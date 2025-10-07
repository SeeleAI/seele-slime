import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def train(args):
    breakpoint()
    assert not args.colocate, "Colocation is not supported for async training."
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    assert not args.offload and not args.colocate, "Offload and colocate are not supported for full async RL training."

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.controller.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    # Note that we initialize it earlier as megatron ckpt loading may have really large peak memory usage.
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    # 1. 启动生产者（训练开始前，持续后台生产）
    ray.get(rollout_manager.start_producer.remote())

    ray.get(rollout_manager.create_consumer.remote())


    # 真正异步解耦的训练循环
    # 使用流水线模式，但避免阻塞等待数据生成
    pending_train_futures = []
    pending_generate_futures = {}
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"\n--- Rollout {rollout_id}/{args.num_rollout} ---")
        
        # 真正异步解耦：立即获取预生成数据，不等待
        print(f"从预填充队列获取rollout数据 {rollout_id}...")
        rollout_data_ref = rollout_manager.async_generate(rollout_id)
        
        # 异步训练，不阻塞循环
        print(f"开始异步训练 {rollout_id}...")
        train_future = actor_model.async_train(rollout_id, rollout_data_ref)
        pending_train_futures.append(train_future)

        # 等待当前训练完成（避免内存积累）
        ray.get(train_future)

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            print(f"保存模型 {rollout_id}...")
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.controller.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            print(f"更新权重 {rollout_id}...")
            ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            print(f"评估 {rollout_id}...")
            eval_future = rollout_manager.async_eval(rollout_id)
            ray.get(eval_future)  # Eval can be blocking as it's infrequent

    # 清理剩余数据（确保没有数据浪费）
    remaining_data = ray.get(rollout_manager.drain_remaining_data.remote())
    
    # 停止生产者（训练结束后）
    ray.get(rollout_manager.stop_producer.remote())



if __name__ == "__main__":
    args = parse_args()
    train(args)
