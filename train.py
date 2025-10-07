import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary

def train(args):
    """
    - Ray：分布式计算框架，用于管理多个GPU和进程
    - Actor：Ray中的分布式对象，可以在不同GPU上运行
    - Placement Group：GPU资源分配组，确保合理的资源分配
    """
    # 第一步：初始化分布式系统和资源分配

    # 创建GPU资源分配组（为不同的任务分配不同的GPU）
    pgs = create_placement_groups(args)
    
    # 初始化实验跟踪系统（用于记录训练过程和结果）
    wandb_run_id = init_wandb_primary(args)

    # 创建训练模型的Actor组（负责实际的模型训练）
    # 这些Actor运行在专门的GPU上，只负责训练，不负责数据生成
    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # 创建数据生成管理器（负责生成训练数据）包含了我们的异步生产者-消费者系统
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # 第二步：计算训练参数和初始化模型
    # 计算每个epoch需要多少个rollout（数据生成轮次）
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        # 从rollout管理器获取每个epoch的rollout数量
        num_rollout_per_epoch = ray.get(rollout_manager.controller.get_num_rollout_per_epoch.remote())
        # 总rollout数 = 每epoch的rollout数 × epoch数
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0, "训练轮次必须大于0"

    print(" 同步初始化所有分布式组件...")
    
    # 同步初始化所有训练Actor（确保所有GPU上的模型都正确加载）
    # with_ref参数决定是否需要参考模型（用于训练算法）
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    # 确保所有worker都从同一个rollout_id开始（保证训练一致性）
    assert len(set(start_rollout_ids)) == 1, "所有worker必须有相同的起始rollout_id"
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    # 如果使用全局数据集，加载之前的训练状态
    if args.rollout_global_dataset:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))

    print("建立训练Actor和数据生成器之间的连接...")
    
    # 初始化权重更新连接（训练Actor需要能够更新rollout_manager中的模型权重）
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    # 内存管理：如果启用了offload，先加载模型权重到GPU
    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    print("同步模型权重...")
    
    # 确保rollout_manager中的模型权重与训练Actor一致；数据生成需要使用最新的模型权重
    ray.get(actor_model.async_update_weights())

    # 内存管理：加载KV缓存到GPU（用于推理生成数据）
    if args.offload:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
    
    # 第三步：启动异步生产者-消费者解耦系统 
    # 启动生产者（在训练循环外，持续后台生产数据）
    # 这是异步解耦的关键：生产者独立运行，不会阻塞训练过程
    print("启动数据生产者...")
    ray.get(rollout_manager.start_producer.remote())
    
    # 创建消费者并等待队列预填充
    # 预填充确保训练开始时就有足够的数据可用，避免等待
    print("\n创建数据消费者...")
    ray.get(rollout_manager.create_consumer.remote())
    print("消费者已创建，队列已预填充")

    # 第四步：异步训练主循环 
    print("开始异步训练主循环")
    
    # 用于跟踪待完成的训练任务（异步训练的关键）
    pending_train_futures = []
    
    # 主训练循环：每个rollout_id代表一轮数据生成和训练
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        print(f"\n === Rollout {rollout_id}/{args.num_rollout} ===")
        
        # 可选：在第一轮进行模型评估
        if args.eval_interval is not None and rollout_id == 0:
            print("执行初始模型评估...")
            eval_future = rollout_manager.async_eval(rollout_id)
            ray.get(eval_future)  # 评估可以是阻塞的，因为不频繁

        # 关键步骤：从预填充队列获取数据（非阻塞）
        print(f"从预填充队列获取rollout数据 {rollout_id}...")
        rollout_data_ref = rollout_manager.async_generate(rollout_id)
        
        #  启动异步训练（不等待完成，继续下一轮）
        print(f" 开始异步训练 {rollout_id}...")
        print("   - 训练在后台进行，不阻塞数据获取")
        train_future = actor_model.async_train(rollout_id, rollout_data_ref)
        pending_train_futures.append(train_future)

        # 内存管理：如果启用了offload，释放不需要的GPU内存
        if args.offload:
            offload_futures = rollout_manager.async_offload()
            ray.get(offload_futures)  # Offload可以是阻塞的

        #  等待当前训练完成（避免内存无限积累）
        ray.get(train_future)
        print(f" 训练 {rollout_id} 完成")
        print("   - 在等待训练时，生产者仍在后台生成新数据")

        #  定期保存模型检查点
        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            print(f" 保存模型检查点 {rollout_id}...")
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.controller.save.remote(rollout_id))

        #  内存管理和权重更新循环
        if args.offload:
            # 释放不需要的GPU内存
            ray.get(rollout_manager.async_offload())
            # 重新加载模型权重到GPU
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

        print(" 更新rollout_manager中的模型权重...")
        # 将训练后的最新权重同步到数据生成器
        # 这确保生产者使用最新的模型来生成数据
        ray.get(actor_model.async_update_weights())

        if args.offload:
            # 重新加载KV缓存到GPU（用于数据生成）
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            eval_future = rollout_manager.async_eval(rollout_id)
            ray.get(eval_future)  # 评估可以是阻塞的

    # 清理队列中剩余的数据（确保没有数据浪费）
    remaining_data = ray.get(rollout_manager.drain_remaining_data.remote())
    print(f"清理完成：回收了 {len(remaining_data)} 个剩余样本")
    print(" 停止数据生产者...")
    ray.get(rollout_manager.stop_producer.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
