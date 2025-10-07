import asyncio
import copy
import time
import torch
from typing import List, Optional
import uuid
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from agent.base.protocal import MySample
from slime.utils.async_utils import run
from tqdm import tqdm
from agent.base.env import Env
from agent.utils import get_env
from agent.gym_rollout_datasource import GymRolloutDataSource
from agent.rollout_producer_consumer import RolloutProducer, RolloutConsumer

__all__ = ["generate_rollout", "create_rollout_producer", "create_rollout_consumer", "shutdown_global_producer"]

# Lynx: Qwen3 sometimes generate this, which is not the instruction-tuned EOS token
DUMMY_EOS_TOKENS = ["<|endoftext|>"]

def remove_eos_token(tokenizer, txt: str):
    eos_tokens = DUMMY_EOS_TOKENS + [tokenizer.eos_token]
    for token in eos_tokens:
        if txt.endswith(token):
            return txt[:-len(token)]

def get_remaining_tokens(memory, current_length):
    return max(0, memory - current_length)

async def agent_loop_generate(
    args,
    sample: MySample,
    trajectory_id: str,
    prompt_group_id: str,
    sampling_params: dict
) -> List[MySample]:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."
    max_turns = getattr(args, 'max_turns', 10)
    max_len = getattr(args, 'rollout_max_response_len', 4096)
    virtual_memory = 2048

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    env: Env = get_env(sample.metadata)
    
    # get the initial messages (sys prompt and question)
    sample.messages = await env.reset()
    input_token_len = len(state.tokenizer.apply_chat_template(sample.messages, add_generation_prompt=True, tokenize=True))
    sample.messages[1]["content"] += f" Remaining tokens: {get_remaining_tokens(virtual_memory, input_token_len)}"
    
    current_prompt = state.tokenizer.apply_chat_template(
            sample.messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    # current_prompt_ids = state.tokenizer(current_prompt, add_special_tokens=False)["input_ids"]
    # tool_call_count = 0  # Track actual tool call rounds
    
    sample.rollout_log_probs = []
    sample.assistant_tokens = []
    
    # maybe record the extra sample when swap out happens
    all_samples = []
    reward = 0.0
    for turn in range(max_turns):
        # 1. Post generate request to SGLang
        payload = {
            "text": current_prompt,
            "sampling_params": sampling_params,
            "return_logprob": True
        }
        output = await post(url, payload, use_http2=args.use_http2)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = MySample.Status.ABORTED
            break
        
        # Handle truncation
        if output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = MySample.Status.TRUNCATED
            break
        
        # 2. collect text, logprobs
        cur_response = output["text"]
        # manage logprobs at turn level
        sample.rollout_log_probs.append([item[0] for item in output["meta_info"]["output_token_logprobs"]])
        sample.assistant_tokens.append([item[1] for item in output["meta_info"]["output_token_logprobs"]])
        
        # 3. interact with enviroment
        # Lynx: here we remove the EOS token, because tokenizer.apply_chat_template will add an EOS token
        # However, we should let the SGLang return the EOS token, because we need log prob of EOS token
        # Lynx: **IMPORTANT** this algorithm assumes that the LLM will always generate a EOS token as a stop token
        # if some other stop token kicks in, this will cause a bug!!
        sample.messages.append({"role": "assistant", "content": remove_eos_token(state.tokenizer, cur_response)})
        step_result = await env.step(sample.messages)
        
        if step_result.reward is not None:
            reward += step_result.reward
            
        if step_result.done:
            sample.status = MySample.Status.COMPLETED
            sample.metadata = {
                "trajectory_id": trajectory_id,
                "prompt_group_id": prompt_group_id,
                "turn_number": turn,
                "env_config": sample.metadata.get("env_name"),
            }
            all_samples.append(sample)
            break
            
        # swap out happend
        if step_result.modified_context:
            # In this case, we need to save the previous messages as a
            # new sample, and continue the interaction with modified context
            original_sample = copy.deepcopy(sample)
            original_sample.metadata = {
                "trajectory_id": trajectory_id,  # belongs to the same trajectory
                "prompt_group_id": prompt_group_id,
                "turn_number": turn,
                "env_config": sample.metadata.get("env_name"),
            }
            all_samples.append(original_sample)
        
        sample.messages = step_result.next_observation
        next_input_length = len(state.tokenizer.apply_chat_template(sample.messages, add_generation_prompt=True, tokenize=True))
        if next_input_length > max_len:
            sample.status = MySample.Status.TRUNCATED
            break
        
        if sample.messages[-1]["role"] == "assistant":
            sample.messages.append({"role": "user", "content": f"Remaining tokens: {get_remaining_tokens(virtual_memory, next_input_length)}"})
        else:
            sample.messages[-1]["content"] += f"\nRemaining tokens: {get_remaining_tokens(virtual_memory, next_input_length)}"
        
        # 4. update to next observation
        current_prompt = state.tokenizer.apply_chat_template(
            sample.messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
    # no sample generated due to abort or truncate
    if not all_samples:
        empty_sample = MySample(
            index=sample.index,
            prompt=sample.prompt,
            messages=[],
            tokens=[],
            response="",
            response_length=0,
            reward=0.0,
            status=MySample.Status.ABORTED,
            metadata={
                "trajectory_id": trajectory_id,
                "prompt_group_id": prompt_group_id,
                "error": "No samples generated",
            }
        )
        all_samples.append(empty_sample)
        
    # in case swap out happened, we assign identical reward to all samples
    for sample in all_samples:
        sample.reward = reward
        
    return all_samples
    


# async def generate_trajectory_as_samples(
#     args,
#     prompt_sample: Sample,
#     trajectory_id: str,
#     prompt_group_id: str,
#     sampling_params: dict
# ) -> List[Sample]:
#     """
#     生成一个trajectory并拆分为多个samples（steps）
    
#     Returns:
#         List[Sample]: 每个step作为一个独立的Sample
#     loss_mask should be the same length as response_length, with 1 for tokens that should be included in the loss calculation and 0 for those that should be masked out.
#     """
#     state = GenerateState(args)
#     url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
#     # 从prompt_sample中获取环境配置
#     env: Env = get_env(prompt_sample.metadata)
    
#     # Reset environment获取初始messages
#     messages = await env.reset()
#     # empty_system_tokens_index = len(state.tokenizer.apply_chat_template(
#     #     [{"role": "system", "content": ""}],
#     #     add_generation_prompt=False,
#     #     tokenize=False,
#     # ))
#     # 生成的samples列表
#     samples = []
    
#     # 配置参数
#     max_turns = getattr(args, 'max_turns', 10)
#     max_len = getattr(args, 'rollout_max_response_len', 4096)
    
#     # 跟踪时间
#     generation_time = 0.0
#     tool_time = 0.0
#     start_time = time.time()
    
#     # 初始化当前step的记录
#     step_prompt_ids = []
#     step_response_ids = []
#     step_loss_masks = []
#     step_number = 0
#     log_probs = []
#     step_prompt_text = ""
#     reponse_text = ""
#     current_turn = 1
#     done = False
#     trajectory_reward = 0.0  # 累积trajectory奖励
    
#     while not done and current_turn <= max_turns:
#         # 1. Tokenize current messages作为prompt
#         current_prompt = state.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=False,
#             enable_thinking=False
#         )
#         current_prompt_ids = state.tokenizer(current_prompt, add_special_tokens=False)["input_ids"] #只支持non-think model
        
#         # 如果是step的第一次生成，记录prompt
#         if not step_prompt_ids:
#             step_prompt_ids = current_prompt_ids
#             step_prompt_text = current_prompt
        
#         # 2. Generate response
#         gen_start = time.time()
#         # current_prompt = state.tokenizer.apply_chat_template(
#         #     [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello! Who are you?"}],
#         #     add_generation_prompt=True,
#         #     tokenize=False,
#         #     enable_thinking=False
#         # )
#         payload = {
#             "text": current_prompt,
#             "sampling_params": sampling_params,
#             "return_logprob": args.use_tis
#         }
        
#         try:
#             output = await post(url, payload, use_http2=args.use_http2)
#         except Exception as e:
#             print(f"Generation failed: {e}")
#             done = True
#             break
#         generation_time += time.time() - gen_start
        
#         # 检查abort
#         if output["meta_info"]["finish_reason"]["type"] == "abort":
#             done = True
#             break
        
#         # 处理生成结果
#         assistant_response = output["text"]
#         reponse_text += assistant_response
        
#         # 获取log probabilities（如果有）
#         log_probs += [item[0] for item in output["meta_info"]["output_token_logprobs"]]
#         assistant_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        
#         # 3. 记录assistant response
#         step_response_ids.extend(assistant_token_ids)
#         step_loss_masks.extend([1] * len(assistant_token_ids))  # assistant tokens标记为1
        
#         # 4. 更新messages
#         messages.append({"role": "assistant", "content": remove_eos_token(state.tokenizer, assistant_response)})
        
#         # 5. Environment step
#         #<im_start>system<im_end><im_start>user<im_end>...<im_start>assistant<im_end>
#         tool_start = time.time()
#         step_result = await env.step(messages)
#         tool_time += time.time() - tool_start
#         pre_msg = state.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=False,
#             tokenize=False,
#             enable_thinking=False,
#         )
#         pre_inpudt_ids = state.tokenizer(pre_msg, add_special_tokens=False)["input_ids"]
#         all_content = state.tokenizer.apply_chat_template(
#             step_result.next_observation,
#             add_generation_prompt=False,
#             tokenize=False,
#             enable_thinking=False,
#         )
        
#         all_ids = state.tokenizer(all_content, add_special_tokens=False)["input_ids"]
#         user_content_ids = all_ids[len(pre_inpudt_ids):] #仅适用non-thinking model
#         user_content = state.tokenizer.decode(user_content_ids)  # 
#         # 累积奖励
#         if step_result.reward is not None:
#             trajectory_reward += step_result.reward
        

#         # 检查是否超过最大长度
#         total_len = len(step_prompt_ids) + len(step_response_ids)
#         if total_len >= max_len - 100:
#             step_result.done = True
        
#         # 7. 决定是否保存当前step为Sample
#         if step_result.done or step_result.modified_context:
#             # 创建Sample
#             sample_tokens = step_prompt_ids + step_response_ids
            
#             # 创建Sample对象
#             step_sample = Sample(
#                 index=prompt_sample.index,  # 继承原始index
#                 prompt=step_prompt_text if step_number == 0 else "",  # 只第一个step保存prompt文本
#                 tokens=sample_tokens,
#                 response=reponse_text,  # response文本可选，用于debug
#                 response_length=len(step_response_ids),
#                 label=prompt_sample.label,  # 继承label
#                 reward=trajectory_reward,  # 设置trajectory reward
#                 loss_mask=step_loss_masks,
#                 weight_versions=[output["meta_info"].get("weight_version", "unknown")],
#                 rollout_log_probs=log_probs if log_probs else None,
#                 status=Sample.Status.COMPLETED if step_result.done else Sample.Status.PENDING,
#                 metadata={
#                     "trajectory_id": trajectory_id,
#                     "prompt_group_id": prompt_group_id,
#                     "step_number": step_number,
#                     "turn_number": current_turn,
#                     "env_config": prompt_sample.metadata,
#                     "generation_time": generation_time,
#                     "tool_time": tool_time,
#                 }
#             )
            
#             samples.append(step_sample)
#             step_number += 1
#             #query promptids
#             # response
#             #tool_call 1
#             #tool_result 0
#             # user compress
#             #too_call 1 #modified_context  user compress

#             # 如果context被修改，开始新的step
#             if step_result.modified_context and not step_result.done: #如果修改了context并且还没有完成，则把修改后的所有上下文作为prompt_ids
#                 step_prompt_text = state.tokenizer(step_result.next_observation,add_generation_prompt=True,tokenize=False)
#                 step_prompt_ids = state.tokenizer(step_prompt_text,add_special_tokens=False)["input_ids"]
#                 step_response_ids = []
#                 step_loss_masks = []
#                 reponse_text = ""
#                 messages = copy.deepcopy(step_result.next_observation)
#                 generation_time = 0.0
#                 tool_time = 0.0
        
#         # 更新状态
#         if not step_result.done:
#             messages = step_result.next_observation
#             if not step_result.modified_context: #如果没有完成，并且没有修改context，则正常添加loss mask
#                 step_loss_masks += [0] * len(user_content_ids)
#                 log_probs += [0] * len(user_content_ids)
#                 step_response_ids += user_content_ids
#                 reponse_text += user_content
            
        
#         done = step_result.done
#         current_turn += 1
        
#         # 检查length限制
#         if output["meta_info"]["finish_reason"]["type"] == "length":
#             done = True
    
#     # 关闭环境
#     await env.close()

#     # 更新所有samples的最终信息
#     for sample in samples:
#         sample.reward = trajectory_reward  # 确保所有steps使用相同的trajectory reward
#         sample.metadata["trajectory_reward"] = trajectory_reward
#         sample.metadata["trajectory_done"] = done
#         sample.metadata["trajectory_total_time"] = time.time() - start_time
#         sample.metadata["trajectory_num_steps"] = len(samples)
    
#     # 如果没有生成任何sample，创建一个空的
#     if not samples:
#         empty_sample = Sample(
#             index=prompt_sample.index,
#             prompt=prompt_sample.prompt,
#             tokens=[],
#             response="",
#             response_length=0,
#             loss_masks=[],
#             reward=0.0,
#             status=Sample.Status.ABORTED,
#             metadata={
#                 "trajectory_id": trajectory_id,
#                 "prompt_group_id": prompt_group_id,
#                 "error": "No samples generated",
#             }
#         )
#         samples = [empty_sample]
    
#     return samples



# ============================================================================
# 全局生产者-消费者管理
# ============================================================================
# 使用全局变量管理生产者和队列，确保整个训练过程中只有一个生产者实例
# 生产者在后台持续运行，不依赖于单个rollout的生命周期

_global_producer: Optional[RolloutProducer] = None
_global_queue: Optional[asyncio.Queue] = None


async def create_rollout_producer(args, data_source: GymRolloutDataSource) -> RolloutProducer:
    """
    创建并启动全局rollout生产者
    支持真正的异步解耦：大容量队列 + 持续后台生产
    
    Args:
        args: 配置参数
        data_source: 数据源
        
    Returns:
        RolloutProducer: 生产者实例
    """
    global _global_producer, _global_queue
    
    if _global_producer is not None:
        print("Global producer already exists, returning existing instance")
        return _global_producer
        
    # 创建大容量队列，支持10个批次的缓冲
    max_queue_size = args.global_batch_size * 10
    _global_queue = asyncio.Queue(maxsize=max_queue_size)
    
    # 创建并启动生产者
    _global_producer = RolloutProducer(args, _global_queue, data_source)
    await _global_producer.start()
    
    print(f"Global rollout producer created with queue size: {max_queue_size}")
    print(f"Producer will continuously generate data in background")
    return _global_producer


async def create_rollout_consumer(args) -> RolloutConsumer:
    """
    创建rollout消费者
    支持真正的异步解耦：非阻塞获取 + 预填充机制
    
    Args:
        args: 配置参数
        
    Returns:
        RolloutConsumer: 消费者实例
    """
    global _global_queue
    
    if _global_queue is None:
        raise RuntimeError("Global queue not initialized, call create_rollout_producer first")
        
    consumer = RolloutConsumer(_global_queue, args)
    
    # 等待队列预填充，确保训练开始时就有足够数据
    await consumer.ensure_queue_ready()
    
    return consumer


async def shutdown_global_producer():
    """关闭全局生产者"""
    global _global_producer
    
    if _global_producer:
        await _global_producer.stop()
        _global_producer = None
        print("Global producer shutdown")

# ============================================================================
# GRPO优势计算
# ============================================================================

def compute_group_advantages(
    samples: List[List[MySample]],
    args
):
    """
    计算一个prompt group内所有trajectories的GRPO优势值
    
    GRPO (Group Relative Policy Optimization) 算法：
    - 对同一个prompt的多个响应进行组内对比
    - advantage = reward - mean(group_rewards)
    - 正值：比平均好，应该增强
    - 负值：比平均差，应该抑制
    
    Args:
        samples: List[List[MySample]] - 每个trajectory可能有多个steps（swap out）
        args: 配置参数
        
    Returns:
        List[float]: 每个trajectory的优势值
    """
    # Although swap out gives us more samples, they have identical rewards
    # and we only need one of them as representative.
    raw_rewards = [sample[0].reward for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        mean = rewards.mean()  # calculate group mean
        rewards = rewards - mean  # remove bias as in GRPO

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std()  # calculate group std
            rewards = rewards / (std + 1e-6)  # remove variance and avoid zero division

        return rewards.flatten().tolist()

    return raw_rewards

def compute_prompt_group_advantages(
    trajectory_samples_by_prompt: List[List[MySample]], 
    args
) -> List[MySample]:
    """
    计算一个prompt group内所有trajectories的advantages（备用函数）
    
    注意：这个函数目前未被使用，保留用于可能的future扩展
    主要使用的是 compute_group_advantages()
    """
    import torch
    
    # 获取每个trajectory的reward
    trajectory_rewards = []
    for trajectory_steps in trajectory_samples_by_prompt:
        if trajectory_steps:
            # 所有steps共享相同的trajectory reward
            trajectory_rewards.append(trajectory_steps[0].reward)
    
    if len(trajectory_rewards) <= 1:
        # 只有一个trajectory，advantage设为0
        all_steps = []
        for trajectory_steps in trajectory_samples_by_prompt:
            for step in trajectory_steps:
                step.advantage = 0.0
                all_steps.append(step)
        return all_steps
    
    # 计算GRPO advantages
    rewards = torch.tensor(trajectory_rewards, dtype=torch.float32)
    mean = rewards.mean()
    advantages = rewards - mean
    
    if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
        std = rewards.std()
        if std > 0:
            advantages = advantages / std
    
    # 分配advantages到所有steps
    all_steps = []
    for i, trajectory_steps in enumerate(trajectory_samples_by_prompt):
        advantage = advantages[i].item()
        for step in trajectory_steps:
            step.advantage = advantage
            if not hasattr(step, 'metadata'):
                step.metadata = {}
            step.metadata["advantage"] = advantage
        all_steps.extend(trajectory_steps)
    
    return all_steps


def generate_rollout(args, rollout_id, data_source: GymRolloutDataSource, evaluation=False, consumer=None):
    """
    生成rollout数据的入口函数（向后兼容接口）
    
    注意：训练模式已废弃，请使用生产者-消费者模式
    
    Args:
        args: 配置参数
        rollout_id: rollout ID
        data_source: 数据源（训练模式已废弃）
        evaluation: 是否为评估模式
        consumer: 消费者实例（训练模式已废弃）
        
    Returns:
        List[MySample]: 生成的samples
    """
    if evaluation:
        # 评估模式：使用独立的实现
        return run(eval_rollout(args, rollout_id))
    else:
        # 训练模式已废弃
        raise RuntimeError(
            "Training mode is deprecated in generate_rollout(). "
            "The system now uses producer-consumer pattern. "
            "Data generation is handled by RolloutProducer in background, "
            "and GymRolloutController.generate() fetches data from queue directly."
        )


# ============================================================================
# 评估（Evaluation）相关函数
# ============================================================================
# 评估模式不使用生产者-消费者，而是直接生成数据
# 因为评估不需要训练，不需要异步解耦

# 全局变量存储eval数据集（缓存）
EVAL_PROMPT_DATASET = {}

async def eval_rollout(args, rollout_id):
    """
    Gym环境的eval rollout入口函数
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return results, []


async def eval_rollout_single_dataset(args, rollout_id, name, path):
    """
    对单个数据集进行eval，使用gym环境生成trajectory
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    # 加载数据集（缓存）
    if name not in EVAL_PROMPT_DATASET:
        from transformers import AutoTokenizer
        from slime.utils.data import Dataset
        
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
        )
    dataset = EVAL_PROMPT_DATASET[name]

    # eval的采样参数
    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    # 创建eval任务
    tasks = []
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # 为每个prompt生成多个trajectory
            eval_sample = copy.deepcopy(prompt_sample)
            eval_sample.index = sample_index
            sample_index += 1
            
            # 添加环境配置到metadata
            if not eval_sample.metadata:
                eval_sample.metadata = {}
            # eval_sample.metadata["env_config"] = getattr(args, 'eval_env_config', {})
            
            trajectory_id = f"eval_{rollout_id}_{name}_prompt_{i}_sample_{j}"
            prompt_group_id = f"eval_group_{i}"  # eval不需要GRPO，但保持一致
            
            task = generate_trajectory_as_samples(
                args,
                eval_sample,
                trajectory_id=trajectory_id,
                prompt_group_id=prompt_group_id,
                sampling_params=sampling_params
            )
            tasks.append(task)

    # 并发执行所有eval任务
    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {name}", disable=not do_print)
    
    for coro in asyncio.as_completed(tasks):
        trajectory_samples = await coro  # 返回的是一个trajectory的所有steps
        
        if do_print:
            # 打印第一个trajectory的信息
            if trajectory_samples:
                final_reward = trajectory_samples[0].reward  # 所有steps共享相同reward
                trajectory_id = trajectory_samples[0].metadata["trajectory_id"]
                num_steps = len(trajectory_samples)
                print(f"Eval trajectory {trajectory_id}: {num_steps} steps, reward: {final_reward}", flush=True)
                do_print = False
        
        # 对于eval，我们只关心trajectory级别的结果
        # 创建一个代表整个trajectory的sample用于统计
        if trajectory_samples:
            # 使用第一个step的信息，但合并所有response
            trajectory_sample = copy.deepcopy(trajectory_samples[0])
            trajectory_sample.response = ""  # 可以选择合并所有steps的response
            trajectory_sample.metadata["num_steps"] = len(trajectory_samples)
            trajectory_sample.metadata["trajectory_done"] = trajectory_samples[0].metadata.get("trajectory_done", True)
            
            data.append(trajectory_sample)
        
        pbar.update(1)
    
    pbar.close()

    # 按index排序
    data.sort(key=lambda sample: sample.index)

    # 提取结果
    reward_key = args.reward_key or args.eval_reward_key
    results = {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == MySample.Status.TRUNCATED for sample in data],
            # 添加gym环境特有的统计
            "num_steps": [sample.metadata.get("num_steps", 1) for sample in data],
            "trajectory_done": [sample.metadata.get("trajectory_done", True) for sample in data],
        }
    }
    
    # 打印统计信息
    avg_reward = sum(results[name]["rewards"]) / len(results[name]["rewards"])
    avg_steps = sum(results[name]["num_steps"]) / len(results[name]["num_steps"])
    done_ratio = sum(results[name]["trajectory_done"]) / len(results[name]["trajectory_done"])
    
    print(f"Eval {name} results: avg_reward={avg_reward:.4f}, avg_steps={avg_steps:.2f}, done_ratio={done_ratio:.2f}")
    
    return results


# sanity check
if __name__ == "__main__":
    pass


