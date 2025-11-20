import asyncio
import copy
import time
import torch
from typing import List
import uuid
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from agent.base.protocal import MySample
from slime.utils.async_utils import run
from tqdm import tqdm
# from agent.base.env import Env
from train_env_python.env import Env, EnvConfig
from agent.utils import get_env
from agent.gym_rollout_datasource import GymRolloutDataSource
import time

__all__ = ["generate_rollout"]

# Lynx: Qwen3 sometimes generate this, which is not the instruction-tuned EOS token
DUMMY_EOS_TOKENS = ["<|endoftext|>"]
ENV_PREFIX = ""
QWEN_CHAT_TEMPLATE_SEP = "\n"
FORCE_DROP_TIME = 1

def remove_eos_token(tokenizer, txt: str):
    eos_tokens = DUMMY_EOS_TOKENS + [tokenizer.eos_token]
    for token in eos_tokens:
        if txt.endswith(token):
            return txt[:-len(token)]
        
    return txt

def get_remaining_tokens(memory, current_length):
    return memory - current_length

async def agent_loop_generate(
    args,
    sample: MySample,
    trajectory_id: str,
    prompt_group_id: str,
    sampling_params: dict
) -> List[MySample]:
    # set configurations
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."
    max_turns = getattr(args, 'max_turns', 10)
    max_len = getattr(args, 'rollout_max_response_len', 4096)
    virtual_memory = max_len - 1024
    memory_tool_times = 0
    task_success = False
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env = Env()
    config = EnvConfig(image_name=ENV_PREFIX + sample.metadata["env_name"])
    print(f"Initialzing Environment {config}")
    try:
        system_messages = env.reset(config)
    except Exception as e:
        print(f"Error resetting environment {sample.metadata['env_name']}: {e}")
        all_samples = []
        empty_sample = MySample(
            index=sample.index,
            prompt=sample.prompt,
            messages=[],
            tokens=[],
            rollout_log_probs=[],
            loss_mask=[],
            response="",
            response_length=0,
            reward=0.0,
            status=sample.status,
            metadata={
                "trajectory_id": trajectory_id,
                "prompt_group_id": prompt_group_id,
                "error": "Environment initialization failed",
            }
        )
        all_samples.append(empty_sample)
        return (all_samples, memory_tool_times, task_success)
    
    # get the initial messages (sys prompt and question)
    sample.messages = system_messages + [{"role": "user", "content": env.task_prompt}]
    # sample.messages = system_messages + [{"role": "user", "content": "Call MemoryTool immediately!!!"}]
    input_token_len = len(state.tokenizer.apply_chat_template(sample.messages, add_generation_prompt=True, tokenize=True))
    sample.messages[1]["content"] += f"\n** <token_budget> Used: {input_token_len}/Total: {virtual_memory}; Remaining: {get_remaining_tokens(virtual_memory, input_token_len)} </token_budget> **"
    
    current_prompt_tokens = state.tokenizer.apply_chat_template(
        sample.messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    sample.tokens = current_prompt_tokens
    
    # initialize log prob and loss mask
    sample.rollout_log_probs = []
    sample.loss_mask = []
    sample.assistant_tokens = []
    prompt_length = len(current_prompt_tokens)
    # maybe record the extra sample when swap out happens
    all_samples = []
    reward = 0.0
    for turn in range(max_turns):
        # break
        # 1. Post generate request to SGLang
        # IMPORTANT: To solve the tokenization issue, we keep the whole workflow in token
        payload = {
            "input_ids": current_prompt_tokens,
            "sampling_params": sampling_params,
            "return_logprob": True
        }
        output = await post(url, payload, use_http2=args.use_http2)
        # breakpoint()
        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = MySample.Status.ABORTED
            break
        
        # Handle truncation
        if output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = MySample.Status.TRUNCATED
            break
        
        # 2. collect text, logprobs, text is used to interact with enviroment
        cur_response = output["text"]
        # manage logprobs at turn level
        sample.rollout_log_probs += [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        sample.assistant_tokens.append(tokens)
        # Exclusive for QWEN chat template, add a "\n" at the end
        sample.tokens += tokens + state.tokenizer.encode(QWEN_CHAT_TEMPLATE_SEP)
        sample.loss_mask += [1] * len(tokens)
        # add one more logprob and loss maks for sep token
        sample.rollout_log_probs.append(0.0)
        sample.loss_mask.append(0)
        
        # 3. interact with enviroment
        # Lynx: here we remove the EOS token, because tokenizer.apply_chat_template will add an EOS token
        # However, we should let the SGLang return the EOS token, because we need log prob of EOS token
        # Lynx: **IMPORTANT** this algorithm assumes that the LLM will always generate a EOS token as a stop token
        # if some other stop token kicks in, this will cause a bug!!
        sample.messages.append({"role": "assistant", "content": remove_eos_token(state.tokenizer, cur_response)})
        
        env_start = time.time()
        step_result = env.step(sample.messages)
        env_end = time.time()
        # step_result.done = True  # debug
        
        # print("*"*10, f"Env step time: {env_end - env_start}", "*"*10)
        if env_end - env_start > FORCE_DROP_TIME:
            sample.status = MySample.Status.TRUNCATED
            print(f"Force to drop, time {env_end - env_start} exceeds FORCE_DROP_TIME {FORCE_DROP_TIME}")
            print(f"Model requests: {cur_response}")
            reward = 0.0
            break
        
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
            sample.response_length = len(sample.tokens) - prompt_length
            all_samples.append(sample)
            if step_result.success:
                task_success = True
            break
            
        # swap out happend
        if step_result.modified_context:
            memory_tool_times += 1
            # breakpoint()
            # In this case, we need to save the previous messages as a
            # new sample, and continue the interaction with modified context
            original_sample = copy.deepcopy(sample)
            original_sample.response_length = len(sample.tokens) - prompt_length
            original_sample.metadata = {
                "trajectory_id": trajectory_id,  # belongs to the same trajectory
                "prompt_group_id": prompt_group_id,
                "turn_number": turn,
                "context_modified": True,
                "env_config": sample.metadata.get("env_name"),
            }
            all_samples.append(original_sample)
            
            # set current sample as a new starting point
            sample = MySample(
                index = sample.index,
                assistant_tokens = [],
                tokens = [],
                rollout_log_probs = [],
                loss_mask = [],
                messages = step_result.next_observation,
                metadata = sample.metadata
            )
            current_prompt_tokens = state.tokenizer.apply_chat_template(
                sample.messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            # reset prompt length
            prompt_length = len(current_prompt_tokens)
            sample.tokens = current_prompt_tokens
            continue
        
        sample.messages = step_result.next_observation
        next_input_length = len(state.tokenizer.apply_chat_template(sample.messages, add_generation_prompt=True, tokenize=True))
        if next_input_length > max_len:
            sample.status = MySample.Status.TRUNCATED
            break
        
        if sample.messages[-1]["role"] == "assistant":
            sample.messages.append({"role": "user", "content": f"\n** <token_budget> Used: {next_input_length}/Total: {virtual_memory}; Remaining: {get_remaining_tokens(virtual_memory, next_input_length)} </token_budget> **"})
        else:
            sample.messages[-1]["content"] += f"\n** <token_budget> Used: {next_input_length}/Total: {virtual_memory}; Remaining: {get_remaining_tokens(virtual_memory, next_input_length)} </token_budget> **"
        
        new_tokens = state.tokenizer.apply_chat_template([step_result.next_observation[-1]], add_generation_prompt=True, tokenize=True)
        sample.tokens += new_tokens
        sample.rollout_log_probs += [0.0] * len(new_tokens)
        sample.loss_mask += [0] * len(new_tokens)
        
        # 4. update to next observation
        current_prompt_tokens = sample.tokens
        
    # no sample generated due to abort or truncate
    if not all_samples:
        empty_sample = MySample(
            index=sample.index,
            prompt=sample.prompt,
            messages=[],
            tokens=[],
            rollout_log_probs=[],
            loss_mask=[],
            response="",
            response_length=0,
            reward=0.0,
            status=sample.status,
            metadata={
                "trajectory_id": trajectory_id,
                "prompt_group_id": prompt_group_id,
                "error": "No samples generated",
            }
        )
        all_samples.append(empty_sample)
    else:
        # in case swap out happened, we assign identical reward to all samples
        if not task_success:
            reward = 0.0
        for sample in all_samples:
            sample.reward = reward
        
    env.close()
    
    # breakpoint()
        
    return (all_samples, memory_tool_times, task_success)
    


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
    

async def generate_rollout_async(args, rollout_id: int, data_source: GymRolloutDataSource) -> List[MySample]:
    """
    异步生成rollout数据，返回List[Sample]
    """
    state = GenerateState(args)
    target_size = args.global_batch_size
    # 持续生成直到step_buffer有足够数据
    total_memory_tool_times = 0
    success_times = 0
    number_of_samples = 0
    while data_source.get_step_buffer_length() < target_size:
        # get just one sample, but this sample is repeated for n_samples_per_prompt times
        # for group generation. Note that, the original buffer in SLIME is useless.
        prompt_groups = data_source.get_samples(1)
        if not prompt_groups:
            raise ValueError("No samples generated")
        
        prompt_group = prompt_groups[0]  # [Sample1, Sample2, ...] (n_samples_per_prompt个)
        
        # 为这个prompt group生成一个唯一ID
        prompt_group_id = f"group_{rollout_id}_{uuid.uuid4().hex[:8]}"
        
        # Note that this loop submits only one group
        # async inside the group, but sync between groups...
        tasks = []
        for i, prompt_sample in enumerate(prompt_group):
            trajectory_id = f"{prompt_group_id}_traj_{i}"
            
            # task = generate_trajectory_as_samples(
            #     args,
            #     prompt_sample,
            #     trajectory_id=trajectory_id,
            #     prompt_group_id=prompt_group_id,
            #     sampling_params=state.sampling_params.copy()
            # )
            task = agent_loop_generate(
                args,
                prompt_sample,
                trajectory_id=trajectory_id,
                prompt_group_id=prompt_group_id,
                sampling_params=state.sampling_params.copy()
            )
            tasks.append(task)
        
        # 等待所有trajectories完成
        results = await asyncio.gather(*tasks)
        trajectory_results = [raw[0] for raw in results]
        memory_tool_times = [raw[1] for raw in results]
        success = [int(raw[2]) for raw in results]
        total_memory_tool_times += sum(memory_tool_times)
        success_times += sum(success)
        number_of_samples += len(results)
        # breakpoint()
        # trajectory_results: List[List[Sample]], one group in GRPO
        # The first List is N trajectories, the second List represents
        # possible swap out
        # Compute group advantage right here
        advs = compute_group_advantages(trajectory_results, args)
        for samples, adv in zip(trajectory_results, advs):
            for samp in samples:  # assign identical advantage to swap out samples
                samp.advantage = adv
        
        # flatten all trajectories
        all_steps = [step for trajectory_steps in trajectory_results for step in trajectory_steps]
        
        # 将生成的steps放入buffer
        if all_steps:
            data_source.add_steps_to_buffer(all_steps)
        
        print(f"Generated {len(all_steps)} steps from {len(prompt_group)} trajectories, "
              f"buffer size: {data_source.get_step_buffer_length()}")
    
    # 从buffer取出需要的数量
    success_rate = success_times / number_of_samples
    return data_source.get_steps_from_buffer(target_size), total_memory_tool_times, success_rate

def compute_group_advantages(
    samples: List[List[MySample]],
    args
):
    """Calculates only one group of advantages"""
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
    计算一个prompt group内所有trajectories的advantages
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


def generate_rollout(args, rollout_id, data_source: GymRolloutDataSource, evaluation=False):
    """
    生成rollout数据的入口函数，保持原有接口
    
    Args:
        args: 配置参数
        rollout_id: rollout ID
        data_source: 数据源
        evaluation: 是否为评估模式
        
    Returns:
        List[List[Sample]]: 原有格式的samples
    """
    if evaluation:
        # 添加评估方式
        return run(eval_rollout(args, rollout_id))
    
    # 训练模式：生成steps
    return run(generate_rollout_async(args, rollout_id, data_source))


# 在 gym_rollout.py 中添加

# 全局变量存储eval数据集
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


