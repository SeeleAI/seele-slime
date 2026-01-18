import asyncio
import copy
import logging
import torch
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable, Optional, Union, List

import numpy as np
import sglang_router
import time
from packaging.version import parse
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from slime.rollout.rm_hub import async_rm, batched_async_rm

from swe_env.environment import SWEEnv
from dataclasses import dataclass, field
from rollout_buffer import GymRolloutDataSource
import uuid
import functools

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


QWEN_CHAT_TEMPLATE_SEP = "\n"
FORCE_DROP_TIME = 35
DUMMY_EOS_TOKENS = ["<|endoftext|>"]

@dataclass
class RolloutStatus:
    samples: List[Sample] = field(default_factory=list)  # List of Sample, for possible swap out
    memory_tool_times: int = field(default=0)
    trajectory_success: bool = field(default=False)
    task_finished: bool = field(default=False)
    # W&B Swap Out Metrics: 记录每次 swap 的详细信息
    swap_out_infos: List[dict] = field(default_factory=list)
    reward_dict: dict = field(default_factory=dict)

@dataclass
class LoopState:
    """Helper class to track the mutable state of the agent loop."""
    reward: float = 0.0
    turn: int = 0

class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.valid_groups = 0
        self.pendings = set()
        self.aborted = False

    # def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
    #     for group in samples:
    #         self.pendings.add(
    #             asyncio.create_task(
    #                 # submit a group of samples as a single task.
    #                 generate_and_rm_group(
    #                     self.args,
    #                     group,
    #                     sampling_params=self.sampling_params.copy(),
    #                     evaluation=False,
    #                 )
    #             )
    #         )
    #     self.remaining_batch_size += len(samples)
        
def _create_error_result(original_sample: Sample, traj_id: str, pg_id: str, error_msg: str, advantage: float=0.0) -> List[Sample]:
    """Creates a dummy sample to return on critical failure."""
    return [Sample(
        index=original_sample.index,
        prompt=original_sample.prompt,
        tokens=[1, 1],
        rollout_log_probs=[0.0],
        loss_mask=[0],
        response="",
        response_length=1,
        reward=0.0,
        advantage=advantage,
        status=original_sample.status,
        metadata={
            "trajectory_id": traj_id,
            "prompt_group_id": pg_id,
            "error": error_msg,
        }
    )]

def _update_sample_stats(sample: Sample, logprobs: List[float], tokens: List[int], tokenizer):
    """Updates the sample with generation results, including Qwen specific logic."""
    sample.rollout_log_probs += logprobs
    
    # Qwen-specific: Add separator \n
    sample.tokens += tokens + tokenizer.encode(QWEN_CHAT_TEMPLATE_SEP)
    
    sample.loss_mask += [1] * len(tokens)
    
    # Add one more log prob and loss mask for \n
    sample.rollout_log_probs.append(0.0)
    sample.loss_mask.append(0)
    
def _inject_token_budget(
    sample: Sample, 
    total_memory: int, 
    tokenizer, messages: List[dict] = None, 
    observation: dict = None, 
    tool_set: list = None
):
    """Calculates used tokens and injects the budget string into the last message."""
    # Test with this message
    # Lynx: I think this is not exactly correct
    test_msg = (
        f"\n** <token_budget> Used: 1234/Total: 1234; "
        f"Remaining: 1234 </token_budget> **"
    )
    # The provided media is OpenAI message
    if messages is not None:
        assert observation is None
        # We assume the user prompt is the last message
        # Lynx: Avoid modifying the original dictionary references
        # We copy the list logic, but create a fresh copy of the last dict
        _message = messages[:-1] + [messages[-1].copy()]
        # Estimate the input length
        _message[-1]['content'] += test_msg
        input_tokens = tokenizer.apply_chat_template(_message, tokenize=True, add_generation_prompt=True, tools=tool_set)
        input_token_len = len(input_tokens)
    # The provided media is a dict observation (role user)
    elif observation is not None:
        assert messages is None
        _observation = observation.copy()
        _observation['content'] += test_msg
        new_tokens = tokenizer.apply_chat_template([_observation], add_generation_prompt=True, tokenize=True)
        input_token_len = len(sample.tokens) + len(new_tokens)
    else:
        raise RuntimeError(f"Should provide either messages or observation")
    budget_msg = (
        f"\n** <token_budget> Used: {input_token_len}/Total: {total_memory}; "
        f"Remaining: {total_memory - input_token_len} </token_budget> **"
    )
    
    return budget_msg

def _create_reset_sample(previous_sample: Sample, new_messages: List[dict]) -> Sample:
    """Creates a fresh sample object for a modified context (swap out)."""
    return Sample(
        index=previous_sample.index,
        tokens=[],
        rollout_log_probs=[],
        loss_mask=[],
        messages=new_messages,
        metadata=previous_sample.metadata
    )

def remove_eos_token(tokenizer, txt: str):
    eos_tokens = DUMMY_EOS_TOKENS + [tokenizer.eos_token]
    for token in eos_tokens:
        if txt.endswith(token):
            return txt[:-len(token)]
        
    return txt

def _extract_content_preview(messages: List[dict], max_chars: int = 1000) -> str:
    """提取 swap 后新 context 的内容预览，用于 W&B Table 记录
    
    设计依据：
    - 保留最后 3 条消息作为上下文快照
    - 每条消息截断 300 字符避免过长
    - 总长度限制 1000 字符
    """
    if not messages:
        return ""
    
    parts = []
    for msg in messages[-3:]:  # 最后 3 条消息
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))[:300]
        parts.append(f"[{role}]: {content}")
    
    return "\n---\n".join(parts)[:max_chars]

def _sample_random_trajectory(samples: List[Sample], tokenizer) -> Optional[dict]:
    """随机选择一个完整轨迹用于 W&B 记录，监控 swap 内容质量
    
    用途：定性分析 swap 后的内容是否：
    1. 完整保留用户需求
    2. 详细列出已做的事情
    3. 历史经验不过于冗余
    4. 有明确的下一步指示
    """
    import random
    
    # 按 trajectory_id 分组
    trajectories = defaultdict(list)
    for s in samples:
        traj_id = s.metadata.get("trajectory_id")
        # Sample dataclass 总是有 messages 字段
        if traj_id and s.messages and len(s.messages) > 0:
            trajectories[traj_id].append(s)
    
    if not trajectories:
        return None
    
    # 随机选择一个轨迹
    selected_traj_id = random.choice(list(trajectories.keys()))
    selected_samples = trajectories[selected_traj_id]
    
    # 取最后一个 sample 的完整 messages
    final_sample = selected_samples[-1]
    
    return {
        "trajectory_id": selected_traj_id,
        "num_steps": len(selected_samples),
        "has_swap": any(s.metadata.get("context_modified") for s in selected_samples),
        "reward": final_sample.reward,
        "success": final_sample.reward > 0 if final_sample.reward else False,
        "messages": final_sample.messages,  # 原始消息列表，供 wandb.Table 使用
        "messages_text": tokenizer.apply_chat_template(
            final_sample.messages, add_generation_prompt=False, tokenize=False
        ) if final_sample.messages else "",
    }

# Lynx: First turn bug fix done with Gemini
async def generate(
    args: Namespace, 
    sample: Sample, 
    sampling_params: dict[str, Any],
    prompt_group_id: str,
    trajectory_id: str
) -> RolloutStatus:
    """Generate using traditional SGLang router with token-based workflow"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    MAX_TURNS = getattr(args, 'max_turns', 40)
    MAX_LEN = getattr(args, 'rollout_max_response_len', 4096)
    VIRTUAL_MEMORY = MAX_LEN - 1024

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # Process prompt to create text and image payload
    assert isinstance(sample.prompt, str), f"Multimodal rollout is not supported!"
    # Lynx: Need to do careful investigation on how to adapt this function
    assert not args.use_rollout_routing_replay, f"Routing replay not supported!"

    try:
        env = SWEEnv(task_instance=sample.metadata["task_instance"], run_id=trajectory_id)
        print(f"Initializing Environment...")
        task_suit = env.get_initial_prompt()
        system_messages = task_suit["message"]
        tool_set = task_suit["tools"]
    except Exception as e:
        print(f"Error resetting environment {sample.metadata['task_instance']['instance_id']}: {e}")
        return RolloutStatus(
            samples=_create_error_result(sample, trajectory_id, prompt_group_id, "Environment initialization failed"),
            memory_tool_times=0,
            trajectory_success=False
        )
    
    text_prompt = sample.prompt
    if len(sample.response) > 0:
        # Adjust max_new_tokens for subsequent generation turns
        prompt_len = len(state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"])
        sampling_params["max_new_tokens"] -= len(sample.tokens) - prompt_len

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample
        
    # Fresh sample
    if not len(sample.response) > 0:
        init_message = system_messages
        budget_msg = _inject_token_budget(sample, VIRTUAL_MEMORY, state.tokenizer, init_message, None, tool_set)
        init_message[-1]['content'] += budget_msg
        prompt_token_ids = state.tokenizer.apply_chat_template(
            init_message,
            add_generation_prompt=True,
            tokenize=True,
            tools=tool_set
        )
        prompt = state.tokenizer.apply_chat_template(
            init_message,
            add_generation_prompt=True,
            tokenize=False,
            tools=tool_set
        )
        sample.prompt = prompt
        sample.messages = init_message
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_token_ids
            
        sample.rollout_log_probs = []
        sample.loss_mask = []
        
        initial_prompt_len = len(prompt_token_ids)
    else:
        initial_prompt_len = len(state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"])
        
    # Initialize Loop State
    loop_state = LoopState()
        
    ###################
    # Main Agent Loop #
    ###################
    status = RolloutStatus()
    collected_samples: List[Sample] = []
    for turn in range(MAX_TURNS):
        loop_state.turn = turn
        
        # Prepare payload for sglang server
        payload = {
            "input_ids": sample.tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        
        # 1. SGLang generation
        output = await post(url, payload)
        
        # Early exit from SGLang backend
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
                break
            case "abort":
                sample.status = Sample.Status.ABORTED
                break
        
        response_text = output["text"]
        clean_response = remove_eos_token(state.tokenizer, response_text)
        sample.messages.append({"role": "assistant", "content": clean_response})
        logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        output_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]

        # Update Sample Stats (Logprobs, Loss Mask, Tokens)
        _update_sample_stats(sample, logprobs, output_tokens, state.tokenizer)
        
        # 3. Environment Interaction
        env_start = time.time()
        loop = asyncio.get_running_loop()
        step_result = await loop.run_in_executor(None, env.step, sample.messages)
        env_duration = time.time() - env_start
        # Lynx: Do not punish unit test time!!!
        # if env_duration > FORCE_DROP_TIME:
        #     print(f"Force drop: Env step took {env_duration}s > {FORCE_DROP_TIME}s")
        #     sample.status = Sample.Status.TRUNCATED
        #     loop_state.reward = 0.0
        #     break
        
        if step_result.reward is not None:
            loop_state.reward += step_result.reward

        # 4. Handle Step Result
        if step_result.done:
            sample.status = Sample.Status.COMPLETED
            status.task_finished = True
            sample.response_length = len(sample.tokens) - initial_prompt_len
            sample.response = state.tokenizer.decode(sample.tokens[-sample.response_length:])
            sample.metadata.update({
                "trajectory_id": trajectory_id,
                "prompt_group_id": prompt_group_id,
                "turn_number": turn,
                "env_config": sample.metadata.get("env_name"),
            })
            collected_samples.append(sample)
            if step_result.success:
                status.trajectory_success = True
                
            print(f"Trajectory {trajectory_id} done, task_success {step_result.success}")
            break
        
        # 5. Handle Context Swap (Memory Tool)
        if step_result.modified_context:
            # breakpoint()
            status.memory_tool_times += 1
            
            # W&B Metrics: 记录 swap 前的 token 数量
            tokens_before = len(sample.tokens)
            
            # Archive current sample
            archived_sample = copy.deepcopy(sample)
            archived_sample.response_length = len(sample.tokens) - initial_prompt_len
            archived_sample.response = state.tokenizer.decode(archived_sample.tokens[-archived_sample.response_length:])
            archived_sample.metadata.update({
                "trajectory_id": trajectory_id,
                "prompt_group_id": prompt_group_id,
                "turn_number": turn,
                "context_modified": True,
                "env_config": sample.metadata.get("env_name"),
            })
            collected_samples.append(archived_sample)

            # Reset Loop State with new context, step_result.updated_message includes only the swapped context
            sample = _create_reset_sample(sample, step_result.updated_message)
            # Inject budget token again
            budget_msg = _inject_token_budget(sample, VIRTUAL_MEMORY, state.tokenizer, sample.messages, None, tool_set)
            sample.messages[-1]['content'] += budget_msg
            # Re-tokenize entire new context
            sample.tokens = state.tokenizer.apply_chat_template(
                sample.messages, add_generation_prompt=True, tokenize=True, tools=tool_set
            )
            # Set swapped context as the new prompt
            sample.prompt = state.tokenizer.apply_chat_template(
                sample.messages, add_generation_prompt=True, tokenize=False, tools=tool_set
            )
            initial_prompt_len = len(sample.tokens) # Reset baseline
            
            # W&B Metrics: 记录 swap 后的详细信息（复用已计算的 sample.tokens，无需重复 tokenize）
            tokens_after = len(sample.tokens)
            swap_info = {
                "trajectory_id": trajectory_id,
                "turn_number": turn,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "compression_ratio": tokens_before / tokens_after if tokens_after > 0 else 0,
                "content_preview": _extract_content_preview(step_result.updated_message),
            }
            status.swap_out_infos.append(swap_info)
            
            continue
        
        # 6. Prepare for Next Turn (Standard Continuation)
        sample.messages = step_result.updated_message
        # Lynx: Shallow copy to avoid in-place modification
        observation = step_result.updated_message[-1].copy()
        # Inject Token Budget
        budget_msg = _inject_token_budget(
            sample, VIRTUAL_MEMORY, state.tokenizer,
            None, observation
        )
        observation['content'] += budget_msg
        
        # Tokenize ONLY the new observation to append to current_tokens
        # Note: We take the last message which is the Observation from Env
        # Lynx: this method is verified, it's exactly the new token when
        # tokenizing the entire message
        new_obs_tokens = state.tokenizer.apply_chat_template(
            [observation], 
            add_generation_prompt=True, 
            tokenize=True
        )
        
        # Update state
        sample.tokens += new_obs_tokens
        sample.rollout_log_probs += [0.0] * len(new_obs_tokens)
        sample.loss_mask += [0] * len(new_obs_tokens)

        # Check Token Limit
        next_input_len = len(sample.tokens)
        if next_input_len > MAX_LEN:
            sample.status = Sample.Status.TRUNCATED
            break
        
    env.close()
    # if collected_samples:
    #     for _sample in collected_samples:
    #         print(state.tokenizer.decode(_sample.tokens))
    
    # --- Finalization ---
    if not collected_samples or not status.task_finished:
        # If no trajectory generated or the task is not finished,
        # the collected_samples may contain unneccessary swap out samples
        # we do not intend to train them
        print(f"Task not finished or max turn exceeded.")
        error_sample = _create_error_result(sample, trajectory_id, prompt_group_id, "No samples generated")
        collected_samples = error_sample
    else:
        ########################
        # Idea 2: Finer-reward #
        ########################
        # 1. task success
        success_reward = 1.0 if status.trajectory_success else 0.0
        # 2. swap length
        # if len(collected_samples) > 1:
        #     avg_compression_ratio = sum([info["compression_ratio"] for info in status.swap_out_infos]) / len(status.swap_out_infos)
        #     # compression ratio is usually 8-9, let's design a Gaussian function that the mean is 3
        #     target = 3.0
        #     sigma = 3.0
        #     ratio_reward = np.exp(-((avg_compression_ratio - target) ** 2) / (2 * sigma ** 2))
        #     r_tool = 0.1 * ratio_reward - 0.1
        # else:
        #     r_tool = 0.0
        # 3. Force the model to use less turns
        # progress = loop_state.turn / MAX_TURNS
        # alpha = 1
        # efficiency_reward = (1.0 - progress) ** alpha
        
        # if success_reward == 1.0:
        #     final_reward = 1.0 + efficiency_reward # final_reward = 1.0 + r_tool + efficiency_reward
        # else:
        #     final_reward = 0.0 # final_reward = r_tool
        # print("="*100)
        # print(f"Traj {trajectory_id}, reward: {final_reward}, success: {success_reward},  efficiency: {efficiency_reward}") # r_tool: {r_tool},
        # print("="*100)
        # status.reward_dict = {
        #     "success": success_reward,
        #     "efficiency": efficiency_reward
        # } # "r_tool": r_tool,
        # Normalize reward across samples in trajectory
        final_reward = 1.0 if status.trajectory_success else 0.0
        for s in collected_samples:
            s.reward = final_reward
            # The last output info is enough
            if "weight_version" in output["meta_info"]:
                s.weight_versions.append(output["meta_info"]["weight_version"])
                
    status.samples = collected_samples
    
    return status

def compute_group_advantages(
    samples: List[List[Sample]],
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


class GroupResult:
    def __init__(self, steps, metrics_delta, swap_infos):
        self.steps = steps
        self.metrics_delta = metrics_delta
        self.swap_infos = swap_infos

async def _process_single_group(
    rollout_id: int, 
    data_source: GymRolloutDataSource, 
    args, 
    state: GenerateState
) -> GroupResult:
    """
    Worker function: Handles generation for ONE group.
    """
    # 1. Fetch Prompt
    prompt_groups = data_source.get_samples(1)
    if not prompt_groups:
        return GroupResult([], {}, [])
    
    prompt_group = prompt_groups[0]
    prompt_group_id = f"group_{rollout_id}_{uuid.uuid4().hex[:8]}"
    
    # 2. Launch Generation Tasks
    tasks = []
    for i, prompt_sample in enumerate(prompt_group):
        trajectory_id = f"{prompt_group_id}_traj_{i}"
        task = generate(
            args,
            prompt_sample,
            trajectory_id=trajectory_id,
            prompt_group_id=prompt_group_id,
            sampling_params=state.sampling_params.copy()
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)

    # 3. Extract Raw Data
    trajectory_results = [raw.samples for raw in results]
    
    # Metrics collection helpers
    memory_tool_times = [raw.memory_tool_times for raw in results]
    success = [int(raw.trajectory_success) for raw in results]
    swap_reward_list = [raw.reward_dict.get("r_tool", 0) for raw in results]
    efficiency_reward_list = [raw.reward_dict.get("efficiency", 0) for raw in results]
    swap_success = [int(raw.trajectory_success and (len(raw.samples) > 1)) for raw in results]
    swap_out_infos_list = [raw.swap_out_infos for raw in results]
    
    # Flatten swap infos
    group_swap_infos = []
    for infos in swap_out_infos_list:
        group_swap_infos.extend(infos)

    # 4. Compute Advantages (GRPO Logic)
    advs = compute_group_advantages(trajectory_results, args)
    for samples, adv in zip(trajectory_results, advs):
        num_trajs = len(samples)
        for samp in samples:
            samp.advantage = adv / num_trajs
    
    # 5. Add to Buffer (This triggers the dynamic filter!)
    all_steps = [step for trajectory_steps in trajectory_results for step in trajectory_steps]
    
    # Important: The buffer might DROP this group if variance is 0
    if all_steps:
        data_source.add_steps_to_buffer(all_steps)

    # 6. Package Metrics
    metrics_delta = {
        "total_memory_tool_times": sum(memory_tool_times),
        "success_times": sum(success),
        "swap_success_times": sum(swap_success),
        "number_of_samples": len(results),
        "swap_reward": sum(swap_reward_list),
        "efficiency_reward": sum(efficiency_reward_list),
        "number_of_swap_rollouts": sum([int(len(raw.samples) > 1) for raw in results]),
        "total_trajectory_count": len(results),
        "steps_generated_count": len(all_steps),
        "group_size": len(prompt_group)
    }

    return GroupResult(all_steps, metrics_delta, group_swap_infos)


async def generate_rollout_async(args, rollout_id: int, data_source: GymRolloutDataSource) -> List[Sample]:
    """
    Optimized async rollout with precise task accounting to prevent overshooting.
    """
    state = GenerateState(args)
    
    # Metrics Accumulators
    acc_metrics = {
        "total_memory_tool_times": 0, "success_times": 0, "swap_success_times": 0,
        "number_of_samples": 0, "number_of_swap_rollouts": 0, "swap_reward": 0,
        "efficiency_reward": 0, "total_trajectory_count": 0
    }
    all_swap_infos: List[dict] = []
    
    pending_tasks = set()

    # --- Wrapper to manage semaphore ---
    async def sem_task():
        async with state.semaphore:
            return await _process_single_group(rollout_id, data_source, args, state)

    print(f"Starting rollout. Target: {args.num_training_groups} groups. Concurrency: {args.sglang_server_concurrency}")
    
    # --- Main Producer-Consumer Loop ---
    while True:
        # Check status
        current_groups_in_buffer = data_source.get_step_buffer_num_groups()
        groups_needed = args.num_training_groups - current_groups_in_buffer
        
        # STOP CONDITION: We have enough in buffer AND no pending tasks
        # (We must wait for pending tasks to finish to ensure we don't exit early 
        # while some are still processing, although arguably we could exit if buffer is full)
        if groups_needed <= 0 and not pending_tasks:
            break
            
        # 1. Fill the pipeline precisely
        # We only spawn if:
        #   A) We have concurrency slots open
        #   B) (Pending + Current) < Target. This prevents the overshooting bug.
        while len(pending_tasks) < (args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine):
            # Re-calculate inside loop because pending_tasks grows
            potential_total = current_groups_in_buffer + len(pending_tasks)
            if potential_total >= args.num_training_groups:
                break
                
            task = asyncio.create_task(sem_task())
            pending_tasks.add(task)
        
        # If we broke the inner loop because potential_total reached target, 
        # but have 0 pending tasks (and target not met), it implies an error state or empty source
        if not pending_tasks and groups_needed > 0:
            raise ValueError("Warning: No pending tasks and target not reached (Source exhausted?)")

        # 2. Wait for at least one task to finish
        done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # 3. Process results
        for task in done:
            try:
                result = task.result()
                
                # Update Metrics
                for k, v in result.metrics_delta.items():
                    if k in acc_metrics.keys():
                        acc_metrics[k] += v
                all_swap_infos.extend(result.swap_infos)
                
                print(
                    f"Generated {result.metrics_delta['steps_generated_count']} steps. | "
                    f"Buffer Groups: {data_source.get_step_buffer_num_groups()}/{args.num_training_groups} | "
                    f"Buffer Size: {data_source.get_step_buffer_length()}"
                )
                      
            except Exception as e:
                print(f"Error in rollout task: {e}")
                import traceback
                traceback.print_exc()
        
        # Loop continues... 
        # If the task that just finished was DROPPED by buffer, 'current_groups_in_buffer' won't increase.
        # 'pending_tasks' decreased by 1.
        # So 'potential_total' decreases by 1, allowing the loop to spawn a replacement task.

    # --- Post-Processing ---
    final_samples = data_source.get_complete_traj(args.num_training_groups)
    
    # Padding logic (Redundant check removed, safe padding implementation)
    original_len = len(final_samples)
    remainder = len(final_samples) % args.global_batch_size
    if remainder != 0:
        pad_len = args.global_batch_size - remainder
        pad_sample = _create_error_result(
            final_samples[0], 
            final_samples[0].metadata["trajectory_id"], 
            final_samples[0].metadata["prompt_group_id"],
            "pad"
        )
        final_samples.extend(pad_sample * pad_len)
        print(f"Original length {original_len}, padded to {len(final_samples)}")

    # Calculate Final Stats
    debug_sample = next((s for s in final_samples if len(s.messages) > 0), None)
    valid_samples = sum(1 for s in final_samples if len(s.messages) > 0)
    
    if debug_sample:
        print(f"Prompt: {debug_sample.prompt}")
        print(f"Response: {debug_sample.response}")

    def safe_div(n, d): return n / d if d > 0 else 0

    success_rate = safe_div(acc_metrics["success_times"], acc_metrics["number_of_samples"])
    swap_success_rate = safe_div(acc_metrics["swap_success_times"], acc_metrics["number_of_swap_rollouts"])
    
    valid_for_stats = [s for s in final_samples if s.metadata.get("error", "") != "pad"]
    true_reward = torch.tensor([s.reward for s in valid_for_stats], dtype=torch.float).mean().item() if valid_for_stats else 0.0
    true_adv = torch.tensor([s.advantage for s in valid_for_stats], dtype=torch.float).mean().item() if valid_for_stats else 0.0
    
    trajectory_ids = {s.metadata["trajectory_id"] for s in final_samples}
    prompt_group_ids = {s.metadata["prompt_group_id"] for s in final_samples}
    total_steps = len(final_samples)

    swap_out_sample_count = sum(1 for s in final_samples if s.metadata.get("context_modified", False))
    
    avg_swap_content_length = 0.0
    avg_compression_ratio = 0.0
    avg_swap_turn_number = 0.0
    
    if all_swap_infos:
        avg_swap_content_length = safe_div(sum(i["tokens_after"] for i in all_swap_infos), len(all_swap_infos))
        avg_compression_ratio = safe_div(sum(i["compression_ratio"] for i in all_swap_infos), len(all_swap_infos))
        avg_swap_turn_number = safe_div(sum(i["turn_number"] for i in all_swap_infos), len(all_swap_infos))

    swap_frequency_per_trajectory = safe_div(acc_metrics["total_memory_tool_times"], acc_metrics["total_trajectory_count"])
    sampled_trajectory = _sample_random_trajectory(final_samples, state.tokenizer)

    metrics = {
        "rollout/success_rate": success_rate,
        "rollout/swap_success_rate": swap_success_rate,
        "rollout/swap_reward": safe_div(acc_metrics["swap_reward"], acc_metrics["number_of_swap_rollouts"]),
        "rollout/efficiency_reward": safe_div(acc_metrics["efficiency_reward"], acc_metrics["success_times"]),
        "rollout/true_reward": true_reward,
        "rollout/true_advantage": true_adv,
        "rollout/memory_tool_times": acc_metrics["total_memory_tool_times"],
        "rollout/num_trajectories": len(trajectory_ids),
        "rollout/num_prompt_groups": len(prompt_group_ids),
        "rollout/avg_steps_per_trajectory": safe_div(total_steps, len(trajectory_ids)),
        "rollout/valid_samples_ratio": safe_div(valid_samples, total_steps),
        "rollout/swap_out_sample_ratio": safe_div(swap_out_sample_count, valid_samples),
        "rollout/avg_swap_content_length": avg_swap_content_length,
        "rollout/swap_compression_ratio": avg_compression_ratio,
        "rollout/swap_frequency_per_trajectory": swap_frequency_per_trajectory,
        "rollout/avg_swap_turn_number": avg_swap_turn_number,
        "_swap_infos": all_swap_infos[:50],
        "_sampled_trajectory": sampled_trajectory
    }

    return RolloutFnTrainOutput(samples=final_samples, metrics=metrics)



# async def generate_rollout_async(args, rollout_id: int, data_source: GymRolloutDataSource) -> List[Sample]:
#     """
#     异步生成rollout数据，返回List[Sample]
#     """
#     state = GenerateState(args)
#     target_size = args.global_batch_size
#     total_memory_tool_times = 0
#     success_times = 0
#     swap_success_times = 0
#     number_of_samples = 0
#     number_of_swap_rollouts = 0
#     swap_reward = 0
#     efficiency_reward = 0
    
#     # W&B Swap Out Metrics: 在 while 循环外部初始化
#     all_swap_infos: List[dict] = []
#     total_trajectory_count = 0  # 记录真实的 trajectory 数量
    
#     if args.train_complete_traj:
#         assert args.num_training_groups is not None, f"Should set args.num_training_groups when training with complete trajectories!"
#     def traj_level_target():
#         return data_source.get_step_buffer_length() < target_size
#     def group_level_target():
#         return data_source.get_step_buffer_num_groups() < args.num_training_groups
    
#     condition_func = traj_level_target if not args.train_complete_traj else group_level_target
    
#     while condition_func():
#         # get just one sample, but this sample is repeated for n_samples_per_prompt times
#         # for group generation. Note that, the original buffer in SLIME is useless.
#         prompt_groups = data_source.get_samples(1)
#         if not prompt_groups:
#             raise ValueError("No samples generated")
        
#         prompt_group = prompt_groups[0]  # [Sample1, Sample2, ...] (n_samples_per_prompt个)
        
#         # 为这个prompt group生成一个唯一ID
#         prompt_group_id = f"group_{rollout_id}_{uuid.uuid4().hex[:8]}"
        
#         # Note that this loop submits only one group
#         # async inside the group, but sync between groups...
#         tasks = []
#         for i, prompt_sample in enumerate(prompt_group):
#             trajectory_id = f"{prompt_group_id}_traj_{i}"
            
#             task = generate(
#                 args,
#                 prompt_sample,
#                 trajectory_id=trajectory_id,
#                 prompt_group_id=prompt_group_id,
#                 sampling_params=state.sampling_params.copy()
#             )
#             tasks.append(task)
        
#         # 等待所有trajectories完成
#         results = await asyncio.gather(*tasks)
#         trajectory_results = [raw.samples for raw in results]
#         memory_tool_times = [raw.memory_tool_times for raw in results]
#         success = [int(raw.trajectory_success) for raw in results]
#         swap_reward_list = [raw.reward_dict.get("r_tool", 0) for raw in results]
#         efficiency_reward_list = [raw.reward_dict.get("efficiency", 0) for raw in results]
#         swap_success = [int(raw.trajectory_success and (len(raw.samples) > 1)) for raw in results]
#         # W&B Swap Out Metrics: 收集 swap_out_infos
#         swap_out_infos_list = [raw.swap_out_infos for raw in results]
        
#         total_memory_tool_times += sum(memory_tool_times)
#         success_times += sum(success)
#         swap_success_times += sum(swap_success)
#         number_of_samples += len(results)
#         swap_reward += sum(swap_reward_list)
#         efficiency_reward += sum(efficiency_reward_list)
#         number_of_swap_rollouts += sum([int(len(raw.samples) > 1) for raw in results])
#         total_trajectory_count += len(results)  # 同步累加，与 total_memory_tool_times 分母一致
        
#         # W&B Swap Out Metrics: 累加而不是重置
#         for infos in swap_out_infos_list:
#             all_swap_infos.extend(infos)
#         # breakpoint()
#         # trajectory_results: List[List[Sample]], one group in GRPO
#         # The first List is N trajectories, the second List represents
#         # possible swap out
#         # Compute group advantage right here
#         advs = compute_group_advantages(trajectory_results, args)
#         for samples, adv in zip(trajectory_results, advs):
#             ####################################
#             # IMPORTANT! Experimental Feature! #
#             ####################################
#             # Lynx: Trying to fix the reward bias
#             num_trajs = len(samples)
#             for samp in samples:  # assign identical advantage to swap out samples
#                 samp.advantage = adv / num_trajs
                
#             ####################################################
#             # Idea 1: Final answer should gain equal advantage #
#             # Verified: Not work                               #
#             ####################################################
#             # samples[-1].advantage = adv
        
#         # flatten all trajectories
#         all_steps = [step for trajectory_steps in trajectory_results for step in trajectory_steps]
        
#         # 将生成的steps放入buffer
#         if all_steps:
#             data_source.add_steps_to_buffer(all_steps)
        
#         print(f"Generated {len(all_steps)} steps from {len(prompt_group)} trajectories, "
#               f"buffer size: {data_source.get_step_buffer_length()}")
    
#     # 从buffer取出需要的数量
#     if not args.train_complete_traj:
#         final_samples = data_source.get_steps_from_buffer(target_size)
#     else:
#         final_samples = data_source.get_complete_traj(args.num_training_groups)
#         original_len = len(final_samples)
#         # Pad to multiplier of global batch size, is it safe?
#         remainder = len(final_samples) % args.global_batch_size
#         if not remainder == 0:
#             pad_len = args.global_batch_size - remainder
#             pad_sample = _create_error_result(
#                 final_samples[0], 
#                 final_samples[0].metadata["trajectory_id"], 
#                 final_samples[0].metadata["prompt_group_id"],
#                 "pad"
#             )
#             final_samples.extend(pad_sample * pad_len)
#             print(f"Original length {original_len}, padded to {len(final_samples)}")
            
#     debug_sample = None
#     valid_samples = 0
#     for sample in final_samples:
#         if len(sample.messages) > 0:
#             # find the first valid sample
#             if not debug_sample:
#                 debug_sample = sample
#             # else count valid samples (with real trajectories)
#             valid_samples += 1
#     print(f"Prompt: {debug_sample.prompt}")
#     print(f"Response: {debug_sample.response}")
#     success_rate = success_times / number_of_samples if number_of_samples > 0 else 0
#     swap_success_rate = swap_success_times / number_of_swap_rollouts if number_of_swap_rollouts > 0 else 0
#     true_reward = torch.tensor([samp.reward for samp in final_samples if not samp.metadata.get("error", "") == "pad"], dtype=torch.float).mean().item()
#     true_adv = torch.tensor([samp.advantage for samp in final_samples if not samp.metadata.get("error", "") == "pad"], dtype=torch.float).mean().item()
#     trajectory_ids = set()
#     prompt_group_ids = set()
#     total_steps = len(final_samples)
#     for sample in final_samples:
#         trajectory_ids.add(sample.metadata["trajectory_id"])
#         prompt_group_ids.add(sample.metadata["prompt_group_id"])
        
#     # W&B Swap Out Metrics: 计算 swap out 相关指标
#     swap_out_sample_count = sum(
#         1 for s in final_samples if s.metadata.get("context_modified", False)
#     )
    
#     avg_swap_content_length = 0.0
#     avg_compression_ratio = 0.0
#     avg_swap_turn_number = 0.0
#     if all_swap_infos:
#         avg_swap_content_length = sum(info["tokens_after"] for info in all_swap_infos) / len(all_swap_infos)
#         avg_compression_ratio = sum(info["compression_ratio"] for info in all_swap_infos) / len(all_swap_infos)
#         # 平均在第几个 turn 发生 swap
#         avg_swap_turn_number = sum(info["turn_number"] for info in all_swap_infos) / len(all_swap_infos)
    
#     # 每条轨迹平均 swap 次数（使用 total_trajectory_count 确保分母一致）
#     swap_frequency_per_trajectory = (
#         total_memory_tool_times / total_trajectory_count if total_trajectory_count > 0 else 0
#     )
    
#     # 随机采样一个完整轨迹用于质量监控
#     sampled_trajectory = _sample_random_trajectory(final_samples, state.tokenizer)
    
#     metrics = {
#         "rollout/success_rate": success_rate,
#         "rollout/swap_success_rate": swap_success_rate,
#         # we have swap reward only when we swap, then we should normalize by number of swap rollouts
#         "rollout/swap_reward": swap_reward / number_of_swap_rollouts if number_of_swap_rollouts > 0 else 0,
#         # we have efficiency reward only in success samples
#         "rollout/efficiency_reward": efficiency_reward / success_times if success_times > 0 else 0,
#         "rollout/true_reward": true_reward,
#         "rollout/true_advantage": true_adv,
#         "rollout/memory_tool_times": total_memory_tool_times,
#         "rollout/num_trajectories": len(trajectory_ids),
#         "rollout/num_prompt_groups": len(prompt_group_ids),
#         "rollout/avg_steps_per_trajectory": total_steps / len(trajectory_ids) if trajectory_ids else 0,
#         "rollout/valid_samples_ratio": valid_samples / total_steps,
#         # W&B Swap Out Metrics: 新增指标
#         "rollout/swap_out_sample_ratio": swap_out_sample_count / total_steps if total_steps > 0 else 0,
#         "rollout/avg_swap_content_length": avg_swap_content_length,
#         "rollout/swap_compression_ratio": avg_compression_ratio,
#         "rollout/swap_frequency_per_trajectory": swap_frequency_per_trajectory,
#         "rollout/avg_swap_turn_number": avg_swap_turn_number
#     }
    
#     # 内部字段，传递给 _log_rollout_data 处理 W&B Table（以 _ 开头表示内部使用）
#     metrics["_swap_infos"] = all_swap_infos[:50]  # 限制最多 50 条
#     metrics["_sampled_trajectory"] = sampled_trajectory
#     # assert all(samp.reward is not None for samp in final_samples)
#     # assert all(samp.advantage is not None for samp in final_samples)
    
#     return RolloutFnTrainOutput(samples=final_samples, metrics=metrics)

def _call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class _MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: Optional[str]):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }

# TODO remove this temp function
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> Union[RolloutFnTrainOutput, RolloutFnEvalOutput]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    if evaluation:
        pass
    
    # 训练模式：生成steps
    return run(generate_rollout_async(args, rollout_id, data_source))
