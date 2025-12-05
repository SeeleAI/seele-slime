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

from train_env_python import Env, EnvConfig
from dataclasses import dataclass, field
from rollout_buffer import GymRolloutDataSource
import uuid

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
        
def _create_error_result(original_sample: Sample, traj_id: str, pg_id: str, error_msg: str) -> List[Sample]:
    """Creates a dummy sample to return on critical failure."""
    return [Sample(
        index=original_sample.index,
        prompt=original_sample.prompt,
        tokens=[],
        rollout_log_probs=[],
        loss_mask=[],
        response="",
        response_length=0,
        reward=0.0,
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
    
def _inject_token_budget(sample: Sample, total_memory: int, tokenizer, messages: List[dict] = None, observation: dict = None):
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
        input_tokens = tokenizer.apply_chat_template(_message, tokenize=True, add_generation_prompt=True)
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
    
    env = Env()
    env_config = EnvConfig(image_name=sample.metadata["env_name"])
    print(f"Initializing Environment {env_config}")

    try:
        system_messages = env.reset(env_config)
    except Exception as e:
        print(f"Error resetting environment {sample.metadata['env_name']}: {e}")
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
        init_message = system_messages + [{"role": "user", "content": env.task_prompt}]
        budget_msg = _inject_token_budget(sample, VIRTUAL_MEMORY, state.tokenizer, init_message, None)
        init_message[-1]['content'] += budget_msg
        prompt_token_ids = state.tokenizer.apply_chat_template(
            init_message,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt = state.tokenizer.apply_chat_template(
            init_message,
            add_generation_prompt=True,
            tokenize=False,
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
        if env_duration > FORCE_DROP_TIME:
            print(f"Force drop: Env step took {env_duration}s > {FORCE_DROP_TIME}s")
            sample.status = Sample.Status.TRUNCATED
            loop_state.reward = 0.0
            break
        
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
            break
        
        # 5. Handle Context Swap (Memory Tool)
        if step_result.modified_context:
            # breakpoint()
            status.memory_tool_times += 1
            
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

            # Reset Loop State with new context, step_result.next_observation includes only the swapped context
            sample = _create_reset_sample(sample, step_result.next_observation)
            # Inject budget token again
            budget_msg = _inject_token_budget(sample, VIRTUAL_MEMORY, state.tokenizer, sample.messages, None)
            sample.messages[-1]['content'] += budget_msg
            # Re-tokenize entire new context
            sample.tokens = state.tokenizer.apply_chat_template(
                sample.messages, add_generation_prompt=True, tokenize=True
            )
            # Set swapped context as the new prompt
            sample.prompt = state.tokenizer.apply_chat_template(
                sample.messages, add_generation_prompt=True, tokenize=False
            )
            initial_prompt_len = len(sample.tokens) # Reset baseline
            continue
        
        # 6. Prepare for Next Turn (Standard Continuation)
        sample.messages = step_result.next_observation
        # Lynx: Shallow copy to avoid in-place modification
        observation = step_result.next_observation[-1].copy()
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
    
    # --- Finalization ---
    if not collected_samples or not status.task_finished:
        # If no trajectory generated or the task is not finished,
        # the collected_samples may contain unneccessary swap out samples
        # we do not intend to train them
        error_sample = _create_error_result(sample, trajectory_id, prompt_group_id, "No samples generated")
        collected_samples = error_sample
    else:
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


async def generate_rollout_async(args, rollout_id: int, data_source: GymRolloutDataSource) -> List[Sample]:
    """
    异步生成rollout数据，返回List[Sample]
    """
    state = GenerateState(args)
    target_size = args.global_batch_size
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
            
            task = generate(
                args,
                prompt_sample,
                trajectory_id=trajectory_id,
                prompt_group_id=prompt_group_id,
                sampling_params=state.sampling_params.copy()
            )
            tasks.append(task)
        
        # 等待所有trajectories完成
        results = await asyncio.gather(*tasks)
        trajectory_results = [raw.samples for raw in results]
        memory_tool_times = [raw.memory_tool_times for raw in results]
        success = [int(raw.trajectory_success) for raw in results]
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
    final_samples = data_source.get_steps_from_buffer(target_size)
    success_rate = success_times / number_of_samples if number_of_samples > 0 else 0
    trajectory_ids = set()
    prompt_group_ids = set()
    total_steps = len(final_samples)
    for sample in final_samples:
        trajectory_ids.add(sample.metadata["trajectory_id"])
        prompt_group_ids.add(sample.metadata["prompt_group_id"])
    metrics = {
        "rollout/success_rate": success_rate,
        "rollout/memory_tool_times": total_memory_tool_times,
        "rollout/num_trajectories": len(trajectory_ids),
        "rollout/num_prompt_groups": len(prompt_group_ids),
        "rollout/avg_steps_per_trajectory": total_steps / len(trajectory_ids) if trajectory_ids else 0
    }
    return RolloutFnTrainOutput(samples=final_samples, metrics=metrics)

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
