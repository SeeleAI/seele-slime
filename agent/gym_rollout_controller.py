# gym_rollout_controller.py
"""
# config.yaml
rollout:
  # 使用自定义的rollout函数
  rollout_function_path: "gym_rollout.generate_rollout"
  
  # gym环境相关配置
  max_turns: 10
  expected_steps_per_trajectory: 3.0  # 平均每个trajectory产生的steps数
  
  # GRPO相关
  advantage_estimator: "grpo"
  grpo_std_normalization: true
  rewards_normalization: true
  
  # 其他配置保持不变
  rollout_batch_size: 64  # 每次rollout产生的samples数
  n_samples_per_prompt: 4

training:
  global_batch_size: 32  # 训练batch大小
"""
import logging
from pathlib import Path
from time import time
from typing import Union, List, Optional
from collections import defaultdict

import ray
import torch
import wandb

from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from agent.base.protocal import MySample
from slime.utils.wandb_utils import init_wandb_secondary
from agent.gym_rollout_datasource import GymRolloutDataSource
from slime.rollout.sglang_rollout import GenerateState
from agent.rollout_producer_consumer import RolloutProducer, RolloutConsumer
from agent.gym_rollout import create_rollout_producer, create_rollout_consumer, shutdown_global_producer

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class GymRolloutController:
    """
    自定义的RolloutController，支持gym环境和trajectory->steps转换
    """

    def __init__(self, args, wandb_run_id):
        self.args = args
        init_wandb_secondary(args, wandb_run_id)

        # 使用自定义的数据源
        self.data_source = GymRolloutDataSource(args)

        # 加载生成函数
        self.generate_rollout = load_function(self.args.rollout_function_path) #改成gym_rollout.generate_rollout
        self.eval_generate_rollout = load_function(self.args.eval_function_path) #改成gym_rollout.generate_rollout
         
        # default is: slime.rollout.sglang_rollout.generate_rollout
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")
        
        # 异步生产者-消费者模式
        self.producer: Optional[RolloutProducer] = None
        self.consumer: Optional[RolloutConsumer] = None
        
        print("RolloutController initialized with async producer-consumer mode")

    def get_num_rollout_per_epoch(self):
        """
        计算每个epoch需要的rollout数量
        对于gym环境，这个计算方式需要调整
        """
        assert self.args.rollout_global_dataset
        
        # 原始计算方式
        dataset_size = len(self.data_source.dataset)
        
        # 估算每个prompt产生的平均steps数
        expected_steps_per_trajectory = getattr(self.args, 'expected_steps_per_trajectory', 2.0)
        
        # 计算总的expected steps
        total_trajectories_per_epoch = dataset_size * self.args.n_samples_per_prompt
        total_steps_per_epoch = int(total_trajectories_per_epoch * expected_steps_per_trajectory)
        
        # 计算需要的rollout数（每个rollout消费global_batch_size个steps）
        num_rollout_per_epoch = (total_steps_per_epoch + self.args.global_batch_size - 1) // self.args.global_batch_size
        
        print(f"Dataset size: {dataset_size}")
        print(f"Expected steps per trajectory: {expected_steps_per_trajectory}")
        print(f"Total expected steps per epoch: {total_steps_per_epoch}")
        print(f"Calculated rollouts per epoch: {num_rollout_per_epoch}")
        
        return num_rollout_per_epoch

    async def start_producer(self):
        """启动生产者（异步方法）"""
        if self.producer is None:
            self.producer = await create_rollout_producer(self.args, self.data_source)
            print("Producer started in RolloutController")
            
    async def create_consumer(self) -> RolloutConsumer:
        """创建消费者（异步方法），支持预填充机制"""
        self.consumer = await create_rollout_consumer(self.args)
        print("Consumer created and queue pre-filled in RolloutController")
        return self.consumer
            
    async def stop_producer(self):
        """停止生产者（异步方法）"""
        if self.producer:
            await shutdown_global_producer()
            self.producer = None
            print("Producer stopped in RolloutController")
    
    async def drain_remaining_data(self) -> List[MySample]:
        """
        清空队列中剩余的数据，确保没有数据浪费
        在训练结束后调用
        
        Returns:
            List[MySample]: 剩余的所有数据
        """
        if self.consumer is None:
            print("No consumer available to drain remaining data")
            return []
        
        print("Draining remaining data from queue...")
        remaining_data = await self.consumer.drain_remaining_data()
        print(f"Drained {len(remaining_data)} remaining samples")
        return remaining_data

    async def generate(self, rollout_id):
        """
        生成rollout数据
        - 使用阻塞式获取，确保返回完整批次
        - 预填充机制确保训练开始时不等待
        - 队列空时协程挂起，不消耗 CPU
        """
        self.rollout_id = rollout_id
        start_time = time()
        
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )["samples"]
            data = [MySample.from_dict(sample) for sample in data]
        else:
            # 从预填充队列立即获取数据
            if self.consumer is None:
                raise RuntimeError(
                    "Consumer not initialized. "
                    "Call start_producer() and create_consumer() before training."
                )
            
            # 阻塞式获取完整批次（队列空时等待，不消耗 CPU）
            print(f"Rollout {rollout_id}: fetching batch from queue")
            data = await self.consumer.get_batch(self.args.global_batch_size)
            
            # 改进后的 get_batch 总是返回完整批次（除非遇到结束信号）
            if not data:
                raise RuntimeError(
                    "No data available from consumer. "
                    "Producer has finished and no more data available."
                )
            
            # 数据量检查（只有遇到结束信号时才会不足）
            if len(data) < self.args.global_batch_size:
                if self.consumer.is_finished:
                    print(f"Info: got {len(data)} samples (last batch, producer finished)")
                else:
                    # 这不应该发生，因为 get_batch 会阻塞等待完整批次
                    print(f"Warning: got {len(data)} samples but producer not finished, "
                          f"this should not happen with blocking get_batch")

        # 保存debug数据
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=self.rollout_id))
            print(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    rollout_id=self.rollout_id,
                    samples=[sample.to_dict() for sample in data],
                ),
                path,
            )
        
        # 记录日志
        log_rollout_data(rollout_id, self.args, data, time() - start_time)
        
        # 转换为训练数据
        data = self._convert_messages_to_train_data(data)
        return Box(ray.put(data))

    def eval(self, rollout_id):
        """
        评估
        """
        if self.args.debug_train_only:
            return

        data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
        log_eval_data(rollout_id, self.args, data)
    
    def _create_loss_mask_from_messages(self, messages: List[dict[str, str]]) -> tuple[List[int], int]:
        """Create loss mask for a message input

        Args:
            messages (List[dict[str, str]]): multi-turn conversation messages

        Returns:
            tuple[List[int], int]: loss mask, 0 for user/sys input, 1 for assistant output, and the response length
        """
        state = GenerateState(self.args)
        if not messages:
            return [], 0
        # all_input_ids = state.tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=False, tokenize=True
        # )
        all_input_ids = self._get_input_ids_from_messages(messages)

        # Start with zeros for the whole conversation token length.
        loss_mask = [0] * len(all_input_ids)

        def prefix_token_len(msgs_prefix: List[dict[str, str]], add_gen_prompt: bool) -> int:
            """Helper: returns length of tokenized chat template for a prefix of messages."""
            return len(
                state.tokenizer.apply_chat_template(msgs_prefix, add_generation_prompt=add_gen_prompt, tokenize=True)
            )

        # Mark assistant response ranges.
        # We compute, for each assistant message at index i, the token index where that assistant response begins
        # (using the chat template with generation prompt enabled for the prefix), then mark the length of the response.
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            # Token length of everything before the assistant's generated response.
            # We set add_generation_prompt=True so the tokenizer formats the prompt exactly like the generation-time input.
            start = prefix_token_len(messages[:idx], add_gen_prompt=True)

            # Token length for this assistant message content (+1 for EOS as in original code).
            assistant_content_tokens = len(state.tokenizer.encode(msg["content"]))
            response_len = assistant_content_tokens + 1  # +1 for EOS token

            # Cap ranges to avoid indexing errors in case of slight mismatches.
            end = min(start + response_len, len(loss_mask))

            loss_mask[start:end] = [1] * (end - start)

        # Compute the prompt length used by the generator for the first user/sys turn (keeps original behavior).
        len_of_prompt = prefix_token_len(messages[:2], add_gen_prompt=True)

        # Generation portion length
        gen_len = len(all_input_ids) - len_of_prompt
        gen_len = max(0, gen_len)
        
        return loss_mask[len_of_prompt:], gen_len
    
    def _get_input_ids_from_messages(self, messages: List[dict[str, str]]):
        state = GenerateState(self.args)
        # Lynx: Since apply_chat_template may include an additional token after the assistant
        # output, we should take care of it.
        # e.g., <|im_start|>assistant\nBecause it was on the other side of the road.<|im_end|>\n
        # the last '\n' is not a generation of the assistant.
        # first apply chat template before the last assistant output
        if not messages:
            return []
        input_ids = state.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
        final_assistant_response = messages[-1]["content"] + state.tokenizer.eos_token
        input_ids = input_ids + state.tokenizer.encode(final_assistant_response)
        
        return input_ids
    
    def _format_rollout_logprobs(self, logprobs: List[List[float]], loss_mask: List[int]):
        """
        Formats a flattened list of log probabilities according to a loss mask.
        """
        if not logprobs or not loss_mask:
            return []
        # 1. Flatten the nested list of log probabilities.
        flat_logprobs = [prob for conv_probs in logprobs for prob in conv_probs]
        
        assert len(flat_logprobs) == sum(loss_mask), (
            f"Logprobs length {len(flat_logprobs)} does not match "
            f"loss_mask sum {sum(loss_mask)}"
        )
        
        # 3. Create an iterator to consume the logprobs sequentially.
        logprobs_iter = iter(flat_logprobs)
        
        # 4. Use a list comprehension: if the mask is 1, take the next logprob; otherwise, use 0.0.
        return [next(logprobs_iter) if mask_value else 0.0 for mask_value in loss_mask]

    def _convert_messages_to_train_data(self, samples: Union[list[MySample], list[list[MySample]]]):
        """Convert message based Samples into training data."""
        assert isinstance(samples, list), "samples must be a list"
        # assert isinstance(samples[0], MySample), f"samples must be a list of Sample, got{type(samples[0])}"
        
        input_ids = [self._get_input_ids_from_messages(sample.messages) for sample in samples]
        response_len = []
        loss_mask = []
        for sample in samples:
            loss_m, res_len = self._create_loss_mask_from_messages(sample.messages)
            response_len.append(res_len)
            loss_mask.append(loss_m)
        
        train_data = {
            "tokens": input_ids,
            "response_lengths": response_len,
            "rewards": [sample.advantage for sample in samples],  # Already calculated rewards during the rollout
            "raw_reward": [sample.reward for sample in samples],
            "truncated": [1 if sample.status == MySample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }
        train_data["loss_masks"] = loss_mask
        if samples and samples[0].metadata:
            if "trajectory_id" in samples[0].metadata:
                train_data["trajectory_ids"] = [sample.metadata["trajectory_id"] for sample in samples]
            
            if "prompt_group_id" in samples[0].metadata:
                train_data["prompt_group_ids"] = [sample.metadata["prompt_group_id"] for sample in samples]
            
            if "step_number" in samples[0].metadata:
                train_data["step_numbers"] = [sample.metadata["step_number"] for sample in samples]

            if "round_number" in samples[0].metadata:
                train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        # Lynx: We have assumsed that the LLM will always use EOS token as the stop token,
        # only then the log probs are aligned with the loss mask.
        # Lynx: NOTE: some samples still hit sum(loss_mask) != len(logprobs) because
        # Qwen sometimes output the EOS token used in pre-training, WTF?
        if samples and any(samp.rollout_log_probs for samp in samples):
            rollout_logprobs = []
            for sample, mask in zip(samples, loss_mask):
                try:
                    rollout_logprobs.append(self._format_rollout_logprobs(sample.rollout_log_probs, mask))
                except:
                    breakpoint()
                
            train_data["rollout_log_probs"] = rollout_logprobs

        if samples and samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        return train_data

    def save(self, rollout_id):
        """
        保存状态
        """
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        """
        加载状态
        """
        self.data_source.load(rollout_id)

    def get_buffer_stats(self):
        """
        获取buffer统计信息
        """
        stats = {
            "step_buffer_length": self.data_source.get_step_buffer_length(),
            "main_buffer_length": self.data_source.get_buffer_length(),
        }
        return stats

def log_eval_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    print(f"eval {rollout_id}: {log_dict}")
    if args.use_wandb:
        # 修改：现在每个rollout对应一个训练step
        log_dict["eval/step"] = rollout_id if not args.wandb_always_use_train_step else rollout_id
        wandb.log(log_dict)


def log_rollout_data(rollout_id, args, samples, rollout_time):
    if args.load_debug_rollout_data:
        return

    log_dict = {}
    response_lengths = [
        sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length 
        for sample in samples
    ]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus is not None:
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time if response_lengths else 0
    
    # 添加gym环境特有的统计
    if samples and samples[0].metadata:
        # 统计trajectory相关信息
        trajectory_ids = set()
        prompt_group_ids = set()
        total_steps = len(samples)
        
        for sample in samples:
            if "trajectory_id" in sample.metadata:
                trajectory_ids.add(sample.metadata["trajectory_id"])
            if "prompt_group_id" in sample.metadata:
                prompt_group_ids.add(sample.metadata["prompt_group_id"])
        
        log_dict["rollout/num_trajectories"] = len(trajectory_ids)
        log_dict["rollout/num_prompt_groups"] = len(prompt_group_ids)
        log_dict["rollout/avg_steps_per_trajectory"] = total_steps / len(trajectory_ids) if trajectory_ids else 0
    
    print(f"perf {rollout_id}: {log_dict}")
    
    if args.use_wandb:
        # 修改：现在每个rollout对应一个训练step
        log_dict["rollout/step"] = rollout_id if not args.wandb_always_use_train_step else rollout_id
        wandb.log(log_dict)