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
from typing import Union, List
from collections import defaultdict

import ray
import torch
import wandb

from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from slime.utils.types import Sample
from slime.utils.wandb_utils import init_wandb_secondary
from agent.gym_rollout_datasource import GymRolloutDataSource

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
         
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

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

    def generate(self, rollout_id):
        """
        生成rollout数据
        """
        self.rollout_id = rollout_id
        start_time = time()
        
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            # 生成数据，现在直接返回List[Sample]
            data = self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False)
            
            # 不再需要flatten，因为已经是List[Sample]
            
            # 确保数据量是global_batch_size
            if len(data) != self.args.global_batch_size:
                print(f"Warning: expected {self.args.global_batch_size} samples, got {len(data)}")

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
        data = self._convert_samples_to_train_data(data)
        return Box(ray.put(data))

    def eval(self, rollout_id):
        """
        评估
        """
        if self.args.debug_train_only:
            return

        data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
        log_eval_data(rollout_id, self.args, data)

    def post_process_rewards(self, samples: List[Sample]):
        """
        处理rewards，支持gym环境的trajectory-based rewards
        按prompt_group_id计算GRPO advantages
        """
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # 按prompt_group_id分组计算advantages
            prompt_groups = defaultdict(list)
            for i, sample in enumerate(samples):
                group_id = sample.metadata.get("prompt_group_id", f"default_{i}")
                prompt_groups[group_id].append((i, sample))
            
            # 初始化advantages tensor
            advantages = torch.zeros(len(samples), dtype=torch.float32)
            
            for group_id, group_samples in prompt_groups.items():
                indices = [idx for idx, _ in group_samples]
                
                # 获取该group中每个trajectory的reward
                trajectory_rewards = {}
                for idx, sample in group_samples:
                    traj_id = sample.metadata.get("trajectory_id", f"traj_{idx}")
                    if traj_id not in trajectory_rewards:
                        trajectory_rewards[traj_id] = sample.reward
                
                # 转换为tensor计算advantages
                unique_rewards = list(trajectory_rewards.values())
                if len(unique_rewards) > 1:
                    rewards_tensor = torch.tensor(unique_rewards, dtype=torch.float32)
                    mean = rewards_tensor.mean()
                    group_advantages = rewards_tensor - mean
                    
                    if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                        std = rewards_tensor.std()
                        if std > 0:
                            group_advantages = group_advantages / std
                    
                    # 将advantages分配给每个sample
                    traj_to_advantage = dict(zip(trajectory_rewards.keys(), group_advantages.tolist()))
                    for idx, sample in group_samples:
                        traj_id = sample.metadata.get("trajectory_id", f"traj_{idx}")
                        advantages[idx] = traj_to_advantage[traj_id]
                else:
                    # 只有一个trajectory或所有rewards相同，advantages为0
                    for idx, _ in group_samples:
                        advantages[idx] = 0.0
            
            return raw_rewards, advantages.tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: List[Sample]):
        """
        Convert inference generated samples to training data.
        """
        raw_rewards, rewards = self.post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "rewards": rewards,  # 这里实际上是advantages
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        loss_masks = []
        for sample in samples:
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # 添加gym环境特有的metadata
        if samples and samples[0].metadata:
            if "trajectory_id" in samples[0].metadata:
                train_data["trajectory_ids"] = [sample.metadata["trajectory_id"] for sample in samples]
            
            if "prompt_group_id" in samples[0].metadata:
                train_data["prompt_group_ids"] = [sample.metadata["prompt_group_id"] for sample in samples]
            
            if "step_number" in samples[0].metadata:
                train_data["step_numbers"] = [sample.metadata["step_number"] for sample in samples]
            
            if "raw_reward" in samples[0].metadata:
                train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

            if "round_number" in samples[0].metadata:
                train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples and samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

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