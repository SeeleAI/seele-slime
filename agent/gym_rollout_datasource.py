from typing import List
from collections import defaultdict
from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample


def filter_uniform_reward_groups(samples: List[Sample]) -> List[Sample]:
    """
    过滤掉reward完全相同的prompt groups（这些会导致advantages为0）
    """
    if not samples:
        return samples
    
    # 按prompt_group_id分组
    prompt_groups = defaultdict(list)
    for sample in samples:
        group_id = sample.metadata.get("prompt_group_id", "unknown")
        prompt_groups[group_id].append(sample)
    
    filtered_samples = []
    filtered_count = 0
    
    for group_id, group_samples in prompt_groups.items():
        # 获取该group所有trajectory的rewards
        trajectory_rewards = {}
        for sample in group_samples:
            traj_id = sample.metadata.get("trajectory_id", "unknown")
            if traj_id not in trajectory_rewards:
                trajectory_rewards[traj_id] = sample.reward
        
        # 检查是否所有rewards都相同
        unique_rewards = set(trajectory_rewards.values())
        if len(unique_rewards) > 1:  # rewards不全相同
            filtered_samples.extend(group_samples)
        else:
            filtered_count += len(group_samples)
            if len(unique_rewards) == 1:
                print(f"Filtered group {group_id}: all {len(trajectory_rewards)} trajectories "
                      f"have same reward {unique_rewards.pop():.4f}, {len(group_samples)} steps removed")
    
    if filtered_count > 0:
        print(f"Total filtered {filtered_count} samples with uniform rewards")
    
    return filtered_samples


class GymRolloutDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        # 专门存储生成的steps
        self.step_buffer = []
    
    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """
        获取原始prompt数据（用于生成），保持父类行为
        返回List[List[Sample]]，其中每个内层list是n_samples_per_prompt个samples
        """
        # 只调用父类方法获取原始数据，不涉及step_buffer
        return super().get_samples(num_samples)
    
    def add_steps_to_buffer(self, steps: List[Sample]):
        """
        将生成的steps添加到step_buffer
        可选进行zero advantage过滤
        """
        if getattr(self.args, 'filter_zero_advantage', False):
            steps = filter_uniform_reward_groups(steps)
        
        self.step_buffer.extend(steps)
    
    def get_steps_from_buffer(self, num_steps: int) -> List[Sample]:
        """
        从step_buffer取出指定数量的steps
        """
        if len(self.step_buffer) < num_steps:
            # 如果buffer不够，返回所有的
            print(f"Warning: step_buffer only has {len(self.step_buffer)} steps, "
                  f"but requested {num_steps}")
            result = self.step_buffer.copy()
            self.step_buffer.clear()
            return result
        
        # 取出需要的数量
        result = self.step_buffer[:num_steps]
        self.step_buffer = self.step_buffer[num_steps:]
        return result
    
    def get_step_buffer_length(self) -> int:
        """
        获取step_buffer中的样本数量
        """
        return len(self.step_buffer)
    
    def save(self, rollout_id):
        """
        保存状态，包括step_buffer
        """
        super().save(rollout_id)
        # TODO: 可以选择保存step_buffer的状态
    
    def load(self, rollout_id=None):
        """
        加载状态
        """
        super().load(rollout_id)
        # 重置step_buffer
        self.step_buffer = []