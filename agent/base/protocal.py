from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import uuid

Message = Dict[str, Union[str, List[Dict[str, Any]], List[int], None]]
Messages = List[Message]
@dataclass
class StepResult:
    """Environment step execution result"""
    next_observation: Messages
    reward: float
    done: bool
    modified_context: bool


@dataclass
class TrajectoryStep:
    """单个trajectory step，直接记录tokenized的ids"""
    prompt_ids: List[int]  # prompt部分的token ids
    response_ids: List[int]  # response部分的token ids  
    response_mask: List[int]  # response部分的mask (1=assistant, 0=tool/user)
    response_logprobs: Optional[List[float]] = None  # response部分的logprobs
    messages: List[Dict[str, Any]] = field(default_factory=list)  # 原始messages，用于debug
    reward: float = 0.0
    turn_number: int = 1
    
@dataclass
class Trajectory:
    """完整轨迹"""
    trajectory_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    generation_time: float = 0.0
    tool_time: float = 0.0
    total_time: float = 0.0
    done: bool = False
    
    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)
    
    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

