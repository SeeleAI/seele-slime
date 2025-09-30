from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import uuid
from enum import Enum

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


@dataclass
class MySample:
    """The sample generated"""

    index: Optional[int] = None
    # prompt
    prompt: Union[str, list[dict[str, str]]] = ""
    assistant_tokens: list[int] = field(default_factory=list)
    # single turn response
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: Optional[str] = None
    reward: Optional[Union[float, dict[str, Any]]] = None
    advantage: Optional[float] = None
    loss_mask: Optional[list[int]] = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: Optional[list[float]] = None  # Log probabilities from rollout engine
    # multi-turn messages
    messages: Messages = field(default_factory=list)
    end_of_turn: bool = False

    class Status(Enum):
        PENDING = "pending"   # 正在生成中
        COMPLETED = "completed"  # 完成
        TRUNCATED = "truncated"  # 超出max length被截断
        ABORTED = "aborted"  # 被打断

    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: Optional[dict] = None

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data["status"] = MySample.Status(data["status"])
        return MySample(**data)

    def get_reward_value(self, args) -> float:
        """return reward if args.reward_key is not specified, otherwise return reward[args.reward_key]

        Args:
            args (_type_): _description_

        Returns:
            float: reward value
        """
        return self.reward if not args.reward_key else self.reward[args.reward_key]
