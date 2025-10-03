from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from agent.base.protocal import Messages, StepResult

class Env(ABC):
    """Base environment interface for GRPO training."""

    def __init__(self, env_config):
        """Initialize environment."""
        self.env_config = env_config
        self._current_step = 0
        self._trajectory_id = None

    @abstractmethod
    async def reset(self) -> Messages:
        """Reset environment to initial state.
        Args:
            turn:current turn
        Returns:
            Messages
        """
        pass

    @abstractmethod
    async def step(self, action: Messages) -> StepResult:
        """Execute one step in the environment.

        Args:
            action: Messages containing the conversation state with agent's action

        Returns:
            StepResult containing next_observation, reward, done, info
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up environment resources."""
        pass
