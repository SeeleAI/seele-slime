from swesmith.profiles import registry
from utils import parse_tool_call, execute_in_container, parse_complete_signal
from constants import SYS_PROMPT, get_tool, get_user_prompt
from dataclasses import dataclass

@dataclass
class StepResult:
    """Environment step execution result"""
    observation: list[dict]
    reward: float
    done: bool
    modified_context: bool

class SWEEnv():
    def __init__(self, task_instance):
        rp = registry.get_from_inst(task_instance)
        container = rp.get_container(task_instance)
        
        
    def step(model_output):
        maybe_complete = parse_complete_signal(model_output)
        if maybe_complete:
            pass
        maybe_tool_call = parse_tool_call(model_output)
        if maybe_tool_call:
            pass