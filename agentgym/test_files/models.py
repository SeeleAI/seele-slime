from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
import time

# Type definitions
Message = Dict[str, Union[str, List[Dict[str, Any]], List[int], None]]
Messages = List[Message]

class StepResult(BaseModel):
    """Environment step execution result"""
    next_observation: Messages
    reward: float
    done: bool
    modified_context: bool

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool

class Tool(BaseModel):
    name: str
    description: str
    params: List[ToolParameter]

class ToolCall(BaseModel):
    id: str
    name: str
    retry: Optional[int] = None

    class Config:
        extra = "allow"

class ToolExecutionRecord(BaseModel):
    id: str
    tool_name: str
    params: Dict[str, Any]
    result: Any
    success: bool
    timestamp: int
    turn: int
    execution_time: Optional[int] = None

class EnvInfo(BaseModel):
    id: str
    container_id: str
    created: int
    tools: List[Tool]
    tool_history: List[ToolExecutionRecord] = []

class EnvConfig(BaseModel):
    #image_name: str = "http-tools:1.4"
    #image_name: str = "http1.4-hello-world"
    #image_name: str = "http1.4-assign-seats" 
    #image_name: str = "http1.4-intrusion-detection"
    image_name: str


class StepInfo(BaseModel):
    env_id: str
    turn: int
    input_messages: List[Dict[str, str]]
    total_extracted_tools: List[ToolCall]
    successful_executed_tools: List[Dict[str, Any]]
    failed_executed_tools: List[Dict[str, Any]]
    observation: str
