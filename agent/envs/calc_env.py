import json
import re
from typing import Dict, Any
from agent.base.protocal import Messages, StepResult
from agent.base.env import Env
from agent.tools.schema import TestMathToolsSchema, register_tool_schema, register_tool_executor, MemoryToolsSchema
from agent.constant import TEST_MATH_SYS_PROMPT
from agent.tools.test_math_tools import TestMathTools
from agent.tools.memory_tools import MemoryTools


class CalcEnv(Env):
    """A test environment for calculating math problems.

    Args:
        env_config (dict): configuration for the environment.
    """
    def __init__(self, env_config: dict):
        self.question = env_config.get("question")
        self.current_answer = env_config.get("answer")
        self.conversation_history = []
        
        # 定义四个运算工具（不透露具体规则）
        self.tool_schema = register_tool_schema("all", TestMathToolsSchema(), MemoryToolsSchema())
        self.tools = register_tool_executor(TestMathTools(avaialble_tools="all"), MemoryTools(avaialble_tools="all"))
        
        # 系统提示词（不透露运算规则）
        self.system_prompt = TEST_MATH_SYS_PROMPT.replace("{{tools}}", json.dumps(self.tool_schema, indent=2,ensure_ascii=False))
        
    async def reset(self) -> Messages:
        """重置环境到初始状态"""
        self._current_step = 0
        
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user", 
                "content": self.question
            }
        ]
        
        return messages


    async def step(self, action: Messages) -> StepResult:
        """Parse and reward message from agent.

        Args:
            action (Messages): Total messages from agent.

        Returns:
            StepResult: Result of the current step, with the following fields:
                - next_observation: The next observation from the environment.
                - reward: The reward for the current step.
                - done: Whether the episode has ended.
                - modified_context: Whether the context was modified during the step.
        """
        self._current_step += 1
        
        # 获取最新的助手消息
        assistant_message = None
        for msg in reversed(action):
            if msg.get("role") == "assistant":
                assistant_message = msg
                break
                
        if not assistant_message:
            return StepResult(
                next_observation=action,
                reward=0.0,
                done=False,
                modified_context=False
            )
        
        content = assistant_message.get("content", "")
        
        # 检查是否有最终答案
        answer_match = re.search(r'<answer>(\d+)</answer>', content)
        if answer_match:
            predicted_answer = int(answer_match.group(1))
            reward = 1.0 if predicted_answer == self.current_answer else 0.0
            return StepResult(
                next_observation=action,
                reward=reward,
                done=True,
                modified_context=False
            )
        
        # 解析工具调用（自定义格式）
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
        if tool_call_match:
            try:
                tool_call_data = json.loads(tool_call_match.group(1))
                function_name = tool_call_data["name"]
                arguments = tool_call_data["arguments"]
                
                # 验证工具名称
                valid_tools = ["star_operation", "flower_operation", "moon_operation", "sun_operation", "summarize"]
                if function_name not in valid_tools:
                    print(f"[step] Invalid tool name: {function_name}")
                    return StepResult(
                        next_observation=action,
                        reward=0.0,
                        done=True,
                        modified_context=False
                    )
                
                # 执行工具函数
                if function_name != "summarize":
                    result = self._execute_tool(function_name, arguments)
                    
                    # 构建工具响应消息
                    tool_response = {
                        "role": "user",
                        "content": f"Tool {function_name} Result: {result}"
                    }
                    updated_messages = action + [tool_response]
                    return StepResult(
                        next_observation=updated_messages,
                        reward=0.0,
                        done=False,
                        modified_context=False
                    )
                else:
                    arguments["messages"] = action
                    updated_messages = self._execute_tool(function_name, arguments)
                
                    return StepResult(
                        next_observation=updated_messages,
                        reward=0.0,
                        done=False,
                        modified_context=True
                    )
                
            except Exception as e:
                print(f"[step] Failed to parse tool call: {e}")
                return StepResult(
                    next_observation=action,
                    reward=0.0,
                    done=False,
                    modified_context=False
                )
        
        # 如果没有工具调用也没有答案，继续对话
        return StepResult(
            next_observation=action,
            reward=0.0,
            done=False,
            modified_context=False
        )
    
    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> int:
        """执行工具函数"""
        if function_name in ["flower_operation", "moon_operation", "star_operation", "sun_operation"]:
            a = int(arguments["a"])
            b = int(arguments["b"])
            
            result = self.tools.forward(function_name, **{"a": a, "b": b})
        elif function_name == "summarize":
            text = arguments["text"]
            messages = arguments["messages"]
            result = self.tools.forward(function_name, **{"text": text, "messages": messages})
        else:
            raise ValueError(f"Invalid function name: {function_name}")
        
        return result
    
    async def close(self):
        """清理环境资源"""
        pass

