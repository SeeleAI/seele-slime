import json
import re
from typing import Dict, Any
from agent.base.protocal import Messages, StepResult
from agent.base.env import Env

class CalcEnv(Env):
    """数学运算环境"""
    
    def __init__(self, env_config):
        self.question = env_config.get("question")
        self.current_answer = env_config.get("answer")
        self.conversation_history = []
        
        # 定义四个运算工具（不透露具体规则）
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "star_operation",
                    "description": "执行星星运算（☆），输入两个整数，返回运算结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "description": "第一个操作数"
                            },
                            "b": {
                                "type": "integer", 
                                "description": "第二个操作数"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "flower_operation",
                    "description": "执行花朵运算（❀），输入两个整数，返回运算结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "description": "第一个操作数"
                            },
                            "b": {
                                "type": "integer",
                                "description": "第二个操作数"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moon_operation", 
                    "description": "执行月亮运算（☽），输入两个整数，返回运算结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "description": "第一个操作数"
                            },
                            "b": {
                                "type": "integer",
                                "description": "第二个操作数"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sun_operation",
                    "description": "执行太阳运算（☀），输入两个整数，返回运算结果", 
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "description": "第一个操作数"
                            },
                            "b": {
                                "type": "integer",
                                "description": "第二个操作数"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
        
        # 系统提示词（不透露运算规则）
        self.system_prompt = """你是一个数学计算助手，需要帮助解决包含特殊运算符的数学表达式。

你有以下四种特殊运算工具可以使用：
{{tools}}

工具调用格式：
```
<tool_call>
{"name": "工具名称", "arguments": {"a": 数字1, "b": 数字2}}
</tool_call>
```

示例：
用户：计算 5 ☆ 3 ❀ 2
助手：我需要逐步计算。首先计算 5 ☆ 3：
<tool_call>
{"name": "star_operation", "arguments": {"a": 5, "b": 3}}
</tool_call>

（工具返回结果后继续下一步计算）
计算规则：
- 表达式从左到右依次计算
- **每轮对话只能调用一个工具，否则直接失败**
- 需要逐步计算每个运算
- 最终答案请放在<answer>数字</answer>标签中
"""
        self.system_prompt = self.system_prompt.replace("{{tools}}", json.dumps(self.tools, indent=2,ensure_ascii=False))

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
        """执行一步"""
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
                valid_tools = ["star_operation", "flower_operation", "moon_operation", "sun_operation"]
                if function_name not in valid_tools:
                    return StepResult(
                        next_observation=updated_messages,
                        reward=0.0,
                        done=True,
                        modified_context=False
                    )
                
                # 执行工具函数
                result = self._execute_tool(function_name, arguments)
                
                # 构建工具响应消息
                tool_response = {
                    "role": "user",
                    "content": f"工具 {function_name} 执行结果：{result}"
                }
                
                # 添加到对话历史
                updated_messages = action + [tool_response]
                
                return StepResult(
                    next_observation=updated_messages,
                    reward=0.0,
                    done=False,
                    modified_context=False
                )
                
            except Exception as e:
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
            done=True,
            modified_context=False
        )
    
    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> int:
        """执行工具函数"""
        a = int(arguments["a"])
        b = int(arguments["b"])
        
        if function_name == "star_operation":
            return a + b - 1
        elif function_name == "flower_operation":
            return a * 2 + b
        elif function_name == "moon_operation":
            return (a + b) * 2
        elif function_name == "sun_operation":
            return a * b - a
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def close(self):
        """清理环境资源"""
        pass

