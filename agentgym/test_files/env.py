import uuid
import time
import re
from typing import List, Dict, Any, Optional

"""from models import EnvInfo, Tool, ToolExecutionRecord, EnvConfig, StepResult, Messages
from docker_manager import DockerManager
from prompt_tool import PromptTool
from persistence import EnvPersistence"""
from agentgym.test_files.models import EnvInfo, Tool, ToolExecutionRecord, EnvConfig, StepResult, Messages
from agentgym.test_files.docker_manager import DockerManager
from agentgym.test_files.prompt_tool import PromptTool
from agentgym.test_files.persistence import EnvPersistence

class Env:
    def __init__(self):
        self.env_id: Optional[str] = None
        self.container_id: Optional[str] = None
        self.created: Optional[int] = None
        self.tools: List[Tool] = []
        self.tool_history: List[ToolExecutionRecord] = []
        self.turn: int = 0
        self.init_timestamp: Optional[int] = int(time.time() * 1000)
        self.close_timestamp: Optional[int] = None
        self._docker_manager = DockerManager()
        self._persistence = EnvPersistence()
    
    def reset(self, env_config: EnvConfig) -> Messages:
        env_id = str(uuid.uuid4())
        
        if self.container_id:
            try:
                self._docker_manager.remove_container(self.container_id)
            except Exception:
                pass
        
        container = self._docker_manager.create_container(env_config.image_name, f"env-{env_id}")
        container_id = container.id
        self._docker_manager.start_container(container_id)
        
        max_wait_time = 10.0
        poll_interval = 0.5
        start_time = time.time()
        
        tools = []
        system_prompt = ""
        
        while time.time() - start_time < max_wait_time:
            try:
                tool_list_data = self._docker_manager.get_container_tool_list(container_id)
                tools = PromptTool.parse_tools(tool_list_data)
                if tools:
                    system_prompt = PromptTool.construct_system_prompt(tool_list_data)
                    break
            except Exception:
                time.sleep(poll_interval)
        
        if not tools:
            try:
                self._docker_manager.remove_container(container_id)
            except Exception:
                pass
            raise Exception("container service did not become ready within 10 seconds or no valid tools found")
        
        self.env_id = env_id
        self.container_id = container_id
        self.created = int(time.time() * 1000)
        self.tools = tools
        self.tool_history = []
        self.turn = 0

        system_message = {"role": "system", "content": system_prompt}

        try:
            self._persistence.save_env(
                self.get_info(),
                self.tool_history,
                self.init_timestamp,
                None
            )
        except Exception:
            pass

        return [system_message]
    
    def step(self, messages: Messages) -> StepResult:
        if not self.env_id:
            observation = "environment not initialized. call reset() first."
            return StepResult(
                next_observation=[{"role": "user", "content": observation}],
                reward=0,
                done=False,
                modified_context=False
            )

        self.turn += 1

        if not messages:
            observation = "no messages provided"
            return StepResult(
                next_observation=[{"role": "user", "content": observation}],
                reward=0,
                done=False,
                modified_context=False
            )
        
        last_message = messages[-1]
        message_objects = [last_message]
        
        tool_calls = PromptTool.extract_tool_calls_from_messages(message_objects)
        if not tool_calls:
            observation = "no tool calls found in messages"
            return StepResult(
                next_observation=[{"role": "user", "content": observation}],
                reward=0,
                done=True, #done=False,
                modified_context=False
            )
        
        validation_errors = []
        tool_ids = set()
        
        for tool_call in tool_calls:
            if not tool_call.id:
                validation_errors.append(f"tool '{tool_call.name}' is missing required ID")
                continue
            
            if not re.match(r'^[0-9]$', tool_call.id):
                validation_errors.append(
                    f"tool '{tool_call.name}' has invalid id format: '{tool_call.id}' "
                    f"(must be a single digit 0-9)"
                )
                continue
            
            if tool_call.id in tool_ids:
                validation_errors.append(f"duplicate tool ID '{tool_call.id}' found in current request")
                continue
            
            tool_ids.add(tool_call.id)
        
        if validation_errors:
            observation = (
                "tool execution failed due to validation errors:\n" +
                "\n".join(f"- {err}" for err in validation_errors) +
                "\n\nplease ensure all tools have valid single-digit ids (0-9) and no duplicate ids in the same message."
            )
            return StepResult(
                next_observation=[{"role": "user", "content": observation}],
                reward=-15,
                done=False,
                modified_context=False
            )
        
        tools_to_execute = tool_calls

        successful_executed_tools = []
        failed_executed_tools = []
        
        for tool_call in tools_to_execute:
            try:
                tool_name = tool_call.name
                tool_params = {k: v for k, v in tool_call.model_dump().items()
                              if k not in ['id', 'name', 'retry']}
                
                tool_definition = self._get_tool_definition(tool_name)
                if not tool_definition:
                    failed_executed_tools.append({
                        "tool_name": tool_name,
                        "success": False,
                        "error": f"tool '{tool_name}' is not available in this environment",
                        "execution_id": tool_call.id,
                        "retry_count": tool_call.retry or 0
                    })
                    continue

                validation_result = PromptTool.validate_tool(tool_params, tool_definition)
                if not validation_result['valid']:
                    failed_executed_tools.append({
                        "tool_name": tool_name,
                        "success": False,
                        "error": f"invalid parameters: {', '.join(validation_result['errors'])}",
                        "execution_id": tool_call.id,
                        "retry_count": tool_call.retry or 0
                    })
                    continue

                # Use converted parameters for tool execution
                converted_params = validation_result['converted_params']
                tool_result = self._docker_manager.call_container_tool(
                    self.container_id, tool_name, converted_params
                )

                if tool_result.get('status') == 'error':
                    self._add_tool_execution_record(
                        tool_call.id, tool_name, converted_params,
                        {"error": tool_result.get('error', 'unknown error')},
                        False, None
                    )

                    failed_executed_tools.append({
                        "tool_name": tool_name,
                        "success": False,
                        "error": tool_result.get('error', 'unknown error'),
                        "execution_id": tool_call.id,
                        "retry_count": tool_call.retry or 0
                    })
                else:
                    self._add_tool_execution_record(
                        tool_call.id, tool_name, converted_params,
                        tool_result.get('result'), True,
                        tool_result.get('execution_time')
                    )

                    successful_executed_tools.append({
                        "tool_name": tool_name,
                        "success": True,
                        "result": tool_result.get('result'),
                        "execution_time": tool_result.get('execution_time'),
                        "execution_id": tool_call.id,
                        "retry_count": tool_call.retry or 0
                    })
                    
            except Exception as e:
                tool_params = {k: v for k, v in tool_call.model_dump().items()
                              if k not in ['id', 'name', 'retry']}
                self._add_tool_execution_record(
                    tool_call.id, tool_call.name, tool_params,
                    {"error": str(e)}, False
                )

                failed_executed_tools.append({
                    "tool_name": tool_call.name,
                    "success": False,
                    "error": str(e),
                    "execution_id": tool_call.id,
                    "retry_count": tool_call.retry or 0
                })

        success_count = len(successful_executed_tools)
        failed_count = len(failed_executed_tools)

        response_text = f"executed {len(tools_to_execute)} tools:\n"
        response_text += f"- {success_count} succeeded\n"
        if failed_count > 0:
            response_text += f"- {failed_count} failed\n"
        
        successful_tools = successful_executed_tools
        failed_tools = failed_executed_tools
        
        if successful_tools:
            response_text += "\nsuccessful executions:\n"
            for tool in successful_tools:
                description = self._generate_tool_result_description(
                    tool["tool_name"], True, tool.get("result")
                )
                response_text += f"- {tool['tool_name']} (id: {tool['execution_id']}): {description}\n"
        
        if failed_tools:
            response_text += "\nfailed executions:\n"
            for tool in failed_tools:
                description = self._generate_tool_result_description(
                    tool["tool_name"], False, None, tool.get("error")
                )
                response_text += f"- {tool['tool_name']} (id: {tool['execution_id']}): {description}\n"

        done = len(failed_executed_tools) == 0

        reward = self._calculate_reward(successful_executed_tools, failed_executed_tools, len(tool_calls))

        observation = response_text.strip()

        result = StepResult(
                next_observation=[{"role": "user", "content": observation}],
                reward=reward,
                done=done,
                modified_context=done
            )

        try:
            self._persistence.save_env(
                self.get_info(),
                self.tool_history,
                self.init_timestamp,
                self.close_timestamp
            )
        except Exception:
            pass

        return result

    def close(self) -> None:
        if not self.env_id:
            raise ValueError("environment not initialized, nothing to close")

        self.close_timestamp = int(time.time() * 1000)

        try:
            self._persistence.save_env(
                self.get_info(),
                self.tool_history,
                self.init_timestamp,
                self.close_timestamp
            )
        except Exception:
            pass

        try:
            self._docker_manager.remove_container(self.container_id)
        except Exception as e:
            raise RuntimeError(f"failed to remove container: {str(e)}") from e
        finally:
            self.env_id = None
            self.container_id = None
            self.created = None
            self.tools = []
            self.tool_history = []
            self.turn = 0

    def get_tool_list(self) -> Dict[str, Any]:
        if not self.env_id:
            raise ValueError("environment not initialized.")
        
        tool_list = self._docker_manager.get_container_tool_list(self.container_id)
        return {
            "env_id": self.env_id,
            **tool_list
        }
    
    def get_tool_history(self, limit: Optional[int] = None) -> Dict[str, Any]:
        if not self.env_id:
            raise ValueError("environment not initialized.")
        
        history = self.tool_history
        if limit and limit > 0:
            history = history[-limit:]
        
        return {
            "env_id": self.env_id,
            "tool_history": [record.model_dump() for record in history],
            "total_records": len(history)
        }

    def save_env_data(self) -> Optional[str]:
        if not self.env_id:
            return None

        try:
            return self._persistence.save_env(
                self.get_info(),
                self.tool_history,
                self.init_timestamp,
                self.close_timestamp
            )
        except Exception:
            return None

    def load_env_data(self, env_id: str, timestamp: Optional[int] = None) -> Optional[Dict[str, Any]]:
        return self._persistence.load_env(env_id, timestamp)

    
    @property
    def id(self) -> Optional[str]:
        return self.env_id
    
    def get_info(self) -> EnvInfo:
        if not self.env_id:
            raise ValueError("environment not initialized.")
        
        return EnvInfo(
            id=self.env_id,
            container_id=self.container_id,
            created=self.created,
            tools=self.tools,
            tool_history=self.tool_history
        )
    
    def _calculate_reward(self, successful_tools: List[Dict[str, Any]],
                         failed_tools: List[Dict[str, Any]],
                         total_tools: int) -> float:
        if total_tools == 0:
            return 0.0

        base_success_reward = 10.0
        base_failure_penalty = 5.0
        success_decay_factor = 0.8
        failure_growth_factor = 1.2
        perfect_bonus = 5.0 * total_tools
        turn_efficiency_factor = 0.97

        total_reward = 0.0

        max_retry_for_calculation = 5

        positive_reward = 0.0
        negative_reward = 0.0

        for tool in successful_tools:
            retry_count = min(tool.get('retry_count', 0), max_retry_for_calculation)
            success_reward = base_success_reward * (success_decay_factor ** retry_count)
            positive_reward += success_reward

        for tool in failed_tools:
            retry_count = min(tool.get('retry_count', 0), max_retry_for_calculation)
            failure_penalty = base_failure_penalty * (failure_growth_factor ** retry_count)
            negative_reward -= failure_penalty

        if len(failed_tools) == 0 and len(successful_tools) > 0:
            positive_reward += perfect_bonus

        positive_reward *= (turn_efficiency_factor ** (self.turn - 1))

        total_reward = positive_reward + negative_reward

        normalized_reward = total_reward / total_tools

        return round(normalized_reward, 2)

    def _add_tool_execution_record(self, execution_id: str, tool_name: str,
                                  params: Dict[str, Any], result: Any,
                                  success: bool, execution_time: Optional[int] = None):
        record = ToolExecutionRecord(
            id=execution_id,
            tool_name=tool_name,
            params=params,
            result=result,
            success=success,
            timestamp=int(time.time() * 1000),
            turn=self.turn,
            execution_time=execution_time
        )
        
        self.tool_history.append(record)
        
        if len(self.tool_history) > 1000:
            self.tool_history = self.tool_history[-1000:]
    
    def _get_tool_definition(self, tool_name: str) -> Optional[Tool]:
        return next((tool for tool in self.tools if tool.name == tool_name), None)
    
    def _generate_tool_result_description(self, tool_name: str, success: bool, 
                                        result: Any, error: Optional[str] = None) -> str:
        if not success:
            return f"failed with error: {error}"
        
        if tool_name == 'BashTool':
            exit_code = result.get('exitCode', 0) if result else 0
            description = f"executed successfully (exit code: {exit_code})"
            
            if result and (result.get('stdout') or result.get('stderr')):
                description += '\n'
                if result.get('stdout'):
                    description += f"stdout:\n{result['stdout']}"
                if result.get('stderr'):
                    if result.get('stdout'):
                        description += '\n'
                    description += f"stderr:\n{result['stderr']}"
            return description
        
        elif tool_name == 'ReadFileTool':
            if result and result.get('formattedContent'):
                line_count = result.get('lineCount', 0)
                return f"read file successfully ({line_count} lines):\n{result['formattedContent']}"
            elif result and result.get('fullContent'):
                line_count = result.get('lineCount', 0)
                return f"read file successfully ({line_count} lines):\n{result['fullContent']}"
            return 'read file successfully (no content returned)'
        
        elif tool_name == 'GrepTool':
            if result and result.get('lineCount', 0) > 0 and result.get('stdout'):
                return f"found {result['lineCount']} matches:\n{result['stdout']}"
            elif result and result.get('lineCount') == 0:
                return 'no matches found'
            return 'search completed successfully'
        
        elif tool_name == 'Edit':
            if result and result.get('success'):
                replacements = result.get('replacements', 0)
                description = f"file edited successfully ({replacements} replacements)"
                if result.get('diff'):
                    description += f"\nDiff:\n{result['diff']}"
                return description
            return 'file edited successfully'
        
        elif tool_name == 'AgentTool':
            if result and result.get('content'):
                return f"agent completed task successfully. Response:\n{result['content']}"
            elif result and result.get('success'):
                return 'agent task completed successfully'
            return 'agent task completed'
        
        else:
            if result:
                if hasattr(result, 'get'):
                    if result.get('content'):
                        return f"completed successfully. Content:\n{result['content']}"
                    elif result.get('output'):
                        return f"completed successfully. Output:\n{result['output']}"
                    elif result.get('response'):
                        return f"completed successfully. Response:\n{result['response']}"
                
                try:
                    import json
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                    return f"completed successfully. Result:\n{result_str}"
                except Exception:
                    pass
            
            return 'completed successfully'