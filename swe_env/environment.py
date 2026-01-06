from swesmith.profiles import registry
from swesmith.harness.eval import run_evaluation
from swe_env.utils import parse_tool_call, execute_in_container, parse_complete_signal
from swe_env.constants import SYS_PROMPT, get_tool, get_user_prompt
from dataclasses import dataclass, field
from typing import Dict, Any, List
import time
import re
import asyncio
import base64
import functools

def swap_context(summarize: dict, messages: list[dict], user_request: str) -> list[dict]:
    assert messages[0]['role'] == "system", f"Message[0] role {messages[0]['role']} is incorrect, should be system"
    system_prompt = messages[0]
    next_session = summarize["next_session_context"]
    think = summarize["think"]
    summary_content = (
f"""
# User Request
{user_request}

<Important> You just called ClearContextTool, check the message from Previous Context </Important>
# Previous Context
{next_session}
"""
    )
    new_message = [
        system_prompt,
        {"role": "user", "content": f"You called ClearContextTool, the new context is\n{summary_content}"}
    ]
    
    return new_message

def generate_patch(container):
    print("Generating patch...")
    cmd_list = [
        'cd /testbed',
        "git add -A",
        "git diff --cached > /testbed/model.patch",
        'cat /testbed/model.patch'
    ]
    for command in cmd_list:
        try:
            # Encode command to avoid shell escaping issues
            b64_cmd = base64.b64encode(command.encode('utf-8')).decode('utf-8')
            safe_cmd = f"bash -c 'timeout -k 5 40s bash -c \"echo {b64_cmd} | base64 -d | bash\"'"
            
            # Blocking call (no signal logic here)
            exit_code, output = container.exec_run(cmd=safe_cmd)
            
            # Decode output
            output_str = output.decode("utf-8", errors="replace")
            # print(f"\n[Tool Execution Completed] {command}")
            if exit_code != 0:
                print(f"Error generating patch: {output_str}")   
                break
                
            if command == cmd_list[-1]:
                # print(f"Patch:\n{output_str}")
                return output_str if output_str.strip() else None
            
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            break
    return None

async def run_blocking(func, *args, **kwargs):
    """
    Helper to run blocking (synchronous) functions in a thread pool.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

@dataclass
class StepResult:
    """Environment step execution result"""
    updated_message: list[dict]
    reward: float
    done: bool = field(default=False)
    success: bool = field(default=False)
    modified_context: bool = field(default=False)
    info: dict = field(default=None)

class SWEEnv:
    def __init__(self, task_instance: Dict, run_id: str):
        """
        Initialize the environment.
        :param task_instance: The specific task/issue to solve.
        :param registry: The registry object to fetch container providers (Dependency Injection).
        """
        self.task_instance = task_instance
        self.run_id = run_id
        self.rp = registry.get_from_inst(task_instance)
        print(f"Initializing container for task: {task_instance["instance_id"]}...")
        self.container = self.rp.get_container(task_instance, run_id)
        self.history: List[Dict[str, str]] = []
        
    def get_initial_prompt(self):
        tools = get_tool()
        user_prompt = get_user_prompt(".", self.task_instance["problem_statement"])
        self.user_prompt = user_prompt
        message = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        return {"message": message, "tools": tools}

    def step(self, messages: List[Dict[str, str]]) -> StepResult:
        """
        Executes a step in the environment based on the agent's message history.
        """
        # Ensure we are working with the latest state
        self.history = messages
        
        # 1. Validate Input
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("The last message in the history must be from the assistant.")

        last_message = messages[-1]
        model_output = last_message.get("content", "")

        # 2. Check for Task Completion
        # if parse_complete_signal(model_output):
        #     # Important: first close the docker
        #     self.close()
        #     task_status = verify_task(model_output, self.task_instance, self.run_id)
        #     task_success = task_status["success"]
        #     return StepResult(
        #         updated_message=self.history,
        #         reward=1.0 if task_success else 0.0,
        #         done=True,
        #         success=task_success,
        #         info={"reason": "agent_submitted"}
        #     )

        # 3. Check for Tool Calls
        tool_call = parse_tool_call(model_output)
        
        if tool_call:
            return self._handle_tool_execution(tool_call)
        
        # 4. Fallback: No command found
        observation = "No valid command or completion signal detected. Please check your format."
        self._append_user_message(f"System Notification:\n{observation}")
        
        return StepResult(
            updated_message=self.history,
            reward=0.0,
            done=False,
            success=False,
            info={"reason": "no_command"}
        )

    def _handle_tool_execution(self, tool_call: Dict[str, Any]) -> StepResult:
        """Internal helper to handle tool logic."""
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        try:
            if name == "ClearContextTool":
                print(f"Detected Swap Tool!!!")
                # Assuming swap_context returns a NEW list of messages (compressed history)
                # We do not append an observation here, we replace the history.
                new_history = swap_context(args, self.history, self.user_prompt)
                return StepResult(
                    updated_message=new_history,
                    reward=0.0,
                    done=False,
                    success=False,
                    modified_context=True,
                    info={"reason": "memory_compressed"}
                )

            elif name == "BashTool":
                command = args.get("command")
                # print(f"{self.run_id} executing command {command}")
                observation = asyncio.run(execute_in_container(self.container, command))
                formatted_obs = f"Tool Execution Result:\n{observation}"
                self._append_user_message(formatted_obs)
            
            elif name == "SubmitTool":
                task_status = asyncio.run(self._verify_task(self.task_instance, self.run_id))
                task_success = task_status["success"]
                return StepResult(
                    updated_message=self.history,
                    reward=1.0 if task_success else 0.0,
                    done=True,
                    success=task_success,
                    info={"reason": "agent_submitted"}
                )
            else:
                self._append_user_message(f"Error: Tool '{name}' is not supported.")

        except Exception as e:
            # Catch execution errors so the agent can try to recover
            error_msg = f"Runtime Error during tool execution: {str(e)}"
            self._append_user_message(error_msg)

        return StepResult(
            updated_message=self.history,
            reward=0.0,
            done=False,
            success=False,
            info={"reason": "tool_executed", "tool": name}
        )
        
    async def _verify_task(
        self,
        task_instance: dict, 
        run_id: str, 
        model_name: str = "Qwen3-Coder"
    ) -> dict:
        """
        Parses model output, extracts a patch, and runs evaluation.
        """
        print(f"Evaluating {run_id}")
        patch = generate_patch(self.container)
        if patch:
            # 3. Construct Payload
            prediction = {
                "instance_id": task_instance.get("instance_id"),
                "model_patch": patch,
                "model_name_or_path": model_name
            }

            payload = {
                "pred": prediction,
                "instance": task_instance,
                "run_id": run_id,
                "f2p_only": False,
                "is_gold": False
            }

            # 4. Execute Evaluation
            self.close()
            # Lynx: run_evaluation contains auto cleanup of the container
            # Lynx: Evaluation could also run infinitely, break long eval and set it unsolved.
            try:
                result = await asyncio.wait_for(
                    run_blocking(run_evaluation, **payload),
                    timeout=60.0
                )
            except Exception as e:
                print(f"Cannot evaluate due to {e}")
                result = dict(resolved=False)
            # result = run_evaluation(**payload)

            # 5. Determine Success
            # Success requires status to be completed AND resolved to be True
            is_success = result.get("resolved", False)
            print("*"*100)
            print(f"{patch}\nSuccess {is_success}")
            print("*"*100)

            return {"success": is_success, "completed": True}
        
        return {"success": False, "completed": True}

    def _append_user_message(self, content: str):
        """Helper to append a message safely."""
        self.history.append({"role": "user", "content": content})

    # def close(self):
    #     """Clean up resources."""
    #     # Lynx: Should we use cleanup_container from swebench/harness/docker_utils.py?
    #     if self.container:
    #         try:
    #             self.container.stop()
    #             self.container.remove()
    #             print("Container stopped and removed.")
    #         except Exception as e:
    #             print(f"Error cleaning up container: {e}")
    #         finally:
    #             time.sleep(2) # Grace period

    #Kerwin: 修改 close 方法，防止重复调用
    def close(self):
        """Clean up resources."""
        if self.container is None:
            return  # 已清理，直接返回
        
        try:
            self.container.reload()  # 检查容器是否存在
            self.container.stop()
            self.container.remove()
            print("Container stopped and removed.")
        except Exception as e:
            # 容器可能已被 run_evaluation 清理
            print(f"Container cleanup: {e}")
        finally:
            self.container = None  # 标记已清理，防止重复调用
            time.sleep(2)
