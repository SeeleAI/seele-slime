from swesmith.profiles import registry
from swesmith.harness.eval import run_evaluation
from swe_env.utils import parse_tool_call, execute_in_container, parse_complete_signal
from swe_env.constants import SYS_PROMPT, get_tool, get_user_prompt
from dataclasses import dataclass, field
from typing import Dict, Any, List
import time
import re
import asyncio

def swap_context(summarize: dict, messages: list[dict], user_request: str) -> list[dict]:
    assert messages[0]['role'] == "system", f"Message[0] role {messages[0]['role']} is incorrect, should be system"
    system_prompt = messages[0]
    next_session = summarize["next_session_context"]
    summary_content = (
f"""
# User Request
{user_request}

# Previous Context
{next_session}
"""
    )
    new_message = [
        system_prompt,
        {"role": "user", "content": f"You called ClearContextTool, the new context is\n{summary_content}"}
    ]
    
    return new_message

def parse_llm_diff_output(text: str):
    """
    Parses LLM output to extract unified diff content and identify the type of output.
    
    Args:
        text (str): The raw string output from the LLM.
        
    Returns:
        dict: {
            "type": "diff_content" | "command_only" | "unknown",
            "content": str (The clean .diff content or the command),
            "is_valid_patch": bool (True if ready to be saved as .diff)
        }
    """
    # 1. Clean Markdown code blocks (e.g., ```diff ... ```)
    # We strip the backticks but keep the content inside
    code_block_pattern = r"```(?:diff|bash|sh)?\n(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    
    # If code blocks exist, prioritize searching inside them
    # Otherwise, search the whole text (the LLM might have forgotten markdown)
    search_text = "\n".join(code_blocks) if code_blocks else text

    # 2. Define patterns
    # A standard unified diff header: 
    # --- a/path/to/file
    # +++ b/path/to/file
    diff_header_pattern = re.compile(r"(^--- .*?\n\+\+\+ .*?)(?=\n|$)", re.MULTILINE)
    
    # A standard diff command
    diff_cmd_pattern = re.compile(r"^\s*(diff\s+-u\s+.*|git\s+diff\s+.*)", re.MULTILINE)

    # 3. extraction Logic
    diff_match = diff_header_pattern.search(search_text)
    
    if diff_match:
        # Found the start of a diff. 
        # Extract from the '---' line to the end of the text/block
        # Note: This includes trailing text (LLM comments). 
        # Robust patch tools usually ignore trailing garbage, but we can try to trim.
        start_index = diff_match.start()
        raw_diff = search_text[start_index:]
        
        # Optional: Trim trailing lines that don't look like diff content
        # (Lines that don't start with ' ', '+', '-', '@', '\', or 'index')
        cleaned_lines = []
        for line in raw_diff.splitlines():
            if re.match(r"^([+\-@\\ ]|index|diff|---|\+\+\+)", line):
                cleaned_lines.append(line)
            else:
                # Once we hit a line that isn't diff syntax, we stop (assuming LLM commentary follows)
                # However, be careful not to stop on blank lines inside a diff
                if line.strip() == "":
                    cleaned_lines.append(line)
                    continue
                break
                
        return {
            "type": "diff_content",
            "content": "\n".join(cleaned_lines),
            "is_valid_patch": True
        }

    # 4. If no diff content, check if it's just a command
    cmd_match = diff_cmd_pattern.search(search_text)
    if cmd_match:
        return {
            "type": "command_only",
            "content": cmd_match.group(1),
            "is_valid_patch": False
        }

    return {
        "type": "unknown",
        "content": "",
        "is_valid_patch": False
    }

def parse_final_patch(model_output: str, trim_spaces: bool = True):
    pattern = r'<final>(.*?)</final>'
    matches = re.findall(pattern, model_output, re.DOTALL)  
    
    if trim_spaces:
        matches = [match.strip() for match in matches]
    
    return matches

# def verify_task(
#     model_output: str, 
#     task_instance: dict, 
#     run_id: str, 
#     model_name: str = "Qwen3-Coder"
# ) -> dict:
#     """
#     Parses model output, extracts a patch, and runs evaluation.
#     """

#     # 3. Construct Payload
#     prediction = {
#         "instance_id": task_instance.get("instance_id"),
#         "model_patch": model_output,
#         "model_name_or_path": model_name
#     }

#     payload = {
#         "pred": prediction,
#         "instance": task_instance,
#         "run_id": run_id,
#         "f2p_only": False,
#         "is_gold": False
#     }

#     # 4. Execute Evaluation
#     # Lynx: run_evaluation contains auto cleanup of the container
#     result = run_evaluation(**payload)

#     # 5. Determine Success
#     # Success requires status to be completed AND resolved to be True
#     is_success = result.get("resolved", False)
#     print("*"*100)
#     print(f"Evaluating {run_id}\n {model_output}\nSuccess {is_success}")
#     print("*"*100)

#     return {"success": is_success, "completed": True}

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
                print(f"{self.run_id} executing command {command}")
                observation = asyncio.run(execute_in_container(self.container, command))
                formatted_obs = f"Tool Execution Result:\n{observation}"
                self._append_user_message(formatted_obs)
            
            elif name == "SubmitTool":
                patch = args.get("patch_path")
                print(f"model submitted {patch}")
                task_status = self._verify_task(patch, self.task_instance, self.run_id)
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
        
    def _verify_task(
        self,
        model_output: str, 
        task_instance: dict, 
        run_id: str, 
        model_name: str = "Qwen3-Coder"
    ) -> dict:
        """
        Parses model output, extracts a patch, and runs evaluation.
        """
        print(f"Evaluating {run_id}")
        # 1. read the file from docker
        file_path = model_output
        command = f"cat {file_path}"
        content = asyncio.run(execute_in_container(self.container, command))
        
        # 3. Construct Payload
        prediction = {
            "instance_id": task_instance.get("instance_id"),
            "model_patch": content,
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
        result = run_evaluation(**payload)

        # 5. Determine Success
        # Success requires status to be completed AND resolved to be True
        is_success = result.get("resolved", False)
        print("*"*100)
        print(f"{content}\nSuccess {is_success}")
        print("*"*100)

        return {"success": is_success, "completed": True}

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
