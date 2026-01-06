import re
import base64
import signal
import asyncio
from concurrent.futures import ThreadPoolExecutor

from concurrent.futures import ThreadPoolExecutor

# Use a dedicated thread pool to avoid blocking your main app
DOCKER_EXECUTOR = ThreadPoolExecutor(max_workers=20)

TIMEOUT = 40

def timeout_handler(signum, frame):
    raise TimeoutError("Command execution timed out")

def parse_complete_signal(text: str):
    pattern = r'<final>(.*?)</final>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def parse_tool_call(text):
    """
    Parses the custom tool call. Returns None if:
    - The function tag is missing or malformed.
    - Any text inside the function block is not a valid parameter tag.
    - Parameter tags are malformed (e.g. missing closing tag).
    """
    # 1. Extract the full function block: <function=NAME> ...CONTENT... </function>
    # We use re.DOTALL to handle newlines inside the block
    func_match = re.search(r"<function=(.*?)>(.*?)</function>", text, re.DOTALL)
    
    if not func_match:
        return None

    tool_name = func_match.group(1).strip()
    content = func_match.group(2)

    # 2. Extract all valid parameters
    # This finds all strings that strictly match the <parameter> format
    param_pattern = r"<parameter=(.*?)>(.*?)</parameter>"
    params_matches = re.findall(param_pattern, content, re.DOTALL)
    
    # 3. Validation: specific check for malformed content
    # We remove all valid parameter strings from the original content.
    # If the agent wrote something like <parameter=p1>val (missing closing),
    # it won't be removed, leaving residual text.
    cleaned_content = re.sub(param_pattern, "", content, flags=re.DOTALL)
    
    # If anything significant remains (ignoring whitespace), the parse failed.
    if cleaned_content.strip():
        return None

    # 4. Construct the dictionary
    arguments = {k.strip(): v.strip() for k, v in params_matches}

    return {
        "name": tool_name,
        "arguments": arguments
    }

# def execute_in_container(container, command):
#     """
#     Executes command in the docker container and handles output decoding.
#     """
#     # print(f"\n[Tool Execution] Running: {command}")
#     # print([command])
#     try:
#         # exec_run usually returns (exit_code, output_bytes)
#         # exit_code, output = container.exec_run(cmd=f"bash -c '{command}'")
#         b64_cmd = base64.b64encode(command.encode('utf-8')).decode('utf-8')
    
#         safe_cmd = f"bash -c 'echo {b64_cmd} | base64 -d | bash'"
#         timeout_seconds = TIMEOUT
#         signal.signal(signal.SIGALRM, timeout_handler)
#         signal.alarm(timeout_seconds)
#         exit_code, output = container.exec_run(cmd=safe_cmd)
#         signal.alarm(0)
        
#         # Decode output (handle potential encoding errors)
#         output_str = output.decode("utf-8", errors="replace")
        
#         if exit_code != 0:
#             return f"Command failed with exit code {exit_code}.\nOutput:\n{output_str}"
        
#         # Truncate very long outputs to save context window
#         if len(output_str) > 2000:
#             output_str = output_str[:1000] + "\n...[Output Truncated]...\n" + output_str[-1000:]
            
#         return output_str if output_str.strip() else "Command executed successfully with no output."
        
#     except Exception as e:
#         return f"System Error during container execution: {str(e)}"


# async def execute_in_container(container, command):
#     """
#     Executes command in the docker container asynchronously and handles output decoding.
#     """
#     try:
#         # Prepare the safe command (same logic as before)
#         b64_cmd = base64.b64encode(command.encode('utf-8')).decode('utf-8')
#         # safe_cmd = f"bash -c 'echo {b64_cmd} | base64 -d | bash'"
#         safe_cmd = f"bash -c 'timeout -k 5 {TIMEOUT}s echo {b64_cmd} | base64 -d | bash'"

#         # Get the current asyncio loop
#         loop = asyncio.get_running_loop()

#         # Define a synchronous wrapper for the blocking docker call
#         # We use a lambda or partial because run_in_executor doesn't support kwargs directly
#         def _exec_blocking():
#             return container.exec_run(cmd=safe_cmd)

#         # Run the blocking call in a separate thread to avoid blocking the event loop
#         # asyncio.wait_for handles the timeout logic natively
#         exit_code, output = await asyncio.wait_for(
#             loop.run_in_executor(None, _exec_blocking), 
#             timeout=TIMEOUT
#         )
        
#         # Decode output (handle potential encoding errors)
#         output_str = output.decode("utf-8", errors="replace")
        
#         if exit_code != 0:
#             return f"Command failed with exit code {exit_code}.\nOutput:\n{output_str}"
        
#         # Truncate very long outputs to save context window
#         if len(output_str) > 2000:
#             output_str = output_str[:1000] + "\n...[Output Truncated]...\n" + output_str[-1000:]
            
#         return output_str if output_str.strip() else "Command executed successfully with no output."

#     except asyncio.TimeoutError:
#         return f"System Error: Execution timed out after {TIMEOUT} seconds."
        
#     except Exception as e:
#         return f"System Error during container execution: {str(e)}"


async def execute_in_container(container, command, timeout=60):
    """
    Executes a command. If it times out, RESTARTS the container to 
    force-kill the process and unblock the thread.
    """
    loop = asyncio.get_running_loop()

    # 1. Prepare the command
    b64_cmd = base64.b64encode(command.encode('utf-8')).decode('utf-8')
    # We remove the bash-level timeout because we will handle it at the container level
    safe_cmd = f"bash -c 'echo {b64_cmd} | base64 -d | bash'"

    # 2. Define the blocking task
    def _exec_blocking():
        return container.exec_run(cmd=safe_cmd)

    try:
        # 3. Wait for result with a hard limit
        exit_code, output = await asyncio.wait_for(
            loop.run_in_executor(DOCKER_EXECUTOR, _exec_blocking),
            timeout=timeout
        )
        
        output_str = output.decode("utf-8", errors="replace")
        
        # print(output_str)
        
        # Truncate very long outputs to save context window
        if len(output_str) > 2500:
            output_str = output_str[:1250] + "\n...[Output Truncated]...\n" + output_str[-1250:]
        
        if exit_code != 0:
            return f"Command failed (Exit {exit_code}):\n{output_str}"
        return output_str if output_str.strip() else "Command executed successfully with no output."

    except asyncio.TimeoutError:
        # 4. THE FIX: Timeout occurred. 
        # The background thread is STUCK waiting on the socket.
        # We must kill the container to break that connection.
        print("%"*100)
        print(f"Container {container.short_id} timed out. Restarting to kill process...")
        print(f"The command was {command}")
        print("%"*100)
        # Run the restart in a separate thread (since restart() is also blocking)
        # timeout=5 ensures we don't hang if restart fails
        try:
            await loop.run_in_executor(None, lambda: container.restart(timeout=5))
        except Exception as e:
            return f"Critical System Error: Failed to restart container: {e}"

        return f"Execution timed out after {timeout}s."

    except Exception as e:
        return f"System Error: {str(e)}"
