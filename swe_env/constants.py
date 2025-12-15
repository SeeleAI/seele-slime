SYS_PROMPT = "You are a helpful assistant that can interact with a computer to solve tasks."

def get_user_prompt(working_dir: str, task_context: str):
    prompt = (
f"""
<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a python code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{task_context['problem_statement']}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied.

When executing multi-line Python code, the use of python3 -c is strictly prohibited; the cat << 'EOF' | python3 pattern must be used instead.

If you find ModuleNotFoundError, try install with `pip install -e .` first. But usually I already installed all required dependencies.

When you think you have resolved the problem, generate a .diff file that can be applied to this repository, output the content of the .diff file in the followng format EXACTLY:
<final>
(Content of the .diff file)
<final>
"""
    )
    return prompt

def get_tool():
    MEM_TOOL_DESC = (
        "Summarize the conversation and flush history except the system prompt. "
        "It returns a minimal context: [system, user(summary)]."
        "<WARNING> This tool will flush all the history messages, including the "
        "initial user requests, please summarize with adaquate information carefully. </WARNING>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "BashTool",
                "description": "Interact with a Linux terminal with bash",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        'command': {
                            'type': 'text',
                            'description': 'Any bash command.'
                        }
                    },
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "MemoryTool",
                "description": MEM_TOOL_DESC,
                "parameters": {
                    "type": "object",
                    "required": ["context"],
                    "properties": {
                        'context': {
                            'type': "text",
                            'description': "Summary of the conversation history."
                        }
                    }
                }
            }
        }
    ]
    
    return tools
