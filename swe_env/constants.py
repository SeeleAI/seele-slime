# SYS_PROMPT = (
#     "You are a helpful assistant that can interact with a computer to solve tasks. "
#     "Meanwhile, you must efficiently manage your limited token budget. In each round of conversation, "
#     "the user will inform you of the remaining token budget. Once you realize that your remaining token "
#     "budget is low, or that upcoming responses or tool call returns may exceed the token budget, "
#     "you should consider call ClearContextTool to clear the context. When you decide to invoke the ClearContextTool, think carefully. "
#     "The epxerience and instruction should retain sufficient detailed information so that you can continue with the task "
#     "even after the context is cleared."
# )
SYS_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks. "
    "The user will inform you of the remaining token budget. You must efficiently manage your limited token budget. \n\n"
    "CRITICAL MEMORY MANAGEMENT RULE:\n"
    "You are strictly prohibited from calling the ClearContextTool UNLESS your "
    "remaining token budget is CRITICALLY LOW (e.g., less than 10% remaining). "
    "Do NOT use this tool to 'checkpoint' or 'save' your progress if you still have plenty of "
    "tokens available. Using this tool wipes your short-term memory, which is dangerous "
    "and should be a last resort to prevent crashing."
)

# SYS_PROMPT = (
#     "You are a helpful assistant that can interact with a computer to solve tasks. "
# )

def get_user_prompt(working_dir: str, task_context: str):
    prompt = (
f"""
<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a python code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{task_context}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied.

When executing multi-line Python code, the use of python3 -c is strictly prohibited; the cat << 'EOF' | python3 pattern must be used instead.

If you find ModuleNotFoundError, try install with `pip install -e .` first. But usually I already installed all required dependencies.

When you think you have resolved the problem, call the SubmitTool.
"""
    )
    return prompt

def get_tool():
    MEM_TOOL_DESC = (
        "EMERGENCY ONLY. A tool for freeing up memory when you are about to run out of tokens. "
        "Calling this DELETES all conversation history. "
        "Only call this if you calculate that the next step will exceed your remaining token limit. "
        "And when you call this tool, first think step by step what information should be passed to the next "
        "session."
    )
    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "BashTool",
    #             "description": "Interact with a Linux terminal with bash",
    #             "parameters": {
    #                 "type": "object",
    #                 "required": ["command"],
    #                 "properties": {
    #                     'command': {
    #                         'type': 'text',
    #                         'description': 'Any bash command.'
    #                     }
    #                 },
    #             }
    #         }
    #     }
    # ]
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
                "name": "ClearContextTool",
                "description": MEM_TOOL_DESC,
                "parameters": {
                    "type": "object",
                    "required": ["think", "next_session_context"],
                    "properties": {
                        'next_session_context': {
                            'type': "text",
                            'description': (
                                "A comprehensive, standalone summary of the state of the world. "
                                "This string will be the ONLY memory available to you after the reset. "
                                "Summary with this format:\n# What I Did\nWhat I Should Do Next, "
                                "put all the important information that you think is necessary."
                            )
                        },
                        "think": {
                            "type": "text",
                            "description": (
                                "Write your reasoning trace here, think step by step, and list in detail with bullet points what specific information you should pass on to the next session."
                            )
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "SubmitTool",
                "description": "Call this tool when you think you have resolved the problem.",
                "parameters": {}
            }
        }
    ]
    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "BashTool",
    #             "description": "Interact with a Linux terminal with bash",
    #             "parameters": {
    #                 "type": "object",
    #                 "required": ["command"],
    #                 "properties": {
    #                     'command': {
    #                         'type': 'text',
    #                         'description': 'Any bash command.'
    #                     }
    #                 },
    #             }
    #         }
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "SwapTool",
    #             "description": MEM_TOOL_DESC,
    #             "parameters": {
    #                 "type": "object",
    #                 "required": ["user_request", "what_i_did", "what_i_should_do_next"],
    #                 "properties": {
    #                     'user_request': {
    #                         'type': "text",
    #                         'description': "Summary of the user request, should include as many details as possible."
    #                     },
    #                     'what_i_did': {
    #                         'type': "text",
    #                         'description': "Summary of what you have done for completing the task."
    #                     },
    #                     'what_i_should_do_next':{
    #                         'type': "text",
    #                         'description': "Give brief instruction of what you should do after the context is replaced."
    #                     }
    #                 }
    #             }
    #         }
    #     }
    # ]
    
    return tools
