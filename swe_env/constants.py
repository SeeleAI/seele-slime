# SYS_PROMPT = (
#     "You are a helpful assistant that can interact with a computer to solve tasks. "
#     "Meanwhile, you must efficiently manage your limited token budget. In each round of conversation, "
#     "the user will inform you of the remaining token budget. Once you realize that your remaining token "
#     "budget is low, or that upcoming responses or tool call returns may exceed the token budget, "
#     "you should consider call ClearContextTool to clear the context. When you decide to invoke the ClearContextTool, think carefully. "
#     "The epxerience and instruction should retain sufficient detailed information so that you can continue with the task "
#     "even after the context is cleared."
# )
# SYS_PROMPT = (
#     "You are a helpful assistant that can interact with a computer to solve tasks. "
#     "The user will inform you of the remaining token budget. You must efficiently manage your limited token budget. \n\n"
#     "CRITICAL MEMORY MANAGEMENT RULE:\n"
#     "You are strictly prohibited from calling the ClearContextTool UNLESS your "
#     "remaining token budget is CRITICALLY LOW (e.g., less than 10% remaining). "
#     "Do NOT use this tool to 'checkpoint' or 'save' your progress if you still have plenty of "
#     "tokens available. Using this tool wipes your short-term memory, which is dangerous "
#     "and should be a last resort to prevent crashing."
# )

FEW_SHOTS = """
## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""

SYS_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks. "
    "The user will inform you of the remaining token budget. You must efficiently manage your limited token budget. \n\n"
    "CRITICAL MANAGEMENT RULE:\n"
    "When you notice your token budget is running low, you may be unable to proceed. In this case, you must invoke the "
    "HandoffTool to transfer your current tasks to another Agent. The primary principle of a handoff is to provide "
    "the successor with sufficiently detailed requirements, context, and goals. The tool will guide you through completing "
    "a handoff form, but remember: providing comprehensive information is essential for the successor to successfully "
    "complete the work you left unfinished."
    "You are strictly prohibited from calling the HandoffTool UNLESS your "
    "remaining token budget is CRITICALLY LOW (e.g., less than 10% remaining). "
    "Do NOT use this tool to 'checkpoint' or 'save' your progress if you still have plenty of "
    "tokens available."
    f"\n{FEW_SHOTS}"
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

When you think you have completed the task, call the SubmitTool.
"""
    )
    return prompt

def get_tool():
    # MEM_TOOL_DESC = (
    #     "EMERGENCY ONLY. A tool for freeing up memory when you are about to run out of tokens. "
    #     "Calling this DELETES all conversation history. "
    #     "Only call this if you calculate that the next step will exceed your remaining token limit. "
    #     "And when you call this tool, first think step by step what information should be passed to the next "
    #     "session."
    # )
    MEM_TOOL_DESC = (
        "EMERGENCY ONLY. A tool to handoff the current task to another Agent. "
        "Crytical Rules:\n"
        "1. Call this tool ONLY when you think you don't have enough token budget to complete the task.\n"
        "2. Your successor has **ZERO access** to the previous conversation history, code bases, or file contents you have already read. They only see the [HANDOFF REPORT] you generate now.\n"
        "3. If you force them to re-read a file you already read, **you fail**.\n"
        "4. If you force them to re-test a bug you already analyzed, **you fail**.\n"
        "5. If you write vague summaries like 'I analyzed the code,' **you fail**.\n"
        "Generate a strict **[HANDOFF REPORT]** containing specific, actionable data. You must transfer **Knowledge**, not just a Summary."
    )
    SUBMIT_TOOL_DESC = (
"""
## Submission Rule

When you've completed the task, you can call this tool submit. Before you call this tool, follow these steps to check your changes.

Step 1: Create the patch file
Run `git diff > patch.txt` to inspect all files that have changed. Do NOT commit your changes. 

<IMPORTANT>
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains changes that relate to the problem.

Step 3: Submit
Submit the **relavent** files you modified.
Example:
[
    "path_to_file1",
    "path_to_file2",
]

<CRITICAL>
You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
"""
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
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "SubmitTool",
    #             "description": "Call this tool when you think you have resolved the problem.",
    #             "parameters": {}
    #         }
    #     }
    # ]
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
    #             "name": "ClearContextTool",
    #             "description": MEM_TOOL_DESC,
    #             "parameters": {
    #                 "type": "object",
    #                 "required": ["think", "next_session_context"],
    #                 "properties": {
    #                     'next_session_context': {
    #                         'type': "text",
    #                         'description': (
    #                             "A comprehensive, standalone summary of the state of the world. "
    #                             "This string will be the ONLY memory available to you after the reset. "
    #                             "Summary with this format:\n# What I Did\nWhat I Should Do Next, "
    #                             "put all the important information that you think is necessary."
    #                         )
    #                     },
    #                     "think": {
    #                         "type": "text",
    #                         "description": (
    #                             "Write your reasoning trace here, think step by step, and list in detail with bullet points what specific information you should pass on to the next session."
    #                         )
    #                     }
    #                 }
    #             }
    #         }
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "SubmitTool",
    #             "description": "Call this tool when you think you have resolved the problem.",
    #             "parameters": {}
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
                "name": "SubmitTool",
                "description": SUBMIT_TOOL_DESC,
                "parameters": {
                    "type": "object",
                    "required": ["files"],
                    "properties": {
                        'files': {
                            'type': 'list[str]',
                            'description': 'List of absolute file path'
                        }
                    },
                }
            }
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "ReadFileTool",
        #         "description": "Reads content from a file. Can read the entire file or specific lines to handle large files. Always use line numbers for files likely to be large (e.g., logs, big source files).",
        #         "parameters": {
        #             "type": "object",
        #             "required": ["file_path"],
        #             "properties": {
        #                 "file_path": {
        #                 "type": "text",
        #                 "description": "The path to the file to read."
        #                 },
        #                 "start_line": {
        #                 "type": "integer",
        #                 "description": "The line number to start reading from (1-indexed). Optional."
        #                 },
        #                 "end_line": {
        #                 "type": "integer",
        #                 "description": "The last line number to read. Optional. If omitted, reads to the end."
        #                 }
        #             }
        #         }
        #     }
        # },
        {
            "type": "function",
            "function": {
                "name": "HandoffTool",
                "description": MEM_TOOL_DESC,
                "parameters": {
                    "type": "object",
                    "required": ["mission_anchor", "acquired_environmental_knowledge", "pruned_paths", "immediate_next_step", "others"],
                    "properties": {
                        'mission_anchor': {
                            'type': "text",
                            'description': (
                                "CRYTICAL: Write the mission anchor as detail as possible!!!"
                                "- **Original Goal:** (Verbatim, what was the user's initial request?)\n"
                                "- **Current Status:** (e.g., 'Phase 1: Exploration Complete. Phase 2: Implementation In-progress.')\n"
                                "- **Completion Plan:** (What remains to be done? Be concise.)\n"
                            )
                        },
                        'acquired_environmental_knowledge':{
                            'type': "text",
                            'description': (
                                "*List all high-cost information you have retrieved from the environment (tools/APIs/files). Save your successor the token cost of retrieving them again.*\n"
                                "- **File/Data Context:** (e.g., file/respository structure)\n"
                                "- **Key Variables:** (e.g., 'AWS Instance ID: i-12345', 'User ID: 888')\n"
                                "- **Experiences:** (e.g., 'Where the bug is, what you have done to it')\n"
                            )
                        },
                        'pruned_paths':{
                            'type': "text",
                            'description': (
                                "*List what you have TRIED but FAILED. Prevent your successor from entering a retry loop.*\n"
                                "- **Failed Attempts:** (e.g., 'Tried modifying `config.json` but it caused a syntax error.')\n"
                                "- **Invalid Hypotheses:** (e.g., 'The bug is NOT in the database connection string; verified via logs.')"
                            )
                        },
                        'immediate_next_step':{
                            'type': "text",
                            'description': (
                                "**Atomic Action:** (The exact single step the successor must take immediately upon waking up. e.g., "
                                "Write the unit test for `fix_login_bug` in `tests/test_auth.py`.)"
                            )
                        },
                        'others':{
                            'type': "text",
                            'description': (
                                "*List all other information you have that may be useful to your successor."
                            )
                        }
                    }
                }
            }
        }
    ]
    
    return tools
