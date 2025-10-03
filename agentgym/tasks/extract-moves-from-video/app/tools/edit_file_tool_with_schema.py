# edit_file_tool_with_schema.py
import os
import io
import time
import difflib
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Any


@dataclass
class CachedFile:
    content: str
    timestamp_ns: int


class ToolContext:
    def __init__(self):
        self.read_file_state: Dict[str, CachedFile] = {}
        self.current_tool_use_id: str = ""


class EditFileTool:
    name = "EditTool"
    description = """
    Performs exact string replacements in existing files only.

      Usage:

      - You must use your `Read` tool at least once in the conversation before
        editing. This tool will error if you attempt an edit without reading the file.
        
      - This tool cannot create new files. If you need to create a new file, you must use the Bash tool instead. 
        Attempting to use EditTool for file creation will result in an error.

      - When editing text from Read tool output, ensure you preserve the exact
        indentation (tabs/spaces) as it appears AFTER the line number prefix. The line
        number prefix format is: spaces + line number + tab. Everything after that tab
        is the actual file content to match. Never include any part of the line number
        prefix in the old_string or new_string.

      - ALWAYS prefer editing existing files in the codebase. NEVER write new files
        unless explicitly required.

      - Only use emojis if the user explicitly requests it. Avoid adding emojis to
        files unless asked.

      - The edit will FAIL if `old_string` is not unique in the file. Either provide
        a larger string with more surrounding context to make it unique or use
        `replace_all` to change every instance of `old_string`.

      - Use `replace_all` for replacing and renaming strings across the file. This
        parameter is useful if you want to rename a variable for instance.
    """
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to modify",
            },
            "old_string": {
                "type": "string",
                "description": "The text to replace",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with (must be different from old_string)",
            },
            "replace_all": {
                "type": "boolean",
                "default": False,
                "description": "Replace all occurrences of old_string (default false)",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }

    async def call(
        self,
        input: Dict[str, Any],
        context: ToolContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        file_path = input["file_path"]
        old_string = input["old_string"]
        new_string = input["new_string"]
        replace_all = bool(input.get("replace_all", False))

        cached = context.read_file_state.get(file_path)
        if cached is None:
            raise RuntimeError("File must be read with Read tool before editing")

        st = os.stat(file_path)
        if st.st_mtime_ns != cached.timestamp_ns:
            raise RuntimeError("File has been modified externally since last read")

        if old_string == new_string:
            raise RuntimeError("old_string and new_string cannot be identical")

        yield {
            "type": "progress",
            "toolUseID": context.current_tool_use_id,
            "data": {"status": "Validating edit..."},
        }

        occurrences = self.count_occurrences(cached.content, old_string)
        if occurrences == 0:
            raise RuntimeError("old_string not found in file")

        if not replace_all and occurrences > 1:
            raise RuntimeError(
                f"old_string is not unique ({occurrences} matches). "
                "Either provide more context or use replace_all."
            )

        new_content = (
            cached.content.replace(old_string, new_string)
            if replace_all else
            cached.content.replace(old_string, new_string, 1)
        )

        diff_text = self.generate_diff(cached.content, new_content, file_path)

        yield {
            "type": "progress",
            "toolUseID": context.current_tool_use_id,
            "data": {"status": "Applying edit...", "preview": diff_text},
        }

        self.write_file_with_backup(file_path, new_content)
        st_after = os.stat(file_path)
        context.read_file_state[file_path] = CachedFile(
            content=new_content, timestamp_ns=st_after.st_mtime_ns
        )

        result = {
            "success": True,
            "diff": diff_text,
            "replacements": occurrences if replace_all else 1,
        }
        yield {"type": "result", "data": result}

    @staticmethod
    def count_occurrences(content: str, search: str) -> int:
        return content.count(search)

    @staticmethod
    def generate_diff(old_text: str, new_text: str, file_path: str) -> str:
        diff_lines = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"{file_path} (before)",
            tofile=f"{file_path} (after)",
            lineterm="",
        )
        return "\n".join(diff_lines)

    @staticmethod
    def write_file_with_backup(file_path: str, content: str) -> None:
        ts = time.strftime("%Y%m%d%H%M%S")
        backup_path = f"{file_path}.{ts}.bak"
        with open(file_path, "r", encoding="utf-8") as rf:
            original = rf.read()
        with open(backup_path, "w", encoding="utf-8") as wf:
            wf.write(original)
        with open(file_path, "w", encoding="utf-8") as wf:
            wf.write(content)
