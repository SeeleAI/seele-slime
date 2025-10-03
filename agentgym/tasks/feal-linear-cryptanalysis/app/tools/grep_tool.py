import re
import subprocess
from typing import Dict, Any, Generator, List


class GrepTool:
    name = "GrepTool"
    description = """A powerful search tool built on ripgrep

    Usage:
    - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. 
      The Grep tool has been optimized for correct permissions and access.
    - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
    - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
    - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
    - Use Task tool for open-ended searches requiring multiple rounds
    - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
    - Multiline matching: By default patterns match within single lines only. 
      For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`
    """

    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "The regex pattern"},
            "path": {
                "type": "string",
                "description": "File or directory to search in",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern filter (maps to rg --glob)",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode: content/files_with_matches/count",
            },
            "-B": {"type": "number", "description": "Lines before match"},
            "-A": {"type": "number", "description": "Lines after match"},
            "-C": {"type": "number", "description": "Lines before & after match"},
            "-n": {"type": "boolean", "description": "Show line numbers"},
            "-i": {"type": "boolean", "description": "Case insensitive"},
            "type": {"type": "string", "description": "File type filter"},
            "head_limit": {"type": "number", "description": "Limit N results"},
            "multiline": {
                "type": "boolean",
                "description": "Enable multiline mode (rg -U --multiline-dotall)",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    @classmethod
    def build_command(cls, input: Dict[str, Any]) -> List[str]:
        cmd = ["rg"]

        # Output mode
        output_mode = input.get("output_mode", "files_with_matches")
        if output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif output_mode == "count":
            cmd.append("--count")
        # default "content": no extra flag needed

        # Context options
        for flag in ["-B", "-A", "-C"]:
            if flag in input:
                cmd.extend([flag, str(input[flag])])

        if input.get("-n"):
            cmd.append("-n")
        if input.get("-i"):
            cmd.append("-i")
        if input.get("multiline"):
            cmd.extend(["-U", "--multiline-dotall"])
        if "type" in input:
            cmd.extend(["--type", input["type"]])
        if "glob" in input:
            cmd.extend(["--glob", input["glob"]])

        # Regex
        cmd.extend(["-e", input["pattern"]])
        cmd.append(input.get("path", "."))

        return cmd

    @classmethod
    async def call(cls, input: Dict[str, Any], context) -> Generator[Dict[str, Any], None, None]:
        # Validate regex
        try:
            re.compile(input["pattern"])
        except re.error as e:
            raise ValueError(f"Invalid regex: {e}")

        yield {
            "type": "progress",
            "toolUseID": getattr(context, "currentToolUseId", None),
            "data": {"status": "Searching files..."},
        }

        cmd = cls.build_command(input)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )

        # Limit head if needed
        stdout_lines = proc.stdout.splitlines()
        if "head_limit" in input:
            stdout_lines = stdout_lines[: int(input["head_limit"])]

        yield {
            "type": "result",
            "data": {
                "exitCode": proc.returncode,
                "stdout": "\n".join(stdout_lines),
                "stderr": proc.stderr,
                "lineCount": len(stdout_lines),
            },
        }
