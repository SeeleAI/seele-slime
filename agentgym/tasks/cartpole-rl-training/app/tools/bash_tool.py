# bash_tool.py
import asyncio
import os
import platform
import shlex
import textwrap
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional, Tuple


# ----------------- Minimal Context (customize to your orchestrator) -----------------
@dataclass
class ToolContext:
    cwd: str = os.getcwd()
    current_tool_use_id: str = ""
    # 可被外部设置为 asyncio.Event()，用于取消
    abort_event: Optional[asyncio.Event] = None

    def yield_progress(self, payload: Dict[str, Any]) -> None:
        """
        你的编排器可以覆写此方法以接收流式输出。
        这里默认 no-op，因为 call() 会直接 yield 进度字典。
        """
        pass


# ----------------- Bash Tool -----------------
class BashTool:
    """
    name: BashTool
    """

    # ==== MANDATED DOCS ====
    description = textwrap.dedent("""
      Executes a given bash command in a persistent shell session with optional
      timeout, ensuring proper handling and security measures.

      Before executing the command, please follow these steps:

      1. Directory Verification:
         - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is the correct location
         - For example, before running "mkdir foo/bar", first use LS to check that "foo" exists and is the intended parent directory

      2. Command Execution:
         - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
         - Examples of proper quoting:
           - cd "/Users/name/My Documents" (correct)
           - cd /Users/name/My Documents (incorrect - will fail)
           - python "/path/with spaces/script.py" (correct)
           - python /path/with spaces/script.py (incorrect - will fail)
         - After ensuring proper quoting, execute the command.
         - Capture the output of the command.

      Usage notes:
        - The command argument is required.
        - You can specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). If not specified, commands will timeout after 120000ms (2 minutes).
        - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.
        - If the output exceeds 30000 characters, output will be truncated before being returned to you.
        - VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. Instead use Grep, Glob, or Task to search. You MUST avoid read tools like `cat`, `head`, `tail`, and `ls`, and use Read and LS to read files.
         - If you _still_ need to run `grep`, STOP. ALWAYS USE ripgrep at `rg` first, which all ${PRODUCT_NAME} users have pre-installed.
        - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings).
        - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.
          <good-example>
          pytest /foo/bar/tests
          </good-example>
          <bad-example>
          cd /foo/bar && pytest tests
          </bad-example>

      # Committing changes with git

      When the user asks you to create a new git commit, follow these steps carefully:

      1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel, each using the Bash tool:
        - Run a git status command to see all untracked files.
        - Run a git diff command to see both staged and unstaged changes that will be committed.
        - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
      2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
        - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. "add" means a wholly new feature, "update" means an enhancement to an existing feature, "fix" means a bug fix, etc.).
        - Check for any sensitive information that shouldn't be committed
        - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what"
        - Ensure it accurately reflects the changes and their purpose
      3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:
         - Add relevant untracked files to the staging area.
         - Create the commit with a message ending with:
         Generated with [Claude Code](https://claude.ai/code)

         Co-Authored-By: Claude <noreply@anthropic.com>
         - Run git status to make sure the commit succeeded.
      4. If the commit fails due to pre-commit hook changes, retry the commit ONCE
      to include these automated changes. If it fails again, it usually means a
      pre-commit hook is preventing the commit. If the commit succeeds but you
      notice that files were modified by the pre-commit hook, you MUST amend your
      commit to include them.

      Important notes:
      - NEVER update the git config
      - NEVER run additional commands to read or explore code, besides git bash commands
      - NEVER use the TodoWrite or Task tools
      - DO NOT push to the remote repository unless the user explicitly asks you to do so
      - IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
      - If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit
      - In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:

      <example>
      git commit -m "$(cat <<'EOF'
         Commit message here.

         Generated with [Claude Code](https://claude.ai/code)

         Co-Authored-By: Claude <noreply@anthropic.com>
         EOF
         )"
      </example>

      # Creating pull requests
      Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.

      IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

      1. ... (omitted for brevity in runtime; see spec above)
    """)

    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"type": "number", "description": "Optional timeout in milliseconds (max 600000)"},
            "description": {
                "type": "string",
                "description": (
                    "Clear, concise description of what this command does in 5-10 words.\n"
                    "Examples:\n"
                    "Input: ls\nOutput: Lists files in current directory\n\n"
                    "Input: git status\nOutput: Shows working tree status\n\n"
                    "Input: npm install\nOutput: Installs package dependencies\n\n"
                    "Input: mkdir foo\nOutput: Creates directory 'foo'"
                ),
            },
            "sandbox": {"type": "boolean", "description": "Run with macOS sandbox-exec if available"},
            "shellExecutable": {"type": "string", "description": "Custom shell executable path"},
        },
        "required": ["command"],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }

    FORBIDDEN = {"find", "grep", "cat", "head", "tail", "ls"}
    DANGEROUS_TOKENS = {"rm", "dd", "mkfs", "fdisk", "kill"}
    MAX_TIMEOUT_MS = 600_000
    DEFAULT_TIMEOUT_MS = 120_000
    OUTPUT_TRUNC_LIMIT = 30_000  # chars

    # ---------------- Permission Checking ----------------
    async def check_permissions(self, input: Dict[str, Any], context: ToolContext, perm_context: Dict[str, Any]) -> Dict[str, Any]:
        command = str(input.get("command", "")).strip()
        sandbox = bool(input.get("sandbox", False))

        base = self._first_token(command)
        # 禁用基础指令，除非 bypass
        if base in self.FORBIDDEN and not ("bypass" in (perm_context.get("mode") or "")):
            return {"behavior": "deny", "message": f"Use dedicated tools instead of `{base}`"}

        # 危险指令需询问
        if any(tok in command for tok in self.DANGEROUS_TOKENS):
            return {
                "behavior": "ask",
                "message": "This command could be dangerous",
                "ruleSuggestions": [f"BashTool({base}/*)"] if base else [],
            }

        # 沙箱模式：给出微弱白名单提示（不强制）
        if sandbox is True and base not in {"echo", "pwd", "env", "date", "which", "true"}:
            # allow anyway, but caller could warn user
            return {"behavior": "allow", "note": "Sandbox requested; non-trivial commands may be limited on macOS."}

        # 默认允许
        return {"behavior": "allow"}

    # ---------------- Main Entry ----------------
    async def call(self, input: Dict[str, Any], context: ToolContext) -> AsyncGenerator[Dict[str, Any], None]:
        command = str(input["command"]).strip()
        timeout_ms = int(input.get("timeout", self.DEFAULT_TIMEOUT_MS))
        timeout_ms = max(1, min(timeout_ms, self.MAX_TIMEOUT_MS))
        sandbox = bool(input.get("sandbox", False))
        shell_exe = str(input.get("shellExecutable") or "/bin/bash")

        # 准备阶段
        yield {
            "type": "progress",
            "toolUseID": context.current_tool_use_id,
            "data": {
                "status": "Preparing command execution...",
                "command": command[:100],
                "sandbox": sandbox,
                "timeout_ms": timeout_ms,
            },
        }

        # macOS 沙箱封装（仅 darwin & sandbox=True 有效）
        final_cmd = command
        if sandbox and platform.system().lower() == "darwin":
            profile = self._generate_sandbox_profile()
            # 用 -c 传脚本前先 echo profile > 临时文件也可，这里 inline 简化
            # 注意转义单引号
            profile_escaped = profile.replace("'", "'\"'\"'")
            final_cmd = f"sandbox-exec -p '{profile_escaped}' {command}"

        # 执行
        result = await self._execute_streaming(
            final_cmd,
            cwd=context.cwd,
            env={**os.environ, "CLAUDE_CODE": "true"},
            timeout_ms=timeout_ms,
            shell_exe=shell_exe,
            context=context,
        )

        # 结果
        yield {"type": "result", "data": result}

    # ---------------- Internals ----------------
    async def _execute_streaming(
        self,
        command: str,
        cwd: str,
        env: Dict[str, str],
        timeout_ms: int,
        shell_exe: str,
        context: ToolContext,
    ) -> Dict[str, Any]:
        """
        以 bash -lc 运行命令；流式读取 stdout/stderr；支持 abort/timeout；输出截断。
        """
        # 使用 bash -lc 确保别名/登录环境（视需求可改 -c）
        proc = await asyncio.create_subprocess_exec(
            shell_exe, "-lc", command,
            cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_buf = []
        stderr_buf = []
        total_len = 0

        async def _reader(stream, tag):
            nonlocal total_len
            try:
                while True:
                    chunk = await stream.readline()
                    if not chunk:
                        break
                    text = chunk.decode(errors="replace")
                    # 流式进度（不保证每行完整）
                    if total_len < self.OUTPUT_TRUNC_LIMIT:
                        yield_payload = {
                            "type": "progress",
                            "toolUseID": context.current_tool_use_id,
                            "data": {tag: text, "partial": True},
                        }
                        # 既通过 context hook，也通过 call() 的 yield 返回
                        context.yield_progress(yield_payload)
                        # 这里不能直接 yield（在子协程中），交由外层通过 hook 处理
                    # 累计并截断
                    append_target = stdout_buf if tag == "stdout" else stderr_buf
                    append_target.append(text)
                    total_len += len(text)
                    if total_len >= self.OUTPUT_TRUNC_LIMIT:
                        break
            except Exception:
                pass  # 安全兜底，读流失败不影响进程结束

        # 并发读取
        reader_stdout = asyncio.create_task(_reader(proc.stdout, "stdout"))
        reader_stderr = asyncio.create_task(_reader(proc.stderr, "stderr"))

        # 处理 abort/timeout
        async def _wait_with_abort(p: asyncio.subprocess.Process) -> Tuple[Optional[int], Optional[str]]:
            wait_task = asyncio.create_task(p.wait())
            try:
                if context.abort_event:
                    abort_task = asyncio.create_task(context.abort_event.wait())
                else:
                    abort_task = None

                done, pending = await asyncio.wait(
                    {wait_task, abort_task} if abort_task else {wait_task},
                    timeout=timeout_ms / 1000.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if wait_task in done:
                    return p.returncode, None
                # 超时或 abort
                if abort_task and abort_task in done:
                    try:
                        p.terminate()
                    except ProcessLookupError:
                        pass
                    return None, "aborted"
                # 超时
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass
                return None, "timeout"
            finally:
                # 清理遗留任务
                for t in (wait_task, abort_task):
                    if t and not t.done():
                        t.cancel()

        code, reason = await _wait_with_abort(proc)

        # 等待读取完成
        with contextlib_suppress():
            await asyncio.wait_for(reader_stdout, timeout=1.0)
        with contextlib_suppress():
            await asyncio.wait_for(reader_stderr, timeout=1.0)

        # 如果仍在运行且不是正常结束，尝试 kill
        if code is None:
            with contextlib_suppress():
                proc.kill()
            with contextlib_suppress():
                await asyncio.wait_for(proc.wait(), timeout=1.0)

        # 拼接并截断
        stdout = "".join(stdout_buf)[: self.OUTPUT_TRUNC_LIMIT]
        stderr = "".join(stderr_buf)[: self.OUTPUT_TRUNC_LIMIT]

        return {
            "exitCode": code if code is not None else -1,
            "signal": reason or "",
            "stdout": stdout,
            "stderr": stderr,
            "truncated": (len(stdout) + len(stderr)) >= self.OUTPUT_TRUNC_LIMIT,
            "duration_ms": None,  # 可按需补充计时
        }

    @staticmethod
    def _first_token(command: str) -> str:
        try:
            lex = shlex.split(command, posix=True)
            return lex[0] if lex else ""
        except Exception:
            # 极端情况下失败，退化成简单拆分
            return command.strip().split(" ", 1)[0]

    @staticmethod
    def _generate_sandbox_profile() -> str:
        # 与 JS 版一致的严格配置（仅 macOS 有效）
        return textwrap.dedent("""
          (version 1)
          (deny default)
          (allow process-exec (literal "/bin/bash"))
          (allow process-exec (literal "/usr/bin/env"))
          (allow file-read*)
          (deny file-write*)
          (deny network*)
          (allow sysctl-read)
        """).strip()

    # ---------------- Convenience helpers (Git workflow etc.) ----------------
    async def run_command(self, command: str, context: ToolContext, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        """便捷的单次命令执行，返回一次性结果（不流式）。"""
        params = {"command": command, "timeout": timeout_ms or self.DEFAULT_TIMEOUT_MS}
        # 消费掉 progress，仅返回 result
        result = None
        async for msg in self.call(params, context):
            if msg["type"] == "result":
                result = msg["data"]
        return result or {"exitCode": -1, "stdout": "", "stderr": "no result", "signal": "internal"}

    async def handle_git_commit(self, context: ToolContext) -> AsyncGenerator[Dict[str, Any], None]:
        """
        对齐 JS 版：收集信息 -> 生成提交信息 -> heredoc 提交。
        真实使用中，建议将三个命令并发跑（下方就是并发）。
        """
        # Phase 1: parallel info gathering
        status_cmd = "git status --porcelain"
        diff_cmd = "git diff"
        log_cmd = "git log -5 --oneline"

        status_task = asyncio.create_task(self.run_command(status_cmd, context))
        diff_task = asyncio.create_task(self.run_command(diff_cmd, context))
        log_task = asyncio.create_task(self.run_command(log_cmd, context))

        status_res, diff_res, log_res = await asyncio.gather(status_task, diff_task, log_task)

        files_changed = len([ln for ln in status_res.get("stdout", "").splitlines() if ln.strip()])
        yield {
            "type": "progress",
            "toolUseID": context.current_tool_use_id,
            "data": {"status": "Analyzing changes...", "files": files_changed},
        }

        # Phase 2: naive commit analysis (你可以替换为更智能的摘要)
        message = self._draft_commit_message(status_res.get("stdout", ""), diff_res.get("stdout", ""))

        # Phase 3: commit via HEREDOC
        commit_cmd = f"""git commit -m "$(cat <<'EOF'
{message}

Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)\""""
        # 注意：若需要先 git add，可在外部先行调用；这里仅示例 commit。
        async for out in self.call({"command": commit_cmd}, context):
            yield out

    @staticmethod
    def _draft_commit_message(status_stdout: str, diff_stdout: str) -> str:
        # 非智能版：根据 status 粗略生成；生产可替换为 LLM 生成摘要
        if not status_stdout.strip():
            return "chore: no changes staged (empty commit skipped)"
        touched = []
        for ln in status_stdout.splitlines():
            if not ln.strip():
                continue
            # 形如 " M path" / "A  path"
            parts = ln.strip().split(maxsplit=1)
            if len(parts) == 2:
                touched.append(parts[1])
        touched_preview = ", ".join(touched[:5]) + (" ..." if len(touched) > 5 else "")
        return f"update: apply changes to {touched_preview}"


# ----------------- Utility: suppress context manager -----------------
class contextlib_suppress:
    def __enter__(self):  # noqa
        return self
    def __exit__(self, exc_type, exc, tb):  # noqa
        return True  # swallow exceptions
