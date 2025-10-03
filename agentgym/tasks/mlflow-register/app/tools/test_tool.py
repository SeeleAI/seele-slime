from __future__ import annotations
import asyncio
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

@dataclass
class ToolContext:
    # 缺省用 os.getcwd()（见 call() 内部兜底）
    cwd: str = ""

class TestTool:
    """
    Run tests inside the *current* container only.
    Priority:
    1) run-script (e.g., ./run-tests.sh) if exists
    2) pytest -q ./tests  (if ./tests exists)
    3) pytest -q ./test   (if  ./test exists)
    4) pytest -q          (pytest discovery)

    Behavior:
    - Auto-detect test command (priority above)
    - Optional setup_script runs first if provided.
    - Timeout kills process (default 600s, rc=124).

    Result:
    exit_code, passed, stdout/stderr (tail 20k), test_cmd, mode="local".
    """

    name = "TestTool"
    description = """
    Runs unit tests in the current working directory, *inside this container only*.
    No cross-container execution is supported.
    """

    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "workdir": {
                "type": "string",
                "description": "Working directory (if not provided, use current process cwd)"
            },
            "results_dir": {
                "type": "string",
                "description": "Directory to store junit.xml, default <workdir>/.test_results"
            },
            "setup_script": {
                "type": "string",
                "description": "(Optional) Pre-execution script, e.g., ./tests/setup-uv-pytest.sh"
            },
            "test_cmd": {
                "type": "string",
                "description": "(Optional) Fully customized test command, overrides auto detection if provided"
            },
            "run_script": {
                "type": "string",
                "description": "Script name to run first in auto mode, default run-tests.sh",
                "default": "run-tests.sh"
            },
            "timeout": {
                "type": "number",
                "description": "Test timeout (seconds)",
                "default": 600
            },
            "env": {
                "type": "object",
                "description": "(Optional) Extra environment variables, e.g., {\"http_proxy\":\"...\"}"
            }
        },
        "required": []
    }

    @staticmethod
    def _mk_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = os.environ.copy()
        if extra:
            for k, v in extra.items():
                if isinstance(k, str) and isinstance(v, str):
                    env[k] = v
        return env

    async def _run(self, cmd: str, cwd: str, env: Dict[str, str], timeout: int) -> Dict[str, Any]:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        out_lines, err_lines = [], []
        try:
            loop = asyncio.get_event_loop()

            async def read_stream(stream, sink):
                while True:
                    line = await loop.run_in_executor(None, stream.readline)
                    if not line:
                        break
                    sink.append(line)

            await asyncio.wait_for(asyncio.gather(
                read_stream(proc.stdout, out_lines),
                read_stream(proc.stderr, err_lines),
            ), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"rc": 124, "out": "".join(out_lines), "err": f"[TIMEOUT] {timeout}s"}
        finally:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass

        return {"rc": proc.returncode, "out": "".join(out_lines), "err": "".join(err_lines)}

    def _auto_pick_test_cmd(self, workdir: str, run_script: str, junit_path: str) -> str:
        """
        根据优先级选择测试命令。返回完整 shell 命令字符串。
        """
        candidate = os.path.join(workdir, run_script)
        if os.path.isfile(candidate):
            return f"chmod +x {shlex.quote(candidate)} && {shlex.quote(candidate)}"

        tests_dir = os.path.join(workdir, "tests")
        if os.path.isdir(tests_dir):
            return f"pytest -q --maxfail=1 --disable-warnings --junitxml {shlex.quote(junit_path)} ./tests"

        test_dir = os.path.join(workdir, "test")
        if os.path.isdir(test_dir):
            return f"pytest -q --maxfail=1 --disable-warnings --junitxml {shlex.quote(junit_path)} ./test"

        return f"pytest -q --maxfail=1 --disable-warnings --junitxml {shlex.quote(junit_path)}"

    async def call(self, args: Dict[str, Any], context: Optional[ToolContext] = None
                   ) -> AsyncGenerator[Dict[str, Any], None]:
        # 解析参数
        workdir = args.get("workdir") or (context.cwd if context and context.cwd else os.getcwd())
        results_dir = args.get("results_dir") or os.path.join(workdir, ".test_results")
        setup_script = args.get("setup_script") or ""
        test_cmd = args.get("test_cmd") or ""
        run_script = args.get("run_script") or "run-tests.sh"
        timeout = int(args.get("timeout") or 600)
        extra_env = args.get("env") or {}

        # 结果目录
        os.makedirs(results_dir, exist_ok=True)
        junit_path = os.path.join(results_dir, "junit.xml")
        env = self._mk_env(extra_env)

        # 事件：准备
        yield {"type": "event", "data": {"stage": "prepare", "workdir": workdir}}

        # 可选预处理
        if setup_script:
            # 允许相对路径脚本，按 workdir 解析
            setup_cmd = f"chmod +x {shlex.quote(setup_script)} && {shlex.quote(setup_script)}"
            yield {"type": "event", "data": {"stage": "setup", "cmd": setup_cmd}}
            ret = await self._run(setup_cmd, cwd=workdir, env=env, timeout=timeout)
            yield {"type": "event", "data": {"stage": "setup_done", "rc": ret['rc'], "stderr": ret['err']}}

        # 选择测试命令
        if not test_cmd:
            test_cmd = self._auto_pick_test_cmd(workdir, run_script, junit_path)

        final_cmd = test_cmd  # 仅本容器内执行

        yield {"type": "event", "data": {"stage": "run_tests", "cmd": final_cmd}}

        # 注意这里把 cwd 设为 workdir，保持与 _auto_pick_test_cmd 语义一致
        ret = await self._run(final_cmd, cwd=workdir, env=env, timeout=timeout)

        passed = (ret["rc"] == 0)
        result = {
            "exit_code": ret["rc"],
            "passed": passed,
            "workdir": workdir,
            "results_dir": results_dir,
            "junit_xml": junit_path if os.path.exists(junit_path) else "",
            "stdout": ret["out"][-20000:],
            "stderr": ret["err"][-20000:],
            "timeout_sec": timeout,
            "test_cmd": final_cmd,
            "mode": "local",
        }
        yield {"type": "result", "data": result}
