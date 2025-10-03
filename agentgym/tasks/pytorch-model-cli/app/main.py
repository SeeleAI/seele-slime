# app/main.py
from __future__ import annotations
import os
import time
import traceback
import shutil
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# ==== 引入工具实现 ====
from app.tools.bash_tool import BashTool, ToolContext as BashCtx
from app.tools.edit_file_tool_with_schema import EditFileTool, ToolContext as EditCtx, CachedFile
from app.tools.grep_tool import GrepTool
from app.tools.read_file_tool import ReadFileTool
from app.tools.test_tool import TestTool, ToolContext as TestCtx
from app.tools.memory_tool import MemoryTool, ToolContext as MemoryCtx


app = FastAPI(title="Tool Gateway", version="0.1.0")

# ---------- 工具注册 ----------
REGISTRY: Dict[str, Dict[str, Any]] = {
    "BashTool": {
        "impl": BashTool(),
        "schema": BashTool.input_schema,
        "description": BashTool.description,
        "kind": "async_gen",
    },
    "Edit": {
        "impl": EditFileTool(),
        "schema": EditFileTool.input_schema,
        "description": EditFileTool.description,
        "kind": "async_gen",
    },
    "GrepTool": {
        "impl": GrepTool,
        "schema": GrepTool.input_schema,
        "description": GrepTool.description,
        "kind": "async_gen",
    },
    "ReadFileTool": {
        "impl": ReadFileTool(),
        "schema": ReadFileTool.input_schema,
        "description": ReadFileTool.description,
        "kind": "read",
    },
    "TestTool": {
        "impl": TestTool(),
        "schema": TestTool.input_schema,
        "description": TestTool.description,
        "kind": "async_gen",
    },
    "MemoryTool": {
        "impl": MemoryTool(),               
        "schema": MemoryTool.input_schema,  
        "description": MemoryTool.description,
        "kind": "async_gen",                
    },  
}

# ---------- 上传目录 ----------
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件到容器 /tmp/uploads 下，返回可用于 ReadFileTool 的路径
    """
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    return {"status": "success", "path": save_path}


# ---------- 统一请求/响应模型 ----------
class RunRequest(BaseModel):
    tool: str = Field(..., description="工具名，如 BashTool / Edit / GrepTool / ReadFileTool / AgentTool")
    args: Dict[str, Any] = Field(default_factory=dict, description="工具入参（按各自 input_schema 提供）")


def schema_to_param_list(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out: List[Dict[str, Any]] = []
    for name, meta in props.items():
        out.append({
            "name": name,
            "type": meta.get("type", "string"),
            "description": meta.get("description", ""),
            "required": name in required,
        })
    return out


@app.get("/get_tool_list")
def get_tool_list():
    tools = []
    for name, spec in REGISTRY.items():
        tools.append({
            "name": name,
            "description": str(spec.get("description") or "").strip(),
            "params": schema_to_param_list(spec.get("schema") or {}),
        })
    return {"tools": tools, "count": len(tools)}


# ---------- 共享上下文（Read -> Edit） ----------
class SharedEditContext:
    def __init__(self):
        self.read_file_state: Dict[str, CachedFile] = {}
        self.current_tool_use_id: str = ""


SHARED_EDIT_CTX = SharedEditContext()


async def _consume_async_gen(gen: AsyncGenerator[Dict[str, Any], None]) -> Dict[str, Any]:
    result: Optional[Dict[str, Any]] = None
    async for msg in gen:
        if isinstance(msg, dict) and msg.get("type") == "result":
            result = msg.get("data")
    if result is None:
        raise RuntimeError("tool returned no result")
    return result

# ---------- 统一封装：成功=200，业务失败=422 ----------
def _wrap_result(tool_name: str, result: Any) -> Dict[str, Any]:
    if isinstance(result, dict) and result.get("passed") is False:
        # 业务失败
        raise HTTPException(status_code=422, detail=result)
    return {"status": "success", "tool": tool_name, "result": result}


@app.post("/run")
async def run(req: RunRequest):
    spec = REGISTRY.get(req.tool)
    if not spec:
        raise HTTPException(status_code=404, detail=f"未知工具: {req.tool}")

    schema = spec.get("schema") or {}
    required = schema.get("required") or []
    for k in required:
        if k not in req.args:
            raise HTTPException(status_code=400, detail=f"缺少必要参数: {k}")

    try:
        impl = spec["impl"]

        # ------ ReadFileTool ------
        if req.tool == "ReadFileTool":
            read_tool: ReadFileTool = impl  # type: ignore
            try:
                result = await read_tool.call_with_params(req.args)
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "FILE_NOT_FOUND",
                        "message": str(e),
                        "file_path": req.args.get("file_path")
                    }
                )
            except PermissionError as e:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "PERMISSION_DENIED",
                        "message": str(e),
                        "file_path": req.args.get("file_path")
                    }
                )
            # 缓存逻辑保持
            file_path = req.args.get("file_path")
            if isinstance(file_path, str) and os.path.exists(file_path) and isinstance(result, dict):
                full = result.get("fullContent")
                if isinstance(full, str):
                    st = os.stat(file_path)
                    SHARED_EDIT_CTX.read_file_state[file_path] = CachedFile(
                        content=full,
                        timestamp_ns=st.st_mtime_ns,
                    )
            return _wrap_result(req.tool, result)

        # ------ Edit ------
        if req.tool == "Edit":
            file_path = req.args.get("file_path")
            if not isinstance(file_path, str):
                raise HTTPException(status_code=400, detail="Edit 需要 string 类型的 file_path")
            if file_path not in SHARED_EDIT_CTX.read_file_state:
                raise HTTPException(status_code=400, detail="File must be read with Read tool before editing")

            edit_tool: EditFileTool = impl  # type: ignore
            ctx = EditCtx()
            ctx.read_file_state.update(SHARED_EDIT_CTX.read_file_state)
            gen = edit_tool.call(req.args, ctx)
            result = await _consume_async_gen(gen)
            return _wrap_result(req.tool, result)

        # ------ BashTool ------
        if req.tool == "BashTool":
            bash: BashTool = impl  # type: ignore
            ctx = BashCtx(cwd=os.getcwd())
            gen = bash.call(req.args, ctx)
            result = await _consume_async_gen(gen)
            return _wrap_result(req.tool, result)

        # ------ GrepTool ------
        if req.tool == "GrepTool":
            gen = impl.call(req.args, context=type("C", (), {"currentToolUseId": ""})())
            result = await _consume_async_gen(gen)
            return _wrap_result(req.tool, result)
        
        # ------ TestTool ------
        if req.tool == "TestTool":
            test_tool: TestTool = spec["impl"]  # type: ignore
            import os as _os
            ctx = TestCtx(cwd=req.args.get("workdir") or _os.getcwd())
            gen = test_tool.call(req.args, context=ctx)
            result = await _consume_async_gen(gen)
            return _wrap_result(req.tool, result)
        
        # ------ MemoryTool ------
        if req.tool == "MemoryTool":
            ctx = MemoryCtx()
            gen = impl.call(req.args, context=ctx)
            result = await _consume_async_gen(gen)
            return _wrap_result(req.tool, result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"执行失败: {e}")
