import os
import base64
import mimetypes
from pathlib import Path

# 处理图片用 pillow
from PIL import Image
from io import BytesIO


class ReadFileTool:
    name = "ReadFileTool"

    # === 新增：长描述（结合你给的说明，已明确 PDF 需额外依赖/当前未实现） ===
    description = (
        "Reads a file from the local filesystem with line numbers (cat -n style) and basic image support.\n\n"
        "Assumptions & Usage:\n"
        "- The file_path parameter must be an absolute path (not relative).\n"
        "- By default, reads up to 2000 lines, starting from line 1.\n"
        "- You can optionally specify a line offset and limit for long files.\n"
        "- Any line longer than 2000 characters will be truncated.\n"
        "- Text results are returned in cat -n style with 1-based line numbers.\n"
        "- This tool can read images (e.g., PNG, JPG): images are resized to fit within 1024x1024 and returned as base64.\n"
        "- For Jupyter notebooks (.ipynb), cells can be paged using offset/limit.\n"
        "- PDF reading is not enabled by default in this implementation. If needed, add a PDF handler "
        "(e.g., with PyPDF2 or PyMuPDF) and route .pdf files accordingly.\n"
        "- Reading a non-existent file will raise an error.\n"
    )

    # === 新增：输入模式（JSON Schema Draft-07 结构） ===
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to read"
            },
            "offset": {
                "type": "number",
                "description": "The line/cell number to start reading from (1-based)."
            },
            "limit": {
                "type": "number",
                "description": "The number of lines/cells to read (max items)."
            }
        },
        "required": ["file_path"],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
    }

    is_read_only = True

    def __init__(self):
        # 用来缓存已读文件状态
        self.read_file_state = {}

    # === 可选：支持以 dict 入参，按 input_schema 做轻量校验 ===
    async def call_with_params(self, params: dict):
        file_path, offset, limit = self._validate_and_extract(params)
        return await self.call(file_path=file_path, offset=offset, limit=limit)

    def _validate_and_extract(self, params: dict):
        # additionalProperties: false
        allowed = set(self.input_schema["properties"].keys())
        extra_keys = set(params.keys()) - allowed
        if extra_keys:
            raise ValueError(f"Unexpected parameter(s): {sorted(extra_keys)}")

        # required
        if "file_path" not in params:
            raise ValueError("Missing required parameter: file_path")

        file_path = params["file_path"]
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string (absolute path).")
        if not file_path.startswith("/"):
            # 强制绝对路径（可按需放宽）
            raise ValueError("file_path must be an absolute path.")

        # offset/limit（可选）
        offset = params.get("offset", 1)
        limit = params.get("limit", 2000)

        # 数值类型检查（schema 中是 number，这里更严格用 int）
        if not isinstance(offset, (int, float)) or int(offset) != offset or offset < 1:
            raise ValueError("offset must be a positive integer (>=1).")
        if not isinstance(limit, (int, float)) or int(limit) != limit or limit < 1:
            raise ValueError("limit must be a positive integer (>=1).")

        return file_path, int(offset), int(limit)

    async def call(self, file_path: str, offset: int = 1, limit: int = 2000):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # PDF（描述中有提及，这里明确未实现，防止误导）
        if file_path.lower().endswith(".pdf"):
            raise NotImplementedError(
                "PDF reading is not implemented in this class. "
                "You can add a PDF handler (e.g., PyPDF2/PyMuPDF) and route .pdf here."
            )

        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type and isinstance(mime_type, str) and mime_type.startswith("image/"):
            # 处理图片
            return await self.read_image(file_path)

        if file_path.endswith(".ipynb"):
            # 处理 Jupyter Notebook
            return await self.read_notebook(file_path, offset, limit)

        # 默认处理文本
        content = await self.read_text_file(file_path, offset, limit)

        # 更新缓存
        self.read_file_state[file_path] = {
            "content": content["fullContent"],
            "timestamp": os.path.getmtime(file_path),
        }

        return content

    async def read_text_file(self, file_path: str, offset: int, limit: int):
        lines = []
        line_number = 0
        truncated = False

        # 若编码失败，可将 errors 改为 "replace" 以容错
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line_number += 1
                if line_number >= offset and len(lines) < limit:
                    line = raw_line.rstrip("\n")
                    truncated_line = line[:2000] + "... (truncated)" if len(line) > 2000 else line
                    lines.append(f"{line_number}\t{truncated_line}")

                if len(lines) >= limit:
                    truncated = True
                    break

        return {
            "formattedContent": "\n".join(lines),
            "fullContent": Path(file_path).read_text(encoding="utf-8"),
            "lineCount": line_number,
            "truncated": truncated,
        }

    async def read_image(self, file_path: str):
        with open(file_path, "rb") as f:
            buffer = f.read()

        img = Image.open(BytesIO(buffer))
        original_size = img.size

        # resize if too large
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024))

        output_buffer = BytesIO()
        fmt = (img.format or "PNG")
        try:
            img.save(output_buffer, format=fmt)
        except Exception:
            # fallback: 转 PNG
            fmt = "PNG"
            img = img.convert("RGBA")
            img.save(output_buffer, format=fmt)

        base64_data = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return {
            "type": "image",
            "mimeType": f"image/{fmt.lower()}",
            "base64": base64_data,
            "dimensions": {
                "original": {"width": original_size[0], "height": original_size[1]},
                "processed": {"width": img.width, "height": img.height},
            },
        }

    async def read_notebook(self, file_path: str, offset: int, limit: int):
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 简单返回部分 cell 内容
        cells = data.get("cells", [])[offset - 1: offset - 1 + limit]
        cell_texts = []
        for idx, cell in enumerate(cells, start=offset):
            src = "".join(cell.get("source", []))
            cell_texts.append(f"{idx}\t{src.strip()}")

        total = len(data.get("cells", []))
        truncated = (offset - 1 + len(cells)) < total

        return {
            "formattedContent": "\n".join(cell_texts),
            "fullContent": json.dumps(data, ensure_ascii=False, indent=2),
            "lineCount": total,
            "truncated": truncated
        }

    def map_tool_result(self, result):
        if result.get("type") == "image":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": result["mimeType"],
                    "data": result["base64"],
                },
            }

        if not result.get("formattedContent"):
            return {
                "type": "text",
                "text": "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>",
            }

        return {
            "type": "text",
            "text": result["formattedContent"] + ("\n... (content truncated)" if result.get("truncated") else "")
        }
