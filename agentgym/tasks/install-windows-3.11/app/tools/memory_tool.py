# app/tools/memory_tool.py
from __future__ import annotations
from typing import Any, Dict, AsyncGenerator, Optional, List

class ToolContext:
    def __init__(self) -> None:
        pass

class MemoryTool:
    description = (
        "Summarize the conversation and flush history except the system prompt. "
        "It returns a minimal context: [system, user(summary_with_prefix)]."
    )

    # —— 注意：这里是 JSON Schema 形式，方便 /get_tool_list -> schema_to_param_list 解析 ——
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Concise summary for the conversation history (will replace history)."
            },
            "system_prompt": {
                "type": "string",
                "description": "Optional new system prompt. If omitted, keep the existing one (upstream should fill)."
            }
        },
        "required": ["text"]
    }

    def __init__(self) -> None:
        pass

    async def call(
        self,
        input_data: Dict[str, Any],
        context: Optional[ToolContext] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        text = input_data.get("text")
        system_override = input_data.get("system_prompt")

        if not isinstance(text, str) or not text.strip():
            yield {"type": "result", "data": {"success": False, "error": "parameter 'text' must be a non-empty string"}}
            return

        base_notification = "You called the summarize tool, the new context is: "
        user_msg = {"role": "user", "content": base_notification + text.strip()}

        # 如果没有传入 system_override，这里用一个空 system 占位，上层收到后用当前会话的 system 覆盖
        system_msg = {"role": "system", "content": (system_override or "")}

        new_messages: List[Dict[str, str]] = [system_msg, user_msg]

        yield {"type": "result", "data": {"success": True, "new_messages": new_messages}}
