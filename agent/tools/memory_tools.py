

class MemoryTools():
    def __init__(self, avaialble_tools: tuple[str] | str = None):
        if avaialble_tools == "all":
            self.available_tools = (
                "summarize",
            )
        else:
            self.available_tools = avaialble_tools
        assert self.available_tools, f"Should provide at least one tool."
        
    def summarize(self, text: str, messages: list[dict]):
        sys_prompt = messages[0]
        assert messages[0]["role"] == "system", f"The first message should be system message, got {messages[0]["role"]}"
        base_notification = "You called the summarize tool, the new context is: "
        new_context = {"role": "user", "content": base_notification + text}
        new_messages = [sys_prompt, new_context]
        
        return new_messages
