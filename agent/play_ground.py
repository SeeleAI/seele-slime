import asyncio
import os
import sys
sys.path.append(os.getcwd())
import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge

from transformers import AutoTokenizer
from agent.constant import TEST_MATH_SYS_PROMPT
import json
from agent.tools.schema import register_tool_schema, TestMathToolsSchema, MemoryToolsSchema

if __name__ == "__main__":
    path = "/root/qwen25-14b-agent/qwen25-14b/"
    llm = sgl.Engine(model_path=path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    sampling_params = {"temperature": 0.8, "top_p": 1.0, "skip_special_tokens": False}
    tool_schema = register_tool_schema("all", TestMathToolsSchema(), MemoryToolsSchema())
    sys_prompt = TEST_MATH_SYS_PROMPT.replace("{{tools}}", json.dumps(tool_schema, indent=2,ensure_ascii=False))

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Calculate the following expression: 3 ☽ 1 ☽ 2 ❀ 2, remaining tokens: 2"}
    ]
    prompt = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

    outputs = llm.generate(prompt, sampling_params)
    for prompt, output in zip(prompt, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        
    llm.shutdown()