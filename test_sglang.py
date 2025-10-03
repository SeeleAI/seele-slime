from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
from transformers import AutoTokenizer

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0

server_process, port = launch_server_cmd(
"""
python3 -m sglang.launch_server --model-path /root/GLM-Z1-9B-0414 \
 --host 0.0.0.0 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")

import requests

url = f"http://localhost:{port}/v1/chat/generate"

tokenizer = AutoTokenizer.from_pretrained("/root/GLM-Z1-9B-0414", trust_remote_code=True)

message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
print(prompt)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)

print_highlight(response.json())

