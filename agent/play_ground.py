import asyncio
import os
import sys
sys.path.append(os.getcwd())
import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge

from transformers import AutoTokenizer
import json
from agent.tools.schema import register_tool_schema, TestMathToolsSchema, MemoryToolsSchema

from train_env_python.env import Env, EnvConfig
import pandas as pd
import json


def get_remaining_tokens(memory, current_length):
    return memory - current_length


def read_file(path):
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")
    for _, row in df.iterrows():
        yield row.to_dict()


if __name__ == "__main__":
    # /root/qwen3-30b-instruct/
    # /root/qwen3-coder-30b-agent/qwen3-coder-30b
    path = "/root/qwen3-30b-instruct/"
    llm = sgl.Engine(model_path=path, max_total_tokens=64000)
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    file_path = "agent/agent_gym_data.jsonl"

    sampling_params = {"temperature": 0.8, "top_p": 1.0, "skip_special_tokens": False, "max_new_tokens": 8192}
    
    max_turns = 20
    max_len = 6000
    virtual_memory = max_len - 1024
    timeout_tasks = []
    ctx_limit_tasks = []
    # for data in read_file(file_path):
    env_name = "test-fibonacci-server"
    env = Env()
    config = EnvConfig(image_name=env_name)
    
    try:
        system_messages = env.reset(config)
    except Exception as e:
        print(f"Error resetting environment {env_name}: {e}")
        timeout_tasks.append(env_name)
    # print(system_messages[0]["content"])
    # exit(0)
    
    print("\n", system_messages[0]['content'], "\n")
    

    # messages = system_messages + [{"role": "user", "content": env.task_prompt}]
    messages = system_messages + [{"role": "user", "content": "Call TestTool immeidiately!!!"}]
    input_token_len = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
    # sample.messages[1]["content"] += f" Remaining tokens: {get_remaining_tokens(virtual_memory, input_token_len)}"
    warning_message = f"\n** <token_budget> Used: {input_token_len}/Total: {virtual_memory}; Remaining: {get_remaining_tokens(virtual_memory, input_token_len)} </token_budget> **"
    messages[1]["content"] += warning_message
    
    # maybe record the extra sample when swap out happens
    all_samples = []
    reward = 0.0
    print(f"User task: {env.task_prompt}")
    for turn in range(max_turns):
        current_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = llm.generate(current_prompt, sampling_params)
        print(f"======turn {turn}=======")
        print(f"User: {messages[-1]["content"]}")
        print(f"Response: {outputs["text"]}")
        messages.append({"role": "assistant", "content": outputs["text"]})
        step_result = env.step(messages)
        print(f"Reward: {step_result.reward}")
        print("\n")
        messages = step_result.next_observation
        if any(["failed with error: request timeout" in messages[i]["content"] for i in range(len(messages))]):
            print(f"Task {env_name} timeout")
            timeout_tasks.append(env_name)
            break
        next_input_length = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
        if next_input_length > 64000:
            print(f"Task {env_name} input length exceeded")
            # timeout_tasks.append(env_name)
            break
        if messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": f"\n** <token_budget> Used: {next_input_length}/Total: {virtual_memory}; Remaining: {get_remaining_tokens(virtual_memory, next_input_length)} </token_budget> **"})
        else:
            messages[-1]["content"] += f"\n** <token_budget> Used: {next_input_length}/Total: {virtual_memory}; Remaining: {get_remaining_tokens(virtual_memory, next_input_length)} </token_budget> **"
        if step_result.reward is not None:
            reward += step_result.reward
            
        if step_result.done:
            break
            
        x = input()
        
    env.close()
        
    # with open("timeout_tasks.json", "w") as f:
    #     json.dump(timeout_tasks, f)
    #     f.close()
    
    # print(f"Timeout tasks: {timeout_tasks}")