#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import sys
import warnings

# 屏蔽 torch.cuda 的 pynvml FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")

sys.path.append(os.getcwd())

import sglang as sgl
import sglang.test.doc_patch  # noqa: F401
from transformers import AutoTokenizer


from agentgym.test_files.env import Env
from agentgym.test_files.models import EnvConfig


def _looks_like_no_tool_calls(obs_list):
    """根据观察判断是否是“无工具调用”的终止信号。"""
    if not obs_list:
        return False
    text = " ".join(str(m.get("content", "")).lower() for m in obs_list)
    return ("no tool calls" in text) or ("no tool" in text and "call" in text)


def _format_observations_as_user(obs_list):
    """
    把环境返回的多条 observation 合成为一条 user 消息，格式如下：
    Observations:
    - user: <第一条内容（可多行）>
    - system: <第二条内容（可多行）>
    ...
    保留换行与原始文本，便于模型继续思考/解析。
    """
    lines = ["Observations:"]
    for m in obs_list:
        role = m.get("role", "user")
        content = str(m.get("content", "")).rstrip("\n")
        if "\n" in content:
            head, *tail = content.splitlines()
            lines.append(f"- {role}: {head}")
            for t in tail:
                lines.append(t)
        else:
            lines.append(f"- {role}: {content}")
    text = "\n".join(lines)
    return {"role": "user", "content": f"{text}"}


def _append_testtool_call_and_step(env, messages):
    """向对话中追加一条 TestTool 的 <tool> 调用块，并立刻 env.step 执行一次，将观察并回对话。"""
    messages.append({
        "role": "assistant",
        "content": """<tool>
<id>9</id>
<name>TestTool</name>
<workdir>/app</workdir>
<timeout>600</timeout>
<results_dir>/app/.test_results</results_dir>
</tool>"""
    })
    sr = env.step(messages)
    print("--- StepResult (forced TestTool) ---")
    print("Reward          :", getattr(sr, "reward", None))
    print("Done            :", getattr(sr, "done", None))
    print("Modified Context:", getattr(sr, "modified_context", None))
    if getattr(sr, "next_observation", None):
        obs_user = _format_observations_as_user(sr.next_observation)
        print("---- Synthesized Observation (as user) ----")
        print(obs_user["content"])
        messages.append(obs_user)


def main():
    # ===== 模型 =====
    model_path = "/root/qwen3-30b-agent/qwen3-30b-coder/"
    llm = sgl.Engine(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = {
        "temperature": 0.8,
        "top_p": 1.0,
        "skip_special_tokens": False,
    }

    # ===== 初始化环境 =====
    env = Env()
    exit_reason = None

    try:
        messages = env.reset(EnvConfig(image_name="http1.4-hello-world"))

        with open("/root/seele-slime/agentgym/tasks/hello-world/task_prompt.txt", "r", encoding="utf-8") as f:
            task_prompt = f.read()

        messages.append({
            "role": "user",
            "content": task_prompt
        })

        # ===== 对话循环 =====
        done = False
        turn = 0
        max_turns = 100

        while not done and turn < max_turns:
            turn += 1
            print(f"\n===== Turn {turn} =====")

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = llm.generate([prompt_text], sampling_params)
            assistant_response = outputs[0]["text"]
            print("Assistant generated:\n", assistant_response)

            # 1) 模型回复
            messages.append({"role": "assistant", "content": assistant_response})

            # 2) 环境执行（解析最后一条 assistant 的工具调用）
            try:
                step_result = env.step(messages)
            except Exception as e:
                exit_reason = f"error: env.step exception: {repr(e)}"
                print(exit_reason)
                break

            print("--- StepResult ---")
            print("Reward          :", step_result.reward)
            print("Done            :", step_result.done)
            print("Modified Context:", step_result.modified_context)

            # 3) 合并观察 -> 一条新的 user 消息
            if step_result.next_observation:
                obs_user = _format_observations_as_user(step_result.next_observation)
                print("---- Synthesized Observation (as user) ----")
                print(obs_user["content"])
                messages.append(obs_user)

            # 4) 退出判定（含 done 纠偏）
            no_tool = _looks_like_no_tool_calls(step_result.next_observation)

            if step_result.done:
                if step_result.modified_context and step_result.next_observation and not no_tool:
                    print("Note: env reported done=True but observations contain actionable output; continuing.")
                    done = False
                    continue
                else:
                    # 先强制补一条 TestTool 调用并执行一次
                    print("\n=== Forcing TestTool before exit (done path) ===")
                    _append_testtool_call_and_step(env, messages)

                    exit_reason = "no_tool_calls" if no_tool else "done"
                    break

            if no_tool:
                # 无工具调用，也先补一条 TestTool 调用并执行一次
                print("\n=== Forcing TestTool before exit (no_tool_calls path) ===")
                _append_testtool_call_and_step(env, messages)

                exit_reason = "no_tool_calls"
                break

        if exit_reason is None and turn >= max_turns:
            exit_reason = "max_turns"

    except Exception as e:
        exit_reason = f"error: {repr(e)}"
    finally:
        try:
            env.close()
        except Exception:
            pass
        llm.shutdown()

    print(f"\n=== Loop finished. reason = {exit_reason} ===")


if __name__ == "__main__":
    main()
