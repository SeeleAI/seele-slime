# from typing import List, Dict
# import torch
from transformers import AutoTokenizer
# import re

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user",   "content": "Hi!"},
#     {"role": "assistant", "content": "Hello! How can I help?"},
#     {"role": "user",   "content": "Tell me a joke."},
#     {"role": "assistant", "content": "Why did the chicken cross the road?"},
#     {"role": "user", "content": "I don't know."},
#     {"role": "assistant", "content": "Because it was on the other side of the road."}
# ]

tokenizer = AutoTokenizer.from_pretrained("/root/qwen3-30b-agent/qwen3-30b-coder/")

# # m1 = messages[:2]
# # m2 = messages[:3]

# # print(tokenizer.apply_chat_template(m1, add_generation_prompt=True, tokenize=False))
# # print("-------------------------")
# # print(tokenizer.apply_chat_template(m2, add_generation_prompt=False, tokenize=False))
# # print("-------------------------")
# # # print(tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False))
# # len1 = len(tokenizer.encode("Hello! How can I help?")) + 1  # 1 for eos

# # t1 = tokenizer.apply_chat_template(m1, add_generation_prompt=True, tokenize=True)
# # t2 = tokenizer.apply_chat_template(m2, add_generation_prompt=False, tokenize=True)
# # loss_mask = [0] * len(t2)
# # loss_mask[len(t1):len(t1)+len1] = [1] * len1
# # ret = []
# # for t, m in zip(t2, loss_mask):
# #     if m == 1:
# #         ret.append(t)
        
# # print("filtered: ", [tokenizer.decode(ret)])


# # all_input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
# # loss_mask = [0] * len(all_input_ids)
# # for i in range(1, len(messages[1:])//2+1):
# #     prompt = messages[:i*2]
# #     prompt_response = messages[:i*2+1]
# #     t1 = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
# #     t2 = tokenizer.apply_chat_template(prompt_response, add_generation_prompt=False, tokenize=True)
# #     len1 = len(tokenizer.encode(messages[i*2]["content"])) + 1
# #     loss_mask[len(t1):len(t1)+len1] = [1] * len1
    
# # len_of_prompt = len(tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True, tokenize=True))
# # len_of_response = len(all_input_ids) - len_of_prompt

# # print(len_of_response)
# # print(len(loss_mask[len_of_prompt:]))

# # print([tokenizer.decode(all_input_ids[len_of_prompt:])])

# all_input_ids = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=False, tokenize=True
# )

# # Start with zeros for the whole conversation token length.
# loss_mask = [0] * len(all_input_ids)

# def prefix_token_len(msgs_prefix: List[dict[str, str]], add_gen_prompt: bool) -> int:
#     """Helper: returns length of tokenized chat template for a prefix of messages."""
#     return len(
#         tokenizer.apply_chat_template(msgs_prefix, add_generation_prompt=add_gen_prompt, tokenize=True)
#     )

# # Mark assistant response ranges.
# # We compute, for each assistant message at index i, the token index where that assistant response begins
# # (using the chat template with generation prompt enabled for the prefix), then mark the length of the response.
# for idx, msg in enumerate(messages):
#     if msg.get("role") != "assistant":
#         continue

#     # Token length of everything before the assistant's generated response.
#     # We set add_generation_prompt=True so the tokenizer formats the prompt exactly like the generation-time input.
#     start = prefix_token_len(messages[:idx], add_gen_prompt=True)

#     # Token length for this assistant message content (+1 for EOS as in original code).
#     assistant_content_tokens = len(tokenizer.encode(msg["content"]))
#     response_len = assistant_content_tokens + 1  # +1 for EOS token

#     # Cap ranges to avoid indexing errors in case of slight mismatches.
#     end = min(start + response_len, len(loss_mask))

#     loss_mask[start:end] = [1] * (end - start)

# # Compute the prompt length used by the generator for the first user/sys turn (keeps original behavior).
# len_of_prompt = prefix_token_len(messages[:2], add_gen_prompt=True)

# # Generation portion length
# gen_len = len(all_input_ids) - len_of_prompt
# gen_len = max(0, gen_len)


    
# filtered_token = []
# for t, m in zip(all_input_ids[len_of_prompt:], loss_mask[len_of_prompt:]):
#     if m == 1:
#         filtered_token.append(t)
        
# print([tokenizer.decode(filtered_token)])

def square_the_number(num: float) -> dict:
    return num ** 2

tools=[
    {
        "type":"function",
        "function":{
            "name": "square_the_number",
            "description": "output the square of the number.",
            "parameters": {
                "type": "object",
                "required": ["input_num"],
                "properties": {
                    'input_num': {
                        'type': 'number', 
                        'description': 'input_num is a number that will be squared'
                        }
                },
            }
        }
    }
]

# Define LLM
# client = OpenAI(
#     # Use a custom endpoint compatible with OpenAI API
#     base_url='http://localhost:8000/v1',  # api_base
#     api_key="EMPTY"
# )
 
messages = [{"role": "system", "content": "you are an assistant"}, {'role': 'user', 'content': 'square the number 1024'}]

# completion = client.chat.completions.create(
#     messages=messages,
#     model="Qwen3-Coder-30B-A3B-Instruct",
#     max_tokens=65536,
#     tools=tools,
# )
print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, tools=tools))
