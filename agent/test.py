from typing import List, Dict
import torch
from transformers import AutoTokenizer
import re

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user",   "content": "Tell me a joke."},
    {"role": "assistant", "content": "Why did the chicken cross the road?"},
]

tokenizer = AutoTokenizer.from_pretrained("/root/qwen3-4b-agent/qwen3-4b")

m1 = messages[:2]
m2 = messages[:3]

print(tokenizer.apply_chat_template(m1, add_generation_prompt=True, tokenize=False))
print("-------------------------")
print(tokenizer.apply_chat_template(m2, add_generation_prompt=False, tokenize=False))
print("-------------------------")
# print(tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False))
len1 = len(tokenizer.encode("Hello! How can I help?")) + 1  # 1 for eos

t1 = tokenizer.apply_chat_template(m1, add_generation_prompt=True, tokenize=True)
t2 = tokenizer.apply_chat_template(m2, add_generation_prompt=False, tokenize=True)
loss_mask = [0] * len(t2)
loss_mask[len(t1):len(t1)+len1] = [1] * len1
ret = []
for t, m in zip(t2, loss_mask):
    if m == 1:
        ret.append(t)
        
print("filtered: ", [tokenizer.decode(ret)])

# print(f"eos_id : {tokenizer.eos_token}")
# print(input_ids)
# print(tokenizer.decode(input_ids))
# labels = labels[labels != -100]
# print(tokenizer.decode(labels))
# probe = [{"role":"user", "content": "||PROBE||"}]
# probe_message = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
# # print(probe_message)
# assistant_header = probe_message.split("||PROBE||")[1]
# if "\n" in assistant_header:
#     assistant_header = assistant_header.split("\n")[0]
# assistant_header_id = tokenizer.encode(assistant_header)[0]
# print(assistant_header_id)
# eos_id = tokenizer.eos_token_id

# full_seq = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
# # ptr1 = 0
# # ptr2 = 0
# # loss_mask = [0] * len(full_seq) 
# # while ptr1 < len(full_seq):
# #     # 1. ptr1 finds assistant header
# #     while full_seq[ptr1] != assistant_header_id:
# #         ptr1 += 1
# #         if ptr1 >= len(full_seq):
# #             break
# #     if ptr1 >= len(full_seq):
# #             break
# #     ptr2 = ptr1
# #     while full_seq[ptr2] != eos_id:
# #         ptr2 += 1
# #     # print(ptr1, ptr2)
# #     loss_mask[ptr1+1:ptr2+1] = [1] * (ptr2 - ptr1)
# #     ptr1 += 1

# loss_mask = [0] * len(full_seq)
# in_assistant_span = False  # start masking after we see an assistant header

# for i, tok in enumerate(full_seq):
#     if tok == assistant_header_id:
#         # Start masking from the *next* token
#         in_assistant_span = True
#         continue

#     if in_assistant_span:
#         loss_mask[i] = 1
#         if tok == eos_id:
#             in_assistant_span = False

    
# print(full_seq)
# print(loss_mask)

# print(len(full_seq), len(loss_mask))

# filtered = []
# for tid, m in zip(full_seq, loss_mask):
#     if m == 1:
#         filtered.append(tid)
        
# print(tokenizer.decode(filtered))