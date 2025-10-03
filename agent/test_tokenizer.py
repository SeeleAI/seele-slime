from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/GLM-Z1-9B-0414/")


only_system = [{"role": "system", "content": ""}]

system_with_user = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "你好"},
]

system_with_user_with_assistant = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"},
]

system_with_user_with_assistant_with_tool = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"},
    {"role": "user", "content": "tool"}
]

system_with_user_with_assistant_with_tool_with_assistant = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"},
    {"role": "user", "content": "tool"},
    {"role": "assistant", "content": "observation"}
]

only_tool = [{"role": "user", "content": "tool"}]


all_msg = tokenizer.apply_chat_template(system_with_user_with_assistant_with_tool, add_generation_prompt=True,tokenize=False)
# print("raw_msg: ",all_msg)
# print("all raw tokens",tokenizer(all_msg,add_special_tokens=False))
assistant_msg = tokenizer.apply_chat_template(system_with_user_with_assistant, add_generation_prompt=False,tokenize=False)
# print("assistant_msg: ",assistant_msg)
# print("assistant raw tokens",tokenizer(assistant_msg,add_special_tokens=False))
only_tool = tokenizer.apply_chat_template(only_tool, add_generation_prompt=True,tokenize=False)
# print("user_msg: ",only_tool)
# print("tool raw tokens",tokenizer(only_tool,add_special_tokens=False))

obersaved_msg = tokenizer.apply_chat_template(system_with_user_with_assistant_with_tool_with_assistant, add_generation_prompt=False,tokenize=False)
# print("obersaved_msg: ",obersaved_msg)
# print("obersaved raw tokens",tokenizer(obersaved_msg,add_special_tokens=False))
# 151336, 198, 14163, 151337,
print(tokenizer(all_msg,add_special_tokens=False)["input_ids"][len(tokenizer(assistant_msg,add_special_tokens=False)["input_ids"]):])
