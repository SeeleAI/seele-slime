import json
import argparse

def convert_to_slime_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in data:
            # 提取必要字段
            prompt = item.get("prompt", [])
            label = item.get("reward_model", {}).get("ground_truth", "").split("\n####")[0]
            
            # 构建 metadata：复制 extra_info + 添加 env_config
            metadata = item.get("extra_info", {}).copy()
            metadata["env_name"] = "calc"

            # 构造 slime 所需格式
            new_item = {
                "prompt": prompt,
                "label": label,
                "metadata": metadata
            }

            # 写入一行 JSON 对象
            out_f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    print(f"✅ 转换完成！输出文件：{output_file}")

if __name__ == "__main__":
    convert_to_slime_format("agent/test/data/test.json", "agent/test/data/test.jsonl")