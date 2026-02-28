import os
import json

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
        
    return data

def write_jsonl(data: list[dict], filepath: str) -> None:
    """Write a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to write
        filepath: Path to the output .jsonl file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            # Write each dictionary as a JSON line
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
        
def create_slime_format(data_dict, index):
    _format = {
        "prompt": [{"role": "user", "content": ""}],
        "label": "None", 
        "metadata": {
            "split": "train", 
            "index": index, 
            "answer": "None", 
            "task_instance": data_dict
        }
    }
    return _format

data_path = "/root/seele-slime/subset_2_6.json"
save_path = "/root/swe_gym_data_2_6.jsonl"
data = load_json(data_path)
print(data[0])
all_data = [create_slime_format(raw, index) for index, raw in enumerate(data)]
print(all_data[0])
write_jsonl(all_data, save_path)

    
