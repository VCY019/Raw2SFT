from datasets import load_dataset
import pdb
import json
import os

parquet_path = "/GPFS/data/yuchifengting-1/OpenFedLLM/data/parquet/Wish-QA-MED-Falcon.parquet"
json_path = "/GPFS/data/yuchifengting-1/OpenFedLLM/data/train/Wish-QA-MED-Falcon.json"

dataset = load_dataset("parquet", data_files={'train': parquet_path})['train']
dataset = dataset.select_columns(["title_question", "long_answer", "text"])
data = []

for sample in dataset:
    data.append({
        "text": sample["text"],
        "input": "",
        "instruction": sample["title_question"],
        "output": sample["long_answer"]
    })

train_data = data[:-100]
test_data = data[-100:]
data = {"train": train_data, "test": test_data}

# Save the final data to a JSON file
os.makedirs(os.path.dirname(json_path), exist_ok=True)

with open(json_path,'w',encoding='utf-8') as output:
    json.dump(data, output, indent=4, ensure_ascii=False)



