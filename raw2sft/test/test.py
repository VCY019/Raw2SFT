# Read wishqafalcon_data.json
from datasets import load_dataset
import datasets
import pdb
def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example

local_data_dir="/GPFS/data/yuchifengting-1/OpenFedLLM/data" 
dataset_name="/Wish-QA-Falcon-generated"
dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
dataset = load_dataset(dataset_name, split="train", field="train")
pdb.set_trace()

dataset = dataset.map(alpaca_format, desc=f"Preprocessing {dataset_name} for unified format.")
pdb.set_trace()

local_datasets = []
for i in range(10):
    local_datasets.append(dataset.shard(10, i))

pdb.set_trace()