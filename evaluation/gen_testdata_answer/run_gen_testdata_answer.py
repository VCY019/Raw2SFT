import datasets
import argparse
import json
import sys
sys.path.append("../../")
from tqdm import tqdm
import os
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from utils.template import TEMPLATE_DICT

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--dataset_path", type=str, default=None)
args = parser.parse_args()
print(args)

if args.use_vllm and args.lora_path is not None:
    raise ValueError("Cannot use both VLLM and LORA, need to merge the lora and then use VLLM")

template = TEMPLATE_DICT[args.template][0]
print(f">> You are using template: {template}")

# ============= Load dataset =============
eval_set = datasets.load_dataset("json", data_files=args.dataset_path, field="test")['train']
max_new_tokens=1024

# ============= Extract model name and dataset name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model
_, json_name = os.path.split(args.dataset_path)
dataset_name = json_name.split(".")[0]

# ============= Load previous results if exists =============
if args.use_vllm:
    result_path = f"./data/{dataset_name}/model_answer_vllm/{model_name}.json"
else:
    result_path = f"./data/{dataset_name}/model_answer/{model_name}.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)

if os.path.exists(result_path):
    with open(result_path, "r") as f:
        result_list = json.load(f)
else:
    result_list = []
existing_len = len(result_list)
print(f">> Existing length: {existing_len}")

# ============= Generate responses =============
if args.use_vllm:
    model = LLM(model=args.base_model_path)
    input_list = [template.format(example["instruction"], "", "")[:-1] for example in eval_set] 
    input_list = input_list[existing_len:]
    print(f">> Example input: {input_list[0]}")
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=max_new_tokens)
    generations = model.generate(input_list, sampling_params)
    generations = [generation.outputs[0].text for generation in generations]

    for i, example in tqdm(enumerate(eval_set)):
        if i < existing_len:
            continue
        example['response'] = generations[i-existing_len]
        example['generator'] = exp_name
        result_list.append(example)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)

else:
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to(device)
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    for i, example in tqdm(enumerate(eval_set)):
        if i < existing_len:
            continue
        instruction = template.format(example["instruction"], "", "")[:-1]     
        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        output_ids = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
        output_ids = output_ids[0][len(input_ids[0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        example['response'] = result # generated ones
        example['generator'] = model_name

        print(f"\nInput: \n{instruction}")
        print(f"\nOutput: \n{result}")
        print("="*100)
        result_list.append(example)
        with open(result_path, "w", encoding='utf-8') as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

print(f">> You are using template: {template}")
