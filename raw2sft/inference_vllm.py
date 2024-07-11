import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from vllm import LLM, SamplingParams
from template import TEMPLATE_DICT
import pdb
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_first_question_and_answer(text):
    question_start = text.find("[question]:")
    if question_start == -1:
        return None, None
    
    answer_start = text.find("[answer]:", question_start)
    if answer_start == -1:
        return None, None
    
    document_start = text.find("[document]:", answer_start)
    
    question = text[question_start + len("[question]:"):answer_start].strip()
    
    if document_start == -1:
        answer = text[answer_start + len("[answer]:"):].strip()
    else:
        answer = text[answer_start + len("[answer]:"):document_start].strip()
    
    return question, answer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/GPFS/data/xhpang-1/LLM/alpaca_recovered")
parser.add_argument("--json_path", type=str, default="/GPFS/data/yuchifengting-1/OpenFedLLM/data/med/WISH-QA-MED_human_split.json", help="artificially crafted and split data, with datapoints already in the prompt removed")
parser.add_argument("--test", action="store_true")
parser.add_argument("--output_path", type=str, default="/GPFS/data/yuchifengting-1/OpenFedLLM/data/Wish-QA-MED-Falcon-generated/Wish-QA-MED-Falcon-generated.json")
parser.add_argument("--template", type=str, default="Wish_QA_MED_TEMPLATE")
args = parser.parse_args()

trust_remote_code = True
tensor_parallel_size = 1
dtype = "float16"
# You could find corresponding templates in template.py
TEMPLATE = TEMPLATE_DICT[args.template]

loaded_json = json.load(open(args.json_path))
prompts = [TEMPLATE.format(sample['text']) for sample in loaded_json['train']]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

llm = LLM(
        model=args.model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )

outputs = llm.generate(prompts, sampling_params)

data = []
# Some generated samples may not contain the question and answer, ensure the corresponding QA pairs in human data are removed.
for sample, output in zip(loaded_json, outputs):
    generated_text = output.outputs[0].text
    question, answer = extract_first_question_and_answer(generated_text)
    if args.test:
        print("[Generated_text]:")
        print(generated_text)
        print("[Text]:")
        print(sample["text"])
        print("[Generated Question]:")
        print(question)
        print("[Generated Answer]:")
        print(answer)
        print("=" * 100)
        continue
    if question is None or answer is None:
        continue
    datapoint = {}
    datapoint["text"] = sample["text"]
    datapoint["input"] = ""
    datapoint["instruction"] = question
    datapoint["output"] = answer
    data.append(datapoint)

if args.test:
    exit()
# split the data into train and test datasets
train_data = data[:-100]
test_data = data[-100:]
data = {"train": train_data, "test": test_data}

# Save the final data to a JSON file
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

