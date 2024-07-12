import os
from vllm import LLM, SamplingParams
from template import TEMPLATE_DICT
import pdb
import json
import argparse

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
parser.add_argument("--old_human_json_path", type=str, default="/GPFS/data/yuchifengting-1/OpenFedLLM/data/med/Wish-QA-MED_human_split.json", help="artificially crafted and split data, with datapoints already in the prompt removed")
parser.add_argument("--test", action="store_true")
parser.add_argument("--human_output_path", type=str, default="/GPFS/data/yuchifengting-1/OpenFedLLM/med/Wish-QA-MED_vicuna_human_split.json")
parser.add_argument("--gen_output_path", type=str, default="/GPFS/data/yuchifengting-1/OpenFedLLM/med/Wish-QA-MED_vicuna_gen_split.json")
parser.add_argument("--template", type=str, default="Wish_QA_MED")
args = parser.parse_args()

# ---------------Set up parameters------------------
trust_remote_code = True
tensor_parallel_size = 1
dtype = "float16"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

llm = LLM(
        model=args.model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )

# ---------------Deal with train data------------------
old_human_train_data = json.load(open(args.old_human_json_path))["train"]
test_data = json.load(open(args.old_human_json_path))["test"]

# You could find corresponding templates in template.py
TEMPLATE = TEMPLATE_DICT[args.template]
prompts = [TEMPLATE.format(sample['text']) for sample in old_human_train_data]
outputs = llm.generate(prompts, sampling_params)

human_train_data = []
gen_train_data = []

retry_prompts = []
retry_datapoints = []

# Process the initial outputs
for output, old_human_train_datapoint in zip(outputs, old_human_train_data):
    generated_text = output.outputs[0].text
    question, answer = extract_first_question_and_answer(generated_text)

    # If question or answer is None, add to retry lists
    if question is None or answer is None:
        retry_prompts.append(TEMPLATE.format(old_human_train_datapoint['text']))
        retry_datapoints.append(old_human_train_datapoint)
    else:
        # Create new human and generated training data points
        human_train_datapoint = old_human_train_datapoint
        gen_train_datapoint = {
            "text": old_human_train_datapoint["text"],
            "input": "",
            "instruction": question,
            "output": answer
        }
        human_train_data.append(human_train_datapoint)
        gen_train_data.append(gen_train_datapoint)

# Retry generating outputs up to 4 additional times if necessary
for i in range(4):
    if not retry_prompts:
        break  # Exit loop if there are no prompts to retry

    # Generate outputs for retry prompts
    retry_outputs = llm.generate(retry_prompts, sampling_params)
    
    # Reset retry lists for next iteration
    retry_prompts = []
    new_retry_datapoints = []

    # Process the retry outputs
    for output, old_human_train_datapoint in zip(retry_outputs, retry_datapoints):
        generated_text = output.outputs[0].text
        question, answer = extract_first_question_and_answer(generated_text)

        # If question or answer is None, add to retry lists again
        if question is None or answer is None:
            retry_prompts.append(TEMPLATE.format(old_human_train_datapoint['text']))
            new_retry_datapoints.append(old_human_train_datapoint)
        else:
            # Create new human and generated training data points
            human_train_datapoint = old_human_train_datapoint
            gen_train_datapoint = {
                "text": old_human_train_datapoint["text"],
                "input": "",
                "instruction": question,
                "output": answer
            }
            human_train_data.append(human_train_datapoint)
            gen_train_data.append(gen_train_datapoint)
    
    # Update retry datapoints for next iteration
    retry_datapoints = new_retry_datapoints

if args.test:
    exit()

# ---------------Deal with test data------------------
human_data = {"train": human_train_data, "test": test_data}
gen_data = {"train": gen_train_data, "test": test_data}

# Save the final data to a JSON file
os.makedirs(os.path.dirname(args.human_output_path), exist_ok=True)
with open(args.human_output_path, 'w', encoding='utf-8') as f:
    json.dump(human_data, f, ensure_ascii=False, indent=4)

os.makedirs(os.path.dirname(args.gen_output_path), exist_ok=True)
with open(args.gen_output_path, 'w', encoding='utf-8') as f:
    json.dump(gen_data, f, ensure_ascii=False, indent=4)

