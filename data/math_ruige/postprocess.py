import json
import os
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

def remove_options(a):
    parts = a.split("\nOptions:", 1)
    text = parts[0] + "\nLet's think step by step."
    return text

gen_json_path = "generate_math.json"
global_json_path = "global_data.json"
gen_output_path = "postprocessed_generate_math.json"
global_output_path = "postprocessed_global_data.json"

gen_json = json.load(open(gen_json_path))
global_json = json.load(open(global_json_path))

gen_data = []
global_data = []
for gen_sample, global_sample in zip(gen_json, global_json):
    generated_text = gen_sample["answer"]
    question, answer = extract_first_question_and_answer(generated_text)
    if question is None or answer is None:
        continue
    gen_datapoint = {}
    gen_datapoint["input"] = ""
    gen_datapoint["instruction"] = question
    gen_datapoint["output"] = answer
    gen_datapoint["context"] = global_sample["context"]
    gen_data.append(gen_datapoint)
    
    global_sample["input"] = ""
    global_sample["instruction"] = remove_options(global_sample["instruction"])
    global_sample["output"] = global_sample["response"]
    del global_sample["response"]
    global_data.append(global_sample)


# split the data into train and test datasets   
gen_data = {"train": gen_data[:-100], "test": gen_data[-100:]}
global_data = {"train": global_data[:-100], "test": global_data[-100:]}

# Save the final data to a JSON file
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(gen_output_path, 'w', encoding='utf-8') as f:
    json.dump(gen_data, f, ensure_ascii=False, indent=4)

with open(global_output_path, 'w', encoding='utf-8') as f:
    json.dump(global_data, f, ensure_ascii=False, indent=4)