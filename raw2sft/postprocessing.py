import json

# remove null elements in json

json_path = "/GPFS/data/yuchifengting-1/OpenFedLLM/data/Wish-QA-Falcon-generated/Wish-QA-Falcon-generated.json"
loaded_json = json.load(open(json_path))
data = loaded_json
# for sample in loaded_json:
#     sample["input"] = ""
#     sample["output"] = sample["response"]
#     # delete "question" and “answer"
#     del sample["response"]
#     data.append(sample)
train_data = data[:int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]
data = {"train": train_data, "test": test_data}

with open('Wish-QA-Falcon-generated.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


