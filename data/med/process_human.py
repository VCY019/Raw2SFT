import json

json_path = "Wish-QA-MED_human_split.json"
output_path = "Wish-QA-MED_human_split1.json"

file = json.load(open(json_path))
train_data = []
test_data = []
for sample in file["train"]:
    datapoint = {
        "text": sample["text"],
        "input": "",
        "instruction": sample["title_question"],
        "output": sample["long_answer"]
    }
    train_data.append(datapoint)
for sample in file["test"]:
    datapoint = {
        "text": sample["text"],
        "input": "",
        "instruction": sample["title_question"],
        "output": sample["long_answer"]
    }
    test_data.append(datapoint)

data = {"train": train_data, "test": test_data}
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)