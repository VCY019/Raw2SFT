import json
import os

file_path = "/GPFS/data/yuchifengting-1/Raw2SFT/data/raw/Wish-QA-MED_split.json"
file = json.load(open(file_path))["train"]
text_list = []
for sample in file:
    text_list.append(sample["text"]+sample["title_question"]+sample["long_answer"])

# find two shortst text in file
two_shortest_elements = sorted(text_list, key=len)[:2]
print(two_shortest_elements)

## outputs:

