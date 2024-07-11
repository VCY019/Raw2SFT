import json

global_list = []
with open("/GPFS/data/ruige-1/Raw_to_SFT/dataset/math/AQUA_RAT_yes_opt_shard_00000.jsonl",'r',encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        new_data = {
            "context": data["context"],
            "instruction": data["QA_list"][0]["Q"],
            "response": data["QA_list"][0]["A"]
        }
        global_list.append(new_data)

with open("global_data.json",'w',encoding='utf-8') as output:
    json.dump(global_list,output,indent=4,ensure_ascii=False)
