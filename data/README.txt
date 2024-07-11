Data Removal:
    Data in "raw" folder: Datapoints used in prompt are not removed.
    Otherwise is removed.

Generate data:
    Use human_split data, whose datapoints already used in prompt are removed.

Naming:
    old_human_json_path: "artificially crafted and split data, with datapoints already in the prompt removed"

eg:
    old_human_json_path="/GPFS/data/yuchifengting-1/OpenFedLLM/data/med/Wish-QA-MED_human_split.json"
    human_output_path="/GPFS/data/yuchifengting-1/OpenFedLLM/data/med/Wish-QA-MED_vicuna_human_split.json"
    gen_output_path="/GPFS/data/yuchifengting-1/OpenFedLLM/data/med/Wish-QA-MED_vicuna_gen_split.json"
