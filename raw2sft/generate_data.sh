export TZ=Asia/Shanghai
START_TIME=`date +%Y%m%d_%H%M%S` # 20210831_235959
log_dir="/GPFS/data/yuchifengting-1/Raw2SFT/logs/gen_data"
log_name="Wish_QA_MED_vicuna_$START_TIME.log"
if [ ! -d "$log_dir" ]; then
    mkdir -p $log_dir
    echo "log_dir created"
else
    echo "log_dir exists"
fi

# model_path="/GPFS/data/xhpang-1/LLM/alpaca_recovered"
model_path="/GPFS/data/ruiye-1/models/Wizard-Vicuna-7B-Uncensored"

# old_human_json_path: artificially crafted and split data, with datapoints already in the prompt removed
old_human_json_path="/GPFS/data/yuchifengting-1/Raw2SFT/data/med/Wish-QA-MED_human_split.json"

human_output_path="/GPFS/data/yuchifengting-1/Raw2SFT/data/med/Wish-QA-MED_vicuna_human_split.json"
gen_output_path="/GPFS/data/yuchifengting-1/Raw2SFT/data/med/Wish-QA-MED_vicuna_gen_split.json"

template="Wish_QA_MED"
gpu=5

CUDA_VISIBLE_DEVICES=$gpu python -u generate_data.py \
  --model_path $model_path \
  --old_human_json_path $old_human_json_path \
  --human_output_path $human_output_path \
  --gen_output_path $gen_output_path \
  --template $template > $log_dir/$log_name 2>&1
#  --test