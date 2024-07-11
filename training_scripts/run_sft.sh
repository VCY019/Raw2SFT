max_steps=10
num_rounds=100
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=5
sample_clients=2
lora_r=16
lora_alpha=32   # twice of lora_r
lr=2e-5

local_data_dir="/GPFS/data/yuchifengting-1/OpenFedLLM/data/math_ruige/" # note that there should be a / at the end
dataset_name="postprocessed_generate_math"
dataset_filename="$dataset_name.json"
dataset_sample=20000
# model_name_or_path="meta-llama/Llama-2-7b-hf"
# model_name_or_path="/GPFS/data/xhpang-1/LLM/alpaca_recovered"
model_name_or_path="/GPFS/data/ruiye-1/models/Wizard-Vicuna-7B-Uncensored"
template="vicuna"
output_dir="./output/$dataset_name"

gpu=1
fed_alg="fedavg"

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --local_data_dir $local_data_dir \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_filename \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template $template 