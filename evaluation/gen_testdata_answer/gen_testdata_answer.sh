gpus=1
base_model_path="/GPFS/data/xhpang-1/LLM/alpaca_recovered"
checkpoint=100
judge_model=gpt-4-1106-preview
model_name=Wish-QA-MED-Falcon-generated_20000_fedavg_c5s2_i10_b16a1_l512_r16a32_20240706150545
model_dir="/GPFS/data/yuchifengting-1/OpenFedLLM/output/Wish-QA-MED-Falcon-generated"

dataset_path="/GPFS/data/yuchifengting-1/OpenFedLLM/data/train/Wish-QA-MED-Falcon.json"

lora_path=$model_dir/$model_name/checkpoint-$checkpoint
merged_lora_path=$model_dir/$model_name/full-$checkpoint
model_list=${model_name}_$checkpoint

# merge lora if necessary
if [ -d "$merged_lora_path" ]; then
    echo "Lora is already merged."
else
    echo "Lora is not merged"
    CUDA_VISIBLE_DEVICES=$gpus python ../../utils/merge_lora.py --lora_path $lora_path --base_model_path $base_model_path
fi

CUDA_VISIBLE_DEVICES=$gpus python run_gen_testdata_answer.py --base_model_path $merged_lora_path --dataset_path $dataset_path --use_vllm
