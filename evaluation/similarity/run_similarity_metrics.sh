export https_proxy=http://db93:5000
export http_proxy=http://db93:5000

gpus=1
dataset_paths=(
"/GPFS/data/ruige-1/Raw_to_SFT/answers/math/Wizard-Vicuna-7B-Uncensored.json"
"/GPFS/data/ruige-1/Raw_to_SFT/answers/math/postprocessed_generate_math.json_20000_fedavg_c5s2_i10_b16a1_l512_r16a32_20240710055339_100.json"
"/GPFS/data/ruige-1/Raw_to_SFT/answers/math/postprocessed_global_data.json_20000_fedavg_c5s2_i10_b16a1_l512_r16a32_20240710055306_100.json"
)

gold_answer_path="/GPFS/data/ruige-1/Raw_to_SFT/dataset/math/test.json"

export CUDA_VISIBLE_DEVICES=$gpus
for dataset_path in "${dataset_paths[@]}"
do
    python similarity_metrics.py --dataset_path $dataset_path --gold_answer_path $gold_answer_path
done