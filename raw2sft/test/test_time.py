import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from vllm import LLM, SamplingParams
trust_remote_code = True
tensor_parallel_size = 1
dtype = "float16"

model_path = "/GPFS/data/ruiye-1/models/Wizard-Vicuna-7B-Uncensored"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )

prompt = "Tell me where is China and what is the capital of China."
prompts = []
for i in range(25):
    prompts.append(prompt)

# batch generation
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
print("batch generation time:", end_time - start_time)

# single generation
start_time = time.time()
for i in range(25):
    output = llm.generate(prompt, sampling_params)

end_time = time.time()
print("single generation time:", end_time - start_time)
