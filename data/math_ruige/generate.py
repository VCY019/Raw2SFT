from vllm import LLM, SamplingParams
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_path = "/GPFS/data/ruiye-1/models/Wizard-Vicuna-7B-Uncensored"
trust_remote_code = True
tensor_parallel_size = 1
dtype = "float16"

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: """

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: """

few_shot_template = """Given the next [document], create a [question] and [answer] pair that are grounded in the main point of the document, don't add any additional information that is not in the document. The [question] is by an information-seeking user and the [answer] is provided by a helping AI Agent.

[document]: The first five numbers in a regular sequence are 4, 10, X, 46, and 94. What is x ?\n\nit is a good Q to learn to pick up number properties in the given set, even if we do not get a Q on these line in actuals..\nwhen we see the sequence 4, 10, x, 46, and 94, we see that each succeeding number is some value more than twice the previous number..\n10=2*4+2..\nx=2*10+2 = 22..\nand so on..\nso next number= 22.

### Response:
[question]: The first five numbers in a regular sequence are 4, 10, X, 46, and 94. What is x ?\nLet's think step by step.
[answer]: it is a good Q to learn to pick up number properties in the given set, even if we do not get a Q on these line in actuals..\nwhen we see the sequence 4, 10, x, 46, and 94, we see that each succeeding number is some value more than twice the previous number..\n10=2*4+2..\nx=2*10+2 = 22..\nand so on..\nso next number= 22\nTherefore, the answer is 22.

[document]: A school currently maintains a fixed number of students per class. If the ratio of students per class were to be increased by 1, 10 fewer classes would be run for a total of 120 students. What is the current ratio Q of students per class?\n\nAnother way to look at the problem...\nSince the total is 120, RATIO * CLASSES = R*C = 120.....(i)\nwe are looking where ratio increases by 1 and # of classes decreases by 10 = (R+1)(C-10) = RC+C-10R-10=120....(ii)\n(ii)-(i)....\nC=10R+10 = 10(R+1).......\nso # of classes has to be multiple of 10\nAS RC=120.... 10(R+1)*R = 120...................R(R+1) = 12..\nso 12 is a multiple of consecutive numbers ONLY 3 *4 fits in..... and R=3.

### Response:
[question]: A school currently maintains a fixed number of students per class. If the ratio of students per class were to be increased by 1, 10 fewer classes would be run for a total of 120 students. What is the current ratio Q of students per class?\nLet's think step by step.
[answer]: Another way to look at the problem...\nSince the total is 120, RATIO * CLASSES = R*C = 120.....(i)\nwe are looking where ratio increases by 1 and # of classes decreases by 10 = (R+1)(C-10) = RC+C-10R-10=120....(ii)\n(ii)-(i)....\nC=10R+10 = 10(R+1).......\nso # of classes has to be multiple of 10\nAS RC=120.... 10(R+1)*R = 120...................R(R+1) = 12..\nso 12 is a multiple of consecutive numbers ONLY 3 *4 fits in..... and R=3\nTherefore, the answer is Q=3.

[document]: |x+3| – |4-x| = |8+x| How many S solutions will this equation have?\n\n|x| = x when x >= 0 (x is either positive or 0)\n|x| = -x when x < 0 (note here that you can put the equal to sign here as well x <= 0 because if x = 0,\n|0| = 0 = -0 (all are the same)\nSo the '=' sign can be put with x > 0 or with x < 0. We usually put it with 'x > 0' for consistency.

### Response:
[question]: |x+3| – |4-x| = |8+x| How many S solutions will this equation have?\nLet's think step by step.
[answer]: |x| = x when x >= 0 (x is either positive or 0)\n|x| = -x when x < 0 (note here that you can put the equal to sign here as well x <= 0 because if x = 0,\n|0| = 0 = -0 (all are the same)\nSo the '=' sign can be put with x > 0 or with x < 0. We usually put it with 'x > 0' for consistency.\nTherefore, the answer is 0.

[document]: {}

### Response:

"""

with open('global_data.json','r',encoding='utf-8') as f:
    data = json.load(f)

sample_list = [example["context"] for example in data]


prompts = [few_shot_template.format(sample) for sample in sample_list]
# prompts = [template.format(content_template.format(sample)) for sample in sample_list]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )

outputs = llm.generate(prompts, sampling_params)

answers = []
# Print the outputs.
for i,output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    answer_dict = {
        "prompt":prompt,
        "answer":generated_text
    }
    answers.append(answer_dict)

fileName = "generate_math.json"
with open(fileName,'w',encoding='utf-8') as output:
    json.dump(answers,output,indent=4,ensure_ascii=False)