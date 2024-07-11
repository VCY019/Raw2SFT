import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
import torch
import pdb

TEMPLATE = """
Instruction: Given the next [document], create only one [question] and [answer] pair that are grounded in the main point of the document. Do not include [document] in your response and do not add any additional information that is not in the document. The [question] is by an information-seeking user and the [answer] is provided by a helping AI Agent. Provide only the [question] and [answer]. 

[document]: Information for those who do not speak Lower Sorbian Lower Sorbian is a West Slavic language. It is spoken by about 14000 people in the BrandenburgianLower Lusatia.
### Response:
[question]: What is Lower Sorbian?
[answer]: Lower Sorbian is a West Slavic language spoken by about 14000 people in the Brandenburgian Lower Lusatia.

[document]: Organisation Cape Town International Convention Centre Cape Town International Convention Centre Host venue Host venue Location Cape Town, South Africa Constructed 2003 Capacity Varies The host city and venue was announced by the International Netball Federation (INF) on 8 March 2019, only months prior to the staging of the 2019 edition in Liverpool, England. Cape Town's bid, supported by the South African Government and the Western Cape province, was selected by the INF ahead of a bid by Auckland, New Zealand. The INF stated the Cape Town bid would "deliver a greater impact on the development of global netball" and cited the pledges by the South African Government to invest heavily in preparation and development of the sport in the lead-up to the tournament.
### Response:
[question]: Where is the 2023 Netball World Cup being held?
[answer]: The 2023 Netball World Cup is being held in Cape Town, South Africa.

[document]: Squads The sixteen competing nations selected 12-player squads for the tournament, with three additional reserve players named. Reserve players would be permanent replacements in the event of injury.
### Response:
[question]: How many players are on a netball team?
[answer]: A netball team consists of 12 players, with three additional reserve players named.

[document]: {}
### Response:\n"""


# base_model_name = "/GPFS/data/ruiye/models/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"

# adapter_model_name = "/GPFS/rhome/ruige/OpenFedLLMBenchmark/output1/medical-dialogue_20000_fedavg_c20s2_i10_b16a1_l512_r32a64_20240220141121/checkpoint-200"
base_model_name = "/GPFS/data/xhpang-1/LLM/alpaca_recovered"
adapter_model_name = "/GPFS/data/yuchifengting-1/OpenFedLLM/output/Wish-QA-Falcon-generated/Wish-QA-Falcon-generated_20000_fedavg_c10s2_i10_b16a1_l512_r32a64_20240703160950/checkpoint-200"
# adapter_model_name = "/GPFS/rhome/ruige/OpenFedLLMBenchmark/dpo_model/hh-rlhf_20000_fedavg_c5s2_i10_b16a1_l512_r8a16_20240222072647/checkpoint-200"
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)
model = PeftModel.from_pretrained(model, adapter_model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

data_list = []
# loaded_data = json.load(open("test.json"))
# for data in loaded_data:
#     data_list.append(data['text'])


sample_list = [
    "Holidays\nThe most notable holiday associated with patriotism in Cameroon is National Day, also called Unity Day. Among the most notable religious holidays are Assumption Day, and Ascension Day, which is typically 39 days after Easter. In the Northwest and Southwest provinces, collectively called Ambazonia, October 1 is considered a national holiday, a date Ambazonians consider the day of their independence from Cameroon.",
    "In dance and ballet\nIn 2011, the world premiere of Boris Eifman's new ballet Rodin took place in St Petersburg, Russia. The ballet is dedicated to the life and creative work of sculptor Auguste Rodin and his apprentice, lover and muse, Camille Claudel.\nIn 2014, the Columbus Dance Theatre and the Carpe Diem String Quartet performed the premiere of Claudel in Columbus, Ohio, with music by Korine Fujiwara, original poetry by Kathleen Kirk, and choreography by Tim Veach.",
    "Predators\nSeveral bird species prey on Canada jays, including great grey owls (Strix nebulosa), northern hawk-owls (Surnia ulula), and Mexican spotted owls (Strix occidentalis lucida). Canada jay remains have been recovered from the lairs of fisher (Pekania pennanti) and American marten (Martes americana).Red squirrels (Tamiasciurus hudsonicus) eat Canada jay eggs. Canada jays alert each other to threats by whistling alarm notes, screaming, chattering, or imitating and/or mobbing predators."
]

for sample in sample_list:
    formated_input = TEMPLATE.format(sample)
    inputs = tokenizer.encode(formated_input, return_tensors="pt").to(device)
    outputs = model.generate(inputs=inputs, max_new_tokens=2048, do_sample=True, top_p=1.0, temperature=0.7)
    #outputs = model.generate(inputs=inputs, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0]))
    print("="*100)