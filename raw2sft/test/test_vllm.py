from vllm import LLM, SamplingParams

model_path = "/GPFS/data/xhpang-1/LLM/alpaca_recovered"
trust_remote_code = True
tensor_parallel_size = 1
dtype = "float16"

TEMPLATE = """
Instruction:  Given the next [document], create a [question] and [answer] pair that are grounded in the main point of the document, don't add any additional information that is not in the document. The [question] is by an information-seeking user and the [answer] is provided by a helping AI Agent.
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

sample_list = [
    "Holidays\nThe most notable holiday associated with patriotism in Cameroon is National Day, also called Unity Day. Among the most notable religious holidays are Assumption Day, and Ascension Day, which is typically 39 days after Easter. In the Northwest and Southwest provinces, collectively called Ambazonia, October 1 is considered a national holiday, a date Ambazonians consider the day of their independence from Cameroon.",
    "In dance and ballet\nIn 2011, the world premiere of Boris Eifman's new ballet Rodin took place in St Petersburg, Russia. The ballet is dedicated to the life and creative work of sculptor Auguste Rodin and his apprentice, lover and muse, Camille Claudel.\nIn 2014, the Columbus Dance Theatre and the Carpe Diem String Quartet performed the premiere of Claudel in Columbus, Ohio, with music by Korine Fujiwara, original poetry by Kathleen Kirk, and choreography by Tim Veach.",
    "Predators\nSeveral bird species prey on Canada jays, including great grey owls (Strix nebulosa), northern hawk-owls (Surnia ulula), and Mexican spotted owls (Strix occidentalis lucida). Canada jay remains have been recovered from the lairs of fisher (Pekania pennanti) and American marten (Martes americana).Red squirrels (Tamiasciurus hudsonicus) eat Canada jay eggs. Canada jays alert each other to threats by whistling alarm notes, screaming, chattering, or imitating and/or mobbing predators."
]

prompts = [TEMPLATE.format(sample) for sample in sample_list]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")