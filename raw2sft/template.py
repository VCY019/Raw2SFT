Wish_QA_Falcon_TEMPLATE = """
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

# The current med template with 2 examples
Wish_QA_MED_TEMPLATE = """
Instruction:  Given the next [document], create a [question] and [answer] pair that are grounded in the main point of the document, don't add any additional information that is not in the document. The [question] is by an information-seeking user and the [answer] is provided by a helping AI Agent.

[document]: Background: Preterm infants have reduced liver stores of vitamin A at birth compared to term born infants. Current guidelines recommend an intake of 700-1500 IU/kg/day vitamin A, and there is evidence to support higher doses for infants with significant lung disease. The importance of appropriate early nutrition for preterm infants is increasingly becoming apparent.\n\nObjectives: To examine whether preterm infants admitted to a UK neonatal unit received the currently recommended amounts of vitamin A during the first 4 weeks of life, and to identify the feeding patterns associated with optimal vitamin A intake.\n\nDesign: Retrospective case note review.\n\nSetting: UK tertiary neonatal unit.\n\nPopulation: 36 preterm infants with a median gestation of 30 weeks (range 26-33 weeks) and median birth weight of 1305 g (range 880-1800 g).\n\nMain outcome measure: Mean daily total intake of vitamin A in each of the first 4 weeks of life.\n\nResults: Despite a policy of introducing intravenous lipid supplemented with fat soluble vitamins on day 2, only four infants (11%) consistently met the recommended daily intake of vitamin A during the first 2 weeks of life. Adequacy of vitamin A intake in the first 2 weeks of life was not predicted by gestation or pattern of feeding.

### Response:
[question]: Vitamin A provision for preterm infants: are we meeting current guidelines?
[answer]: Recommendations for intravenous vitamin A supplementation in parenterally fed preterm infants require revision. Intravenous lipid with added fat soluble vitamins should be started as soon as possible after birth, and consideration given to early oral vitamin A supplementation in those infants tolerant of enteral feeds.

[document]: Objectives: The objective was to determine if hospital admission of children with blunt abdominal trauma for observation of possible intraabdominal injury (IAI) is necessary after a normal abdominal computed tomography (CT) scan in the emergency department (ED).\n\nMethods: The authors conducted a prospective observational cohort study of children less than 18 years of age with blunt abdominal trauma who underwent an abdominal CT scan in the ED. Abdominal CT scans were obtained with intravenous contrast but no oral contrast. The decision to hospitalize the patient was made by the attending emergency physician (EP) with the trauma or pediatric surgery teams. An abnormal abdominal CT scan was defined by the presence of any visualized IAI or findings suggestive of possible IAI (e.g., intraperitoneal fluid without solid organ injury). Patients were followed to determine if IAI was later diagnosed and the need for acute therapeutic intervention if IAI was present.\n\nResults: A total of 1,295 patients underwent abdominal CT, and 1,085 (84%) patients had normal abdominal CT scans in the ED and make up the study population. Seven-hundred thirty-seven (68%) were hospitalized, and 348 were discharged to home. None of the 348 patients discharged home and 2 of the 737 hospitalized patients were identified with an IAI after a normal initial abdominal CT. The IAIs in patients with normal initial CT scans included a 10-year-old with a mesenteric hematoma and serosal tear at laparotomy and a 10-year-old with a perinephric hematoma on repeat CT. Neither underwent specific therapy. The negative predictive value (NPV) of a normal abdominal CT scan for IAI was 99.8% (95% confidence interval [CI] = 99.3% to 100%).

### Response:
[question]: Is hospital admission and observation required after a normal abdominal computed tomography scan in children with blunt abdominal trauma?
[answer]: Children with blunt abdominal trauma and a normal abdominal CT scan in the ED are at very low risk of having a subsequently diagnosed IAI and are very unlikely to require a therapeutic intervention. Hospitalization of children for evaluation of possible undiagnosed IAI after a normal abdominal CT scan has a low yield and is generally unnecessary.

[document]: {}

### Response:\n"""

# The previous med template with 3 examples
Wish_QA_MED_Falcon_TEMPLATE = """
Instruction:  Given the next [document], create a [question] and [answer] pair that are grounded in the main point of the document, don't add any additional information that is not in the document. The [question] is by an information-seeking user and the [answer] is provided by a helping AI Agent.

[document]: Unlabelled: The Institute of Medicine (IOM) has recommended instituting clinical education reforms to ensure all health profession graduates acquire five core competencies; providing patient-centered care, working in interdisciplinary teams, employing evidence-based practice, applying quality improvement and utilizing informatics. The IOM has identified 28 specific skills associated with these competencies. This qualitative, exploratory study was conducted to begin to examine the extent to which physical therapy clinical instructors provide students with instruction the students perceived as promoting the acquisition of these skills.\n\nMethods: Two groups of physical therapy students enrolled in a 3-year DPT program (7 on a first-year clinical education experience and 17 on a final year experience) maintained journals describing the types of learning activities used by clinical instructors to promote the acquisition of the competencies. The authors employed NVivo qualitative data analysis software to code the journal entries using 28 codes derived from the skills associated with the five core competencies.\n\nResults: Of the 327 coded learning activities, just over 50% were related to skills associated with providing patient-centered care (21.4%) and working in interdisciplinary teams (30.0%). The remaining 49.6% of the learning activities cited by students were related to skills associated with employing evidence-based practice (18.3%), applying quality improvement (16.5%) and utilizing informatics (13.8%).\n\nDiscussion: Based on student perceptions, physical therapy clinical instructors are providing learning activities that allow students to acquire skills associated with all five of the IOM competencies. However, students reported the least emphasis on instruction pertaining to the competencies of applying quality improvement and utilizing informatics.

### Response:
[question]: Are physical therapy clinical instructors teaching the Institute of Medicine core competencies?
[answer]: This study supports the need for the profession of physical therapy to delineate formal and explicit clinical education instructional content and strategies to promote students' acquisition of the IOM core competency skills.

[document]: Background: Preterm infants have reduced liver stores of vitamin A at birth compared to term born infants. Current guidelines recommend an intake of 700-1500 IU/kg/day vitamin A, and there is evidence to support higher doses for infants with significant lung disease. The importance of appropriate early nutrition for preterm infants is increasingly becoming apparent.\n\nObjectives: To examine whether preterm infants admitted to a UK neonatal unit received the currently recommended amounts of vitamin A during the first 4 weeks of life, and to identify the feeding patterns associated with optimal vitamin A intake.\n\nDesign: Retrospective case note review.\n\nSetting: UK tertiary neonatal unit.\n\nPopulation: 36 preterm infants with a median gestation of 30 weeks (range 26-33 weeks) and median birth weight of 1305 g (range 880-1800 g).\n\nMain outcome measure: Mean daily total intake of vitamin A in each of the first 4 weeks of life.\n\nResults: Despite a policy of introducing intravenous lipid supplemented with fat soluble vitamins on day 2, only four infants (11%) consistently met the recommended daily intake of vitamin A during the first 2 weeks of life. Adequacy of vitamin A intake in the first 2 weeks of life was not predicted by gestation or pattern of feeding.

### Response:
[question]: Vitamin A provision for preterm infants: are we meeting current guidelines?
[answer]: Recommendations for intravenous vitamin A supplementation in parenterally fed preterm infants require revision. Intravenous lipid with added fat soluble vitamins should be started as soon as possible after birth, and consideration given to early oral vitamin A supplementation in those infants tolerant of enteral feeds.

[document]: Objectives: The objective was to determine if hospital admission of children with blunt abdominal trauma for observation of possible intraabdominal injury (IAI) is necessary after a normal abdominal computed tomography (CT) scan in the emergency department (ED).\n\nMethods: The authors conducted a prospective observational cohort study of children less than 18 years of age with blunt abdominal trauma who underwent an abdominal CT scan in the ED. Abdominal CT scans were obtained with intravenous contrast but no oral contrast. The decision to hospitalize the patient was made by the attending emergency physician (EP) with the trauma or pediatric surgery teams. An abnormal abdominal CT scan was defined by the presence of any visualized IAI or findings suggestive of possible IAI (e.g., intraperitoneal fluid without solid organ injury). Patients were followed to determine if IAI was later diagnosed and the need for acute therapeutic intervention if IAI was present.\n\nResults: A total of 1,295 patients underwent abdominal CT, and 1,085 (84%) patients had normal abdominal CT scans in the ED and make up the study population. Seven-hundred thirty-seven (68%) were hospitalized, and 348 were discharged to home. None of the 348 patients discharged home and 2 of the 737 hospitalized patients were identified with an IAI after a normal initial abdominal CT. The IAIs in patients with normal initial CT scans included a 10-year-old with a mesenteric hematoma and serosal tear at laparotomy and a 10-year-old with a perinephric hematoma on repeat CT. Neither underwent specific therapy. The negative predictive value (NPV) of a normal abdominal CT scan for IAI was 99.8% (95% confidence interval [CI] = 99.3% to 100%).

### Response:
[question]: Is hospital admission and observation required after a normal abdominal computed tomography scan in children with blunt abdominal trauma?
[answer]: Children with blunt abdominal trauma and a normal abdominal CT scan in the ED are at very low risk of having a subsequently diagnosed IAI and are very unlikely to require a therapeutic intervention. Hospitalization of children for evaluation of possible undiagnosed IAI after a normal abdominal CT scan has a low yield and is generally unnecessary.

[document]: {}

### Response:\n"""

TEMPLATE_DICT = {
    'Wish_QA_MED_Falcon': Wish_QA_MED_Falcon_TEMPLATE,
    'Wish_QA_MED': Wish_QA_MED_TEMPLATE, 
    'Wish_QA_Falcon': Wish_QA_Falcon_TEMPLATE
}