from evaluate import load
from bert_score import score

import argparse
import pdb
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--gold_answer_path", type=str)
parser.add_argument("--lang", type=str, default="en")
args = parser.parse_args()

# Used in datasets with artificially crafted QA pairs.
bertscore = load("bertscore")
rouge = load("rouge")
eval_set = datasets.load_dataset("json", data_files=args.dataset_path)['train']
gold_answers = datasets.load_dataset("json", data_files=args.gold_answer_path)['train']
predictions = eval_set["answer"]
references = gold_answers["response"]
results_bertscore = bertscore.compute(predictions=predictions, references=references, lang="en", rescale_with_baseline=True)
result_rouge = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"], use_aggregator=True)

bertscore_f1 = sum(results_bertscore["f1"]) / len(results_bertscore["f1"])
rougeL = result_rouge["rougeL"]
print()
print(f"dataset_path: {args.dataset_path}, lang: {args.lang}")
print(f"bertscore_f1: {bertscore_f1}, rougeL: {rougeL}")
print("="*100)
