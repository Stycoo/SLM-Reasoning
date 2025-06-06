import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets
from pathlib import Path
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file_1", type=str, help="Path to the output generation file")
parser.add_argument("--generation_file_2", type=str, help="Path to the output generation file")
parser.add_argument("--question_file", type=str)
parser.add_argument("--reward_model", type=str, help="Path to reward model")

args = parser.parse_args()

question_file = args.question_file
questions = defaultdict(str)
with open(question_file, 'r') as f:
    for line in f:
        json_line = json.loads(line)  # Parse each line as a JSON object
        question_id = json_line['question_id']
        question = json_line['turns'][0]['content']
        questions[question_id] = question

model_1_name = ''
generation_file_1 = args.generation_file_1
output_data_1 = defaultdict(str)
with open(generation_file_1, 'r') as f:
    for line in f:
        json_line = json.loads(line)  # Parse each line as a JSON object
        model_1_name = json_line['model_id']
        question_id = json_line['question_id']
        output = json_line['choices'][0]['turns'][0]['content']
        output_data_1[question_id] = output

model_2_name = ''
generation_file_2 = args.generation_file_2
output_data_2 = defaultdict(str)
with open(generation_file_2, 'r') as f:
    for line in f:
        json_line = json.loads(line)  # Parse each line as a JSON object
        model_2_name = json_line['model_id']
        question_id = json_line['question_id']
        output = json_line['choices'][0]['turns'][0]['content']
        output_data_2[question_id] = output

model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                           device_map="cuda", 
                                                           trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

win_rate = []
for question_id in tqdm.tqdm(questions):
    prompt = questions[question_id]
    output_1 = output_data_1[question_id]
    output_2 = output_data_2[question_id]

    scores = []
    for output in [output_1, output_2]:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": output}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(input_ids)
            # score = output.score.float().item()
            score = output.logits.float().item()
            scores.append(score)
    
    if scores[0] > scores[1]:
        win_rate.append(1)
    else:
        win_rate.append(0)

print(f"{model_1_name} vs {model_2_name} winrate: {sum(win_rate)/len(win_rate)}")

local_eval_res = {f"{model_1_name} vs {model_2_name} winrate": sum(win_rate)/len(win_rate)}
local_eval_res_file = Path(args.generation_file_1).parent / 'rm_eval_res.json'
with open(local_eval_res_file, 'w') as f:
    f.write(json.dumps(local_eval_res) + '\n')