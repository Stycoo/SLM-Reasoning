
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

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file_1", type=str, help="Path to the output generation file")
parser.add_argument("--generation_file_2", type=str, help="Path to the output generation file")
parser.add_argument("--reward_model", type=str, help="Path to reward model")

args = parser.parse_args()

generation_file_1 = args.generation_file_1
with open(generation_file_1, 'r') as f:
    output_data_1 = json.load(f)

folder_path = os.path.dirname(generation_file_1)
model_1_name = os.path.basename(folder_path)

generation_file_2 = args.generation_file_2
with open(generation_file_2, 'r') as f:
    output_data_2 = json.load(f)
    
folder_path = os.path.dirname(generation_file_2)
model_2_name = os.path.basename(folder_path)

model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                           device_map="cuda", 
                                                           trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

win_rate = []
for ind in tqdm.tqdm(range(len(output_data_1))):
    prompt = output_data_1[ind]["instruction"]
    output_1 = output_data_1[ind]["output"]
    output_2 = output_data_2[ind]["output"]

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