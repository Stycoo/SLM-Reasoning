
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import json
import os
import argparse
import tqdm
import numpy as np
import datasets
from pathlib import Path
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default='', help="Path to the output generation file")
parser.add_argument("--model_name_or_path", type=str, default='', help="Path to reward model")
parser.add_argument("--output_dir", type=str, default='', help="Path to output directory")
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)

args = parser.parse_args()

with open(args.input_file, 'r') as f:
    input_data = json.load(f)
    input_data_size = len(input_data)

# input_data = input_data[0:100]

chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, 
                                                        device_map="cuda", 
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True) # use_fast=True

# data: {'prompt': str, 'best_response':str, 'best_response_score':float, 'completion_inputs': [], 'completion_outputs':[[],...], 
# 'target_model_responses':[], 'target_model_response_scores':[]}

for data in tqdm.tqdm(input_data):
    prompt = data["prompt"]
    completion_inputs = data["completion_inputs"]
    completion_outputs = data["completion_outputs"]
    target_model_response_scores = data['target_model_response_scores']

    scores = []
    for i, completion_input in enumerate(completion_inputs):
        _scores = []
        for j, completion_output in enumerate(completion_outputs[i]):
            response_with_completion = completion_input.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1] + completion_output
            messages = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_with_completion}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

            with torch.no_grad():
                output = model(input_ids)
                score = output.logits.float().item()
                _scores.append(score)

        scores.append(_scores)
        if max(_scores) > max(target_model_response_scores) and len(set(completion_outputs[i])) > 1:
            break

    data["completion_rm_scores"] = scores

file_name = f"chunk_{args.chunk_id}.json"
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(os.path.join(output_dir, file_name), 'w') as f:
    json.dump(input_data, f, indent=4)

# data: {'prompt': str, 'best_response':str, 'best_response_score':float, 'completion_inputs': [], 'completion_outputs':[[],...], 'completion_rm_scores':[[],...],
# 'target_model_responses':[], 'target_model_response_scores':[]}