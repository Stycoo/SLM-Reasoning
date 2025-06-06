import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import json
import os
import argparse
import tqdm
from pathlib import Path

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

chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, 
                                                        device_map="cuda", 
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True) # use_fast=True

for data in tqdm.tqdm(input_data):
    prompt = data["prompt"]
    all_generated_responses = data["all_generated_responses"]

    all_generated_response_scores = []
    for response in all_generated_responses:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model(input_ids)
            score = output.logits.float().item()
            all_generated_response_scores.append(score)

    data["orm_scores"] = all_generated_response_scores

file_name = f"chunk_{args.chunk_id}.json"
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(os.path.join(output_dir, file_name), 'w') as f:
    json.dump(input_data, f, indent=4)