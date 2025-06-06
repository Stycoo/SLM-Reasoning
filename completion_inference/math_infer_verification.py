import json
import os
import argparse
import tqdm
from pathlib import Path
from math_verify import parse, verify
import re

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default='', help="Path to the output generation file")
parser.add_argument("--model_name_or_path", type=str, default='', help="Path to reward model")
parser.add_argument("--output_dir", type=str, default='', help="Path to output directory")
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)
parser.add_argument("--verfiy_mode", type=str, default='', help="PRM or Rule or LLM-as-a-judge")

args = parser.parse_args()

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

with open(args.input_file, 'r') as f:
    input_data = json.load(f)
    input_data_size = len(input_data)

if args.verfiy_mode == 'PRM':
    chunk_size = input_data_size // args.chunk_num
    if args.chunk_id < args.chunk_num - 1:
        input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
    else:
        input_data = input_data[args.chunk_id * chunk_size: ]

    device = "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    for data in tqdm.tqdm(input_data):
        prompt = data["prompt"]
        all_generated_responses = data["all_generated_responses"]
        all_generated_response_step_scores = []
        for response in all_generated_responses:
            data = {
            "system": "Please reason step by step, and put your final answer within \\boxed{}.",
            "query": prompt,
            "response": response.split('\n\n')
            }
            messages = [
                {"role": "system", "content": data['system']},
                {"role": "user", "content": data['query']},
                {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
            ]
            conversation_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )

            input_ids = tokenizer.encode(
                conversation_str, 
                return_tensors="pt", 
                ).to(model.device)
            outputs = model(input_ids=input_ids)

            step_sep_id = tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_reward = make_step_rewards(outputs[0], token_masks)
            all_generated_response_step_scores.append(step_reward[0])

        data["prm_step_scores"] = all_generated_response_step_scores

    file_name = f"chunk_{args.chunk_id}.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(input_data, f, indent=4)

if args.verfiy_mode == 'Rule':
    overall_acc = []
    for data in tqdm.tqdm(input_data):
        prompt = data["prompt"]
        answer = data['answer']

        # gold = parse(f"${answer}$")
        last_step_answer = answer.split('\n\n')[-1]
        gold = parse(f"${last_step_answer}$")

        all_generated_responses = data["all_generated_responses"]
        all_generated_response_scores = []

        for response in all_generated_responses:
            last_step = response.split('\n\n')[-1]
            extracted_answer = parse(last_step)
            
            # Order here is important!
            true_or_false = verify(gold, extracted_answer)

            if true_or_false:
                all_generated_response_scores.append(1)
            else:
                all_generated_response_scores.append(0)

        data["rule_scores"] = all_generated_response_scores
        overall_acc.append(sum(all_generated_response_scores) / len(all_generated_response_scores))

    file_name = 'all_generated_response_rule_verified.json'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(input_data, f, indent=4)
    
    overall_acc = sum(overall_acc) / len(overall_acc)
    print(f"overall_acc: {overall_acc:.4f}")
