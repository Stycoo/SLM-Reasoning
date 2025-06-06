from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Dataset
import os
from collections import defaultdict
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--input_file', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='output_dir')
parser.add_argument('--model_name_or_path', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.9,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=8000,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)           
                    
args = parser.parse_args()

input_file = args.input_file
llm = LLM(model=args.model_name_or_path) # max_model_len=127376, gpu_memory_utilization=0.9
tokenizer = llm.get_tokenizer()

with open(input_file, 'r') as f:
    input_data = json.load(f)

input_data_size = len(input_data)
chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

prompts = []
answers = []
for data in input_data:
    # assert data['conversations'][0]['from'] == 'human'
    # prompt = data['conversations'][0]['value']
    # answer = data['conversations'][1]['value']

    prompt = data['prompt']
    answer = data['answer']

    prompts.append(prompt)
    answers.append(answer)

print(f"dataset size: {len(prompts)}")

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
                                             
sampling_params = SamplingParams(temperature=args.temperature, 
                                 max_tokens=args.max_tokens, 
                                 seed=args.seed,)
outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    generated_response = output.outputs[0].text

    output_data.append({
        'prompt': prompts[i],
        'answer': answers[i],
        'generated_response': generated_response,
    })

output_dir = Path(args.output_dir) / f'seed_{args.seed}'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"chunk_{args.chunk_id}.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=4)