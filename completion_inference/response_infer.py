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
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)           
                    
args = parser.parse_args()

chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

input_file = args.input_file
llm = LLM(model=args.model_name_or_path, max_model_len=127376, gpu_memory_utilization=0.9)
tokenizer = llm.get_tokenizer()

with open(input_file, 'r') as f:
    input_data = json.load(f)
input_data = input_data[1000:5000]
input_data_size = len(input_data)
chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

prompts = []
for data in input_data:
    prompt = data['prompt']
    prompts.append(prompt)

prompts = list(set(prompts))
print(f"dataset size: {len(prompts)}")

# conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
conversations = [chat_template.format(prompt) for prompt in prompts]

sampling_params = SamplingParams(temperature=args.temperature, 
                                 top_p=args.top_p, 
                                 max_tokens=args.max_tokens, 
                                 seed=args.seed,
                                 logprobs=1,)

outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    generated_response = output.outputs[0].text

    output_data.append({
        'prompt': prompts[i],
        'generated_response': generated_response,
    })

output_dir = Path(args.output_dir) / f'seed_{args.seed}'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"chunk_{args.chunk_id}.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=4)