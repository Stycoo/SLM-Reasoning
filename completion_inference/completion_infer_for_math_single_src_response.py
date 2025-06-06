from vllm import LLM, SamplingParams
import os
import argparse
import json
from collections import defaultdict
from pathlib import Path
import re
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--input_file', type=str)
parser.add_argument('--model_name_or_path', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--model_id', type=str)
parser.add_argument('--temperature', type=float, default=0.9,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=8000,
                    help='Maximum number of tokens to generate')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='output_dir')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)
args = parser.parse_args()

### 逐个遍历source model response 中的子句
input_file = args.input_file
with open(input_file, 'r') as f:
    input_data = json.load(f)
    # input_data = input_data[:100]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

prompts = []
answers = []
completion_inputs = []

input_data_size = len(input_data)
chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

for data in input_data:
    assert data['conversations'][0]['from'] == 'human'
    prompt = data['conversations'][0]['value']
    answer = data['conversations'][1]['value']

    messages = [{"role": "user", "content": prompt}]

    prompt_normal = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)
    
    completion_steps = answer.split("\n\n")[:-1]

    if completion_steps:
        for j, completion_step in enumerate(completion_steps):
            prompts.append(prompt)
            answers.append(answer)
            completion_input = '\n\n'.join(completion_steps[:j+1])
            completion_inputs.append(f"{prompt_normal}{completion_input}\n\n")

print(f"completion_inputs size: {len(completion_inputs)}")

llm = LLM(model=args.model_name_or_path)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=args.temperature, 
                                 max_tokens=args.max_tokens,
                                 seed=args.seed,)
outputs = llm.generate(completion_inputs, sampling_params) # completion_chunk

output_data = defaultdict(dict) 
for i, output in enumerate(outputs):
    completion_input = output.prompt
    prompt = prompts[i]
    answer = answers[i]
    generated_text = output.outputs[0].text

    if prompt in output_data:
        output_data[prompt]['completion_inputs'].append(completion_input)
        output_data[prompt]['completion_outputs'].append(generated_text)
    else:
        output_data[prompt] = {'prompt': prompt, 'answer': answer, 'completion_inputs': [completion_input], 
                                'completion_outputs':[generated_text]}  
        
output_dir = Path(args.output_dir) / args.model_id / f'seed_{args.seed}'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"chunk_{args.chunk_id}.json"
output_data_list = []
for prompt in output_data:
    output_data_list.append(output_data[prompt])

with open(output_path, 'w') as f:
    json.dump(output_data_list, f, indent=4)

