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
parser.add_argument("--iteration_time", type=int, default=0)
parser.add_argument("--completion_id", type=int, default=0)
args = parser.parse_args()


### 逐个遍历source model response 中的子句
input_file = args.input_file
with open(input_file, 'r') as f:
    input_data = json.load(f)
    # input_data = input_data[:1000]

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

prompts = []
answers = []
completion_inputs_for_accu_q = []

input_data_size = len(input_data)
chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

for data in input_data:
    prompt = data['prompt']
    answer = data['answer']
    completion_outputs = data['completion_outputs']
    completion_input = data['completion_inputs'][0]

    # completion_steps = answer.split("\n\n")
    # if len(completion_steps) < (args.iteration_time):
    #     continue
    
    # part_size = len(completion_steps) // (args.iteration_time + 1)
    # remainder = len(completion_steps) % (args.iteration_time + 1)

    # parts = []
    # start_index = 0
    # for i in range(args.iteration_time):
    #     end_index = start_index + part_size + (1 if i < remainder else 0)
    #     parts.append('\n\n'.join(completion_steps[start_index: end_index]))
    #     start_index = end_index

    # source_segment_completion_i = parts[1-args.completion_id]
    # completion_inputs_for_accu_q.append(f"{completion_input}{source_segment_completion_i}\n\n")
    # prompts.append(prompt)
    # answers.append(answer)

    for comp_output in completion_outputs[0]:
        comp_output_steps = comp_output.split("\n\n")
        comp_output_step_num = len(comp_output_steps)

        if comp_output_step_num < (args.completion_id):
            continue
    
        part_size = comp_output_step_num // (args.completion_id)
        remainder = comp_output_step_num % (args.completion_id)

        parts = []
        start_index = 0
        for i in range(args.completion_id):
            end_index = start_index + part_size + (1 if i < remainder else 0)
            parts.append('\n\n'.join(comp_output_steps[start_index: end_index]))
            start_index = end_index

        completion_inputs_for_accu_q.append(f"{completion_input}{parts[0]}\n\n")
        prompts.append(prompt)
        answers.append(answer)  

print(f"completion_inputs_for_accu_q size: {len(completion_inputs_for_accu_q)}")

llm = LLM(model=args.model_name_or_path)
sampling_params = SamplingParams(temperature=args.temperature, 
                                 max_tokens=args.max_tokens,
                                 seed=args.seed,)
outputs = llm.generate(completion_inputs_for_accu_q, sampling_params) # completion_chunk

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
        
output_dir = Path(args.output_dir) / f'seed_{args.seed}'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"chunk_{args.chunk_id}.json"
output_data_list = []
for prompt in output_data:
    output_data_list.append(output_data[prompt])

with open(output_path, 'w') as f:
    json.dump(output_data_list, f, indent=4)

