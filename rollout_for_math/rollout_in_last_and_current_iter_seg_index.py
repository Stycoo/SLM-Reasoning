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
parser.add_argument('--forward_or_reverse', type=str)
parser.add_argument('--last_iteration_segment_index_file', type=str)

args = parser.parse_args()

### 逐个遍历source model response 中的子句
with open(args.input_file, 'r') as f:
    input_data = json.load(f)
    # input_data = input_data[:100]

if args.completion_id != 1:
    assert args.last_iteration_segment_index_file
    with open(args.last_iteration_segment_index_file, 'r') as f:
        last_iteration_segment_index_dict = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

prompts = []
answers = []
completion_inputs = []
last_step_completion_inputs = []
chosen_rejected_flag = []

input_data_size = len(input_data)
chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

less_than_iteration_time_data_count = 0

for data in input_data:
    assert data['conversations'][0]['from'] == 'human'
    prompt = data['conversations'][0]['value']
    answer = data['conversations'][1]['value']
    
    messages = [{"role": "user", "content": prompt}]
    prompt_normal = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)
    
    completion_steps = answer.split("\n\n")
    if len(completion_steps) < args.iteration_time:
        less_than_iteration_time_data_count += 1
        continue

    part_size = len(completion_steps) // args.iteration_time
    remainder = len(completion_steps) % args.iteration_time
    parts = []
    start_index = 0
    for i in range(args.iteration_time):
        # 如果有多余的元素，当前部分需要多一个元素
        end_index = start_index + part_size + (1 if i < remainder else 0)
        parts.append('\n\n'.join(completion_steps[start_index:end_index]))
        start_index = end_index  # 更新下一个部分的起始索引

    # reverse
    if args.forward_or_reverse == 'reverse':
        if args.completion_id == 1:
            prompts.append(prompt)
            answers.append(answer)

            rejected_completion_input = '\n\n'.join(parts[:args.iteration_time-args.completion_id])
            completion_inputs.append(f"{prompt_normal}{rejected_completion_input}\n\n")
            
            chosen_rejected_flag.append(0)
            last_step_completion_inputs.append(f"{prompt_normal}{answer}")

        else:
            last_iteration_segment_index = last_iteration_segment_index_dict[prompt] 
            chosen_completion_input = '\n\n'.join(parts[:args.iteration_time-last_iteration_segment_index+1])

            if last_iteration_segment_index == 1:
                last_step_completion_inputs.append(f"{prompt_normal}{answer}")
                prompts.append(prompt)
                answers.append(answer)
            else:
                completion_inputs.append(f"{prompt_normal}{chosen_completion_input}\n\n")
                chosen_rejected_flag.append(1)
                last_step_completion_inputs.extend([''] * 2)
                prompts.extend([prompt] * 2)
                answers.extend([answer] * 2)

            rejected_completion_input = '\n\n'.join(parts[:args.iteration_time-args.completion_id])
            if rejected_completion_input:
                completion_inputs.append(f"{prompt_normal}{rejected_completion_input}\n\n")
            else:
                completion_inputs.append(prompt_normal)
            chosen_rejected_flag.append(0)

    # forward
    if args.forward_or_reverse == 'forward':
        if args.completion_id == args.iteration_time:
            prompts.append(prompt)
            answers.append(answer)

            last_step_completion_inputs.append(f"{prompt_normal}{answer}")

            last_iteration_segment_index = last_iteration_segment_index_dict[prompt]
            rejected_completion_input = '\n\n'.join(parts[:last_iteration_segment_index-1])

            if rejected_completion_input:
                completion_inputs.append(f"{prompt_normal}{rejected_completion_input}\n\n")
            else:
                completion_inputs.append(prompt_normal)
            
            chosen_rejected_flag.append(0)

        else:
            prompts.extend([prompt] * 2)
            answers.extend([answer] * 2)
            last_step_completion_inputs.extend([''] * 2)
            
            chosen_completion_input = '\n\n'.join(parts[:args.completion_id])
            completion_inputs.append(f"{prompt_normal}{chosen_completion_input}\n\n")
            chosen_rejected_flag.append(1)

            if args.completion_id == 1:
                rejected_completion_input = '\n\n'.join(parts[:args.completion_id-1])
            else:
                last_iteration_segment_index = last_iteration_segment_index_dict[prompt] 
                rejected_completion_input = '\n\n'.join(parts[:last_iteration_segment_index-1])

            if rejected_completion_input:
                completion_inputs.append(f"{prompt_normal}{rejected_completion_input}\n\n")
            else:
                completion_inputs.append(prompt_normal)

            chosen_rejected_flag.append(0)
            
print(f"completion_inputs size: {len(completion_inputs)}")
print(f"less_than_iteration_time_data_count: {less_than_iteration_time_data_count}")

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
    chosen_rejected = chosen_rejected_flag[i]
    last_step_completion_input = last_step_completion_inputs[i]

    if last_step_completion_input:
        output_data[prompt] = {'prompt': prompt, 'answer': answer, 'completion_inputs': [last_step_completion_input, completion_input], 
                                'completion_outputs':[answer, generated_text], 'chosen_rejected': [1, chosen_rejected]}
    else:
        if prompt in output_data:
            output_data[prompt]['completion_inputs'].append(completion_input)
            output_data[prompt]['completion_outputs'].append(generated_text)
            output_data[prompt]['chosen_rejected'].append(chosen_rejected)
        else:
            output_data[prompt] = {'prompt': prompt, 'answer': answer, 'completion_inputs': [completion_input], 
                                    'completion_outputs':[generated_text], 'chosen_rejected': [chosen_rejected]}  
        
output_dir = Path(args.output_dir) / f'seed_{args.seed}'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"chunk_{args.chunk_id}.json"
output_data_list = []
for prompt in output_data:
    output_data_list.append(output_data[prompt])

with open(output_path, 'w') as f:
    json.dump(output_data_list, f, indent=4)

