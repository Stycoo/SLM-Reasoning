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
parser.add_argument("--segment_num", type=int, default=0)
parser.add_argument("--iteration_id_of_training", type=int, default=0)
parser.add_argument("--iterative_rollout", type=bool, default=False)
parser.add_argument('--start_segment_id_log_file', type=str)
args = parser.parse_args()

### 逐个遍历source model response 中的子句
input_file = args.input_file
with open(input_file, 'r') as f:
    input_data = json.load(f)
    # input_data = input_data[:200]

if args.iteration_id_of_training != 1:
    assert args.start_segment_id_log_file
    with open(args.start_segment_id_log_file, 'r') as f:
        start_segment_id_records = json.load(f)

    if args.iterative_rollout:
        for data in input_data:
            prompt = data['conversations'][0]['value']
            start_segment_id_records[prompt] += 1

        with open(args.start_segment_id_log_file, 'w') as f:
            json.dump(start_segment_id_records, f, indent=4)

else:
    start_segment_id_records = {}
    for data in input_data:
        prompt = data['conversations'][0]['value']
        start_segment_id_records[prompt] = 0
    with open(args.start_segment_id_log_file, 'w') as f:
        json.dump(start_segment_id_records, f, indent=4)

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

iteration_arrive_the_last_segment = {}

for data in input_data:
    assert data['conversations'][0]['from'] == 'human'
    prompt = data['conversations'][0]['value']
    answer = data['conversations'][1]['value']

    start_segment_id = start_segment_id_records[prompt]

    messages = [{"role": "user", "content": prompt}]
    prompt_normal = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)
    
    completion_steps = answer.split("\n\n")

    if len(completion_steps) < args.segment_num:
        segment_num = len(completion_steps)
    else:
        segment_num = args.segment_num

    part_size = len(completion_steps) // segment_num
    remainder = len(completion_steps) % segment_num

    parts = [prompt_normal]
    start_index = 0
    for i in range(segment_num - 1):
        end_index = start_index + part_size + (1 if i < remainder else 0)
        parts.append('\n\n'.join(completion_steps[start_index:end_index]))
        start_index = end_index  # 更新下一个部分的起始索引

    if start_segment_id != len(parts) - 1:
        if start_segment_id == 0:
            completion_inputs.append(prompt_normal)
        else:
            completion_input = '\n\n'.join(parts[1:start_segment_id+1])
            completion_inputs.append(f"{prompt_normal}{completion_input}\n\n")

        prompts.append(prompt)
        answers.append(answer)

    else:
        completion_input = '\n\n'.join(parts[1:start_segment_id+1])
        iteration_arrive_the_last_segment[prompt] = {'completion_input': f"{prompt_normal}{completion_input}", 'answer': answer}
        continue

print(f"completion_inputs size: {len(completion_inputs)}")

llm = LLM(model=args.model_name_or_path)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=args.temperature, 
                                 max_tokens=args.max_tokens,
                                 seed=args.seed,)
outputs = llm.generate(completion_inputs, sampling_params) # completion_chunk

output_data = defaultdict(dict) 
for prompt in iteration_arrive_the_last_segment:
    output_data[prompt] = {'prompt': prompt, 'answer': answers[i], 'completion_inputs': [iteration_arrive_the_last_segment[prompt]['completion_input']], 
                            'completion_outputs':[iteration_arrive_the_last_segment[prompt]['answer']]}

for i, output in enumerate(outputs):
    completion_input = output.prompt
    prompt = prompts[i]
    answer = answers[i]
    generated_text = output.outputs[0].text

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