from vllm import LLM, SamplingParams
import os
import argparse
import json
from collections import defaultdict
from pathlib import Path
import re

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--input_file', type=str)
parser.add_argument('--model_name_or_path', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--model_id', type=str)
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=2048,
                    help='Maximum number of tokens to generate')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='output_dir')
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def split_by_newlines(text):
    pattern = r'(.*?\n+)'
    return re.findall(pattern, text)

# 构建不同粒度的completion输入
def create_node(response, node_level, min_time, max_time):
    # 不同level的node划分: token-level, sentence-level, option-level
    if node_level == "sentence":
        # 以换行符为切分标志: \n
        sub_sentences = response.split('\n')
        sub_sentences_num = 0
        for sub_sent in sub_sentences:
            if sub_sent:
                sub_sentences_num += 1

        completion_inputs = split_by_newlines(response)
        completion_inputs_num = len(completion_inputs)

        if completion_inputs_num < min_time:
            completion_inputs_used = []

        elif completion_inputs_num > min_time and completion_inputs_num <= max_time:
            if completion_inputs_num >= sub_sentences_num:
                completion_inputs = completion_inputs[:-1]
            completion_inputs_used = []
            for i, completion in enumerate(completion_inputs):
                completion_inputs_used.append(''.join(completion_inputs[:i+1]))

        else:
            single_completion_size = completion_inputs_num // max_time
            completion_inputs_used = []
            for i in range(max_time):
                _completion_inputs_used = completion_inputs[0: (i+1)*single_completion_size]
                if (i+1)*single_completion_size < completion_inputs_num:
                    completion_inputs_used.append(''.join(_completion_inputs_used))
                else:
                    break

        return completion_inputs_used

input_file = args.input_file
llm = LLM(model=args.model_name_or_path)
tokenizer = llm.get_tokenizer()

with open(input_file, 'r') as f:
    input_data = json.load(f)
# input_data = input_data[0:100]

prompts = []
best_responses = []
best_response_scores = []
completion_inputs = []

for data in input_data:
    prompt = data['prompt']
    best_response = data['best_response']
    best_response_score = data['best_response_score']

    # prompt_normal = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    prompt_normal = chat_template.format(prompt)

    _completion_inputs = create_node(best_response, 'sentence', 2, 3)
    if _completion_inputs:
        for _completion_input in _completion_inputs:
            prompts.append(prompt)
            best_responses.append(best_response)
            best_response_scores.append(best_response_score)

            completion_inputs.append(f"{prompt_normal}{_completion_input}")

sampling_params = SamplingParams(temperature=args.temperature, 
                                 top_p=args.top_p, 
                                 max_tokens=args.max_tokens,
                                 seed=args.seed,)
outputs = llm.generate(completion_inputs, sampling_params) # completion_chunk

output_data = defaultdict(dict) 
# {'prompt':, 'best_response':, 'best_response_scores':[], 'completion_inputs':[], 'completion_outputs':[], 'completion_rm_scores':[]}
for i, output in enumerate(outputs):
    completion_input = output.prompt
    prompt = prompts[i] # prompt_chunk[i]
    generated_text = output.outputs[0].text
    best_response_score = best_response_scores[i]
    best_response = best_responses[i]
    
    if prompt in output_data:
        output_data[prompt]['completion_inputs'].append(completion_input)
        output_data[prompt]['completion_outputs'].append(generated_text)
    else:
        output_data[prompt] = {'prompt': prompt, 'best_response':best_response, 'best_response_score':best_response_score, 
                               'completion_inputs': [completion_input], 'completion_outputs':[generated_text]} 
                               # 'target_model_response': target_model_response, 'target_model_response_score': target_model_response_score, 

output_dir = Path(args.output_dir) / args.model_id
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f'seed_{args.seed}.json'
output_data_list = []
for prompt in output_data:
    output_data_list.append(output_data[prompt])

with open(output_path, 'w') as f:
    json.dump(output_data_list, f, indent=4)
