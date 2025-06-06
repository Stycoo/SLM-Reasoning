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
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)
parser.add_argument('--max_steps', type=int, default=5)
args = parser.parse_args()

chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# def split_by_newlines(text):
#     # 使用正则表达式按至少两个换行符进行分割，保留换行符作为分隔符
#     paragraphs = re.split(r'(\n{2,})', text)
      
#     result = []
#     temp = paragraphs[0]
    
#     # 合并段落内容，保留换行符
#     for i in range(1, len(paragraphs)):
#         if i % 2 == 1:  # i是换行符的索引
#             temp += paragraphs[i]  # 加入换行符
#         else:  # i是段落内容的索引
#             result.append(temp)  # 将当前段落内容加入结果
#             temp = paragraphs[i]  # 更新临时段落

#     result.append(temp)  # 将最后一个段落加入结果
    
#     return result


def split_by_newlines(text):
    pattern = r'(.*?\n+)'
    return re.findall(pattern, text)


# 构建不同粒度的completion输入
def create_node(response, node_level, max_steps):
    # 不同level的node划分: token-level, sentence-level, option-level
    if node_level == "sentence":
        # 以换行符为切分标志: \n
        completion_inputs = split_by_newlines(response)
        
        if response.endswith('\n'):
            completion_inputs = completion_inputs[:-1]

        completion_inputs_used = []
        # for i, completion in enumerate(completion_inputs):
        #     print(completion)
        #     if completion:
        #         completion_inputs_used.append(''.join(completion_inputs[:i+1]))

        completion_inputs_num = len(completion_inputs)
        single_completion_size = max(1, completion_inputs_num // args.max_steps)
        for i in range(0, completion_inputs_num, single_completion_size):
            _completion_inputs_used = completion_inputs[0: i + single_completion_size]
            completion_inputs_used.append(''.join(_completion_inputs_used))
        assert ''.join(completion_inputs) == completion_inputs_used[-1]

        return completion_inputs_used


### 逐个遍历source model response 中的子句
input_file = args.input_file
with open(input_file, 'r') as f:
    input_data = json.load(f)

source_model_names = ['deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 'Qwen2.5-72B-Instruct']

prompts = []
best_responses = []
best_response_scores = []
completion_inputs = []
completion_ids = []
source_model_responses_scores = []

input_data = input_data[1000:5000]
input_data_size = len(input_data)
chunk_size = input_data_size // args.chunk_num
if args.chunk_id < args.chunk_num - 1:
    input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
else:
    input_data = input_data[args.chunk_id * chunk_size: ]

for data in input_data:
    prompt = data['prompt']
    best_response = data['best_response']
    best_response_score = data['best_response_score']
    prompt_normal = chat_template.format(prompt)

    response_num = 0
    for model_name in source_model_names:
        if model_name in data:
            source_model_responses = data[model_name]['response']
            response_num += len(source_model_responses)
            for i, model_response in enumerate(source_model_responses):
                _completion_inputs = create_node(model_response, 'sentence', args.max_steps)
                
                if _completion_inputs:
                    for _completion_input in _completion_inputs:
                        prompts.append(prompt)
                        best_responses.append(best_response)
                        best_response_scores.append(best_response_score)
                        completion_inputs.append(f"{prompt_normal}{_completion_input}")
                        completion_ids.append(f'{model_name}_{i}')
                        source_model_responses_scores.append(data[model_name]['rm_score'][i])

print(f"completion_inputs size: {len(completion_inputs)}")

llm = LLM(model=args.model_name_or_path, max_model_len=127376)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=args.temperature, 
                                 top_p=args.top_p, 
                                 max_tokens=args.max_tokens,
                                 seed=args.seed,)
outputs = llm.generate(completion_inputs, sampling_params) # completion_chunk

output_data = defaultdict(dict) 
for i, output in enumerate(outputs):
    completion_input = output.prompt
    prompt = prompts[i] # prompt_chunk[i]
    generated_text = output.outputs[0].text
    best_response_score = best_response_scores[i]
    best_response = best_responses[i]
    completion_id = completion_ids[i]
    source_model_response_score = source_model_responses_scores[i]

    if prompt in output_data:
        output_data[prompt]['completion_inputs'].append(completion_input)
        output_data[prompt]['completion_outputs'].append(generated_text)
        output_data[prompt]['completion_ids'].append(completion_id)
        output_data[prompt]['source_model_response_score'].append(source_model_response_score)
    else:
        output_data[prompt] = {'prompt': prompt, 'best_response':best_response, 'best_response_score':best_response_score, 
                               'completion_inputs': [completion_input], 'completion_outputs':[generated_text],
                               'completion_ids': [completion_id], 'source_model_response_score': [source_model_response_score]} 
        
output_dir = Path(args.output_dir) / args.model_id / f'seed_{args.seed}'
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"chunk_{args.chunk_id}.json"
output_data_list = []
for prompt in output_data:
    output_data_list.append(output_data[prompt])

with open(output_path, 'w') as f:
    json.dump(output_data_list, f, indent=4)

