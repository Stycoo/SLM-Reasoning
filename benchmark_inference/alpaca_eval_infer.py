from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Dataset
import os
import argparse
import json
import tiktoken
import pandas as pd

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--question_file', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--model_id', type=str)
parser.add_argument('--temperature', type=float, default=0.9,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=1.0,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='output_dir')

args = parser.parse_args()

question_file = args.question_file
llm = LLM(model=args.model, max_model_len=129472)
tokenizer = llm.get_tokenizer()

with open(question_file, 'r') as f:
    questions = json.load(f)

alpaca_dataset = Dataset.from_list(questions)

dataset_id = alpaca_dataset['dataset']
prompts = alpaca_dataset['instruction']

chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
# conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

conversations = []
for prompt in prompts:
    formatted_conversation = chat_template.format(prompt)
    conversations.append(formatted_conversation)

sampling_params = SamplingParams(temperature=args.temperature, 
                                 top_p=args.top_p, 
                                 max_tokens=args.max_tokens,)
outputs = llm.generate(conversations, sampling_params)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    
    output_data.append({
        'dataset': dataset_id[i],
        'generator': args.model_id,
        'instruction': prompts[i],
        'output': generated_text
    })

output_file = 'model_outputs.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
