import argparse
import json
from collections import defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Decode with vLLM')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default="google/gemma-2-9b-it")
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=8000)
    parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk_num', type=int, default=8)
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--iteration_time', type=int, default=0)
    parser.add_argument('--completion_id', type=int, default=0)
    return parser.parse_args()

# Load and chunk data
def load_input_data(input_file, chunk_num, chunk_id):
    with open(input_file, 'r') as f:
        data = json.load(f)
        # data = data[:200]

    total = len(data)
    chunk_size = total // chunk_num
    start = chunk_id * chunk_size
    end = (chunk_id + 1) * chunk_size if chunk_id < chunk_num - 1 else total
    return data[start:end]

# Flatten segment rollout inputs
def flatten_data(input_data):
    prompts, answers, completion_inputs = [], [], []
    seg_inputs_flat, seg_prefixes_flat = [], []

    for data in input_data:
        prompt = data['prompt']
        answer = data['answer']
        completion_input = data['completion_input']

        for key, val in data.items():
            if key in ['prompt', 'answer', 'completion_input']:
                continue
            seg_inputs = val['segment_rollout_inputs']

            flat_inputs = [item for group in seg_inputs for item in group]
            seg_inputs_flat.extend(flat_inputs)
            seg_prefixes_flat.extend([key] * len(flat_inputs))
            prompts.extend([prompt] * len(flat_inputs))
            answers.extend([answer] * len(flat_inputs))
            completion_inputs.extend([completion_input] * len(flat_inputs))

    return prompts, answers, completion_inputs, seg_inputs_flat, seg_prefixes_flat

# Generate completions with vLLM
def generate_responses(model_name, inputs, temperature, max_tokens, seed):
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
    return llm.generate(inputs, sampling_params)

# Reorganize output into structured dict
def collect_outputs(outputs, prompts, answers, completion_inputs, seg_prefixes):
    result = defaultdict(dict)
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        answer = answers[i]
        completion_input = completion_inputs[i]
        prefix = seg_prefixes[i]
        segment_input = output.prompt
        segment_output = output.outputs[0].text

        if prompt not in result:
            result[prompt] = {
                'prompt': prompt,
                'answer': answer,
                'completion_input': completion_input
            }

        if prefix not in result[prompt]:
            result[prompt][prefix] = {
                'segment_rollout_inputs': [],
                'segment_rollout_outputs': []
            }

        result[prompt][prefix]['segment_rollout_inputs'].append(segment_input)
        result[prompt][prefix]['segment_rollout_outputs'].append(segment_output)
    
    return list(result.values())

# Save output
def save_output(output_dir, chunk_id, seed, data):
    output_dir = Path(output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"chunk_{chunk_id}.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# Main logic
def main():
    args = parse_args()
    input_data = load_input_data(args.input_file, args.chunk_num, args.chunk_id)

    prompts, answers, completion_inputs, seg_inputs, seg_prefixes = flatten_data(input_data)
    print(f"Total segment rollout inputs: {len(seg_inputs)}")

    outputs = generate_responses(args.model_name_or_path, seg_inputs,
                                 args.temperature, args.max_tokens, args.seed)

    output_data = collect_outputs(outputs, prompts, answers, completion_inputs, seg_prefixes)
    save_output(args.output_dir, args.chunk_id, args.seed, output_data)

if __name__ == '__main__':
    main()