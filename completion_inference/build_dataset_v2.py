import os
import json
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import re
import numpy as np
from datasets import load_from_disk
import random

parser = argparse.ArgumentParser()
parser.add_argument("--completion_file_dir", type=str, default='')
parser.add_argument("--response_file_dir", type=str, default='') 
parser.add_argument("--source_model_response_file", type=str, default='')
parser.add_argument("--target_model_response_file", type=str, default='', help="Path to reward model")
parser.add_argument("--train_data_save_dir", type=str, default='', help="Path to output directory")
parser.add_argument("--post_process_type", type=str, default='')
parser.add_argument("--completion_start_index", type=int, default=0)
parser.add_argument('--completion_rank_file', type=str, default='')
parser.add_argument("--iteration_id", type=int, default=0)
parser.add_argument("--forward_or_reverse", type=str, default='')
parser.add_argument("--iteration_time", type=int, default=0)

args = parser.parse_args()

def read_json_file(input_file):
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_list = [dt for dt in data]
    return data_list


def write_json_file(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)


def merge_target_multi_responses(generation_file_dir):
    # 将不同seed下target model生成的response合并到单个文件保存
    all_data = []
    for file_name in os.listdir(generation_file_dir):
        if file_name.startswith("seed") and file_name.endswith(".json"):
            generation_file = os.path.join(generation_file_dir, file_name)
            with open(generation_file, 'r') as f:
                output_data = json.load(f)
                data_dict = defaultdict(dict)
                for data in output_data:
                    prompt = data["prompt"]
                    data_dict[prompt] = data
                all_data.append(data_dict)

    all_res = []
    num_identical = 0

    for prompt in all_data[0]:
        generated_responses = []
        prompts = []

        for data in all_data:
            generated_responses.append(data[prompt]["generated_response"])
            prompts.append(data[prompt]["prompt"])

        assert len(set(prompts)) == 1, print(len(set(prompts)))

        if len(set(generated_responses)) == 1:
            # filter out samples where all generated responses are identical
            num_identical += 1
            # continue

        all_res.append(
            {
                "prompt": prompt,
                "all_generated_responses": generated_responses,
            }
        )
    print(f"All outputs size: {len(all_res)}; {num_identical} samples with identical generated responses")

    with open(os.path.join(generation_file_dir, 'all_generated_response.json'), 'w') as f:
        json.dump(all_res, f, indent=4)

    print(f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_generated_response.json')}")


def merge_multi_completions(completion_file_dir):
    # 将不同seed下生成的completion files合并为单个文件
    all_data = []
    for file_name in os.listdir(completion_file_dir):
        if file_name.startswith("seed") and file_name.endswith(".json"):
            completion_file = os.path.join(completion_file_dir, file_name)
            with open(completion_file, 'r') as f:
                output_data = json.load(f)
                data_dict = defaultdict(dict)
                for data in output_data:
                    prompt = data["prompt"]
                    data_dict[prompt] = data
                all_data.append(data_dict)

    merged_data = []
    for prompt in all_data[0]:
        all_generated_completions = []
        completions_num = len(all_data[0][prompt]["completion_outputs"])
        # assert completions_num <= 5
        for i in range(completions_num):
            all_generated_completions_i = []
            for data in all_data:
                completion_output = data[prompt]["completion_outputs"][i]
                all_generated_completions_i.append(completion_output)
            all_generated_completions.append(all_generated_completions_i)

        # all_data[0][prompt]['completion_outputs'] = all_generated_completions
        data_previous = all_data[0][prompt]
        data_previous['completion_outputs'] = all_generated_completions
        merged_data.append(data_previous)

    with open(os.path.join(completion_file_dir, 'all_generated_completions.json'), 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Processed outputs saved to {os.path.join(completion_file_dir, 'all_generated_completions.json')}")


def merge_multi_seeds_completion_chunks(generation_file_dir):
    for file_dir in Path(generation_file_dir).iterdir():
        if file_dir.is_dir():
            name = file_dir.name
            merge_completion_chunks(file_dir, name)
    
    all_data = []
    for file_dir in Path(generation_file_dir).iterdir():
        if file_dir.is_dir():
            for completion_file in Path(file_dir).iterdir():
                file_name = completion_file.name
                if file_name.startswith("seed") and file_name.endswith(".json"):
                    with open(completion_file, 'r') as f:
                        output_data = json.load(f)
                        data_dict = defaultdict(dict)
                        for data in output_data:
                            prompt = data["prompt"]
                            data_dict[prompt] = data
                        all_data.append(data_dict)

    merged_data = []
    invalid_data_count = 0
    for prompt in all_data[0]:
        all_generated_completions = []
        completions_num = len(all_data[0][prompt]["completion_outputs"])
        completion_inputs = all_data[0][prompt]["completion_inputs"]

        # assert completions_num <= 5
        for i in range(completions_num):
            all_generated_completions_i = []
            for data in all_data:
                if not data[prompt]:
                    invalid_data_count += 1
                    continue
                # assert completion_inputs[i] == data[prompt]["completion_inputs"][i]
                completion_output = data[prompt]["completion_outputs"][i]
                all_generated_completions_i.append(completion_output)
            all_generated_completions.append(all_generated_completions_i)

        # all_data[0][prompt]['completion_outputs'] = all_generated_completions
        data_previous = all_data[0][prompt]
        data_previous['completion_outputs'] = all_generated_completions
        merged_data.append(data_previous)

    with open(os.path.join(generation_file_dir, 'all_generated_completions.json'), 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_generated_completions.json')}")
    print(f"Invalid data count: {invalid_data_count}")

def merge_multi_seeds_response_chunks(generation_file_dir):
    for file_dir in Path(generation_file_dir).iterdir():
        if file_dir.is_dir():
            name = file_dir.name
            merge_completion_chunks(file_dir, name)

    all_data = []
    for file_dir in Path(generation_file_dir).iterdir():
        if file_dir.is_dir():
            for response_file in Path(file_dir).iterdir():
                file_name = response_file.name
                if file_name.startswith("seed") and file_name.endswith(".json"):
                    with open(response_file, 'r') as f:
                        output_data = json.load(f)
                        data_dict = defaultdict(dict)
                        for data in output_data:
                            prompt = data["prompt"]
                            data_dict[prompt] = data
                        all_data.append(data_dict)

    all_res = []
    num_identical = 0
    for prompt in all_data[0]:
        generated_responses = []
        prompts = []

        for data in all_data:
            generated_responses.append(data[prompt]["generated_response"])
            prompts.append(data[prompt]["prompt"])

        assert len(set(prompts)) == 1, print(len(set(prompts)))

        if len(set(generated_responses)) == 1:
            # filter out samples where all generated responses are identical
            num_identical += 1
            # continue

        if 'answer' in all_data[0][prompt]:
            answer = all_data[0][prompt]['answer']
        elif 'gt_answer' in all_data[0][prompt]:
            answer = all_data[0][prompt]['gt_answer']

        all_res.append(
            {
                "prompt": prompt,
                "all_generated_responses": generated_responses,
                "answer": answer
            }
        )
    print(f"All outputs size: {len(all_res)}; {num_identical} samples with identical generated responses")

    with open(os.path.join(generation_file_dir, 'all_generated_responses.json'), 'w') as f:
        json.dump(all_res, f, indent=4)

    print(f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_generated_responses.json')}")


def merge_completion_chunks(input_dir, output_file_name):
    merged_data = []

    # Iterate through all the files in the directory
    for filename in os.listdir(input_dir):
        if filename.startswith("chunk") and filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            # Open and load each JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Check if the data is a list
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {file_path} does not contain a list and will be skipped.")

    # Write the merged data to the output file
    output_file = os.path.join(input_dir, f'{output_file_name}.json')
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)


def add_target_model_response_to_completion_file(completion_file, target_model_file, output_file):
    target_model_responses = read_json_file(target_model_file)
    target_model_response_score = defaultdict(dict)
    for data in target_model_responses:
        prompt = data['prompt']
        all_target_response_scores = data['all_rm_scores']
        all_target_responses = data['all_generated_responses']
        target_model_response_score[prompt] = {"response": all_target_responses, "rm_score": all_target_response_scores}

    completion_responses = read_json_file(completion_file)
    for comp_data in completion_responses:
        prompt = comp_data["prompt"]
        all_target_responses = target_model_response_score[prompt]["response"]
        all_target_response_scores = target_model_response_score[prompt]["rm_score"]

        comp_data['target_model_responses'] = all_target_responses
        comp_data['target_model_response_scores'] = all_target_response_scores

    write_json_file(completion_responses, output_file)
    print(f"{len(target_model_responses) - len(completion_responses)} filtered.")
    # {'prompt': str, 'best_response':str, 'best_response_score':float, 'completion_inputs': [], 'completion_outputs':[[],...]}, 'completion_scores':[[],...]:,
    # 'target_model_responses':[], 'target_model_response_scores': []'}


def build_target_model_dpo_dataset(completion_dpo_file, target_model_response_file, save_dir):
    # 根据completion dpo dataset中指令数据，将chosen-rejected替换为target model response
    target_model_response_data = read_json_file(target_model_response_file)
    target_model_chosen_rejected_data = defaultdict(dict)
    for data in target_model_response_data:
        prompt = data['prompt']
        all_target_response_scores = data['all_rm_scores']
        all_target_responses = data['all_generated_responses']

        if len(set(all_target_responses)) == 1:
            continue
        
        target_model_chosen = all_target_responses[all_target_response_scores.index(max(all_target_response_scores))]
        target_model_rejected = all_target_responses[all_target_response_scores.index(min(all_target_response_scores))]
        target_model_chosen_rejected_data[prompt] = {"conversations":[{'from':'human', 'value':prompt}], 
                                                     "chosen":{'from':'gpt', 'value':target_model_chosen},
                                                     "rejected":{'from':'gpt', 'value':target_model_rejected}}

    completion_dpo_data = read_json_file(completion_dpo_file)
    target_model_chosen_rejected_data_filtered = []
    for data in completion_dpo_data:
        prompt = data['conversations'][0]['value']
        if prompt in target_model_chosen_rejected_data:
            target_model_chosen_rejected_data_filtered.append(target_model_chosen_rejected_data[prompt])

    print(f"completion_dpo_datasize: {len(completion_dpo_data)}, target_model_response_dpo_datasize:{len(target_model_chosen_rejected_data_filtered)}")
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'dpo_target_model_1.0.json'
    write_json_file(target_model_chosen_rejected_data_filtered, save_path)


def build_source_model_dpo_dataset(completion_dpo_file, source_model_response_dir, src_model_names, save_dir):
    # 根据completion dpo dataset中指令数据，将chosen-rejected替换为source model response(属于同一source model)
    source_model_response_dir = Path(source_model_response_dir)
    source_model_response_dict = defaultdict(dict)

    for model_name in src_model_names:
        src_model_response_file = source_model_response_dir / model_name / 'all_outputs_rm.json'
        src_model_responses = read_json_file(src_model_response_file)
        for model_response in src_model_responses:
            prompt = model_response['prompt']
            src_model_response = {'all_generated_responses': model_response['all_generated_responses'], 
                                'all_rm_scores': model_response['all_rm_scores']}
            if prompt in source_model_response_dict:
                source_model_response_dict[prompt][model_name] = src_model_response
            else:
                source_model_response_dict[prompt] = {model_name: src_model_response}
    
    completion_dpo_data = read_json_file(completion_dpo_file)
    source_model_chosen_rejected_data = []
    chosen_statis = defaultdict(int)
    for data in completion_dpo_data:
        prompt = data['conversations'][0]['value']
        model_names = []
        model_responses = []
        model_responses_scores = []

        source_model_responses = source_model_response_dict[prompt]
        for model_name in source_model_responses:
            model_responses.extend(source_model_responses[model_name]['all_generated_responses'])
            model_responses_scores.extend(source_model_responses[model_name]['all_rm_scores'])
            model_names.extend([model_name] * len(source_model_responses[model_name]['all_generated_responses']))
    
        max_score_index = model_responses_scores.index(max(model_responses_scores))
        max_score_model_name = model_names[max_score_index]

        max_score_all_generated_responses = source_model_responses[max_score_model_name]['all_generated_responses']
        max_score_all_rm_scores = source_model_responses[max_score_model_name]['all_rm_scores']

        if len(set(max_score_all_generated_responses)) == 1:
            continue

        chosen_statis[max_score_model_name] += 1
        chosen_index = max_score_all_rm_scores.index(max(max_score_all_rm_scores))
        rejected_index = max_score_all_rm_scores.index(min(max_score_all_rm_scores))

        source_model_chosen_rejected = {"conversations":[{'from':'human', 'value':prompt}], 
                                        "chosen":{'from':'gpt', 'value':max_score_all_generated_responses[chosen_index]},
                                        "rejected":{'from':'gpt', 'value':max_score_all_generated_responses[rejected_index]}}

        source_model_chosen_rejected_data.append(source_model_chosen_rejected)

    print(f"completion_dpo_datasize: {len(completion_dpo_data)}, source_model_response_dpo_datasize:{len(source_model_chosen_rejected_data)}")
    print(chosen_statis)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'dpo_source_model_1.0.json'
    write_json_file(source_model_chosen_rejected_data, save_path)


def build_sft_dataset_sharegpt_format(prompt_file, src_model_output_dir, src_model_name, save_dir):
    # 从source model responses 中选出得分最高的
    prompts = read_json_file(prompt_file)
    # prompts = prompts[7446:]

    prompt_dict = defaultdict(dict)
    for prompt in prompts:
        prompt_text = prompt['prompt']
        prompt_dict[prompt_text] = prompt
    # print(len(prompt_dict))

    src_model_output_dir = Path(src_model_output_dir)
    src_model_output_dict = defaultdict(dict)

    for model_name in src_model_name:
        src_model_output_file = src_model_output_dir / model_name / 'all_outputs_rm.json'
        src_model_outputs = read_json_file(src_model_output_file)
        for model_output in src_model_outputs:
            prompt = model_output['prompt']
            src_model_output = {'all_generated_responses': model_output['all_generated_responses'], 
                                'all_rm_scores': model_output['all_rm_scores']}
            if prompt in src_model_output_dict:
                src_model_output_dict[prompt][model_name] = src_model_output
            else:
                src_model_output_dict[prompt] = {model_name: src_model_output}
            
    model_sample_counts = {}
    best_responses = []
    for prompt in prompt_dict:
        max_model_name = ''
        if prompt in src_model_output_dict:
            _best_response = defaultdict(str)
            _best_response['prompt'] = prompt
            for model_name, model_output in src_model_output_dict[prompt].items():
                all_generated_responses = model_output['all_generated_responses']
                all_rm_scores = model_output['all_rm_scores']
                
                max_rm_score = max(all_rm_scores)
                max_rm_score_index = all_rm_scores.index(max_rm_score)
                max_response = all_generated_responses[max_rm_score_index]

                if 'rm_score' in _best_response:
                    if _best_response['rm_score'] < max_rm_score:
                        _best_response['rm_score'] = max_rm_score
                        _best_response['response'] = max_response
                        max_model_name = model_name
                else:
                    _best_response = {'response':max_response, 'rm_score': max_rm_score}
                    max_model_name = model_name                    

            conversations = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': _best_response['response']}]
            best_responses.append({'conversations': conversations})

            if max_model_name in model_sample_counts:
                model_sample_counts[max_model_name] += 1
            else:
                model_sample_counts[max_model_name] = 1

    # print(model_sample_counts)
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f'dpo_{len(prompts)}.json'
    write_json_file(best_responses, output_path)


def build_chosen_sft_dataset_sharegpt_format(dpo_input_file, save_dir, iteration_index):
    # use the dpo chosen data to implement sft warm-up
    dpo_pairs = read_json_file(dpo_input_file)
    chosen_data = []
    for dpo_pair in dpo_pairs:
        prompt = dpo_pair['conversations'][0]['value']
        chosen_response = dpo_pair['chosen']['value']
        conversations = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': chosen_response}]
        chosen_data.append({'conversations': conversations})
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f'chosen_sft_iteration_{iteration_index+1}.json'
    write_json_file(chosen_data, output_path)


def build_sft_data_sharegpt(input_file, save_dir):
    # 构建用于sft训练的数据，格式满足sharegpt要求
    input_data = read_json_file(input_file)
    sft_data = []
    for data in input_data:
        prompt = data['prompt']
        best_response = data['best_response']
        conversations = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': best_response}]
        sft_data.append({'conversations': conversations})
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f'sft_{len(input_data)}.json'
    write_json_file(sft_data, output_path)


def build_completion_dpo_dataset(input_file, save_dir):
    # 构建用于dpo训练的数据，格式满足llama-factory要求
    # 1. target model response 最优时chosen，rejected从target model response中选取
    # 2. source model response 最优时，遍历不同位置的completions，在第一次completion score>target model response 位置处的多次采样中选出chosen，rejected
    
    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
    chosen_rejected_statis = []
    chosen_rejected_same_statis = []
    completion_subsentence_num_statis = []

    for data in input_data:
        target_model_responses = data['target_model_responses']
        target_model_response_scores = data['target_model_response_scores']
        max_target_model_response_score = max(target_model_response_scores)
        min_target_model_response_score = min(target_model_response_scores)
        
        chosen = ''
        rejected = ''
        
        if max_target_model_response_score < data['best_response_score']:
            completion_inputs = data['completion_inputs']
            completion_outputs = data['completion_outputs']
            completion_scores = data['completion_rm_scores']

            # for i, completion_input in enumerate(completion_inputs): 
            #     if len(set(completion_outputs[i])) == 1:
            #         chosen_rejected_same_statis.append(1)
            #         continue
                
            #     completion_scores_i = completion_scores[i]
            #     max_completion_score_i = max(completion_scores_i)
            #     min_completion_score_i = min(completion_scores_i)

            #     if max_completion_score_i > max_target_model_response_score:
            #         chosen_completion_index = completion_scores_i.index(max_completion_score_i)
            #         rejected_completion_index = completion_scores_i.index(min_completion_score_i)

            #         completion_input = completion_input.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
            #         chosen = completion_input + ' ' + completion_outputs[i][chosen_completion_index]
            #         rejected = completion_input + ' ' + completion_outputs[i][rejected_completion_index]
            #         break
            
            for i, completion_scores_i in enumerate(completion_scores): 
                if len(set(completion_outputs[i])) == 1:
                    chosen_rejected_same_statis.append(1)
                    continue
                
                max_completion_score_i = max(completion_scores_i)
                min_completion_score_i = min(completion_scores_i)

                if max_completion_score_i > max_target_model_response_score:
                    chosen_completion_index = completion_scores_i.index(max_completion_score_i)
                    rejected_completion_index = completion_scores_i.index(min_completion_score_i)

                    completion_input = completion_inputs[i].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                    chosen = completion_input + ' ' + completion_outputs[i][chosen_completion_index]
                    rejected = completion_input + ' ' + completion_outputs[i][rejected_completion_index]
                    completion_subsentence_num_statis.append(i + 1)
                    break
                    
        if not chosen:
            if len(set(data['target_model_responses'])) == 1:
                chosen_rejected_same_statis.append(0)
                continue
            else:
                max_score_index = target_model_response_scores.index(max_target_model_response_score)
                min_score_index = target_model_response_scores.index(min_target_model_response_score)
                
                chosen = target_model_responses[max_score_index]
                rejected = target_model_responses[min_score_index]
                chosen_rejected_statis.append(0)
        else:
            chosen_rejected_statis.append(1)
        
        # temp: 让rejected始终来自target model 自身采样
        # min_score_index = target_model_response_scores.index(min_target_model_response_score)
        # rejected = target_model_responses[min_score_index]

        chosen_rejected_data = {"conversations":[{'from':'human', 'value':data['prompt']}], "chosen":{'from':'gpt', 'value':chosen},
                                 "rejected":{'from':'gpt', 'value':rejected}}

        chosen_rejected_dataset.append(chosen_rejected_data)
    
    if len(input_data) != len(chosen_rejected_dataset):
        print(f"input_dataset_size:{len(input_data)}, dataset_used_size:{len(chosen_rejected_dataset)}, {len(input_data) - len(chosen_rejected_dataset)} samples filtered")
        chosen_rejected_same_from_completion = sum(chosen_rejected_same_statis) / len(chosen_rejected_same_statis)
        print(f"{len(chosen_rejected_same_statis)} samples chosen-rejected same, {chosen_rejected_same_from_completion:.2f} from completion.")
    
    data_from_completion_ratio = sum(chosen_rejected_statis) / len(chosen_rejected_statis)
    print(f"{data_from_completion_ratio:.2f} chosen-rejected from completion data")
    print(f"completion_subsentence_num_statis: avg-{sum(completion_subsentence_num_statis)/len(completion_subsentence_num_statis)}")
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'dpo_completion_{data_from_completion_ratio:.2f}_target_model_{1-data_from_completion_ratio:.2f}_mix.json'
    write_json_file(chosen_rejected_dataset, save_path)


def build_completion_dpo_dataset_v2(input_file, save_dir, iteration_index):
    # 构建用于dpo训练的数据，格式满足llama-factory要求
    # 1. target model response 最优时chosen，rejected从target model response中选取
    # 2. source model response 最优时，遍历不同位置的completions，在第一次completion score>target model response 位置处的多次采样中选出chosen，rejected
    
    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
    chosen_rejected_statis = []

    for data in input_data:
        chosen = ''
        rejected = ''
        
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']

        for i, completion_scores_i in enumerate(completion_scores): 
            if len(set(completion_outputs[i])) == 1:
                continue
            
            max_completion_score_i = max(completion_scores_i)
            min_completion_score_i = min(completion_scores_i)

            chosen_completion_index = completion_scores_i.index(max_completion_score_i)
            rejected_completion_index = completion_scores_i.index(min_completion_score_i)

            completion_input = completion_inputs[i].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
            chosen = completion_input + ' ' + completion_outputs[i][chosen_completion_index]
            rejected = completion_input + ' ' + completion_outputs[i][rejected_completion_index]
            break
                    
        chosen_rejected_data = {"conversations":[{'from':'human', 'value':data['prompt']}], "chosen":{'from':'gpt', 'value':chosen},
                                 "rejected":{'from':'gpt', 'value':rejected}}

        chosen_rejected_dataset.append(chosen_rejected_data)
    
    if len(input_data) != len(chosen_rejected_dataset):
        print(f"input_dataset_size:{len(input_data)}, dataset_used_size:{len(chosen_rejected_dataset)}, {len(input_data) - len(chosen_rejected_dataset)} samples filtered")
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'dpo_completion_iter{iteration_index+1}.json'
    write_json_file(chosen_rejected_dataset, save_path)


def build_completion_dpo_dataset_v3(input_file, save_dir, iteration_index):
    # 构建用于dpo训练的数据，格式满足llama-factory要求
    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
    chosen_rejected_statis = []

    for data in input_data:
        chosen = ''
        rejected = ''
        
        best_response = data['best_response']
        best_response_score = data['best_response_score']

        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']

        if len(completion_inputs) == 2:
            assert completion_inputs[0] in completion_inputs[1]
            more_completion_input_max_score = max(completion_scores[1])
            less_completion_input_max_score = max(completion_scores[0])

            if more_completion_input_max_score == less_completion_input_max_score:
                chosen_rejected_statis.append(0)
                continue

                # completion_output = completion_outputs[1][completion_scores[1].index(more_completion_input_max_score)]
                # if more_completion_input_max_score == best_response_score:
                #     chosen_rejected_statis.append(4)
                #     continue
                
                # chosen_rejected_statis.append(3)
                # if more_completion_input_max_score > best_response_score:
                #     chosen = completion_output
                #     rejected = best_response  
                # else:
                #     rejected = completion_output
                #     chosen = best_response  

            else:
                more_completion_input = completion_inputs[1].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                less_completion_input = completion_inputs[0].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]

                more_completion_input_max_output = completion_outputs[1][completion_scores[1].index(more_completion_input_max_score)]
                less_completion_input_max_output = completion_outputs[0][completion_scores[0].index(less_completion_input_max_score)]

                if more_completion_input_max_score > less_completion_input_max_score:
                    chosen = more_completion_input + ' ' + more_completion_input_max_output
                    rejected = less_completion_input + ' ' + less_completion_input_max_output
                    chosen_rejected_statis.append(1)
                else:
                    rejected = more_completion_input + ' ' + more_completion_input_max_output
                    chosen = less_completion_input + ' ' + less_completion_input_max_output
                    chosen_rejected_statis.append(2)

        elif len(completion_inputs) == 1:
            max_completion_score = max(completion_scores[0])
            max_completion_output = completion_outputs[0][completion_scores[0].index(max_completion_score)]
            completion_input = completion_inputs[0].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]

            if max_completion_score == best_response_score:
                chosen_rejected_statis.append(4)
                continue

            chosen_rejected_statis.append(3)
            if max_completion_score > best_response_score:
                chosen = completion_input + ' ' + max_completion_output
                rejected = best_response
            else:
                rejected = completion_input + ' ' + max_completion_output
                chosen = best_response

        else:
            raise ValueError(f"invalid subsent num: {len(completion_inputs)}")
                    
        chosen_rejected_data = {"conversations":[{'from':'human', 'value':data['prompt']}], "chosen":{'from':'gpt', 'value':chosen},
                                 "rejected":{'from':'gpt', 'value':rejected}}

        chosen_rejected_dataset.append(chosen_rejected_data)
    
    chosen_rejected_statis = Counter(chosen_rejected_statis)
    print(f"chosen_rejected_dataset size: {len(chosen_rejected_dataset)}\n"
    f"adjacent_completion_score_equal: {chosen_rejected_statis[0]}\n"
    f"dpo pair from adjacent_completion: {chosen_rejected_statis[1] + chosen_rejected_statis[2]}\n"
    f"   chosen from the latter position: {chosen_rejected_statis[1]}\n"
    f"   chosen from the previous position: {chosen_rejected_statis[2]}\n"
    f"dpo pair from source and completion: {chosen_rejected_statis[3]}\n"
    f"sample filtered: {chosen_rejected_statis[4]}\n")
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'dpo_completion_adjacent_iter{iteration_index+1}_v2.json'
    write_json_file(chosen_rejected_dataset, save_path)


def build_completion_dpo_dataset_v4(input_file, save_dir, iteration_index, target_response_file):
    if not iteration_index:
        assert target_response_file
        target_model_responses = read_json_file(target_response_file)
        target_model_response_score = defaultdict(dict)
        for data in target_model_responses:
            prompt = data['prompt']
            all_target_response_scores = data['all_rm_scores']
            target_model_response_score[prompt] = all_target_response_scores

    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
    chosen_rejected_statis = []
    wo_better_adv = 0 # 剩下的子句中不存在能够提供positive advantage的子句
    forward_step_statis = [] # 统计前向遍历多少位才满足的adv条件
    for data in input_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']
        completion_inputs_index = data['completion_inputs_index']

        if not iteration_index:
            all_target_response_scores = target_model_response_score[prompt]
            last_state_value = sum(all_target_response_scores) / len(all_target_response_scores)
        else:
            last_state_value = sum(completion_scores[0]) / len(completion_scores[0])

        chosen = ''
        rejected = ''
        for i, completion_scores_i in enumerate(completion_scores): 
            if iteration_index and i == 0: # 如果不是第一轮迭代，就从completion input的第二位开始判断，因为此时第一位是作为last state
                continue

            if len(set(completion_outputs[i])) == 1:
                continue
            
            current_action_value = sum(completion_scores_i) / len(completion_scores_i)
            if current_action_value >= last_state_value:
                max_completion_score_i = max(completion_scores_i)
                min_completion_score_i = min(completion_scores_i)

                chosen_completion_index = completion_scores_i.index(max_completion_score_i)
                rejected_completion_index = completion_scores_i.index(min_completion_score_i)

                completion_input = completion_inputs[i].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                chosen = completion_input + ' ' + completion_outputs[i][chosen_completion_index]
                rejected = completion_input + ' ' + completion_outputs[i][rejected_completion_index]

                data[f'iter_{iteration_index+1}_completion_index'] = completion_inputs_index[i]
                forward_step_statis.append(i)
                break
        
        if not chosen:
            wo_better_adv += 1
            # continue
            if len(set(completion_outputs[0])) == 1:
                continue

            max_completion_score = max(completion_scores[0])
            min_completion_score = min(completion_scores[0])

            chosen_completion_index = completion_scores[0].index(max_completion_score)
            rejected_completion_index = completion_scores[0].index(min_completion_score)

            completion_input = completion_inputs[0].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
            chosen = completion_input + ' ' + completion_outputs[0][chosen_completion_index]
            rejected = completion_input + ' ' + completion_outputs[0][rejected_completion_index]

        chosen_rejected_data = {"conversations":[{'from':'human', 'value':data['prompt']}], "chosen":{'from':'gpt', 'value':chosen},
                                 "rejected":{'from':'gpt', 'value':rejected}}
        chosen_rejected_dataset.append(chosen_rejected_data)

    if len(input_data) != len(chosen_rejected_dataset):
        print(f"input_dataset_size:{len(input_data)}, dataset_used_size:{len(chosen_rejected_dataset)}\n"
        f"{len(input_data) - len(chosen_rejected_dataset)} samples filtered\n"
        f"{wo_better_adv} without positive advantage")
    
    forward_step_statis = Counter(forward_step_statis)
    print(f"forward_step_statis: {dict(forward_step_statis)}")

    # save_dir = Path(save_dir)
    # if not save_dir.exists():
    #     save_dir.mkdir(parents=True, exist_ok=True)
    # save_path = save_dir / f'dpo_completion_adv_iter{iteration_index+1}_v2.json'
    # write_json_file(chosen_rejected_dataset, save_path)

    # # update completion index used in current iteration 
    # write_json_file(input_data, input_file)


def build_completion_dpo_dataset_v4_2(input_file, save_dir, iteration_index, target_response_file):
    if not iteration_index:
        assert target_response_file
        target_model_responses = read_json_file(target_response_file)
        target_model_response_score = defaultdict(dict)
        for data in target_model_responses:
            prompt = data['prompt']
            all_target_response_scores = data['all_rm_scores']
            all_target_responses = data['all_generated_responses']
            target_model_response_score[prompt] = {'rm_scores': all_target_response_scores, 'responses': all_target_responses}

    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
    chosen_rejected_statis = []
    wo_better_adv = 0 # adv <= 0
    strange_sample_num = 0

    for data in input_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']
        completion_inputs_index = data['completion_inputs_index']

        if not iteration_index:
            all_target_response_scores = target_model_response_score[prompt]['rm_scores']
            last_state_value = sum(all_target_response_scores) / len(all_target_response_scores)
            # last_state_value = max(all_target_response_scores)
            last_state_completions = target_model_response_score[prompt]['responses']
            last_state_completion_scores = all_target_response_scores
        else:
            last_state_value = sum(completion_scores[0]) / len(completion_scores[0])
            last_state_completions = completion_outputs[0]
            last_state_completion_scores = completion_scores[0]
        last_state_completion_input = completion_inputs[0]
       
        chosen = ''
        rejected = ''
        for i, completion_scores_i in enumerate(completion_scores): 
            if iteration_index and i == 0: # 如果不是第一轮迭代，就从completion input的第二位开始判断，因为此时第一位是作为last state
                continue

            if len(set(completion_outputs[i])) == 1:
                continue
            
            current_action_value = sum(completion_scores_i) / len(completion_scores_i)
            # current_action_value = max(completion_scores_i)

            if current_action_value > last_state_value:
                max_completion_score_i = max(completion_scores_i)
                min_completion_score_i = min(completion_scores_i)

                chosen_completion_index = completion_scores_i.index(max_completion_score_i)
                rejected_completion_index = completion_scores_i.index(min_completion_score_i)

                chosen = completion_outputs[i][chosen_completion_index]
                rejected = completion_outputs[i][rejected_completion_index]

                if "<|start_header_id|>assistant<|end_header_id|>\n\n" in completion_inputs[i]:
                    completion_input = completion_inputs[i].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                    if completion_input:
                        chosen = completion_input + ' ' + chosen
                        rejected = completion_input + ' ' + rejected
                    else:
                        strange_sample_num += 1

                data[f'iter_{iteration_index+1}_completion_index'] = completion_inputs_index[i]
                break
            
        if not chosen:
            wo_better_adv += 1
            continue

            if len(set(last_state_completions)) == 1:
                continue

            max_completion_score = max(last_state_completion_scores)
            min_completion_score = min(last_state_completion_scores)

            chosen_completion_index = last_state_completion_scores.index(max_completion_score)
            rejected_completion_index = last_state_completion_scores.index(min_completion_score)

            chosen = last_state_completions[chosen_completion_index]
            rejected = last_state_completions[rejected_completion_index]

            if "<|start_header_id|>assistant<|end_header_id|>\n\n" in last_state_completion_input:
                completion_input = last_state_completion_input.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                if completion_input:
                    chosen = completion_input + ' ' + chosen
                    rejected = completion_input + ' ' + rejected
            
            if not iteration_index:
                data[f'iter_{iteration_index+1}_completion_index'] = -1
            else:
                data[f'iter_{iteration_index+1}_completion_index'] = completion_inputs_index[0]

        chosen_rejected_data = {"conversations":[{'from':'human', 'value':data['prompt']}], "chosen":{'from':'gpt', 'value':chosen},
                                 "rejected":{'from':'gpt', 'value':rejected}}
        chosen_rejected_dataset.append(chosen_rejected_data)

    print(f"{wo_better_adv} without positive advantage")
    print(f"strange_sample_num: {strange_sample_num}")
    if len(input_data) != len(chosen_rejected_dataset):
        print(f"input_dataset_size:{len(input_data)}, dataset_used_size:{len(chosen_rejected_dataset)}\n"
        f"{len(input_data) - len(chosen_rejected_dataset)} samples filtered\n")
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'dpo_completion_adv_iter{iteration_index+1}_v3.json'
    write_json_file(chosen_rejected_dataset, save_path)

    # update completion index used in current iteration 
    write_json_file(input_data, input_file)


def build_completion_dpo_dataset_v5(input_file, save_dir, iteration_index, rank_file):
    # 构建用于dpo训练的数据，格式满足llama-factory要求
    # 1. target model response 最优时chosen，rejected从target model response中选取
    # 2. source model response 最优时，遍历不同位置的completions，在第一次completion score>target model response 位置处的多次采样中选出chosen，rejected
    with open(rank_file, 'r', encoding='utf-8') as file:
        completion_rank_dict = json.load(file)

    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
    chosen_rejected_statis = []
    for data in input_data:
        chosen = ''
        rejected = ''
        
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']
        
        if not iteration_index:
            completion_inputs_rank = completion_rank_dict[prompt]
            completion_input_index = completion_inputs_rank[0]
        else:
            assert len(completion_scores) == 1
            completion_input_index = 0

        completion_scores_i = completion_scores[completion_input_index]
        completion_outputs_i = completion_outputs[completion_input_index]

        if len(set(completion_outputs_i)) == 1:
            continue
        
        max_completion_score_i = max(completion_scores_i)
        min_completion_score_i = min(completion_scores_i)

        chosen_completion_index = completion_scores_i.index(max_completion_score_i)
        rejected_completion_index = completion_scores_i.index(min_completion_score_i)

        completion_input = completion_inputs[completion_input_index].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
        chosen = completion_input + ' ' + completion_outputs_i[chosen_completion_index]
        rejected = completion_input + ' ' + completion_outputs_i[rejected_completion_index]
                    
        chosen_rejected_data = {"conversations":[{'from':'human', 'value':data['prompt']}], "chosen":{'from':'gpt', 'value':chosen},
                                 "rejected":{'from':'gpt', 'value':rejected}}

        chosen_rejected_dataset.append(chosen_rejected_data)
    
    if len(input_data) != len(chosen_rejected_dataset):
        print(f"input_dataset_size:{len(input_data)}, dataset_used_size:{len(chosen_rejected_dataset)}, {len(input_data) - len(chosen_rejected_dataset)} samples filtered")
    
    # completion_input_index_statis = Counter(completion_input_index_statis)
    # print(completion_input_index_statis)
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'dpo_completion_adv_iter{iteration_index+1}_test.json'
    write_json_file(chosen_rejected_dataset, save_path)


def rank_subsent_by_adv(input_file, save_file, target_response_file):
    # source model response 中不同子句对target model重要性不同，利用adv代表重要性，按照重要性大小排序遍历所有子句
    target_model_responses = read_json_file(target_response_file)
    target_model_response_score = defaultdict(dict)
    for data in target_model_responses:
        prompt = data['prompt']
        all_target_response_scores = data['all_rm_scores']
        target_model_response_score[prompt] = all_target_response_scores

    rank_by_adv = defaultdict(list)
    input_data = read_json_file(input_file)
    for data in input_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']
        completion_inputs_index = data['completion_inputs_index']

        all_target_response_scores = target_model_response_score[prompt]
        target_response_avg_score = sum(all_target_response_scores) / len(all_target_response_scores)
        adv_list = []
        completion_avg_score = 0
        for i, completion_score in enumerate(completion_scores):
            if i == 0:
                completion_avg_score = sum(completion_score) / len(completion_score)
                adv = completion_avg_score - target_response_avg_score
            else:
                completion_avg_score_i = sum(completion_score) / len(completion_score)
                adv = completion_avg_score_i - completion_avg_score
                completion_avg_score = completion_avg_score_i
            adv_list.append(adv)
        
        paired_list = list(zip(completion_inputs_index, adv_list))
        sorted_paired_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
        sorted_completion_list = [item[0] for item in sorted_paired_list]

        rank_by_adv[prompt] = sorted_completion_list
    
    write_json_file(rank_by_adv, save_file)


def source_response_filtering(source_model_response_file, source_model_names, min_steps_threshold, save_dir):
    # filtering standard: min(source model response steps) > threshold
    source_model_responses = read_json_file(source_model_response_file)
    print(f"source model response size: {len(source_model_responses)}")

    pattern = r'(.*?\n+)'

    valid_data = []
    source_model_statis = []
    for data in source_model_responses:
        valid_source_model_responses = []
        source_model_used = []
        for model_name in source_model_names:
            if model_name in data:
                valid_source_model_responses.extend(data[model_name]['response'])
                source_model_used.append(model_name)
        # source_model_statis.extend(source_model_used)

        valid_source_model_responses_steps = []
        for response in valid_source_model_responses:
            response_steps = re.findall(pattern, response)
            # print(response_steps)
            len_response_steps = len(response_steps)
            valid_source_model_responses_steps.append(len_response_steps)

        if min(valid_source_model_responses_steps) > min_steps_threshold:
            valid_data.append(data)
            source_model_statis.extend(source_model_used)
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'source_model_response_filtered_{len(valid_data)}_over_{min_steps_threshold}_steps.json'
    write_json_file(valid_data, save_path)
    print(f"source model response filtered: {len(valid_data)} over {min_steps_threshold} steps")
    source_model_statis = Counter(source_model_statis)
    print(f"source model used: {source_model_statis}")


def data_select_based_on_adv_score(completion_file, source_response_file, target_model_file, save_dir):
    # 选择adv score最高的response
    source_model_responses = read_json_file(source_response_file)
    source_model_responses_dict = defaultdict(dict)
    for data in source_model_responses:
        prompt = data['prompt']
        source_model_responses_dict[prompt] = data

    target_model_responses = read_json_file(target_model_file)
    target_model_response_score = defaultdict(dict)
    for data in target_model_responses:
        prompt = data['prompt']
        all_target_response_scores = data['all_rm_scores']
        all_target_responses = data['all_generated_responses']
        target_model_response_score[prompt] = {"response": all_target_responses, "rm_score": all_target_response_scores}
    
    completion_data = read_json_file(completion_file)
    best_rm_score_adv_score_diff_statis = 0
    rm_best_response_distrib_statis = defaultdict(int)
    adv_best_response_distrib_statis = defaultdict(int)

    conversations = []
    for data in completion_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        completion_scores = data['completion_rm_scores']
        completion_ids = data['completion_ids']
        best_response = data['best_response']

        _target_model_response_score = target_model_response_score[prompt]['rm_score']
        avg_target_model_response_score = sum(_target_model_response_score) / len(_target_model_response_score)
        
        completion_dict = defaultdict(list)
        previous_completion_id = completion_ids[0]
        avg_completion_scores = []
        completion_num = 1
        for i, current_completion_id in enumerate(completion_ids):
            model_name = current_completion_id.split('_')[0]

            if previous_completion_id != current_completion_id:
                previous_completion_id = current_completion_id
                completion_dict[model_name].append(avg_completion_scores)
                avg_completion_scores = []
                completion_num += 1

            avg_completion_score = sum(completion_scores[i]) / len(completion_scores[i])
            avg_completion_scores.append(avg_completion_score)
        assert completion_num == 4, print(completion_num, completion_ids)

        source_adv_scores = defaultdict(list)
        max_adv_score = -1000
        max_adv_score_index = -1
        max_adv_model = ''
        for model_name, completions_scores in completion_dict.items():
            adv_scores = []
            for completion_scores in completions_scores:
                completion_scores_array = np.array(completion_scores)
                # baseline_scores_array = np.array([avg_target_model_response_score] + completion_scores[:-1])
                # adv_step_scores = completion_scores_array - baseline_scores_array
                # adv_score = adv_step_scores.mean().item()
                # adv_scores.append(adv_score)

                # max return
                # adv_score = completion_scores_array.mean().item()
                # adv_scores.append(adv_score)

                # min Q
                # adv_score = completion_scores_array.min().item()
                # adv_scores.append(adv_score)

                # max Q
                adv_score = completion_scores_array.max().item()
                adv_scores.append(adv_score)
    
            source_adv_scores[model_name] = adv_scores
            
            _max_adv_score = max(adv_scores)
            if _max_adv_score > max_adv_score:
                max_adv_score = _max_adv_score
                max_adv_score_index = adv_scores.index(max_adv_score)
                max_adv_model = model_name

        selected_response = source_model_responses_dict[prompt][max_adv_model]['response'][max_adv_score_index]
        conversation = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': selected_response}]
        conversations.append({'conversations': conversation})

        # check the RM-best response belong to which source model
        source_model_names = ['deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 'Qwen2.5-72B-Instruct']
        rm_best_source_model = ''
        for source_model in source_model_names:
            if source_model in source_model_responses_dict[prompt]:
                if best_response in source_model_responses_dict[prompt][source_model]['response']:
                    rm_best_source_model = source_model
                    break
        assert rm_best_source_model
        rm_best_response_distrib_statis[rm_best_source_model] += 1
        adv_best_response_distrib_statis[max_adv_model] += 1

        if selected_response != best_response:
            best_rm_score_adv_score_diff_statis += 1

    print(f"best_rm_score_adv_score_diff_statis: {best_rm_score_adv_score_diff_statis}")
    print(f"rm_best_response_distrib_statis:{rm_best_response_distrib_statis}\nadv_best_response_distrib_statis:{adv_best_response_distrib_statis}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'sft_{len(conversations)}_max_q.json'
    write_json_file(conversations, save_path)


def build_best_response_sft_subset_tmp(input_file, save_dir):
    # 按照v0226使用的前10000条数据中best rm score对应的best response构建sft数据集
    input_data = read_json_file(input_file)
    conversations = []

    for data in input_data[:5000]:
        prompt = data['prompt']
        best_response = data['best_response']
        conversation = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': best_response}]
        conversations.append({'conversations': conversation})

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'sft_rm_best_{len(conversations)}.json'
    write_json_file(conversations, save_path)


def concat_best_adv_sft_data(file_list, save_path):
    # 将多个sft数据集合并
    conversations = []
    for file in file_list:
        data = read_json_file(file)
        conversations.extend(data)
    
    write_json_file(conversations, save_path)


def create_step_weighted_dataset_v2(input_file, target_model_response_file, save_dir):
    with open(target_model_response_file, 'r', encoding='utf-8') as file:
        target_model_response_verified = json.load(file)

    target_model_response_score_dict = defaultdict(float)
    target_model_zero_score_count = 0
    target_model_one_score_count = 0
    for data in target_model_response_verified:
        prompt = data['prompt']
        rule_scores = data['rule_scores']
        all_generated_responses = data['all_generated_responses']
        target_model_response_score_dict[prompt] = sum(rule_scores) / len(rule_scores)
        if sum(rule_scores) / len(rule_scores) == 0:
            target_model_zero_score_count += 1
        elif sum(rule_scores) / len(rule_scores) == 1:
            target_model_one_score_count += 1
    print(f"target_model_zero_score_count: {target_model_zero_score_count}, target_model_one_score_count: {target_model_one_score_count}")

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"raw data size: {len(data)}")

    first_rollout_allright_index_statis = []
    conversations = []
    for item in data:
        prompt = item['prompt']
        response = item['answer']
        rule_scores = item['rule_scores']

        assert prompt in target_model_response_score_dict
        baseline_score = target_model_response_score_dict[prompt]
        if baseline_score == 1:
            continue

        step_scores = [1-baseline_score]
        for rule_scores_i in rule_scores:
            step_score = 1 - (sum(rule_scores_i) / len(rule_scores_i))
            step_scores.append(step_score)

        if 0 in step_scores:
            first_rollout_allright_index_statis.append(step_scores.index(0))

        conversation = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': response}]
        conversations.append({'conversations': conversation, 'step_scores': step_scores})

    first_rollout_allright_index_statis = Counter(first_rollout_allright_index_statis)
    print(f"first_rollout_allright_index_statis: {dict(first_rollout_allright_index_statis)}, total: {sum(first_rollout_allright_index_statis.values())}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'openmathinstruct2_validation_with_step_score_v2.json'
    write_json_file(conversations, save_path)


def create_step_weighted_dataset_v3(input_file, target_model_response_file, save_dir):
    random.seed(10)

    with open(target_model_response_file, 'r', encoding='utf-8') as file:
        target_model_response_verified = json.load(file)

    target_model_response_score_dict = defaultdict(float)
    target_model_zero_score_count = 0
    target_model_one_score_count = 0
    for data in target_model_response_verified:
        prompt = data['prompt']
        rule_scores = data['rule_scores']
        all_generated_responses = data['all_generated_responses']
        target_model_response_score_dict[prompt] = sum(rule_scores) / len(rule_scores)
        
        if sum(rule_scores) / len(rule_scores) == 0:
            target_model_zero_score_count += 1
        elif sum(rule_scores) / len(rule_scores) == 1:
            target_model_one_score_count += 1
    print(f"target_model_zero_score_count: {target_model_zero_score_count}, target_model_one_score_count: {target_model_one_score_count}")

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"raw data size: {len(data)}")

    first_rollout_allright_index_statis = []
    conversations = []
    for item in data:
        prompt = item['prompt']
        response = item['answer']
        rule_scores = item['rule_scores']
        completion_inputs = item['completion_inputs']
        completion_outputs = item['completion_outputs']

        assert prompt in target_model_response_score_dict
        baseline_score = target_model_response_score_dict[prompt]
        if baseline_score == 1:
            continue

        step_scores = [1-baseline_score]
        for i, rule_scores_i in enumerate(rule_scores):
            step_score = 1 - (sum(rule_scores_i) / len(rule_scores_i))
            if step_score == 0:
                completion_inputs_i = completion_inputs[i].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                completion_outputs_i = random.choice(completion_outputs[i])
                response = completion_inputs_i + completion_outputs_i
                break

            step_scores.append(step_score)

        if 0 in step_scores:
            first_rollout_allright_index_statis.append(step_scores.index(0))

        conversation = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': response}]
        conversations.append({'conversations': conversation, 'step_scores': step_scores})

    first_rollout_allright_index_statis = Counter(first_rollout_allright_index_statis)
    print(f"first_rollout_allright_index_statis: {dict(first_rollout_allright_index_statis)}, total: {sum(first_rollout_allright_index_statis.values())}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'openmathinstruct2_validation_with_step_score_v3.json'
    write_json_file(conversations, save_path)


def create_adv_based_completion_dpo_dataset(completion_file, target_model_response_file, save_dir, iteration_time, iteration_rule):
    random.seed(10)

    with open(target_model_response_file, 'r', encoding='utf-8') as file:
        target_model_response_verified = json.load(file)

    target_model_response_score_dict = defaultdict(float)
    target_model_zero_score_count = 0
    target_model_one_score_count = 0
    for data in target_model_response_verified:
        prompt = data['prompt']
        rule_scores = data['rule_scores']
        all_generated_responses = data['all_generated_responses']
        data['baseline_score'] = sum(rule_scores) / len(rule_scores)
        target_model_response_score_dict[prompt] = data
        
        if sum(rule_scores) / len(rule_scores) == 0:
            target_model_zero_score_count += 1
        elif sum(rule_scores) / len(rule_scores) == 1:
            target_model_one_score_count += 1
    print(f"target_model_zero_score_count: {target_model_zero_score_count}, target_model_one_score_count: {target_model_one_score_count}")

    with open(completion_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"raw data size: {len(data)}")

    step_index_rollout_acc = defaultdict(list)
    step_index_rollout_adv = defaultdict(list)

    top_k_scores_less_than_or_equal_to_0_count = 0
    conversations = defaultdict(list)
    for item in data:
        prompt = item['prompt']
        response = item['answer']
        rule_scores = item['rule_scores']
        completion_inputs = item['completion_inputs']
        completion_outputs = item['completion_outputs']

        assert prompt in target_model_response_score_dict
        baseline_score = target_model_response_score_dict[prompt]['baseline_score']
        if baseline_score == 1:
            continue

        step_scores = [sum(rule_scores_i) / len(rule_scores_i) for rule_scores_i in rule_scores]
        step_adv_scores = np.array(step_scores) - np.array([baseline_score] + step_scores[:-1])

        if iteration_rule == 'sorted_by_adv_score':
            top_k_indices = np.argsort(step_adv_scores)[-iteration_time:][::-1]
        elif iteration_rule == 'sorted_by_step_index':
            top_k_indices = np.argsort(step_adv_scores)[-iteration_time:]
            top_k_indices = np.sort(top_k_indices)[::-1]

        top_k_scores = step_adv_scores[top_k_indices]

        if np.all(top_k_scores <= 0):
            top_k_scores_less_than_or_equal_to_0_count += 1
            continue    

        for i, indice in enumerate(top_k_indices):
            if top_k_scores[i] <= 0:
                continue
            
            step_index_rollout_acc[i].append(step_scores[indice - 1])
            step_index_rollout_adv[i].append(top_k_scores[i])

            completion_input_segments = completion_inputs[indice].split("\n\n")
            assert not completion_input_segments[-1]
            completion_input = "\n\n".join(completion_input_segments[:-2]) + '\n\n'

            # completion_input_segments = completion_inputs[indice].split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            # completion_input = completion_input_segments[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"

            chosen_candidates = []
            completion_scores = rule_scores[indice]
            for j, completion in enumerate(completion_outputs[indice]):
                if completion_scores[j] == 1:
                    chosen_candidates.append(completion)

            chosen_text = completion_input_segments[-2] + '\n\n' + random.choice(chosen_candidates)
            # chosen_text = completion_input_segments[-1] + random.choice(chosen_candidates)
            # chosen_text = completion_input_segments[-2] + '\n\n'

            rejected_candidates = []
            if indice == 0:
                for k, response in enumerate(target_model_response_score_dict[prompt]['all_generated_responses']):
                    if target_model_response_score_dict[prompt]['rule_scores'][k] == 0:
                        rejected_candidates.append(response)
            else:
                # completion_input_segments = completion_inputs[indice-1].split("<|start_header_id|>assistant<|end_header_id|>\n\n")

                completion_scores = rule_scores[indice-1]
                for j, completion in enumerate(completion_outputs[indice-1]):
                    if completion_scores[j] == 0:
                        rejected_candidates.append(completion)
                        # rejected_candidates.append(completion_input_segments[-1] + completion)
                            
            rejected_text = random.choice(rejected_candidates)

            # rejected_text_segs = rejected_text.split('\n\n')
            # if len(rejected_text_segs) == 1:
            #     rejected_text = rejected_text_segs[0] + '<|eot_id|>'
            # else:
            #     rejected_text = rejected_text_segs[0] + '\n\n'

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], "chosen":{'from':'gpt', 'value': chosen_text},
                                    "rejected":{'from':'gpt', 'value': rejected_text}}
            conversations[i].append(chosen_rejected_data)
 
        # 是否需要根据target model avg acc确定completion pair的下限 ？

    print(f"top_k_scores_less_than_or_equal_to_0_count: {top_k_scores_less_than_or_equal_to_0_count}")
    
    for key in step_index_rollout_acc:
        step_index_rollout_acc[key] = np.mean(step_index_rollout_acc[key])
        step_index_rollout_adv[key] = np.mean(step_index_rollout_adv[key])
    print(f"step_index_rollout_acc: {step_index_rollout_acc}")
    print(f"step_index_rollout_adv: {step_index_rollout_adv}")

    # save_dir = Path(save_dir)
    # if not save_dir.exists():
    #     save_dir.mkdir(parents=True, exist_ok=True)

    # for i in range(iteration_time):
    #     save_path = save_dir / f'{i}.json'
    #     print(f"iteration_{i} size: {len(conversations[i])}")
    #     write_json_file(conversations[i], save_path)


def create_completion_dpo_dataset(completion_file, target_model_response_file, save_dir, iteration_time, iteration_rule):
    random.seed(10)

    with open(target_model_response_file, 'r', encoding='utf-8') as file:
        target_model_response_verified = json.load(file)

    target_model_response_score_dict = defaultdict(float)
    target_model_zero_score_count = 0
    target_model_one_score_count = 0
    for data in target_model_response_verified:
        prompt = data['prompt']
        rule_scores = data['rule_scores']
        all_generated_responses = data['all_generated_responses']
        data['baseline_score'] = sum(rule_scores) / len(rule_scores)
        target_model_response_score_dict[prompt] = data
        
        if sum(rule_scores) / len(rule_scores) == 0:
            target_model_zero_score_count += 1
        elif sum(rule_scores) / len(rule_scores) == 1:
            target_model_one_score_count += 1
    print(f"target_model_zero_score_count: {target_model_zero_score_count}, target_model_one_score_count: {target_model_one_score_count}")

    with open(completion_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"raw data size: {len(data)}")

    top_k_scores_less_than_or_equal_to_0_count = 0
    conversations = defaultdict(list)
    for item in data:
        prompt = item['prompt']
        response = item['answer']
        rule_scores = item['rule_scores']
        completion_inputs = item['completion_inputs']
        completion_outputs = item['completion_outputs']

        assert prompt in target_model_response_score_dict
        baseline_score = target_model_response_score_dict[prompt]['baseline_score']
        if baseline_score == 1:
            continue

        step_scores = [sum(rule_scores_i) / len(rule_scores_i) for rule_scores_i in rule_scores]
        step_adv_scores = np.array(step_scores) - np.array([baseline_score] + step_scores[:-1])

        if iteration_rule == 'sorted_by_adv_score':
            top_k_indices = np.argsort(step_adv_scores)[-iteration_time:][::-1]
        elif iteration_rule == 'sorted_by_step_index':
            top_k_indices = np.argsort(step_adv_scores)[-iteration_time:]
            top_k_indices = np.sort(top_k_indices)[::-1]
        elif iteration_rule == 'equal_partitioning':
            step_num = len(step_scores)
            base_size = step_num // iteration_time
            remainder = step_num % iteration_time

            top_k_indices = []
            top_k_indice = 0
            for i in range(iteration_time):
                current_size = base_size + (1 if i < remainder else 0)
                top_k_indice += current_size
                top_k_indices.append(top_k_indice - 1)  # 切分点是当前份的“结尾索引”（非包含）

        # top_k_scores = step_adv_scores[top_k_indices]

        # if np.all(top_k_scores <= 0):
        #     top_k_scores_less_than_or_equal_to_0_count += 1
        #     continue    

        for i, indice in enumerate(top_k_indices):
            # if top_k_scores[i] <= 0:
            #     continue
            
            # completion_input_segments = completion_inputs[indice].split("\n\n")
            # assert not completion_input_segments[-1]
            # completion_input = "\n\n".join(completion_input_segments[:-2]) + '\n\n'

            completion_input = completion_inputs[indice]

            # completion_input_segments = completion_inputs[indice].split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            # completion_input = completion_input_segments[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            rejected_candidates = []

            chosen_candidates = []
            completion_scores = rule_scores[indice]
            for j, completion in enumerate(completion_outputs[indice]):
                if completion_scores[j] == 1:
                    chosen_candidates.append(completion)
                else:
                    rejected_candidates.append(completion)

            # chosen_text = completion_input_segments[-2] + '\n\n' + random.choice(chosen_candidates)
            # chosen_text = completion_input_segments[-1] + random.choice(chosen_candidates)
            # chosen_text = completion_input_segments[-2] + '\n\n'

            if not rejected_candidates or not chosen_candidates:
                continue

            chosen_text = random.choice(chosen_candidates)
            rejected_text = random.choice(rejected_candidates)

            # rejected_candidates = []
            # if indice == 0:
            #     for k, response in enumerate(target_model_response_score_dict[prompt]['all_generated_responses']):
            #         if target_model_response_score_dict[prompt]['rule_scores'][k] == 0:
            #             rejected_candidates.append(response)
            # else:
            #     # completion_input_segments = completion_inputs[indice-1].split("<|start_header_id|>assistant<|end_header_id|>\n\n")

            #     completion_scores = rule_scores[indice-1]
            #     for j, completion in enumerate(completion_outputs[indice-1]):
            #         if completion_scores[j] == 0:
            #             rejected_candidates.append(completion)
            #             # rejected_candidates.append(completion_input_segments[-1] + completion)
                            
            # rejected_text = random.choice(rejected_candidates)

            # rejected_text_segs = rejected_text.split('\n\n')
            # if len(rejected_text_segs) == 1:
            #     rejected_text = rejected_text_segs[0] + '<|eot_id|>'
            # else:
            #     rejected_text = rejected_text_segs[0] + '\n\n'

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], "chosen":{'from':'gpt', 'value': chosen_text},
                                    "rejected":{'from':'gpt', 'value': rejected_text}}
            conversations[i].append(chosen_rejected_data)
 
        # 是否需要根据target model avg acc确定completion pair的下限 ？

    print(f"top_k_scores_less_than_or_equal_to_0_count: {top_k_scores_less_than_or_equal_to_0_count}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(iteration_time):
        save_path = save_dir / f'{i}.json'
        print(f"iteration_{i} size: {len(conversations[i])}")
        write_json_file(conversations[i], save_path)


def merge_iteration_data(input_dir, output_file):
    # 将多个迭代数据集合并
    conversations = []
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            data = read_json_file(os.path.join(input_dir, file))
            conversations.extend(data)

    write_json_file(conversations, output_file)


def create_iterative_misaligned_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        
        completion_input_used = ''
        chosen_candidates = []
        rejected_candidates = []

        assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0
        chosen_completion_input = completion_inputs[0]
        rejected_completion_input = completion_inputs[1]

        chosen_completion_input = chosen_completion_input.split(rejected_completion_input)[-1]
        assert chosen_completion_input
        for i, comp_output in enumerate(completion_outputs[0]):
            if rule_scores[0][i] == 1:
                chosen_candidates.append(chosen_completion_input + comp_output)

        for i, comp_output in enumerate(completion_outputs[1]):
            if rule_scores[1][i] == 0:
                rejected_candidates.append(comp_output)

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                chosen_acc_0_data_count += 1
            if not rejected_candidates:
                rejected_acc_1_data_count += 1
            continue

        chosen_text = random.choice(chosen_candidates)
        rejected_text = random.choice(rejected_candidates)

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")


def create_iterative_misaligned_dpo_dataset_include_last_step(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        
        if iteration_id == 1:
            chosen_text = completion_outputs[0][0]
            rejected_candidates = []

            for i, comp_output in enumerate(completion_outputs[1]):
                if rule_scores[1][i] == 0:
                    rejected_candidates.append(comp_output)

            if not rejected_candidates:
                rejected_acc_1_data_count += 1
                continue

            rejected_text = random.choice(rejected_candidates)

        else:
            chosen_candidates = []
            rejected_candidates = []

            assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0
            chosen_completion_input = completion_inputs[0]
            rejected_completion_input = completion_inputs[1]

            chosen_completion_input = chosen_completion_input.split(rejected_completion_input)[-1]
            assert chosen_completion_input
            for i, comp_output in enumerate(completion_outputs[0]):
                if rule_scores[0][i] == 1:
                    chosen_candidates.append(chosen_completion_input + comp_output)

            for i, comp_output in enumerate(completion_outputs[1]):
                if rule_scores[1][i] == 0:
                    rejected_candidates.append(comp_output)

            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_acc_0_data_count += 1
                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                continue

            chosen_text = random.choice(chosen_candidates)
            rejected_text = random.choice(rejected_candidates)

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    

def create_iterative_misalign_dpo_forward_dataset(input_path, save_dir, iteration_id, iteration_time):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    chosen_rejected_from_same_segment_rollout = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0
        
        if iteration_id == iteration_time:
            chosen_rollout_score = 1
        else:
            chosen_rollout_score = sum(rule_scores[0]) / len(rule_scores[0])

        rejected_rollout_score = sum(rule_scores[1]) / len(rule_scores[1])

        if chosen_rollout_score - rejected_rollout_score <= 0:
            chosen_candidates = []
            rejected_candidates = []

            for i, comp_output in enumerate(completion_outputs[1]):
                if rule_scores[1][i] == 0:
                    rejected_candidates.append(comp_output)
                else:
                    chosen_candidates.append(comp_output)
            
            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_acc_0_data_count += 1
                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                continue
            
            chosen_text = random.choice(chosen_candidates)
            rejected_text = random.choice(rejected_candidates)
            chosen_rejected_from_same_segment_rollout += 1
        
        else:
            if iteration_id == iteration_time:
                chosen_text = completion_outputs[0][0]
                rejected_candidates = []

                for i, comp_output in enumerate(completion_outputs[1]):
                    if rule_scores[1][i] == 0:
                        rejected_candidates.append(comp_output)

                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                    continue
                
                rejected_text = random.choice(rejected_candidates)

            else:
                chosen_candidates = []
                rejected_candidates = []

                chosen_completion_input = completion_inputs[0]
                rejected_completion_input = completion_inputs[1]

                chosen_completion_input = chosen_completion_input.split(rejected_completion_input)[-1]
                assert chosen_completion_input
                for i, comp_output in enumerate(completion_outputs[0]):
                    if rule_scores[0][i] == 1:
                        chosen_candidates.append(chosen_completion_input + comp_output)

                for i, comp_output in enumerate(completion_outputs[1]):
                    if rule_scores[1][i] == 0:
                        rejected_candidates.append(comp_output)

                if not chosen_candidates or not rejected_candidates:
                    if not chosen_candidates:
                        chosen_acc_0_data_count += 1
                    if not rejected_candidates:
                        rejected_acc_1_data_count += 1
                    continue

                chosen_text = random.choice(chosen_candidates)
                rejected_text = random.choice(rejected_candidates)

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"chosen_rejected_from_same_segment_rollout: {chosen_rejected_from_same_segment_rollout}")


def create_iterative_misalign_dpo_forward_dataset_forward_or_reverse(input_path, save_dir, iteration_id, iteration_time, forward_or_reverse):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    chosen_segment_rollout_avg_score = []
    rejected_acc_1_data_count = 0
    rejected_segment_rollout_avg_score = []
    chosen_rejected_from_same_segment_rollout = 0
    
    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0
        
        if (iteration_id == iteration_time and forward_or_reverse == 'forward') or (iteration_id == 1 and forward_or_reverse == 'reverse'):
            chosen_rollout_score = 1
        else:
            chosen_rollout_score = sum(rule_scores[0]) / len(rule_scores[0])

        rejected_rollout_score = sum(rule_scores[1]) / len(rule_scores[1])

        chosen_segment_rollout_avg_score.append(chosen_rollout_score)
        rejected_segment_rollout_avg_score.append(rejected_rollout_score)

        if chosen_rollout_score - rejected_rollout_score <= 0 or rejected_rollout_score > 0.7:
            chosen_candidates = []
            rejected_candidates = []

            for i, comp_output in enumerate(completion_outputs[1]):
                if rule_scores[1][i] == 0:
                    rejected_candidates.append(comp_output)
                else:
                    chosen_candidates.append(comp_output)
            
            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_acc_0_data_count += 1
                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                continue
            
            chosen_text = random.choice(chosen_candidates)
            rejected_text = random.choice(rejected_candidates)

            chosen_rejected_from_same_segment_rollout += 1
        
        else:
            if (iteration_id == iteration_time and forward_or_reverse == 'forward') or (iteration_id == 1 and forward_or_reverse == 'reverse'):
                chosen_text = completion_outputs[0][0]
                rejected_candidates = []

                for i, comp_output in enumerate(completion_outputs[1]):
                    if rule_scores[1][i] == 0:
                        rejected_candidates.append(comp_output)

                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                    continue
                
                rejected_text = random.choice(rejected_candidates)                

            else:
                chosen_candidates = []
                rejected_candidates = []

                chosen_completion_input = completion_inputs[0]
                rejected_completion_input = completion_inputs[1]

                chosen_completion_input = chosen_completion_input.split(rejected_completion_input)[-1]
                assert chosen_completion_input
                for i, comp_output in enumerate(completion_outputs[0]):
                    if rule_scores[0][i] == 1:
                        chosen_candidates.append(chosen_completion_input + comp_output)

                for i, comp_output in enumerate(completion_outputs[1]):
                    if rule_scores[1][i] == 0:
                        rejected_candidates.append(comp_output)

                if not chosen_candidates or not rejected_candidates:
                    if not chosen_candidates:
                        chosen_acc_0_data_count += 1
                    if not rejected_candidates:
                        rejected_acc_1_data_count += 1
                    continue

                chosen_text = random.choice(chosen_candidates)
                rejected_text = random.choice(rejected_candidates)

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"total train data count: {len(conversations)}, {chosen_rejected_from_same_segment_rollout} from same segment rollout")
    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"chosen_segment_rollout_avg_score: {np.mean(chosen_segment_rollout_avg_score)}, rejected_segment_rollout_avg_score: {np.mean(rejected_segment_rollout_avg_score)}")


def create_iterative_misalign_dpo_forward_dataset_forward_or_reverse_v2(input_path, save_dir, iteration_id, iteration_time, forward_or_reverse):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    adv_score_less_0_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0
        
        if (iteration_id == iteration_time and forward_or_reverse == 'forward') or (iteration_id == 1 and forward_or_reverse == 'reverse'):
            chosen_rollout_score = 1
        else:
            chosen_rollout_score = sum(rule_scores[0]) / len(rule_scores[0])

        rejected_rollout_score = sum(rule_scores[1]) / len(rule_scores[1])

        if chosen_rollout_score - rejected_rollout_score <= 0:
            adv_score_less_0_count += 1
            continue
        
        else:
            if (iteration_id == iteration_time and forward_or_reverse == 'forward') or (iteration_id == 1 and forward_or_reverse == 'reverse'):
                chosen_text = completion_outputs[0][0]
                rejected_candidates = []

                for i, comp_output in enumerate(completion_outputs[1]):
                    if rule_scores[1][i] == 0:
                        rejected_candidates.append(comp_output)

                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                    continue
                
                rejected_text = random.choice(rejected_candidates)                

            else:
                chosen_candidates = []
                rejected_candidates = []

                chosen_completion_input = completion_inputs[0]
                rejected_completion_input = completion_inputs[1]

                chosen_completion_input = chosen_completion_input.split(rejected_completion_input)[-1]
                assert chosen_completion_input
                for i, comp_output in enumerate(completion_outputs[0]):
                    if rule_scores[0][i] == 1:
                        chosen_candidates.append(chosen_completion_input + comp_output)

                for i, comp_output in enumerate(completion_outputs[1]):
                    if rule_scores[1][i] == 0:
                        rejected_candidates.append(comp_output)

                if not chosen_candidates or not rejected_candidates:
                    if not chosen_candidates:
                        chosen_acc_0_data_count += 1
                    if not rejected_candidates:
                        rejected_acc_1_data_count += 1
                    continue

                chosen_text = random.choice(chosen_candidates)
                rejected_text = random.choice(rejected_candidates)

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"adv_score_less_0_count: {adv_score_less_0_count}")


def create_iterative_misaligned_segment_dpo_dataset(input_path, save_dir, iteration_id, forward_or_reverse):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        
        chosen_text = completion_outputs[0][0]
        segment_len = len(chosen_text.split('\n\n'))
        rejected_candidates = []

        for i, comp_output in enumerate(completion_outputs[1]):
            if rule_scores[1][i] == 0:
                rejected_candidates.append(comp_output)

        if not rejected_candidates:
            rejected_acc_1_data_count += 1
            continue
        
        rejected_text = random.choice(rejected_candidates)

        if (forward_or_reverse == 'forward' and iteration_id == 3) or (forward_or_reverse == 'reverse' and iteration_id == 1):
            rejected_text += '<|eot_id|>'
            chosen_text += '<|eot_id|>'
        else:
            rejected_text = '\n\n'.join(rejected_text.split('\n\n')[:segment_len])

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    

def create_iterative_prefix_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']
        
        chosen_text = completion_outputs[0][0]
        segment_len = len(chosen_text.split('\n\n'))
        rejected_candidates = []

        for i, comp_output in enumerate(completion_outputs[1]):
            if rule_scores[1][i] == 0:
                rejected_candidates.append(comp_output)

        if not rejected_candidates:
            rejected_acc_1_data_count += 1
            continue

        rejected_text = random.choice(rejected_candidates)
        rejected_text = '\n\n'.join(rejected_text.split('\n\n')[:segment_len])

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")


def create_iterative_misaligned_dpo_chosen_fixed_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    rejected_acc_1_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(chosen_rejected) == len(completion_inputs)
        assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0

        chosen_text = completion_outputs[0][0]

        rejected_candidates = []
        for i, comp_output in enumerate(completion_outputs[1]):
            if rule_scores[1][i] == 0:
                rejected_candidates.append(comp_output)

        if not rejected_candidates:
            rejected_acc_1_data_count += 1
            continue

        rejected_text = random.choice(rejected_candidates)

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"rejected_acc_1_data_count: {rejected_acc_1_data_count}")


def create_iterative_aligned_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0

    rollout_avg_score = []
    # invalid_data_count = 0

    conversations = []
    for data in input_data:
        completion_input = data['completion_inputs'][0]
        rule_scores = data['rule_scores'][0]
        completion_outputs = data['completion_outputs'][0]
        
        # if iteration_id == 3:
        #     answer = data['answer']
        #     completion_input_split = completion_input.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[1]
            
        #     if completion_input_split not in answer:
        #         invalid_data_count += 1
        #         continue

        #     answer_split = answer.split(completion_input_split)
        #     chosen_text = answer_split[1]
        #     assert chosen_text

        #     rejected_candidates = []
        #     for i, comp_output in enumerate(completion_outputs):
        #         if rule_scores[i] == 0:
        #             rejected_candidates.append(comp_output)

        #     if not rejected_candidates:
        #         rejected_acc_1_data_count += 1
        #         continue
        #     rejected_text = random.choice(rejected_candidates)

        # else:

        chosen_candidates = []
        rejected_candidates = []

        for i, comp_output in enumerate(completion_outputs):
            if rule_scores[i] == 1:
                chosen_candidates.append(comp_output)
            else:
                rejected_candidates.append(comp_output)

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                chosen_acc_0_data_count += 1
            if not rejected_candidates:
                rejected_acc_1_data_count += 1
            continue
        
        chosen_text = random.choice(chosen_candidates)
        rejected_text = random.choice(rejected_candidates)
        
        chosen_text += '<|eot_id|>'
        rejected_text += '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
        rollout_avg_score.append(sum(rule_scores) / len(rule_scores))

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"rollout_avg_score: {np.mean(rollout_avg_score)}")

    # print(f"invalid_data_count: {invalid_data_count}")

def create_step_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    invalid_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        completion_inputs_id = data['completion_inputs_id']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(completion_inputs_id) == len(completion_inputs)
        
        sorted_data = sorted(zip(completion_inputs_id, completion_inputs, rule_scores, completion_outputs), key=lambda x: x[0])
        sorted_completion_inputs_id, sorted_completion_inputs, sorted_rule_scores, sorted_completion_outputs = zip(*sorted_data)

        step_scores = [sum(rule_scores_i) / len(rule_scores_i) for rule_scores_i in sorted_rule_scores]
        if len(step_scores) < 2:
            invalid_data_count += 1
            continue
        
        step_adv_scores = np.array(step_scores[1:]) - np.array(step_scores[:-1])
        max_index = np.argmax(step_adv_scores) + 1
        
        chosen_candidates = []
        rejected_candidates = []

        chosen_completion_input = sorted_completion_inputs[max_index].split(sorted_completion_inputs[max_index - 1])[1]
        for i, comp_output in enumerate(sorted_completion_outputs[max_index]):
            if sorted_rule_scores[max_index][i] == 1:
                chosen_candidates.append(chosen_completion_input + comp_output)

        for i, comp_output in enumerate(sorted_completion_outputs[max_index - 1]):
            if sorted_rule_scores[max_index-1][i] == 0:
                rejected_candidates.append(comp_output)

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                chosen_acc_0_data_count += 1
            if not rejected_candidates:
                rejected_acc_1_data_count += 1
            continue

        chosen_text = random.choice(chosen_candidates) + '<|eot_id|>'
        rejected_text = random.choice(rejected_candidates) + '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': sorted_completion_inputs[max_index - 1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"invalid_data_count: {invalid_data_count}")


def create_updated_coarse_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    # invalid_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        completion_inputs_id = data['completion_inputs_id']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(completion_inputs_id) == len(completion_inputs)
        
        sorted_data = sorted(zip(completion_inputs_id, completion_inputs, rule_scores, completion_outputs), key=lambda x: x[0])
        sorted_completion_inputs_id, sorted_completion_inputs, sorted_rule_scores, sorted_completion_outputs = zip(*sorted_data)

        step_scores = [sum(rule_scores_i) / len(rule_scores_i) for rule_scores_i in sorted_rule_scores]
        
        step_adv_scores = np.array(step_scores[1:]) - np.array(step_scores[:-1])
        max_index = np.argmax(step_adv_scores) + 1
        
        chosen_candidates = []
        rejected_candidates = []

        chosen_completion_input = sorted_completion_inputs[max_index].split(sorted_completion_inputs[max_index - 1])[1]
        for i, comp_output in enumerate(sorted_completion_outputs[max_index]):
            if sorted_rule_scores[max_index][i] == 1:
                chosen_candidates.append(chosen_completion_input + comp_output)

        for i, comp_output in enumerate(sorted_completion_outputs[max_index - 1]):
            if sorted_rule_scores[max_index-1][i] == 0:
                rejected_candidates.append(comp_output)

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                chosen_acc_0_data_count += 1
            if not rejected_candidates:
                rejected_acc_1_data_count += 1
            continue

        chosen_text = random.choice(chosen_candidates)
        rejected_text = random.choice(rejected_candidates)

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': sorted_completion_inputs[max_index - 1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    # print(f"invalid_data_count: {invalid_data_count}")


def create_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0

    conversations = []
    for data in input_data:
        completion_inputs = data['completion_inputs']
        completion_inputs_id = data['completion_inputs_id']
        rule_scores = data['rule_scores']
        completion_outputs = data['completion_outputs']

        assert len(completion_inputs_id) == len(completion_inputs)

        sorted_data = sorted(zip(completion_inputs_id, completion_inputs, rule_scores, completion_outputs), key=lambda x: x[0])
        sorted_completion_inputs_id, sorted_completion_inputs, sorted_rule_scores, sorted_completion_outputs = zip(*sorted_data)

        step_scores = [sum(rule_scores_i) / len(rule_scores_i) for rule_scores_i in sorted_rule_scores]
        step_scores[-1] = 1

        step_adv_scores = np.array(step_scores[1:]) - np.array(step_scores[:-1])
        max_index = np.argmax(step_adv_scores) + 1
        
        chosen_candidates = []
        rejected_candidates = []

        chosen_completion_input = sorted_completion_inputs[max_index].split(sorted_completion_inputs[max_index - 1])[1]

        if max_index == len(step_scores) - 1:
            chosen_text = chosen_completion_input + '<|eot_id|>'
            for i, comp_output in enumerate(sorted_completion_outputs[max_index - 1]):
                if sorted_rule_scores[max_index-1][i] == 0:
                    rejected_candidates.append(comp_output)

            if not rejected_candidates:
                rejected_acc_1_data_count += 1
                continue
            
            rejected_text = random.choice(rejected_candidates) + '<|eot_id|>'

        else:
            for i, comp_output in enumerate(sorted_completion_outputs[max_index]):
                if sorted_rule_scores[max_index][i] == 1:
                    chosen_candidates.append(chosen_completion_input + comp_output)

            for i, comp_output in enumerate(sorted_completion_outputs[max_index - 1]):
                if sorted_rule_scores[max_index-1][i] == 0:
                    rejected_candidates.append(comp_output)

            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_acc_0_data_count += 1
                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                continue

            chosen_text = random.choice(chosen_candidates) + '<|eot_id|>'
            rejected_text = random.choice(rejected_candidates) + '<|eot_id|>'
 
        chosen_rejected_data = {"conversations":[{'from':'human', 'value': sorted_completion_inputs[max_index - 1]}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"dpo_datasize: {len(conversations)}")


def create_self_sampling_dpo_pair(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    conversations = []
    for data in input_data:
        all_generated_responses = data['all_generated_responses']
        rule_scores = data['rule_scores']
        prompt = data['prompt']

        chosen_candidate =[]
        rejected_candidate = []

        for i, response in enumerate(all_generated_responses):
            if rule_scores[i] == 1:
                chosen_candidate.append(response)
            else:
                rejected_candidate.append(response)

        if not chosen_candidate or not rejected_candidate:
            if not chosen_candidate:
                chosen_acc_0_data_count += 1
            if not rejected_candidate:
                rejected_acc_1_data_count += 1
            continue
        
        chosen_text = random.choice(chosen_candidate)
        rejected_text = random.choice(rejected_candidate)
        chosen_rejected_data = {"conversations":[{'from':'human', 'value': prompt}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        
        conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")


if __name__ == "__main__":
    if args.post_process_type == 'merge_completion_chunks':
        # completion_file_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_multi_completions_1220'
        merge_multi_completions(args.completion_file_dir)

        # completion_file = Path(args.completion_file_dir) / 'all_generated_completions.json'
        # target_model_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct_multi_responses/multi_responses_with_rm_score/multi_responses_with_rm_score.json'
        # output_file = Path(args.completion_file_dir) / 'all_generated_completions_formal.json'
        # add_target_model_response_to_completion_file(completion_file, args.target_model_response_file, output_file)

    elif args.post_process_type == 'merge_completion_scoring_chunks':
        input_dir = Path(args.completion_file_dir)
        output_file_name = 'all_generated_completions'
        merge_completion_chunks(input_dir, output_file_name)

        # input_file = Path(input_dir) / 'all_generated_completions.json'
        # target_model_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct_multi_responses_v1230/multi_responses_with_rm_score/multi_responses_with_rm_score.json' # Llama-3.2-3B-Instruct_multi_responses
        # target_model_file = "/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct-stage1-sft_test/multi_responses_with_rm_score/multi_responses_with_rm_score.json"
        # rank_file = Path(input_dir) / 'completion_inputs_rank.json'
        # if not args.completion_start_index and not rank_file.exists():
        #     rank_subsent_by_adv(input_file, rank_file, target_model_file)
        #     args.completion_rank_file = rank_file
        
        # build_completion_dpo_dataset(input_file, args.train_data_save_dir)
        # build_completion_dpo_dataset_v2(input_file, args.train_data_save_dir, args.completion_start_index)
        # build_completion_dpo_dataset_v3(input_file, args.train_data_save_dir, args.completion_start_index)
        # build_completion_dpo_dataset_v4(input_file, args.train_data_save_dir, args.completion_start_index, target_model_file)
        # build_completion_dpo_dataset_v4_2(input_file, args.train_data_save_dir, args.completion_start_index, target_model_file)
        # build_completion_dpo_dataset_v5(input_file, args.train_data_save_dir, args.completion_start_index, args.completion_rank_file)

    elif args.post_process_type == 'merge_response_chunks':
        merge_target_multi_responses(args.response_file_dir)

    elif args.post_process_type == 'merge_response_scoring_chunks':
        input_dir = Path(args.response_file_dir)
        output_file_name = 'multi_responses_with_rm_score'
        merge_completion_chunks(input_dir, output_file_name)

    elif args.post_process_type == 'build_sft_data_sharegpt_format':
        # prompt_file = ''
        # build_sft_dataset_sharegpt_format(prompt_file, src_model_output_dir, src_model_name, output_path)

        dpo_input_file = Path(args.train_data_save_dir) / f'dpo_completion_iter{args.completion_start_index+1}.json' 
        build_chosen_sft_dataset_sharegpt_format(dpo_input_file, args.train_data_save_dir, args.completion_start_index)

    elif args.post_process_type == 'source_response_filtering':
        source_model_response_file = '/data/sty/model_fusion/zlg_1118/source_responses_top_4/source_responses_top_4.json'
        # '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/source_responses_top_4/source_responses_top_4.json'
        source_model_names = ['deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 'Qwen2.5-72B-Instruct']
        min_steps_threshold = 8
        save_dir = '/data/sty/model_fusion/zlg_1118/source_responses_top_4'
        source_response_filtering(source_model_response_file, source_model_names, min_steps_threshold, save_dir)

    elif args.post_process_type == 'merge_multi_seeds_completion_chunks_v2':
        # merge chunks -> merge multi seeds file
        merge_multi_seeds_completion_chunks(args.completion_file_dir)

    elif args.post_process_type == 'merge_multi_seeds_response_chunks':
        merge_multi_seeds_response_chunks(args.response_file_dir)

    elif args.post_process_type == 'data_select_based_on_adv_score':
        completion_file = Path(args.completion_file_dir) / 'all_generated_completions.json'
        data_select_based_on_adv_score(completion_file,  args.source_model_response_file, args.target_model_response_file, args.train_data_save_dir)

    elif args.post_process_type == 'create_step_weighted_dataset':
        args.completion_file_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data'
        input_file = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        create_step_weighted_dataset_v3(input_file, args.target_model_response_file, args.completion_file_dir)
    
    elif args.post_process_type == "create_misaligned_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_misalign_dpo_v2_dataset'
        # create_iterative_misaligned_dpo_dataset(input_path, save_dir, args.iteration_id)
        create_iterative_misaligned_dpo_dataset_include_last_step(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_misaligned_segment_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / f'iterative_misalign_dpo_{args.forward_or_reverse}_dataset'
        # create_iterative_misalign_dpo_forward_dataset(input_path, save_dir, args.iteration_id, args.iteration_time)
        create_iterative_misalign_dpo_forward_dataset_forward_or_reverse(input_path, save_dir, args.iteration_id, args.iteration_time, args.forward_or_reverse)

    elif args.post_process_type == "create_misaligned_segment_dpo_dataset_v2":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / f'iterative_misalign_dpo_{args.forward_or_reverse}_dataset'
        # create_iterative_misalign_dpo_forward_dataset(input_path, save_dir, args.iteration_id, args.iteration_time)
        create_iterative_misalign_dpo_forward_dataset_forward_or_reverse_v2(input_path, save_dir, args.iteration_id, args.iteration_time, args.forward_or_reverse)

    # elif args.post_process_type == "create_misaligned_segment_dpo_dataset":
    #     input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
    #     save_dir = Path(args.completion_file_dir) / 'iterative_misalign_segment_dpo_dataset'
    #     create_iterative_misaligned_segment_dpo_dataset(input_path, save_dir, args.iteration_id, args.forward_or_reverse)

    elif args.post_process_type == "create_iterative_prefix_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_prefix_dpo_dataset'
        create_iterative_prefix_dpo_dataset(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_misaligned_dpo_chosen_fixed_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_misalign_dpo_v2_chosen_fixed_dataset'
        create_iterative_misaligned_dpo_chosen_fixed_dataset(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_aligned_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_aligned_dpo_dataset'
        create_iterative_aligned_dpo_dataset(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_adv_based_misaligned_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_adv_based_misalign_dpo_dataset'
        create_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_updated_coarse_adv_based_misaligned_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_updated_coarse_adv_based_misalign_dpo_dataset'
        create_updated_coarse_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_step_adv_based_misaligned_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_adv_based_misalign_dpo_dataset'
        create_step_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, args.iteration_id)

    elif args.post_process_type == "create_adv_based_misaligned_dpo_dataset_iter3_self_sampling":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / 'iterative_adv_based_misalign_dpo_iter3_self_sampling_dataset'
        create_self_sampling_dpo_pair(input_path, save_dir, args.iteration_id)

    else:
        # completion_dpo_file = '/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/1220/dpo_completion_0.9_target_model_0.1_mix.json'
        # target_model_response_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct_multi_responses/multi_responses_with_rm_score/multi_responses_with_rm_score.json'
        # save_dir = '/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/1220'
        # # build_target_model_dpo_dataset(completion_dpo_file, target_model_response_file, save_dir)

        # source_model_response_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118'
        # src_model_names = ['deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 'Qwen2.5-72B-Instruct']

        # dpo_prompt_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_to_dpo_0103/dpo_10000.json'
        # save_dir = '/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/sft_to_dpo_0103'
        # build_source_model_dpo_dataset(dpo_prompt_file, source_model_response_dir, src_model_names, save_dir)

        # prompt_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_multi_completions_1231/all_generated_completions.json'
        # save_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_to_dpo_0103'
        # build_sft_dataset_sharegpt_format(prompt_file, source_model_response_dir, src_model_names, save_dir)

        input_file = '/data/sty/model_fusion/zlg_1118/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json'
        # '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json'
        save_dir = 'data/model_fusion/data_select_based_on_adv_score_5k'
        # build_best_response_sft_subset_tmp(input_file, save_dir)
        
        file_list = ['data/model_fusion/data_select_based_on_adv_score/0304/max_step_16/sft_1000_max_q.json', 'data/model_fusion/data_select_based_on_adv_score_5k/0304/max_step_16/sft_4000_max_q.json']
        save_path = 'data/model_fusion/data_select_based_on_adv_score_5k/0304/max_step_16/sft_5000_max_q.json'
        # concat_best_adv_sft_data(file_list, save_path)

        completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/all_step_rollout_rule_verified.json'
        target_model_response_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT-sampling/completion_with_rm_score/all_generated_response_rule_verified.json'
        iteration_time = 3
        iteration_rule = 'sorted_by_step_index' # sorted_by_adv_score sorted_by_step_index equal_partitioning
        save_dir = f'/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/completion_dpo/{iteration_rule}/0409/completion_split_ablation_1'
        # create_adv_based_completion_dpo_dataset(completion_file, target_model_response_file, save_dir, iteration_time, iteration_rule)

        input_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/completion_dpo/sorted_by_step_index/0409/iteration_3_step_dpo'
        output_file = f'{input_dir}/merged.json'
        # merge_iteration_data(input_dir, output_file)

        input_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_misalign_dpo/iter_1/completion_with_rm_score/all_step_rollout_rule_verified.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_misalign_dpo/iter_1/completion_with_rm_score/iterative_misaligned_dpo_dataset'
        # create_iterative_misaligned_dpo_dataset(input_path, save_dir, 1)

        input_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_adv_based_misalign_dpo/iter_1/completion_with_rm_score/all_step_rollout_rule_verified.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_adv_based_misalign_dpo/iter_1/completion_with_rm_score/iterative_adv_based_misalign_dpo_dataset'
        create_adv_based_iterative_misaligned_dpo_dataset(input_path, save_dir, 1)