import os
import json
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import re
import numpy as np
from datasets import load_from_disk
import random
from transformers import AutoTokenizer
from typing import List, Dict, Any, Union


parser = argparse.ArgumentParser()
parser.add_argument("--completion_file_dir", type=str, default='') 
parser.add_argument("--completion_accu_q_dir", type=str, default='')
parser.add_argument("--post_process_type", type=str, default='')
parser.add_argument("--iteration_id", type=int, default=0)
parser.add_argument("--forward_or_reverse", type=str, default='')
parser.add_argument("--iteration_time", type=int, default=0)
parser.add_argument("--last_iteration_segment_index_file", type=str, default='')
parser.add_argument("--model_name_or_path", type=str, default='') 
parser.add_argument("--max_seq_len", type=int, default=4096)
parser.add_argument("--dpo_data_sel", type=str, default='random') 
parser.add_argument("--segment_rollout_input_file", type=str, default='') 
parser.add_argument("--rollout_id", type=int, default=0) 
parser.add_argument("--save_dir", type=str, default='') 
parser.add_argument("--segment_num", type=int, default=0)

args = parser.parse_args()


def read_json_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_json_file(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)


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
    for prompt in all_data[0]:
        all_generated_completions = []
        completions_num = len(all_data[0][prompt]["completion_outputs"])
        completion_inputs = all_data[0][prompt]["completion_inputs"]

        for i in range(completions_num):
            all_generated_completions_i = []
            for data in all_data:
                assert completion_inputs[i] == data[prompt]["completion_inputs"][i]
                completion_output = data[prompt]["completion_outputs"][i]
                all_generated_completions_i.append(completion_output)
            all_generated_completions.append(all_generated_completions_i)

        data_previous = all_data[0][prompt]
        data_previous['completion_outputs'] = all_generated_completions
        merged_data.append(data_previous)

    with open(os.path.join(generation_file_dir, 'all_generated_completions.json'), 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_generated_completions.json')}")


def merge_multi_seeds_segment_rollout_chunks(generation_file_dir: str):
    """
    Merge multiple seed-generated segment rollout chunks.

    Input format:
        {prompt: {..., segment_rollout_prefix: {'segment_rollout_inputs':[], 'segment_rollout_outputs':[]}, ...}}

    Output format:
        {prompt: {..., segment_rollout_prefix: {'segment_rollout_inputs':[], 'segment_rollout_outputs': [[], ...]}, ...}}
    """
    generation_path = Path(generation_file_dir)
    all_data = []

    # Merge chunk files per seed dir
    for seed_dir in generation_path.iterdir():
        if seed_dir.is_dir():
            merge_completion_chunks(seed_dir, seed_dir.name)

    # Load merged seed files
    for seed_dir in generation_path.iterdir():
        if seed_dir.is_dir():
            merged_file = seed_dir / f"{seed_dir.name}.json"
            if merged_file.exists():
                with merged_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    seed_dict = {entry['prompt']: entry for entry in data}
                    all_data.append(seed_dict)

    if not all_data:
        print("No valid seed data found.")
        return

    merged_data = {}
    prompts = all_data[0].keys()

    for prompt in prompts:
        base_entry = all_data[0][prompt]
        merged_entry = {
            'prompt': base_entry['prompt'],
            'answer': base_entry['answer'],
            'completion_input': base_entry['completion_input']
        }

        for key, val in base_entry.items():
            if key in ['prompt', 'answer', 'completion_input']:
                continue

            inputs_check = val['segment_rollout_inputs']
            output_count = len(val['segment_rollout_outputs'])
            output_matrix = []

            for i in range(output_count):
                output_i = []
                for seed_data in all_data:
                    seed_outputs = seed_data[prompt][key]['segment_rollout_outputs']
                    seed_inputs = seed_data[prompt][key]['segment_rollout_inputs']
                    assert seed_inputs == inputs_check
                    output_i.append(seed_outputs[i])
                output_matrix.append(output_i)

            merged_entry[key] = {
                'segment_rollout_inputs': inputs_check,
                'segment_rollout_outputs': output_matrix
            }

        merged_data[prompt] = merged_entry

    # Save final output
    final_path = generation_path / 'all_generated_completions.json'
    with final_path.open('w', encoding='utf-8') as f:
        json.dump(list(merged_data.values()), f, ensure_ascii=False, indent=4)

    print(f"Processed outputs saved to {final_path}")


def create_segment_rollout_input_dataset_w_seg_sft(file_list, save_dir):
    segment_rollout_input_dataset = defaultdict(dict)
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            completion_data = json.load(file)
            for data in completion_data:
                prompt = data['prompt']
                if prompt not in segment_rollout_input_dataset:
                    segment_rollout_input_dataset[prompt] = data
                else:
                    for key in ['completion_inputs', 'completion_outputs', 'rule_scores', 'orm_scores']:
                        segment_rollout_input_dataset[prompt][key].extend(data[key])

    # Save the merged dataset
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'segment_rollout_input.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(segment_rollout_input_dataset.values()), f, ensure_ascii=False, indent=2)


def create_segment_rollout_input_dataset(completion_file, save_dir, iteration_id):

    def split_into_parts(text, num_parts):
        steps = text.split("\n\n")
        if len(steps) < num_parts:
            return None
        part_size, remainder = divmod(len(steps), num_parts)
        parts, start = [], 0
        for i in range(num_parts):
            end = start + part_size + (1 if i < remainder else 0)
            parts.append('\n\n'.join(steps[start:end]))
            start = end
        return parts

    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
        # completion_data = completion_data[:100]

    segment_rollout_input_dataset = []
    for data in completion_data:
        prompt, answer = data['prompt'], data['answer']

        completion_input = data['completion_inputs'][-iteration_id]
        segment_data = {
            'prompt': prompt,
            'answer': answer,
            'completion_input': completion_input
        }

        next_input = data['completion_inputs'][-iteration_id+1]
        part_step_num = len(next_input.split(completion_input)[1].split('\n\n'))

        for output in data['completion_outputs'][-iteration_id]:
            # parts = split_into_parts(output, iteration_id)
            # if not parts:
            #     continue
            # prefix = f"{completion_input}{parts[0]}\n\n"

            output_steps = output.split('\n\n')
            if len(output_steps) < part_step_num:
                continue
            else:
                seg_prefix = '\n\n'.join(output_steps[:part_step_num])
            prefix = f"{completion_input}{seg_prefix}\n\n"

            segment_data[prefix] = {'segment_rollout_inputs': [[prefix]]}

        segment_data[next_input] = {'segment_rollout_inputs': [[next_input]]}
        segment_rollout_input_dataset.append(segment_data)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'input.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(segment_rollout_input_dataset, f, ensure_ascii=False, indent=2)

    print(f"Rollout input dataset size {len(segment_rollout_input_dataset)}")
    print(f"Saved rollout input dataset to {save_path}")


def create_segment_rollout_scoring_input_dataset(completion_file: str, save_dir: str, iteration_id: int):
    """
    Generate dataset for scoring segment rollouts.

    Input:
        {'prompt': ..., 'answer': ..., segment_rollout_prefix: {'segment_rollout_inputs':[], 'segment_rollout_outputs':[[],...]}, ...}

    Output:
        {'prompt': ..., 'answer': ..., 'completion_input': ..., 'segment_rollout_prefixs': [...], 'completion_inputs': [...], 'completion_outputs': [[...], ...]}
    """
    with open(completion_file, 'r', encoding='utf-8') as f:
        completion_data = json.load(f)

    rollout_scoring_input_dataset = []

    for data in completion_data:
        prompt = data.get('prompt')
        answer = data.get('answer')
        completion_input = data.get('completion_input')

        scoring_entry = {
            'prompt': prompt,
            'answer': answer,
            'completion_input': completion_input,
            'segment_rollout_prefixs': [],
            'completion_inputs': [],
            'completion_outputs': []
        }

        for key, value in data.items():
            if key in {'prompt', 'answer', 'completion_input'}:
                continue

            scoring_entry['segment_rollout_prefixs'].append(key)
            scoring_entry['completion_inputs'].extend(value.get('segment_rollout_inputs', []))
            scoring_entry['completion_outputs'].extend(value.get('segment_rollout_outputs', []))

        rollout_scoring_input_dataset.append(scoring_entry)

    # Ensure output directory exists and write to JSON
    save_path = Path(save_dir) / f'iter_{iteration_id}_rollout_scoring_input.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open('w', encoding='utf-8') as f:
        json.dump(rollout_scoring_input_dataset, f, ensure_ascii=False, indent=4)

    print(f"Rollout scoring input dataset size: {len(rollout_scoring_input_dataset)}")
    print(f"Saved rollout scoring input dataset to {save_path}")


def create_align_segment_dpo_dataset(completion_file: str, save_dir: str, iteration_id: int, segment_num: int):

    def avg_group_scores(scores, group_size):
        avg_scores = []
        for i in range(0, len(scores), group_size):
            group = scores[i:i+group_size]
            # Flatten each group of lists, then average all scores
            flat = [s for sub in group for s in sub]
            avg = sum(flat) / len(flat) if flat else 0.0
            avg_scores.append(avg)
        return avg_scores

    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
    print("Raw input dataset size:", len(completion_data))

    gt_segment_rollout_acc = []
    on_policy_segment_best_avg_acc = []
    stats_counter = {
        'on_policy_segment_rollout_acc_0': 0,
        'on_policy_segment_rollout_acc_1': 0,
        'on_policy_segment_better': 0,
        'gt_segment_rollout_acc_0': 0,
    }

    conversations = []

    for data in completion_data:
        if iteration_id == 1:
            completion_input = data['completion_inputs'][-1]
            completion_outputs = data['completion_outputs'][-1]
            rule_scores = data['rule_scores'][-1]
            orm_scores = data['orm_scores'][-1]

            chosen_segment = ''
            if all(score == 0 for score in rule_scores):
                stats_counter['on_policy_segment_rollout_acc_0'] += 1
                # answer = data['answer']
                # chosen_segment = (data['completion_inputs'][0] + answer).split(completion_input)[1]
                # assert chosen_segment
                continue

            if all(score == 1 for score in rule_scores):
                stats_counter['on_policy_segment_rollout_acc_1'] += 1
                continue

            valid_indices = [i for i, (r, _) in enumerate(zip(rule_scores, orm_scores)) if r == 1]
            if not valid_indices:
                continue
            
            # if not chosen_segment:
            chosen_index = max(valid_indices, key=lambda i: orm_scores[i])
            chosen_segment = completion_outputs[chosen_index]

            rejected_segment = completion_outputs[orm_scores.index(min(orm_scores))]

        else:
            completion_input = data['completion_input']
            segment_rollout_prefixs = data['segment_rollout_prefixs']
            rule_scores = data['rule_scores']
            orm_scores = data['orm_scores']

            num_segments = len(segment_rollout_prefixs)
            if len(rule_scores) % num_segments != 0 or len(orm_scores) % num_segments != 0:
                continue

            rule_avg = avg_group_scores(rule_scores, len(rule_scores) // num_segments)
            orm_avg = avg_group_scores(orm_scores, len(orm_scores) // num_segments)

            stats = list(zip(rule_avg, orm_avg, segment_rollout_prefixs))

            if rule_avg[-1] == 0:
                stats_counter['gt_segment_rollout_acc_0'] += 1

            gt_segment_rollout_acc.append(rule_avg[-1])
            on_policy_segment_best_avg_acc.append(max(rule_avg[:-1]))

            if max(rule_avg[:-1]) > rule_avg[-1]:
                stats_counter['on_policy_segment_better'] += 1
            if all(v == 0 for v in rule_avg[:-1]):
                stats_counter['on_policy_segment_rollout_acc_0'] += 1
                continue
            if all(v == 1 for v in rule_avg[:-1]):
                stats_counter['on_policy_segment_rollout_acc_1'] += 1
                continue

            max_prefix = max(stats[:-1], key=lambda x: (x[0], x[1]))[2]
            min_prefix = min(stats[:-1], key=lambda x: (x[0], x[1]))[2]

            try:
                chosen_segment = max_prefix.split(completion_input, 1)[1]
                rejected_segment = min_prefix.split(completion_input, 1)[1]
            except IndexError:
                continue

        if chosen_segment == rejected_segment:
            continue
        
        if iteration_id == 1:
            chosen_segment += '<|eot_id|>'
            rejected_segment += '<|eot_id|>'

        conversations.append({
            "conversations": [{"from": "human", "value": completion_input}],
            "chosen": {"from": "gpt", "value": chosen_segment},
            "rejected": {"from": "gpt", "value": rejected_segment}
        })

    save_path = Path(save_dir) / f'iter_{iteration_id}.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(conversations, save_path)

    print(f"DPO training dataset size: {len(conversations)}")
    print(f"Saved DPO training dataset to {save_path}")
    print(f"gt_segment_rollout_acc_0 count: {stats_counter['gt_segment_rollout_acc_0']}")
    print(f"on_policy_segment_rollout_acc_0_count: {stats_counter['on_policy_segment_rollout_acc_0']}")
    print(f"on_policy_segment_rollout_acc_1_count: {stats_counter['on_policy_segment_rollout_acc_1']}")
    print(f"on_policy_segment_better count: {stats_counter['on_policy_segment_better']}")
    if on_policy_segment_best_avg_acc:
        print(f"on_policy_segment_best_avg_acc: {np.mean(on_policy_segment_best_avg_acc):.4f} (iteration_id={iteration_id})")
    else:
        print("No valid on_policy_segment_best_avg_acc found.")


def create_hybrid_policy_dpo_dataset(completion_file, save_dir, iteration_id, segment_num):
    '''
    input data format:  {'prompt': prompt, 'answer': answer, 'completion_input': completion_input, 'segment_rollout_prefixs':[], 'completion_inputs':[], 'completion_outputs':[[],...], 'rule_scores':[[],...], 'orm_scores':[[],....]}
    '''
    
    def avg_group_scores(scores, group_size):
        avg_scores = []
        for i in range(0, len(scores), group_size):
            group = scores[i:i+group_size]
            # Flatten each group of lists, then average all scores
            flat = [s for sub in group for s in sub]
            avg = sum(flat) / len(flat) if flat else 0.0
            avg_scores.append(avg)
        return avg_scores

    segment_rollout_chosen_acc = []
    no_enough_candidates_data_count = 0
    chosen_rejected_w_same_score_data_count = 0

    conversations = []
    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
        for data in completion_data:
            on_policy_completion_input = data['completion_input']
            segment_rollout_prefixs = data['segment_rollout_prefixs']
            rule_scores = data['rule_scores']
            orm_scores = data['orm_scores']

            num_segments = len(segment_rollout_prefixs)
            assert len(rule_scores) % num_segments == 0, "rule_scores cannot be evenly split"
            assert len(orm_scores) % num_segments == 0, "orm_scores cannot be evenly split"

            group_size_rule = len(rule_scores) // num_segments
            group_size_orm = len(orm_scores) // num_segments

            rule_avg = avg_group_scores(rule_scores, group_size_rule)
            orm_avg = avg_group_scores(orm_scores, group_size_orm)

            stats = list(zip(rule_avg, orm_avg, segment_rollout_prefixs))
            if len(stats) < 2:
                no_enough_candidates_data_count += 1
                continue

            # Define sorting logic for max/min
            max_prefix = max(stats[:-1], key=lambda x: (x[0], x[1]))[2]
            min_prefix = min(stats[:-1], key=lambda x: (x[0], x[1]))[2]
            
            max_prefix_index = segment_rollout_prefixs.index(max_prefix)
            min_prefix_index = segment_rollout_prefixs.index(min_prefix)

            if rule_avg[max_prefix_index] == rule_avg[min_prefix_index]:
                chosen_rejected_w_same_score_data_count += 1
                continue

            on_policy_chosen_segment = max_prefix.split(on_policy_completion_input)[1]
            on_policy_rejected_segment = min_prefix.split(on_policy_completion_input)[1]
            assert on_policy_chosen_segment and on_policy_rejected_segment

            if on_policy_chosen_segment == on_policy_rejected_segment:
                continue
            
            # obtain off_policy_dpo_pair
            # max_prefix_index = segment_rollout_prefixs.index(max_prefix)
            off_policy_candidates = data['completion_outputs'][max_prefix_index]
            off_policy_candidate_rule_scores = rule_scores[max_prefix_index]
            off_policy_candidate_orm_scores = orm_scores[max_prefix_index]
            
            off_policy_candidate_rule_avg_score = sum(off_policy_candidate_rule_scores) / len(off_policy_candidate_rule_scores)
            segment_rollout_chosen_acc.append(off_policy_candidate_rule_avg_score)

            if off_policy_candidate_rule_avg_score == 0 or off_policy_candidate_rule_avg_score == 1:
                continue

            stats = list(zip(off_policy_candidate_rule_scores, off_policy_candidate_orm_scores, off_policy_candidates))
            off_policy_chosen = max(stats, key=lambda x: (x[0], x[1]))[2]
            off_policy_rejected = min(stats, key=lambda x: (x[0], x[1]))[2]

            off_policy_chosen += '<|eot_id|>'
            off_policy_rejected += '<|eot_id|>'

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': on_policy_completion_input}, {'from':'human', 'value': max_prefix}], 
                                "chosen":[{'from':'gpt', 'value': on_policy_chosen_segment}, {'from':'gpt', 'value': off_policy_chosen}],
                                "rejected":[{'from':'gpt', 'value': on_policy_rejected_segment}, {'from':'gpt', 'value': off_policy_rejected}]}
                                
            conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'iter_{iteration_id}.json'
    write_json_file(conversations, save_path)
    print(f"Saved dpo training dataset to {save_path}")

    print(f"DPO training dataset size: {len(conversations)}")
    print(f"segment_rollout_chosen_acc: {np.mean(segment_rollout_chosen_acc):.4f} (iteration_id={iteration_id})")
    
    chosen_acc_counter = Counter(segment_rollout_chosen_acc)
    print(f"Chosen acc 0 count: {chosen_acc_counter[0]}")
    print(f"Chosen acc 1 count: {chosen_acc_counter[1]}")
    print(f"No enough candidates data count: {no_enough_candidates_data_count}")
    print(f"Chosen and rejected segments with same score count: {chosen_rejected_w_same_score_data_count}")


def create_segment_rollout_input_valid_dataset(completion_file, save_dir, iteration_time, iteration_id):

    def split_into_parts(text, num_parts):
        steps = text.split("\n\n")
        if len(steps) < num_parts:
            return None
        part_size, remainder = divmod(len(steps), num_parts)
        parts, start = [], 0
        for i in range(num_parts):
            end = start + part_size + (1 if i < remainder else 0)
            parts.append('\n\n'.join(steps[start:end]))
            start = end
        return parts

    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)

    segment_rollout_input_dataset = []
    for data in completion_data:
        prompt, answer = data['prompt'], data['answer']

        completion_input = data['completion_inputs'][0]
        segment_data = {
            'prompt': prompt,
            'answer': answer,
            'completion_input': completion_input
        }

        for output in data['completion_outputs'][0]:
            parts = split_into_parts(output, iteration_time)
            if not parts:
                continue

            prefix = '\n\n'.join(parts[0: -iteration_id])
            if prefix:
                prefix = f"{completion_input}{prefix}\n\n"
            else:
                prefix = completion_input
            segment_data[prefix] = {'segment_rollout_inputs': [[prefix]]}

        segment_rollout_input_dataset.append(segment_data)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'input.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(segment_rollout_input_dataset, f, ensure_ascii=False, indent=2)

    print(f"Rollout input dataset size {len(segment_rollout_input_dataset)}")
    print(f"Saved rollout input dataset to {save_path}")


def compare_acc_before_after_seg_rollout(before_seg_rollout_file, after_seg_rollout_file):
    # Load before-seg rollout data
    with open(before_seg_rollout_file, 'r', encoding='utf-8') as f:
        before_data = json.load(f)

    # Build a lookup dictionary for before-seg scores
    before_scores = {
        entry['prompt']: entry['rule_scores'][0][0]
        for entry in before_data
    }

    # Load after-seg rollout data
    with open(after_seg_rollout_file, 'r', encoding='utf-8') as f:
        after_data = json.load(f)

    before_acc = []
    after_acc = []

    for entry in after_data:
        prompt = entry['prompt']
        after_score = entry['rule_scores'][0][0]
        before_score = before_scores.get(prompt, 0)

        after_acc.append(int(bool(after_score)))
        before_acc.append(int(bool(before_score)))

    # Optional: print summary statistics
    total = len(after_acc)
    before_correct = sum(before_acc)
    after_correct = sum(after_acc)
    print(f"Before accuracy: {before_correct}/{total} = {before_correct / total:.2%}")
    print(f"After  accuracy: {after_correct}/{total} = {after_correct / total:.2%}")


def create_on_policy_response_dpo_dataset(on_policy_file, save_dir):
    with open(on_policy_file, 'r', encoding='utf-8') as f:
        on_policy_dataset = json.load(f)
    print(f"Raw input dataset size: {len(on_policy_dataset)}")

    conversations = []
    on_policy_avg_acc = []
    for data in on_policy_dataset:
        prompt = data['prompt']
        all_generated_responses = data['all_generated_responses']
        rule_scores = data['rule_scores']
        orm_scores = data['orm_scores']

        avg_acc = sum(rule_scores) / len(rule_scores)
        on_policy_avg_acc.append(avg_acc)

        if avg_acc == 0 or avg_acc == 1:
            continue

        stats = list(zip(rule_scores, orm_scores, all_generated_responses))

        chosen_response = max(stats, key=lambda x: (x[0], x[1]))[2]
        rejected_response = min(stats, key=lambda x: (x[0], x[1]))[2]
        
        conversations.append({
            "conversations": [{"from": "human", "value": prompt}],
            "chosen": {"from": "gpt", "value": chosen_response},
            "rejected": {"from": "gpt", "value": rejected_response}
        })

    save_path = Path(save_dir) / 'on_policy_dpo_dataset.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
    
    print(f"On-policy DPO dataset size: {len(conversations)}")

    print(f"on_policy_avg_acc mean: {np.mean(on_policy_avg_acc):.4f}")
    acc_counter = Counter(on_policy_avg_acc)
    print(f"on_policy_avg_acc 0 count: {acc_counter[0]}")
    print(f"on_policy_avg_acc 1 count: {acc_counter[1]}")


def create_segment_sft_dataset(
    source_dpo_file: Union[str, Path],
    save_dir: Union[str, Path],
    segment_num: int,
    model_name_or_path: str
) -> None:
    """
    Splits each example’s completion into `segment_num` parts and
    builds prefix/label pairs for SFT data. Saves one JSON per segment index.
    """
    source_path = Path(source_dpo_file)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # load raw data
    with source_path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    print(f"Raw input dataset size: {len(raw)}")

    datasets = defaultdict(list)

    for entry in raw:
        # prompt = entry['conversations'][0]['value']
        # answer = entry['chosen']['value']

        prompt = entry['conversations'][0]['value']
        answer = entry['conversations'][1]['value']

        # apply chat template
        messages = [{"role": "user", "content": prompt}]
        prompt_normal = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # split into steps and skip short examples
        steps = answer.split("\n\n")
        if len(steps) < segment_num:
            continue

        # compute segment boundaries
        base_size, rem = divmod(len(steps), segment_num)
        sizes = [base_size + (1 if i < rem else 0) for i in range(segment_num)]
        boundaries = [0] + [sum(sizes[:i+1]) for i in range(segment_num)]

        # extract segments
        segments = [
            "\n\n".join(steps[boundaries[i]:boundaries[i+1]])
            for i in range(segment_num)
        ]

        # build prefix/label for each segment index
        for idx in range(segment_num):
            prefix = "\n\n".join(segments[:idx])
            # if there's a non-empty prefix, add the separator
            inp = prompt_normal + (prefix + '\n\n' if prefix else "")
            # label is the rest of the segments plus end-of-text token
            label = "\n\n".join(segments[idx:]) + "<|eot_id|>"

            datasets[segment_num - idx].append({
                "conversations": [
                    {"value": inp,   "from": "human"},
                    {"value": label, "from": "gpt"}
                ]
            })

    # save each segment dataset
    for seg_id, items in datasets.items():
        out_file = save_path / f"segment_sft_dataset_{seg_id}.json"
        with out_file.open('w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        print(f"Saved segment {seg_id} dataset to {out_file} (size: {len(items)})")


if __name__ == "__main__":
    if args.post_process_type == 'merge_multi_seeds_completion_chunks_v2':
        merge_multi_seeds_completion_chunks(args.completion_file_dir)

    elif args.post_process_type == "merge_completion_chunks":
        merge_completion_chunks(args.completion_file_dir, 'all_step_rollout_orm_verified')

    elif args.post_process_type == "merge_multi_seeds_segment_rollout_chunks":
        merge_multi_seeds_segment_rollout_chunks(args.completion_file_dir)
    
    elif args.post_process_type == "create_reverse_segment_rollout_dataset":
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        save_dir = Path(args.completion_file_dir).parent
        avg_acc_threshold = 0.8
        create_reverse_segment_rollout_dataset(completion_file, save_dir, avg_acc_threshold, args.iteration_time)
    
    elif args.post_process_type == "create_segment_rollout_input_dataset":
        create_segment_rollout_input_dataset(args.segment_rollout_input_file, args.completion_file_dir, args.iteration_id)

    elif args.post_process_type == 'create_segment_rollout_scoring_input_dataset':
        completion_file = Path(args.completion_file_dir) / 'all_generated_completions.json'
        create_segment_rollout_scoring_input_dataset(completion_file, Path(args.completion_file_dir), args.iteration_id)

    elif args.post_process_type == 'create_align_segment_dpo_dataset':
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        create_align_segment_dpo_dataset(completion_file, args.save_dir, args.iteration_id, args.segment_num)

    elif args.post_process_type == 'create_hybrid_policy_dpo_dataset':
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        create_hybrid_policy_dpo_dataset(completion_file, args.save_dir, args.iteration_id, args.segment_num)

    elif args.post_process_type == 'create_segment_rollout_input_valid_dataset':
        create_segment_rollout_input_valid_dataset(args.segment_rollout_input_file, args.completion_file_dir, args.iteration_time, args.iteration_id)

    elif args.post_process_type == 'create_segment_rollout_input_dataset_w_seg_sft':
        completion_file_dir = Path(args.completion_file_dir)
        file_list = []
        for i in range(args.segment_num):
            file_list.append(completion_file_dir / f'segment_{args.segment_num - i}' / 'completion_with_rm_score' / 'orm_verify' / 'all_step_rollout_orm_verified.json')
        create_segment_rollout_input_dataset_w_seg_sft(file_list, completion_file_dir)

    else:
        before_seg_rollout_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/valid_rollout_input.json'
        after_seg_rollout_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0520/Llama-3.2-3B-Instruct/iterative_reverse_segment_dpo_w_accu_q/iter_3/valid_rollout/completion_with_rm_score/all_step_rollout_rule_verified.json'
        # compare_acc_before_after_seg_rollout(before_seg_rollout_file, after_seg_rollout_file)

        on_policy_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0525/Llama-3.2-3B-Instruct/on_policy/completion_with_rm_score/orm_verify/all_generated_response_orm_verified.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0525/Llama-3.2-3B-Instruct/on_policy'
        # create_on_policy_response_dpo_dataset(on_policy_file, save_dir)

        source_dpo_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/openmathinstruct2-Qwen2.5-72B-Instruct-2527.json'
        source_sft_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-SFT-Data/openmathinstruct2-Qwen2.5-72B-Instruct-6873.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0525/Llama-3.2-3B-Instruct/segment_sft_dataset'
        segment_num = 3
        model_name_or_path = '/GLOBALFS/gznwp_3/qxj/models/Llama-3.2-3B-Instruct'

        create_segment_sft_dataset(source_sft_file, save_dir, segment_num, model_name_or_path)