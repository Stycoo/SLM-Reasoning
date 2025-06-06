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


def create_iterative_misalign_forward_or_reverse_dpo_dataset(input_path, save_dir, iteration_id, iteration_time, forward_or_reverse, last_iteration_segment_index_file):
    random.seed(42)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    chosen_acc_0_data_count = 0
    chosen_rollout_avg_scores = []
    
    rejected_acc_1_data_count = 0
    rejected_rollout_avg_scores = []

    misalign_dpo_pair_count = 0
    align_dpo_pair_count = 0

    if iteration_id == 1:
        last_iteration_segment_index_dict = {}
        for data in input_data:
            prompt = data['prompt']
            last_iteration_segment_index_dict[prompt] = 1
    else:
        with open(last_iteration_segment_index_file, 'r') as f:
            last_iteration_segment_index_dict = json.load(f)

    conversations = []
    for data in input_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']

        assert len(chosen_rejected) == len(completion_inputs)
        assert chosen_rejected[0] == 1 and chosen_rejected[1] == 0

        rejected_rollouts = completion_outputs[1]
        rejected_rollout_scores = rule_scores[1]
        rejected_rollout_avg_score = sum(rejected_rollout_scores) / len(rejected_rollout_scores)
        rejected_rollout_avg_scores.append(rejected_rollout_avg_score)                

        chosen_completion_input = completion_inputs[0]
        rejected_completion_input = completion_inputs[1]

        if (forward_or_reverse == 'reverse' and last_iteration_segment_index_dict[prompt] == 1) or (forward_or_reverse == 'forward' and iteration_id == iteration_time):
            chosen_rollout_avg_score = 1
            chosen_prefix = ''
            chosen_rollout = chosen_completion_input.split(rejected_completion_input)[-1]
            chosen_rollouts = [chosen_rollout]
            chosen_rollout_scores = [1]
        else:
            chosen_prefix = chosen_completion_input.split(rejected_completion_input)[-1]
            chosen_rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
            chosen_rollouts = completion_outputs[0]
            chosen_rollout_scores = rule_scores[0]
            chosen_rollout_avg_scores.append(chosen_rollout_avg_score)

        chosen_candidates = []
        rejected_candidates = []

        if chosen_rollout_avg_score - rejected_rollout_avg_score > 0.2:
            for i, comp_output in enumerate(chosen_rollouts):
                if chosen_rollout_scores[i] == 1:
                    chosen_candidates.append(chosen_prefix + comp_output)
            
            for i, comp_output in enumerate(rejected_rollouts):
                if rejected_rollout_scores[i] == 0:
                    rejected_candidates.append(comp_output)

            if not rejected_candidates:
                rejected_acc_1_data_count += 1
                continue
            
            misalign_dpo_pair_count += 1
            last_iteration_segment_index_dict[prompt] = iteration_id + 1

        else:
            for i, comp_output in enumerate(rejected_rollouts):
                if rejected_rollout_scores[i] == 0:
                    rejected_candidates.append(comp_output)
                else:
                    chosen_candidates.append(comp_output)
        
            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_acc_0_data_count += 1
                if not rejected_candidates:
                    rejected_acc_1_data_count += 1
                continue
            align_dpo_pair_count += 1

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
    write_json_file(last_iteration_segment_index_dict, last_iteration_segment_index_file)

    print(f"total train data count: {len(conversations)}, misalign dpo pair count: {misalign_dpo_pair_count}, align dpo pair count: {align_dpo_pair_count}")
    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"chosen_rollout_avg_scores: {np.mean(chosen_rollout_avg_scores)}, rejected_rollout_avg_score: {np.mean(rejected_rollout_avg_scores)}")


def create_on_policy_dpo_dataset(input_path, save_dir, iteration_id):
    random.seed(10)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)
    
    conversations = []
    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    on_policy_avg_scores = []

    for data in input_data:
        prompt = data['prompt']
        completion_outputs = data['completion_outputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']

        rejected_candidates = []
        chosen_candidates = []

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
        
        on_policy_avg_score = sum(rule_scores[1]) / len(rule_scores[1])
        on_policy_avg_scores.append(on_policy_avg_score)

        chosen_text = random.choice(chosen_candidates)
        rejected_text = random.choice(rejected_candidates)
    
        chosen_rejected_data = {"conversations":[{'from':'human', 'value': prompt}], "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"total train data count: {len(conversations)}")
    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"on_policy_avg_scores: {np.mean(on_policy_avg_scores)}")


def create_hybrid_policy_dpo_dataset(on_policy_path, off_policy_path, save_dir, iteration_id):
    random.seed(10)
    with open(on_policy_path, 'r', encoding='utf-8') as file:
        on_policy_data = json.load(file)
    
    with open(off_policy_path, 'r', encoding='utf-8') as file:
        off_policy_data = json.load(file)
    off_policy_data_dict = defaultdict(dict)
    for data in off_policy_data:
        prompt = data['conversations'][0]['value']
        chosen = data['chosen']['value']
        rejected = data['rejected']['value']
        off_policy_data_dict[prompt] = {'chosen': chosen, 'rejected': rejected}

    conversations = []
    chosen_acc_0_data_count = 0
    rejected_acc_1_data_count = 0
    on_policy_avg_scores = []

    for data in on_policy_data:
        prompt = data['prompt']
        completion_outputs = data['completion_outputs']
        chosen_rejected = data['chosen_rejected']
        rule_scores = data['rule_scores']
        completion_inputs = data['completion_inputs']

        rejected_candidates = []
        chosen_candidates = []

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
        
        on_policy_avg_score = sum(rule_scores[1]) / len(rule_scores[1])
        on_policy_avg_scores.append(on_policy_avg_score)

        on_policy_chosen = random.choice(chosen_candidates) + '<|eot_id|>'
        on_policy_rejected = random.choice(rejected_candidates) + '<|eot_id|>'

        off_policy_chosen = off_policy_data_dict[prompt]['chosen'] + '<|eot_id|>'
        off_policy_rejected = off_policy_data_dict[prompt]['rejected'] + '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[1]}, {'from':'human', 'value': completion_inputs[1]}], 
                                "chosen":[{'from':'gpt', 'value': on_policy_chosen}, {'from':'gpt', 'value': off_policy_chosen}],
                                "rejected":[{'from':'gpt', 'value': on_policy_rejected}, {'from':'gpt', 'value': off_policy_rejected}]}
        conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"total train data count: {len(conversations)}")
    print(f"chosen_acc_0_data_count: {chosen_acc_0_data_count}, rejected_acc_1_data_count: {rejected_acc_1_data_count}")
    print(f"on_policy_avg_scores: {np.mean(on_policy_avg_scores)}")


def create_iterative_align_forward_or_reverse_dpo_dataset(input_path, save_dir, iteration_id, iteration_time, forward_or_reverse, model_name_or_path, max_seq_len):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    rollout_acc_1_count = 0
    rollout_acc_0_count = 0
    rollout_avg_scores = []

    conversations = []
    for data in input_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        rule_scores = data['rule_scores']

        rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
        rollout_avg_scores.append(rollout_avg_score)   

        chosen_candidates = []
        rejected_candidates = []

        for i, comp_output in enumerate(completion_outputs[0]):
            if rule_scores[0][i] == 0:
                rejected_candidates.append(comp_output)
            elif rule_scores[0][i] == 1:
                chosen_candidates.append(comp_output)
            else:
                continue

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                rollout_acc_0_count += 1
            if not rejected_candidates:
                rollout_acc_1_count += 1
            continue

        chosen_text = random.choice(chosen_candidates)
        rejected_text = random.choice(rejected_candidates)       

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'
        
        # chosen_rejected_pair = [completion_inputs[0] + chosen_text, completion_inputs[0] + rejected_text]
        # chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]
        # if chosen_rejected_len[0] > max_seq_len or chosen_rejected_len[1] > max_seq_len:
        #     continue

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[0]}], 
                                "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"total train data count: {len(conversations)}")
    print(f"rollout_acc_0_count: {rollout_acc_0_count}, rollout_acc_1_count: {rollout_acc_1_count}")
    print(f"rollout_avg_scores: {np.mean(rollout_avg_scores)}")


def create_iterative_align_forward_or_reverse_dpo_dataset_v2(input_path, save_dir, iteration_id, iteration_time, forward_or_reverse, model_name_or_path, max_seq_len):
    random.seed(42)
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    rollout_acc_1_count = 0
    rollout_acc_0_count = 0
    rollout_avg_scores = []

    conversations = []
    for data in input_data:
        prompt = data['prompt']
        answer = data['answer']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        rule_scores = data['rule_scores']

        rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
        rollout_avg_scores.append(rollout_avg_score)   

        chosen_candidates = []
        rejected_candidates = []

        for i, comp_output in enumerate(completion_outputs[0]):
            if rule_scores[0][i] == 0:
                rejected_candidates.append(comp_output)
            else:
                chosen_candidates.append(comp_output)

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                messages = [{"role": "user", "content": prompt}]
                prompt_normal = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)
                completion_prefix = completion_inputs[0].split(prompt_normal)[1]
                assert completion_prefix
                
                chosen_candidate = answer.split(completion_prefix)[1]
                assert chosen_candidate
                chosen_candidates = [chosen_candidate]

                rollout_acc_0_count += 1

            if not rejected_candidates:
                rollout_acc_1_count += 1
                continue

        chosen_text = random.choice(chosen_candidates)
        rejected_text = random.choice(rejected_candidates)       

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'

        chosen_rejected_pair = [completion_inputs[0] + chosen_text, completion_inputs[0] + rejected_text]
        chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]
        if chosen_rejected_len[0] > max_seq_len or chosen_rejected_len[1] > max_seq_len:
            continue

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[0]}], 
                                "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"total train data count: {len(conversations)}")
    print(f"rollout_acc_0_count: {rollout_acc_0_count}, rollout_acc_1_count: {rollout_acc_1_count}")
    print(f"rollout_avg_scores: {np.mean(rollout_avg_scores)}")


def create_iterative_align_forward_or_reverse_dpo_dataset_w_orm_score(input_path, save_dir, iteration_id, iteration_time, forward_or_reverse, model_name_or_path, max_seq_len, dpo_data_sel):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    rollout_acc_1_count = 0
    rollout_acc_0_count = 0
    rollout_avg_scores = []

    conversations = []
    for data in input_data:
        prompt = data['prompt']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']
        rule_scores = data['rule_scores']
        orm_scores = data['orm_scores']

        rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
        rollout_avg_scores.append(rollout_avg_score)   

        chosen_candidates = []
        rejected_candidates = []

        chosen_candidates_orm_scores = []
        rejected_candidates_orm_scores = []

        for i, comp_output in enumerate(completion_outputs[0]):
            if rule_scores[0][i] == 0: # ! there exists -1 case
                rejected_candidates.append(comp_output)
                rejected_candidates_orm_scores.append(orm_scores[0][i])
            elif rule_scores[0][i] == 1: 
                chosen_candidates.append(comp_output)
                chosen_candidates_orm_scores.append(orm_scores[0][i])
            else:
                continue

        if not chosen_candidates or not rejected_candidates:
            if not chosen_candidates:
                rollout_acc_0_count += 1
            if not rejected_candidates:
                rollout_acc_1_count += 1
            continue
        
        if dpo_data_sel == 'random':
            chosen_text = random.choice(chosen_candidates)
            rejected_text = random.choice(rejected_candidates)    

        elif dpo_data_sel == 'orm':
            chosen_index = chosen_candidates_orm_scores.index(max(chosen_candidates_orm_scores))
            chosen_text = chosen_candidates[chosen_index]

            rejected_index = rejected_candidates_orm_scores.index(min(rejected_candidates_orm_scores))
            rejected_text = rejected_candidates[rejected_index]
        
        else:
            raise ValueError("please input valid dpo_data_sel: [random, orm]")

        rejected_text += '<|eot_id|>'
        chosen_text += '<|eot_id|>'
        
        # chosen_rejected_pair = [completion_inputs[0] + chosen_text, completion_inputs[0] + rejected_text]
        # chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]
        # if chosen_rejected_len[0] > max_seq_len or chosen_rejected_len[1] > max_seq_len:
        #     continue

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_inputs[0]}], 
                                "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
        conversations.append(chosen_rejected_data)
        
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"total train data count: {len(conversations)}")
    print(f"rollout_acc_0_count: {rollout_acc_0_count}, rollout_acc_1_count: {rollout_acc_1_count}")
    print(f"rollout_avg_scores: {np.mean(rollout_avg_scores)}")


def create_iterative_align_dpo_for_accumulate_q_dataset(completion_file, completion_w_q_file, save_dir, iteration_id, model_name_or_path, max_seq_len):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
        # completion_data = completion_data[:1000]
        completion_data_dict = defaultdict(dict)
        for data in completion_data:
            prompt = data['prompt']
            rule_scores = data['rule_scores']
            rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
            data['rollout_avg_score'] = rollout_avg_score
            completion_data_dict[prompt] = data

    print(f"raw_completion_data_size: {len(completion_data_dict)}")

    completion_rollout_avg_scores = []
    completion_for_accu_q_rollout_avg_scores = []
    completion_for_accu_q_rollout_avg_improve_scores = []

    chosen_rollout_acc_0_count = 0
    rejected_rollout_acc_1_count = 0
    over_len_data_count = 0

    conversations = []
    with open(completion_w_q_file, 'r', encoding='utf-8') as file:
        completion_w_q_data = json.load(file)
        for data in completion_w_q_data:
            prompt = data['prompt']
            rule_scores_accu_q = data['rule_scores']
            completion_inputs_accu_q = data['completion_inputs']

            baseline_score = completion_data_dict[prompt]['rollout_avg_score']
            completion_input = completion_data_dict[prompt]['completion_inputs'][0]

            _completion_for_accu_q_rollout_avg_scores = []
            for i in range(len(rule_scores_accu_q)):
                rollout_avg_score_i = sum(rule_scores_accu_q[i]) / len(rule_scores_accu_q[i])
                _completion_for_accu_q_rollout_avg_scores.append(rollout_avg_score_i)

            chosen_candidates = []
            rejected_candidates = []

            completion_for_accu_q_rollout_avg_score = sum(_completion_for_accu_q_rollout_avg_scores) / len(_completion_for_accu_q_rollout_avg_scores)
            
            if completion_for_accu_q_rollout_avg_score >= baseline_score and baseline_score < 0.8:
                max_index = _completion_for_accu_q_rollout_avg_scores.index(max(_completion_for_accu_q_rollout_avg_scores))
                min_index = _completion_for_accu_q_rollout_avg_scores.index(min(_completion_for_accu_q_rollout_avg_scores))
                
                chosen_prefix = completion_inputs_accu_q[max_index].split(completion_input)[1]
                rejected_prefix = completion_inputs_accu_q[min_index].split(completion_input)[1]
                
                assert chosen_prefix and rejected_prefix

                for i, comp_output in enumerate(data['completion_outputs'][max_index]):
                    if rule_scores_accu_q[max_index][i] == 1:
                        chosen_candidates.append(chosen_prefix + comp_output)
                
                for i, comp_output in enumerate(data['completion_outputs'][min_index]):
                    if rule_scores_accu_q[min_index][i] == 0:
                        rejected_candidates.append(rejected_prefix + comp_output)

                if chosen_candidates and rejected_candidates:
                    completion_for_accu_q_rollout_avg_scores.append(completion_for_accu_q_rollout_avg_score)
                    completion_rollout_avg_scores.append(baseline_score)

            else:
                completion_outputs = completion_data_dict[prompt]['completion_outputs'][0]
                rule_scores = completion_data_dict[prompt]['rule_scores'][0]
            
                for i, comp_output in enumerate(completion_outputs):
                    if rule_scores[i] == 1:
                        chosen_candidates.append(comp_output)
                    elif rule_scores[i] == 0:
                        rejected_candidates.append(comp_output)
                    else:
                        continue

                if chosen_candidates and rejected_candidates:
                    completion_for_accu_q_rollout_avg_scores.append(baseline_score)
                    completion_rollout_avg_scores.append(baseline_score)

            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_rollout_acc_0_count += 1
                if not rejected_candidates:
                    rejected_rollout_acc_1_count += 1
                continue
            else:
                chosen_text = random.choice(chosen_candidates)
                rejected_text = random.choice(rejected_candidates)       

                rejected_text += '<|eot_id|>'
                chosen_text += '<|eot_id|>'

            # chosen_rejected_pair = [completion_input + chosen_text, completion_input + rejected_text]
            # chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]
            # if chosen_rejected_len[0] > max_seq_len or chosen_rejected_len[1] > max_seq_len:
            #     over_len_data_count += 1
            #     continue

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], 
                                "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
            conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}_re.json'
    write_json_file(conversations, save_path)

    print(f"accumulate_q_data_count: {len(conversations)}")
    print(f"accu_q_rollout_avg_scores: {sum(completion_for_accu_q_rollout_avg_scores) / len(completion_for_accu_q_rollout_avg_scores)}")
    print(f"rollout_avg_scores: {sum(completion_rollout_avg_scores) / len(completion_rollout_avg_scores)}")
    print(f"chosen_rollout_acc_0_count: {chosen_rollout_acc_0_count}, rejected_rollout_acc_1_count: {rejected_rollout_acc_1_count}")
    print(f"max_seq_len:{max_seq_len}, over_len_data_count: {over_len_data_count}")


def create_iterative_align_dpo_for_accumulate_q_dataset_v2(completion_file, completion_w_q_file, save_dir, iteration_id, model_name_or_path, max_seq_len):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
        # completion_data = completion_data[:1000]
        completion_data_dict = defaultdict(dict)
        for data in completion_data:
            prompt = data['prompt']
            rule_scores = data['rule_scores']
            rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
            data['rollout_avg_score'] = rollout_avg_score
            completion_data_dict[prompt] = data

    print(f"raw_completion_data_size: {len(completion_data_dict)}")

    completion_rollout_avg_scores = []
    completion_for_accu_q_rollout_avg_scores = []
    completion_for_accu_q_rollout_avg_improve_scores = []

    chosen_rollout_acc_0_count = 0
    rejected_rollout_acc_1_count = 0

    conversations = []
    with open(completion_w_q_file, 'r', encoding='utf-8') as file:
        completion_w_q_data = json.load(file)
        for data in completion_w_q_data:
            prompt = data['prompt']
            rule_scores_accu_q = data['rule_scores']
            completion_inputs_accu_q = data['completion_inputs']
            answer = data['answer']

            baseline_score = completion_data_dict[prompt]['rollout_avg_score']
            completion_input = completion_data_dict[prompt]['completion_inputs'][0]

            _completion_for_accu_q_rollout_avg_scores = []
            for i in range(len(rule_scores_accu_q)):
                rollout_avg_score_i = sum(rule_scores_accu_q[i]) / len(rule_scores_accu_q[i])
                _completion_for_accu_q_rollout_avg_scores.append(rollout_avg_score_i)

            chosen_candidates = []
            rejected_candidates = []

            completion_for_accu_q_rollout_avg_score = sum(_completion_for_accu_q_rollout_avg_scores) / len(_completion_for_accu_q_rollout_avg_scores)
            
            if completion_for_accu_q_rollout_avg_score >= baseline_score and baseline_score < 0.8:
                max_index = _completion_for_accu_q_rollout_avg_scores.index(max(_completion_for_accu_q_rollout_avg_scores))
                min_index = _completion_for_accu_q_rollout_avg_scores.index(min(_completion_for_accu_q_rollout_avg_scores))
                
                chosen_prefix = completion_inputs_accu_q[max_index].split(completion_input)[1]
                rejected_prefix = completion_inputs_accu_q[min_index].split(completion_input)[1]
                
                assert chosen_prefix and rejected_prefix

                for i, comp_output in enumerate(data['completion_outputs'][max_index]):
                    if rule_scores_accu_q[max_index][i] == 1:
                        chosen_candidates.append(chosen_prefix + comp_output)
                
                for i, comp_output in enumerate(data['completion_outputs'][min_index]):
                    if rule_scores_accu_q[min_index][i] == 0:
                        rejected_candidates.append(rejected_prefix + comp_output)

                if chosen_candidates and rejected_candidates:
                    completion_for_accu_q_rollout_avg_scores.append(completion_for_accu_q_rollout_avg_score)
                    completion_rollout_avg_scores.append(baseline_score)

            else:
                completion_outputs = completion_data_dict[prompt]['completion_outputs'][0]
                rule_scores = completion_data_dict[prompt]['rule_scores'][0]
            
                for i, comp_output in enumerate(completion_outputs):
                    if rule_scores[i] == 1:
                        chosen_candidates.append(comp_output)
                    else:
                        rejected_candidates.append(comp_output)

                if chosen_candidates and rejected_candidates:
                    completion_for_accu_q_rollout_avg_scores.append(baseline_score)
                    completion_rollout_avg_scores.append(baseline_score)

            if not chosen_candidates or not rejected_candidates:

                if not chosen_candidates:
                    messages = [{"role": "user", "content": prompt}]
                    prompt_normal = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)
                    completion_prefix = completion_input.split(prompt_normal)[1]
                    assert completion_prefix
                    
                    chosen_candidate = answer.split(completion_prefix)[1]
                    assert chosen_candidate
                    chosen_candidates = [chosen_candidate]

                    chosen_rollout_acc_0_count += 1

                if not rejected_candidates:
                    rejected_rollout_acc_1_count += 1
                    continue

            # else:
            chosen_text = random.choice(chosen_candidates)
            rejected_text = random.choice(rejected_candidates)       

            rejected_text += '<|eot_id|>'
            chosen_text += '<|eot_id|>'

            chosen_rejected_pair = [completion_input + chosen_text, completion_input + rejected_text]
            chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]
            if chosen_rejected_len[0] > max_seq_len or chosen_rejected_len[1] > max_seq_len:
                continue

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], 
                                "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
            conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"accumulate_q_data_count: {len(conversations)}")
    print(f"accu_q_rollout_avg_scores: {sum(completion_for_accu_q_rollout_avg_scores) / len(completion_for_accu_q_rollout_avg_scores)}")
    print(f"rollout_avg_scores: {sum(completion_rollout_avg_scores) / len(completion_rollout_avg_scores)}")
    print(f"chosen_rollout_acc_0_count: {chosen_rollout_acc_0_count}, rejected_rollout_acc_1_count: {rejected_rollout_acc_1_count}")


def create_iterative_align_dpo_for_accumulate_q_dataset_w_orm_score(completion_file, completion_w_q_file, save_dir, iteration_id, model_name_or_path, max_seq_len, dpo_data_sel):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
        # completion_data = completion_data[:1000]
        completion_data_dict = defaultdict(dict)
        for data in completion_data:
            prompt = data['prompt']
            rule_scores = data['rule_scores']
            rollout_avg_score = sum(rule_scores[0]) / len(rule_scores[0])
            data['rollout_avg_score'] = rollout_avg_score
            completion_data_dict[prompt] = data

    print(f"raw_completion_data_size: {len(completion_data_dict)}")

    completion_rollout_avg_scores = []
    completion_for_accu_q_rollout_avg_scores = []
    completion_for_accu_q_rollout_avg_improve_scores = []

    chosen_rollout_acc_0_count = 0
    rejected_rollout_acc_1_count = 0
    over_len_data_count = 0

    conversations = []
    with open(completion_w_q_file, 'r', encoding='utf-8') as file:
        completion_w_q_data = json.load(file)
        for data in completion_w_q_data:
            prompt = data['prompt']
            rule_scores_accu_q = data['rule_scores']
            completion_inputs_accu_q = data['completion_inputs']
            orm_scores = data['orm_scores']

            baseline_score = completion_data_dict[prompt]['rollout_avg_score']
            completion_input = completion_data_dict[prompt]['completion_inputs'][0]

            _completion_for_accu_q_rollout_avg_scores = []
            for i in range(len(rule_scores_accu_q)):
                rollout_avg_score_i = sum(rule_scores_accu_q[i]) / len(rule_scores_accu_q[i])
                _completion_for_accu_q_rollout_avg_scores.append(rollout_avg_score_i)

            chosen_candidates = []
            rejected_candidates = []

            chosen_candidates_orm_scores = []
            rejected_candidates_orm_scores = []

            completion_for_accu_q_rollout_avg_score = sum(_completion_for_accu_q_rollout_avg_scores) / len(_completion_for_accu_q_rollout_avg_scores)
            
            if completion_for_accu_q_rollout_avg_score >= baseline_score and baseline_score < 0.8:
                max_index = _completion_for_accu_q_rollout_avg_scores.index(max(_completion_for_accu_q_rollout_avg_scores))
                min_index = _completion_for_accu_q_rollout_avg_scores.index(min(_completion_for_accu_q_rollout_avg_scores))
                
                chosen_prefix = completion_inputs_accu_q[max_index].split(completion_input)[1]
                rejected_prefix = completion_inputs_accu_q[min_index].split(completion_input)[1]
                
                assert chosen_prefix and rejected_prefix

                for i, comp_output in enumerate(data['completion_outputs'][max_index]):
                    if rule_scores_accu_q[max_index][i] == 1:
                        chosen_candidates.append(chosen_prefix + comp_output)
                        chosen_candidates_orm_scores.append(orm_scores[max_index][i])

                for i, comp_output in enumerate(data['completion_outputs'][min_index]):
                    if rule_scores_accu_q[min_index][i] == 0:
                        rejected_candidates.append(rejected_prefix + comp_output)
                        rejected_candidates_orm_scores.append(orm_scores[min_index][i])

                if chosen_candidates and rejected_candidates:
                    completion_for_accu_q_rollout_avg_scores.append(completion_for_accu_q_rollout_avg_score)
                    completion_rollout_avg_scores.append(baseline_score)

            else:
                completion_outputs = completion_data_dict[prompt]['completion_outputs'][0]
                rule_scores = completion_data_dict[prompt]['rule_scores'][0]
                orm_scores = completion_data_dict[prompt]['orm_scores'][0]

                for i, comp_output in enumerate(completion_outputs):
                    if rule_scores[i] == 1:
                        chosen_candidates.append(comp_output)
                        chosen_candidates_orm_scores.append(orm_scores[i])
                    elif rule_scores[i] == 0:
                        rejected_candidates.append(comp_output)
                        rejected_candidates_orm_scores.append(orm_scores[i])
                    else:
                        continue

                if chosen_candidates and rejected_candidates:
                    completion_for_accu_q_rollout_avg_scores.append(baseline_score)
                    completion_rollout_avg_scores.append(baseline_score)

            if not chosen_candidates or not rejected_candidates:
                if not chosen_candidates:
                    chosen_rollout_acc_0_count += 1
                if not rejected_candidates:
                    rejected_rollout_acc_1_count += 1
                continue
            else:
                if dpo_data_sel == 'random':
                    chosen_text = random.choice(chosen_candidates)
                    rejected_text = random.choice(rejected_candidates)   

                elif dpo_data_sel == 'orm':
                    chosen_index = chosen_candidates_orm_scores.index(max(chosen_candidates_orm_scores))
                    chosen_text = chosen_candidates[chosen_index]

                    rejected_index = rejected_candidates_orm_scores.index(min(rejected_candidates_orm_scores))
                    rejected_text = rejected_candidates[rejected_index]
        
                else:
                    raise ValueError("please input valid dpo_data_sel: [random, orm]")

                rejected_text += '<|eot_id|>'
                chosen_text += '<|eot_id|>'

            # chosen_rejected_pair = [completion_input + chosen_text, completion_input + rejected_text]
            # chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]
            # if chosen_rejected_len[0] > max_seq_len or chosen_rejected_len[1] > max_seq_len:
            #     over_len_data_count += 1
            #     continue

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], 
                                "chosen":{'from':'gpt', 'value': chosen_text},
                                "rejected":{'from':'gpt', 'value': rejected_text}}
            conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{iteration_id-1}.json'
    write_json_file(conversations, save_path)

    print(f"accumulate_q_data_count: {len(conversations)}")
    print(f"accu_q_rollout_avg_scores: {sum(completion_for_accu_q_rollout_avg_scores) / len(completion_for_accu_q_rollout_avg_scores)}")
    print(f"rollout_avg_scores: {sum(completion_rollout_avg_scores) / len(completion_rollout_avg_scores)}")
    print(f"chosen_rollout_acc_0_count: {chosen_rollout_acc_0_count}, rejected_rollout_acc_1_count: {rejected_rollout_acc_1_count}")
    print(f"max_seq_len:{max_seq_len}, over_len_data_count: {over_len_data_count}")


def create_reverse_segment_rollout_dataset(completion_file, save_dir, avg_acc_threshold, iteration_time):
    '''
    input data format: {'prompt':, 'answer':, 'completion_inputs':[], 'completion_outputs':[[],...], 'rule_scores':[[],...], 'orm_scores':[[],...]}
    output data format: {'prompt':, 'answer':, 'completion_inputs':[], 'completion_outputs':[[],...], 'rule_scores':[[],...], 'orm_scores':[[],...]}
    '''
    reverse_segment_rollout_input_dataset = []

    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)

    for data in completion_data:
        prompt = data['prompt']
        answer = data['answer']
        inputs_all = data['completion_inputs']
        outputs_all = data['completion_outputs']
        scores_all = data['rule_scores']
        orm_scores_all = data['orm_scores']

        for i, rule_score in enumerate(scores_all):
            if i < len(scores_all) - 1:
                assert inputs_all[i] in inputs_all[i + 1]

            avg_score = sum(rule_score) / len(rule_score)
            # if avg_score >= avg_acc_threshold and i >= iteration_time:
            if i >= iteration_time:
                # Initialize with prefix
                used_inputs = [inputs_all[0]]
                used_outputs = [outputs_all[0]]
                used_scores = [scores_all[0]]
                used_orm_scores = [orm_scores_all[0]]

                rollout_inputs = inputs_all[1:i+1]
                rollout_outputs = outputs_all[1:i+1]
                rollout_scores = scores_all[1:i+1]
                rollout_orm_scores = orm_scores_all[1:i+1]

                total_len = len(rollout_inputs)
                chunk_size, remainder = divmod(total_len, iteration_time)

                start = 0
                for t in range(iteration_time):
                    end = start + chunk_size + (1 if t < remainder else 0)
                    if end > start:
                        used_inputs.append(rollout_inputs[end - 1])
                        used_outputs.append(rollout_outputs[end - 1])
                        used_scores.append(rollout_scores[end - 1])
                        used_orm_scores.append(rollout_orm_scores[end - 1])
                    start = end

                reverse_segment_rollout_input_dataset.append({
                    'prompt': prompt,
                    'answer': answer,
                    'completion_inputs': used_inputs,
                    'completion_outputs': used_outputs,
                    'rule_scores': used_scores,
                    'orm_scores': used_orm_scores
                })
                break  # Stop checking further segments once condition is met

    # Save the processed dataset
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'reverse_segment_rollout_input_iter_{iteration_time}.json'
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(reverse_segment_rollout_input_dataset, f, ensure_ascii=False, indent=2)

    print(f"Start rollout input dataset size {len(reverse_segment_rollout_input_dataset)}")
    print(f"Saved start rollout input dataset to {save_path}")


def create_segment_rollout_input_dataset(completion_file, save_dir, rollout_id, iteration_id):

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

        if rollout_id == 1:
            completion_input = data['completion_inputs'][-iteration_id - 1]
            segment_data = {
                'prompt': prompt,
                'answer': answer,
                'completion_input': completion_input
            }

            for output in data['completion_outputs'][-iteration_id - 1]:
                parts = split_into_parts(output, iteration_id + 1)
                if not parts:
                    continue
                prefix = f"{completion_input}{parts[0]}\n\n"
                segment_data[prefix] = {'segment_rollout_inputs': [[prefix]]}

            next_input = data['completion_inputs'][-iteration_id]
            segment_data[next_input] = {'segment_rollout_inputs': [[next_input]]}

        else:
            segment_data = {
                'prompt': prompt,
                'answer': answer,
                'completion_input': data['completion_input']
            }

            seg_num = iteration_id - rollout_id + 2
            for key, value in data.items():
                if key in segment_data:
                    continue
                combined = []
                for inp, out_group in zip(value['segment_rollout_inputs'], value['segment_rollout_outputs']):
                    current_segment = []
                    for out in out_group:
                        parts = split_into_parts(out, seg_num)
                        if not parts:
                            continue
                        current_segment.append(f"{inp}{parts[0]}\n\n")
                    combined.append(current_segment)
                segment_data[key] = {'segment_rollout_inputs': combined}

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


def create_align_segment_dpo_dataset(completion_file, save_dir, iteration_id):
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

    invalid_data_count = 0
    conversations = []
    with open(completion_file, 'r', encoding='utf-8') as file:
        completion_data = json.load(file)
        for data in completion_data:
            completion_input = data['completion_input']
            segment_rollout_prefixs = data['segment_rollout_prefixs']
            rule_scores = data['rule_scores']
            orm_scores = data['orm_scores']

            num_segments = len(segment_rollout_prefixs)
            # assert len(rule_scores) % num_segments == 0, "rule_scores cannot be evenly split"
            # assert len(orm_scores) % num_segments == 0, "orm_scores cannot be evenly split"

            if len(rule_scores) % num_segments:
                invalid_data_count += 1
                continue

            group_size_rule = len(rule_scores) // num_segments
            group_size_orm = len(orm_scores) // num_segments

            rule_avg = avg_group_scores(rule_scores, group_size_rule)
            orm_avg = avg_group_scores(orm_scores, group_size_orm)

            stats = list(zip(rule_avg, orm_avg, segment_rollout_prefixs))

            # Define sorting logic for max/min
            max_prefix = max(stats, key=lambda x: (x[0], x[1]))[2]
            min_prefix = min(stats, key=lambda x: (x[0], x[1]))[2]
            
            chosen_segment = max_prefix.split(completion_input)[1]
            rejected_segment = min_prefix.split(completion_input)[1]
            assert chosen_segment and rejected_segment

            if chosen_segment == rejected_segment:
                continue

            chosen_segment += '<|eot_id|>'
            rejected_segment += '<|eot_id|>'

            chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], 
                                "chosen":{'from':'gpt', 'value': chosen_segment},
                                "rejected":{'from':'gpt', 'value': rejected_segment}}
            conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'iter_{iteration_id}.json'
    write_json_file(conversations, save_path)

    print(f"DPO training dataset size: {len(conversations)}")
    print(f"Saved dpo training dataset to {save_path}")
    print(f"invalid_data_count: {invalid_data_count}")


def create_segment_rollout_input_valid_dataset(completion_file, save_dir, iteration_time, rollout_id, iteration_id):

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

        if rollout_id == 1:
            completion_input = data['completion_inputs'][0]
            segment_data = {
                'prompt': prompt,
                'answer': answer,
                'completion_input': completion_input
            }

            for output in data['completion_outputs'][0]:
                parts = split_into_parts(output, iteration_time + 1)
                if not parts:
                    continue
                prefix = '\n\n'.join(parts[0: -iteration_id - 1])
                prefix = f"{completion_input}{prefix}\n\n"
                segment_data[prefix] = {'segment_rollout_inputs': [[prefix]]}

        else:
            segment_data = {
                'prompt': prompt,
                'answer': answer,
                'completion_input': data['completion_input']
            }

            seg_num = iteration_id - rollout_id + 3
            for key, value in data.items():
                if key in segment_data:
                    continue
                combined = []
                for inp, out_group in zip(value['segment_rollout_inputs'], value['segment_rollout_outputs']):
                    current_segment = []
                    for out in out_group:
                        parts = split_into_parts(out, seg_num)
                        if not parts:
                            continue
                        current_segment.append(f"{inp}{parts[0]}\n\n")
                    combined.append(current_segment)
                segment_data[key] = {'segment_rollout_inputs': combined}

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


if __name__ == "__main__":
    if args.post_process_type == 'merge_multi_seeds_completion_chunks_v2':
        merge_multi_seeds_completion_chunks(args.completion_file_dir)

    elif args.post_process_type == "merge_completion_chunks":
        merge_completion_chunks(args.completion_file_dir, 'all_step_rollout_orm_verified')

    elif args.post_process_type == "merge_multi_seeds_segment_rollout_chunks":
        merge_multi_seeds_segment_rollout_chunks(args.completion_file_dir)
    
    elif args.post_process_type == "create_misaligned_segment_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / f'iterative_misalign_dpo_{args.forward_or_reverse}_dataset'
        create_iterative_misalign_forward_or_reverse_dpo_dataset(input_path, save_dir, args.iteration_id, args.iteration_time, args.forward_or_reverse, args.last_iteration_segment_index_file)

    elif args.post_process_type == "create_aligned_segment_dpo_dataset":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / f'iterative_align_dpo_{args.forward_or_reverse}_dataset'
        create_iterative_align_forward_or_reverse_dpo_dataset(input_path, save_dir, args.iteration_id, args.iteration_time, args.forward_or_reverse, args.model_name_or_path, args.max_seq_len)

    elif args.post_process_type == "create_aligned_segment_accu_q_dpo_dataset":
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
        completion_w_q_file = Path(args.completion_accu_q_dir) / 'all_step_rollout_rule_verified.json'
        save_dir = Path(args.completion_file_dir) / f'iterative_align_dpo_reverse_dataset'
        create_iterative_align_dpo_for_accumulate_q_dataset(completion_file, completion_w_q_file, save_dir, args.iteration_id, args.model_name_or_path, args.max_seq_len)
    
    # elif args.post_process_type == "create_aligned_segment_dpo_dataset_v2":
    #     input_path = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
    #     save_dir = Path(args.completion_file_dir) / f'iterative_align_dpo_{args.forward_or_reverse}_dataset'
    #     create_iterative_align_forward_or_reverse_dpo_dataset_v2(input_path, save_dir, args.iteration_id, args.iteration_time, args.forward_or_reverse, args.model_name_or_path, args.max_seq_len)

    # elif args.post_process_type == "create_aligned_segment_accu_q_dpo_dataset_v2":
    #     completion_file = Path(args.completion_file_dir) / 'all_step_rollout_rule_verified.json'
    #     completion_w_q_file = Path(args.completion_accu_q_dir) / 'all_step_rollout_rule_verified.json'
    #     save_dir = Path(args.completion_file_dir) / f'iterative_align_dpo_reverse_dataset'
    #     create_iterative_align_dpo_for_accumulate_q_dataset_v2(completion_file, completion_w_q_file, save_dir, args.iteration_id, args.model_name_or_path, args.max_seq_len)

    elif args.post_process_type == "create_aligned_segment_dpo_dataset_w_orm":
        input_path = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        save_dir = Path(args.completion_file_dir).parent / f'iterative_align_dpo_{args.forward_or_reverse}_dataset'
        create_iterative_align_forward_or_reverse_dpo_dataset_w_orm_score(input_path, save_dir, args.iteration_id, args.iteration_time, args.forward_or_reverse, args.model_name_or_path, args.max_seq_len, args.dpo_data_sel)

    elif args.post_process_type == "create_aligned_segment_accu_q_dpo_dataset_w_orm":
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        completion_w_q_file = Path(args.completion_accu_q_dir) / 'all_step_rollout_orm_verified.json'
        save_dir = Path(args.completion_file_dir).parent / f'iterative_align_dpo_reverse_dataset'
        create_iterative_align_dpo_for_accumulate_q_dataset_w_orm_score(completion_file, completion_w_q_file, save_dir, args.iteration_id, args.model_name_or_path, args.max_seq_len, args.dpo_data_sel)

    elif args.post_process_type == "create_reverse_segment_rollout_dataset":
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        save_dir = Path(args.completion_file_dir).parent
        avg_acc_threshold = 0.8
        create_reverse_segment_rollout_dataset(completion_file, save_dir, avg_acc_threshold, args.iteration_time)
    
    elif args.post_process_type == "create_segment_rollout_input_dataset":
        create_segment_rollout_input_dataset(args.segment_rollout_input_file, args.completion_file_dir, args.rollout_id, args.iteration_id)

    elif args.post_process_type == 'create_segment_rollout_scoring_input_dataset':
        completion_file = Path(args.completion_file_dir) / 'all_generated_completions.json'
        create_segment_rollout_scoring_input_dataset(completion_file, Path(args.completion_file_dir), args.iteration_id)

    elif args.post_process_type == 'create_align_segment_dpo_dataset':
        completion_file = Path(args.completion_file_dir) / 'all_step_rollout_orm_verified.json'
        create_align_segment_dpo_dataset(completion_file, args.save_dir, args.iteration_id)

    elif args.post_process_type == 'create_segment_rollout_input_valid_dataset':
        create_segment_rollout_input_valid_dataset(args.segment_rollout_input_file, args.completion_file_dir, args.iteration_time, args.rollout_id, args.iteration_id)

    else:
        # raise ValueError("please set the valid post_process_type")
        input_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/Llama-3.2-3B-Instruct/iterative_misalign_dpo_forward_seg_4/iter_1/completion_with_rm_score/all_step_rollout_rule_verified.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/Llama-3.2-3B-Instruct/on_policy_dpo'
        iteration_id = 1
        # create_on_policy_dpo_dataset(input_path, save_dir, iteration_id)

        on_policy_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/Llama-3.2-3B-Instruct/iterative_misalign_dpo_forward_seg_4/iter_1/completion_with_rm_score/all_step_rollout_rule_verified.json'
        off_policy_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/openmathinstruct2_source_response_dpo.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/Llama-3.2-3B-Instruct/hybrid_policy_dpo'
        # create_hybrid_policy_dpo_dataset(on_policy_path, off_policy_path, save_dir, iteration_id)

        completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0504/Llama-3.2-3B-Instruct/iterative_align_dpo_reverse_seg_4_w_accu_q/iter_2/completion_with_rm_score/all_step_rollout_rule_verified.json'
        completion_w_q_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0504/Llama-3.2-3B-Instruct/iterative_align_dpo_reverse_seg_4_w_accu_q/iter_2/completion_for_accu_q_iter_1/completion_with_rm_score/all_step_rollout_rule_verified.json'
        # create_iterative_align_dpo_for_accumulate_q_dataset(completion_file, completion_w_q_file)

        completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/train_hard_prompt.json'
        save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/iterative_reverse_segment_dpo_w_accu_q'
        avg_acc_threshold = 0.8
        # create_reverse_segment_rollout_dataset(completion_file, save_dir, avg_acc_threshold, 2)

        before_seg_rollout_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/valid_rollout_input.json'
        after_seg_rollout_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/iterative_reverse_segment_dpo_w_accu_q/iter_2/valid_rollout_3/completion_with_rm_score/all_step_rollout_rule_verified.json'
        compare_acc_before_after_seg_rollout(before_seg_rollout_file, after_seg_rollout_file)