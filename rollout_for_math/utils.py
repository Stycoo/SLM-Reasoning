from datasets import load_from_disk, concatenate_datasets, Dataset
from pathlib import Path
import json
import random
import os
import statistics
from transformers import AutoTokenizer
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
from typing import Dict, List

def write_json_file(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)


def r1_solution_extraction(dataset_dir, save_dir):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")
    dataset_used = []
    for data in trainset:
        problem = data['problem']
        messages = data['messages']
        answer = data['answer']
        generation = messages[1]['content']
        assert messages[1]['role'] == 'assistant'
        solution = generation.split('</think>\n\n')[-1].strip()
        dataset_used.append({'problem': problem, 'deepseek-r1': solution, 'answer': answer})

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'openr1_solution.json'
    write_json_file(dataset_used, save_path)


def dpo_chosen_to_sft(dataset_dir, save_dir, save_name, keep_think_token):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    dataset_used = []
    for data in trainset:
        prompt = data['conversations'][0]['value']
        chosen = data['chosen']['value']
        
        if chosen.startswith('\n'):
            chosen = chosen[1:]

        if not keep_think_token:
            if chosen.startswith('<think>\n'):
                chosen = chosen[len('<think>\n'):]

        dataset_used.append({"conversations":[{'value':prompt, 'from':'human'}, {'value':chosen, 'from':'gpt'}]})
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{save_name}.json'
    write_json_file(dataset_used, save_path)


def convert_HF_sft_data_format_to_sharegpt(dataset_dir, save_dir, save_name):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    conversations = []
    for conversation in trainset['conversations']:
        conversations.append({'conversations': conversation})
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{save_name}.json'
    write_json_file(conversations, save_path)


def convert_HF_dpo_data_format_to_sharegpt(dataset_dir, save_dir, keep_think_token, model_name_or_path):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    conversations = []
    for data in trainset:
        prompt = data['conversations'][0]['value']
        messages = [{"role": "user", "content": prompt}]
        prompt_normal = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        chosen = data['chosen']['value']
        rejected = data['rejected']['value']

        if chosen.startswith('\n'):
            chosen = chosen[1:]
        if rejected.startswith('\n'):
            rejected = rejected[1:]

        if not keep_think_token:
            if chosen.startswith('<think>\n'):
                chosen = chosen[len('<think>\n'):]
            if rejected.startswith('<think>\n'):
                rejected = rejected[len('<think>\n'):]

        # chosen += '<|eot_id|>'
        # rejected += '<|eot_id|>'

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': prompt_normal}], "chosen":{'from':'gpt', 'value': chosen},
                                "rejected":{'from':'gpt', 'value': rejected}}
        conversations.append(chosen_rejected_data)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'dpo_sharegpt.json'
    # write_json_file(conversations, save_path)


def convert_UWNSL_MATH_long_cot_to_sharegpt(dataset_dir, save_dir):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    conversations = []
    for i, problem in enumerate(trainset['problem']):
        solution = trainset['solution'][i]
        conversation = {"conversations":[{'value': problem, 'from':'human'}, {'value': solution, 'from':'gpt'}]}
        conversations.append(conversation)
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'UWNSL_MATH_long_cot_sharegpt.json'
    write_json_file(conversations, save_path)


def statis_response_len(input_path, model_name_or_path, max_seq_len):
    with open(input_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    over_max_seq_len_statis = defaultdict(list)
    for data in input_data:
        prompt = data['conversations'][0]['value']
        chosen = data['chosen']['value']
        rejected = data['rejected']['value']
        chosen_rejected_pair = [prompt + chosen, prompt + rejected]
        chosen_rejected_len = [len(input_ids) for input_ids in tokenizer(chosen_rejected_pair)['input_ids']]

        if chosen_rejected_len[0] > max_seq_len:
            over_max_seq_len_statis['chosen'].append(chosen_rejected_len[0])
        if chosen_rejected_len[1] > max_seq_len:
            over_max_seq_len_statis['rejected'].append(chosen_rejected_len[1])

    print(f"over_max_seq_len_statis: {over_max_seq_len_statis}")


def create_math_sft_warm_up_dataset(dataset_dir, save_dir, save_name, sample_size, model_name_or_path):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    math_sft_dataset = []
    source_model_statis = defaultdict(int)
    response_len_statis = defaultdict(list)
    for dataset, conversations, source_model in tqdm(zip(trainset['dataset'], trainset['conversations'], trainset['source_model']), 
                                                    total=len(trainset['dataset']), desc="Processing dataset", unit="item"):
        if dataset == 'openmathinstruct2': #and source_model == 'Llama-3.1-405B-Instruct':
            source_model_statis[source_model] += 1

            if source_model == 'Qwen2.5-72B-Instruct':
                math_sft_dataset.append({'conversations': conversations})
            
            # statis response len
            res_len = len(tokenizer(conversations[1]['value']).input_ids)
            response_len_statis[source_model].append(res_len)

    print(f"raw math sample num: {len(math_sft_dataset)}")
    print(f"math source model statis: {source_model_statis}")

    # if sample_size > 0:
    #     math_sft_dataset = random.sample(math_sft_dataset, sample_size)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{save_name}-{len(math_sft_dataset)}.json'
    write_json_file(math_sft_dataset, save_path)

    for source_model, response_len in response_len_statis.items():
        print(f"source model: {source_model}, response len mean: {np.mean(response_len)}, std: {np.std(response_len)}")


def create_math_chosen_sft_warm_up_dataset(dataset_dir, save_dir, save_name, sample_size, model_name_or_path):
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load dataset
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"Raw trainset size: {len(trainset)}")

    math_dpo_dataset = []
    source_model_statis = defaultdict(int)
    response_len_statis = defaultdict(list)

    # Use tqdm to track progress
    for i, (dataset, source_model, prompt, chosen, rejected) in tqdm(enumerate(zip(trainset['dataset'], trainset['source_model'], trainset['prompt'], trainset['chosen'], trainset['rejected'])),
                                            total=len(trainset['dataset']), desc="Processing dataset", unit="item"):
        if dataset == 'openmathinstruct2':
            prompt = prompt[0]['content']
            chosen = chosen[0]['content']
            rejected = rejected[0]['content']

            # Structure the dataset as needed
            chosen_rejected_data = {
                "conversations": [{'from': 'human', 'value': prompt}],
                "chosen": {'from': 'gpt', 'value': chosen},
                "rejected": {'from': 'gpt', 'value': rejected}
            }

            if source_model == 'Qwen2.5-72B-Instruct':
                math_dpo_dataset.append(chosen_rejected_data)

            source_model_statis[source_model] += 1

            res_len = len(tokenizer(chosen).input_ids)
            response_len_statis[source_model].append(res_len)

    print(f"Raw math sample num: {len(math_dpo_dataset)}")
    print(f"Math source model statistics: {source_model_statis}")

    # Apply sample size if specified
    # if sample_size > 0:
    #     math_dpo_dataset = random.sample(math_dpo_dataset, sample_size)

    # Save dataset to the specified directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # Create save directory if not exists
    save_path = save_dir / f'{save_name}-{len(math_dpo_dataset)}.json'
    write_json_file(math_dpo_dataset, save_path)
    print(f"Dataset saved to: {save_path}")

    for source_model, response_len in response_len_statis.items():
        print(f"source model: {source_model}, response len mean: {np.mean(response_len)}, std: {np.std(response_len)}")


def concat_trainset(input_file_list, output_file):
    datasets = []
    for input_file in input_file_list:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            datasets.extend(input_data)

    write_json_file(datasets, output_file)


def create_hybrid_dpo_dataset(replay_file, current_dpo_file, output_file):
    with open(replay_file, 'r', encoding='utf-8') as f:
        replay_data = json.load(f)
        replay_data_dict = defaultdict(dict)
        for date in replay_data:
            completion_input = date['conversations'][0]['value']
            prompt = completion_input.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[0]
            replay_data_dict[prompt] = date
    print(f"replay data size: {len(replay_data_dict)}")

    hybrid_data = []
    with open(current_dpo_file, 'r', encoding='utf-8') as f:
        current_dpo_data = json.load(f)
        print(f"current dpo data size: {len(current_dpo_data)}")
        for data in current_dpo_data:
            completion_input = data['conversations'][0]['value']
            prompt = completion_input.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[0]
            # print(prompt)
            if prompt in replay_data_dict:
                hybrid_pair = {'conversations': [{'value': completion_input, 'from': 'human'}, {'value': replay_data_dict[prompt]['conversations'][0]['value'], 'from': 'human'}],
                               'chosen': [{'value': data['chosen']['value'], 'from': 'gpt'}, {'value': replay_data_dict[prompt]['chosen']['value'], 'from': 'gpt'}],
                               'rejected': [{'value': data['rejected']['value'], 'from': 'gpt'}, {'value': replay_data_dict[prompt]['rejected']['value'], 'from': 'gpt'}]}
                hybrid_data.append(hybrid_pair)

    print(f"hybrid data size: {len(hybrid_data)}")                
    write_json_file(hybrid_data, output_file)


def check_datasize(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        print(len(dataset))


def filter_and_split_data(completion_file, save_dir):
    with open(completion_file, 'r', encoding='utf-8') as f:
        completion_data = json.load(f)
    print(f"Raw data size: {len(completion_data)}")

    completion_data_dict = defaultdict(list)
    filtered_count = 0

    for comp_data in completion_data:
        scores_all = comp_data['rule_scores']
        avg_scores = [sum(scores) / len(scores) for scores in scores_all]
        base_score = avg_scores[0]

        max_index = 0
        for i, avg_score in enumerate(avg_scores):
            if avg_score >= 0.8:
                max_index = i
                break

        if not max_index:
            max_score = max(avg_scores)
            max_index = avg_scores.index(max_score)
        else:
            max_score = avg_scores[max_index]

        if (max_score - base_score) >= 0.4:
            entry = {
                'prompt': comp_data['prompt'],
                'answer': comp_data['answer'],
                **{k: comp_data[k][:max_index + 1] for k in ['completion_inputs', 'completion_outputs', 'rule_scores', 'orm_scores']}
            }
            completion_data_dict[max_index].append(entry)
            filtered_count += 1

    print(f"Filtered data size: {filtered_count}")

    for seg_num, group in sorted(completion_data_dict.items()):
        rel_improvements = [
            max(sum(step) / len(step) for step in sample['rule_scores']) -
            sum(sample['rule_scores'][0]) / len(sample['rule_scores'][0])
            for sample in group
        ]
        avg_improvement = sum(rel_improvements) / len(group) if group else 0.0
        print(f"Segment count: {seg_num}, Samples: {len(group)}, "
              f"Avg. relative improvement: {avg_improvement:.2%}")

    # Select top-50% data by largest seg_num groups
    sorted_groups = sorted(completion_data_dict.items(), key=lambda x: x[0], reverse=True)
    selected_data = []
    target_size = filtered_count // 2

    for _, group in sorted_groups:
        selected_data.extend(group)
        if len(selected_data) >= target_size:
            selected_data = selected_data[:target_size]
            break

    # Split into train/valid (80/20)
    train_size = int(0.8 * len(selected_data))
    train_data = selected_data[:train_size]
    valid_data = selected_data[train_size:]

    # Save outputs
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'train_hard_prompt.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(save_dir / 'valid_hard_prompt.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

    print(f"train_hard_prompt num: {len(train_data)}, valid_hard_prompt num: {len(valid_data)}")


def create_response_sampling_valid_dataset(completion_file, save_dir):
    # Load completions from file
    with open(completion_file, 'r', encoding='utf-8') as f:
        completion_data = json.load(f)

    # Construct the validation dataset
    valid_dataset = [
        {
            'prompt': data['prompt'],
            'answer': data['answer'],
            'completion_inputs': [data['completion_inputs'][0]],
            'completion_outputs': [[data['completion_outputs'][0][0]]],
            'rule_scores': [[data['rule_scores'][0][0]]]
        }
        for data in completion_data
    ]

    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Write to file
    output_file = save_path / 'valid_rollout_input.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_dataset, f, ensure_ascii=False, indent=2)


def create_hard_prompt_dpo_subset(ori_file, hard_prompt_file, save_dir):
    # Load original DPO data
    with open(ori_file, 'r', encoding='utf-8') as f:
        ori_data = json.load(f)
    ori_data_dict = {data['conversations'][0]['value']: data for data in ori_data}

    # Load hard prompts
    with open(hard_prompt_file, 'r', encoding='utf-8') as f:
        hard_prompt_data = json.load(f)
    hard_prompts = {entry['prompt'] for entry in hard_prompt_data}

    # Collect matching subset
    hard_dpo_subset = [
        ori_data_dict[prompt]
        for prompt in hard_prompts
        if prompt in ori_data_dict
    ]

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save to file
    output_file = save_path / 'hard_dpo_subset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hard_dpo_subset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(hard_dpo_subset)} hard DPO examples to {output_file}")


def build_train_subsets(scored_file: str, save_dir):

    def _closest_center(value: float) -> float:
        """返回 value 最接近的中心点（0.2/0.4/0.6/0.8）。"""
        TARGET_CENTERS = [0.2, 0.4, 0.6, 0.8]
        return min(TARGET_CENTERS, key=lambda c: abs(c - value))

    BUCKET_RADIUS = 0.05        # 可根据数据分布调整
    SUBSET_SIZE = 1500          # 每个子集 1500 条

    path = Path(scored_file)
    if not path.is_file():
        raise FileNotFoundError(scored_file)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Raw data size: {len(data)}")

    # 1. 将样本分桶
    buckets: Dict[float, List[dict]] = defaultdict(list)

    for rec in data:
        try:
            rule_scores = rec["rule_scores"][0]
            orm_scores = rec["orm_scores"][0]
            outputs = rec["completion_outputs"][0]
        except (KeyError, IndexError, TypeError):
            continue
        if not rule_scores:
            continue

        avg_acc = sum(rule_scores) / len(rule_scores)

        # 找出最佳/最差 completion
        stats = list(zip(rule_scores, orm_scores, outputs))
        max_completion = max(stats, key=lambda x: (x[0], x[1]))[2]
        min_completion = min(stats, key=lambda x: (x[0], x[1]))[2]

        sample = {
            "conversations": [{"from": "human", "value": rec["prompt"]}],
            "chosen": {"from": "gpt", "value": max_completion},
            "rejected": {"from": "gpt", "value": min_completion},
        }

        center = _closest_center(avg_acc)
        # 只收集四个中心附近的数据
        if abs(avg_acc - center) <= BUCKET_RADIUS:
            buckets[center].append(sample)

    # 2. 生成目标子集
    subsets: Dict[str, List[dict]] = {}

    # 0.2 / 0.4 子集
    for center in (0.2, 0.4):
        if len(buckets[center]) < SUBSET_SIZE:
            raise ValueError(
                f"Bucket {center} 仅有 {len(buckets[center])} 条，无法抽样 {SUBSET_SIZE}"
            )
        subsets[f"avg_{center}"] = random.sample(buckets[center], SUBSET_SIZE)

    # 第三个子集：优先用 0.6，不足则由 0.8 补齐
    bucket_06 = buckets[0.6]
    bucket_08 = buckets[0.8]

    if len(bucket_06) >= SUBSET_SIZE:
        # 0.6 样本足够
        subsets["avg_0.6"] = random.sample(bucket_06, SUBSET_SIZE)
    else:
        # 0.6 不足，需 0.8 补足
        needed = SUBSET_SIZE - len(bucket_06)
        if len(bucket_08) < needed:
            raise ValueError(
                f"Bucket 0.6 + 0.8 总计不足 {SUBSET_SIZE} 条 "
                f"(0.6 有 {len(bucket_06)}，0.8 有 {len(bucket_08)})"
            )
        supplemented = bucket_06 + random.sample(bucket_08, needed)
        random.shuffle(supplemented)  # 打乱顺序
        subsets["avg_0.6_0.8"] = supplemented

    print({k: len(v) for k, v in subsets.items()})

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for name, records in subsets.items():
        filename = f"train_{name}.json"
        path = Path(save_dir) / filename

        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"Saved {path}  ({len(records)} records)")


def create_segment_sft_dataset(input_file, save_dir, iteration_time, model_name_or_path):
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(scored_file)
        
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    segment_sft_datasets = defaultdict(list)
    for i, rec in enumerate(data):
        prompt = rec['prompt']
        answer = rec['answer']
        
        messages = [{"role": "user", "content": prompt}]

        prompt_normal = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        
        completion_steps = answer.split("\n\n")
        if len(completion_steps) < (iteration_time):
            continue
        
        part_size = len(completion_steps) // iteration_time
        remainder = len(completion_steps) % iteration_time

        parts = []
        start_index = 0
        for i in range(iteration_time):
            end_index = start_index + part_size + (1 if i < remainder else 0)
            parts.append('\n\n'.join(completion_steps[start_index: end_index]))
            start_index = end_index

        for i, part in enumerate(parts):
            if i == 0:
                completion_input = prompt_normal
            else: 
                completion_input = prompt_normal + '\n\n'.join(parts[:i]) + '\n\n'

            part += '<|eot_id|>'
            segment_sft_datasets[i].append({"conversations":[{'value': completion_input, 'from':'human'}, 
                                                            {'value': part, 'from':'gpt'}]})

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for segment_id, dataset in segment_sft_datasets.items():
        output_file = save_path / f'segment_sft_dataset_{segment_id}.json'
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"Saved segment {segment_id} dataset to {output_file}, size: {len(dataset)}")


def create_chosen_sft_dataset(input_file, save_dir):
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(input_file)
        
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chosen_sft_dataset = []
    for rec in data:
        prompt = rec['conversations'][0]['value']
        chosen = rec['chosen']['value']

        chosen_sft_dataset.append({"conversations":[{'value':prompt, 'from':'human'}, {'value':chosen, 'from':'gpt'}]})

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_file = save_path / 'openmathinstruct2-Qwen2.5-72B-Instruct-2527-chosen-sft.json'
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(chosen_sft_dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Saved chosen SFT dataset to {output_file}, size: {len(chosen_sft_dataset)}")


if __name__ == '__main__':
    dataset_dir = '/GLOBALFS/gznwp_3/qxj/datasets/Light-R1-DPOData'
    save_name = 'chosen_sft'
    keep_think_token = False
    
    # dpo_chosen_to_sft(dataset_dir, dataset_dir, save_name, keep_think_token)

    model_name_or_path = '/GLOBALFS/gznwp_3/qxj/models/Llama-3.2-3B-Instruct'
    # convert_HF_dpo_data_format_to_sharegpt(dataset_dir, dataset_dir, keep_think_token, model_name_or_path)

    input_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0504/Llama-3.2-3B-Instruct/iterative_align_dpo_reverse_seg_4_w_accu_q_re/iter_2/completion_with_rm_score/iterative_align_dpo_reverse_dataset/1.json'
    # '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_adv_based_misalign_dpo_forward/iter_3/completion_with_rm_score/iterative_adv_based_misalign_dpo_dataset/2.json'
    max_seq_len = 4096
    # statis_response_len(input_path, model_name_or_path, max_seq_len)

    dataset_dir = '/GLOBALFS/gznwp_3/qxj/datasets/FuseChat-3.0-SFT-Data'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-SFT-Data'
    save_name = 'openmathinstruct2-Qwen2.5-72B-Instruct'
    sample_size = 10000
    # create_math_sft_warm_up_dataset(dataset_dir, save_dir, save_name, sample_size, model_name_or_path)

    dataset_dir = '/GLOBALFS/gznwp_3/qxj/datasets/FuseChat-3.0-DPO-Data'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data' 
    save_name = 'openmathinstruct2-Qwen2.5-72B-Instruct'
    # create_math_chosen_sft_warm_up_dataset(dataset_dir, save_dir, save_name, sample_size, model_name_or_path)

    input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/openmathinstruct2-Qwen2.5-72B-Instruct-2527.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-SFT-Data'
    create_chosen_sft_dataset(input_file, save_dir)

    input_file_list = ['/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0430/Llama-3.2-3B-Instruct/iterative_misalign_dpo_reverse_seg_4/iter_1/completion_with_rm_score/iterative_misalign_dpo_reverse_dataset/0.json',
                       '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0430/Llama-3.2-3B-Instruct/iterative_misalign_dpo_reverse_seg_4/iter_2/completion_with_rm_score/iterative_misalign_dpo_reverse_dataset/1.json']
    
    output_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0430/Llama-3.2-3B-Instruct/iterative_misalign_dpo_reverse_seg_4/iter_2/completion_with_rm_score/iterative_misalign_dpo_reverse_dataset/1_w_replay.json'
    # concat_trainset(input_file_list, output_file)
    
    replay_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0430/Llama-3.2-3B-Instruct/iterative_misalign_dpo_reverse_seg_4/iter_1/completion_with_rm_score/iterative_misalign_dpo_reverse_dataset/0.json'
    current_dpo_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0430/Llama-3.2-3B-Instruct/iterative_misalign_dpo_reverse_seg_4/iter_2/completion_with_rm_score/iterative_misalign_dpo_reverse_dataset/1.json'
    output_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0430/Llama-3.2-3B-Instruct/iterative_misalign_dpo_reverse_seg_4/iter_2/completion_with_rm_score/iterative_misalign_dpo_reverse_dataset/1_hybrid.json'
    # create_hybrid_dpo_dataset(replay_file, current_dpo_file, output_file)

    # input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0509/Llama-3.2-3B-Instruct/iterative_reverse_segment_dpo_w_accu_q/rollout_forward/completion_with_rm_score/reverse_segment_rollout_input_iter_4.json'
    completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0509/Llama-3.2-3B-Instruct/iterative_reverse_segment_dpo_w_accu_q/rollout_forward/completion_with_rm_score/reverse_segment_rollout_input_iter_4.json'
    # check_datasize(completion_file)

    completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0509/Llama-3.2-3B-Instruct/segment_rollout_forward_8/completion_with_rm_score/orm_verify/all_step_rollout_orm_verified.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt'
    # filter_and_split_data(completion_file, save_dir)

    completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/valid_hard_prompt.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt'
    # create_response_sampling_valid_dataset(completion_file, save_dir)

    ori_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/openmathinstruct2_source_response_dpo.json'
    hard_prompt_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/train_hard_prompt.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt'
    # create_hard_prompt_dpo_subset(ori_file, hard_prompt_file, save_dir)

    completion_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0509/Llama-3.2-3B-Instruct/segment_rollout_forward_8/completion_with_rm_score/orm_verify/all_step_rollout_orm_verified.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0509/Llama-3.2-3B-Instruct/segment_rollout_forward_8/split_by_avg_acc'
    # build_train_subsets(completion_file, save_dir)

    input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/train_hard_prompt.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public/data/FuseChat-3.0-DPO-Data/0524/Llama-3.2-3B-Instruct/segment_sft_dataset'
    iteration_time = 3

    # create_segment_sft_dataset(input_file, save_dir, iteration_time, model_name_or_path)