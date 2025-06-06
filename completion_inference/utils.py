from datasets import load_from_disk, concatenate_datasets, Dataset
from pathlib import Path
import json
import random
import os
import statistics
from transformers import AutoTokenizer
from collections import defaultdict


def write_json_file(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)


def data_filtering_for_openr1_math(dataset_dir, save_dir):
    # 推理输出个数大于3且答案正确
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")
    dataset_used = []
    for data in trainset:
        correctness_count = data['correctness_count']
        if correctness_count > 3:
            problem = data['problem']
            generations = data['generations']
            answer = data['answer']
            dataset_used.append({'problem': problem, 'deepseek-r1': generations, 'answer': answer})
    print(f"filtered trainset size: {len(dataset_used)}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'openr1_math_reasoning_path_num_over_3_{len(dataset_used)}.json'
    write_json_file(dataset_used, save_path)


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


def create_subset_by_random_sampling_with_step_num_constrain(input_file, save_dir, min_step_num, max_step_num, subset_size):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"raw trainset size: {len(dataset)}")
    dataset_used = []
    max_step_num_statis = 0
    for data in dataset:
        r1_solution = data['deepseek-r1']
        step_num = len(r1_solution.split("\n\n"))
        if step_num >= min_step_num and step_num <= max_step_num:
            dataset_used.append(data)
            if step_num > max_step_num_statis:
                max_step_num_statis = step_num
    
    print(f"max step num: {max_step_num_statis}")
    print(f"filtered trainset size: {len(dataset_used)}")

    if subset_size > 0:
        random.seed(42)
        random.shuffle(dataset_used)
        dataset_used = dataset_used[:subset_size]
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'solution.json'
    write_json_file(dataset_used, save_path)


def convert_sft_data_format_to_sharegpt(input_file, save_dir, file_name):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    sharegpt_dataset = []
    for data in dataset:
        prompt = data['problem'] + " Let's think step by step and output the final answer within \\boxed{}."
        conversation = [{'from': 'human', 'value': prompt}, 
                        {'from': 'gpt', 'value': data['deepseek-r1']}]
        sharegpt_dataset.append({'conversations': conversation})
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'sft_{file_name}.json'
    write_json_file(sharegpt_dataset, save_path)


def select_dpo_chosen_to_sft(dataset_dir, save_dir, save_name):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    dataset_used = []
    for data in trainset:
        # conversation = data['conversations'][0]
        prompt = data['prompt'][0]['content']
        chosen = data['chosen'][0]['content']
        dataset_used.append({"conversations":[{'value':prompt, 'from':'human'}, {'value':chosen, 'from':'gpt'}]})
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{save_name}.json'
    write_json_file(dataset_used, save_path)


def data_check(dataset_dir):
    trainset = load_from_disk(Path(dataset_dir) / 'train')
    print(f"raw trainset size: {len(trainset)}")
    think_step_statis = []
    for data in trainset:
        chosen = data['chosen']
        response = chosen['value']
        think_step_statis.append(len(response.split('</think>')[0].split('\n\n')))

    print(f"think_step_statis: min-{min(think_step_statis)}, max-{max(think_step_statis)}, avg-{sum(think_step_statis)/len(think_step_statis)}")


def data_len_check(input_file, split_token):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"raw trainset size: {len(dataset)}")
    data_len_statis = []
    for data in dataset:
        conversations = data['conversations']
        response = conversations[1]['value']
        data_len = len(response.split(split_token))
        data_len_statis.append(data_len)

    min_len = min(data_len_statis)
    max_len = max(data_len_statis)
    gap = (max_len - min_len) // 10

    length_distribution = {f"{min_len + i * gap}-{min_len + (i + 1) * gap}": 0 for i in range(10)}
    length_distribution[f"{min_len + 10 * gap}+"] = 0

    for length in data_len_statis:
        index = (length - min_len) // gap
        if index >= 10:
            length_distribution[f"{min_len + 10 * gap}+"] += 1
        else:
            length_distribution[f"{min_len + index * gap}-{min_len + (index + 1) * gap}"] += 1
    
    print(f"Split token: {repr(split_token)}, Length distribution: {length_distribution}")
    # print(f"data_len_statis min: {min(data_len_statis)}, max: {max(data_len_statis)}, avg: {sum(data_len_statis) / len(data_len_statis)}")


def chunk_dataset_concat(parent_dir, save_dir):
    chunk_datasets = []
    for folder in sorted(os.listdir(parent_dir)):
        if folder.startswith("chunk_"):
            chunk_dir = Path(parent_dir) / folder / 'train'
            # 加载数据集
            ds = load_from_disk(chunk_dir)
            chunk_datasets.append(ds)

    # 将所有chunk数据集合并为一个数据集
    combined_dataset = concatenate_datasets(chunk_datasets)
    combined_dataset.save_to_disk(save_dir)


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


def split_dataset_by_difficulty(original_file, save_dir, sampling_size, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # 读取原始数据
    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算每个 item 的 step 数（以 "\n\n" 切分并过滤空字符串）
    for item in data:
        steps = [s for s in item['deepseek-r1'].split("\n\n") if s.strip()]
        item['step_count'] = len(steps)
    
    # 计算所有 step 数的中位数
    step_counts = [item['step_count'] for item in data]
    median_steps = statistics.median(step_counts)
    print("中位数 step 数:", median_steps)
    
    # 划分数据：低难度 (step_count ≤ 中位数) 与 高难度 (step_count > 中位数)
    low_diff = [item for item in data if item['step_count'] <= median_steps]
    high_diff = [item for item in data if item['step_count'] > median_steps]
    
    # 定义采样函数：从组中随机采样 sample_size 条数据作为训练集
    def sample_training(group, sample_size):
        available = len(group)
        if available < sample_size:
            print(f"警告: 当前组数据量 {available} 小于要求采样数 {sample_size}，采样全部数据。")
            return group.copy()
        return random.sample(group, sample_size)
    
    # 对低难度和高难度组分别采样训练集
    low_train = sample_training(low_diff, sampling_size)
    high_train = sample_training(high_diff, sampling_size)
    
    def save_as_sharegpt_format(dataset):
        dataset_new = []
        for data in dataset:
            conversation = [{'from': 'human', 'value': data['problem']}, {'from': 'gpt', 'value': data['deepseek-r1']}]
            dataset_new.append({'conversations': conversation})
        return dataset_new

    low_train = save_as_sharegpt_format(low_train)
    high_train = save_as_sharegpt_format(high_train)
    
    # 创建保存目录（若不存在则自动创建）
    save_dir = Path(save_dir) / f'difficulty_split_by_{median_steps}_steps'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 定义保存文件路径
    low_train_file = os.path.join(save_dir, "low_train.json")
    high_train_file = os.path.join(save_dir, "high_train.json")
    
    # 保存训练集数据到文件
    with open(low_train_file, 'w', encoding='utf-8') as f:
        json.dump(low_train, f, ensure_ascii=False, indent=4)
    with open(high_train_file, 'w', encoding='utf-8') as f:
        json.dump(high_train, f, ensure_ascii=False, indent=4)
    
    print(f"低难度训练集：{len(low_train)} 条")
    print(f"高难度训练集：{len(high_train)} 条")
    print(f"所有文件已保存至目录: {save_dir}")


def create_subset_by_random_sampling(input_file, save_dir, sampling_rate):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"raw trainset size: {len(dataset)}")

    subset_size = int(len(dataset) * sampling_rate)
    if subset_size > 0:
        random.seed(42)
        random.shuffle(dataset)
        dataset = dataset[:subset_size]
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'openmathinstruct2_validation_{subset_size}.json'
    write_json_file(dataset, save_path)


def creat_math_valid_dataset_for_fusechat3_from_DPO_trainset(dataset_path, save_dir):
    trainset = load_from_disk(Path(dataset_path) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    dataset_used = []
    for data in trainset:
        if data['dataset'] == 'openmathinstruct2':
            prompt = data['prompt'][0]['content']
            chosen = data['chosen'][0]['content']
            rejected = data['rejected'][0]['content']
            # dataset_used.append({"conversations":[{'value':prompt, 'from':'human'}, {'value':chosen, 'from':'gpt'}]})
            chosen_rejected_data = {"conversations":[{'from':'human', 'value': prompt}], "chosen":{'from':'gpt', 'value': chosen},
                                    "rejected":{'from':'gpt', 'value': rejected}}
            dataset_used.append(chosen_rejected_data)

    print(f"openmathinstruct2 subset size: {len(dataset_used)}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'openmathinstruct2_source_response_dpo.json'
    write_json_file(dataset_used, save_path)


def creat_math_valid_dataset_for_fusechat3_from_SFT_trainset(dataset_path, save_dir):
    trainset = load_from_disk(Path(dataset_path) / 'train')
    print(f"raw trainset size: {len(trainset)}")

    dataset_used = []
    for data in trainset:
        if data['dataset'] == 'openmathinstruct2':
            dataset_used.append({"conversations": data['conversations']})
    print(f"openmathinstruct2 subset size: {len(dataset_used)}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'openmathinstruct2_validation.json'
    write_json_file(dataset_used, save_path)


def add_rm_score_to_step_ce_dataset(dataset_path, tokenizer_path, rm_score_path):
    trainset = load_from_disk(Path(dataset_path))
    print(f"raw trainset size: {len(trainset)}")
    
    rm_score_dict = defaultdict()
    with open(rm_score_path, 'r', encoding='utf-8') as f:
        rm_score = json.load(f)
        for data in rm_score:
            prompt = data['prompt'].strip()
            rm_score_dict[prompt] = data

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=False,
        use_fast=False,
    )

    error_data_count = 0
    data_used = []
    for data in trainset:
        input_ids = data['input_ids']
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
       
        if "llama-3"in tokenizer.name_or_path.lower():
            prompt = input_text.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].split('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n')[-1].strip()
        # assert prompt in rm_score_dict, print(f"prompt: {prompt} not in rm_score_dict") 
        if prompt not in rm_score_dict:
            error_data_count += 1
            continue
        # data['rm_score'] = rm_score_dict[prompt]['all_rm_scores']
        data['rm_score'] = rm_score_dict[prompt]['rule_scores']
        data_used.append(data)
    print(f"error_data_count: {error_data_count}")
    
    save_dir = Path(dataset_path) / 'rm_score_added'
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    data_used = Dataset.from_list(data_used)
    data_used.save_to_disk(save_dir)


def create_pseudo_step_weighted_data(input_file, save_path):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    for data in dataset:
        step_num = len(data['conversations'][1]['value'].split('\n\n'))
        step_scores = [1] * step_num
        data['step_scores'] = step_scores

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


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


def create_self_sampling_dpo_pair_from_all_step_rollout_rule_verified_data(input_path, output_path):
    random.seed(42)
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    self_samping_dpo_pair = []
    all_wrong_data_count = 0
    all_correct_data_count = 0
    for data in dataset:
        completion_input = data['completion_inputs'][0] 
        assert data['chosen_rejected'][1] == 0

        chosen_candidates = []
        rejected_candidates = []

        for i, completion_output in enumerate(data['completion_outputs'][1]):
            if data['rule_scores'][1][i] == 1:
                chosen_candidates.append(completion_output)
            else:
                rejected_candidates.append(completion_output)
        
        if len(chosen_candidates) == 0 or len(rejected_candidates) == 0:
            if len(chosen_candidates) == 0:
                all_wrong_data_count += 1
            else:
                all_correct_data_count += 1
            continue
        
        chosen = random.choice(chosen_candidates)
        rejected = random.choice(rejected_candidates)

        chosen_rejected_data = {"conversations":[{'from':'human', 'value': completion_input}], "chosen":{'from':'gpt', 'value': chosen},
                                    "rejected":{'from':'gpt', 'value': rejected}}
        self_samping_dpo_pair.append(chosen_rejected_data)
    
    print(f"all_wrong_data_count: {all_wrong_data_count}")
    print(f"all_correct_data_count: {all_correct_data_count}")
    print(f"self_samping_dpo_pair size: {len(self_samping_dpo_pair)}")

    save_path = Path(output_path)
    write_json_file(self_samping_dpo_pair, save_path)


if __name__ == '__main__':
    dataset_dir = '/GLOBALFS/gznwp_3/qxj/datasets/OpenR1-Math-220k'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math'
    # r1_solution_extraction(dataset_dir, save_dir)

    openr1_solution_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution.json'
    min_step_num = 8
    max_step_num = 10
    subset_size = 10000
    save_dir = f'/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution_max_{max_step_num}_step_min_{min_step_num}_step_datasize_{subset_size}'
    # create_subset_by_random_sampling_with_step_num_constrain(openr1_solution_file, save_dir, min_step_num, max_step_num, subset_size)

    input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution_max_10_step_min_8_step_datasize_10000/solution.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/data/model_fusion/0317'
    file_name = 'base'
    # convert_sft_data_format_to_sharegpt(input_file, save_dir, file_name)
    
    dataset_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/UWNSL_MATH_training_split_long_cot'
    # '/GLOBALFS/gznwp_3/qxj/datasets/FuseChat-3.0-DPO-Data'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data'
    save_name = 'chosen_content'
    # select_dpo_chosen_to_sft(dataset_dir, save_dir, save_name)

    dataset_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/UWNSL_MATH_training_split_long_cot'
    # '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-SFT-Data'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/UWNSL_MATH_long_cot'
    save_name = 'gt_content'
    # convert_HF_sft_data_format_to_sharegpt(dataset_dir, save_dir, save_name)
    # convert_UWNSL_MATH_long_cot_to_sharegpt(dataset_dir, save_dir)

    # data_check(dataset_dir)

    Light_R1_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/chosen_logp_checking/chosen_content.json'
    # '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/data/model_fusion/chosen_sft/Light-R1-DPOData_chosen_sft.json'
    # data_len_check(Light_R1_file, '\n\n')

    s1_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/data/model_fusion/sharegpt-s1k-math-r1.json'
    # data_len_check(s1_file)

    parent_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/chosen_logp_checking'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/chosen_logp_checking/all_log_checking_dataset'
    # chunk_dataset_concat(parent_dir, save_dir)

    original_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math'
    sampling_size = 10000
    # split_dataset_by_difficulty(original_file, save_dir, sampling_size, seed=42)

    input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-SFT-Data/openmathinstruct2_validation.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-SFT-Data'
    sampling_rate = 0.25
    # create_subset_by_random_sampling(input_file, save_dir, sampling_rate)

    dataset_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-SFT-Data/FuseChat-Llama-3.2-3B-SFT/trainset_math_gt_logp_checking/step_ce_dataset'
    tokenizer_path = '/GLOBALFS/gznwp_3/qxj/models/FuseChat-Llama-3.2-3B-SFT'
    rm_score_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-SFT-Data/FuseChat-Llama-3.2-3B-SFT_openmathinstruct2_trainset_valid_infer/response_with_rm_score/all_generated_response_rule_verified.json'
    # add_rm_score_to_step_ce_dataset(dataset_path, tokenizer_path, rm_score_path)

    dataset_path = '/GLOBALFS/gznwp_3/qxj/datasets/FuseChat-3.0-DPO-Data'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data'
    # creat_math_valid_dataset_for_fusechat3_from_DPO_trainset(dataset_path, save_dir)
    # creat_math_valid_dataset_for_fusechat3_from_SFT_trainset(dataset_path, save_dir)

    input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/openmathinstruct2_validation.json'
    save_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/openmathinstruct2_validation_with_step_score_tmp.json'
    # create_pseudo_step_weighted_data(input_file, save_path)


    input_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_misalign_dpo/iter_3/completion_with_rm_score/all_step_rollout_rule_verified.json'
    output_path = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT/iterative_misalign_dpo/iter_3/completion_with_rm_score/iterative_misaligned_dpo_dataset/2_self_sampling.json'
    create_self_sampling_dpo_pair_from_all_step_rollout_rule_verified_data(input_path, output_path)