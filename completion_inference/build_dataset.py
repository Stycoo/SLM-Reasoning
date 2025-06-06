from pathlib import Path
import json
import random
from collections import defaultdict, Counter
import datasets
import os
import heapq


def read_json_file(input_file):
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_list = [dt for dt in data]
    return data_list


def write_json_file(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)
    

def build_prompt_subset(input_file, output_dir, sampling_type, subset_ratio):
    input_data = read_json_file(input_file)
    input_data_size = len(input_data)
    input_subdata_size = int(input_data_size * subset_ratio)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if sampling_type == 'sequential':
        input_subset = input_data[:input_subdata_size]
        output_file = output_dir / f'subset_sequential_{input_subdata_size}.json'
        
    elif sampling_type == 'random':
        input_subset = random.sample(input_data, input_subdata_size)
        output_file = output_dir / f'subset_random_{input_subdata_size}.json'

    else:
        raise ValueError('Unsupport sampling type')

    write_json_file(input_subset, output_file)


def build_completion_prompt_set(prompt_file, src_model_output_dir, src_model_name, output_dir, top_k):
    prompts = read_json_file(prompt_file)
    # prompts = prompts[7446:]
    prompt_dict = defaultdict(dict)
    for prompt in prompts:
        prompt_text = prompt['prompt']
        prompt_dict[prompt_text] = prompt

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

    best_model_statis = []
    best_response_per_model = []
    for prompt in prompt_dict:
        if prompt in src_model_output_dict:
            all_model_responses = []
            all_model_scores = []
            model_name_expand = []

            for model_name, model_output in src_model_output_dict[prompt].items():
                all_generated_responses = model_output['all_generated_responses']
                all_rm_scores = model_output['all_rm_scores']
                
                all_model_responses.extend(all_generated_responses)
                all_model_scores.extend(all_rm_scores)
                model_name_expand.extend([model_name] * len(all_generated_responses))
            
            assert len(model_name_expand) == len(all_model_scores)
            _best_response_per_model = defaultdict(dict)
            top_k_score_index = [i for _, i in heapq.nlargest(top_k, zip(all_model_scores, range(len(all_model_scores))))]
            best_response = all_model_responses[top_k_score_index[0]]
            best_response_score = all_model_scores[top_k_score_index[0]]
            assert best_response_score == max(all_model_scores)
            best_model_statis.append(model_name_expand[top_k_score_index[0]])

            for index in top_k_score_index:
                model_name = model_name_expand[index]
                if model_name not in _best_response_per_model:
                    _best_response_per_model[model_name] = {'response': [all_model_responses[index]], 'rm_score': [all_model_scores[index]]}
                else:
                    _best_response_per_model[model_name]['response'].append(all_model_responses[index])
                    _best_response_per_model[model_name]['rm_score'].append(all_model_scores[index])

            _best_response_per_model['prompt'] = prompt
            _best_response_per_model['best_response'] = best_response
            _best_response_per_model['best_response_score'] = best_response_score
            best_response_per_model.append(_best_response_per_model)

    output_dir = Path(output_dir) / f'source_responses_top_{top_k}'
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'source_responses_top_{top_k}.json'
    write_json_file(best_response_per_model, output_file)

    print(f"best_model_statis: {Counter(best_model_statis)}")


def build_sft_dataset(prompt_file, src_model_output_dir, src_model_name, sft_type, output_dir):
    prompts = read_json_file(prompt_file)
    prompt_dict = defaultdict(dict)
    for prompt in prompts:
        prompt_text = prompt['prompt']
        prompt_dict[prompt_text] = prompt

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
    if sft_type == 'single_response':
        best_response = []
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

                messages = [{'content':prompt, 'role':'user'}, {'content': _best_response['response'], 'role': 'assistant'}]
                best_response.append({'prompt': prompt, 'messages': messages, 'source_model': max_model_name})

                if max_model_name in model_sample_counts:
                    model_sample_counts[max_model_name] += 1
                else:
                    model_sample_counts[max_model_name] = 1
    
        print(model_sample_counts)

        output_dir = Path(output_dir) / f'sft_{sft_type}'
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # train_dataset = datasets.Dataset.from_list(best_response)
        # train_dataset.save_to_disk(output_dir / 'train')

        # test_dataset = best_response[-1000:]
        # test_dataset = datasets.Dataset.from_list(test_dataset)
        # test_dataset.save_to_disk(output_dir / 'test')    

        output_file = output_dir / f'sft_{sft_type}.json'
        write_json_file(best_response, output_file)

    elif sft_type == 'multiple_response':
        best_response_per_model = []
        for prompt in prompt_dict:
            best_response = ''
            best_rm_score = -10
            if prompt in src_model_output_dict:
                _best_response_per_model = defaultdict(dict)
                for model_name, model_output in src_model_output_dict[prompt].items():
                    all_generated_responses = model_output['all_generated_responses']
                    all_rm_scores = model_output['all_rm_scores']

                    max_rm_score = max(all_rm_scores)
                    max_rm_score_index = all_rm_scores.index(max_rm_score)
                    max_response = all_generated_responses[max_rm_score_index]

                    if max_rm_score > best_rm_score:
                        best_rm_score = max_rm_score
                        best_response = max_response

                    _best_response_per_model[model_name] = {'response': max_response, 'rm_score': max_rm_score}
                _best_response_per_model['prompt'] = prompt
                _best_response_per_model['best_response'] = best_response
                best_response_per_model.append(_best_response_per_model)

        output_dir = Path(output_dir) / f'sft_{sft_type}'
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'sft_{sft_type}.json'
        write_json_file(best_response_per_model, output_file)


def merge_completion_chunks(input_dir, output_file_name):
    merged_data = []

    # Iterate through all the files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
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


# {'prompt':, 'best_response':, 'source_model_outputs':[], 'source_rm_scores':[], 'completion_inputs':[], 'completion_outputs':[], 'completion_rm_scores':[]}
def build_multiple_sft_dataset(input_file, output_dir):
    input_data = read_json_file(input_file)

    completion_is_better_than_init = []
    for data in input_data:
        source_rm_scores = data["source_rm_scores"]
        completion_rm_scores = data["completion_rm_scores"]

        for i, score in enumerate(source_rm_scores):
            if score > completion_rm_scores[i]:
                completion_is_better_than_init.append(0)
            else:
                completion_is_better_than_init.append(1)

        source_model_outputs = data['source_model_outputs']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']

        sorted_tuples = sorted(
            zip(source_rm_scores, source_model_outputs, completion_inputs, completion_outputs, completion_rm_scores),
            key=lambda x: x[-1],
            reverse=True
        )

        data["source_rm_scores"], data["source_model_outputs"], data['completion_inputs'], data['completion_outputs'], data['completion_rm_scores'] = map(list, zip(*sorted_tuples))
    
    print(f"completion_is_better_than_init ratio: {sum(completion_is_better_than_init) / len(completion_is_better_than_init)}")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.Dataset.from_list(input_data)
    train_dataset.save_to_disk(output_dir / 'train')

    test_dataset = input_data[-1000:]
    test_dataset = datasets.Dataset.from_list(test_dataset)
    test_dataset.save_to_disk(output_dir / 'test')    


### LLaMA-factory sft
def build_sft_dataset_sharegpt_format(prompt_file, src_model_output_dir, src_model_name, output_path):
    # 从source model responses 中选出得分最高的
    prompts = read_json_file(prompt_file)
    prompt_dict = defaultdict(dict)
    for prompt in prompts:
        prompt_text = prompt['prompt']
        prompt_dict[prompt_text] = prompt

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

    print(model_sample_counts)
    write_json_file(best_responses, output_path)


def build_sft_dataset_sharegpt_format_v2(input_file, output_path):
    # 从source model response & target model completion 中选出得分最高的
    input_data = read_json_file(input_file)

    completion_is_better_than_init = []
    best_responses = []
    for data in input_data:
        prompt = data['prompt']
        source_rm_scores = data["source_rm_scores"]
        completion_rm_scores = data["completion_rm_scores"]
        source_model_outputs = data['source_model_outputs']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']

        max_source_rm_score = max(source_rm_scores)
        max_completion_rm_score = max(completion_rm_scores)

        if max_source_rm_score > max_completion_rm_score:
            _best_response = source_model_outputs[source_rm_scores.index(max_source_rm_score)]
            completion_is_better_than_init.append(0)
        else:
            max_completion_rm_score_index = completion_rm_scores.index(max_completion_rm_score)
            _best_completion = completion_outputs[max_completion_rm_score_index]
            _best_response = completion_inputs[max_completion_rm_score_index].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1] + ' ' + _best_completion
            completion_is_better_than_init.append(1)

        conversations = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': _best_response}]
        best_responses.append({'conversations': conversations})
    
    print(f"completion_is_better_than_init ratio: {sum(completion_is_better_than_init) / len(completion_is_better_than_init)}")

    write_json_file(best_responses, output_path)   


def build_multiple_sft_dataset_for_llama_factory(input_file, output_path):
    input_data = read_json_file(input_file)

    completion_is_better_than_init = []
    for data in input_data:
        source_rm_scores = data["source_rm_scores"]
        completion_rm_scores = data["completion_rm_scores"]

        for i, score in enumerate(source_rm_scores):
            if score > completion_rm_scores[i]:
                completion_is_better_than_init.append(0)
            else:
                completion_is_better_than_init.append(1)

        source_model_outputs = data['source_model_outputs']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']

        sorted_tuples = sorted(
            zip(source_rm_scores, source_model_outputs, completion_inputs, completion_outputs, completion_rm_scores),
            key=lambda x: x[-1],
            reverse=True
        )

        data["source_rm_scores"], data["source_model_outputs"], data['completion_inputs'], data['completion_outputs'], data['completion_rm_scores'] = map(list, zip(*sorted_tuples))
    
    print(f"completion_is_better_than_init ratio: {sum(completion_is_better_than_init) / len(completion_is_better_than_init)}")

    write_json_file(input_data, output_path)   


def build_multiple_sft_dataset_for_llama_factory_v2(input_file, output_path):
    # response label: source model responses & target model completions 中得分最高的; completion label: target model completions 中除分数最高之外的其他completions
    input_data = read_json_file(input_file)

    completion_is_better_than_init = []
    response_num_statis = 0
    data_return = []

    for data in input_data:
        source_rm_scores = data["source_rm_scores"]
        completion_rm_scores = data["completion_rm_scores"]

        source_model_outputs = data['source_model_outputs']
        completion_inputs = data['completion_inputs']
        completion_outputs = data['completion_outputs']

        max_source_rm_score = max(source_rm_scores)
        max_completion_rm_score = max(completion_rm_scores)

        if len(source_model_outputs) != 4:
            response_num_statis += 1
            continue

        if max_source_rm_score > max_completion_rm_score:
            max_score_index = source_rm_scores.index(max_source_rm_score)
            _best_response = source_model_outputs[max_score_index]
            completion_is_better_than_init.append(0)

        else:
            max_score_index = completion_rm_scores.index(max_completion_rm_score)
            _best_completion = completion_outputs[max_score_index]
            _best_response = completion_inputs[max_score_index].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1] + ' ' + _best_completion
            completion_is_better_than_init.append(1)

        data["best_response"] = _best_response

        del source_rm_scores[max_score_index]
        del source_model_outputs[max_score_index]
        del completion_inputs[max_score_index]
        del completion_outputs[max_score_index]
        del completion_rm_scores[max_score_index]
    
        sorted_tuples = sorted(
            zip(source_rm_scores, source_model_outputs, completion_inputs, completion_outputs, completion_rm_scores),
            key=lambda x: x[-1],
            reverse=True
        )

        data["source_rm_scores"], data["source_model_outputs"], data['completion_inputs'], data['completion_outputs'], data['completion_rm_scores'] = map(list, zip(*sorted_tuples))
        data_return.append(data)

    print(f"completion_is_better_than_init ratio: {sum(completion_is_better_than_init) / len(completion_is_better_than_init)}")

    write_json_file(data_return, output_path)  
    

def rm_score_compare_tgt_src(src_file, tgt_file, filtered_file):
    # 过滤掉target model response更好的数据
    source_model_responses = read_json_file(src_file)
    target_model_responses = read_json_file(tgt_file)
    
    target_model_response_score = defaultdict(dict)
    for data in target_model_responses:
        prompt = data['prompt']
        target_response_score = data['rm_score']
        target_response = data['generated_response']
        target_model_response_score[prompt] = {"response": target_response, "rm_score": target_response_score}

    tgt_is_better = []
    source_model_responses_filtered = []
    for data in source_model_responses:
        prompt = data['prompt']
        src_response_score = data['best_response_score']
        
        tgt_model_response_score = target_model_response_score[prompt]['rm_score']
        if tgt_model_response_score > src_response_score:
            tgt_is_better.append(1)
        else:
            tgt_is_better.append(0)
            data['target_model_response'] = target_model_response_score[prompt]['response']
            data["target_model_response_score"] = tgt_model_response_score
            source_model_responses_filtered.append(data)

    print(f"tgt_is_better ratio: {sum(tgt_is_better) / len(tgt_is_better)}")

    write_json_file(source_model_responses_filtered, filtered_file)


def check_completion_validation(input_file, output_file): 
    # 过滤掉completion为空的数据
    input_data = read_json_file(input_file)
    null_response_num = 0
    completion_data_filtered = []
    for data in input_data:
        null_response_flag = 0
        completion_outputs = data["completion_outputs"]
        for completion_output in completion_outputs:
            if not completion_output:
                null_response_num += 1
                null_response_flag = 1
                break

        if not null_response_flag:
            completion_data_filtered.append(data)

    print(f"null_response_num: {null_response_num}")
    write_json_file(completion_data_filtered, output_file)


def add_target_model_response_to_completion_file(completion_file, target_model_file, output_file):
    target_model_responses = read_json_file(target_model_file)
    target_model_response_score = defaultdict(dict)
    for data in target_model_responses:
        prompt = data['prompt']
        all_target_response_scores = data['all_rm_scores']
        all_target_responses = data['all_generated_responses']
        target_model_response_score[prompt] = {"response": all_target_response, "rm_score": all_target_response_scores}

    completion_responses = read_json_file(completion_file)
    for comp_data in completion_responses:
        prompt = comp_data["prompt"]
        all_target_responses = target_model_response_score[prompt]["response"]
        all_target_response_scores = target_model_response_score[prompt]["rm_score"]

        comp_data['target_model_responses'] = all_target_responses
        comp_data['target_model_response_scores'] = all_target_response_scores

    write_json_file(completion_responses, output_file)
    # {'prompt': str, 'best_response':str, 'best_response_score':float, 'completion_inputs': [], 'completion_outputs':[[],...]}, 'completion_scores':[[],...]:,
    # 'target_model_responses':[], 'target_model_response_scores': []'}


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
    print(f"All outputs size: {len(all_res)}. {num_identical} samples with identical generated responses")

    with open(os.path.join(generation_file_dir, 'all_generated_response.json'), 'w') as f:
        json.dump(all_res, f, indent=4)

    print(f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_generated_response.json')}")


def build_dpo_dataset(input_file, save_dir):
    # 构建用于dpo训练的数据，格式满足llama-factory要求
    # 1. target model response 最优时chosen，rejected从target model response中选取
    # 2. source model response 最优时，遍历不同位置的completions，在第一次completion score>target model response 位置处的多次采样中选出chosen，rejected
    
    input_data = read_json_file(input_file)
    chosen_rejected_dataset = []
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
            completion_scores = data['completion_scores']

            for i, completion_input in enumerate(completion_inputs): 
                if len(set(completion_outputs[i])) == 1:
                    continue
            
                completion_scores_i = completion_scores[i]
                max_completion_score_i = max(completion_scores_i)
                min_completion_score_i = min(completion_scores_i)

                if max_completion_score_i > max_target_model_response_score:
                    chosen_completion_index = completion_scores_i.index(max_completion_score_i)
                    rejected_completion_index = completion_scores_i.index(min_completion_score_i)

                    completion_input = completion_input.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
                    chosen = completion_input + ' ' + completion_outputs[i][chosen_completion_index]
                    rejected = completion_input + ' ' + completion_outputs[i][rejected_completion_index]
                    break
                    
        if not chosen:
            if len(set(data['target_model_responses'])) == 1:
                continue
            else:
                max_score_index = target_model_response_scores.index(max_target_model_response_score)
                min_score_index = target_model_response_scores.index(min_target_model_response_score)
                
                chosen = target_model_responses[max_score_index]
                rejected = target_model_responses[min_score_index]

        chosen_rejected_data = {'instruction': data['prompt'], 'chosen':chosen, 'rejected': rejected}
        chosen_rejected_dataset.append(chosen_rejected_data)
        
    return chosen_rejected_dataset


def data_preprocess(prompt_file, source_model_file, save_dir):
    # 根据prompt从多source model预测文件中抽取数据
    prompt_data = read_json_file(prompt_file)
    prompt_list = []
    for data in prompt_data:
        assert data['conversations'][0]['from'] == 'human'
        prompt = data['conversations'][0]['value']
        prompt_list.append(prompt)

    source_model_data_dict = defaultdict(dict)
    source_model_data = read_json_file(source_model_file)
    for data in source_model_data:
        prompt = data['prompt']
        source_model_data_dict[prompt] = data
    
    source_model_data_used = []
    for prompt in prompt_list:
        source_model_data_used.append(source_model_data_dict[prompt])

    output_dir = Path(save_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(len(source_model_data_used))
    output_path = Path(save_dir) / 'source_model_data_for_chosen_sft.json'
    write_json_file(source_model_data_used, output_path)


if __name__ == "__main__":
    src_model_outputs_path = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/deepseek-chat/all_outputs_rm.json'
    prompt_path = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/ultra_train_prompt.json'
    
    sampling_type = 'sequential'
    subset_ratio = 0.4
    subset_save_dir = '/data/sty/model_fusion/zlg_1118'
    # f'/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_{sampling_type}_ratio_{subset_ratio}'
    
    ### 1.从全量数据集中顺序截取40%数据作为训练数据
    # build_prompt_subset(prompt_path, subset_save_dir, sampling_type, subset_ratio)


    ### 2.构建single response sft 数据
    # subset_prompt_path = Path(subset_save_dir) / 'subset_sequential_23950.json'
    # src_model_output_dir = Path(subset_save_dir)
    # '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118'
    # src_model_name = ['deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 'Qwen2.5-72B-Instruct'] #'deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 
    # sft_type = 'single_response' # multiple_response single_response
    # output_dir = subset_save_dir
    # build_sft_dataset(subset_prompt_path, src_model_output_dir, src_model_name, sft_type, output_dir)


    ### 3. 构建用于completion的数据，并未包含completion input构造函数，直接在completion_infer步构建
    subset_prompt_path = Path(subset_save_dir) / 'ultra_train_prompt.json'
    src_model_output_dir = Path(subset_save_dir)
    src_model_name = ['deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 'Qwen2.5-72B-Instruct'] #'deepseek-chat', 'gemma-2-27b-it', 'Mistral-Large', 
    output_dir = Path(subset_save_dir) #/ 'sft_multiple_response_top_5'/ 'sft_to_dpo_0103'
    top_k = 4
    build_completion_prompt_set(subset_prompt_path, src_model_output_dir, src_model_name, output_dir, top_k)

    # 将不同seed下target model生成的response合并到单个文件保存
    generation_file_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct_multi_responses_v1230'
    # merge_target_multi_responses(generation_file_dir)

    # 将打分后的source model response chunk files 合成单个文件
    input_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct_multi_responses/multi_responses_with_rm_score'
    output_file_name = 'multi_responses_with_rm_score'
    # merge_completion_chunks(input_dir, output_file_name)

    src_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_multiple_response_top_5.json'
    tgt_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/target_model_response/Llama-3.2-3B-Instruct_single_response_with_rm_score/Llama-3.2-3B-Instruct_single_response_with_rm_score.json'
    filtered_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_multiple_response_top_5_filtered.json'
    # rm_score_compare_tgt_src(src_file, tgt_file, filtered_file)


    ### 4. 将completion-infer结果保存的chunks合并为单个文件 
    input_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_completion_1219'
    # merge_completion_chunks(input_dir, 'best_response_with_completion')
    
    input_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response/target_model_response/Llama-3.2-3B-Instruct_single_response_with_rm_score'
    output_file_name = 'Llama-3.2-3B-Instruct_single_response_with_rm_score'
    # merge_completion_chunks(input_dir, output_file_name)

    input_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_completion_1219/best_response_with_completion.json'
    output_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_completion_1219/best_response_with_completion_filtered.json'
    # check_completion_validation(input_file, output_file)

    input_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_completion_1219/best_response_with_completion_scored'
    output_file_name = 'best_response_with_completion_scored'
    # merge_completion_chunks(input_dir, output_file_name)

    completion_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_completion_1219/best_response_with_completion_scored/best_response_with_completion_scored.json'
    target_model_file = tgt_file
    output_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/best_response_with_completion_1219/best_response_with_completion_scored/best_response_with_completion_scored_formal.json'
    # add_target_model_response_to_completion_file(completion_file, target_model_file, output_file)

    ### 5. 构建multi-response sft格式数据
    input_file = f"{input_dir}/sft_multiple_response_with_completion_scored.json"
    # build_multiple_sft_dataset(input_file, input_dir)
    
    # build_sft_dataset_sharegpt_format(subset_prompt_path, src_model_output_dir, src_model_name, output_path)

    output_path = '/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/sft_single_response/sft_single_response_w_completion.json'
    # build_sft_dataset_sharegpt_format_v2(input_file, output_path)

    output_path = "/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_wo_overlap/sft_multiple_response_wo_overlap.json"
    # build_multiple_sft_dataset_for_llama_factory(input_file, output_path)

    # build_multiple_sft_dataset_for_llama_factory_v2(input_file, output_path)

    prompt_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_to_dpo_0103/sft_7446.json'
    source_model_file = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_multiple_response_top_5.json'
    save_dir = '/nas-wulanchabu/shitianyuan.sty/huggingface_dataset/zlg_1118/subset_sequential_ratio_0.4/sft_multiple_response_top_5/sft_to_dpo_0220_v2/chosen_sft'
    # data_preprocess(prompt_file, source_model_file, save_dir)