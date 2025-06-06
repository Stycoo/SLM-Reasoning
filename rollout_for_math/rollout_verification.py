import json
import os
import argparse
import tqdm
import numpy as np
import datasets
from pathlib import Path
from math_verify import parse, verify
import re
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default='', help="Path to the output generation file")
parser.add_argument("--model_name_or_path", type=str, default='', help="Path to reward model")
parser.add_argument("--output_dir", type=str, default='', help="Path to output directory")
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)
parser.add_argument("--verfiy_mode", type=str, default='', help="PRM or Rule or LLM-as-a-judge")
parser.add_argument("--check_file", type=str, default='')

args = parser.parse_args()

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def has_repeated_chars(s, threshold):
    pattern = rf"(.)\1{{{threshold},}}"  # `\1`表示匹配的第一个分组，`{threshold,}`表示重复 threshold 次及以上
    return bool(re.search(pattern, s))

def _process_one(completion_output, answer):
    # exactly your “worker” logic, returning 0/1 or -1 on error
    try:
        last_step = completion_output.split('\n\n')[-1]
        if has_repeated_chars(last_step, 20):
            return 0

        extracted = parse(last_step)
        gold_last = answer.split('\n\n')[-1]
        gold = parse(f"${gold_last}$")
        return 1 if verify(gold, extracted) else 0

    except Exception:
        return 0  # or -1 if you prefer

with open(args.input_file, 'r') as f:
    input_data = json.load(f)
    # input_data = input_data[:1000]
    input_data_size = len(input_data)

if args.verfiy_mode == 'PRM':
    chunk_size = input_data_size // args.chunk_num
    if args.chunk_id < args.chunk_num - 1:
        input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
    else:
        input_data = input_data[args.chunk_id * chunk_size: ]

    device = "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    for data in tqdm.tqdm(input_data):
        prompt = data["prompt"]
        completion_inputs = data["completion_inputs"]
        completion_outputs = data["completion_outputs"]

        step_scores = []
        for i, completion_input in enumerate(completion_inputs):
            _step_scores = []
            for j, completion_output in enumerate(completion_outputs[i]):
                response = completion_input.split("<|im_start|>assistant\n")[1] + completion_output
            
                data = {
                    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
                    "query": prompt,
                    "response": response.split('\n\n')
                    }
                messages = [
                    {"role": "system", "content": data['system']},
                    {"role": "user", "content": data['query']},
                    {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
                ]
                conversation_str = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = tokenizer.encode(
                    conversation_str, 
                    return_tensors="pt", 
                    ).to(model.device)
                outputs = model(input_ids=input_ids)

                step_sep_id = tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_reward = make_step_rewards(outputs[0], token_masks)
                print(step_reward)  # eg: [[1.0, 0.1904296875, 0.9765625, 1.0]]
                _step_scores.append(step_reward[0])

            step_scores.append(_step_scores)

        data["prm_step_scores"] = step_scores

    file_name = f"chunk_{args.chunk_id}.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(input_data, f, indent=4)


elif args.verfiy_mode == 'ORM':

    # Split into chunks
    chunk_size = input_data_size // args.chunk_num
    if args.chunk_id < args.chunk_num - 1:
        input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
    else:
        input_data = input_data[args.chunk_id * chunk_size:]

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model.eval()

    # Batched scoring
    for data in tqdm.tqdm(input_data):
        prompt = data["prompt"]
        completion_outputs = data["completion_outputs"]
        completion_inputs = data["completion_inputs"]

        batched_messages = []
        mapping = []  # (i, j) index for scores

        for i, completion_input in enumerate(completion_inputs):
            prefix = completion_input.split("assistant<|end_header_id|>\n\n")[1]
            for j, completion_output in enumerate(completion_outputs[i]):
                response = prefix + completion_output
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}]
                batched_messages.append(messages)
                mapping.append((i, j))

        # Tokenize in batch
        encoded = tokenizer.apply_chat_template(
            batched_messages,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(encoded)
            scores = outputs.logits.float().squeeze(-1).tolist()

        # Reconstruct nested score structure
        completion_outputs_scores = [[] for _ in completion_inputs]
        for (i, j), score in zip(mapping, scores):
            completion_outputs_scores[i].append(score)

        data["orm_scores"] = completion_outputs_scores

    # Save result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"chunk_{args.chunk_id}.json"
    with open(output_dir / file_name, 'w') as f:
        json.dump(input_data, f, indent=4)


elif args.verfiy_mode == 'Rule':
    CHECKPOINT=1000

    def verify_all_modify(input_data, output_file,
               timeout: int = 2,
               max_workers: int | None = None,
               checkpoint: int = CHECKPOINT):
        """并发验证并将结果流式写成 JSON 数组；结束后保证无残留子进程。"""
        max_workers = max_workers or os.cpu_count()
        total = len(input_data)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # ---------- ❶ 启动进程池 ----------
        executor = ProcessPoolExecutor(max_workers=max_workers)
        try:
            with open(output_file, "w", encoding="utf-8") as fout:
                fout.write("[\n")
                first = True
                batch = []

                for data_idx, data in enumerate(tqdm.tqdm(input_data, desc="data"), 1):
                    outputs = data["completion_outputs"]
                    answer = data["answer"]

                    # --- 提交任务 ---
                    flat = [out for roll in outputs for out in roll]
                    futures = [executor.submit(_process_one, out, answer) for out in flat]

                    # --- 收集结果（带超时） ---
                    results = []
                    for fut in futures:
                        try:
                            results.append(fut.result(timeout=timeout))
                        except TimeoutError:
                            fut.cancel()          # 立即取消；若已运行会返回 False
                            results.append(0)

                    # --- 重新分组 ---
                    idx_ptr = 0
                    rule_scores = []
                    for roll in outputs:
                        n = len(roll)
                        rule_scores.append(results[idx_ptr: idx_ptr + n])
                        idx_ptr += n

                    data["rule_scores"] = rule_scores
                    batch.append(data)

                    # --- 到 checkpoint / 最后一条就写盘 ---
                    if (data_idx % checkpoint == 0) or (data_idx == total):
                        for item in batch:
                            if not first:
                                fout.write(",\n")
                            fout.write(json.dumps(item, ensure_ascii=False))
                            first = False
                        batch.clear()

                        fout.flush()
                        os.fsync(fout.fileno())
                        print(f"💾 已追加 {data_idx}/{total} 条", flush=True)

                fout.write("\n]\n")

        finally:
            # ---------- ❷ 关闭进程池 ----------
            executor.shutdown(wait=True, cancel_futures=True)

            # ---------- ❸ 双保险：杀掉仍存活的子进程 ----------
            for p in multiprocessing.active_children():
                try:
                    p.terminate()
                    p.join(1)
                    if p.is_alive():
                        p.kill()  # 发送 SIGKILL
                except Exception:
                    pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_file = output_dir / "all_step_rollout_rule_verified.json"

    verify_all_modify(input_data, final_file, timeout=2, max_workers=10)

else:
    raise ValueError("please input valid verfiy_mode: [PRM, ORM, Rule]")