# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import greedy_knapsack, infer_seqlen

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    # print(messages)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len] # ?? bug: 直接从后往前截断会导致<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n缺失
        target_ids = target_ids[:target_len] 
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    valid_num = 0
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")


def print_supervised_dataset_example_for_multi_sft(example: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer") -> None:
    for i, input_ids in enumerate(example["input_ids"]):
        print("input_ids:\n{}".format(input_ids))
        print("inputs:\n{}".format(tokenizer.decode(input_ids, skip_special_tokens=False)))

        labels = example["labels"][i]
        print("label_ids:\n{}".format(labels))
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, labels))
        print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")


def _encode_supervised_example_for_multi_sft(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    tokenizer: "PreTrainedTokenizer",
    cutoff_len: int,
) -> Tuple[List[int], List[int]]:

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    total_length = len(prompt_ids) + len(response_ids)

    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    if total_length > cutoff_len:
        prompt_len, response_len = infer_seqlen(prompt_len, response_len, cutoff_len)
        prompt_ids = prompt_ids[:prompt_len]
        response_ids = response_ids[:response_len]
    
    prompt_label = [IGNORE_INDEX] * prompt_len
    response_label = response_ids

    prompt_ids += response_ids
    response_labels = prompt_label + response_label

    return prompt_ids, response_labels


def preprocess_multi_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    model_inputs = defaultdict(list)

    prompts = [chat_template.format(prompt) for prompt in examples['prompt']] # str
    best_responses = [best_response + tokenizer.eos_token for best_response in examples['best_response']] # str

    completion_inputs = examples["completion_inputs"] # list
    completion_outputs = examples["completion_outputs"] # list

    for i, prompt in enumerate(prompts):
        _input_ids = []
        _labels = []
        _attention_mask = []

        prompt_ids, gt_response_labels = _encode_supervised_example_for_multi_sft(prompt, best_responses[i], tokenizer, data_args.cutoff_len)
        _input_ids.append(prompt_ids)
        _labels.append(gt_response_labels)
        _attention_mask.append([1] * len(prompt_ids))

        for j, completion_input in enumerate(completion_inputs[i]):
            completion_output = completion_outputs[i][j] + tokenizer.eos_token
            _completion_input_ids, _completion_output_labels = _encode_supervised_example_for_multi_sft(completion_input, completion_output, tokenizer, data_args.cutoff_len)
            
            _input_ids.append(_completion_input_ids)
            _labels.append(_completion_output_labels)
            _attention_mask.append([1] * len(_completion_input_ids))

        model_inputs["input_ids"].append(_input_ids)
        model_inputs["labels"].append(_labels)
        model_inputs["attention_mask"].append(_attention_mask)

    return model_inputs


def preprocess_step_weighted_supervised_dataset_v0(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    model_inputs = defaultdict(list)
    # Iterate over each example in the dataset
    for i, conversations in enumerate(examples['conversations']):
        # Process each response in the example
        prompt = conversations[0]['value']
        assert conversations[0]['from'] == 'human'

        # response = examples['response'][i]
        response = conversations[1]['value']
        step_scores = examples['step_scores'][i]
        step_weights = torch.softmax(torch.tensor(step_scores, dtype=torch.float32), dim=-1).tolist()
        
        # Tokenize the prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        prompt_input_ids = tokenizer(prompt, truncation=True, padding=False).input_ids
        prompt_len = len(prompt_input_ids)

        # Split the response into steps based on the "\n\n" separator
        steps = response.split("\n\n")
        step_num = len(steps)
        assert step_num == len(step_weights), f"Steps: {steps}, Step Weights: {step_weights}"

        # Add \n\n to the end position of each step except the last step
        for j in range(len(steps) - 1):
            steps[j] += "\n\n"

        # Initialize step_weight_mask list
        response_ids = []
        step_weights_expand = []
        step_index = []
        start_id = prompt_len - 1
        # Now apply step_weights for each step in the response:
        # todo: max len clip
        for step_idx, (step, step_weight) in enumerate(zip(steps, step_weights)):
            # Tokenize the step independently to get its token length
            step_input_ids = tokenizer(step, truncation=True, padding=False).input_ids
            step_len = len(step_input_ids)

            # Apply the step_score to all tokens in this step
            step_weights_expand.extend([step_weight] * step_len)
            step_index.append([start_id, start_id + step_len])
            start_id += step_len
            # Add the token IDs and attention masks for this step
            response_ids.extend(step_input_ids)  # Assuming labels are the same as input_ids for now

        response_ids += [tokenizer.eos_token_id]
        step_weights_expand.append(step_weights[-1])  # Add EOS token weight
        step_weights_expand = [IGNORE_INDEX] * prompt_len + step_weights_expand
        # step_index = [IGNORE_INDEX] * prompt_len + step_index + [step_num - 1]

        # After processing all responses for this example, append the lists to model_inputs
        input_ids = prompt_input_ids + response_ids
        labels = [IGNORE_INDEX] * prompt_len + response_ids
        attention_mask = [1] * len(input_ids)

        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['attention_mask'].append(attention_mask)
        model_inputs['step_weights'].append(step_weights_expand)
        model_inputs['step_index'].append(step_index) 

    return model_inputs


# updated
def preprocess_step_weighted_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    model_inputs = defaultdict(list)
    # Iterate over each example in the dataset
    for i, conversations in enumerate(examples['conversations']):
        # Process each response in the example
        prompt = conversations[0]['value']
        assert conversations[0]['from'] == 'human'

        # response = examples['response'][i]
        response = conversations[1]['value']
        # step_scores = examples['step_scores'][i]
        step_scores = examples['step_scores'][i] + [1]
        step_weight_num = len(step_scores)
        step_weights = torch.softmax(torch.tensor(step_scores, dtype=torch.float32), dim=-1).tolist()
        
        # Tokenize the prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        prompt_input_ids = tokenizer(prompt, truncation=True, padding=False).input_ids
        prompt_len = len(prompt_input_ids)

        # Split the response into steps based on the "\n\n" separator
        steps = response.split("\n\n")
        steps = steps[:step_weight_num - 1] + ["\n\n".join(steps[step_weight_num - 1:])]

        # Add \n\n to the end position of each step except the last step
        for j in range(len(steps) - 1):
            steps[j] += "\n\n"
        
        # Initialize step_weight_mask list
        response_ids = []
        step_weights_expand = []
        step_index = []
        start_id = prompt_len - 1
        # Now apply step_weights for each step in the response:
        # todo: max len clip
        for step_idx, (step, step_weight) in enumerate(zip(steps, step_weights)):
            # Tokenize the step independently to get its token length
            step_input_ids = tokenizer(step, truncation=True, padding=False).input_ids
            step_len = len(step_input_ids)

            # Apply the step_score to all tokens in this step
            step_weights_expand.extend([step_weight] * step_len)
            step_index.append([start_id, start_id + step_len])
            start_id += step_len
            # Add the token IDs and attention masks for this step
            response_ids.extend(step_input_ids)  # Assuming labels are the same as input_ids for now

        response_ids += [tokenizer.eos_token_id]
        step_weights_expand.append(step_weights[-1])  # Add EOS token weight
        step_weights_expand = [IGNORE_INDEX] * prompt_len + step_weights_expand
        # step_index = [IGNORE_INDEX] * prompt_len + step_index + [step_num - 1]

        # After processing all responses for this example, append the lists to model_inputs
        input_ids = prompt_input_ids + response_ids
        labels = [IGNORE_INDEX] * prompt_len + response_ids
        attention_mask = [1] * len(input_ids)

        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['attention_mask'].append(attention_mask)
        model_inputs['step_weights'].append(step_weights_expand)
        model_inputs['step_index'].append(step_index) 

    return model_inputs


def preprocess_sft_segment_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:

    model_inputs = defaultdict(list)
    
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        prompt = examples["_prompt"][i][0]['content']
        response = examples["_response"][i][0]['content']

        prompt_ids = tokenizer(prompt, truncation=True, padding=False).input_ids
        response_ids = tokenizer(response, truncation=True, padding=False).input_ids

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        attention_mask = [1] * len(input_ids)

        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['attention_mask'].append(attention_mask)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs