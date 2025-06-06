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
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_pairwise_example(
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
) -> Tuple[List[int], List[int], List[int], List[int]]:
    chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
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
        )
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def preprocess_pairwise_completion_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue
       
        prompt=examples["_prompt"][i][0]['content']
        chosen_response=examples["_response"][i][0]['content']
        rejected_response=examples["_response"][i][1]['content']

        prompt_ids = tokenizer(prompt, truncation=False, padding=False).input_ids # truncation=True,
        source_len = len(prompt_ids)

        chosen_ids = tokenizer(chosen_response, truncation=False, padding=False).input_ids #+ [tokenizer.eos_token_id]
        rejected_ids = tokenizer(rejected_response, truncation=False, padding=False).input_ids #+ [tokenizer.eos_token_id]

        # source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), data_args.cutoff_len)
        # prompt_ids = prompt_ids[:source_len]
        # chosen_ids = chosen_ids[:target_len]
        # rejected_ids = rejected_ids[:target_len]

        if len(chosen_ids) + len(prompt_ids) > data_args.cutoff_len:
            chosen_ids = chosen_ids[: data_args.cutoff_len - len(prompt_ids)]
        if len(rejected_ids) + len(prompt_ids) > data_args.cutoff_len:
            rejected_ids = rejected_ids[: data_args.cutoff_len - len(prompt_ids)]
        
        chosen_input_ids = prompt_ids + chosen_ids 
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids

        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def _make_example(
    prompt_ids: List[int],
    target_ids: List[int],
    cutoff_len: int,
) -> Dict[str, List[int]]:
    """
    Given tokenized prompt_ids and target_ids, truncate target_ids
    if necessary, then build input_ids, attention_mask, labels.
    """
    src_len = len(prompt_ids)
    max_tgt = cutoff_len - src_len

    if max_tgt <= 0:
        max_tgt = 100
    
    if len(target_ids) > max_tgt:
        target_ids = target_ids[:max_tgt]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [IGNORE_INDEX] * src_len + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_pairwise_hybrid_completion_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",            # unused in this snippet?
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    model_inputs = defaultdict(list)

    # for each example i
    for prompt_pair, chosen_pair, rejected_pair in zip(
        examples["conversations"],
        examples["chosen"],
        examples["rejected"]
    ):
        # conversations: [on_policy_conv, off_policy_conv]
        for policy_name, conv, chosen, rejected in (
            ("on_policy", prompt_pair[0], chosen_pair[0], rejected_pair[0]),
            ("off_policy", prompt_pair[1], chosen_pair[1], rejected_pair[1]),
        ):
            # extract raw strings
            prompt_text = conv["value"]
            chosen_text = chosen["value"]
            rejected_text = rejected["value"]

            # batch-tokenize prompt / chosen / rejected
            encodings = tokenizer(
                [prompt_text, chosen_text, rejected_text],
                truncation=False, padding=False
            ).input_ids
            prompt_ids, chosen_ids, rejected_ids = encodings

            # build chosen example
            chosen_ex = _make_example(prompt_ids, chosen_ids, data_args.cutoff_len)
            model_inputs[f"{policy_name}_chosen_input_ids"].append(chosen_ex["input_ids"])
            model_inputs[f"{policy_name}_chosen_attention_mask"].append(chosen_ex["attention_mask"])
            model_inputs[f"{policy_name}_chosen_labels"].append(chosen_ex["labels"])

            # build rejected example
            rej_ex = _make_example(prompt_ids, rejected_ids, data_args.cutoff_len)
            model_inputs[f"{policy_name}_rejected_input_ids"].append(rej_ex["input_ids"])
            model_inputs[f"{policy_name}_rejected_attention_mask"].append(rej_ex["attention_mask"])
            model_inputs[f"{policy_name}_rejected_labels"].append(rej_ex["labels"])

        model_inputs["images"].append([])
        model_inputs["videos"].append([])

    return model_inputs


def print_pairwise_hybrid_completion_example(
    example: Dict[str, List[int]],
    tokenizer: "PreTrainedTokenizer",
) -> None:
    def _print_block(policy: str, outcome: str):
        input_key = f"{policy}_{outcome}_input_ids"
        label_key = f"{policy}_{outcome}_labels"

        input_ids = example[input_key]
        # filter out IGNORE_INDEX in one go
        labels = [tok for tok in example[label_key] if tok != IGNORE_INDEX]

        print(f"{policy}_{outcome}_input_ids:\n{input_ids}")
        print(f"{policy}_{outcome}_inputs:\n"
              f"{tokenizer.decode(input_ids, skip_special_tokens=False)}")
        print(f"{policy}_{outcome}_label_ids:\n{example[label_key]}")
        print(f"{policy}_{outcome}_labels:\n"
              f"{tokenizer.decode(labels, skip_special_tokens=False)}\n")

    print("Example keys: {}".format(example.keys()))
    for policy in ("on_policy", "off_policy"):
        for outcome in ("chosen", "rejected"):
            _print_block(policy, outcome)


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print(f"chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print(f"rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")