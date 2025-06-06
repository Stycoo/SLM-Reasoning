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

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple

from .processors.feedback import preprocess_feedback_dataset
from .processors.pairwise import (preprocess_pairwise_dataset, 
                                  print_pairwise_dataset_example, 
                                  preprocess_pairwise_completion_dataset, 
                                  print_pairwise_hybrid_completion_example, 
                                  preprocess_pairwise_hybrid_completion_dataset)
from .processors.pretrain import preprocess_pretrain_dataset
from .processors.supervised import (
    preprocess_packed_supervised_dataset,
    preprocess_supervised_dataset,
    preprocess_step_weighted_supervised_dataset,
    print_supervised_dataset_example,
    preprocess_multi_supervised_dataset,
    preprocess_sft_segment_supervised_dataset,
    print_supervised_dataset_example_for_multi_sft,
    )
from .processors.unsupervised import preprocess_unsupervised_dataset, print_unsupervised_dataset_example


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments
    from .template import Template


def get_preprocess_and_print_func(
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "multi_sft", "step_weighted_sft", "rm", "ppo", "kto", "completion_dpo", "dpo", "completion_hybrid_dpo", 'sft_segment'],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            preprocess_func = partial(
                preprocess_packed_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    elif stage == "multi_sft":
        preprocess_func = partial(
                preprocess_multi_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        print_function = partial(print_supervised_dataset_example_for_multi_sft, tokenizer=tokenizer)

    elif stage == "sft_segment":
        preprocess_func = partial(
                preprocess_sft_segment_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    elif stage == "step_weighted_sft":
        preprocess_func = partial(
                preprocess_step_weighted_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    elif stage == "completion_dpo":
        preprocess_func = partial(
            preprocess_pairwise_completion_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)

    elif stage == "completion_hybrid_dpo":
        preprocess_func = partial(
            preprocess_pairwise_hybrid_completion_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_hybrid_completion_example, tokenizer=tokenizer)

    elif stage == "rm":
        preprocess_func = partial(
            preprocess_pairwise_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
    
    elif stage == "dpo":
        preprocess_func = partial(
            preprocess_pairwise_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)

    elif stage == "kto":
        preprocess_func = partial(
            preprocess_feedback_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
