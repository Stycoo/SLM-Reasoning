# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    # @override
    # def concatenated_forward(
    #     self,
    #     model: "PreTrainedModel",
    #     batch: Dict[str, "torch.Tensor"],
    # ) -> Tuple[
    #     "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor",  # on-policy: logps, logps_avg, logits
    #     "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"   # off-policy: logps, logps_avg, logits
    # ]:
    #     # if we're scoring with a ref model, avoid accidental grad-flow
    #     if self.finetuning_args.use_ref_model:
    #         batch = {k: v.detach().clone() for k, v in batch.items()}

    #     # get logits + log-probs
    #     outputs = model(**batch, return_dict=True, use_cache=False)
    #     logits = outputs.logits.float()  # cast once
    #     logps, lengths = get_batch_logps(logits=logits, labels=batch["labels"])

    #     # support average-vs-sum variants
    #     if self.loss_type in {"ipo", "orpo", "simpo"}:
    #         logps = logps / lengths

    #     # split into 4 equal chunks: on-chosen, on-rejected, off-chosen, off-rejected
    #     on_chosen_lp, on_rej_lp, off_chosen_lp, off_rej_lp = logps.chunk(4, dim=0)
    #     on_chosen_logits, on_rej_logits, off_chosen_logits, off_rej_logits = logits.chunk(4, dim=0)
    #     on_chosen_len, on_rej_len, off_chosen_len, off_rej_len = lengths.chunk(4, dim=0)

    #     # compute averages
    #     on_chosen_lp_avg = on_chosen_lp / on_chosen_len
    #     off_chosen_lp_avg = off_chosen_lp / off_chosen_len

    #     return (
    #         on_chosen_lp,
    #         on_rej_lp,
    #         on_chosen_logits,
    #         on_rej_logits,
    #         on_chosen_lp_avg,
    #         off_chosen_lp,
    #         off_rej_lp,
    #         off_chosen_logits,
    #         off_rej_logits,
    #         off_chosen_lp_avg,
    #     )
    
    @override
    def concatenated_forward(
            self,
            model: "PreTrainedModel",
            batch: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, ...]:
        """
        Two-pass, memory-efficient forward.

        The **input batch is always ordered** as
            [on-chosen, on-rejected, off-chosen, off-rejected].

        We process it in **two sub-batches**:

        1. sub-batch-A → on-chosen  + on-rejected
        2. sub-batch-B → off-chosen + off-rejected

        This halves peak activation memory (each forward only sees 50 % of the
        original batch), while downstream code still receives tensors in the full
        4-way order it expects.

        Returns (when ``return_logits=True``)
            (on_c_lp, on_r_lp, on_c_logits, on_r_logits, on_c_lp_avg,
            off_c_lp, off_r_lp, off_c_logits, off_r_logits, off_c_lp_avg)

        Otherwise (``return_logits=False``)
            (on_c_lp, on_r_lp, on_c_lp_avg,
            off_c_lp, off_r_lp, off_c_lp_avg)
        """

        # ------------------------------------------------------------------ helpers
        def _run_forward(sub: Dict[str, torch.Tensor]) -> Tuple:
            """
            Forward pass on a *single* (sub-)batch that already fits in memory.
            """
            with torch.autocast("cuda", dtype=next(model.parameters()).dtype):
                out    = model(**sub, return_dict=True, use_cache=False)
                logits = out.logits                                 # (b, L, V)
                labels = sub["labels"]
                logps, lengths = get_batch_logps(logits=logits, labels=labels)

            if self.loss_type in {"ipo", "orpo", "simpo"}:
                logps = logps / lengths

            return logits, logps, lengths

        # ----------------------------------------------------------------- slicing
        B = next(iter(batch.values())).size(0)
        if B % 4:
            raise ValueError("Batch size must be divisible by 4 "
                            "(on_c, on_r, off_c, off_r).")

        grp = B // 4                        # size of each logical group
        idxA = slice(0, 2 * grp)            # on-chosen + on-rejected
        idxB = slice(2 * grp, B)            # off-chosen + off-rejected

        batchA = {k: v[idxA] for k, v in batch.items()}
        batchB = {k: v[idxB] for k, v in batch.items()}

        # optional: detach when evaluating with a reference model
        if self.finetuning_args.use_ref_model:
            batchA = {k: v.detach() for k, v in batchA.items()}
            batchB = {k: v.detach() for k, v in batchB.items()}

        # ---------------------------------------------------------- forward passes
        logitsA, logpsA, lensA = _run_forward(batchA)   # on-chosen/rejected
        torch.cuda.empty_cache()                        # free before pass-B
        logitsB, logpsB, lensB = _run_forward(batchB)   # off-chosen/rejected

        # ------------------------------------------------ split back into 4 groups
        on_c_lp, on_r_lp = logpsA.chunk(2, 0)
        off_c_lp, off_r_lp = logpsB.chunk(2, 0)

        on_c_len, on_r_len = lensA.chunk(2, 0)
        off_c_len, off_r_len = lensB.chunk(2, 0)

        on_c_lp_avg  = on_c_lp  / on_c_len
        off_c_lp_avg = off_c_lp / off_c_len

        on_c_logits, on_r_logits = logitsA.chunk(2, 0)
        off_c_logits, off_r_logits = logitsB.chunk(2, 0)

        return (
            on_c_lp, on_r_lp, on_c_logits, on_r_logits, on_c_lp_avg,
            off_c_lp, off_r_lp, off_c_logits, off_r_logits, off_c_lp_avg,
        )

    @override
    def compute_reference_log_probs(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
    ) -> Tuple[
        Optional["torch.Tensor"], Optional["torch.Tensor"],
        Optional["torch.Tensor"], Optional["torch.Tensor"]
    ]:
        # skip entirely if we're not using a separate ref model
        if not self.finetuning_args.use_ref_model:
            return None, None, None, None

        # pick the right model + context (disable adapters if needed)
        if self.ref_model is not None:
            ref_model = self.ref_model
            ref_context = nullcontext()
        else:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()

        with torch.no_grad(), ref_context:
            out = self.concatenated_forward(ref_model, batch)

            ref_on_chosen_lp      = out[0]
            ref_on_rej_lp         = out[1]
            ref_off_chosen_lp     = out[5]
            ref_off_rej_lp   = out[6]

        return (
            ref_on_chosen_lp,
            ref_on_rej_lp,
            ref_off_chosen_lp,
            ref_off_rej_lp,
        )

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        # forward through main model
        (
            on_chosen_lp,
            on_rej_lp,
            on_chosen_logits,
            on_rej_logits,
            _on_chosen_lp_avg,
            off_chosen_lp,
            off_rej_lp,
            off_chosen_logits,
            off_rej_logits,
            _off_chosen_lp_avg,
        ) = self.concatenated_forward(model, batch)

        # get reference log-probs
        (
            ref_on_chosen_lp,
            ref_on_rej_lp,
            ref_off_chosen_lp,
            ref_off_rej_lp,
        ) = self.compute_reference_log_probs(model, batch)

        # compute per-pair losses + rewards
        on_losses, on_chosen_r, on_rej_r = self.compute_preference_loss(
            on_chosen_lp, on_rej_lp, ref_on_chosen_lp, ref_on_rej_lp
        )
        off_losses, off_chosen_r, off_rej_r = self.compute_preference_loss(
            off_chosen_lp, off_rej_lp, ref_off_chosen_lp, ref_off_rej_lp
        )

        # final scalar loss
        loss = 0.5 * (on_losses + off_losses)
        
        # helper to aggregate into Python floats
        prefix = "eval_" if train_eval == "eval" else ""
        metrics: Dict[str, float] = {}
        def _log(name: str, tensor: "torch.Tensor"):
            metrics[f"{prefix}{name}"] = tensor.mean().item()

        # on-policy
        _log("rewards/on_chosen",      on_chosen_r)
        _log("rewards/on_rejected",    on_rej_r)
        _log("rewards/on_accuracies",  (on_chosen_r > on_rej_r).float())
        _log("rewards/on_margins",     on_chosen_r - on_rej_r)
        _log("logps/on_chosen",        on_chosen_lp)
        _log("logps/on_rejected",      on_rej_lp)
        _log("logits/on_chosen",       on_chosen_logits)
        _log("logits/on_rejected",     on_rej_logits)

        # off-policy
        _log("rewards/off_chosen",     off_chosen_r)
        _log("rewards/off_rejected",   off_rej_r)
        _log("rewards/off_accuracies", (off_chosen_r > off_rej_r).float())
        _log("rewards/off_margins",    off_chosen_r - off_rej_r)
        _log("logps/off_chosen",       off_chosen_lp)
        _log("logps/off_rejected",     off_rej_lp)
        _log("logits/off_chosen",      off_chosen_logits)
        _log("logits/off_rejected",    off_rej_logits)

        return loss, metrics

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)
