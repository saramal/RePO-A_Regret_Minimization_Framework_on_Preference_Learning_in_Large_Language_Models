import os
from abc import ABC

import torch
from flash_attn.utils.distributed import all_gather
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from peft import PeftModelForCausalLM
from contextlib import contextmanager, nullcontext

from openrlhf.models import TDPOLoss
from openrlhf.models.utils import log_probs_from_logits
from openrlhf.utils.distributed_sampler import DistributedSampler


class TDPOTrainer(ABC):
    """
    Trainer for Token-level Direct Preference Optimization (TDPO) training.
    
    TDPO extends DPO by incorporating token-level position KL divergence between the policy 
    and reference model to provide finer-grained control over the optimization process.

    Args:
        model (torch.nn.Module): The primary model to be trained.
        ref_model (torch.nn.Module): The reference model for computing KL divergence and log probability margins.
        strategy (Strategy): The strategy to use for training.
        tokenizer (Tokenizer): The tokenizer for processing input data.
        optim (Optimizer): The optimizer for training the model.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to control learning rate during training.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        beta (float, defaults to 0.01): Temperature parameter for the TDPO loss, typically in range of 0.1 to 0.5.
        alpha (float, defaults to 0.5): Temperature parameter for adjusting the impact of sequential KL divergence.
        if_tdpo2 (bool, defaults to True): If True, use TDPO2 variant with detached chosen KL for stability. If False, use TDPO1.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
    """

    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.1,
        alpha=0.5,
        if_tdpo2=True,
        max_epochs: int = 2,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.beta = beta
        self.alpha = alpha
        self.if_tdpo2 = if_tdpo2
        self.loss_fn = TDPOLoss(self.beta, self.alpha, self.if_tdpo2)
        # check ref_model is correct case
        if not self.ref_model and not isinstance(strategy._unwrap_model(model), PeftModelForCausalLM):
            raise ValueError("ref_model is None, but model is not a PeftModel")

        self.is_peft_model = isinstance(strategy._unwrap_model(model), PeftModelForCausalLM)
        
        self.ref_adapter_name = getattr(self.strategy.args, "ref_adapter_name", None)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.nll_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.eval_steps == -2:
            args.eval_steps = float("inf")  # Evaluate once per epoch

        if args.save_steps == -1:
            args.save_steps = num_update_steps_per_epoch  # do not save ckpt
        if args.save_steps == -2:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        # --------- calculate max_global_step and ratio-based save step ---------  ###
        # global_step = optimizer step
        # assume one epoch has num_update_steps_per_epoch optimizer steps
        self.max_global_step = self.epochs * num_update_steps_per_epoch

        # save_ratios = [0.5]
        save_ratios=None
        if save_ratios:
            # int() down. minimum 1 step is max(1, ...)
            ratio_steps = {max(1, int(self.max_global_step * r)) for r in save_ratios}
            # if same value, remove duplicates and sort
            self.ratio_save_steps = sorted(ratio_steps)
            self.saved_ratio_steps = set()

            if self.strategy.is_rank_0():
                self.strategy.print(
                    f"[DPOTrainer] max_global_step = {self.max_global_step}, "
                    f"ratio_save_steps (0.5) = {self.ratio_save_steps}"
                )
        else:
            self.ratio_save_steps = None
            self.saved_ratio_steps = None
        log_ratio_step = getattr(args, "log_ratio_step", None)
        if log_ratio_step is not None:
            if log_ratio_step < 1:
                log_ratios = [i*log_ratio_step for i in range(1, int(1//log_ratio_step)+2) if i*log_ratio_step <= 1]
            else:
                log_ratios = [int(i*log_ratio_step) for i in range(1, int(self.max_global_step//log_ratio_step)+2) if int(i*log_ratio_step) <= self.max_global_step]
            log_ratio_steps = {max(1, int(self.max_global_step * r)) for r in log_ratios}
            self.ratio_log_steps = sorted(log_ratio_steps)
            self.logged_ratio_steps = set()
        else:
            self.ratio_log_steps = None
            self.logged_ratio_steps = None
 
        # ------------------------------------------------------------------  ###


        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        acc_sum = 0
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model is not None:
                self.ref_model.eval()
            # train
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, aux_loss, nll_loss = self.tdpo_concatenated_forward(
                        self.model, self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, aux_loss, nll_loss = self.tdpo_packed_samples_forward(
                        self.model, self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )

                # loss function
                preference_loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl
                )
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                # nll loss
                if not self.nll_loss:
                    nll_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef + nll_loss * self.args.nll_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # dpo logs
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        # logs
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if args.eval_steps == -2:
            pass
        elif self.ratio_log_steps is not None and global_step in self.ratio_log_steps and global_step not in self.logged_ratio_steps:
            self.logged_ratio_steps.add(global_step)# do eval when len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # ---------------- ratio-based save logic ------------------------------  ###
        save_by_ratio = False
        if self.ratio_save_steps is not None:
            if global_step in self.ratio_save_steps and global_step not in self.saved_ratio_steps:
                save_by_ratio = True
                self.saved_ratio_steps.add(global_step)

        # (optional) if you want to keep the old way, only use it when needed
        save_by_step = False
        if args.save_steps not in (float("inf"), 0) and args.save_steps > 0:
            if global_step % args.save_steps == 0:
                save_by_step = True

        # here, "ratio-based save" is the main requirement,
        # usually save_by_ratio is enough and save_by_step is turned off.
        if save_by_ratio or save_by_step:
            tag = f"global_step{global_step}"
            if self.strategy.is_rank_0():
                self.strategy.print(
                    f"[DPOTrainer] Saving checkpoint at global_step={global_step} "
                    f"(ratio-based={save_by_ratio}, step-based={save_by_step})"
                )
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

        # # save ckpt
        # # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        # if global_step % args.save_steps == 0:
        #     tag = f"global_step{global_step}"
        #     if not self.disable_ds_ckpt:
        #         self.strategy.save_ckpt(
        #             self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
        #         )
        #     if self.save_hf_ckpt:
        #         save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
        #         self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, aux_loss, _ = self.tdpo_concatenated_forward(
                        self.model, self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, aux_loss, _ = self.tdpo_packed_samples_forward(
                        self.model, self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )

                loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times,
                "acc_mean": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def tdpo_concatenated_forward(self, model, ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens_cat = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        
        # Forward pass for policy model
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        
        # Forward pass for reference model
        if ref_model is not None:
            with torch.no_grad():
                ref_output = ref_model(input_ids, attention_mask=att_masks, return_output=True)
                reference_all_logits = ref_output["logits"]
        else:
            # Use model with disabled adapter as reference
            with torch.no_grad():
                with self.null_ref_context():
                    ref_output = model(input_ids, attention_mask=att_masks, return_output=True)
                    reference_all_logits = ref_output["logits"]
        
        # Compute TDPO-specific metrics
        all_logps_margin, all_position_kl, all_logps_sum = self._tdpo_get_batch_logps(
            all_logits, reference_all_logits, input_ids, att_masks, prompt_id_lens_cat, average_log_prob=False
        )
        
        chosen_logps_margin = all_logps_margin[: chosen_ids.shape[0]]
        rejected_logps_margin = all_logps_margin[chosen_ids.shape[0] :]
        chosen_position_kl = all_position_kl[: chosen_ids.shape[0]]
        rejected_position_kl = all_position_kl[chosen_ids.shape[0] :]
        
        aux_loss = output.aux_loss if "aux_loss" in output else []
        nll_loss = -all_logps_sum[: chosen_ids.shape[0]].mean()
        
        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, aux_loss, nll_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        
        with self.strategy._unwrap_model(self.model).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")



    def _tdpo_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        reference_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ):
        """Compute the kl divergence/log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            attention_mask: Attention mask for the input
            prompt_id_lens: List of prompt lengths
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape
        assert reference_logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        reference_logits = reference_logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0

        # Compute log probabilities and KL divergence
        vocab_logps = logits.log_softmax(-1)
        reference_vocab_ps = reference_logits.softmax(-1)
        reference_vocab_logps = reference_vocab_ps.log()

        # Per-position KL divergence: KL(ref || policy)
        per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
        
        # Per-token log probabilities
        per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # Log probability margin: policy - reference
        logps_margin = per_token_logps - per_reference_token_logps

        if average_log_prob:
            return (
                (logps_margin * loss_masks).sum(-1) / loss_masks.sum(-1),
                (per_position_kl * loss_masks).sum(-1) / loss_masks.sum(-1),
                (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
            )
        else:
            return (
                (logps_margin * loss_masks).sum(-1),
                (per_position_kl * loss_masks).sum(-1),
                (per_token_logps * loss_masks).sum(-1)
            )

    def tdpo_packed_samples_forward(self, model, ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        # Forward pass for policy model
        output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        all_logits = output["logits"]
        
        # Forward pass for reference model
        if ref_model is not None:
            with torch.no_grad():
                ref_output = ref_model(
                    packed_input_ids,
                    attention_mask=packed_attention_masks,
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=packed_seq_lens,
                )
                reference_all_logits = ref_output["logits"]
        else:
            # Use model with disabled adapter as reference
            with torch.no_grad():
                with self.null_ref_context():
                    ref_output = model(
                        packed_input_ids,
                        attention_mask=packed_attention_masks,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=packed_seq_lens,
                    )
                    reference_all_logits = ref_output["logits"]
        
        # Compute TDPO-specific metrics
        all_logps_margin, all_position_kl, all_logps_mean = self._tdpo_packed_get_batch_logps(
            all_logits,
            reference_all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=False,
        )
        
        chosen_logps_margin = all_logps_margin[: len(packed_seq_lens) // 2]
        rejected_logps_margin = all_logps_margin[len(packed_seq_lens) // 2 :]
        chosen_position_kl = all_position_kl[: len(packed_seq_lens) // 2]
        rejected_position_kl = all_position_kl[len(packed_seq_lens) // 2 :]
        
        aux_loss = output.aux_loss if "aux_loss" in output else []
        nll_loss = -all_logps_mean[: len(packed_seq_lens) // 2].mean()
        
        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, aux_loss, nll_loss

    def _tdpo_packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        reference_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
    ):
        """Compute TDPO metrics for packed samples."""
        assert average_log_prob == False

        if self.strategy.ring_attn_group is None:
            assert logits.shape[:-1] == labels.shape
            assert reference_logits.shape[:-1] == labels.shape
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]
            reference_logits = reference_logits[:, :-1, :]
            
            # Compute log probabilities and KL divergence
            vocab_logps = logits.log_softmax(-1)
            reference_vocab_ps = reference_logits.softmax(-1)
            reference_vocab_logps = reference_vocab_ps.log()

            # Per-position KL divergence
            per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
            
            # Per-token log probabilities
            per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            
            # Log probability margin
            per_token_logps_margin = per_token_logps - per_reference_token_logps
        else:
            rank = self.strategy.ring_attn_rank
            total_seq_len = labels.numel()
            local_seq_len = total_seq_len // self.strategy.ring_attn_size
            local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
            local_label = labels[:, local_slice]
            if rank == self.strategy.ring_attn_size - 1:
                # add a dummy label to the last logit
                local_label = F.pad(local_label, (0, 1), value=0)

            # Compute for policy model
            vocab_logps = logits.log_softmax(-1)
            local_per_token_logps = torch.gather(vocab_logps, dim=2, index=local_label.unsqueeze(2)).squeeze(2)
            per_token_logps = all_gather(local_per_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
            per_token_logps = per_token_logps[:, :-1]
            
            # Compute for reference model
            reference_vocab_ps = reference_logits.softmax(-1)
            reference_vocab_logps = reference_vocab_ps.log()
            local_per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=local_label.unsqueeze(2)).squeeze(2)
            per_reference_token_logps = all_gather(local_per_reference_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
            per_reference_token_logps = per_reference_token_logps[:, :-1]
            
            # Compute KL
            local_per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
            per_position_kl = all_gather(local_per_position_kl, self.strategy.ring_attn_group).reshape((1, -1))
            per_position_kl = per_position_kl[:, :-1]
            
            # Log probability margin
            per_token_logps_margin = per_token_logps - per_reference_token_logps

        loss_masks = attention_mask.clone().bool()

        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            loss_masks[0, index : index + prompt_id_lens[i]] = False
            index = index + seq_len

        loss_masks = loss_masks[:, 1:]

        logprobs_margin_sums = []
        position_kl_sums = []
        logprobs_sums = []
        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            margin_seq = per_token_logps_margin[0, index : index + seq_len - 1]
            kl_seq = per_position_kl[0, index : index + seq_len - 1]
            logps_seq = per_token_logps[0, index : index + seq_len - 1] if self.strategy.ring_attn_group is None else per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            
            logprobs_margin_sums.append((margin_seq * mask).sum())
            position_kl_sums.append((kl_seq * mask).sum())
            logprobs_sums.append((logps_seq * mask).sum())
            index = index + seq_len

        return torch.stack(logprobs_margin_sums), torch.stack(position_kl_sums), torch.stack(logprobs_sums)
