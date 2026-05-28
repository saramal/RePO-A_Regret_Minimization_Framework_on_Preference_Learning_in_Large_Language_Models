import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from peft import PeftModel, PeftModelForCausalLM
from contextlib import contextmanager, nullcontext

from openrlhf.models import LogExpLoss, PairWiseLoss, RePO_Loss, GPTLMLoss, RePO_Unvalanced_Loss
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.models.utils import log_probs_from_logits
from openrlhf.utils import match_with_answer_labels_v2, extract_last_answer

import re
import os
import json
from openrlhf.utils.utils import extract_first_numeric_answer

from torch import distributed as dist

class RePO_Unbalanced_Trainer(ABC):
    """
    Trainer for training a RePO model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to aRePOy.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        ref_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        loss="unvalanced_RePO",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.ref_model = ref_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt


        self.cpl_lambda = self.strategy.args.cpl_lambda
        self.ref_coef = self.strategy.args.ref_coef
        self.sft_loss_coef = self.strategy.args.sft_loss_coef
        



        self.beta = getattr(self.strategy.args, "beta", None)
        assert self.beta is not None, "beta must be set for unvalanced RePO loss"
        self.positive_ratio = getattr(self.strategy.args, "positive_ratio", 1.0)
        self.negative_ratio = getattr(self.strategy.args, "negative_ratio", 1.0)
        self.loss_fn = RePO_Unvalanced_Loss(self.beta, self.ref_coef)
        

        
        self.sft_loss_fn = GPTLMLoss()
        
        
        # check ref_model is correct case
        if not self.ref_model and not isinstance(strategy._unwrap_model(model), PeftModelForCausalLM):
            raise ValueError("ref_model is None, but model is not a PeftModel")
        
        self.is_peft_model = isinstance(strategy._unwrap_model(model), PeftModelForCausalLM)
        # self.model_adapter_name = strategy.args.model_adapter_name
        self.ref_adapter_name = self.strategy.args.ref_adapter_name
        # self.reference_free = strategy.args.reference_free
        
        
        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.sft_loss = self.strategy.args.sft_loss
        self.disable_ref_loss = self.strategy.args.disable_ref_loss
        
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

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
        # torch.autograd.set_detect_anomaly(True)
        print_examples = True
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        acc_sum = 0
        loss_sum = 0
        RePO_loss_sum = 0
        sft_loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            
            # import pdb
            # pdb.set_trace()



            for local_step, data in enumerate(self.train_dataloader):
                # Resume - skip until stopped step
                if epoch == start_epoch and local_step < step % self.train_dataloader.__len__():
                    continue
                
                chosen_data_mask = int(len(self.train_dataloader)* self.strategy.args.positive_ratio) > local_step
                reject_data_mask = int(len(self.train_dataloader)* self.strategy.args.negative_ratio) > local_step
                
                if not self.packing_samples:
                    # chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_ids, prompt_masks, answer_label, extra = data
                    
                    if print_examples:
                        if self.strategy.is_rank_0():
                            
                            prompt_text = self.tokenizer.decode(data["prompt_ids"][0][0], skip_special_tokens=True)
                            chosen_text = self.tokenizer.decode(data["chosen_ids"][0][0], skip_special_tokens=True)
                            reject_text = self.tokenizer.decode(data["reject_ids"][0][0], skip_special_tokens=True)
                            
                            self.strategy.print(f"\n Training data example: \nprompt:\n{prompt_text}\n")
                            self.strategy.print(f"\n chosen: \n{chosen_text}\n")
                            self.strategy.print(f"\n rejected: \n{reject_text}\n")
                        print_examples=False
                    
                    chosen_ids = data["chosen_ids"].squeeze(1).to(torch.cuda.current_device())
                    c_mask = data["chosen_masks"].squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_logp_labels = data["chosen_logprob_labels"].squeeze(1).to(torch.cuda.current_device())
                    c_label_mask = data["chosen_logprob_masks"].squeeze(1).to(torch.cuda.current_device())
                    
                    
                    reject_ids = data["reject_ids"].squeeze(1).to(torch.cuda.current_device())
                    r_mask = data["rejects_masks"].squeeze(1).to(torch.cuda.current_device())
                    
                    reject_logp_labels = data["rejected_logprob_labels"].squeeze(1).to(torch.cuda.current_device())
                    r_label_mask = data["rejected_logprob_masks"].squeeze(1).to(torch.cuda.current_device())

                    prompt_ids = data["prompt_ids"].to(torch.cuda.current_device())
                    prompt_masks = data["prompt_masks"].to(torch.cuda.current_device())
                    prompt_id_lens = prompt_masks.sum(dim=-1)
                    
                    answers = data["answers"]
                    extras = data["extras"]
                    # import pdb
                    # pdb.set_trace()
                    # forward
                    RePO_forward_output = self.concatenated_forward(
                            self.model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens
                        )
                else:
                    raise ValueError("Packing is not implemented for RePO")
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, reject_reward, aux_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )

                if self.sft_loss:
                    labels = torch.where(
                    c_mask.bool(),
                    chosen_ids,
                    self.sft_loss_fn.IGNORE_INDEX,
                    ).to(dtype=torch.int64)
                    for label, source_len in zip(labels, prompt_masks.sum(dim=-1)):
                        label[: source_len + 1] = self.sft_loss_fn.IGNORE_INDEX
                    chosen_logits = RePO_forward_output["chosen_target_model_logits"][:, :chosen_ids.shape[1], :].contiguous()
                    assert chosen_ids.shape[1] == chosen_logp_labels.shape[1] == c_label_mask.shape[1]
                    sft_loss = self.sft_loss_fn(chosen_logits, labels)
                else:
                    sft_loss = 0
                
                if not self.disable_ref_loss:                    
                    ref_RePO_precalculated = extras["ref_"] if  "ref_" in extras else None
                    if not ref_RePO_precalculated:
                        
                        if self.ref_model:
                            with torch.no_grad():
                                ref_RePO_forward_output = self.concatenated_forward(
                                    self.ref_model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens
                                )
                        else:
                            # treat model with disabled_adpater as ref_model
                            ref_RePO_forward_output = self.ref_concatenated_forward(
                                chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens
                            )
                    else:
                        ref_RePO_forward_output = self.ref_cached_forward(
                            ref_RePO_precalculated["ref_chosen_target_model_logprobs"], c_mask, chosen_logp_labels, c_label_mask,
                            ref_RePO_precalculated["ref_reject_target_model_logprobs"], r_mask, reject_logp_labels, r_label_mask,
                        )
                        
                    RePO_forward_output["ref_chosen_target_model_logprobs"] = ref_RePO_forward_output["chosen_target_model_logprobs"]
                    RePO_forward_output["ref_rejected_target_model_logprobs"] = ref_RePO_forward_output["rejected_target_model_logprobs"]
                    
                    # check_ref_model_work = RePO_forward_output["chosen_target_model_logprobs"] == ref_RePO_forward_output["chosen_target_model_logprobs"]
                    # check_ref_model_work = check_ref_model_work.sum(dim=-1)
                    
                    #TODO: check ref model work
                    # import pdb
                    # pdb.set_trace()
                    # if RePO_forward_output["chosen_target_model_logprobs"].shape[1] == int(check_ref_model_work[0]):
                    #     self.strategy.print("ref model is NOT working!")

                else:
                    ref_loss = 0

                # loss function
                if self.compute_fp32_loss:
                    raise ValueError("compute_fp32_loss is not implemented for RePO")
                    chosen_logp = chosen_logp.float()
                    reject_logp = reject_logp.float()


                # RePO loss
                preference_loss, chosen_negative_regret, reject_negative_regret, \
                chosen_target_regret, reject_target_regret, \
                chosen_ref_regret, reject_ref_regret = self.loss_fn(RePO_forward_output, chosen_data_mask, reject_data_mask)

                
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                
                loss = preference_loss + aux_loss * self.args.aux_loss_coef + sft_loss * self.sft_loss_coef
                
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                
                acc = (chosen_negative_regret > reject_negative_regret).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()
                RePO_loss_sum += preference_loss.item()
                sft_loss_sum += sft_loss.item() if sft_loss else 0
                # optional rm info
                logs_dict = {
                    "loss": loss.item(),
                    "RePO_loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_negative_regret": chosen_negative_regret.mean().item(),
                    "reject_negative_regret": reject_negative_regret.mean().item(),
                    "chosen_target_regret": chosen_target_regret,
                    "reject_target_regret": reject_target_regret,
                    "chosen_ref_regret": chosen_ref_regret if chosen_ref_regret is not None else 0,
                    "reject_ref_regret": reject_ref_regret if reject_ref_regret is not None else 0,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                if self.sft_loss:
                    logs_dict["sft_loss"] = sft_loss.item()
                # if not self.disable_ref_loss:
                #     logs_dict["ref_loss"] = ref_loss.item()

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    RePO_loss_sum = 0
                    sft_loss_sum = 0
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
            pass  # do not eval
        
        elif global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            #imported from dpo_trainer
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)
            # self.strategy.save_ckpt(
            #     self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            # )

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        if self.strategy.is_rank_0():
            os.makedirs(self.strategy.args.generation_log_path, exist_ok=True)
        save_path = os.path.join(self.strategy.args.generation_log_path, f"eval_{steps}.jsonl")
        self.model.eval()
        with torch.no_grad():
            acc = 0
            regrets = []
            chosen_regrets = []
            rejected_regrets = []
            RePO_loss_sum = 0
            sft_loss_sum = 0
            times = 0
            for step, data in enumerate(eval_dataloader):
                chosen_data_mask = int(len(eval_dataloader) * self.strategy.args.positive_ratio) > step
                reject_data_mask = int(len(eval_dataloader) * self.strategy.args.negative_ratio) > step
                if not self.packing_samples:
                    if not self.strategy.args.disable_eval_loss:
                        # print(f"Rank {self.strategy.get_rank()} : for loop start\n")
                        # chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_ids, prompt_masks, answer_label, extra = data
                    
                    
                        chosen_ids = data["chosen_ids"].squeeze(1).to(torch.cuda.current_device())
                        c_mask = data["chosen_masks"].squeeze(1).to(torch.cuda.current_device())
                        
                        chosen_logp_labels = data["chosen_logprob_labels"].squeeze(1).to(torch.cuda.current_device())
                        c_label_mask = data["chosen_logprob_masks"].squeeze(1).to(torch.cuda.current_device())
                        
                        
                        reject_ids = data["reject_ids"].squeeze(1).to(torch.cuda.current_device())
                        r_mask = data["rejects_masks"].squeeze(1).to(torch.cuda.current_device())
                        
                        reject_logp_labels = data["rejected_logprob_labels"].squeeze(1).to(torch.cuda.current_device())
                        r_label_mask = data["rejected_logprob_masks"].squeeze(1).to(torch.cuda.current_device())

                        prompt_ids = data["prompt_ids"].to(torch.cuda.current_device())
                        prompt_masks = data["prompt_masks"].to(torch.cuda.current_device())
                        prompt_id_lens = prompt_masks.sum(dim=-1)
                        
                        answers = data["answers"]
                        extras = data["extras"]


                        # forward
                        RePO_forward_output = self.concatenated_forward(
                            self.model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens
                        )
                        # print(f"Rank {self.strategy.get_rank()} : finish forward")

                        
                        
                        if self.sft_loss:  
                            labels = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.sft_loss_fn.IGNORE_INDEX,
                            ).to(dtype=torch.int64)
                            for label, source_len in zip(labels, prompt_masks.sum(dim=-1)):
                                label[: source_len + 1] = self.sft_loss_fn.IGNORE_INDEX
                            chosen_logits = RePO_forward_output["chosen_target_model_logits"][:, :chosen_ids.shape[1], :].contiguous()
                            assert chosen_ids.shape[1] == chosen_logp_labels.shape[1] == c_label_mask.shape[1]
                            sft_loss = self.sft_loss_fn(chosen_logits, labels)
                        else:
                            sft_loss = 0
                        # print(f"Rank {self.strategy.get_rank()} : finish sft loss cal.\n")
                        if not self.disable_ref_loss:                    
                            ref_RePO_precalculated = extras["ref_"] if  "ref_" in extras else None
                            if not ref_RePO_precalculated:
                                
                                if self.ref_model:
                                    with torch.no_grad():
                                        ref_RePO_forward_output = self.concatenated_forward(
                                            self.ref_model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens
                                        )
                                else:
                                    # treat model with disabled_adapter as ref_model
                                    ref_RePO_forward_output = self.ref_concatenated_forward(
                                        chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens
                                    )
                            else:
                                ref_RePO_forward_output = self.ref_cached_forward(
                                    ref_RePO_precalculated["ref_chosen_target_model_logprobs"], c_mask, chosen_logp_labels, c_label_mask,
                                    ref_RePO_precalculated["ref_reject_target_model_logprobs"], r_mask, reject_logp_labels, r_label_mask,
                                )
                            
                            RePO_forward_output["ref_chosen_target_model_logprobs"] = ref_RePO_forward_output["chosen_target_model_logprobs"]
                            RePO_forward_output["ref_rejected_target_model_logprobs"] = ref_RePO_forward_output["rejected_target_model_logprobs"]
                                                
                        else:
                            ref_loss = 0

                        # loss function
                        if self.compute_fp32_loss:
                            chosen_logp = chosen_logp.float()
                            reject_logp = reject_logp.float()


                        # print(f"Rank {self.strategy.get_rank()} : finish regret cal\n")
                        
                        # RePO loss
                        preference_loss, chosen_negative_regret, reject_negative_regret, \
                        chosen_target_regret, reject_target_regret, \
                        chosen_ref_regret, reject_ref_regret = self.loss_fn(RePO_forward_output, chosen_data_mask, reject_data_mask)
                        
                        # mixtral
                        if not self.aux_loss:
                            aux_loss = 0


                        regrets += [chosen_negative_regret.flatten(), reject_negative_regret.flatten()]
                        chosen_regrets += [chosen_negative_regret.flatten()]
                        rejected_regrets += [reject_negative_regret.flatten()]
                        
                        RePO_loss_sum += preference_loss.item()
                        sft_loss_sum += sft_loss.item() if self.sft_loss else 0
                        times += 1
                        # loss_sum += loss.item()
                        # print(f"Rank {self.strategy.get_rank()} : end of loss calc\n")
                    if self.strategy.args.eval_acc:
                        # print(f"Rank {self.strategy.get_rank()} : start of eval acc\n")
                        # _, _, _, _, _, _, _, _, prompt_ids, prompt_masks, answer_label, extra = data
                        prompt_ids = data["prompt_ids"].squeeze(1).to(torch.cuda.current_device())
                        prompt_masks = data["prompt_masks"].squeeze(1).to(torch.cuda.current_device())
                        answers = data["answers"]

                        #TODO: fix generate function! refer to data generation step in RePO old code
                        
                        # generated_outputs = self.model.generate(prompt_ids, prompt_masks)
                        # model_input_for_generation = {"input_ids": prompt_ids, "attention_mask": prompt_masks}
                        # print(f"Rank {self.strategy.get_rank()} : ready to generate\n")
                        generated_outputs, _, _ = self.model.generate(
                                                input_ids=prompt_ids,
                                                attention_mask=prompt_masks,
                                                use_cache=True,
                                                max_length=None,
                                                max_new_tokens=self.strategy.args.generation_max_len,
                                                do_sample=True,
                                                top_p=self.strategy.args.top_p,
                                                early_stopping=False,
                                                num_beams=1,
                                                temperature=self.strategy.args.temperature,
                                                repetition_penalty=self.strategy.args.repetition_penalty,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                            )
                        # print(f"Rank {self.strategy.get_rank()} : generation end\n")
                        tokenized_output = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
                        # print(f"Rank {self.strategy.get_rank()} : token decode end\n")
                        if self.strategy.args.generation_log_path:
                            # tokenized_output = self.strategy.all_gather(tokenized_output)
                            # gathered_answers = self.strategy.all_gather(answers)
                            # save generation log
                            if self.strategy.is_rank_0():
                                with open(save_path, 'a') as f:
                                    for generation, answer in zip(tokenized_output, answers):
                                        generation_dict = {"generation": generation, 
                                                        # "extracted_answers": self.extract_first_numeric_answer(generation, self.strategy.args.answer_trigger),
                                                        "extracted_answers": extract_last_answer(generation, self.strategy.args.answer_trigger),
                                                        "gold_answers": answer}
                                        f.write(json.dumps(generation_dict, ensure_ascii=False) + "\n")
                        # chosen_reward, reject_reward, _ = self.concatenated_forward(
                        #     self.model, chosen_ids, c_mask, reject_ids, r_mask
                        # )
                        # import pdb
                        # pdb.set_trace()
                        # print(f"Rank {self.strategy.get_rank()} : answer match end\n")
                        # acc += self.match_with_answer_labels(tokenized_output, answers)
                        acc += match_with_answer_labels_v2(tokenized_output, answers, self.strategy.args.answer_trigger)
                        dist.barrier()
                    
                else:
                    raise ValueError("Packing is not implemented for RePO")
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, reject_reward, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )

                
                step_bar.update()





            bar_dict = {}
            if not self.strategy.args.disable_eval_loss:
                RePO_loss_mean = RePO_loss_sum / self.eval_dataloader.__len__()
                sft_loss_mean = sft_loss_sum / self.eval_dataloader.__len__()

                regrets = torch.cat(regrets).float()
                regrets = self.strategy.all_gather(regrets)
                regret_mean = torch.mean(regrets)
                regret_std = torch.std(regrets).clamp(min=1e-8)
                
                chosen_regrets = torch.cat(chosen_regrets).float()
                chosen_regrets = self.strategy.all_gather(chosen_regrets)
                chosen_regret_mean = torch.mean(chosen_regrets)
                chosen_regret_std = torch.std(chosen_regrets).clamp(min=1e-8)
                
                rejected_regrets = torch.cat(rejected_regrets).float()
                rejected_regrets = self.strategy.all_gather(rejected_regrets)
                rejected_regrets_mean = torch.mean(rejected_regrets)
                rejected_regrets_std = torch.std(rejected_regrets).clamp(min=1e-8)

                # save mean std
                self.strategy.print("Set regret mean std")
                unwrap_model = self.strategy._unwrap_model(self.model)
                unwrap_model.config.mean = regret_mean.item()
                unwrap_model.config.std = regret_std.item()
                
                bar_dict["eval_RePO_loss"] = RePO_loss_mean
                bar_dict["eval_sft_loss"] = sft_loss_mean
                bar_dict["regret_mean"] = regret_mean.item()
                bar_dict["regret_std"] = regret_std.item()
                
                bar_dict["chosen_regret_mean"] = chosen_regret_mean.item()
                bar_dict["chosen_regret_std"] = chosen_regret_std.item()
                bar_dict["rejected_regrets_mean"] = rejected_regrets_mean.item()
                bar_dict["rejected_regrets_std"] = rejected_regrets_std.item()
                
            if self.strategy.args.eval_acc:
                acc_mean = acc / self.eval_dataloader.__len__()
                bar_dict["acc_mean"] = acc_mean
            

            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            # Skip print histgram
            
            # histgram = torch.histogram(regrets.cpu(), bins=10, range=(-10, 10), density=True) * 2
            # self.strategy.print("histgram")
            # self.strategy.print(histgram)
            if self.strategy.is_rank_0() and self.strategy.args.eval_acc:
                self.strategy.print(f"Evaluation finished, accuracy: {logs['acc_mean']:.4f}")
                with open(save_path, 'a') as f:
                    f.write(json.dumps(f"Evaluation finished, accuracy: {logs['acc_mean']:.4f}\n") + "\n")

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        label_logprobs, label_masks = self.concatenated_inputs(chosen_logp_labels, c_label_mask, reject_logp_labels, r_label_mask)
        
        ## use num_actions? Actor's forward function can return log probs, but return last 'num_actions' logprobs.
        target_model_outputs = model(input_ids, attention_mask=att_masks, return_output=True)
        
        # import pdb
        # pdb.set_trace()
        # with model.model.disable_adapter():
        #     ref_target_model_outputs_1 = model(input_ids, attention_mask=att_masks, return_output=True)
        


        # Method 1: using log_probs_from_logits
        target_model_outputs["logits"] = target_model_outputs["logits"].contiguous().to(torch.float32)  # ensure logits are float for log_probs_from_logits
        
        labels = input_ids[:, 1:].clone()  # shift input_ids to the right to match the logits
        label_masks = label_masks[:, 1:].bool()
        label_logprobs = label_logprobs[:, 1:]

        labels[~label_masks] = 0  # set labels to -100 where label_masks is False
        per_token_logps = log_probs_from_logits(target_model_outputs["logits"][:,:-1,:], labels)
        
        per_token_logps[~label_masks] = 0  # set log probabilities to 0 where label_masks is False  
        label_logprobs[~label_masks] = 0  # set label log probabilities to 0 where label_masks is False


        # method 2: using _get_batch_logps
        # per_token_logps, all_logps_sum, all_logps_mean = self._get_batch_logps(
        #     target_model_outputs["logits"], input_ids, att_masks, prompt_id_lens, average_log_prob=False
        # )
        
        # dummy_column = torch.full((per_token_logps.size(0), 1), 0, dtype=per_token_logps.dtype, device=per_token_logps.device)
        # resized_per_token_logps = torch.cat([dummy_column, per_token_logps.clone()], dim=1)
        
        RePO_forward_output = {}
        RePO_forward_output["chosen_target_model_logits"] = target_model_outputs["logits"][: chosen_ids.shape[0]].contiguous()
        RePO_forward_output["rejected_target_model_logits"] = target_model_outputs["logits"][chosen_ids.shape[0] :].contiguous()
        
        RePO_forward_output["chosen_target_model_logprobs"] = per_token_logps[: chosen_ids.shape[0]].contiguous()
        RePO_forward_output["rejected_target_model_logprobs"] = per_token_logps[chosen_ids.shape[0] :].contiguous()
        
        RePO_forward_output["chosen_att_masks"] = att_masks[: chosen_ids.shape[0]].contiguous()
        RePO_forward_output["rejected_att_masks"] = att_masks[chosen_ids.shape[0] :].contiguous()
        
        RePO_forward_output["chosen_label_logprobs"] = label_logprobs[: chosen_ids.shape[0]].contiguous()
        RePO_forward_output["rejected_label_logprobs"] = label_logprobs[chosen_ids.shape[0] :].contiguous()
        
        RePO_forward_output["chosen_label_masks"] = label_masks[: chosen_ids.shape[0]].contiguous()
        RePO_forward_output["rejected_label_masks"] = label_masks[chosen_ids.shape[0] :].contiguous()
        
        assert (chosen_ids.shape[0] == chosen_logp_labels.shape[0] == c_label_mask.shape[0])
        assert (reject_ids.shape[0] == reject_logp_labels.shape[0] == r_label_mask.shape[0])
        
        RePO_forward_output["aux_loss"] = target_model_outputs.aux_loss if "aux_loss" in target_model_outputs else []
        
        return RePO_forward_output
        

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
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
                # right pad
                #TODO: adjust pad side for RePO dataset -> right be ok...not left maybe
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
        return inputs_ids, att_masks
    
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:]
        logits = logits[:, :-1, :].clone()

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = log_probs_from_logits(logits, labels)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return per_token_logps * loss_masks, logprobs_sums, logprobs_means
    
    # reference model calculation
    # TODO: read here: https://github.com/huggingface/peft/issues/1523 
    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        
        with self.strategy._unwrap_model(self.model).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")
    
    def ref_concatenated_forward(self, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        # compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        compte_ref_context_manager = nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, chosen_ids, c_mask, chosen_logp_labels, c_label_mask, reject_ids, r_mask, reject_logp_labels, r_label_mask, prompt_id_lens)
        return ref_model_output
    
    
    def ref_cached_forward(self, chosen_ref_logprobs, c_mask, chosen_logp_labels, c_label_mask, rejected_ref_logprobs, r_mask, reject_logp_labels, r_label_mask):
        """
        re-formulate pre-computed reference model's output into RePO output form
        """
        concated_ref_logprobs, att_masks = self.concatenated_inputs(chosen_ref_logprobs, c_mask, rejected_ref_logprobs, r_mask)
        label_logprobs, label_masks = self.concatenated_inputs(chosen_logp_labels, c_label_mask, reject_logp_labels, r_label_mask)
        
        # target_model_outputs = model(input_ids, attention_mask=att_masks, return_output=True)
        
        RePO_forward_output = {}
        RePO_forward_output["chosen_target_model_logprobs"] = concated_ref_logprobs[: chosen_ref_logprobs.shape[0]]
        RePO_forward_output["rejected_target_model_logprobs"] = concated_ref_logprobs[chosen_ref_logprobs.shape[0] :]
        
        RePO_forward_output["chosen_att_masks"] = att_masks[: chosen_ref_logprobs.shape[0]]
        RePO_forward_output["rejected_att_masks"] = att_masks[chosen_ref_logprobs.shape[0] :]
        
        RePO_forward_output["chosen_label_logprobs"] = label_logprobs[: chosen_ref_logprobs.shape[0]]
        RePO_forward_output["rejected_label_logprobs"] = label_logprobs[chosen_ref_logprobs.shape[0] :]
        
        RePO_forward_output["chosen_label_masks"] = label_masks[: chosen_ref_logprobs.shape[0]]
        RePO_forward_output["rejected_label_masks"] = label_masks[chosen_ref_logprobs.shape[0] :]
        
        assert (chosen_ref_logprobs.shape[0] == chosen_logp_labels.shape[0] == c_label_mask.shape[0])
        assert (rejected_ref_logprobs.shape[0] == reject_logp_labels.shape[0] == r_label_mask.shape[0])
        
        
        # RePO_forward_output["aux_loss"] = target_model_outputs.aux_loss if "aux_loss" in target_model_outputs else []
        
        return RePO_forward_output        

    
    
    def match_with_answer_labels(self, tokenized_output, answers):
        #TODO: current case is only work for gsm8k. find appropriate match with MATH or else.
        answer_trigger = self.strategy.args.answer_trigger
        correct_count = 0
        valid_count = 0
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                predicted_answer = self.extract_first_numeric_answer(output, answer_trigger)
                if predicted_answer is None:
                    continue
                is_correct = self.check_correctness(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1
        
        return correct_count/valid_count if valid_count > 0 else 0
    

    def extract_answer(self, output, answer_trigger):
        
        def extract_first_value(s, answer_trigger):
            pattern = rf"\b(?:the {re.escape(answer_trigger)}|therefore, the {re.escape(answer_trigger)})\s*(\d+)\b"
            matches = re.findall(pattern, s, flags=re.IGNORECASE)
            return float(matches[0]) if matches else None
        
        try:
            return extract_first_value(output, answer_trigger)
        except:
            return None
    
    def extract_first_numeric_answer(self, text:str, answer_trigger:str):
        import re
        # matches = []

        # pattern 1: \(\\boxed{ANSWER}\)
        # match1 = re.search(r'boxed\{(.*?)\}', text)
        # if match1:
        #     matches.append(('boxed', match1.start(), match1.group(1).strip()))

        # pattern 2: Therefore, the answer is: ANSWER.
        pattern = re.escape(answer_trigger) + r"\s*['\"]?(\d+(?:\.\d+)?)['\"]?"
        # match2 = re.search(r'Therefore, the answer is: ([^\.\n]+)', text)
        matches = re.findall(pattern, text)

        if not matches:
            return None

        answer = matches[-1].strip()
        return float(answer) if re.match(r'^\d+(\.\d+)?$', answer) else None


    # def extract_first_numeric_answer(self, text:str, answer_trigger:str):
    #     matches = []

    #     # pattern 1: \(\\boxed{ANSWER}\)
    #     match1 = re.search(r'\\\(\\boxed\{(.*?)\}\\\)', text)
    #     if match1:
    #         matches.append(('boxed', match1.start(), match1.group(1).strip()))

    #     # pattern 2: Therefore, the answer is: ANSWER.
    #     pattern2=re.escape(answer_trigger)+r'\s*([^\.\n]+)'
    #     # match2 = re.search(r'Therefore, the answer is: ([^\.\n]+)', text)
    #     match2 = re.search(pattern2, text)
    #     if match2:
    #         matches.append(('therefore', match2.start(), match2.group(1).strip()))

    #     if not matches:
    #         return None

    #     # extract first answer
    #     first = min(matches, key=lambda x: x[1])
    #     answer_text = first[2]

    #     # float / int extract
    #     num_match = re.search(r'\d+(?:\.\d+)?', answer_text)
    #     return float(num_match.group()) if num_match else None
        
        
    def check_correctness(self, prediction, target):
        return abs(float(prediction) - float(target)) <= 1e-3
        
    
    
    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens):
        raise ValueError("packing is not Implemented for RePO")
        all_values, output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        half_len = len(packed_seq_lens) // 2
        chosen_rewards = all_values[:half_len]
        rejected_rewards = all_values[half_len:]
        aux_loss = output.aux_loss if "aux_loss" in output else []

        return chosen_rewards, rejected_rewards, aux_loss
