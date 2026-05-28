import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import RewardDataset, RePODataset, RePO_datasets, RePODataset_fast
from openrlhf.models import Actor
from openrlhf.trainer import RewardModelTrainer, RePOTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from peft import PeftModelForCausalLM

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )

    # configure tokenizer
    # special_tokens = ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>', "<error_info>", "<end_of_error_info>", "<BACK>"]
    special_tokens = None
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer, special_token_list=special_tokens)


    strategy.print(model)

    # load weights for ref model
    if args.ref_pretrain:
        ref_model = Actor(
            args.ref_pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            packing_samples=args.packing_samples,
        )
        if args.ref_offload:
            ref_model._offload = True
        get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    else:
        ref_model=None
        
    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    
    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    # train_data, eval_data = RePO_datasets(
    #     args.train_data_path,
    #     args.eval_data_path,
    # )
    # import pdb; pdb.set_trace()
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=args.return_eval,
        stopping_strategy="all_exhausted",
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    if args.use_fast_dataset:
        train_dataset = RePODataset_fast(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            multiple_of=args.ring_attn_size,
        )
    else:
        train_dataset = RePODataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            multiple_of=args.ring_attn_size,
        )
    strategy.print(f"train_dataset: {len(train_dataset)}")
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    
    if eval_data is not None:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        if args.use_fast_dataset:
            eval_dataset = RePODataset_fast(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                input_template=args.input_template,
                multiple_of=args.ring_attn_size,
            )
        else:
            eval_dataset = RePODataset(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                input_template=args.input_template,
                multiple_of=args.ring_attn_size,
            )
        strategy.print(f"eval_dataset: {len(eval_dataset)}")
        
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_eval_batch_size,
            True,
            False,
            eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
        )
    else:
        eval_dataloader = None

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # strategy prepare
    if ref_model:
        ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)
    else:
        (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RePOTrainer(
        
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # # Save value_head_prefix
    # strategy.print("Save value_head_prefix in config")
    # unwrap_model = strategy._unwrap_model(model)
    # unwrap_model.config.value_head_prefix = args.value_head_prefix

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)
    if strategy.is_rank_0():
        with open("last_model_path.txt", "w") as f:
            f.write(args.save_path)

    if args.save_merged and isinstance(strategy._unwrap_model(model), PeftModelForCausalLM):
        strategy.print("\nSave merged model...\n")
        from transformers import AutoModelForCausalLM
        from torch import distributed as dist
        import torch
        merged_save_path = args.save_path + "_merged"
        if strategy.is_rank_0():
            os.makedirs(merged_save_path, exist_ok=True)
            tokenizer.save_pretrained(merged_save_path)
            #model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.pretrain, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16), args.save_path)
            model_to_merge = strategy._unwrap_model(model)
            merged_model = model_to_merge.merge_and_unload()
            merged_model.save_pretrained(save_directory=merged_save_path)
            # save config
            output_config_file = os.path.join(merged_save_path, "config.json")
            merged_model.config.to_json_file(output_config_file)
            with open("last_model_path.txt", "w") as f:
                f.write(merged_save_path)
        dist.barrier()
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Unvalanced RePO Loss
    parser.add_argument("--positive_ratio", type=float, default=1.0, help="positive ratio for unvalanced RePO loss")
    parser.add_argument("--negative_ratio", type=float, default=1.0, help="negative ratio for unvalanced RePO loss")
    parser.add_argument("--beta", type=float, default=0.5, help="beta for unvalanced RePO loss")
    parser.add_argument("--loss_type", type=str, default="RePO", choices=["RePO", "unvalanced_RePO"], help="loss function to use")
    
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/ckpt/checkpoints_RePO")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)
    parser.add_argument("--save_merged", action="store_true", default=False)

    parser.add_argument("--log_ratio_step", type=float, default=0.2, help="log ratio step for evaluation")
    parser.add_argument("--return_eval", action="store_true", default=False, help="return eval dataset")
    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--micro_eval_batch_size", type=int, default=1, help="batch size per GPU at evaluation")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    # parser.add_argument("--value_head_prefix", type=str, default="score")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    
    # Reference model with lora
    parser.add_argument("--ref_adapter_name", type=str, default=None, help="peft adapter name when reference model is implemeted with lora adapter")
    parser.add_argument("--ref_pretrain", type=str, default=None, help="ref model name or path")
    
    # RM training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--sft_loss", action="store_true", default=False)
    parser.add_argument("--disable_ref_loss", action="store_true", default=False)
    # parser.add_argument("--learning_rate", type=float, default=9e-6)
    # parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    # parser.add_argument("--micro_train_batch_size", type=int, default=1)
    # parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--loss", type=str, default="RePO")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    parser.add_argument("--cpl_lambda", type=float, default=1, help="cofficient for negative score, adopted from cpl")
    parser.add_argument("--ref_coef", type=float, default=1, help="cofficient for reference model's regret")
    parser.add_argument("--alpha", type=float, default=0.1, help="cofficient for alpha term")
    parser.add_argument("--sft_loss_coef", type=float, default=1, help="cofficient for sft loss")
    
    # Eval option
    parser.add_argument("--disable_eval_loss", action="store_true", default=False, help="evaluate loss for eval dataset")
    parser.add_argument("--eval_acc", action="store_true", default=False, help="evaluate accuracy for eval dataset")
    
    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    # parser.add_argument("--train_data_path", type=str, default=None)
    # parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--chosen_logprob_key", type=str, default="chosen_logprob_with_token")
    parser.add_argument("--rejected_logprob_key", type=str, default="rejected_logprob_with_token")
    parser.add_argument("--dummy_value", type=int, default=100, help="dummy value for padding logprob label")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--answer_trigger", type=str, default="The answer is:")
    parser.add_argument("--use_fast_dataset", action="store_true", default=False)
    
    parser.add_argument(
        "--aRePOy_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    
    # Generation configs
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--generation_max_len", type=int, default=768)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--generation_log_path", type=str, default="./generation_logs/RePO_%s" % datetime.now().strftime("%m%dT%H:%M"))

    # wandb parameters
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_RePO")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="RePO",
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    args.save_path = args.save_path + "_" + datetime.now().strftime("%m%dT%H:%M")

    args.wandb_run_name = args.wandb_run_name + "_" + datetime.now().strftime("%m%dT%H:%M")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )
        args.input_template = args.input_template.encode().decode('unicode_escape')

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
