#!/bin/bash


  



set -x
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset "RePO_datasets/MetamathQA/sft_qwen2.5-Math-7B-Instruct_0918T1643/sft_merged.jsonl" \
   --input_key query \
   --output_key response \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen3-4B-Instruct-2507 \
   --save_path ./checkpoint/Qwen3-4B-Instruct-sft-metamathqa \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -2 \
   --zero_stage 2 \
   --max_epochs 5 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --lora_rank 128 \
   --lora_dropout 0.05 \
   --use_wandb \
   --input_template None \
   --wandb_run_name sft_qwen3.4B_metamathqa \
   --save_merged True \
   "$@"


