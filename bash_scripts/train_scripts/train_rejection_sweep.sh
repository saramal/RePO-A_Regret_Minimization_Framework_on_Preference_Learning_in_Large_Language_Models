#!/bin/bash


  



set -x
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl" \
   --input_key question \
   --output_key chosen \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen3-4B-Instruct-2507 \
   --save_path ./checkpoint/Qwen3-4B-Instruct-rejection-metamathqa \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -2 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --lora_rank 128 \
   --lora_dropout 0.05 \
   --use_wandb \
   --wandb_project openrlhf_train_rej \
   --input_template None \
   --save_merged \
   "$@"


