#!/bin/bash


  


export MASTER_PORT=29507
set -x
deepspeed --master_port $MASTER_PORT --module openrlhf.cli.train_RePO_det \
   --save_path ./checkpoint/Qwen3-4B-metamathqa-RePO \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -2 \
   --disable_ds_ckpt \
   --log_ratio_step 0.2 \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain Qwen/Qwen3-4B-Base \
   --bf16 \
   --max_epochs 2 \
   --max_len 2048 \
   --zero_stage 2 \
   --learning_rate 2e-5 \
   --dataset "./RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch/RePO_train_merged.jsonl, ./RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch/RePO_test_merged.jsonl" \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --use_fast_dataset \
   --flash_attn \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --lora_rank 32 \
   --lora_alpha 64 \
   --lora_dropout 0.1 \
   --sft_loss \
   --use_wandb \
   --cpl_lambda 0.5 \
   --alpha 1 \
   --sft_loss_coef 0.0 \
   "$@"


