#!/bin/bash


  

export MASTER_PORT=29503

set -x
deepspeed --master_port $MASTER_PORT --module openrlhf.cli.train_RePO \
   --save_path ./checkpoint/Qwen3-4B-ultrafeedback-RePO \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --disable_ds_ckpt \
   --save_hf_ckpt \
   --log_ratio_step 0.2 \
   --return_eval \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain Qwen/Qwen3-4B-Base \
   --bf16 \
   --max_epochs 2 \
   --max_len 2048 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --dataset "RePO_datasets/Ultrafeedback/biniarized/RePO_train_biniarized_cleaned.jsonl, RePO_datasets/Ultrafeedback/biniarized/RePO_test_Qwen2.5-14B-Instruct.jsonl" \
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
   --wandb_run_name RePO_qwen3.4B_ultrafeedback \
   --cpl_lambda 1 \
   --alpha 1 \
   --sft_loss_coef 0.5 \
   "$@"

