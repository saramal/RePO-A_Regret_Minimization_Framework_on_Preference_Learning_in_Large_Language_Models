#!/bin/bash


  



set -x
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset HuggingFaceH4/ultrachat_200k \
   --input_key messages \
   --aRePOy_chat_template \
   --train_split train_sft \
   --system_prompt "You are a helpful assistant." \
   --multiturn \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen3-4B-Base \
   --save_path ./checkpoint/sft_Qwen/Qwen3-4B-Base-sft-ultrafeedback-multiturn-withsystemprompt \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -2 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --lora_rank 64 \
   --lora_alpha 64 \
   --lora_dropout 0.05 \
   --use_wandb \
   --wandb_run_name sft_qwen3.4B_ultrafeedback_multiturn_withsystemprompt \
   --save_merged \
   "$@"


