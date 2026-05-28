#!/bin/bash


  


export MASTER_PORT=29509
set -x
deepspeed --master_port $MASTER_PORT --module openrlhf.cli.train_RePO_topk \
   --save_path ./checkpoint/Qwen3-1.7B-metamathqa-RePO-topk \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --return_eval \
   --disable_ds_ckpt \
   --log_ratio_step 0.2 \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --use_fast_dataset \
   --pretrain Qwen/Qwen3-1.7B-Base \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --dataset "./RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch_new/RePO_train.jsonl, ./RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch_new/RePO_test.jsonl" \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --lora_rank 64 \
   --lora_alpha 64 \
   --lora_dropout 0.1 \
   --sft_loss_coef 0.0 \
   --cpl_lambda 0.5 \
   --use_wandb \
   --wandb_run_name RePO_qwen3_1.7B_metamathqa_topk_10_sft0.0_cpl0.5 \
   --alpha 0.1 \
   --top_k 10 \
   "$@"


