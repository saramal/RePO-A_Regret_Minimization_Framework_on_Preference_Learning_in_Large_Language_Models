#!/bin/bash

  



set -x

read -r -d '' training_commands <<EOF
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset "./RePO_datasets/stepDPO/RePO_train.jsonl, ./RePO_datasets/stepDPO/RePO_test.jsonl" \
   --input_key prompt \
   --output_key chosen \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-Math-1.5B \
   --save_path ./checkpoint/Qwen2.5-Math-1.5B-sft-stepdpo \
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
   --use_wandb True \
   --input_template None \
   --wandb_run_name sft_qwen2.5-math-1.5B_stepdpo \


EOF

    # --wandb [WANDB_TOKENS]
    # --packing_samples
    # --load_checkpoint \
# time=`date +%y-%m-%d-%H:%M:%S`
# mkdir RePO_exp_logs/${time}
if [[ ${1} != "slurm" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash -c "$training_commands" \
    # 1> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".log) \
    # 2> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".err >&2) | \
    # tee -a "./RePO_exp_logs/${time}/qwen_sft_full".log
fi