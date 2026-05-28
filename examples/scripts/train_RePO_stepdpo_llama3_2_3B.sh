  



set -x

read -r -d '' training_commands <<EOF
deepspeed --module openrlhf.cli.train_RePO \
   --save_path ./checkpoint/Llama3.2-3B-stepdpo-RePO \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -2 \
   --train_batch_size 32 \
   --micro_train_batch_size 2 \
   --pretrain checkpoint/Llama3.2-3B-sft-stepdpo_0828T12:17_merged \
   --bf16 \
   --max_epochs 10 \
   --max_len 2048 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --dataset "./RePO_datasets/stepDPO/RePO_train.jsonl, ./RePO_datasets/stepDPO/RePO_test.jsonl" \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --lora_rank 128 \
   --lora_dropout 0.1 \
   --sft_loss \
   --use_wandb \
   --wandb_run_name RePO_llama3.2-3B_stepdpo \
   --cpl_lambda 1 \
   --save_merged \



   
EOF
    # --pretrain ./checkpoint/Qwen2.5-Math-1.5B-sft_merged \
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




# set -x

# read -r -d '' training_commands <<EOF
# deepspeed --module openrlhf.cli.train_RePO \
#    --save_path ./checkpoint/Qwen2.5-Math-1.5B-RePO \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --train_batch_size 256 \
#    --micro_train_batch_size 4 \
#    --pretrain ./checkpoint/Qwen2.5-Math-1.5B-sft_merged \
#    --bf16 \
#    --max_epochs 1 \
#    --max_len 2048 \
#    --zero_stage 3 \
#    --learning_rate 5e-7 \
#    --dataset ./RePO_datasets/RePO_debug.jsonl \
#    --chosen_key chosen \
#    --rejected_key rejected \
#    --flash_attn \
#    --load_checkpoint \
#    --gradient_checkpointing \
#    --lora_rank 128 \
#    --lora_dropout 0.1 \
#    --use_wandb True 

   
# EOF
#     # --input_template None
#     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
#     # --ipo [for IPO]
#     # --label_smoothing 0.1 [for cDPO]
#     # --ref_offload
#     # --packing_samples
#     # --nll_loss_coef (Regularization with NLL loss)


# if [[ ${1} != "slurm" ]]; then
#     CUDA_VISIBLE_DEVICES=3 bash -c "$training_commands"
# fi
