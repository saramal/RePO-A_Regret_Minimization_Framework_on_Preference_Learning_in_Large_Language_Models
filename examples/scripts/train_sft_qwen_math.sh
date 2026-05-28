# set -x

# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_sft \
#    --max_len 2048 \
#    --dataset ElonTusk2001/rstar_sft \
#    --input_key query \
#    --output_key response \
#    --train_batch_size 256 \
#    --micro_train_batch_size 4 \
#    --max_samples 500000 \
#    --pretrain Qwen/Qwen2.5-Math-1.5B \
#    --save_path ./checkpoint/Qwen2.5-Math-1.5B-sft \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --zero_stage 2 \
#    --max_epochs 2 \
#    --bf16 \
#    --flash_attn \
#    --learning_rate 5e-6 \
#    --gradient_checkpointing \
#    --lora_rank 128 \
#    --lora_dropout 0.1 \
#    --use_wandb True \
#    --input_template "<|user|>:\\n{}\\n<|assistant|>: Let\'s think step by step and solve the problem with code."

# EOF
#     # --wandb [WANDB_TOKENS]
#     # --packing_samples
#     # --load_checkpoint \

# if [[ ${1} != "slurm" ]]; then
#     eval CUDA_VISIBLE_DEVICES=3 deepspeed --module $training_commands
# fi


  



set -x

read -r -d '' training_commands <<EOF
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset ElonTusk2001/rstar_sft \
   --input_key query \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-Math-7B \
   --save_path ./checkpoint/Qwen2.5-Math-7B-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 100 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --lora_rank 128 \
   --lora_dropout 0.05 \
   --use_wandb True \
   --input_template "<|user|>:\\n{}\\n<|assistant|>: Let\'s think step by step and solve the problem with code."

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