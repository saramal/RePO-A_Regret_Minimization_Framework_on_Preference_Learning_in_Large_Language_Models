  



set -x

read -r -d '' training_commands <<EOF
deepspeed --module openrlhf.cli.build_RePO_dataset \
   --max_len 2048 \
   --dataset xinlai/Math-Step-DPO-10K \
   --prompt_key "prompt, initial_reason_steps" \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 500000 \
   --pretrain_A xinlai/Qwen2-7B-SFT \
   --save_path ./RePO_datasets/stepDPO_step \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template "<|user|>:\\n{}\\nPlease reason step by step, and put your final answer with 'The answer is: '.\\n<|assistant|>:\n"

EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples
    # --load_checkpoint \
# time=`date +%y-%m-%d-%H:%M:%S`
# mkdir RePO_exp_logs/${time}
if [[ ${1} != "slurm" ]]; then
    CUDA_VISIBLE_DEVICES=0 bash -c "$training_commands" \
    # 1> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".log) \
    # 2> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".err >&2) | \
    # tee -a "./RePO_exp_logs/${time}/qwen_sft_full".log
fi


# old

# export CUDA_VISIBLE_DEVICES=3

# set -x

# read -r -d '' training_commands <<EOF
# openrlhf.cli.build_RePO_dataset \
#    --max_len 2048 \
#    --dataset ElonTusk2001/rstar_ppm \
#    --prompt_key prompt \
#    --chosen_key pos \
#    --rejected_key neg \
#    --max_samples 500000 \
#    --pretrain_A ./checkpoint/Qwen2.5-Math-1.5B-sft_merged \
#    --save_path ./RePO_datasets \
#    --zero_stage 2 \
#    --bf16 \
#    --flash_attn
# EOF
#     # --wandb [WANDB_TOKENS]
#     # --packing_samples
#     # --load_checkpoint \

# if [[ ${1} != "slurm" ]]; then
#     deepspeed --module $training_commands
# fi