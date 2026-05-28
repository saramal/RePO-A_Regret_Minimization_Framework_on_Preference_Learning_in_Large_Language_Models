

  



set -x

read -r -d '' training_commands <<EOF
deepspeed --module openrlhf.cli.evaluation_benchmarks \
   --pretrain checkpoint/Qwen2.5-Math-7B-dpo_0625T20:11 \
   --generation_log_path ./evaluation/logs/qwen2.5-7B-dpo \
   --dataset ./evaluation/eval_data/ \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \


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