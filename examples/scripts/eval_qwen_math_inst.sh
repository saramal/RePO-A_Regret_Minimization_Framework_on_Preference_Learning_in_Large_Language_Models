

  



set -x

read -r -d '' training_commands <<EOF
deepspeed --module openrlhf.cli.evaluation \
   --pretrain Qwen/Qwen2.5-Math-7B \
   --dataset "./RePO_datasets/stepDPO/RePO_train.jsonl, ./RePO_datasets/stepDPO/RePO_test.jsonl" \
   --generation_log_path ./generation_logs/evals/qwen2.5-7B-Instruct \
   --max_len 2048 \
   --input_key prompt \
   --output_key chosen \
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