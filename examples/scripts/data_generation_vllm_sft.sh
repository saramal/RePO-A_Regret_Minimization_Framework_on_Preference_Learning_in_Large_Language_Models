





CUDA_VISIBLE_DEVICES=0 python -m openrlhf.cli.data_generation_vllm_sft \
   --pretrain Qwen/Qwen2.5-Math-7B-Instruct \
   --generation_log_path ./RePO_datasets/MetamathQA/sft_qwen2.5-Math-7B-Instruct \
   --dataset RePO_datasets/metamathqa_processed.jsonl \
   --tp_size 1 \
   --max_samples 500000 \
   --max_len 2048 \
   --RePO_metadata RePO_datasets/MetamathQA/test_qwen2.5-Math-7B-Instruct_0824T0632/metadata.jsonl \
   --n_samples_per_prompt 1 \
   --temperature 0 \
   --top_p 0.95 \
   --top_k 20 \
   --repetition_penalty 1 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen2-boxed-cot







# few_shot_example: "cot examples..."


# set -x

# read -r -d '' training_commands <<EOF
# deepspeed --module openrlhf.cli.evaluation_benchmarks_vllm \
#    --pretrain checkpoint/Qwen2.5-Math-7B-RePO_kl-fixed_0813T02:08 \
#    --generation_log_path ./evaluation/logs/qwen2.5-7B-RePO \
#    --dataset ./evaluation/eval_data/ \
#    --tp_size 4 \
#    --max_len 2048 \
#    --input_key question \
#    --answer_key answer \
#    --data_id_key index \
#    --micro_eval_batch_size 1 \
#    --max_samples 500000 \
#    --zero_stage 2 \
#    --bf16 \
#    --flash_attn \
#    --input_template None \


# EOF

#     # --wandb [WANDB_TOKENS]
#     # --packing_samples
#     # --load_checkpoint \
# # time=`date +%y-%m-%d-%H:%M:%S`
# # mkdir RePO_exp_logs/${time}
# if [[ ${1} != "slurm" ]]; then
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash -c "$training_commands" \
#     # 1> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".log) \
#     # 2> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".err >&2) | \
#     # tee -a "./RePO_exp_logs/${time}/qwen_sft_full".log
# fi