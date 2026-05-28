





CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain Qwen/Qwen2.5-Math-7B-Instruct \
   --generation_log_path ./evaluation/logs/qwen2.5-Math-7B-Instruct-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen2-boxed-cot

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain Qwen/Qwen2.5-Math-1.5B \
   --generation_log_path ./evaluation/logs/qwen2.5-Math-1.5B-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen2-boxed-cot

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain Qwen/Qwen2.5-Math-7B \
   --generation_log_path ./evaluation/logs/qwen2.5-Math-7B-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen2-boxed-cot

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain Qwen/Qwen2-Math-7B \
   --generation_log_path ./evaluation/logs/qwen2-Math-7B-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen2-boxed-cot


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain Qwen/Qwen3-4B-Instruct-2507 \
   --generation_log_path ./evaluation/logs/qwen3-4B-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen2-boxed-cot \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain microsoft/Phi-3-mini-4k-instruct \
   --generation_log_path ./evaluation/logs/qwen3-4B-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type phi3 \



CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain meta-llama/Llama-3.2-3B-Instruct \
   --generation_log_path ./evaluation/logs/llama3.2-3B-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 4 \
   --max_len 2048 \
   --input_key question \
   --answer_key answer \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type llama3 \





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