





CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
   --pretrain Qwen/Qwen2.5-3B-Instruct \
   --generation_log_path ./evaluation/logs/qwen2.5-3B-Instruct-stepdpo-base-vllm \
   --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
   --tp_size 4 \
   --max_len 2048 \
   --input_key prompt \
   --answer_key answer_label \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type None

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
   --pretrain checkpoint/Qwen2.5-3B-Instruct-sft-stepdpo_0829T03:32_merged \
   --generation_log_path ./evaluation/logs/qwen2.5-3B-Instruct-sft-stepdpo-vllm \
   --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
   --tp_size 4 \
   --max_len 2048 \
   --input_key prompt \
   --answer_key answer_label \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type None

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
   --pretrain checkpoint/Qwen2.5-3B-Instruct-stepdpo-RePO_0829T04:15_merged \
   --generation_log_path ./evaluation/logs/qwen2.5-3B-Instruct-stepdpo-RePO-vllm \
   --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
   --tp_size 4 \
   --max_len 2048 \
   --input_key prompt \
   --answer_key answer_label \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type None


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
   --pretrain Qwen/Qwen2.5-Math-1.5B \
   --generation_log_path ./evaluation/logs/qwen2.5-math-1.5B-stepdpo-base-vllm \
   --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
   --tp_size 4 \
   --max_len 2048 \
   --input_key prompt \
   --answer_key answer_label \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type None


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
   --pretrain checkpoint/Qwen2.5-Math-1.5B-sft-stepdpo_0828T05:12_merged \
   --generation_log_path ./evaluation/logs/qwen2.5-math-1.5B-sft-stepdpo-vllm \
   --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
   --tp_size 4 \
   --max_len 2048 \
   --input_key prompt \
   --answer_key answer_label \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type None

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
   --pretrain checkpoint/Qwen2.5-Math-1.5B-stepdpo-RePO_0828T19:19_merged \
   --generation_log_path ./evaluation/logs/qwen2.5-math-1.5B-stepdpo-RePO-vllm \
   --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
   --tp_size 4 \
   --max_len 2048 \
   --input_key prompt \
   --answer_key answer_label \
   --data_id_key index \
   --micro_eval_batch_size 1 \
   --max_samples 500000 \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type None


######### temp
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_benchmarks_vllm \
   --pretrain   ./checkpoint/Qwen2.5-Math-7B-dpo_0625T20:11_merged \
   --generation_log_path ./evaluation/logs/qwen2.5-Math-7B-dpo_0625T20:11-vllm_benchmark \
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
   --prompt_type qwen-stepdpo

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
#    --pretrain Qwen/Qwen3-4B-Instruct-2507 \
#    --generation_log_path ./evaluation/logs/qwen3.4B-Instruct-stepdpo-base-vllm \
#    --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
#    --tp_size 4 \
#    --max_len 2048 \
#    --input_key prompt \
#    --answer_key answer_label \
#    --micro_eval_batch_size 1 \
#    --max_samples 500000 \
#    --zero_stage 2 \
#    --bf16 \
#    --flash_attn \
#    --input_template None \
#    --prompt_type None


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
#    --pretrain checkpoint/Qwen3-4B-Instruct-sft-stepdpo_0828T04:18_merged \
#    --generation_log_path ./evaluation/logs/qwen3.4B-sft-stepdpo-vllm \
#    --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
#    --tp_size 4 \
#    --max_len 2048 \
#    --input_key prompt \
#    --answer_key answer_label \
#    --micro_eval_batch_size 1 \
#    --max_samples 500000 \
#    --zero_stage 2 \
#    --bf16 \
#    --flash_attn \
#    --input_template None \
#    --prompt_type None


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m openrlhf.cli.evaluation_stepdpo_vllm \
#    --pretrain checkpoint/Qwen3-4B-stepdpo-RePO_0828T14:02_merged \
#    --generation_log_path ./evaluation/logs/qwen3.4B-stepdpo-RePO-vllm \
#    --dataset RePO_datasets/stepDPO/RePO_test.jsonl \
#    --tp_size 4 \
#    --max_len 2048 \
#    --input_key prompt \
#    --answer_key answer_label \
#    --micro_eval_batch_size 1 \
#    --max_samples 500000 \
#    --zero_stage 2 \
#    --bf16 \
#    --flash_attn \
#    --input_template None \
#    --prompt_type None
