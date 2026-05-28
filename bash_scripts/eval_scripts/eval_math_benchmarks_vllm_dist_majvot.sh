  

set -x

# Set the number of GPUs to use (modify this based on your available GPUs)
# GPU detection with CUDA_VISIBLE_DEVICES support
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Using $NUM_GPUS GPUs from CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected and using all $NUM_GPUS GPUs"
fi
NCCL_DEBUG=INFO \
# Use torchrun for multi-GPU distributed processing
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29504 \
   -m openrlhf.cli.evaluation_benchmarks_vllm_dist_majvote \
   --pretrain Qwen/Qwen3-1.7B-Base \
   --generation_log_path ./evaluation/logs/qwen2.5-Math-7B-Instruct-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 1 \
   --num_generations 8 \
   --max_len 2048 \
   --top_p 0.95 \
   --top_k 20 \
   --temperature 0.7 \
   --repetition_penalty 1.05 \
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
   "$@"
   # Note: tp_size is set to 1 since we're using data parallelism instead of tensor parallelism
   # Each GPU will have its own VLLM instance processing a subset of the data
