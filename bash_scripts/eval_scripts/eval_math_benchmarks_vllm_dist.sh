  

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

# Use torchrun for multi-GPU distributed processing
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
   -m openrlhf.cli.evaluation_benchmarks_vllm_dist \
   --pretrain Qwen/Qwen2.5-Math-7B-Instruct \
   --generation_log_path ./evaluation/logs/qwen2.5-Math-7B-Instruct-vllm_qwenboxedtemplate \
   --dataset ./evaluation/eval_data_basic/ \
   --tp_size 1 \
   --max_len 2048 \
   --top_p 1.0 \
   --temperature 0.0 \
   --repetition_penalty 1.0 \
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
