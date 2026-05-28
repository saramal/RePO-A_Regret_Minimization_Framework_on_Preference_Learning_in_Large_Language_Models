  

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
   -m openrlhf.cli.data_generation_vllm_uf_dist \
   --pretrain Qwen/Qwen3-4B-Instruct-2507 \
   --generation_log_path RePO_datasets/Ultrafeedback/generation_logs/qwen3-4b-instruct-2507 \
   --dataset HuggingFaceH4/ultrafeedback_binarized \
   --tp_size 1 \
   --max_samples 500000 \
   --max_len 2048 \
   --n_samples_per_prompt 1 \
   --temperature 0.7 \
   --top_p 0.95 \
   --top_k 40 \
   --repetition_penalty 1.05 \
   --max_new_tokens 1024 \
   --prompt_max_len 1024 \
   --input_key prompt \
   --data_id_key prompt_id \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --input_template None \
   --prompt_type qwen-instruct-basic-prompt \
   "$@"
   # Note: tp_size is set to 1 since we're using data parallelism instead of tensor parallelism
   # Each GPU will have its own VLLM instance processing a subset of the data

