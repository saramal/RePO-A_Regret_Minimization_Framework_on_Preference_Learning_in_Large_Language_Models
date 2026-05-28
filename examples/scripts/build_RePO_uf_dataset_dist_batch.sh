  



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
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29505 \
   -m openrlhf.cli.build_RePO_dataset_uf_dist_batch \
   --max_len 2048 \
   --dataset RePO_datasets/Ultrafeedback/biniarized/binary_pair_templated.jsonl \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-1.5B-Instruct \
   --model_id Qwen/Qwen2.5-1.5B-Instruct \
   --save_name Qwen2.5-1.5B-Instruct \
   --save_path RePO_datasets/Ultrafeedback/biniarized \
   --batch_size 2 \







torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29505 \
   -m openrlhf.cli.build_RePO_dataset_uf_dist_batch \
   --max_len 2048 \
   --dataset RePO_datasets/Ultrafeedback/biniarized/RePO_train_Qwen2.5-1.5B-Instruct.jsonl \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 500000 \
   --pretrain Qwen/Qwen3-4B-Instruct-2507 \
   --model_id Qwen/Qwen3-4B-Instruct-2507 \
   --save_name Qwen3-4B-Instruct-2507 \
   --save_path RePO_datasets/Ultrafeedback/biniarized \
   --batch_size 2 \









   torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29505 \
   -m openrlhf.cli.build_RePO_dataset_uf_dist_batch \
   --max_len 2048 \
   --dataset RePO_datasets/Ultrafeedback/biniarized/RePO_train_Qwen3-4B-Instruct-2507.jsonl \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --model_id Qwen/Qwen2.5-7B-Instruct \
   --save_name Qwen2.5-7B-Instruct \
   --save_path RePO_datasets/Ultrafeedback/biniarized \
   --batch_size 2 \











   torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29505 \
   -m openrlhf.cli.build_RePO_dataset_uf_dist_batch \
   --max_len 2048 \
   --dataset RePO_datasets/Ultrafeedback/biniarized/RePO_train_Qwen2.5-7B-Instruct.jsonl \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-14B-Instruct \
   --model_id Qwen/Qwen2.5-14B-Instruct \
   --save_name Qwen2.5-14B-Instruct \
   --save_path RePO_datasets/Ultrafeedback/biniarized \
   --return_eval \
   --batch_size 1 \











   #--zero_stage 2 \
   # --bf16 \
   # --flash_attn \


    # --input_template "<|user|>:\\n{}\\nPlease reason step by step, and put your final answer with 'The answer is: '.\\n<|assistant|>:\n"

    # --wandb [WANDB_TOKENS]
    # --packing_samples
    # --load_checkpoint \
# time=`date +%y-%m-%d-%H:%M:%S`
# mkdir RePO_exp_logs/${time}
# if [[ ${1} != "slurm" ]]; then
#     CUDA_VISIBLE_DEVICES=0 bash -c "$training_commands" \
#     # 1> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".log) \
#     # 2> >(tee "./RePO_exp_logs"/${time}/"qwen_sft".err >&2) | \
#     # tee -a "./RePO_exp_logs/${time}/qwen_sft_full".log
# fi


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