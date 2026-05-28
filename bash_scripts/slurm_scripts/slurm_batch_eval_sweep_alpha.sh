#!/bin/bash 

#SBATCH --job-name=RePO_eval_sweep
#SBATCH --partition=gpu04
  
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu04

# set conda env
env_name="vllm"

# set variables
JOBNAME=${SLURM_JOB_NAME}
JOBID=${SLURM_JOB_ID}
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="log_cluster/${JOBNAME}_${JOBID}__${TIMESTAMP}"

# create log directory
mkdir -p "$LOG_DIR"

# set log file name
OUTPUT_LOG="output_${JOBNAME}_${JOBID}.log"
ERROR_LOG="error_${JOBNAME}_${JOBID}.log"
COMBINED_LOG="full_${JOBNAME}_${JOBID}.log"

# copy err&output to "combined_log"
exec > >(tee -a "$COMBINED_LOG") 2> >(tee -a "$COMBINED_LOG" >&2)

# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

# move log files
mv output_${JOBID}.log "$LOG_DIR/$OUTPUT_LOG"
mv error_${JOBID}.log "$LOG_DIR/$ERROR_LOG"
mv "$COMBINED_LOG" "$LOG_DIR/"

echo "Logs saved in $LOG_DIR"
####################################
# codes to execute

# model_name, lora_rank, micro_train_batch_size
# experiments=(
#     "Qwen/Qwen3-8B-Base Qwen3-8B-Base"
#     "Qwen/Qwen3-4B-Base Qwen3-4B-Base"
#     "Qwen/Qwen3-1.7B-Base Qwen3-1.7B-Base"
#     "checkpoint/sft_Qwen/Qwen3-8B-Base_lora_128_metamathqa_0920T03:38_merged sft_Qwen3-8B-Base_lora_128_metamathqa"
#     "checkpoint/sft_Qwen/Qwen3-4B-Base_lora_0_metamathqa_0921T04:45 sft_Qwen3-4B-Base_lora_0_metamathqa"
#     "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_0_metamathqa_0924T05:22 sft_Qwen3-1.7B-Base_lora_0_metamathqa"
#     "checkpoint/RePO_Qwen/RePO_Qwen3-8B-Base_lora_64_metamathqa_1001T17:34_merged RePO_Qwen3-8B-Base_lora_64_metamathqa"
#     "checkpoint/RePO_Qwen/RePO_Qwen3-4B-Base_lora_32_metamathqa_1005T08:54_merged RePO_Qwen3-4B-Base_lora_32_metamathqa"
#     "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-Base_lora_32_metamathqa_1007T21:20_merged RePO_Qwen3-1.7B-Base_lora_32_metamathqa"
#     "checkpoint/dpo_Qwen3-8B-Base_lora_128_metamathqa_0926T18:55_merged dpo_Qwen3-8B-Base_lora_128_metamathqa"
#     "checkpoint/dpo_Qwen3-4B-Base_lora_64_metamathqa_0930T12:21_merged dpo_Qwen3-4B-Base_lora_64_metamathqa"
#     "checkpoint/dpo_Qwen3-1.7B-Base_lora_64_metamathqa_1002T21:15_merged dpo_Qwen3-1.7B-Base_lora_64_metamathqa"
# )
experiments=(
    "checkpoint/RePO_Qwen/RePO_Qwen3-8B-Base_lora_128_alpha_3_lr_5e-7_metamathqa_1027T14:38_merged RePO_Qwen3-8B-Base_lora_128_alpha_3_lr_5e-7_metamathqa"
    "checkpoint/RePO_Qwen/RePO_Qwen3-8B-Base_lora_128_alpha_2_lr_5e-7_metamathqa_1031T09:07_merged RePO_Qwen3-8B-Base_lora_128_alpha_2_lr_5e-7_metamathqa"
    "checkpoint/RePO_Qwen/RePO_Qwen3-8B-Base_lora_128_alpha_0.5_lr_5e-7_metamathqa_1104T03:39_merged RePO_Qwen3-8B-Base_lora_128_alpha_0.1_lr_1e-8_metamathqa"
    
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    model_name_for_log=${experiment_array[1]}


    echo "########################################################"
    echo "Running experiment: Evaluate ${model_name}"
    echo "########################################################"

    bash   ./bash_scripts/eval_scripts/eval_math_benchmarks_vllm_dist.sh \
        --pretrain $model_name \
        --generation_log_path ./evaluation/metamath_logs/${model_name_for_log}_vllm_qwenboxedtemplate \
        --dataset ./evaluation/eval_data_basic/ \
        --tp_size 1 \
        --max_len 2048 \
        --top_p 1.0 \
        --temperature 0.0 \
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
        --prompt_type qwen2-boxed-cot

done


######################################





