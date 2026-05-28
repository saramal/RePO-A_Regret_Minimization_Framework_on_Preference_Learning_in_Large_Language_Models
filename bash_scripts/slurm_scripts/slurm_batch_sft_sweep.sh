#!/bin/bash 

#SBATCH --job-name=sft_sweep
#SBATCH --partition=gpu04
  
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu04

# set conda env
env_name="openrlhf"

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
experiments=(
    "Qwen/Qwen3-8B-Base 128 2"
    "Qwen/Qwen3-4B-Base 0 1"
    "Qwen/Qwen3-1.7B-Base 0 2"
    
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    lora_rank=${experiment_array[1]}
    micro_train_batch_size=${experiment_array[2]}

    echo "########################################################"
    echo "Running experiment: ${model_name} with lora rank ${lora_rank}"
    echo "########################################################"

    bash   ./bash_scripts/train_scripts/train_sft_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/sft_${model_name}_lora_${lora_rank}_metamathqa \
        --dataset "RePO_datasets/MetamathQA/sft_qwen2.5-Math-7B-Instruct_0918T1643/sft_merged.jsonl" \
        --input_key query \
        --output_key response \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate 5e-6 \
        --lora_rank $lora_rank \
        --lora_dropout 0.05 \
        --use_wandb True \
        --input_template None \
        --wandb_run_name "sft_${model_name}_lora_${lora_rank}_metamathqa" \
        --save_merged True

done


######################################





