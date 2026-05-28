#!/bin/bash 

#SBATCH --job-name=dpo_logp_sweep
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

####
## Replace base model to SFT_trained model!!!
####

# model_name, lora_rank, micro_train_batch_size
experiments=(
    "checkpoint/sft_Qwen/Qwen3-8B-Base_lora_128_metamathqa_0920T03:38_merged 128 1 Qwen3-8B-logprob_pair"
    "checkpoint/sft_Qwen/Qwen3-4B-Base_lora_0_metamathqa_0921T04:45 128 2 Qwen3-4B-logprob_pair"
    "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_0_metamathqa_0924T05:22 128 2 Qwen3-1.7B-logprob_pair"
    
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    lora_rank=${experiment_array[1]}
    micro_train_batch_size=${experiment_array[2]}
    model_name_short=${experiment_array[3]}
    echo "########################################################"
    echo "Running experiment: DPO - ${model_name} with lora rank ${lora_rank}"
    echo "########################################################"

    bash   ./bash_scripts/train_scripts/train_dpo_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/dpo_${model_name_short}_lora_${lora_rank}_metamathqa \
        --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/logprob_score/RePO_train.jsonl \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate 1e-7 \
        --lora_rank $lora_rank \
        --lora_dropout 0.1 \
        --use_wandb True \
        --input_template None \
        --wandb_run_name "dpo_${model_name_short}_lora_${lora_rank}_metamathqa" \
        --save_merged

done


######################################





