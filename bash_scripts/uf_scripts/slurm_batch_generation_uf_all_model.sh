#!/bin/bash 

#SBATCH --job-name=uf_generation_sweep
#SBATCH --partition=gpu02
  
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu02

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


experiments=(
    "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct-2507"
    "Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3-30b-a3b-instruct-2507"
    "Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b-instruct"
    "Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct"
    "Qwen/Qwen2.5-14B-Instruct qwen2.5-14b-instruct"
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    model_name_for_log=${experiment_array[1]}


    echo "########################################################"
    echo "Running experiment: Evaluate ${model_name}"
    echo "########################################################"

    bash   ./bash_scripts/uf_scripts/data_generation_vllm_uf_dist.sh \
        --pretrain $model_name \
        --generation_log_path RePO_datasets/Ultrafeedback/generation_logs/${model_name_for_log} \
        --dataset HuggingFaceH4/ultrafeedback_binarized \
        --tp_size 1 \
        --max_samples 5000000 \
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

done


######################################

