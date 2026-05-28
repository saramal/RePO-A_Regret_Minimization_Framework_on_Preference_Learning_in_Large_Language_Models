# !/bin/bash 

# SBATCH --job-name=RePO_dpo_ultrafeedback
# SBATCH --partition=gpu01
 
# SBATCH --output=output_%j.log
# SBATCH --error=error_%j.log
# SBATCH --gres=gpu:4
# SBATCH --nodelist=gpu01



# ### RePO


# # set conda env
# env_name="openrlhf"

# # set variables
# JOBNAME=${SLURM_JOB_NAME}
# JOBID=${SLURM_JOB_ID}
# TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
# LOG_DIR="log_cluster/${JOBNAME}_${JOBID}__${TIMESTAMP}"

# # create log directory
# mkdir -p "$LOG_DIR"

# # set log file name
# OUTPUT_LOG="output_${JOBNAME}_${JOBID}.log"
# ERROR_LOG="error_${JOBNAME}_${JOBID}.log"
# COMBINED_LOG="full_${JOBNAME}_${JOBID}.log"

# # copy err&output to "combined_log"
# exec > >(tee -a "$COMBINED_LOG") 2> >(tee -a "$COMBINED_LOG" >&2)

# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# # move log files
# mv output_${JOBID}.log "$LOG_DIR/$OUTPUT_LOG"
# mv error_${JOBID}.log "$LOG_DIR/$ERROR_LOG"
# mv "$COMBINED_LOG" "$LOG_DIR/"

# echo "Logs saved in $LOG_DIR"
# ####################################
# # codes to execute
# export MASTER_PORT=29503

# # model_name, lora_rank, micro_train_batch_size
# experiments=(
 
#     "Qwen/Qwen3-4B-Base 32 1 Qwen3-4B-base_sft0.0 1 5e-7"
    
# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     lora_rank=${experiment_array[1]}
#     micro_train_batch_size=${experiment_array[2]}
#     model_name_short=${experiment_array[3]}
#     alpha=${experiment_array[4]}
#     learning_rate=${experiment_array[5]}
#     echo "########################################################"
#     echo "Running experiment: RePO - ${model_name} with lora rank ${lora_rank}"
#     echo "########################################################"

#     bash   ./bash_scripts/uf_scripts/train_RePO_uf_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_ultrafeedback_custom \
#         --ckpt_path ./checkpoint/ckpt/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_ultrafeedback_custom \
#         --dataset "RePO_datasets/Ultrafeedback/biniarized/RePO_train_biniarized_cleaned.jsonl, RePO_datasets/Ultrafeedback/biniarized/RePO_test_Qwen2.5-14B-Instruct.jsonl" \
#         --prompt_key prompt \
#         --chosen_key chosen \
#         --rejected_key rejected \
#         --alpha $alpha \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 128 \
#         --eval_steps -1 \
#         --log_ratio_step 0.2 \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate $learning_rate \
#         --lora_rank $lora_rank \
#         --lora_alpha 64 \
#         --lora_dropout 0.1 \
#         --sft_loss_coef 0.0 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_ultrafeedback_custom" \
#         --save_merged

# done











### DPO

env_name="openrlhf"

# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(

    "Qwen/Qwen3-8B-Base 64 1 Qwen3-8B-Base_uf_custom_2epochs"

    
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
        --save_path ./checkpoint/dpo_${model_name_short}_lora_${lora_rank}_ultrafeedback_custom \
        --dataset "RePO_datasets/Ultrafeedback/biniarized/RePO_train_biniarized_cleaned.jsonl, RePO_datasets/Ultrafeedback/biniarized/RePO_test_Qwen2.5-14B-Instruct.jsonl" \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -1 \
        --log_ratio_step 0.2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate 5e-7 \
        --lora_rank $lora_rank \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "dpo_${model_name_short}_lora_${lora_rank}_ultrafeedback_custom" \
        --save_merged

done











### RePO_DET_cpl0.5


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
export MASTER_PORT=29503

# model_name, lora_rank, micro_train_batch_size
experiments=(

    "Qwen/Qwen3-8B-Base 64 1 Qwen3-8B-Base_sft0.0_cpl0.5_2epochs 1 5e-7"

     
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    lora_rank=${experiment_array[1]}
    micro_train_batch_size=${experiment_array[2]}
    model_name_short=${experiment_array[3]}
    alpha=${experiment_array[4]}
    learning_rate=${experiment_array[5]}
    echo "########################################################"
    echo "Running experiment: RePO_DET - ${model_name} with lora rank ${lora_rank}"
    echo "########################################################"

    bash   ./bash_scripts/uf_scripts/train_RePO_det_uf_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/RePO_Qwen/RePO_det_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_ultrafeedback_custom \
        --ckpt_path ./checkpoint/ckpt/RePO_Qwen/RePO_det_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_ultrafeedback_custom \
        --dataset "RePO_datasets/Ultrafeedback/biniarized/RePO_train_biniarized_cleaned.jsonl, RePO_datasets/Ultrafeedback/biniarized/RePO_test_Qwen2.5-14B-Instruct.jsonl" \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --alpha $alpha \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -1 \
        --return_eval \
        --log_ratio_step 0.2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate $learning_rate \
        --lora_rank $lora_rank \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --sft_loss_coef 0.0 \
        --cpl_lambda 0.5 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "RePO_det_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_ultrafeedback_custom" \
        --save_merged

done










### RePO cpl0.5


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
export MASTER_PORT=29503

# model_name, lora_rank, micro_train_batch_size
experiments=(

    
    "Qwen/Qwen3-8B-Base 64 1 Qwen3-8B-Base_sft0.0_cpl0.5_2epochs 1 5e-7"

    
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    lora_rank=${experiment_array[1]}
    micro_train_batch_size=${experiment_array[2]}
    model_name_short=${experiment_array[3]}
    alpha=${experiment_array[4]}
    learning_rate=${experiment_array[5]}
    echo "########################################################"
    echo "Running experiment: RePO - ${model_name} with lora rank ${lora_rank}"
    echo "########################################################"

    bash   ./bash_scripts/uf_scripts/train_RePO_uf_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_ultrafeedback_custom \
        --ckpt_path ./checkpoint/ckpt/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_ultrafeedback_custom \
        --dataset "RePO_datasets/Ultrafeedback/biniarized/RePO_train_biniarized_cleaned.jsonl, RePO_datasets/Ultrafeedback/biniarized/RePO_test_Qwen2.5-14B-Instruct.jsonl" \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --alpha $alpha \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -1 \
        --log_ratio_step 0.2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate $learning_rate \
        --lora_rank $lora_rank \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --sft_loss_coef 0.0 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_ultrafeedback_custom" \
        --save_merged \
        --cpl_lambda 0.5
done




