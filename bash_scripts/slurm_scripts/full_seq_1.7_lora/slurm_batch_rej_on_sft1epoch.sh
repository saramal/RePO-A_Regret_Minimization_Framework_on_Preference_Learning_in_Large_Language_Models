# #!/bin/bash 

# #SBATCH --job-name=RePO_base_sweep
# #SBATCH --partition=gpu02
#   
# #SBATCH --output=output_%j.log
# #SBATCH --error=error_%j.log
# #SBATCH --gres=gpu:2
# #SBATCH --nodelist=gpu02




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

# # model_name, lora_rank, micro_train_batch_size
# experiments=(
#     # "checkpoint/sft_Qwen/Qwen3-8B-Base_lora_128_metamathqa_0920T03:38_merged 128 1 Qwen3-8B-base_pair 1 1e-7"
#     # "checkpoint/sft_Qwen/Qwen3-4B-Base_lora_0_metamathqa_0921T04:45 128 1 Qwen3-4B-base_pair 1 1e-7"
#     # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_0_metamathqa_0924T05:22 128 2 Qwen3-1.7B-base_pair 1 1e-7"
    
#     "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_merged 32 4 Qwen3-1.7B-Base_sft_1epoch_metamathqa_filtered"
#     # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_0.5_merged 128 1 Qwen3-1.7B-half_base_pair 1 5e-7"
    
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
#     echo "Running experiment: Rej - ${model_name} with lora rank ${lora_rank}"
#     echo "########################################################"

 
#     bash   ./bash_scripts/train_scripts/train_rejection_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/rejection_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --dataset "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl" \
#         --ckpt_path ./checkpoint/ckpt/rejection_Qwen/rejection_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --input_key prompt \
#         --output_key chosen \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 32 \
#         --eval_steps -2 \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate 5e-6 \
#         --lora_rank $lora_rank \
#         --lora_alpha 64 \
#         --lora_dropout 0.05 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "rejection_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
#         --save_merged

# done



# #####################################

# MODEL_PATH=$(cat last_model_path_.txt)


# ######################################
# ######################################

# env_name="vllm"
# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "$MODEL_PATH rejection_Qwen3-1.7B-Base_sft_1epoch_metamathqa_filtered_lora_32_metamathqa_filtered"

# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     model_name_for_log=${experiment_array[1]}


#     echo "########################################################"
#     echo "Running experiment: Evaluate ${model_name}"
#     echo "########################################################"

#     bash   ./bash_scripts/eval_scripts/eval_math_benchmarks_vllm_dist.sh \
#         --pretrain $model_name \
#         --generation_log_path ./evaluation/metamath_logs/${model_name_for_log}_vllm_qwenboxedtemplate \
#         --dataset ./evaluation/eval_data_basic/ \
#         --tp_size 1 \
#         --max_len 2048 \
#         --top_p 1.0 \
#         --temperature 0.0 \
#         --repetition_penalty 1.05 \
#         --input_key question \
#         --answer_key answer \
#         --data_id_key index \
#         --micro_eval_batch_size 1 \
#         --max_samples 500000 \
#         --zero_stage 2 \
#         --bf16 \
#         --flash_attn \
#         --input_template None \
#         --prompt_type qwen2-boxed-cot

# done


# ######################################












######################################
### RePO - on sft 0.0 - 2epoch
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
    # "checkpoint/sft_Qwen/Qwen3-8B-Base_lora_128_metamathqa_0920T03:38_merged 128 1 Qwen3-8B-base_pair 1 1e-7"
    # "checkpoint/sft_Qwen/Qwen3-4B-Base_lora_0_metamathqa_0921T04:45 128 1 Qwen3-4B-base_pair 1 1e-7"
    # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_0_metamathqa_0924T05:22 128 2 Qwen3-1.7B-base_pair 1 1e-7"
    
    "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_merged 32 2 Qwen3-1.7B-Base_on_sft_sft0.0_2epoch_metamathqa_filtered 1 5e-7"
    # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_0.5_merged 128 1 Qwen3-1.7B-half_base_pair 1 5e-7"
    
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

    bash   ./bash_scripts/train_scripts/train_RePO_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_metamathqa \
        --ckpt_path ./checkpoint/ckpt/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_metamathqa \
        --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --alpha $alpha \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate $learning_rate \
        --lora_rank $lora_rank \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --sft_loss_coef 0.0 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_metamathqa" \
        --save_merged

done


#####################################

MODEL_PATH=$(cat last_model_path.txt)


######################################
######################################

env_name="vllm"
# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(
    "$MODEL_PATH RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_4epoch_lora_32_metamathqa_filtered"

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
















######################################
### RePO - on sft 0.5 - 2epoch
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
    # "checkpoint/sft_Qwen/Qwen3-8B-Base_lora_128_metamathqa_0920T03:38_merged 128 1 Qwen3-8B-base_pair 1 1e-7"
    # "checkpoint/sft_Qwen/Qwen3-4B-Base_lora_0_metamathqa_0921T04:45 128 1 Qwen3-4B-base_pair 1 1e-7"
    # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_0_metamathqa_0924T05:22 128 2 Qwen3-1.7B-base_pair 1 1e-7"
    
    "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_merged 32 2 Qwen3-1.7B-Base_on_sft_sft0.5_2epoch_metamathqa_filtered 1 5e-7"
    # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_0.5_merged 128 1 Qwen3-1.7B-half_base_pair 1 5e-7"
    
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

    bash   ./bash_scripts/train_scripts/train_RePO_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_metamathqa \
        --ckpt_path ./checkpoint/ckpt/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_metamathqa \
        --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --alpha $alpha \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -2 \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate $learning_rate \
        --lora_rank $lora_rank \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --sft_loss_coef 0.5 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_metamathqa" \
        --save_merged

done


#####################################

MODEL_PATH=$(cat last_model_path.txt)


######################################
######################################

env_name="vllm"
# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(
    "$MODEL_PATH RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_4epoch_lora_32_metamathqa_filtered"

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









