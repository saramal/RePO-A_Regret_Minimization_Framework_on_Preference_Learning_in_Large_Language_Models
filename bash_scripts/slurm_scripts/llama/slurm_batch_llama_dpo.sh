



# # ######################################
# # ### RePO


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
    
#     "Qwen/Qwen3-4B-Base 32 1 Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5 1 5e-7"
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
#     echo "Running experiment: RePO - ${model_name_short} with lora rank ${lora_rank}"
#     echo "########################################################"

#     bash   ./bash_scripts/train_scripts/train_RePO_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_metamathqa \
#         --ckpt_path ./checkpoint/ckpt/RePO_Qwen/RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_lr_${learning_rate}_metamathqa \
#         --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
#         --prompt_key prompt \
#         --chosen_key chosen \
#         --rejected_key rejected \
#         --alpha $alpha \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 128 \
#         --eval_steps -1 \
#         --return_eval \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate $learning_rate \
#         --lora_rank $lora_rank \
#         --lora_alpha 64 \
#         --lora_dropout 0.1 \
#         --sft_loss_coef 0.0 \
#         --cpl_lambda 0.5 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "RePO_${model_name_short}_lora_${lora_rank}_alpha_${alpha}_metamathqa" \
#         --save_merged

# done


# ####################################

# MODEL_PATH=$(cat last_model_path.txt)


# ######################################
# ######################################

# env_name="vllm"
# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "$MODEL_PATH RePO_Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_metamathqa_filtered"

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
### DPO

env_name="openrlhf"

# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(
    # "meta-llama/Llama-3.1-8B-Instruct 64 1 llama3.1-8B-Instruct_metamathqa_filtered"
    "meta-llama/Llama-3.1-8B 64 1 llama3.1-8B-Base_metamathqa_filtered_2e-5"

    
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    lora_rank=${experiment_array[1]}
    micro_train_batch_size=${experiment_array[2]}
    model_name_short=${experiment_array[3]}
    echo "########################################################"
    echo "Running experiment: DPO - ${model_name_short} with lora rank ${lora_rank}"
    echo "########################################################"

    bash   ./bash_scripts/train_scripts/train_dpo_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/dpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
        --ckpt_path ./checkpoint/ckpt/dpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
        --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -1 \
        --log_ratio_step 0.2 \
        --disable_ds_ckpt \
        --save_hf_ckpt \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate 2e-5 \
        --lora_rank $lora_rank \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "dpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
        --save_merged

done

MODEL_PATH=$(cat last_model_path.txt)

######################################

######################################

env_name="vllm"
# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(
    "$MODEL_PATH dpo_llama3.1-8B-Base_metamathqa_filtered_2e-5_lora_64_metamathqa_filtered"

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
        --prompt_type llama3

done


######################################




######################################
### DPO

env_name="openrlhf"

# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(
    "meta-llama/Llama-3.1-8B-Instruct 64 1 llama3.1-8B-Instruct_metamathqa_filtered_5e-6"
    # "meta-llama/Llama-3.1-8B 64 2 llama3.1-8B-Base_metamathqa_filtered_2e-5"

    
)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    lora_rank=${experiment_array[1]}
    micro_train_batch_size=${experiment_array[2]}
    model_name_short=${experiment_array[3]}
    echo "########################################################"
    echo "Running experiment: DPO - ${model_name_short} with lora rank ${lora_rank}"
    echo "########################################################"

    bash   ./bash_scripts/train_scripts/train_dpo_sweep.sh \
        --pretrain $model_name \
        --save_path ./checkpoint/dpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
        --ckpt_path ./checkpoint/ckpt/dpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
        --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
        --prompt_key prompt \
        --chosen_key chosen \
        --rejected_key rejected \
        --micro_train_batch_size $micro_train_batch_size \
        --train_batch_size 128 \
        --eval_steps -1 \
        --log_ratio_step 0.2 \
        --disable_ds_ckpt \
        --save_hf_ckpt \
        --max_epochs 2 \
        --zero_stage 2 \
        --learning_rate 5e-6 \
        --lora_rank $lora_rank \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --use_wandb \
        --input_template None \
        --wandb_run_name "dpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
        --save_merged

done

MODEL_PATH=$(cat last_model_path.txt)

######################################

######################################

env_name="vllm"
# Conda activate
CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
ENV_PATH=" ./conda_envs/${env_name}"
source $CONDA_BIN_PATH/activate $ENV_PATH

experiments=(
    "$MODEL_PATH dpo_llama3.1-8B-Instruct_metamathqa_filtered_5e-6_lora_64_metamathqa_filtered"

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
        --prompt_type llama3

done


######################################


# ######################################
# ### RPO

# env_name="openrlhf"

# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "Qwen/Qwen3-4B-Base 32 2 Qwen3-4B-non_sft_base_pair"

    
# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     lora_rank=${experiment_array[1]}
#     micro_train_batch_size=${experiment_array[2]}
#     model_name_short=${experiment_array[3]}
#     echo "########################################################"
#     echo "Running experiment: RPO - ${model_name_short} with lora rank ${lora_rank}"
#     echo "########################################################"

#     bash   ./bash_scripts/train_scripts/train_rpo_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/rpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
#         --prompt_key prompt \
#         --chosen_key chosen \
#         --rejected_key rejected \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 128 \
#         --eval_steps -1 \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate 5e-7 \
#         --lora_rank $lora_rank \
#         --lora_alpha 128 \
#         --lora_dropout 0.1 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "rpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
#         --save_merged

# done

# MODEL_PATH=$(cat last_model_path.txt)

# ######################################

# ######################################

# env_name="vllm"
# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "$MODEL_PATH rpo_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered"

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



# ######################################
# ### IPO

# env_name="openrlhf"

# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "Qwen/Qwen3-4B-Base 32 2 Qwen3-4B-non_sft_base_pair"

    
# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     lora_rank=${experiment_array[1]}
#     micro_train_batch_size=${experiment_array[2]}
#     model_name_short=${experiment_array[3]}
#     echo "########################################################"
#     echo "Running experiment: IPO - ${model_name_short} with lora rank ${lora_rank}"
#     echo "########################################################"

#     bash   ./bash_scripts/train_scripts/train_ipo_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/ipo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
#         --prompt_key prompt \
#         --chosen_key chosen \
#         --rejected_key rejected \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 128 \
#         --eval_steps -1 \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate 5e-7 \
#         --lora_rank $lora_rank \
#         --lora_alpha 128 \
#         --lora_dropout 0.1 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "ipo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
#         --save_merged

# done

# MODEL_PATH=$(cat last_model_path.txt)

# ######################################

# ######################################

# env_name="vllm"
# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "$MODEL_PATH ipo_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered"

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




# ######################################
# ### TDPO

# env_name="openrlhf"

# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "Qwen/Qwen3-4B-Base 32 2 Qwen3-4B-non_sft_base_pair"

    
# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     lora_rank=${experiment_array[1]}
#     micro_train_batch_size=${experiment_array[2]}
#     model_name_short=${experiment_array[3]}
#     echo "########################################################"
#     echo "Running experiment: TDPO - ${model_name_short} with lora rank ${lora_rank}"
#     echo "########################################################"

#     bash   ./bash_scripts/train_scripts/train_tdpo_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/tdpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
#         --prompt_key prompt \
#         --chosen_key chosen \
#         --rejected_key rejected \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 128 \
#         --eval_steps -1 \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate 5e-7 \
#         --lora_rank $lora_rank \
#         --lora_alpha 128 \
#         --lora_dropout 0.1 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "tdpo_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
#         --save_merged

# done

# MODEL_PATH=$(cat last_model_path.txt)

# ######################################

# ######################################

# env_name="vllm"
# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "$MODEL_PATH tdpo_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered"

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




# ######################################
# ### KTO

# env_name="openrlhf"

# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "Qwen/Qwen3-4B-Base 32 2 Qwen3-4B-non_sft_base_pair_2epoch"

    
# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     lora_rank=${experiment_array[1]}
#     micro_train_batch_size=${experiment_array[2]}
#     model_name_short=${experiment_array[3]}
#     echo "########################################################"
#     echo "Running experiment: KTO - ${model_name_short} with lora rank ${lora_rank}"
#     echo "########################################################"

#     bash   ./bash_scripts/train_scripts/train_kto_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/kto_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
#         --prompt_key prompt \
#         --chosen_key chosen \
#         --rejected_key rejected \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 64 \
#         --eval_steps -1 \
#         --max_epochs 2 \
#         --zero_stage 2 \
#         --learning_rate 5e-7 \
#         --lora_rank $lora_rank \
#         --lora_alpha 128 \
#         --lora_dropout 0.1 \
#         --use_wandb \
#         --input_template None \
#         --wandb_run_name "kto_${model_name_short}_lora_${lora_rank}_metamathqa_filtered" \
#         --save_merged

# done

# MODEL_PATH=$(cat last_model_path.txt)

# ######################################

# ######################################

# env_name="vllm"
# # Conda activate
# CONDA_BIN_PATH=/opt/ohpc/pub/apps/anaconda3/bin
# ENV_PATH=" ./conda_envs/${env_name}"
# source $CONDA_BIN_PATH/activate $ENV_PATH

# experiments=(
#     "$MODEL_PATH kto_Qwen3-4B-non_sft_base_pair_lora_32_2epoch_metamathqa_filtered"

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


######################################

# ##########
# # Rej
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
    
#     "Qwen/Qwen3-4B-Base 32 1 Qwen3-4B-non_sft_base_pair_sft0.0"
#     # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_0.5_merged 128 1 Qwen3-1.7B-half_base_pair 1 5e-7"
    
# )
# for experiment in "${experiments[@]}"; do
#     experiment_array=($experiment)
#     model_name=${experiment_array[0]}
#     lora_rank=${experiment_array[1]}
#     micro_train_batch_size=${experiment_array[2]}
#     model_name_short=${experiment_array[3]}
#     echo "########################################################"
#     echo "Running experiment: Rej - ${model_name_short} with lora rank ${lora_rank}"
#     echo "########################################################"

 
#     bash   ./bash_scripts/train_scripts/train_rejection_sweep.sh \
#         --pretrain $model_name \
#         --save_path ./checkpoint/rejection_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --dataset RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/basic_pair/RePO_train.jsonl \
#         --ckpt_path ./checkpoint/ckpt/rejection_Qwen/rejection_${model_name_short}_lora_${lora_rank}_metamathqa_filtered \
#         --input_key prompt \
#         --output_key chosen \
#         --micro_train_batch_size $micro_train_batch_size \
#         --train_batch_size 128 \
#         --eval_steps -1 \
#         --return_eval \
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
#     "$MODEL_PATH rejection_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered"

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










