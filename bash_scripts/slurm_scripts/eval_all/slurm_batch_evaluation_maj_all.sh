#!/bin/bash 

#SBATCH --job-name=RePO_evalmaj_sweep
#SBATCH --partition=gpu01
  
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu01

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
    # "Qwen/Qwen3-1.7B-Base Qwen3-1.7B-Base"
    # "checkpoint/dpo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_1125T02:51_merged dpo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1124T19:55_merged RePO_Qwen3-1.7B-non_sft_base_pair_sft0.5_lora_32_alpha_1_lr_5e-7_metamathqa"
    
    # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_merged sft_Qwen3-1.7B-Base_lora_128_metamathqa_filtered"
    # "checkpoint/dpo_Qwen3-1.7B-base_pair_filtered_lora_128_metamathqa_filtered_1120T05:55_merged dpo_Qwen3-1.7B-1epochsftbase_pair_filtered_lora_128_metamathqa_filtered"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-base_pair_nosft_lora_32_alpha_1_lr_5e-7_metamathqa_1124T20:36_merged RePO_Qwen3-1.7B-1epochsft_base_pair_nosft_lora_32_alpha_1_lr_5e-7_metamathqa"

    # "checkpoint/dpo_Qwen3-1.7B-Base_lora_64_metamathqa_1002T21:15_merged dpo_Qwen3-1.7B-Base_lora_64_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-Base_lora_32_metamathqa_1007T21:20_merged RePO_Qwen3-1.7B-Base_lora_32_metamathqa"
    
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1125T13:10_merged RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1125T13:10_1epoch_ckpt_merged RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_1epoch_ckpt_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/rejection_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_1125T22:04_merged rej_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa"
    # "checkpoint/rejection_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_1125T22:04_1epoch_ckpt_merged rej_Qwen3-1.7B-non_sft_base_pair_1epoch_ckpt_lora_32_metamathqa"
    
    # "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_128_metamathqa_filtered_1120T02:06_merged sft_Qwen3-1.7B-Base_lora_128_metamathqa_filtered"
    # "checkpoint/rejection_Qwen3-1.7B-Base_sft_1epoch_metamathqa_filtered_lora_32_metamathqa_filtered_1129T00:26_merged rej_Qwen3-1.7B-Base_sft_1epoch_metamathqa_filtered_lora_32_metamathqa_filtered"
    # "checkpoint/dpo_Qwen3-1.7B-base_pair_filtered_lora_128_metamathqa_filtered_1120T05:55_merged dpo_Qwen3-1.7B-base_pair_filtered_lora_128_metamathqa_filtered"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-Base_on_sft_sft0.0_2epoch_metamathqa_filtered_lora_32_alpha_1_lr_5e-7_metamathqa_1130T01:16_merged RePO_Qwen3-1.7B-Base_on_sft_sft0.0_2epoch_metamathqa_filtered_lora_32_alpha_1_lr_5e-7_metamathqa"

    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_logprob_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1127T17:40_merged RePO_Qwen3-1.7B-non_sft_logprob_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa"    
    # "checkpoint/dpo_Qwen3-1.7B-non_sft_logprob_pair_lora_32_metamathqa_filtered_1128T03:55_merged dpo_Qwen3-1.7B-non_sft_logprob_pair_lora_32_metamathqa_filtered"

    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_masked_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1201T02:53_merged RePO_Qwen3-1.7B-non_sft_masked_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/dpo_Qwen3-1.7B-non_sft_masked_pair_lora_32_metamathqa_filtered_1130T00:02_merged dpo_Qwen3-1.7B-non_sft_masked_pair_lora_32_metamathqa_filtered"

    # "Qwen/Qwen3-4B-Base Qwen3-4B-Base"
    # "checkpoint/rejection_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_metamathqa_filtered_1203T11:44_merged rej_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_metamathqa_filtered"
    # "checkpoint/dpo_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_metamathqa_filtered_1203T00:43_merged dpo_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_metamathqa_filtered"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1203T17:37_merged RePO_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-non_sft_base_pair_sft0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1204T10:35_merged RePO_Qwen3-4B-non_sft_base_pair_sft0.5_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1205T00:10_merged RePO_Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1205T12:22_merged RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa"

    "Qwen/Qwen3-1.7B-Base Qwen3-1.7B-Base"
    
    
    "Qwen/Qwen3-4B-Base Qwen3-4B-Base"


)
for experiment in "${experiments[@]}"; do
    experiment_array=($experiment)
    model_name=${experiment_array[0]}
    model_name_for_log=${experiment_array[1]}


    echo "########################################################"
    echo "Running experiment: Evaluate ${model_name}"
    echo "########################################################"

    bash   ./bash_scripts/eval_scripts/eval_math_benchmarks_vllm_dist_majvot.sh \
        --pretrain $model_name \
        --generation_log_path ./evaluation/metamath_logs/${model_name_for_log}_vllm_qwenboxedtemplate \
        --dataset ./evaluation/eval_data_basic/ \
        --tp_size 1 \
        --num_generations 8 \
        --max_len 2048 \
        --top_p 0.95 \
        --top_k 20 \
        --temperature 0.7 \
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





