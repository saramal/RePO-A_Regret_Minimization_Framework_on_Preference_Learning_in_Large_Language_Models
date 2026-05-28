#!/bin/bash 

#SBATCH --job-name=RePO_eval_sweep
#SBATCH --partition=gpu03
  
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu03

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
# ####################################
# # codes to execute

# # model_name, lora_rank, micro_train_batch_size
# # experiments=(
# #     "Qwen/Qwen3-8B-Base Qwen3-8B-Base"
# #     "Qwen/Qwen3-4B-Base Qwen3-4B-Base"
# #     "Qwen/Qwen3-1.7B-Base Qwen3-1.7B-Base"
# #     "checkpoint/sft_Qwen/Qwen3-8B-Base_lora_128_metamathqa_0920T03:38_merged sft_Qwen3-8B-Base_lora_128_metamathqa"
# #     "checkpoint/sft_Qwen/Qwen3-4B-Base_lora_0_metamathqa_0921T04:45 sft_Qwen3-4B-Base_lora_0_metamathqa"
# #     "checkpoint/sft_Qwen/Qwen3-1.7B-Base_lora_0_metamathqa_0924T05:22 sft_Qwen3-1.7B-Base_lora_0_metamathqa"
# #     "checkpoint/RePO_Qwen/RePO_Qwen3-8B-Base_lora_64_metamathqa_1001T17:34_merged RePO_Qwen3-8B-Base_lora_64_metamathqa"
# #     "checkpoint/RePO_Qwen/RePO_Qwen3-4B-Base_lora_32_metamathqa_1005T08:54_merged RePO_Qwen3-4B-Base_lora_32_metamathqa"
# #     "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-Base_lora_32_metamathqa_1007T21:20_merged RePO_Qwen3-1.7B-Base_lora_32_metamathqa"
# #     "checkpoint/dpo_Qwen3-8B-Base_lora_128_metamathqa_0926T18:55_merged dpo_Qwen3-8B-Base_lora_128_metamathqa"
# #     "checkpoint/dpo_Qwen3-4B-Base_lora_64_metamathqa_0930T12:21_merged dpo_Qwen3-4B-Base_lora_64_metamathqa"
# #     "checkpoint/dpo_Qwen3-1.7B-Base_lora_64_metamathqa_1002T21:15_merged dpo_Qwen3-1.7B-Base_lora_64_metamathqa"
# # )
# experiments=(
#     # "checkpoint/RePO_Qwen/RePO_Qwen3-8B-logprob_pair_lora_128_alpha_1_lr_1e-7_metamathqa_1114T21:20_merged RePO_Qwen3-8B-logprob_lora_128_alpha_1_lr_e-7_metamathqa"
#     # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-logprob_pair_lora_128_alpha_1_lr_1e-7_metamathqa_1116T01:34_merged RePO_Qwen3-4B-logprob_lora_128_alpha_1_lr_1e-7_metamathqa"
#     # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-masked_pair_lora_128_alpha_1_lr_1e-7_metamathqa_1117T17:44_merged RePO_Qwen3-4B-masked_lora_128_alpha_1_lr_1e-7_metamathqa"
#     # "checkpoint/dpo_Qwen3-8B-logprob_pair_lora_128_metamathqa_1117T06:33_merged dpo_Qwen3-8B-logprob_lora_128_lr_1e-7_metamathqa"
#     # "checkpoint/dpo_Qwen3-4B-logprob_pair_lora_128_metamathqa_1118T06:57_merged dpo_Qwen3-4B-logprob_lora_128_lr_1e-7_metamathqa"
#     # "checkpoint/dpo_Qwen3-1.7B-Base_lora_64_metamathqa_1002T21:15_merged"
#     # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1125T13:10_1epoch_ckpt_merged RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_1epoch_ckpt_lora_32_alpha_1_lr_5e-7_metamathqa"
#     # "checkpoint/rejection_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_1125T22:04_1epoch_ckpt_merged rejection_Qwen3-1.7B-non_sft_base_pair_1epoch_ckpt_lora_32_metamathqa"


#     # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_1_lora_32_topk_1_metamathqa_1213T14:58_merged RePO_topk_Qwen3-1.7B-Base_topk_1"
#     # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_2_lora_32_topk_2_metamathqa_1215T05:51_merged RePO_topk_Qwen3-1.7B-Base_topk_2"
#     # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_4_lora_32_topk_4_metamathqa_1215T16:48_merged RePO_topk_Qwen3-1.7B-Base_topk_4"
#     # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_8_lora_32_topk_8_metamathqa_1215T12:35_merged RePO_topk_Qwen3-1.7B-Base_topk_8"
#     # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_10_lora_32_topk_10_metamathqa_1215T23:20_merged RePO_topk_Qwen3-1.7B-Base_topk_10"

#     # "Qwen/Qwen3-1.7B-Base Qwen3-1.7B-Base"
#     # "Qwen/Qwen3-4B-Base Qwen3-4B-Base"
#     # "Qwen/Qwen3-8B-Base Qwen3-8B-Base"

#     # "meta-llama/Llama-3.1-8B Llama-3.1-8B-Base"
#     # "meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct"



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
#         --result_txt_path evaluation_results_txt/math_results_pass_1.txt \
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



experiments=(
    # "checkpoint/RePO_Qwen/RePO_Qwen3-8B-logprob_pair_lora_128_alpha_1_lr_1e-7_metamathqa_1114T21:20_merged RePO_Qwen3-8B-logprob_lora_128_alpha_1_lr_e-7_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-logprob_pair_lora_128_alpha_1_lr_1e-7_metamathqa_1116T01:34_merged RePO_Qwen3-4B-logprob_lora_128_alpha_1_lr_1e-7_metamathqa"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-4B-masked_pair_lora_128_alpha_1_lr_1e-7_metamathqa_1117T17:44_merged RePO_Qwen3-4B-masked_lora_128_alpha_1_lr_1e-7_metamathqa"
    # "checkpoint/dpo_Qwen3-8B-logprob_pair_lora_128_metamathqa_1117T06:33_merged dpo_Qwen3-8B-logprob_lora_128_lr_1e-7_metamathqa"
    # "checkpoint/dpo_Qwen3-4B-logprob_pair_lora_128_metamathqa_1118T06:57_merged dpo_Qwen3-4B-logprob_lora_128_lr_1e-7_metamathqa"
    # "checkpoint/dpo_Qwen3-1.7B-Base_lora_64_metamathqa_1002T21:15_merged"
    # "checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_lora_32_alpha_1_lr_5e-7_metamathqa_1125T13:10_1epoch_ckpt_merged RePO_Qwen3-1.7B-non_sft_base_pair_sft0.0_1epoch_ckpt_lora_32_alpha_1_lr_5e-7_metamathqa"
    # "checkpoint/rejection_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_1125T22:04_1epoch_ckpt_merged rejection_Qwen3-1.7B-non_sft_base_pair_1epoch_ckpt_lora_32_metamathqa"


    # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_1_lora_32_topk_1_metamathqa_1213T14:58_merged RePO_topk_Qwen3-1.7B-Base_topk_1"
    # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_2_lora_32_topk_2_metamathqa_1215T05:51_merged RePO_topk_Qwen3-1.7B-Base_topk_2"
    # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_4_lora_32_topk_4_metamathqa_1215T16:48_merged RePO_topk_Qwen3-1.7B-Base_topk_4"
    # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_8_lora_32_topk_8_metamathqa_1215T12:35_merged RePO_topk_Qwen3-1.7B-Base_topk_8"
    # "checkpoint/RePO_Qwen/RePO_topk_Qwen3-1.7B-non_sft_base_pair_sft0.0_cpl0.5_topk_10_lora_32_topk_10_metamathqa_1215T23:20_merged RePO_topk_Qwen3-1.7B-Base_topk_10"

    # "Qwen/Qwen3-1.7B-Base Qwen3-1.7B-Base"
    # "Qwen/Qwen3-4B-Base Qwen3-4B-Base"
    # "Qwen/Qwen3-8B-Base Qwen3-8B-Base"

    # "meta-llama/Llama-3.1-8B Llama-3.1-8B-Base"
    # "meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct"
    # "checkpoint/RePO_llama/RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_lora_64_alpha_1_lr_5e-7_metamathqa_0117T09:56_merged RePO_det_llama3.1-8B-Instruct_metamathqa_filtered"

    # "checkpoint/ckpt/RePO_llama/RePO_det_llama3.1-8B-Base_metamathqa_filtered_lora_64_alpha_1_lr_2e-5_metamathqa/global_step505_hf_merged RePO_det_llama3.1-8B-Base_metamathqa_filtered_2e-5_2epochs_0.5"
    # "checkpoint/ckpt/RePO_llama/RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_lora_64_alpha_1_lr_2e-5_metamathqa/global_step505_hf_merged RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_2e-5_2epochs_0.5"
    # "checkpoint/ckpt/RePO_llama/RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_lora_64_alpha_1_lr_2e-6_metamathqa/global_step505_hf_merged RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_2e-6_2epochs_0.5"
    # "checkpoint/ckpt/RePO_Qwen/RePO_Qwen3-1.7B-Base_2epochs_2e-5_lora_32_alpha_1_lr_2e-5_metamathqa/global_step505_hf_merged RePO_Qwen3-1.7B-Base_2epochs_2e-5_lora_32_alpha_1_lr_2e-5_metamathqa"


    # # Qwen/Qwen3-1.7B-Base  
    # # ./checkpoint/dpo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_1125T02:51_merged
    # ./checkpoint/rpo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_0106T17:31_merged
    # ./checkpoint/ipo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_0106T23:23_merged
    # ./checkpoint/tdpo_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_0106T18:54_merged
    # ./checkpoint/kto_Qwen3-1.7B-non_sft_base_pair_lora_32_metamathqa_filtered_0116T02:58_merged
    # ./checkpoint/RePO_Qwen/RePO_Qwen3-1.7B-Base_2epochs_5e-6_lora_32_alpha_1_lr_5e-6_metamathqa_0127T16:15_merged
    # ./checkpoint/RePO_Qwen/RePO_det_Qwen3-1.7B-Base_2epochs_5e-6_lora_32_alpha_1_lr_5e-6_metamathqa_0128T04:21_merged



    # # Qwen/Qwen3-4B-Base
    # # ./checkpoint/dpo_Qwen3-4B-non_sft_base_pair_sft0.0_lora_32_metamathqa_filtered_1203T00:43_merged
    # ./checkpoint/rpo_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered_0107T11:59_merged
    # ./checkpoint/ipo_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered_0107T23:30_merged
    # ./checkpoint/kto_Qwen3-4B-non_sft_base_pair_2epoch_lora_32_metamathqa_filtered_0116T04:30_merged
    # ./checkpoint/tdpo_Qwen3-4B-non_sft_base_pair_lora_32_metamathqa_filtered_0108T12:26_merged
    # ./checkpoint/RePO_Qwen/RePO_Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1205T00:10_merged
    # ./checkpoint/RePO_Qwen/RePO_det_Qwen3-4B-non_sft_base_pair_sft0.0_cpl0.5_lora_32_alpha_1_lr_5e-7_metamathqa_1213T04:08_merged


    # "meta-llama/Llama-3.1-8B llama3.1-8B-Base"
    # "checkpoint/dpo_llama3.1-8B-Base_metamathqa_filtered_2e-5_lora_64_metamathqa_filtered_0126T04:49_merged llama3.1-8B-Base_metamathqa_filtered_2e-5"
    # "checkpoint/RePO_llama/RePO_det_llama3.1-8B-Base_metamathqa_filtered_lora_64_alpha_1_lr_2e-5_metamathqa_0123T15:55_merged llama3.1-8B-Base_metamathqa_filtered_2e-5"

    # "meta-llama/Llama-3.1-8B-Instruct llama3.1-8B-Instruct"
    # "checkpoint/dpo_llama3.1-8B-Instruct_metamathqa_filtered_5e-6_lora_64_metamathqa_filtered_0126T21:03_merged llama3.1-8B-Instruct_metamathqa_filtered_5e-6"
    # "./checkpoint/RePO_llama/RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_lora_64_alpha_1_lr_5e-7_metamathqa_0117T09:56_merged RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_5e-"
    # "./checkpoint/RePO_llama/RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_5e-6_lora_64_alpha_1_lr_5e-6_metamathqa_0125T10:24_merged RePO_det_llama3.1-8B-Instruct_metamathqa_filtered_5e-6"

    "./checkpoint/dpo_Qwen3-1.7B-Base_2epochs_5e-6_lora_32_metamathqa_filtered_0127T22:32_merged dpo_Qwen3-1.7B-Base_2epochs_5e-6"
    "./checkpoint/rpo_Qwen3-1.7B-Base_2epochs_5e-6_lora_32_metamathqa_filtered_0129T01:08_merged rpo_Qwen3-1.7B-Base_2epochs_5e-6"
    "./checkpoint/tdpo_Qwen3-1.7B-Base_2epochs_5e-6_lora_32_metamathqa_filtered_0128T18:52_merged tdpo_Qwen3-1.7B-Base_2epochs_5e-6"
    "./checkpoint/ipo_Qwen3-1.7B-Base_2epochs_5e-6_lora_32_metamathqa_filtered_0129T06:56_merged ipo_Qwen3-1.7B-Base_2epochs_5e-6"

    "./checkpoint/dpo_Qwen3-4B-Base_2epochs_5e-6_lora_64_metamathqa_filtered_0128T06:00_merged dpo_Qwen3-4B-Base_2epochs_5e-6"


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
        --result_txt_path evaluation_results_txt/math_results_pass_1_minerva.txt \
        --dataset ./evaluation/eval_data_minerva/ \
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

