#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MASTER_PORT="${MASTER_PORT:-29507}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
DATASET="${DATASET:-RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch/RePO_train.jsonl}"
SAVE_PATH="${SAVE_PATH:-checkpoint/Qwen3-4B-metamathqa-RePO_det}"

EXTRA_FLAGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--use_wandb)
fi
if [[ "${SAVE_MERGED:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--save_merged)
fi
if [[ "${RETURN_EVAL:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--return_eval)
fi

deepspeed --master_port "$MASTER_PORT" --module openrlhf.cli.train_RePO_det \
  --save_path "$SAVE_PATH" \
  --save_steps "${SAVE_STEPS:--1}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --eval_steps "${EVAL_STEPS:--2}" \
  --disable_ds_ckpt \
  --log_ratio_step "${LOG_RATIO_STEP:-0.2}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-128}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE:-2}" \
  --pretrain "$MODEL" \
  --bf16 \
  --max_epochs "${MAX_EPOCHS:-2}" \
  --max_len "${MAX_LEN:-2048}" \
  --zero_stage "${ZERO_STAGE:-2}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --dataset "$DATASET" \
  --dataset_probs "${DATASET_PROBS:-1.0}" \
  --prompt_key "${PROMPT_KEY:-prompt}" \
  --chosen_key "${CHOSEN_KEY:-chosen}" \
  --rejected_key "${REJECTED_KEY:-rejected}" \
  --use_fast_dataset \
  --flash_attn \
  --gradient_checkpointing \
  --gradient_checkpointing_use_reentrant \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout "${LORA_DROPOUT:-0.1}" \
  --sft_loss \
  --cpl_lambda "${CPL_LAMBDA:-0.5}" \
  --alpha "${ALPHA:-1}" \
  --sft_loss_coef "${SFT_LOSS_COEF:-0.0}" \
  --wandb_run_name "${WANDB_RUN_NAME:-RePO_det_public}" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
