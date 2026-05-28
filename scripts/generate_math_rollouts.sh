#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
DATASET="${DATASET:-RePO_datasets/metamathqa_processed.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-RePO_datasets/MetamathQA/test_qwen2.5-Math-7B-Instruct}"
TP_SIZE="${TP_SIZE:-4}"

python -m openrlhf.cli.data_generation_vllm \
  --pretrain "$MODEL" \
  --generation_log_path "$OUTPUT_DIR" \
  --dataset "$DATASET" \
  --tp_size "$TP_SIZE" \
  --max_samples "${MAX_SAMPLES:-500000}" \
  --max_len "${MAX_LEN:-2048}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT:-8}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.95}" \
  --top_k "${TOP_K:-40}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.05}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}" \
  --prompt_max_len "${PROMPT_MAX_LEN:-1024}" \
  --input_key "${INPUT_KEY:-question}" \
  --answer_key "${ANSWER_KEY:-answer}" \
  --data_id_key "${DATA_ID_KEY:-index}" \
  --micro_eval_batch_size "${MICRO_EVAL_BATCH_SIZE:-1}" \
  --zero_stage "${ZERO_STAGE:-2}" \
  --bf16 \
  --flash_attn \
  --input_template "${INPUT_TEMPLATE:-None}" \
  --prompt_type "${PROMPT_TYPE:-qwen2-boxed-cot}" \
  "$@"
