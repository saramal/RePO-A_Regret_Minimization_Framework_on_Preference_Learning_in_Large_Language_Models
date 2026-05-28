#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

detect_num_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | sed '/^[[:space:]]*$/d' | wc -l
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus | wc -l
  else
    echo 1
  fi
}

NUM_GPUS="${NUM_GPUS:-$(detect_num_gpus)}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_NAME_FOR_LOG="${MODEL_NAME_FOR_LOG:-qwen3-4b-instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-RePO_datasets/Ultrafeedback/generation_logs/${MODEL_NAME_FOR_LOG}}"

torchrun --nproc_per_node="$NUM_GPUS" --nnodes=1 --node_rank=0 \
  --master_addr="${MASTER_ADDR:-localhost}" --master_port="$MASTER_PORT" \
  -m openrlhf.cli.data_generation_vllm_uf_dist \
  --pretrain "$MODEL" \
  --generation_log_path "$OUTPUT_DIR" \
  --dataset "${DATASET:-HuggingFaceH4/ultrafeedback_binarized}" \
  --tp_size "${TP_SIZE:-1}" \
  --max_samples "${MAX_SAMPLES:-500000}" \
  --max_len "${MAX_LEN:-2048}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT:-1}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.95}" \
  --top_k "${TOP_K:-40}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.05}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}" \
  --prompt_max_len "${PROMPT_MAX_LEN:-1024}" \
  --input_key "${INPUT_KEY:-prompt}" \
  --data_id_key "${DATA_ID_KEY:-prompt_id}" \
  --zero_stage "${ZERO_STAGE:-2}" \
  --bf16 \
  --flash_attn \
  --input_template "${INPUT_TEMPLATE:-None}" \
  --prompt_type "${PROMPT_TYPE:-qwen-instruct-basic-prompt}" \
  "$@"
