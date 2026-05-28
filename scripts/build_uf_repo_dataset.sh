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
MASTER_PORT="${MASTER_PORT:-29505}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_ID="${MODEL_ID:-$MODEL}"
SAVE_NAME="${SAVE_NAME:-Qwen3-4B-Instruct-2507}"
PAIR_DATASET="${PAIR_DATASET:-RePO_datasets/Ultrafeedback/biniarized/binary_pair_templated.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-RePO_datasets/Ultrafeedback/biniarized}"

EXTRA_FLAGS=()
if [[ "${RETURN_EVAL:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--return_eval)
fi

torchrun --nproc_per_node="$NUM_GPUS" --nnodes=1 --node_rank=0 \
  --master_addr="${MASTER_ADDR:-localhost}" --master_port="$MASTER_PORT" \
  -m openrlhf.cli.build_RePO_dataset_uf_dist_batch \
  --max_len "${MAX_LEN:-2048}" \
  --dataset "$PAIR_DATASET" \
  --prompt_key "${PROMPT_KEY:-prompt}" \
  --chosen_key "${CHOSEN_KEY:-chosen}" \
  --rejected_key "${REJECTED_KEY:-rejected}" \
  --max_samples "${MAX_SAMPLES:-500000}" \
  --pretrain "$MODEL" \
  --model_id "$MODEL_ID" \
  --save_name "$SAVE_NAME" \
  --save_path "$OUTPUT_DIR" \
  --batch_size "${BATCH_SIZE:-2}" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
