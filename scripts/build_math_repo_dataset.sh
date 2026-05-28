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
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
PAIR_DATASET="${PAIR_DATASET:-RePO_datasets/MetamathQA/preference_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch}"

torchrun --nproc_per_node="$NUM_GPUS" --nnodes=1 --node_rank=0 \
  --master_addr="${MASTER_ADDR:-localhost}" --master_port="$MASTER_PORT" \
  -m openrlhf.cli.build_RePO_dataset_topk_dist_batch \
  --max_len "${MAX_LEN:-2048}" \
  --dataset "$PAIR_DATASET" \
  --prompt_key "${PROMPT_KEY:-question}" \
  --chosen_key "${CHOSEN_KEY:-chosen}" \
  --rejected_key "${REJECTED_KEY:-rejected}" \
  --max_samples "${MAX_SAMPLES:-500000}" \
  --pretrain_A "$ROLLOUT_MODEL" \
  --save_path "$OUTPUT_DIR" \
  --top_k "${TOP_K_LOGPROBS:-10}" \
  --batch_size "${BATCH_SIZE:-2}" \
  "$@"
