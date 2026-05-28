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
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/logs/qwen2.5-Math-7B-Instruct-vllm_qwenboxedtemplate}"
RESULT_TXT_PATH="${RESULT_TXT_PATH:-evaluation_results_txt/math_results_pass_1.txt}"
mkdir -p "$(dirname "$RESULT_TXT_PATH")"

torchrun --nproc_per_node="$NUM_GPUS" --nnodes=1 --node_rank=0 \
  --master_addr="${MASTER_ADDR:-localhost}" --master_port="$MASTER_PORT" \
  -m openrlhf.cli.evaluation_benchmarks_vllm_dist \
  --pretrain "$MODEL" \
  --generation_log_path "$OUTPUT_DIR" \
  --dataset "${DATASET:-evaluation/eval_data_basic/}" \
  --tp_size "${TP_SIZE:-1}" \
  --max_len "${MAX_LEN:-2048}" \
  --top_p "${TOP_P:-1.0}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.0}" \
  --input_key "${INPUT_KEY:-question}" \
  --answer_key "${ANSWER_KEY:-answer}" \
  --data_id_key "${DATA_ID_KEY:-index}" \
  --micro_eval_batch_size "${MICRO_EVAL_BATCH_SIZE:-1}" \
  --max_samples "${MAX_SAMPLES:-500000}" \
  --zero_stage "${ZERO_STAGE:-2}" \
  --bf16 \
  --flash_attn \
  --input_template "${INPUT_TEMPLATE:-None}" \
  --prompt_type "${PROMPT_TYPE:-qwen2-boxed-cot}" \
  --result_txt_path "$RESULT_TXT_PATH" \
  "$@"
