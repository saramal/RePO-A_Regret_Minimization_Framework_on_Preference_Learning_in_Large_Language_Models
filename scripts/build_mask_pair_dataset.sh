#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

INPUT="${INPUT:-RePO_datasets/MetamathQA/logp_scored_qwen2.5-Math-7B-Instruct/logprob_grouped.jsonl}"
OUTPUT="${OUTPUT:-RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/mask_pair_chosen_fixed/RePO_train.jsonl}"

python -m openrlhf.cli.build_mask_pair_dataset \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --basic-pairs \
  --add-mask-pairs \
  --fix-chosen \
  --seed "${SEED:-42}" \
  "$@"
