# RePO: Regret-based Preference Optimization

This repository contains the implementation for the ICML 2026 spotlight paper
**A Regret Minimization Framework on Preference Learning in Large Language Models**.

- Paper: [OpenReview PDF](https://openreview.net/pdf?id=genVnYBAV7)
- Poster/project page: [RePO-regret-based-rlhf](https://saramal.github.io/RePO-regret-based-rlhf/)


RePO studies preference learning through regret minimization. The key implementation detail is that a RePO preference example is not only a `(prompt, chosen, rejected)` triple: it also stores token-level log probabilities from the rollout/data-generation policy. `RePO_det` is the deterministic/approximated variant used when that rollout logprob term is treated as a constant proxy, referred to in our experiments as assuming the generation logprob is 1.

This codebase is built on and adapted from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), with additional RePO datasets, losses, trainers, and experiment scripts.

## Contents

- `openrlhf/cli/train_RePO.py`, `openrlhf/trainer/RePO_trainer.py`: RePO training.
- `openrlhf/cli/train_RePO_det.py`, `openrlhf/trainer/RePO_trainer_det.py`: deterministic RePO training.
- `openrlhf/datasets/RePO_dataset*.py`: RePO dataset loaders with token-level rollout logprob labels.
- `openrlhf/cli/build_RePO_dataset*_*.py`: scripts for attaching rollout-policy logprobs to preference data.
- `openrlhf/cli/build_mask_pair_dataset.py`: masked pair augmentation.
- `scripts/`: public entrypoints for data generation, training, and evaluation.

## Setup

This repository follows the OpenRLHF environment setup. The recommended path is to start from NVIDIA's PyTorch container and install OpenRLHF with vLLM support, then put this repository at the front of `PYTHONPATH` so the local RePO implementation overrides the upstream package code.

```bash
# Launch the docker container. Adjust the mount path for your machine.
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash

# Match the OpenRLHF recommendation for this base image.
sudo pip uninstall xgboost transformer_engine flash_attn pynvml -y

# Install OpenRLHF with vLLM acceleration.
pip install openrlhf[vllm]

# Alternatively, use the latest vLLM-supported OpenRLHF extras.
# pip install openrlhf[vllm_latest]

# Install RePO-specific utilities used by data generation, judging, and math evaluation.
pip install openai jinja2 tabulate regex sympy antlr4-python3-runtime \
  timeout-decorator pebble multiprocess jsonlines

# Clone this repository and use its local openrlhf package.
git clone <this-repo-url> RePO
cd RePO
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

OpenRLHF recommends vLLM 0.8.2 or higher. For vLLM 0.8.2 or nightly builds, you may also use:

```bash
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

See the upstream [OpenRLHF installation guide](https://github.com/OpenRLHF/OpenRLHF) for more environment details.

Or, if you want to try a simple setup:

```bash
pip install vllm==0.11.2
pip install https://github.com/mjun0812/flash-attention-prebuild- \
  wheels/releases/download/{COMPATIBLE_WITH_YOUR_ENVIRONMENT}.whl
pip install -r requirements.txt
```
## Dataset Format

RePO training expects JSONL records with `prompt`, `chosen`, `rejected`, `chosen_logprob_with_token`, and `rejected_logprob_with_token`:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "chosen_logprob_with_token": {
    "tokens": ["...", "..."],
    "logprobs": [-0.12, -0.34]
  },
  "rejected_logprob_with_token": {
    "tokens": ["...", "..."],
    "logprobs": [-0.56, -0.78]
  }
}
```

`RePO_det` is the deterministic approximation and only requires ordinary preference records:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

The same plain `(prompt, chosen, rejected)` format is usable for baselines such as DPO, TDPO, RPO, KTO, and IPO. Full RePO uses the additional rollout-policy logprob fields.

## Math Data Pipeline

The math pipeline first generates multiple rollouts per problem, filters them into correct/incorrect pairs, and then attaches token-level rollout-policy logprobs.

### 1. Generate math rollouts

The default math rollout input is `RePO_datasets/metamathqa_processed.jsonl`. With the wrapper defaults, each JSONL row must contain `question`, `answer`, and `index`:

```json
{"index": 0, "question": "Solve ...", "answer": "42"}
```

`question` is formatted by the selected prompt template, `answer` is used to filter generated responses into correct/incorrect candidates, and `index` is written as the generated record's `data_id`. For custom JSONL files, override `INPUT_KEY`, `ANSWER_KEY`, and `DATA_ID_KEY`.

```bash
# The generation CLI appends a timestamp suffix to OUTPUT_DIR.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=Qwen/Qwen2.5-Math-7B-Instruct \
DATASET=RePO_datasets/metamathqa_processed.jsonl \
OUTPUT_DIR=RePO_datasets/MetamathQA/test_qwen2.5-Math-7B-Instruct \
bash scripts/generate_math_rollouts.sh
```

### 2. Convert rollouts into preference pairs

Use the actual timestamped rollout directory produced above as `--input_dir`.

```bash
python -m openrlhf.cli.data_preprocessing_pairdata \
  --input_dir RePO_datasets/MetamathQA/test_qwen2.5-Math-7B-Instruct_<timestamp> \
  --output RePO_datasets/MetamathQA/preference_pairs.jsonl \
  --metadata_output RePO_datasets/MetamathQA/metadata.jsonl \
  --portion_output RePO_datasets/MetamathQA/portion_preference_pairs.jsonl
```

### 3. Build the RePO dataset with rollout logprobs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PAIR_DATASET=RePO_datasets/MetamathQA/preference_pairs.jsonl \
ROLLOUT_MODEL=Qwen/Qwen2.5-Math-7B-Instruct \
OUTPUT_DIR=RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch \
bash scripts/build_math_repo_dataset.sh
```

This wrapper uses `openrlhf.cli.build_RePO_dataset_topk_dist_batch`.

## UltraFeedback / Human Feedback Pipeline

The UltraFeedback-style pipeline generates responses from multiple models, rates them, binarizes them into chosen/rejected pairs, and then computes rollout logprobs for each generator model.

### 1. Generate responses

Run once per generator model. The default base dataset is `HuggingFaceH4/ultrafeedback_binarized`, loaded from its `train_sft` split by the current generation CLI. Generation requires `prompt` and `prompt_id`; the upstream dataset also includes `chosen`, `rejected`, `messages`, `score_chosen`, and `score_rejected`.

For a custom JSONL dataset, use one row per prompt:

```json
{"prompt_id": "unique-id", "prompt": "Write a helpful answer to ..."}
```

The generated output is merged into `generation_merged.jsonl` records containing `prompt_id`, `prompt`, `response`, and `model`. Those records are then merged across generator models, judged, and binarized.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL=Qwen/Qwen3-4B-Instruct-2507 \
MODEL_NAME_FOR_LOG=qwen3-4b-instruct-2507 \
bash scripts/generate_uf_rollouts.sh
```

### 2. Merge, rate, and binarize model outputs

```bash
python -m openrlhf.cli.merge_model_uf_outputs \
  --root_dir RePO_datasets/Ultrafeedback/generation_logs \
  --output RePO_datasets/Ultrafeedback/merged_model_outputs/merged_all_models.jsonl

# If using this repo's OpenAI-judge script, configure the API/model settings inside the script.
python rating_model_outputs_with_openai_multiproc.py

python -m openrlhf.cli.uf_dataset_binarize \
  --input RePO_datasets/Ultrafeedback/model_output_with_rating/eval_combined.jsonl \
  --output RePO_datasets/Ultrafeedback/biniarized/binary_pair_templated.jsonl
```

The directory name `biniarized` is retained for compatibility with the existing experiment scripts.

### 3. Attach rollout logprobs

Run this once for each generator model that appears in `chosen_model` or `rejected_model`. Each pass adds logprob fields for one `MODEL_ID` while preserving fields that already exist, so multi-model UltraFeedback runs should be chained: use the previous pass's `RePO_train_${SAVE_NAME}.jsonl` as the next pass's `PAIR_DATASET`.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL=Qwen/Qwen3-4B-Instruct-2507 \
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
SAVE_NAME=Qwen3-4B-Instruct-2507 \
PAIR_DATASET=RePO_datasets/Ultrafeedback/biniarized/binary_pair_templated.jsonl \
OUTPUT_DIR=RePO_datasets/Ultrafeedback/biniarized \
bash scripts/build_uf_repo_dataset.sh
```

For an optional held-out/eval pass, add `RETURN_EVAL=1`.

## Training

### RePO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=Qwen/Qwen3-4B-Base \
DATASET=RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch/RePO_train.jsonl \
SAVE_PATH=checkpoint/Qwen3-4B-metamathqa-RePO \
bash scripts/train_repo.sh
```

### RePO_det

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=Qwen/Qwen3-4B-Base \
DATASET=RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch/RePO_train.jsonl \
SAVE_PATH=checkpoint/Qwen3-4B-metamathqa-RePO_det \
bash scripts/train_repo_det.sh
```

For UltraFeedback, use the same wrappers with the UltraFeedback RePO dataset:

```bash
DATASET="RePO_datasets/Ultrafeedback/biniarized/RePO_train_biniarized_cleaned.jsonl" \
SAVE_PATH=checkpoint/Qwen3-4B-ultrafeedback-RePO \
bash scripts/train_repo.sh
```

Baseline implementations are available through the corresponding OpenRLHF CLI modules:

- DPO/RPO/IPO/SimPO: `openrlhf.cli.train_dpo`
- TDPO: `openrlhf.cli.train_tdpo`
- KTO: `openrlhf.cli.train_kto`

## Masked Pair Augmentation

Masked pair augmentation builds additional preference pairs by truncating/masking selected continuations and preferring less-masked variants. The input is a grouped JSONL file with `chosen_set` and `rejected_set` fields, such as the output expected by `openrlhf.cli.build_mask_pair_dataset`.

```bash
INPUT=RePO_datasets/MetamathQA/logp_scored_qwen2.5-Math-7B-Instruct/logprob_grouped.jsonl \
OUTPUT=RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/mask_pair_chosen_fixed/RePO_train.jsonl \
bash scripts/build_mask_pair_dataset.sh

DATASET=RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/mask_pair_chosen_fixed/RePO_train.jsonl \
SAVE_PATH=checkpoint/Qwen3-4B-metamathqa-RePO-mask-pair \
bash scripts/train_repo.sh
```

The wrapper uses `--basic-pairs --add-mask-pairs --fix-chosen` by default.

## Evaluation

Math benchmark evaluation uses distributed vLLM generation and writes one JSONL per benchmark plus a summary table.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=checkpoint/Qwen3-4B-metamathqa-RePO_merged \
DATASET=evaluation/eval_data_basic/ \
OUTPUT_DIR=evaluation/logs/Qwen3-4B-metamathqa-RePO \
RESULT_TXT_PATH=evaluation_results_txt/math_results_pass_1.txt \
bash scripts/eval_math.sh
```

This wraps `openrlhf.cli.evaluation_benchmarks_vllm_dist`.

## Acknowledgements and License

This repository is derived from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), an Apache-2.0 licensed RLHF framework. We retain OpenRLHF attribution in `NOTICE` and distribute this repository under the Apache License 2.0. See `LICENSE` for details.

## Citation

If you use this repository, please cite RePO:

```bibtex
@inproceedings{kim2026repo,
  title = {A Regret Minimization Framework on Preference Learning in Large Language Models},
  author = {Kim, Suhwan and Cho, Taehyun and Kim, GeonHyeong and Kim, YuJin and Jang, Youngsoo and Lee, Moontae and Lee, Jungwoo},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year = {2026}
}
```

Please also cite OpenRLHF when using the inherited framework components:

```bibtex
@article{hu2024openrlhf,
  title = {OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author = {Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal = {arXiv preprint arXiv:2405.11143},
  year = {2024}
}
```
