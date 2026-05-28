#!/usr/bin/env python
import json
import random
import argparse
from typing import Dict, Any
from tqdm import tqdm
import pdb

def process_line(example: Dict[str, Any], rng: random.Random):
    """
    take one original example (dict) and convert it to a binary pair (chosen, rejected) example.
    return None if it cannot be used.
    """
    prompt_id = example.get("prompt_id")
    prompt = example.get("prompt")
    responses = example.get("response", {})
    ratings = example.get("ratings", {})
    mean_ratings = example.get("mean_ratings", {})
    annotator = example.get("annotator")

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if not mean_ratings or len(mean_ratings) < 2:
         
        return None

     
    values = list(mean_ratings.values())

     
    unique_values = set(values)
    if len(unique_values) == 1:
        return "same_rating"

     
    max_mean = max(values)
    max_models = [m for m, v in mean_ratings.items() if v == max_mean]

     
    chosen_model = rng.choice(max_models)

     
    lower_models = [m for m, v in mean_ratings.items() if v < max_mean]

     
    if not lower_models:
         
        return None

    rejected_model = rng.choice(lower_models)

     
    chosen_response = responses.get(chosen_model)
    rejected_response = responses.get(rejected_model)
    chosen_rating = ratings.get(chosen_model)
    rejected_rating = ratings.get(rejected_model)

     
    if chosen_response is None or rejected_response is None:
        return None
    if chosen_rating is None or rejected_rating is None:
        return None

     
    out = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "chosen_rating": chosen_rating,
        "rejected_rating": rejected_rating,
        "chosen_model": chosen_model,
        "rejected_model": rejected_model,
        "annotator": annotator,
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="convert multi-model rating jsonl to (prompt, chosen, rejected) jsonl"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="RePO_datasets/Ultrafeedback/model_output_with_rating/eval_combined.jsonl",
        help="original multi-response jsonl file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="RePO_datasets/Ultrafeedback/biniarized/binary_pair_templated.jsonl",
        help="path to save the generated binary pair jsonl file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    num_in = 0
    num_out = 0
    num_same_rating = 0
    num_skipped = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin):
            # pdb.set_trace()
            line = line.strip()
            if not line:
                continue
            num_in += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                num_skipped += 1
                continue

            processed = process_line(example, rng)
            if processed is None:
                num_skipped += 1
                continue
            if processed == "same_rating":
                num_same_rating += 1
                continue

            fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
            num_out += 1

    print(f"Input lines:   {num_in}")
    print(f"Output lines:  {num_out}")
    print(f"Skipped lines: {num_skipped}")
    print(f"Same rating lines: {num_same_rating}")


if __name__ == "__main__":
    main()
