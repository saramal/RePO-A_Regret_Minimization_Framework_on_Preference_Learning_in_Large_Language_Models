#!/usr/bin/env python
import json
import argparse
import random
from typing import List, Dict, Any
import os
from tqdm import tqdm
 
DEFAULT_INPUT = "RePO_datasets/MetamathQA/logp_scored_qwen2.5-Math-7B-Instruct/logprob_grouped.jsonl"
DEFAULT_OUTPUT = "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/mask_pair_chosen_fixed_only/RePO_train.jsonl"


def select_top_k(items: List[Dict[str, Any]], score_key: str, k: int = 2) -> List[Dict[str, Any]]:
    """select top k items based on score_key."""
    if not items:
        return []
    return sorted(items, key=lambda x: x.get(score_key, float("-inf")), reverse=True)[:k]


def build_masked_variants(
    chosen_item: Dict[str, Any],
    mask_range: range = range(0, 5),
    tokens_per_mask: int = 16,
) -> List[Dict[str, Any]]:
    """
    create masked variants for chosen_item with mask_len=0..4.

    chosen_item structure:
      {
        "chosen": str,
        "chosen_logprob_with_token": {"token": List[str], "logprobs": List[float]},
        "chosen_logprob_score": float,
        ...
      }

    each returned element is:
      {
        "chosen": masked_text,
        "chosen_logprob_with_token": {"token": masked_tokens, "logprobs": masked_logprobs},
        "mask_len": mask
      }
    """
    variants = []

    logprob_with_token = chosen_item.get("chosen_logprob_with_token", {})
    tokens = list(logprob_with_token.get("tokens", []))
    logprobs = list(logprob_with_token.get("logprobs", []))

    if len(tokens) != len(logprobs) or len(tokens) == 0:
        raise ValueError(f"Tokens and logprobs length mismatch: {len(tokens)} != {len(logprobs)}")
         
        min_len = min(len(tokens), len(logprobs))
        tokens = tokens[:min_len]
        logprobs = logprobs[:min_len]

    base_len = len(tokens)

    for mask in mask_range:
         
        cut = max(0, base_len - mask * tokens_per_mask)
        if cut <= 0:
             
            continue

        masked_tokens = tokens[:cut]
        masked_logprobs = logprobs[:cut]

         
        masked_text = "".join(masked_tokens)

        variants.append(
            {
                "chosen": masked_text,
                "chosen_logprob_with_token": {
                    "tokens": masked_tokens,
                    "logprobs": masked_logprobs,
                },
                "mask_len": mask,
            }
        )

    return variants


def process_record(
    record: Dict[str, Any],
    add_mask_pairs: bool = False,
    basic_pairs: bool = True,
    fix_chosen: bool = False,
) -> List[Dict[str, Any]]:
    """
    take one line (record) and create a list of new pair lines (dict).
    """
    prompt = record.get("prompt", "")

    chosen_set = record.get("chosen_set", []) or []
    rejected_set = record.get("rejected_set", []) or []

     
    selected_chosen = select_top_k(chosen_set, "chosen_logprob_score", k=2)

     
    selected_rejected = select_top_k(rejected_set, "rejected_logprob_score", k=2)

    new_lines: List[Dict[str, Any]] = []

     
    if basic_pairs:
        if selected_chosen and selected_rejected:
            for c in selected_chosen:
                for r in selected_rejected:
                    new_lines.append(
                        {
                            "prompt": prompt,
                            "chosen": c["chosen"],
                            "rejected": r["rejected"],
                            "chosen_logprob_with_token": c["chosen_logprob_with_token"],
                            "rejected_logprob_with_token": r["rejected_logprob_with_token"],
                        }
                    )

     
    if add_mask_pairs and selected_chosen:
        for c in selected_chosen:
            masked_selected_chosen: List[Dict[str, Any]] = []
            masked_selected_chosen.extend(build_masked_variants(c))

             
            if len(masked_selected_chosen) >= 2:
                num_pairs_to_make = 2
                pairs_made = 0
                max_attempts = num_pairs_to_make * 10   
                attempts = 0
                n = len(masked_selected_chosen)

                while pairs_made < num_pairs_to_make and attempts < max_attempts:
                    attempts += 1
                    if fix_chosen:
                        i=0
                        j = random.sample(range(1, n), 1)[0]
                    else:
                        i, j = random.sample(range(n), 2)
                    a = masked_selected_chosen[i]
                    b = masked_selected_chosen[j]

                     
                    if a["mask_len"] == b["mask_len"]:
                        continue

                    if a["mask_len"] < b["mask_len"]:
                        chosen_item, rejected_item = a, b
                    else:
                        chosen_item, rejected_item = b, a

                    new_lines.append(
                        {
                            "prompt": prompt,
                            "chosen": chosen_item["chosen"],
                            "rejected": rejected_item["chosen"],
                            "chosen_logprob_with_token": chosen_item[
                                "chosen_logprob_with_token"
                            ],
                            "rejected_logprob_with_token": rejected_item[
                                "chosen_logprob_with_token"
                            ],
                             
                            "chosen_mask_len": chosen_item["mask_len"],
                            "rejected_mask_len": rejected_item["mask_len"],
                        }
                    )
                    pairs_made += 1

    return new_lines


def process_jsonl(
    input_path: str,
    output_path: str,
    add_mask_pairs: bool = True,
    basic_pairs: bool = True,
    fix_chosen: bool = False,
):
    """
    read input jsonl and save as new jsonl according to the algorithm.
    """
    print(f"Processing {input_path} to {output_path}")
    print(f"Adding mask pairs: {add_mask_pairs}")
    print(f"Basic pairs: {basic_pairs}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_size = os.path.getsize(input_path)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing")
    dataset_size=0
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(tqdm(fin)):
            pbar.update(len(line))
            pbar.set_postfix(line_no=line_no)
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            new_records = process_record(record, add_mask_pairs=add_mask_pairs, basic_pairs=basic_pairs, fix_chosen=fix_chosen)
            for r in new_records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            dataset_size += len(new_records)
    print(f"Total dataset size: {dataset_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess jsonl into pairwise (prompt, chosen, rejected) jsonl with optional mask-based intra-chosen pairs."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input jsonl file path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output jsonl file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--add-mask-pairs",
        action="store_true",
        default=False,
        help="enable 4th way (creating mask-based chosen intra-pair line).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility of random sampling (default: 42).",
    )
    parser.add_argument(
        "--basic-pairs",
        action="store_true",
        default=False,
        help="create basic pairs (prompt, chosen, rejected).",
    )
    parser.add_argument(
        "--fix-chosen",
        action="store_true",
        default=False,
        help="fix chosen as unmasked, sample mask for rejected.",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    process_jsonl(
        args.input,
        args.output,
        add_mask_pairs=args.add_mask_pairs,
        basic_pairs=args.basic_pairs,
        fix_chosen=args.fix_chosen,
    )


if __name__ == "__main__":
    main()
