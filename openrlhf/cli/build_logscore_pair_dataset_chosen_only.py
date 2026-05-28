#!/usr/bin/env python
import json
import argparse
from typing import List, Dict, Any
import os
from tqdm import tqdm
import random


def select_top_k(items: List[Dict[str, Any]], score_key: str, k: int = 4) -> List[Dict[str, Any]]:
    """select top k items based on score_key."""
    if not items:
        return []
     
    sorted_items = sorted(items, key=lambda x: x.get(score_key, float("-inf")), reverse=True)
    return sorted_items[:k] if len(sorted_items) >= k else sorted_items


def process_record(
    record: Dict[str, Any],
    add_intraset_pairs: bool = True,
    basic_pairs: bool = False,
) -> List[Dict[str, Any]]:
    """
    take one line (record) and create a list of new lines (dict).
    """
    prompt = record.get("prompt", "")

    chosen_set = record.get("chosen_set", []) or []
    rejected_set = record.get("rejected_set", []) or []

     
    selected_chosen = select_top_k(chosen_set, "chosen_logprob_score", k=6)

     
    selected_rejected = select_top_k(rejected_set, "rejected_logprob_score", k=6)

    new_lines: List[Dict[str, Any]] = []

     
    if basic_pairs and selected_chosen and selected_rejected:
        for c in selected_chosen:
            for r in selected_rejected:
                new_lines.append(
                    {
                        "prompt": prompt,
                        "chosen": c["chosen"],
                        "rejected": r["rejected"],
                        "chosen_logprob_with_token": c["chosen_logprob_with_token"],
                        "rejected_logprob_with_token": r[
                            "rejected_logprob_with_token"
                        ],
                    }
                )

     
    if add_intraset_pairs and len(selected_chosen) >= 2:


        n = len(selected_chosen)
        pairs_made = 0
        num_pairs_to_make = 4
        max_attempts  = 20
        attempts = 0
        while pairs_made < num_pairs_to_make and attempts < max_attempts:
            attempts += 1
            i, j = random.sample(range(n), 2)
            a = selected_chosen[i]
            b = selected_chosen[j]
            if a["chosen_logprob_score"] > b["chosen_logprob_score"]:
                chosen_item, rejected_item = a, b
            elif a["chosen_logprob_score"] < b["chosen_logprob_score"]:
                chosen_item, rejected_item = b, a
            else:
                continue
            new_lines.append(
                {
                    "prompt": prompt,
                    "chosen": chosen_item["chosen"],
                    "rejected": rejected_item["chosen"],
                    "chosen_logprob_with_token": chosen_item["chosen_logprob_with_token"],
                    "rejected_logprob_with_token": rejected_item["chosen_logprob_with_token"],
                }
            )
            pairs_made += 1

       

    return new_lines


def process_jsonl(
    input_path: str,
    output_path: str,
    add_intraset_pairs: bool = True,
    basic_pairs: bool = False,
):
    """
    read input jsonl and save as new jsonl according to the algorithm.
    """
    print(f"Processing {input_path} to {output_path}")
    print(f"Adding intra-set pairs: {add_intraset_pairs}")
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
            new_records = process_record(record, add_intraset_pairs=add_intraset_pairs, basic_pairs=basic_pairs)
            for r in new_records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            dataset_size += len(new_records)
    print(f"Total dataset size: {dataset_size}")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess jsonl into pairwise (prompt, chosen, rejected) jsonl."
    )
    parser.add_argument("--input", type=str, default="RePO_datasets/MetamathQA/logp_scored_qwen2.5-Math-7B-Instruct/logprob_grouped.jsonl", help="Input jsonl file path")
    parser.add_argument("--output", type=str, default="RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct/logscore_pair_chosen_only_for_merge/RePO_train.jsonl", help="Output jsonl file path")
    parser.add_argument(
        "--no-extra-pairs",
        action="store_true",
        help="disable 4th way (creating one more line in selected_list based on high/low).",
    )
    parser.add_argument(
        "--basic-pairs",
        action="store_true",
        default=False,
        help="create basic pairs (prompt, chosen, rejected).",
    )
    

    args = parser.parse_args()

    process_jsonl(
        args.input,
        args.output,
        add_intraset_pairs=not args.no_extra_pairs,
        basic_pairs=args.basic_pairs,
    )


if __name__ == "__main__":
    main()
