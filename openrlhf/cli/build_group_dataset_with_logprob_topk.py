import json
from collections import defaultdict
from typing import Dict, Any
from tqdm import tqdm
from statistics import mean
def regroup_jsonl(
    input_path: str,
    output_path: str,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    chosen_logprob_key: str = "chosen_logprob_with_token",
    rejected_logprob_key: str = "rejected_logprob_with_token",
    chosen_top_k_key: str = "chosen_top_k_data",
    rejected_top_k_key: str = "rejected_top_k_data",
    max_samples: int = None,
):
    """
    read jsonl file and group by prompt,
    collect chosen / rejected into set (remove duplicates) and save as new jsonl file.
    """ 

     
    # groups[prompt] = {
    #     "chosen": { chosen_text: chosen_logprob_with_token, ... },
    #     "rejected": { rejected_text: rejected_logprob_with_token, ... },
    # }
    groups: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: {"chosen": {}, "rejected": {}}
    )

    file_size = os.path.getsize(input_path)

    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing")
     
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        #for line_no, line in enumerate(tqdm(f, desc="Processing lines"), start=1):
        for line_no, line in enumerate(f):
            pbar.update(len(line))
            pbar.set_postfix(line_no=line_no)
            if max_samples is not None and line_no > max_samples:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                 
                # print(f"Warning: JSON decode error at line {line_no}")
                continue

            if prompt_key not in obj:
                 
                # print(f"Warning: no prompt at line {line_no}")
                continue

            prompt = obj[prompt_key]
            group = groups[prompt]

             
            if chosen_key in obj:
                chosen_text = obj[chosen_key]
                 
                if chosen_text not in group["chosen"]:
                    if chosen_logprob_key in obj:
                        group["chosen"][chosen_text] = [obj[chosen_logprob_key], obj[chosen_top_k_key]]
                    else:
                        group["chosen"][chosen_text] = None   

             
            if rejected_key in obj:
                rejected_text = obj[rejected_key]
                if rejected_text not in group["rejected"]:
                    if rejected_logprob_key in obj:
                        group["rejected"][rejected_text] = [obj[rejected_logprob_key], obj[rejected_top_k_key]]
                    else:
                        group["rejected"][rejected_text] = None

     
    print(f"Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for prompt, group in tqdm(groups.items(), desc="Writing groups"):
            chosen_set = [
                {
                    "chosen": chosen_text,
                    "chosen_logprob_with_token": group["chosen"][chosen_text][0],
                    "chosen_logprob_score": mean(group["chosen"][chosen_text]["logprobs"]) if group["chosen"][chosen_text] is not None else None,
                    "chosen_top_k_data": group["chosen"][chosen_text][1]
                }
                for chosen_text in group["chosen"].keys()
            ]
            rejected_set = [
                {
                    "rejected": rejected_text,
                    "rejected_logprob_with_token": group["rejected"][rejected_text][0],
                    "rejected_logprob_score": mean(group["rejected"][rejected_text]["logprobs"]) if group["rejected"][rejected_text] is not None else None,
                    "rejected_top_k_data": group["rejected"][rejected_text][1]
                }
                for rejected_text in group["rejected"].keys()
            ]

            new_obj = {
                "prompt": prompt,
                "chosen_set": chosen_set,
                "rejected_set": rejected_set,
            }

            out_f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
     
    input_file = "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch_new/RePO_train.jsonl"
    output_file = "RePO_datasets/MetamathQA/logp_scored_qwen2.5-Math-7B-Instruct/topk_logprob_grouped.jsonl"
    max_samples = 500000
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Processing {max_samples} samples from {input_file} to {output_file}")
    regroup_jsonl(input_file, output_file, max_samples=max_samples)
    print(f"Done. Wrote grouped data to {output_file}")
