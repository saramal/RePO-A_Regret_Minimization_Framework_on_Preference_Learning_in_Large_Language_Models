import os
import json
import shutil
import hashlib
import tempfile
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from statistics import mean


def _bucket_id(prompt: str, num_buckets: int) -> int:
     
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return int(h, 16) % num_buckets


def regroup_jsonl_lowmem_bucketdir(
    input_path: str,
    output_path: str,
    bucket_dir: str,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    chosen_logprob_key: str = "chosen_logprob_with_token",
    rejected_logprob_key: str = "rejected_logprob_with_token",
    chosen_top_k_key: str = "chosen_top_k_data",
    rejected_top_k_key: str = "rejected_top_k_data",
    max_samples: Optional[int] = None,
    num_buckets: int = 256,
    max_open_files: int = 64,
    cleanup_bucket_dir: bool = True,
    overwrite_buckets: bool = True,
):
    """
    use disk (bucket files) instead of RAM to perform prompt-wise regrouping.

    - Pass1: distribute input jsonl into N bucket files based on prompt hash
    - Pass2: read bucket files one by one to create groups and append to output jsonl

    bucket_dir:
      - directory to store temporary bucket files (user-specified)
    cleanup_bucket_dir:
      - delete bucket_dir after completion (recommended)
    overwrite_buckets:
      - delete existing bucket_*.jsonl in bucket_dir and create new ones
    """

    if num_buckets <= 0:
        raise ValueError("num_buckets must be > 0")

    if max_open_files <= 0:
        raise ValueError("max_open_files must be > 0")

    max_open_files = min(max_open_files, num_buckets)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(bucket_dir, exist_ok=True)

     
    if overwrite_buckets:
        for i in range(num_buckets):
            p = os.path.join(bucket_dir, f"bucket_{i:04d}.jsonl")
            if os.path.exists(p):
                os.remove(p)

    bucket_paths = [os.path.join(bucket_dir, f"bucket_{i:04d}.jsonl") for i in range(num_buckets)]

     
    with open(output_path, "w", encoding="utf-8") as _:
        pass

    file_size = os.path.getsize(input_path)

    # --------------------
    # Pass 1: Partition
    # --------------------
    print(f"[Pass1] Partitioning into {num_buckets} buckets at: {bucket_dir}")

    open_files: Dict[int, Any] = {}

    def _close_one_file():
         
        bid, f = open_files.popitem()
        f.close()

    def get_bucket_file(bid: int):
        if bid in open_files:
            return open_files[bid]
        if len(open_files) >= max_open_files:
            _close_one_file()
        f = open(bucket_paths[bid], "a", encoding="utf-8")
        open_files[bid] = f
        return f

    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Pass1")
    line_no = 0

    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line_no += 1
                pbar.update(len(line))
                pbar.set_postfix(line_no=line_no)

                if max_samples is not None and line_no > max_samples:
                    break

                s = line.strip()
                if not s:
                    continue

                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue

                prompt = obj.get(prompt_key)
                if not prompt:
                    continue

                bid = _bucket_id(prompt, num_buckets)
                bf = get_bucket_file(bid)
                bf.write(s + "\n")
    finally:
        pbar.close()
         
        for f in open_files.values():
            try:
                f.close()
            except Exception:
                pass
        open_files.clear()

    # --------------------
    # Pass 2: Group & Write
    # --------------------
    print(f"[Pass2] Grouping bucket-by-bucket and writing: {output_path}")

    def _score_from_logprob(lp: Any) -> Optional[float]:
         
        if isinstance(lp, dict):
            lps = lp.get("logprobs")
            if isinstance(lps, list) and len(lps) > 0:
                return mean(lps)
        return None

    try:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for bpath in tqdm(bucket_paths, desc="Buckets"):
                if (not os.path.exists(bpath)) or os.path.getsize(bpath) == 0:
                    continue

                groups: Dict[str, Dict[str, Dict[str, Tuple[Any, Any]]]] = defaultdict(
                    lambda: {"chosen": {}, "rejected": {}}
                )

                with open(bpath, "r", encoding="utf-8") as bf:
                    for line in bf:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except json.JSONDecodeError:
                            continue

                        prompt = obj.get(prompt_key)
                        if not prompt:
                            continue

                        group = groups[prompt]

                        # chosen
                        if chosen_key in obj:
                            chosen_text = obj[chosen_key]
                            if chosen_text not in group["chosen"]:
                                lp = obj.get(chosen_logprob_key)
                                tk = obj.get(chosen_top_k_key)
                                group["chosen"][chosen_text] = (lp, tk)

                        # rejected
                        if rejected_key in obj:
                            rejected_text = obj[rejected_key]
                            if rejected_text not in group["rejected"]:
                                lp = obj.get(rejected_logprob_key)
                                tk = obj.get(rejected_top_k_key)
                                group["rejected"][rejected_text] = (lp, tk)

                # write grouped results for this bucket
                for prompt, group in groups.items():
                    chosen_set = []
                    for chosen_text, (lp, tk) in group["chosen"].items():
                        chosen_set.append(
                            {
                                "chosen": chosen_text,
                                "chosen_logprob_with_token": lp,
                                "chosen_logprob_score": _score_from_logprob(lp),
                                "chosen_top_k_data": tk,
                            }
                        )

                    rejected_set = []
                    for rejected_text, (lp, tk) in group["rejected"].items():
                        rejected_set.append(
                            {
                                "rejected": rejected_text,
                                "rejected_logprob_with_token": lp,
                                "rejected_logprob_score": _score_from_logprob(lp),
                                "rejected_top_k_data": tk,
                            }
                        )

                    new_obj = {
                        prompt_key: prompt,
                        "chosen_set": chosen_set,
                        "rejected_set": rejected_set,
                    }
                    out_f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

                 

    finally:
        # --------------------
        # Cleanup
        # --------------------
        if cleanup_bucket_dir:
            try:
                shutil.rmtree(bucket_dir)
                print(f"[Cleanup] Removed bucket_dir: {bucket_dir}")
            except Exception as e:
                print(f"[Cleanup] Failed to remove bucket_dir {bucket_dir}: {e}")


if __name__ == "__main__":
    input_file = "RePO_datasets/MetamathQA/RePO/qwen2.5-Math-7B-Instruct_dist_batch_new/RePO_train.jsonl"
    output_file = "RePO_datasets/MetamathQA/logp_scored_qwen2.5-Math-7B-Instruct/topk_logprob_grouped.jsonl"
    max_samples = 500000

     
    bucket_dir = "RePO_datasets/MetamathQA/tmp_buckets_qwen2_5_math_7b"

    print(f"Processing {max_samples} samples from {input_file} to {output_file}")
    regroup_jsonl_lowmem_bucketdir(
        input_path=input_file,
        output_path=output_file,
        bucket_dir=bucket_dir,
        max_samples=max_samples,
        num_buckets=1024,          
        max_open_files=64,        
        cleanup_bucket_dir=True,  
        overwrite_buckets=True,   
    )
    print(f"Done. Wrote grouped data to {output_file}")
