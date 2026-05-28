import os
import json
import argparse
from tqdm import tqdm

def merge_generations(root_dir, output_path):
    """
    root_dir: model folders containing the model folders
    output_path: final merged jsonl file path
    """

    # save information by prompt_id
    merged = {}  # key = prompt_id, value = {"prompt": ..., "response": {model_id: response}}

    # iterate over all model folders in the root directory
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print("Found model folders:", model_dirs)

    for i, model_id in enumerate(model_dirs):
        jsonl_path = os.path.join(root_dir, model_id, "generation_merged.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"generation_merged.jsonl not found in {model_id}, skipping...")
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {model_id}"):
            
                data = json.loads(line)

                pid = data["prompt_id"]
                prompt = data["prompt"]
                response = data["response"]
                model = data["model"]

                if pid not in merged:
                    merged[pid] = {
                        "prompt": prompt,
                        "response": {}
                    }

                # model ID -> response mapping
                merged[pid]["response"][model] = response

    count_missing_responses = 0
    missing_prompts = []
    for pid, info in tqdm(merged.items(), desc="Checking missing responses"):
        if len(info["response"]) != len(model_dirs):
            # print(f"Prompt ID {pid} has missing responses")
            count_missing_responses += 1
            missing_prompts.append(pid)

    print(f"Count of missing responses: {count_missing_responses}")
    for pid in missing_prompts:
        merged.pop(pid)
    print(f"Count of prompts after removing missing responses: {len(merged)}")
    # save final jsonl
    with open(output_path, "w", encoding="utf-8") as out:
        for pid, info in merged.items():
            record = {
                "prompt_id": pid,
                "prompt": info["prompt"],
                "response": info["response"]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved merged jsonl at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge UltraFeedback generations from multiple model subdirectories."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="RePO_datasets/Ultrafeedback/generation_logs",
        help="Directory containing one subdirectory per generator model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="RePO_datasets/Ultrafeedback/merged_model_outputs/merged_all_models.jsonl",
        help="Merged jsonl output path.",
    )
    args = parser.parse_args()

    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    merge_generations(root_dir=args.root_dir, output_path=args.output)


if __name__ == "__main__":
    main()
