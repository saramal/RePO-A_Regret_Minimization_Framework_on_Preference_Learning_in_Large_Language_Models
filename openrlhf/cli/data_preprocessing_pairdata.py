import os
import json
import itertools
import tqdm
import argparse

import re
import unicodedata

# wide-allow(unicode math/alphabet symbols)
VALID_LATEXY = re.compile(r"""
    \A[ 
        A-Za-z0-9
        \s                                  # whitespace, tab, newline
        \.\,\;\:\!\?\'\"\_\-\+\=\*\/\^\|\%\<\>\~\#\@\&\$
        \\ \(\) \[ \] \{ \}                 # backslash and parentheses

        \u0370-\u03FF                       # Greek and Coptic
        \u1F00-\u1FFF                       # Greek Extended

        \u2190-\u21FF                       # Arrows
        \u2200-\u22FF                       # Mathematical Operators
        \u27C0-\u27EF                       # Misc Math Symbols A
        \u2980-\u29FF                       # Misc Math Symbols B
        \u2A00-\u2AFF                       # SuRePOemental Math Operators

        \u2100-\u214F                       # Letterlike Symbols (ℝ, ℤ, ℵ ...)
        \u2070-\u209F                       # Superscripts & Subscripts (⁰ⁱⁿ, ₓᵢ etc.)
        \u1D400-\u1D7FF                     # Mathematical Alphanumeric Symbols (𝔸, 𝕽, 𝒙 ...)

        \u00B0\u00B1\u00B2\u00B3\u00B7\u00B9\u00D7\u00F7  # ° ± ² ³ · ¹ × ÷
    ]*\Z
""", re.VERBOSE)

def is_valid_generated_text(text: str) -> bool:
    return bool(VALID_LATEXY.match(text))


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_preference_pairs(input_dir, output_path, metadata_path, portion_output_path=None, portion_size=64):
    all_pairs = []
    metadata = []

    batch_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jsonl"))

    for bf in tqdm.tqdm(batch_files, desc="Processing batch files"):
        path = os.path.join(input_dir, bf)
        with open(path, "r", encoding="utf-8") as f:
            batch_data = [json.loads(line) for line in f]

        # query-wise pair generation
        for item in batch_data:
            query = item["query"]
            responses = item["responses"]
            data_id = item["data_id"]

            answers = [r["gold_answer"] for r in responses]
            if len(set(answers)) == 1:
                answer = answers[0]
            else:
                print(f"Multiple answers found: {answers}")
                continue

            positives = [r["text"] for r in responses if r["is_correct"] == 1]
            negatives = [r["text"] for r in responses if r["is_correct"] == 0]
            positives = [p for p in positives if is_valid_generated_text(p)]
            negatives = [n for n in negatives if is_valid_generated_text(n)]
            if len(positives) == 0 or len(negatives) == 0:
                continue

            for pos, neg in itertools.product(positives, negatives):
                all_pairs.append(
                    {
                        "question": query,
                        "chosen": pos,
                        "rejected": neg,
                        "data_id": data_id,
                        "answer": answer,
                    }
                )

            metadata.append(
                {
                    "data_id": data_id,
                    "query": query,
                    "answer": answer,
                    "num_positives": len(positives),
                    "num_negatives": len(negatives),
                    "num_pairs": len(all_pairs),
                }
            )

    ensure_parent_dir(output_path)
    ensure_parent_dir(metadata_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")

    if portion_output_path:
        ensure_parent_dir(portion_output_path)
        with open(portion_output_path, "w", encoding="utf-8") as f:
            for pair in all_pairs[:portion_size]:
                json.dump(pair, f, ensure_ascii=False)
                f.write("\n")

    print(f"Finished. Saved {len(all_pairs)} pairs to {output_path}")


def main():
    default_data_path = "RePO_datasets/MetamathQA/test_qwen2.5-Math-7B-Instruct_0824T0632"
    parser = argparse.ArgumentParser(
        description="Build math preference pairs from vLLM rollout batches."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(default_data_path, "batchs"),
        help="Directory containing batch_*.jsonl rollout files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(default_data_path, "preference_pairs.jsonl"),
        help="Output preference-pair jsonl path.",
    )
    parser.add_argument(
        "--metadata_output",
        type=str,
        default=os.path.join(default_data_path, "metadata.jsonl"),
        help="Output metadata jsonl path.",
    )
    parser.add_argument(
        "--portion_output",
        type=str,
        default=os.path.join(default_data_path, "portion_preference_pairs.jsonl"),
        help="Optional small debug subset output path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--portion_size",
        type=int,
        default=64,
        help="Number of examples to write to --portion_output.",
    )
    args = parser.parse_args()

    build_preference_pairs(
        input_dir=args.input_dir,
        output_path=args.output,
        metadata_path=args.metadata_output,
        portion_output_path=args.portion_output or None,
        portion_size=args.portion_size,
    )


if __name__ == "__main__":
    main()
