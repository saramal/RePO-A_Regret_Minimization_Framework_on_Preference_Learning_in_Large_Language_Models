from datasets import load_dataset
import re
import json

# 1. load dataset
dataset = load_dataset("hkust-nlp/dart-math-hard", split="train")

def preprocess(example):
    question = example["query"]

    # extract everything after "The answer is:"
    matches = list(re.finditer(r"The answer is:\s*(.+)", example["response"], flags=re.DOTALL))

    # if there is no match or multiple matches, exclude
    if len(matches) != 1:
        return None

    answer = matches[0].group(1).strip()

    # find the last match position
    last_match = matches[-1]
    start, end = last_match.span(0)

    # replace response (only the last match)
    response_new = (
        example["response"][:start]
        + f"The answer is: \\boxed{{{{{answer}}}}}"
        + example["response"][end:]
    )

    return {
        "question": question,
        "response": response_new,
        "answer": answer,
    }

# 2. aRePOy preprocessing
processed_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 3. filter None
processed_dataset = processed_dataset.filter(lambda x: x is not None)

# 4. re-index
def add_index(example, idx):
    example["index"] = idx + 1  # start from 1
    return example

processed_dataset = processed_dataset.map(add_index, with_indices=True)

# 5. save (jsonl) - specify column order
output_file = "RePO_datasets/dart_math_hard_processed.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in processed_dataset:
        ordered_ex = {
            "index": ex["index"],
            "question": ex["question"],
            "response": ex["response"],
            "answer": ex["answer"],
        }
        line = json.dumps(ordered_ex, ensure_ascii=False)
        line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        f.write(line + "\n")

print(f"✅ Preprocessing complete! {len(processed_dataset)} samples saved to {output_file}")
