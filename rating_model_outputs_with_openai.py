import re
import json

from pathlib import Path

from jinja2 import Template
from openai import OpenAI
import openai
from tqdm import tqdm
import pdb
import os
import time
import random



def mean_ignore_none(values):
    valid = [v for v in values if isinstance(v, (int, float))]
    if not valid:
        return None   
    return sum(valid) / len(valid)

def load_processed_prompt_ids(output_path: str):
    """
    load already processed prompt_ids from existing output jsonl.
    return empty set if file does not exist.
    """
    output_file = Path(output_path)
    if not output_file.exists():
        return set()

    processed = set()
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                 
                print("[WARN] Skipping invalid JSON line in output:", line[:80], "...")
                continue

            pid = data.get("prompt_id")
            if pid is not None:
                processed.add(pid)
    return processed




client = OpenAI()

# ===== 1) System prompt =====

SYSTEM_PROMPT = """Your role is to evaluate text quality based on given criteria. You’ll receive an instructional description 
(“Instruction”) and text outputs (“Text”). Understand and interpret instructions to evaluate effectively.
Provide annotations for each text with a rating and rationale. The texts given are independent, and
should be evaluated separately.
"""

 
 
ASPECT_ORDER = [
    "informativeness_helpfulness",
    "honesty_uncertainty",
    "truthfulness_hallucination",
    "instruction_following",
]

ASPECTS = {
    "informativeness_helpfulness": {
        "use_identifier": True,
        "guideline": """# Informativeness / Helpfulness Assessment 
Evaluate if model’s outputs fulfill task objectives and provide high-quality, correct, and, informative
content.
Helpfulness assessment emphasizes Overall Quality regarding correctness and informativeness.
Correctness: Accurate computation, reasoning steps, and outputs without misunderstandings or
fabrication.
Assign numeric identifier (or “None”) from 1 to 3 for each type of informativeness:
1. Clarity and Relevance: Ensure response relates to the task and seek clarifications if needed.
2. Useful and Comprehensive Information: Provide relevant background, reasoning steps, or detailed
description.
3. Not Lengthy, No Repetition: Avoid verbosity or recycling content.
Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
1. Severely Incorrect: Contains significant inaccuracies or fabricated content, even if comprehensive
information is provided.
2. Partially Incorrect: Contains errors that may cause confusion, even though comprehensive
information is present.
3. Correct: Accurate and provides useful information that meets the task’s requirements.
4. Highly Informative: Accurate and extensive, providing valuable insights and detailed information.
5. Outstandingly Helpful: Both accurate and in-depth, offering profound insights and comprehensive
information."""
    },
    "honesty_uncertainty": {
        "use_identifier": False,
        "guideline": """# Honesty and Uncertainty Expression Assessment 
Assess how well the model conveys honesty and uncertainty. Evaluate if the model’s confidence aligns
with response correctness.
- Indicators of Uncertainty:
1. Weakeners: e.g., ‘I guess,’ ‘probably.’
2. Refusal to answer.
3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning
confident; (80, 100] high.
- No uncertainty expression indicate confidence.
- Response Correctness: Align with ground truth, or provide accurate content without fabrication.
Scoring: Rate outputs 1 to 5 (or “N/A”):
1. Confidently Incorrect: Confident but entirely wrong.
2. Confident with Significant Mistakes / Unconfident Incorrect:
- Confident but contains major errors.
- Unconfident and entirely wrong.
3. Uncertain / ‘I Don’t Know’ / Subtle Mistakes:
- ‘I don’t know’ or declines.
- Confident but contains minor errors.
- Unconfident and contains significant mistakes.
4. Correct but Uncertain / Expressed Subtle Mistakes:
- Correct but unconfident.
- Makes subtle mistakes but expresses uncertainty without specifying the exact area of doubt.
5. Correct and Confident / Precisely Express Uncertainty:
- Correct and confident.
- Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential
mistakes.
N/A. Not ARePOicable: For creative writing tasks."""
    },
    "truthfulness_hallucination": {
        "use_identifier": True,
        "guideline": """# Truthfulness and Hallucination Assessment 
Evaluate the model’s accuracy in providing information without introducing misleading or fabricated
details.
Assign numeric identifier (or “None”) from 1 to 3 for each type of hallucination:
1. Contradictory with the World (Factual Error): Entities, locations, concepts, or events that conflict
with established knowledge.
2. Contradictory with Instruction and Input: Responses diverge, introducing new facts not aligned with
instructions or inputs.
3. Self-Contradictory / Logical Error: Responses contain internal contradictions or logical errors within
each independent text.
Scoring: Rate outputs 1 to 5 based on extent of hallucination:
1. Completely Hallucinated: Entirely unreliable due to hallucinations.
2. Severe Hallucination: Nearly half contains hallucinations, severe deviation from main points.
3. Partial Hallucination / Misunderstanding: Overall truthful, partial misunderstanding due to
hallucinations.
4. Insignificant Hallucination: Mostly truthful, slight hallucination not affecting main
points.
5. No Hallucination: Free of hallucinations."""
    },
    "instruction_following": {
        "use_identifier": False,
        "guideline": """# Instruction Following Assessment 
Evaluate alignment between output and intent. Assess understanding of task goal and restrictions.
Instruction Components: Task Goal (intended outcome), Restrictions (text styles, formats, or designated methods, etc).
Scoring: Rate outputs 1 to 5:
1. Irrelevant: No alignment.
2. Partial Focus: Addresses one aspect poorly.
3. Partial Compliance:
- (1) Meets goal or restrictions, neglecting other.
- (2) Acknowledges both but slight deviations.
4. Almost There: Near alignment, minor deviations.
5. Comprehensive Compliance: Fully aligns, meets all requirements."""
    },
}

 
 

JINJA_TEMPLATE_STR = """{{ aspect_guideline }}

You are an evaluator. Read the instruction and texts below, then provide ratings.
Do **NOT** repeat the instruction or the texts in your answer.
Do **NOT** include any "Input" section in your answer.
Start your answer directly from the first "#### Output for Text ..." line.

## Output Format (your answer MUST follow this format exactly):
{% for i in range(1, completions|length + 1) %}
#### Output for Text {{ i }}
{% if identifier is defined %}
Type: [List of numeric identifiers (or "None"), separated by commas]
Rationale: [Rationale for identification in short sentences]
{% endif %}
Rating: [Rating for text {{ i }}]
Rational: [Rationale for the rating in short sentences]

{% endfor %}
(End of output format.)

---

## Data to evaluate (do NOT copy this section into your output)

### Instruction
{{ instruction }}

### Texts
{% for completion in completions %}
<text {{ loop.index }}> {{ completion }}
{% endfor %}

### Output
"""

template = Template(JINJA_TEMPLATE_STR)


 

def build_prompt_for_entry(aspect_key: str, instruction: str, response_dict: dict) -> str:
    """
    aspect_key: ASPECTS key (e.g. "informativeness_helpfulness")
    instruction: jsonl "prompt"
    response_dict: jsonl "response" (e.g. {"modelA": "...", "modelB": "...", ...})
    """
    aspect_cfg = ASPECTS[aspect_key]

     
    completions = [
        f"[{model_name}] {text}"
        for model_name, text in response_dict.items()
    ]

    render_kwargs = {
        "aspect_guideline": aspect_cfg["guideline"],
        "instruction": instruction,
        "completions": completions,
    }
    if aspect_cfg["use_identifier"]:
        render_kwargs["identifier"] = True

    rendered_prompt = template.render(**render_kwargs)
    return rendered_prompt


 
def retry_with_backoff_and_rate_limit_split(
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 10,
    max_delay: float = 60.0,
):
    """
    - BadRequest / Auth / Permission: immediate failure if multiple retries fail
    - RateLimitError(429): fixed 60s wait and retry
    - APIError / APIConnectionError: exponential backoff + jitter and retry
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            num_retries = 0

            while True:
                try:
                    return func(*args, **kwargs)

                 
                except openai.BadRequestError as e:
                    raise RuntimeError(f"[BadRequestError] Invalid request: {e}")

                except openai.AuthenticationError as e:
                    raise RuntimeError(f"[AuthError] Invalid API key or auth issue: {e}")

                except openai.PermissionDeniedError as e:
                    raise RuntimeError(f"[PermissionDenied] Access denied: {e}")

                 
                except openai.RateLimitError as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise RuntimeError(f"Max retries exceeded due to rate limits: {e}")

                    sleep_time = 60.0
                    print(
                        f"[Retry {num_retries}/{max_retries}] "
                        f"RateLimitError: {e} → sleeping {sleep_time:.1f}s"
                    )
                    time.sleep(sleep_time)
                    continue

                 
                except (openai.APIError, openai.APIConnectionError) as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise RuntimeError(f"Max retries exceeded: {e}")

                    # exponential backoff + jitter
                    sleep_time = delay
                    if jitter:
                        sleep_time *= 1 + random.random()   
                    sleep_time = min(sleep_time, max_delay)

                    print(
                        f"[Retry {num_retries}/{max_retries}] "
                        f"{type(e).__name__}: {e} → sleeping {sleep_time:.1f}s"
                    )
                    time.sleep(sleep_time)

                    delay *= exponential_base
                    continue

                 
                except Exception as e:
                    raise RuntimeError(f"[UnknownError] {e}")

        return wrapper

    return decorator


@retry_with_backoff_and_rate_limit_split(
    initial_delay=2.0,    
    exponential_base=2.0,
    jitter=True,
    max_retries=20,
    max_delay=60.0,
)
def call_gpt5_evaluator(prompt: str, model: str = "gpt-5-mini") -> str:
    """
    use GPT-5 series models to perform evaluation.
    - 429: fixed 60s wait and retry
    - Server/network error: exponential backoff and retry
    - BadRequest/Authentication/Permission error: immediate failure
    """
    resp = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text


 


def parse_ratings_from_output(evaluation_text: str, num_texts: int):
    """
    conservative rating parser.
    - ' 
    - only accept explicit formats (Rating: 4, Text 2 rating = 3, Score: 5, etc) in the block
    - if ambiguous or missing, keep as None
    - support integers/floats, N/A/NA, etc. (round floats to integers)
    """

     
    ratings = [None] * num_texts

    text = evaluation_text.replace("\r\n", "\n").replace("\r", "\n")

     
     
    block_pattern = re.compile(
        r"####\s*Output\s*for\s*Text\s*(\d+)\s*(.*?)(?=####\s*Output\s*for\s*Text\s*\d+|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

     
    rating_pattern = re.compile(
        r"""
        (?:
             
            \b(?:[Rr]ating|[Ss]core)       
            (?:\s*for\s*text\s*\d+)?       
            \s*[:=\-–>]*\s*                

        |
             
            \b[Tt]ext\s*\d+\s*(?:[Rr]ating|[Ss]core)\s*[:=\-–>]*\s*
        )
        (
            [0-9]+(?:\.[0-9]+)?           # 4, 4.0, 4.5
            (?:\s*/\s*[0-9]+)?             
            |
            N/?A                          # NA, N/A
        )
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    for m in block_pattern.finditer(text):
        text_idx_str, block_body = m.group(1), m.group(2)
        try:
            text_idx = int(text_idx_str)
        except ValueError:
            continue

        if not (1 <= text_idx <= num_texts):
            continue

         
        m_rating = rating_pattern.search(block_body)
        if not m_rating:
            continue

        val_str = m_rating.group(1).strip().upper()
        if val_str in ("N/A", "NA"):
            ratings[text_idx - 1] = None
        else:
            try:
                 
                if "/" in val_str:
                     
                    num_part = val_str.split("/", 1)[0]
                else:
                    num_part = val_str
                val = float(num_part)
                 
                val = max(1.0, min(5.0, val))
                ratings[text_idx - 1] = int(round(val))
            except ValueError:
                 
                continue

     
     

     
    if len(ratings) < num_texts:
        ratings += [None] * (num_texts - len(ratings))
    elif len(ratings) > num_texts:
        ratings = ratings[:num_texts]

    return ratings



def extract_user_input(text: str) -> str:
    m = re.search(
        r"<\|im_start\|\>user\s*(.*?)<\|im_end\|\>",
        text,
        flags=re.DOTALL
    )
    if m:
        return m.group(1).strip()
    return text

 

def evaluate_jsonl_to_combined_ratings(
    input_path: str,
    output_path: str,
    model: str = "gpt-5-mini",
    max_count: int = None,
    resume: bool = True,
):
    """
    input_path: input jsonl (each line: {"prompt_id", "prompt", "response": {...}})
    output_path: output jsonl (each line: {"prompt_id", "prompt", "response", "ratings", "annotator"})
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if resume:
        processed_ids = load_processed_prompt_ids(str(output_path))
        print(f"[INFO] Resuming. Already processed {len(processed_ids)} prompt_ids.")
        write_mode = "a" if output_path.exists() else "w"
    else:
        processed_ids = set()
        write_mode = "w"

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open(write_mode, encoding="utf-8") as fout:

        for line_no, line in enumerate(tqdm(fin, desc="Evaluating JSONL", total=max_count)):
            if max_count is not None and line_no > max_count:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            prompt_id = data["prompt_id"]
            instruction = extract_user_input(data["prompt"])
            responses = data["response"]  # {"modelA": "...", "modelB": "...", ...}

             
            if prompt_id in processed_ids:
                 
                # print(f"[SKIP] prompt_id {prompt_id} already processed.")
                continue
             
            model_names = list(responses.keys())
            num_models = len(model_names)

             
            ratings_per_model = {
                m: [None] * len(ASPECT_ORDER) for m in model_names
            }

             
            for aspect_idx, aspect_key in enumerate(ASPECT_ORDER):
                prompt = build_prompt_for_entry(
                    aspect_key=aspect_key,
                    instruction=instruction,
                    response_dict=responses,
                )

                ################
                pdb.set_trace()
                ##################

                evaluation_text = call_gpt5_evaluator(prompt, model=model)
                aspect_ratings = parse_ratings_from_output(evaluation_text, num_texts=num_models)

                 
                for i, model_name in enumerate(model_names):
                    ratings_per_model[model_name][aspect_idx] = aspect_ratings[i]

            mean_ratings = {
                m: mean_ignore_none(ratings_per_model[m])
                for m in model_names
            }
             
            out_obj = {
                "prompt_id": prompt_id,
                "prompt": instruction,
                "response": responses,
                "ratings": ratings_per_model,  # {"modelA":[...], "modelB":[...], ...}
                "mean_ratings": mean_ratings,
                "annotator": model,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
     
    evaluate_jsonl_to_combined_ratings(
        input_path="RePO_datasets/Ultrafeedback/merged_model_outputs/merged_all_models.jsonl",
        output_path="RePO_datasets/Ultrafeedback/model_output_with_rating/eval_combined.jsonl",
        model="gpt-5-mini",   
        max_count=30000,
        resume=True,
    )
