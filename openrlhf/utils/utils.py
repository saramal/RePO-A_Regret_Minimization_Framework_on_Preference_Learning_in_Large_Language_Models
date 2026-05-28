import os
import ray
import re
from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True, special_token_list: list=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    if special_token_list:
        strategy.print(f"Add special tokens specified: {special_token_list}")
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": special_token_list
            },
            replace_additional_special_tokens=False,
        )
        model.resize_token_embeddings(len(tokenizer))
    
    # tokenizer.add_special_tokens({"additional_special_tokens": ["ки"]})

    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy



# # TODO: AFTER
# def blending_datasets(
#     datasets,
#     probabilities=None,
#     strategy=None,
#     seed=42,
#     max_count=1e8,
#     stopping_strategy="all_exhausted",
#     dataset_split="train",
# ):
#     """Blend multiple datasets with optional probability sampling.

#     Args:
#         datasets (str): Comma-separated list of dataset paths
#         probabilities (str, optional): Comma-separated list of probabilities for sampling.
#             If None, datasets will be concatenated without probability sampling.
#         strategy: Training strategy object
#         seed (int): Random seed
#         max_count (int): Maximum number of samples per dataset
#     """
#     datasets = datasets.split(",")
#     if probabilities is not None:
#         probabilities = list(map(float, probabilities.split(",")))
#         assert len(probabilities) == len(datasets)

#     data_list = []
#     for i, dataset in enumerate(datasets):
#         dataset = dataset.strip()
#         strategy.print(f"dataset: {dataset}")

#         data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
#         dataset = dataset.split("@")[0].strip()
#         dataset_basename = os.path.basename(dataset)

#         ext = os.path.splitext(dataset)[-1]
#         # local python script
#         if ext == ".py" or (
#             os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
#         ):
#             data = load_dataset(dataset, trust_remote_code=True)
#             strategy.print(f"loaded {dataset} with python script")
#         # local text file
#         elif ext in [".json", ".jsonl", ".csv", ".parquet"]:
#             ext = ext.lower().strip(".")
#             if ext == "jsonl":
#                 ext = "json"
#             data = load_dataset(ext, data_files=dataset)
#             strategy.print(f"loaded {dataset} with data_files={dataset}")
#         # local dataset saved with `datasets.Dataset.save_to_disk`
#         elif os.path.isdir(dataset):
#             try:
#                 data = load_from_disk(dataset)
#                 strategy.print(f"loaded {dataset} from disk")
#             except Exception as e:
#                 strategy.print(f"failed to load {dataset} from disk: {e}")
#                 data = load_dataset(dataset, data_dir=data_dir)
#                 strategy.print(f"loaded {dataset} from files")
#         # remote/local folder or common file
#         elif strategy.args.use_ms:
#             from modelscope.msdatasets import MsDataset

#             namespace, dataset = dataset.split("/")
#             data = MsDataset.load(dataset, namespace=namespace)
#         else:
#             data = load_dataset(dataset, data_dir=data_dir)
#             strategy.print(f"loaded {dataset} from files")

#         # Select dataset
#         if dataset_split and dataset_split in data:
#             data = data[dataset_split]
#         data = data.select(range(min(max_count, len(data))))
#         data_list.append(data)

#     # merge datasets
#     if strategy.is_rank_0():
#         print(data_list)

#     # If probabilities is None, concatenate datasets directly
#     if probabilities is None:
#         from datasets import concatenate_datasets

#         dataset = concatenate_datasets(data_list)
#     else:
#         dataset = interleave_datasets(
#             data_list,
#             probabilities=probabilities,
#             seed=seed,
#             stopping_strategy=stopping_strategy,
#         )

#     return dataset

# TODO: BEFORE
def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
    split_ratio=0.01,
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    is_combined_datapath = all(any(split in dst for dst in datasets) for split in (train_split, eval_split))
    if is_combined_datapath:
        strategy.print("\ntrain, eval data path is given:\n")
        assert len(probabilities) == len(datasets) // 2
    else:
        assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            # _, data = data.train_test_split(test_size=0.7)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        elif strategy.args.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset = dataset.split("/")
            data = MsDataset.load(dataset, namespace=namespace)
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        # split dataset without overlap
        # import pdb
        # pdb.set_trace()
        if return_eval:
            if not is_combined_datapath and train_split and train_split in data:
                if eval_split and eval_split in data:
                    train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
                    eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
                else:
                    # Split train dataset into two portion
                    eval_data_len = int(min(max_count, len(data[train_split])) * split_ratio)
                    train_data_len = min(max_count, len(data[train_split])) - eval_data_len
                    
                    # print(f"train_data_len : {train_data_len}, eval_data_len : {eval_data_len}")
                    # print(f"max_data_len : {min(max_count, len(data[train_split]))}")
                    # print(f"full_data_len : {len(data[train_split])}")
                    # print(f"train_data_idxs : {[id for id in range(train_data_len)]}")
                    # print(f"eval_data_idxs : {[id for id in range(train_data_len, train_data_len + eval_data_len)]}")
                    train_data = data[train_split].select(range(train_data_len))
                    eval_data = data[train_split].select(range(train_data_len, train_data_len + eval_data_len))
            else:
                # if two datasets are given:
                if is_combined_datapath:
                    if train_split in dataset:
                        train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
                        eval_data = None
                    elif eval_split in dataset:
                        train_data = None
                        if eval_split and eval_split in data:
                            eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
                        else:
                            eval_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
                    else:
                        raise ValueError("some of dataset's names is neither 'train' nor 'eval'")
                else:
                    # Split full dataset into two portion
                    eval_data_len = int(min(max_count, len(data["train"])) * split_ratio)
                    train_data_len = min(max_count, len(data["train"])) - eval_data_len
                    
                    train_data = data["train"].select(range(train_data_len))
                    eval_data = data["train"].select(range(train_data_len, train_data_len + eval_data_len))


            if train_data is not None: train_data_list.append(train_data) 
            if eval_data is not None : eval_data_list.append(eval_data)
            
        # only return train dataset
        else:
            if train_split and train_split in data:
                train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
            else:
                train_data = data["train"].select(range(min(max_count, len(data["train"]))))
            train_data_list.append(train_data)

        # if train_split and train_split in data:
        #     train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        # else:
        #     train_data = data.select(range(min(max_count, len(data))))
        #     train_data_last_idx = min(max_count, len(data))
        
        # train_data_list.append(train_data)

        # if return_eval:
        #     if eval_split and eval_split in data:
        #         eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
        #     # train will contains eval? TODO
        #     else:
        #         eval_data = train_data.select(range(train_data_last_idx, min(max_count, int(len(train_data) * 0.03))))
        #     eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(f"train_data:{train_data_list}")
        print(f"test_data:{eval_data_list}")

    if probabilities is None or len(probabilities) == 1:
        train_dataset = train_data_list[0]
    else:
        train_dataset = interleave_datasets(
            train_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    if return_eval:
        if probabilities is None or len(probabilities) == 1:
            eval_dataset = eval_data_list[0]
        else:
            eval_dataset = interleave_datasets(
                eval_data_list,
                probabilities=probabilities,
                seed=seed,
                stopping_strategy=stopping_strategy,
            )
        return train_dataset, eval_dataset
    else:
        return train_dataset, None


def convert_token_to_id(token, tokenizer):
    try:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(token_ids) == 1
        return token_ids[0]
    
    except:

        tokenizer.add_tokens([token], special_tokens=True)
        if isinstance(token, str):
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            assert len(token_ids) == 1
            return token_ids[0]
        else:
            raise ValueError("token should be int or str")


def extract_first_numeric_answer(text):
    matches = []

    # pattern 1: \(\\boxed{ANSWER}\)
    match1 = re.search(r'\\\(\\boxed\{(.*?)\}\\\)', text)
    if match1:
        matches.append(('boxed', match1.start(), match1.group(1).strip()))

    # pattern 2: Therefore, the answer is: ANSWER.
    match2 = re.search(r'Therefore, the answer is: ([^\.\n]+)', text)
    if match2:
        matches.append(('therefore', match2.start(), match2.group(1).strip()))

    if not matches:
        return None

    # extract first answer
    first = min(matches, key=lambda x: x[1])
    answer_text = first[2]

    # float / int extract
    num_match = re.search(r'\d+(?:\.\d+)?', answer_text)
    return float(num_match.group()) if num_match else None


## match answer include non-numeric answer

def extract_last_answer(text: str, answer_trigger: str):
    """
    Extracts the last expression following the answer_trigger in the text.
    """
    pattern = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches = re.findall(pattern, text)
    if matches:
        last_answer = matches[-1].strip()
        if last_answer:
            return last_answer
    
    # 2. Fallback to \\boxed{...} with nested {} support
    # Match \boxed{...} where ... may include nested {...}
    pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    matches_boxed = re.findall(pattern_boxed, text)
    if matches_boxed:
        # Unwrap double braces if present (e.g., {{...}})
        boxed_content = matches_boxed[-1].strip()
        if boxed_content.startswith('{') and boxed_content.endswith('}'):
            boxed_content = boxed_content[1:-1].strip()
        return boxed_content

    # Nothing found
    return None
import re

def extract_all_answers(text: str, answer_trigger: str):
    """
    Extracts all expressions following the answer_trigger and all \boxed{...} contents.
    Returns a list of all matches.
    """
    results = []

    # 1. Find all matches after answer_trigger
    pattern_trigger = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches_trigger = re.findall(pattern_trigger, text)
    for match in matches_trigger:
        val = match.strip()
        if val:
            results.append(val)

    # 2. Find all \boxed{...} matches with nested {} support
    pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    matches_boxed = re.findall(pattern_boxed, text)
    for boxed_content in matches_boxed:
        boxed_content = boxed_content.strip()
        # Unwrap double braces if present (e.g., {{...}})
        if boxed_content.startswith('{') and boxed_content.endswith('}'):
            boxed_content = boxed_content[1:-1].strip()
        if boxed_content:
            results.append(boxed_content)

    # 3. Return unique results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)

    return unique_results if unique_results else None

def extract_all_answers_not_boxed(text: str, answer_trigger: str):
    """
    Extracts all expressions following the answer_trigger and all \boxed{...} contents.
    Returns a list of all matches.
    """
    results = []

    # 1. Find all matches after answer_trigger
    pattern_trigger = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches_trigger = re.findall(pattern_trigger, text)
    for match in matches_trigger:
        val = match.strip()
        if val:
            results.append(val)

    # # 2. Find all \boxed{...} matches with nested {} support
    # pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    # matches_boxed = re.findall(pattern_boxed, text)
    # for boxed_content in matches_boxed:
    #     boxed_content = boxed_content.strip()
    #     # Unwrap double braces if present (e.g., {{...}})
    #     if boxed_content.startswith('{') and boxed_content.endswith('}'):
    #         boxed_content = boxed_content[1:-1].strip()
    #     if boxed_content:
    #         results.append(boxed_content)

    # 3. Return unique results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)

    return unique_results if unique_results else None

def normalize_expression(expr: str) -> str:
    """
    Normalize expressions by removing whitespace and wrapping quotes.
    """
    if expr is None:
        return None
    return re.sub(r'\s+', '', expr).strip("'\"")


def try_parse_float(expr: str):
    """
    Tries to convert an expression to float. Returns None if not possible.
    """
    try:
        return float(expr)
    except ValueError:
        return None


def is_answer_equal(predicted: str, target: str) -> bool:
    import math
    """
    Compares predicted and target answers:
    - If both are numeric, compare numerically
    - Otherwise, compare after normalizing expressions
    """
    pred_norm = normalize_expression(predicted)
    target_norm = normalize_expression(target)

    pred_num = try_parse_float(pred_norm)
    target_num = try_parse_float(target_norm)

    if pred_num is not None and target_num is not None:
        return math.isclose(pred_num, target_num, rel_tol=1e-6)  

    return pred_norm == target_norm


def match_with_answer_labels_v2(tokenized_output, answers, answer_trigger):
        correct_count = 0
        valid_count = 0
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                predicted_answer = extract_last_answer(output, answer_trigger)
                if predicted_answer is None:
                    continue
                is_correct = is_answer_equal(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1
        
        return correct_count/valid_count if valid_count > 0 else 0
    
def match_with_answer_labels_v3(tokenized_output, answers, answer_trigger):
        correct_count = 0
        valid_count = 0
        extracted_answer = None
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                # predicted_answer = extract_last_answer(output, answer_trigger)
                predicted_answers = extract_all_answers(output, answer_trigger)
                if predicted_answers is None:
                    continue
                is_correct = False
                # Check if any of the predicted answers match the target answer
                for predicted_answer in predicted_answers:
                    if predicted_answer is None:
                        continue
                    if is_answer_equal(predicted_answer, answer):
                        is_correct = True
                        extracted_answer = predicted_answer
                        break
                # is_correct = is_answer_equal(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1
        
        accuracy = correct_count/valid_count if valid_count > 0 else 0
        return accuracy


def match_with_answer_labels_v4(tokenized_output, answers, answer_trigger):
    from evaluation_scripts.evaluation.eval.eval_utils import math_equal

    correct_count = 0
    valid_count = 0
    extracted_answer = None
    for output, answer in zip(tokenized_output, answers):
        if answer is not None:
            valid_count += 1
            # predicted_answer = self.extract_answer(output, answer_trigger)
            # predicted_answer = extract_last_answer(output, answer_trigger)
            predicted_answers = extract_all_answers(output, answer_trigger)
            if predicted_answers is None:
                continue
            is_correct = False
            # Check if any of the predicted answers match the target answer
            for predicted_answer in predicted_answers:
                if predicted_answer is None:
                    continue
                if is_answer_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
                elif math_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
            # is_correct = is_answer_equal(predicted_answer, answer)
            correct_count += is_correct
            
    
    accuracy = correct_count/valid_count if valid_count > 0 else 0
    return accuracy

def most_frequent_answer(string_list: list[str]) -> str:
    return max(set(string_list), key=string_list.count)


def match_with_answer_labels_v5_majvote(tokenized_output, answer, answer_trigger):
    from evaluation_scripts.evaluation.eval.eval_utils import math_equal

    predicted_answer_list = []
    for output in tokenized_output:
        if answer is not None:
            predicted_answer = extract_last_answer(output, answer_trigger)
            if predicted_answer is None:
                continue
            predicted_answer_list.append(predicted_answer)

    if len(predicted_answer_list) == 0:
        return 0

    maj_vote_answer = most_frequent_answer(predicted_answer_list)
    if is_answer_equal(maj_vote_answer, answer) or math_equal(maj_vote_answer, answer):
        return 1
    return 0




def match_with_answer_labels_not_boxed(tokenized_output, answers, answer_trigger):
    from evaluation_scripts.evaluation.eval.eval_utils import math_equal

    correct_count = 0
    valid_count = 0
    extracted_answer = None
    for output, answer in zip(tokenized_output, answers):
        if answer is not None:
            valid_count += 1
            # predicted_answer = self.extract_answer(output, answer_trigger)
            # predicted_answer = extract_last_answer(output, answer_trigger)
            predicted_answers = extract_all_answers_not_boxed(output, answer_trigger)
            if predicted_answers is None:
                continue
            is_correct = False
            # Check if any of the predicted answers match the target answer
            for predicted_answer in predicted_answers:
                if predicted_answer is None:
                    continue
                if is_answer_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
                elif math_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
            # is_correct = is_answer_equal(predicted_answer, answer)
            correct_count += is_correct
            
    
    accuracy = correct_count/valid_count if valid_count > 0 else 0
    return accuracy