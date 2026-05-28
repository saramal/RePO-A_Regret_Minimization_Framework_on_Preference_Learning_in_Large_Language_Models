import argparse
import math
import os
from datetime import datetime

# import torch
from tqdm import tqdm
import json


from openrlhf.datasets import BenchmarkDataset
from openrlhf.utils import extract_last_answer, extract_all_answers_not_boxed, match_with_answer_labels_v3, match_with_answer_labels_not_boxed

from datasets import load_dataset

from evaluation_scripts.evaluation.eval.eval_utils import math_equal

import torch.multiprocessing as mp


STOP_WORDS = {
    "qwen2": ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:", "<|end|>"],
    "llama": ["</s>", "<|endoftext|>", "\n\nQuestion:"],
} 

def eval(args):
    # configure strategy
    # strategy = get_strategy(args)
    # strategy.setup_distributed()
    # configure strategy
    class Empty:
        pass
    strategy = Empty()
    strategy.print = print
    strategy.is_rank_0 = lambda: True
    strategy.args = args

    strategy.print(f"Evaluating {args.pretrain} with {args.prompt_type} prompt type")
    
    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    # configure model
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=True,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        stop=STOP_WORDS["qwen2" if "qwen2" in args.pretrain.lower() else "llama"],
        stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.pretrain.lower()
                        else None
                    ),
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=True,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )
    
    # loop for all benchmarks
    # from pathlib import Path
    # path = Path(args.dataset)
    
    # ignore_benchmark_list = [""]
    # for benchmark_file in path.glob("*.jsonl"):
    
    benchmark_name = "stepdpo_eval"
    eval_data = load_dataset("json", data_files=args.dataset)["train"]
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    
    print(f"Loaded {len(eval_data)} samples from {args.dataset} for evaluation.")
    # prepare dataset
    eval_dataset = BenchmarkDataset(
        eval_data,
        tokenizer,
        strategy,
        input_template=None,
        prompt_type=None,
    )

    strategy.print(f"Prepared evaluation dataset with {eval_dataset.__len__()} samples.")
    # prepare dataloader

    # eval_dataloader = strategy.setup_dataloader(
    #     eval_dataset,
    #     args.micro_eval_batch_size,
    #     True,
    #     False,
    #     eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    # )

    # strategy.print(f"Prepared evaluation dataloader with {len(eval_dataloader)} batches.")



    # # load checkpoint
    # consumed_samples = 0
    # if args.load_checkpoint and os.path.exists(args.ckpt_path):
    #     _, states = strategy.load_ckpt(model.model, args.ckpt_path)
    #     consumed_samples = states["consumed_samples"]
    #     strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
    # else:
    #     strategy.print(f"Checkpoint not found: {args.ckpt_path}, start from scratch.")

    

    # evaluate
    strategy.print(f"Start evaluation for {benchmark_name}...")
    evaluate(llm, sampling_params, tokenizer, strategy, eval_dataset, benchmark_name)
    


def evaluate(llm, sampling_params, tokenizer, strategy, eval_dataset, benchmark_name: str):
    import time, datetime
    start_time = time.time()
    times = 0
    acc = 0

    os.makedirs(strategy.args.generation_log_path, exist_ok=True)
    save_path = os.path.join(strategy.args.generation_log_path, f"{benchmark_name}_eval.jsonl")

    
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(f"args: \n\n{strategy.args}\n") + "\n")
        
    
    data_ids = []
    prompts = []
    labels = []
    for data_id, prompt, label in eval_dataset:    
        data_ids.append(data_id)
        prompts.append(prompt)
        labels.append(label)
    
    output_dataset = []
    output_list = []
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time()
    # import pdb
    # pdb.set_trace()
    for output in outputs:
        prompt = output.prompt
        output = output.outputs[0].text
        output_dataset.append({"input": prompt, "output": output})
        output_list.append(output)
        
    with open(save_path, 'a', encoding="utf-8") as f:
        for data_id, prompt, generation, answer in zip(data_ids, prompts, output_list, labels):
            generation_dict = {"data_id": data_id, "prompt": prompt, "generation": generation, 
                                #"extracted_answers": extract_first_numeric_answer(generation, strategy.args.answer_trigger),
                                "extracted_last_answers": extract_last_answer(generation, strategy.args.answer_trigger),
                                "extracted_all_answers": extract_all_answers_not_boxed(generation, strategy.args.answer_trigger),
                                "gold_answers": answer,
                                "is_correct": bool(match_with_answer_labels_not_boxed([generation], [answer], answer_trigger=strategy.args.answer_trigger))}
            f.write(json.dumps(generation_dict, ensure_ascii=False) + "\n")

            # acc += match_with_answer_labels(tokenized_output, answers, answer_trigger=strategy.args.answer_trigger)
            # acc += match_with_answer_labels_v3(outputs, labels, answer_trigger=strategy.args.answer_trigger)
    evaluation_time = time.time()
    acc = match_with_answer_labels_not_boxed(output_list, labels, answer_trigger=strategy.args.answer_trigger)
    strategy.print(f"Evaluation finished, accuracy: {acc:.4f}")
    with open(save_path, 'a', encoding="utf-8") as f:
        f.write(json.dumps(f"Evaluation finished, accuracy: {acc:.4f}, generation time: {generation_time - start_time:.2f}s, evaluation time: {evaluation_time - generation_time:.2f}s\n") + "\n")





if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    
    # VLLM
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    
    # Checkpoint
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--micro_eval_batch_size", type=int, default=1, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # SFT

    parser.add_argument("--pretrain", type=str, default=None)

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--save_merged", type=bool, default=True)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")

    parser.add_argument("--input_key", type=str, default="prompt", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--answer_key", type=str, default="answer_label", help="JSON dataset key")
    parser.add_argument("--data_id_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--answer_trigger", type=str, default="The answer is:", help="Trigger for answer extraction")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default=None)
    parser.add_argument(
        "--aRePOy_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # Generation configs
    # parser.add_argument("--max_len", type=int, default=512)
    # parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty for generation")
    parser.add_argument("--generation_log_path", type=str, default="./generation_logs/evals/RePO")
    
    
    
    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    args.generation_log_path = args.generation_log_path + f"_{datetime.now().strftime('%m%dT%H%M')}"

    if args.multiturn:
        assert args.aRePOy_chat_template, "aRePOy_chat_template must be enabled when using multiturn format"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )
        args.input_template = args.input_template.encode().decode('unicode_escape')

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    eval(args)
