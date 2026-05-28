import argparse
import math
import os
import datetime
import glob

# import torch
from tqdm import tqdm
import json

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from openrlhf.datasets import BenchmarkDataset
from openrlhf.utils import extract_last_answer, extract_all_answers, match_with_answer_labels_v3, match_with_answer_labels_v4, match_with_answer_labels_not_boxed
from openrlhf.utils import match_with_answer_labels_v5_majvote
from datasets import load_dataset

from evaluation_scripts.evaluation.eval.eval_utils import math_equal

import torch.multiprocessing as mp



from tabulate import tabulate

STOP_WORDS = {
    "qwen2": ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQuestion:", "<|end|>"],
    "llama": ["</s>", "<|endoftext|>", "\n\nQuestion:"],
} 

def init_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl' , timeout=datetime.timedelta(seconds=1800))
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0

def merge_jsonl_files(output_dir, filename_pattern, merged_filename):
    """Merge multiple JSONL files into one"""
    pattern = os.path.join(output_dir, filename_pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    merged_path = os.path.join(output_dir, merged_filename)
    
    with open(merged_path, 'w') as outfile:
        for file_path in files:
            print(f"Merging {file_path}")
            with open(file_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"Merged {len(files)} files into {merged_path}")
    
    # Clean up individual files
    for file_path in files:
        os.remove(file_path)
        print(f"Removed {file_path}")

def split_data_for_rank(data, rank, world_size):
    """Split data across GPUs"""
    total_samples = len(data)
    samples_per_gpu = total_samples // world_size
    remainder = total_samples % world_size
    
    start_idx = rank * samples_per_gpu + min(rank, remainder)
    if rank < remainder:
        end_idx = start_idx + samples_per_gpu + 1
    else:
        end_idx = start_idx + samples_per_gpu
        
    return data.select(range(start_idx, end_idx))

def eval(args):
    os.environ["TORCH_GEOMETRY_USE_AGENT_STORE"]="False"
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    
    # configure strategy
    class Empty:
        pass
    strategy = Empty()
    strategy.print = print if rank == 0 else lambda *args, **kwargs: None
    strategy.is_rank_0 = lambda: rank == 0
    strategy.args = args

    if rank == 0:
        print(f"Evaluating {args.pretrain} with {args.prompt_type} prompt type")
        print(f"World size: {world_size}, Rank: {rank}")
    
    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    # configure model - use single GPU per process instead of tensor parallel
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=1,  # Changed from args.tp_size to 1
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=True,
        distributed_executor_backend="external_launcher",
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        n=args.num_generations,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
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
    
    # Create output directory on rank 0
    if rank == 0:
        os.makedirs(args.generation_log_path, exist_ok=True)
    
    # Wait for rank 0 to create directory
    if world_size > 1:
        dist.barrier()
    
    # Store global metrics for all benchmarks
    benchmark_global_metrics = {}
    
    # loop for all benchmarks
    from pathlib import Path
    path = Path(args.dataset)
    
    ignore_benchmark_list = [""]
    for benchmark_file in path.glob("*.jsonl"):
        benchmark_name = benchmark_file.stem
        benchmark_name = benchmark_name.strip()
        
        benchmark_file = str(benchmark_file)
        if benchmark_name in ignore_benchmark_list:
            strategy.print(f"Skipping benchmark: {benchmark_name}")
            continue
        
        strategy.print(f"Evaluating benchmark: {benchmark_name}")
        
        # Load full dataset
        eval_data = load_dataset("json", data_files=benchmark_file)["train"]
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        
        if rank == 0:
            print(f"Total dataset size: {len(eval_data)} samples from {benchmark_file}")
        
        # Split data across GPUs
        eval_data_split = split_data_for_rank(eval_data, rank, world_size)
        
        print(f"Rank {rank}: Processing {len(eval_data_split)} samples")
        
        # prepare dataset
        eval_dataset = BenchmarkDataset(
            eval_data_split,
            tokenizer,
            strategy,
            input_template=None,
            prompt_type=args.prompt_type,
        )

        print(f"Rank {rank}: Prepared evaluation dataset with {eval_dataset.__len__()} samples.")

        # evaluate
        print(f"Rank {rank}: Start evaluation for {benchmark_name}...")
        local_metrics = evaluate_distributed(llm, sampling_params, tokenizer, strategy, eval_dataset, benchmark_name, rank)
        
        print(f"[DEBUG] Rank {rank}: evaluate_distributed finished for {benchmark_name}")
        dist.barrier()
        # Gather metrics from all ranks using all_gather_object
        gathered_metrics = [None] * world_size
        if world_size > 1:
            dist.barrier()
            print(f"[DEBUG] Rank {rank}: all_gather_object started")
            
            dist.all_gather_object(gathered_metrics, local_metrics)
            
            print(f"[DEBUG] Rank {rank}: all_gather_object finished")
            
        else:
            gathered_metrics = [local_metrics]
            
        if world_size > 1:
            dist.barrier()
        
        # Calculate global metrics on rank 0
        if rank == 0:
            total_question_num = sum(metrics['question_num'] for metrics in gathered_metrics)
            total_valid_question_num = sum(metrics['valid_question_num'] for metrics in gathered_metrics)
            total_correct_num = sum(metrics['correct_num'] for metrics in gathered_metrics)
            total_generation_time = sum(metrics['generation_time'] for metrics in gathered_metrics)
            total_evaluation_time = sum(metrics['evaluation_time'] for metrics in gathered_metrics)
            
            global_accuracy = total_correct_num / total_question_num if total_question_num > 0 else 0.0
            
            print(f"Global results for {benchmark_name}:")
            print(f"  Total questions: {total_question_num}")
            print(f"  Valid questions: {total_valid_question_num}")
            print(f"  Correct answers: {total_correct_num}")
            print(f"  Global accuracy: {global_accuracy:.4f}")
            print(f"  Total generation time: {total_generation_time:.2f}s")
            print(f"  Total evaluation time: {total_evaluation_time:.2f}s")
            
            # Store global metrics for this benchmark
            benchmark_global_metrics[benchmark_name] = {
                'total_question_num': total_question_num,
                'total_valid_question_num': total_valid_question_num,
                'total_correct_num': total_correct_num,
                'global_accuracy': global_accuracy,
                'total_generation_time': total_generation_time,
                'total_evaluation_time': total_evaluation_time
            }

    
    # Wait for all processes to finish
    if world_size > 1:
        dist.barrier()
    
    # Merge files and add global metrics on rank 0
    if rank == 0:
        print("Merging output files...")
        from pathlib import Path
        path = Path(args.dataset)
        
        for benchmark_file in path.glob("*.jsonl"):
            benchmark_name = benchmark_file.stem.strip()
            if benchmark_name not in ignore_benchmark_list:
                # First merge the files
                merge_jsonl_files(args.generation_log_path, f"{benchmark_name}_eval_rank_*.jsonl", f"{benchmark_name}_eval.jsonl")
                
                # Then add global metrics if available
                if benchmark_name in benchmark_global_metrics:
                    merged_path = os.path.join(args.generation_log_path, f"{benchmark_name}_eval.jsonl")
                    metrics = benchmark_global_metrics[benchmark_name]
                    
                    with open(merged_path, 'a') as f:
                        global_summary = {
                            "global_metrics": metrics
                        }
                        f.write(json.dumps(global_summary) + "\n")
                        f.write(json.dumps(f"Global Evaluation Summary: accuracy: {metrics['global_accuracy']:.4f}, total questions: {metrics['total_question_num']}, correct answers: {metrics['total_correct_num']}, total generation time: {metrics['total_generation_time']:.2f}s, total evaluation time: {metrics['total_evaluation_time']:.2f}s") + "\n")
        
        print("All files merged successfully!")

        # save results to txt

        txt_path = f"evaluation_results_txt/math_results_majvote_{args.num_generations}.txt"

        # sorted benchmark list
        benchmarks = sorted(benchmark_global_metrics.keys())
        columns = ["model_name"] + [f"{b}_accuracy_maj@{args.num_generations}" for b in benchmarks]

        # if file exists, read existing content and keep it as a table
        table = []

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # header is the first line, the rest is data
                # tabulate based table is easy to parse so split
                if len(lines) > 2:
                    # table shape so parse only the data lines
                    for line in lines[2:]:
                        if line.strip().startswith("|"):
                            vals = [v.strip() for v in line.strip().split("|")[1:-1]]
                            table.append(vals)

        # current execution result row
        row = [args.pretrain] + [f"{benchmark_global_metrics[b]['global_accuracy']:.4f}" for b in benchmarks]
        table.append(row)

        # new table
        text_table = tabulate(table, headers=columns, tablefmt="github")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_table)

        print("Saved nicely formatted results to results.txt")
    
    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()
    


def evaluate_distributed(llm, sampling_params, tokenizer, strategy, eval_dataset, benchmark_name: str, rank: int):
    import time, datetime
    start_time = time.time()
    times = 0
    acc = 0

    # Create rank-specific output file
    save_path = os.path.join(strategy.args.generation_log_path, f"{benchmark_name}_eval_rank_{rank}.jsonl")

    
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(f"args: \n\n{strategy.args}\n") + "\n")
        f.write(json.dumps(f"rank: {rank}\n") + "\n")
        
    
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
        one_sample_outputs = [output.outputs[i].text for i in range(len(output.outputs))]
        output_dataset.append({"input": prompt, "output": one_sample_outputs})
        output_list.append(one_sample_outputs)
        
    with open(save_path, 'a', encoding="utf-8") as f:
        for data_id, prompt, generation_list, answer in zip(data_ids, prompts, output_list, labels):
            for generation in generation_list:
                generation_dict = {"data_id": data_id, "prompt": prompt, "generation": generation, 
                                    #"extracted_answers": extract_first_numeric_answer(generation, strategy.args.answer_trigger),
                                    "extracted_last_answers": extract_last_answer(generation, strategy.args.answer_trigger),
                                    "extracted_all_answers": extract_all_answers(generation, strategy.args.answer_trigger),
                                    "gold_answers": answer,
                                    "is_correct": bool(match_with_answer_labels_v5_majvote([generation], answer, answer_trigger=strategy.args.answer_trigger))}
                f.write(json.dumps(generation_dict, ensure_ascii=False) + "\n")

            # acc += match_with_answer_labels(tokenized_output, answers, answer_trigger=strategy.args.answer_trigger)
            # acc += match_with_answer_labels_v3(outputs, labels, answer_trigger=strategy.args.answer_trigger)
    evaluation_time = time.time()
    for one_sample_outputs, label in zip(output_list, labels):
        acc += match_with_answer_labels_v5_majvote(one_sample_outputs, label, answer_trigger=strategy.args.answer_trigger)
    acc = acc / len(output_list)
    question_num = len(labels)
    assert question_num == len(output_list)
    valid_question_num = sum(1 for label in labels if label is not None)
    correct_num = int(acc * question_num)
    print(f"Rank {rank}: Evaluation finished, accuracy: {acc:.4f}, question_num: {question_num}, valid_question_num: {valid_question_num}, correct_num: {correct_num}")
    with open(save_path, 'a', encoding="utf-8") as f:
        f.write(json.dumps(f"Rank {rank}: Evaluation finished, accuracy: {acc:.4f}, question_num: {question_num}, valid_question_num: {valid_question_num}, correct_num: {correct_num}, generation time: {generation_time - start_time:.2f}s, evaluation time: {evaluation_time - generation_time:.2f}s\n") + "\n")
    
    # Return metrics for gathering
    return {
        'question_num': question_num,
        'valid_question_num': valid_question_num,
        'correct_num': correct_num,
        'generation_time': generation_time - start_time,
        'evaluation_time': evaluation_time - generation_time
    }





if __name__ == "__main__":
    os.environ["TORCH_GEOMETRY_USE_AGENT_STORE"]="False"
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    
    # VLLM
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
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

    parser.add_argument("--input_key", type=str, default="question", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--answer_key", type=str, default="answer", help="JSON dataset key")
    parser.add_argument("--data_id_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--answer_trigger", type=str, default="The answer is:", help="Trigger for answer extraction")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--prompt_type", type=str, default="qwen2-boxed")
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
        default="sft_%s" % datetime.datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    args.generation_log_path = args.generation_log_path + f"_{datetime.datetime.now().strftime('%m%dT%H%M')}"

    if args.prompt_type == "None":
        args.prompt_type = None
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
