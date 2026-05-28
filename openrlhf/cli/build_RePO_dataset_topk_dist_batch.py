import argparse
import math
import os
from datetime import datetime
import json
import glob

from transformers.trainer import get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.models.utils import log_probs_from_logits
from peft import PeftModel

from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

import re
def append_endoftext_if_needed(s: str, eos_token, answer_prefix: str = "The answer is:") -> str:
    """
    This function is designed for step-level RePO dataset.
    If given step is final step, it appends the end of text token.
    Appends the end of text token if the string ends with the answer prefix or the boxed pattern.

    Args:
        s: The string to append the end of text token to.
        eos_token: The end of text token to append.
        answer_prefix: The answer prefix to check for.

    Returns:
        The string with the end of text token appended.
    """

    escaped_prefix = re.escape(answer_prefix)
    answer_pattern = rf"{escaped_prefix} .+\s*$"
    boxed_pattern = r"\\boxed\{[^}]*\}"
    
    if re.search(answer_pattern, s) or re.search(boxed_pattern, s):
        return s.rstrip() + f" {eos_token}"
    else:
        return s
    
def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask,
    prompt_id_lens,
    average_log_prob: bool = False,
    k: int = 10,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        k: Number of top tokens to return log probabilities for.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        A tensor of shape (batch_size, sequence_length, k) containing top-k log probabilities for each position.
    """
    assert average_log_prob == False
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:]
    logits = logits[:, :-1, :].clone()

    loss_masks = attention_mask.clone().bool()
    # mask prompts

    for mask, source_len in zip(loss_masks, prompt_id_lens):
        mask[:source_len-1] = False
    loss_masks = loss_masks[:, :-1]

    # dummy token; we'll ignore the losses on these tokens later
    # labels[loss_masks == False] = 0
    
    # per_token_logps = log_probs_from_logits(logits, labels)
    logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # Calculate log probabilities for all tokens using _logsumexp_by_chunk approach
    
    # log_probs_all = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    
    # This is more memory efficient than torch.log_softmax for large tensors
    batch_dim = logits.shape[:-1]  # (batch_size, seq_len)
    last_dim = logits.shape[-1]    # vocab_size
    
    # Calculate logsumexp values for normalization using chunked approach
    from openrlhf.models.utils import _logsumexp_by_chunk
    logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
    logsumexp_values = logsumexp_values.view(*batch_dim)  # (batch_size, seq_len, 1)
    
    # Calculate log probabilities: log_softmax(x_i) = x_i - logsumexp(x)
    per_token_logps = logits_labels - logsumexp_values
    log_probs_all = logits - logsumexp_values.unsqueeze(-1)  # (batch_size, seq_len, vocab_size)
    
    # Get top-k log probabilities
    top_k_logprobs, top_k_indices = torch.topk(log_probs_all, k, dim=-1)  # (batch_size, seq_len, k)
    
    # Mask top-k logprobs where tokens are padded
    top_k_logprobs_masked = top_k_logprobs.clone()
    # Expand loss_masks to match top_k_logprobs shape
    loss_masks_expanded = loss_masks.unsqueeze(-1).expand(-1, -1, k)  # (batch_size, seq_len, k)
    top_k_logprobs_masked[~loss_masks_expanded] = 0

    per_token_logps = per_token_logps * loss_masks
    logprobs_sums = (per_token_logps * loss_masks).sum(-1)
    logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)

    # restore logps size to original size, fill with zeros at left side.
    per_token_logps = torch.cat([torch.zeros_like(per_token_logps[:, :1]), per_token_logps], dim=1)
    top_k_logprobs_masked = torch.cat([torch.zeros_like(top_k_logprobs_masked[:, :1]), top_k_logprobs_masked], dim=1)
    top_k_indices = torch.cat([torch.zeros_like(top_k_indices[:, :1]), top_k_indices], dim=1)

    return per_token_logps, logprobs_sums, logprobs_means, top_k_logprobs_masked, top_k_indices

def extract_text_from_keys(data, key, input_template=None):
    """Extract text from data using comma-separated keys"""
    if "," in key and len(key.split(",")) > 1:
        text = ""
        for sub_key in key.split(","):
            sub_key = sub_key.strip()
            if input_template is not None and sub_key == "prompt":
                text += input_template.format(data[sub_key])
            else:
                text += data[sub_key]
            text += "\n"
        return text.rstrip("\n")
    else:
        if input_template is not None and key.strip() == "prompt":
            return input_template.format(data[key])
        else:
            return data[key]

def process_response_batch(prompts, responses, model, tokenizer, k):
    """Process a batch of prompt-response pairs"""
    batch_size = len(prompts)
    device = next(model.parameters()).device
    
    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize all prompts and responses
    all_prompt_tokens = []
    all_response_tokens = []
    all_full_tokens = []
    all_prompt_lens = []
    
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt
        prompt_tokens = tokenizer(
            prompt, padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        
        # Tokenize response
        response_tokens = tokenizer(
            response, padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        
        # Concatenate
        full_tokens = torch.cat([prompt_tokens, response_tokens], dim=0)
        
        all_prompt_tokens.append(prompt_tokens)
        all_response_tokens.append(response_tokens)
        all_full_tokens.append(full_tokens)
        all_prompt_lens.append(len(prompt_tokens))
    
    # Pad sequences for batch processing
    padded_tokens = pad_sequence(all_full_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (padded_tokens != tokenizer.pad_token_id).long()
    
    # Move to device
    padded_tokens = padded_tokens.to(device)
    attention_mask = attention_mask.to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(padded_tokens, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Process logits with batch function
    per_token_logps, _, _, top_k_logprobs, top_k_indices = _get_batch_logps(
        logits, padded_tokens, attention_mask, all_prompt_lens, k=k
    )
    
    # Extract results for each sample
    results = []
    for i in range(batch_size):
        response_len = len(all_response_tokens[i])
        
        # Extract response part
        response_logps = per_token_logps[i, -response_len:]
        response_top_k_logprobs = top_k_logprobs[i, -response_len:, :]
        response_top_k_indices = top_k_indices[i, -response_len:, :]
        
        # Create result
        result = {
            'logprob_with_token': {
                "tokens": [tokenizer.decode(tok_id) for tok_id in all_response_tokens[i].tolist()],
                "logprobs": response_logps.tolist()
            },
            'top_k_data': {
                "top_k_tokens": [[tokenizer.decode(idx) for idx in indices.tolist()] 
                               for indices in response_top_k_indices],
                "top_k_logprobs": response_top_k_logprobs.tolist()
            }
        }
        results.append(result)
    
    return results

def validate_reconstruction(result, original_text):
    """Validate that tokens can be reconstructed to original text"""
    reconstructed = "".join(result['logprob_with_token']['tokens'])
    if reconstructed != original_text:
        print(f"Reconstruction mismatch:\nOriginal: {original_text}\nReconstructed: {reconstructed}")
        return False
    return True

def process_batch(batch_data, model_A, tokenizer_A, model_B, tokenizer_B, 
                 prompt_key, chosen_key, rejected_key, input_template, answer_key, k):
    """Process a batch of data samples"""
    batch_results = []
    batch_metadata = []
    
    # Prepare metadata for each sample
    for data in batch_data:
        # Prepare text data for each sample
        chosen_model_type = data.get("chosen_model", None)
        rejected_model_type = data.get("rejected_model", None)

        # Determine which models to use
        if chosen_model_type is not None and "A" in chosen_model_type:
            chosen_model, chosen_tokenizer = model_A, tokenizer_A
            rejected_model, rejected_tokenizer = model_B, tokenizer_B
        elif chosen_model_type is not None and "B" in chosen_model_type:
            chosen_model, chosen_tokenizer = model_B, tokenizer_B
            rejected_model, rejected_tokenizer = model_A, tokenizer_A
        else:
            chosen_model = rejected_model = model_A
            chosen_tokenizer = rejected_tokenizer = tokenizer_A
        
        # Extract and format text
        prompt = extract_text_from_keys(data, prompt_key, input_template)
        chosen = extract_text_from_keys(data, chosen_key, input_template)
        rejected = extract_text_from_keys(data, rejected_key, input_template)
        
        # Append EOS token if needed
        if not chosen.endswith(chosen_tokenizer.eos_token):
            chosen = append_endoftext_if_needed(chosen, chosen_tokenizer.eos_token)
        if not rejected.endswith(rejected_tokenizer.eos_token):
            rejected = append_endoftext_if_needed(rejected, rejected_tokenizer.eos_token)
        
        batch_metadata.append({
            'data': data,
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'chosen_model': chosen_model,
            'rejected_model': rejected_model,
            'chosen_tokenizer': chosen_tokenizer,
            'rejected_tokenizer': rejected_tokenizer,
        })
    
    # Group samples by model type for efficient batching
    model_groups = {}
    for i, meta in enumerate(batch_metadata):
        key = (id(meta['chosen_model']), id(meta['rejected_model']))
        if key not in model_groups:
            model_groups[key] = []
        model_groups[key].append((i, meta))
    
    # Initialize results list with None values
    batch_results = [None] * len(batch_metadata)
    
    # Process each model group
    for (chosen_model_id, rejected_model_id), samples in model_groups.items():
        if not samples:
            continue
            
        # Extract batch data
        batch_prompts = [meta['prompt'] for _, meta in samples]
        batch_chosen = [meta['chosen'] for _, meta in samples]
        batch_rejected = [meta['rejected'] for _, meta in samples]
        
        chosen_model = samples[0][1]['chosen_model']
        rejected_model = samples[0][1]['rejected_model']
        chosen_tokenizer = samples[0][1]['chosen_tokenizer']
        rejected_tokenizer = samples[0][1]['rejected_tokenizer']
        
        # Process chosen responses in batch
        chosen_results = process_response_batch(
            batch_prompts, batch_chosen, chosen_model, chosen_tokenizer, k
        )
        
        # Process rejected responses in batch
        rejected_results = process_response_batch(
            batch_prompts, batch_rejected, rejected_model, rejected_tokenizer, k
        )
        
        # Combine results
        for idx, (sample_idx, meta) in enumerate(samples):
            if chosen_results[idx] is None or rejected_results[idx] is None:
                continue
                
            chosen_result = chosen_results[idx]
            rejected_result = rejected_results[idx]
            
            # Validate reconstruction
            if not validate_reconstruction(chosen_result, meta['chosen']) or \
               not validate_reconstruction(rejected_result, meta['rejected']):
                continue
            
            # Create final result
            result = {
                "prompt": meta['prompt'],
                "chosen": meta['chosen'],
                "rejected": meta['rejected'],
                "chosen_logprob_with_token": chosen_result['logprob_with_token'],
                "rejected_logprob_with_token": rejected_result['logprob_with_token'],
                "chosen_top_k_data": chosen_result['top_k_data'],
                "rejected_top_k_data": rejected_result['top_k_data'],
                "answer_label": meta['data'].get(answer_key, None)
            }
            batch_results[sample_idx] = result
    
    return batch_results

def generate_execute_logprobs(
    dataset,
    save_path,
    model_A,
    tokenizer_A,
    model_B=None,
    tokenizer_B=None,
    prompt_key="prompt",
    chosen_key="chosen",
    rejected_key="rejected",
    input_template=None,
    answer_key="answer",
    k=10,
    batch_size=1,
    ):
    """
    Generate logprobs for the chosen and rejected responses with rollout policy each.

    Args:
        dataset: preference dataset
            dataset: {prompt, chosen, rejected, chosen_model, rejected_model, answer_label}
        model_A: policy model
        tokenizer_A: tokenizer for policy model
        model_B: reference model
        tokenizer_B: tokenizer for reference model
        batch_size: number of samples to process in each batch
    """
    if model_B is None:
        model_B = model_A
        tokenizer_B = tokenizer_A
    
    new_dataset = []
    
    # Process data in batches
    with open(save_path, 'w') as f:
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="logging rollout logprobs"):
            batch_end = min(batch_start + batch_size, len(dataset))
            # Convert HuggingFace dataset slice to list of dictionaries
            batch_data = [dataset[i] for i in range(batch_start, batch_end)]
            
            # Process batch
            batch_results = process_batch(
                batch_data, model_A, tokenizer_A, model_B, tokenizer_B,
                prompt_key, chosen_key, rejected_key, input_template, answer_key, k
            )
            
            # Write results to file
            for result in batch_results:
                if result is not None:  # Skip invalid samples
                    new_dataset.append(result)
                    json_line = json.dumps(result)
                    f.write(json_line + '\n')
            
            if len(new_dataset) % 1000 < batch_size:
                print(f"\nlogged dataset samples: {len(new_dataset)}\n")

    return new_dataset

def init_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
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
    
    # # Clean up individual files
    # for file_path in files:
    #     os.remove(file_path)
    #     print(f"Removed {file_path}")

def build_RePO_dataset_topk(args):
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print if rank == 0 else lambda *args, **kwargs: None
    dummy_strategy.is_rank_0 = lambda: rank == 0
    dummy_strategy.args = args

    # Load models on current device
    device = torch.device(f'cuda:{local_rank}')
    model_A = AutoModelForCausalLM.from_pretrained(args.pretrain_A).to(device)
    tokenizer_A = AutoTokenizer.from_pretrained(args.pretrain_A)
    model_A.eval()

    if rank == 0:
        print(f"Loaded model_A on device: {device}")
        print(f"World size: {world_size}, Rank: {rank}")
        print(f"Using batch size: {getattr(args, 'batch_size', 1)}")
    
    if args.pretrain_B is not None:
        model_B = AutoModelForCausalLM.from_pretrained(args.pretrain_B).to(device)
        tokenizer_B = AutoTokenizer.from_pretrained(args.pretrain_B)
        model_B.eval()
    else:
        model_B = None
        tokenizer_B = None

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
        split_ratio=0.05,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    
    # Split data across GPUs
    def split_data_for_rank(data, rank, world_size):
        total_samples = len(data)
        samples_per_gpu = total_samples // world_size
        remainder = total_samples % world_size
        
        start_idx = rank * samples_per_gpu + min(rank, remainder)
        if rank < remainder:
            end_idx = start_idx + samples_per_gpu + 1
        else:
            end_idx = start_idx + samples_per_gpu
            
        return data.select(range(start_idx, end_idx))
    
    train_data_split = split_data_for_rank(train_data, rank, world_size)
    eval_data_split = split_data_for_rank(eval_data, rank, world_size)
    
    if rank == 0:
        print(f"Total train data: {len(train_data)}")
        print(f"Total eval data: {len(eval_data)}")
        os.makedirs(args.save_path, exist_ok=True)
    
    # Wait for rank 0 to create directory
    if world_size > 1:
        dist.barrier()
    
    print(f"Rank {rank}: Processing {len(train_data_split)} train samples and {len(eval_data_split)} eval samples")

    # Process train data
    train_output_path = os.path.join(args.save_path, f"RePO_train_rank_{rank}.jsonl")
    train_dataset = generate_execute_logprobs(
        train_data_split,
        train_output_path,
        model_A,
        tokenizer_A,
        model_B,
        tokenizer_B,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        input_template=None,
        k=getattr(args, 'top_k', 10),
        batch_size=getattr(args, 'batch_size', 1),
    )
    print(f"Rank {rank}: Train dataset size: {len(train_dataset)}")
    print(f"Rank {rank}: excluded train dataset: {len(train_data_split) - len(train_dataset)}")

    # Process eval data
    eval_output_path = os.path.join(args.save_path, f"RePO_test_rank_{rank}.jsonl")
    eval_dataset = generate_execute_logprobs(
        eval_data_split,
        eval_output_path,
        model_A,
        tokenizer_A,
        model_B,
        tokenizer_B,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        input_template=None,
        k=getattr(args, 'top_k', 10),
        batch_size=getattr(args, 'batch_size', 1),
    )
    print(f"Rank {rank}: Eval dataset size: {len(eval_dataset)}")
    print(f"Rank {rank}: excluded eval dataset: {len(eval_data_split) - len(eval_dataset)}")
    
    # Wait for all processes to finish
    if world_size > 1:
        dist.barrier()
    
    # Merge files on rank 0
    if rank == 0:
        print("Merging output files...")
        merge_jsonl_files(args.save_path, "RePO_train_rank_*.jsonl", "RePO_train.jsonl")
        merge_jsonl_files(args.save_path, "RePO_test_rank_*.jsonl", "RePO_test.jsonl")
        print("All files merged successfully!")
    
    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # main parameters for build_RePO_dataset
    parser.add_argument("--pretrain_A", type=str, required=True)
    parser.add_argument("--pretrain_B", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./RePO_datasets")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON dataset prompt key")
    parser.add_argument("--chosen_key", type=str, default="chosen", help="JSON dataset chosen key")
    parser.add_argument("--rejected_key", type=str, default="rejected", help="JSON dataset rejected key")
    
    # DeepSpeed
    
    
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--deepspeed_port", type=int, default=None)

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


    parser.add_argument("--input_template", type=str, default=None)

    parser.add_argument(
        "--aRePOy_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top-k tokens to store logprobs for")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")


    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)
    args = parser.parse_args()

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



    build_RePO_dataset_topk(args)