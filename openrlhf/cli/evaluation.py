import argparse
import math
import os
from datetime import datetime

import torch
from tqdm import tqdm
import json

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.utils import extract_last_answer, match_with_answer_labels_v2, match_with_answer_labels_v3, extract_all_answers
from peft import PeftModel, PeftModelForCausalLM

def eval(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    # configure tokenizer
    # special_tokens = ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>', "<error_info>", "<end_of_error_info>", "<BACK>"]
    special_tokens = None
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer, special_token_list=special_tokens)
    strategy.print(model)



    # tokenizer.add_special_tokens({"additional_special_tokens": ["¿"]})
    # model.model.resize_token_embeddings(len(tokenizer))

    # import pdb
    # pdb.set_trace()
    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    

    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=False,
        input_template=args.input_template,
        multiple_of=1,
        multiturn=False,
    )

    # prepare dataloader

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_eval_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )




    # prepare models
    model = strategy.prepare(model)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
    else:
        strategy.print(f"Checkpoint not found: {args.ckpt_path}, start from scratch.")

    

    # evaluate
    strategy.print("Start evaluation...")
    evaluate(model, tokenizer, strategy, eval_dataloader)
    


def evaluate(model, tokenizer, strategy, eval_dataloader):
    times = 0
    acc = 0
    model.eval()

    with torch.no_grad():

        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Evaluation",
            disable=not strategy.is_rank_0(),
        )
        os.makedirs(strategy.args.generation_log_path, exist_ok=True)
        save_path = os.path.join(strategy.args.generation_log_path, f"eval.jsonl")

        with open(save_path, 'w') as f:
            f.write(json.dumps(f"args: \n\n{strategy.args}\n") + "\n")
        
        for prompt_id_lens, inputs, attention_masks, answers, infos in eval_dataloader:
            times += 1
            inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
            attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

            
                

            # _, _, _, _, _, _, _, _, prompt_ids, prompt_masks, answer_label, extra = data
            prompt_ids = infos["input"].squeeze(1).to(torch.cuda.current_device())
            # prompt_masks = infos["prompt_masks"].squeeze(1).to(torch.cuda.current_device())


            #TODO: fix generate function! refer to data generation step in RePO old code
            
            # generated_outputs = self.model.generate(prompt_ids, prompt_masks)
            # model_input_for_generation = {"input_ids": prompt_ids, "attention_mask": prompt_masks}
            generated_outputs, _, _ = model.generate(
                                    input_ids=prompt_ids,
                                    # attention_mask=prompt_masks,
                                    use_cache=True,
                                    max_length=strategy.args.max_len,
                                    do_sample=False,
                                    top_p=strategy.args.top_p,
                                    early_stopping=False,
                                    num_beams=3,
                                    temperature=strategy.args.temperature,
                                    repetition_penalty=strategy.args.repetition_penalty,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                )
            tokenized_output = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            # chosen_reward, reject_reward, _ = self.concatenated_forward(
            #     self.model, chosen_ids, c_mask, reject_ids, r_mask
            # )
            # import pdb
            # pdb.set_trace()

            # tokenized_output = strategy.all_gather(tokenized_output)
            # gathered_answers = strategy.all_gather(answers)
            # save generation log
            if strategy.is_rank_0():
                # import pdb
                # pdb.set_trace()
                with open(save_path, 'a') as f:
                    for generation, answer in zip(tokenized_output, answers):
                        generation_dict = {"generation": generation, 
                                            #"extracted_answers": extract_first_numeric_answer(generation, strategy.args.answer_trigger),
                                            "extracted_last_answers": extract_last_answer(generation, strategy.args.answer_trigger),
                                            "extracted_all_answers": extract_all_answers(generation, strategy.args.answer_trigger),
                                            "gold_answers": answer}
                        f.write(json.dumps(generation_dict, ensure_ascii=False) + "\n")
            
            # acc += match_with_answer_labels(tokenized_output, answers, answer_trigger=strategy.args.answer_trigger)
            acc += match_with_answer_labels_v3(tokenized_output, answers, answer_trigger=strategy.args.answer_trigger)
            bar_dict = {"eval accuracy": acc / times}
        
            step_bar.update()
            logs = strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

    if strategy.is_rank_0():
        strategy.print(f"Evaluation finished, accuracy: {logs['eval accuracy']:.4f}")
        with open(save_path, 'a') as f:
            f.write(json.dumps(f"Evaluation finished, accuracy: {logs['eval accuracy']:.4f}\n") + "\n")
            
   
def match_with_answer_labels(tokenized_output, answers, answer_trigger):
        #TODO: current case is only work for gsm8k. find appropriate match with MATH or else.
        correct_count = 0
        valid_count = 0
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                predicted_answer = extract_first_numeric_answer(output, answer_trigger)
                if predicted_answer is None:
                    continue
                is_correct = check_correctness(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1
        
        return correct_count/valid_count if valid_count > 0 else 0
    

def extract_first_numeric_answer( text:str, answer_trigger:str):
    import re

    pattern = re.escape(answer_trigger) + r"\s*['\"]?(\d+(?:\.\d+)?)['\"]?"

    matches = re.findall(pattern, text)

    if not matches:
        return None

    answer = matches[-1].strip()
    return float(answer) if re.match(r'^\d+(\.\d+)?$', answer) else None
    # # extract first answer
    # first = min(matches, key=lambda x: x[1])
    # answer_text = first[2]

    # # float / int extract
    # num_match = re.search(r'\d+(?:\.\d+)?', answer_text)
    # return float(num_match.group()) if num_match else None
    
    
def check_correctness(prediction, target):
    return abs(float(prediction) - float(target)) <= 1e-3



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--answer_key", type=str, default="answer_label", help="JSON dataset key")
    parser.add_argument("--answer_trigger", type=str, default="The answer is:", help="Trigger for answer extraction")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--aRePOy_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # Generation configs
    # parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty for generation")
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
