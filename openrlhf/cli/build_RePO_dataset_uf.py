import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from peft import PeftModel

from tqdm import tqdm
import torch

import re
def append_endoftext_if_needed(s: str, eos_token, answer_prefix: str = "The answer is:") -> str:

    escaped_prefix = re.escape(answer_prefix)
    pattern = rf"{escaped_prefix} .+\s*$"
    
    if re.search(pattern, s):
        return s.rstrip() + f" {eos_token}"
    else:
        return s

def generate_execute_logprobs(
    dataset,
    save_path,
    model,
    tokenizer,
    prompt_key="prompt",
    response_key="response",
    input_template=None,
    answer_key="answer",
    ):
    """
    Generate logprobs for single responses.

    Args:
        dataset: dataset with single responses
            dataset: {prompt, response, answer_label}
        model: policy model
        tokenizer: tokenizer for policy model
    """
    import json
    
    new_dataset = []
    with open(save_path, 'w') as f:
        
        for data in tqdm(dataset, desc="logging response logprobs"):
            
            # tokenize
            if prompt_key.split(",") and len(prompt_key.split(",")) > 1:
                prompt=""
                for _prompt_key in prompt_key.split(","):
                    if input_template is not None and _prompt_key.strip() == "prompt":
                        prompt += input_template.format(data[_prompt_key.strip()])
                    else:
                        prompt += data[_prompt_key.strip()]
                    prompt+="\n"
            else:
                if input_template is not None and prompt_key.strip() == "prompt":
                    prompt = input_template.format(data[prompt_key])
                else:
                    prompt = data[prompt_key] + "\n"
            
            
            if response_key.split(",") and len(response_key.split(",")) > 1:
                response=""
                for _response_key in response_key.split(","):
                    if input_template is not None and _response_key.strip() == "prompt":
                        response += input_template.format(data[_response_key.strip()])
                    else:
                        response += data[_response_key.strip()]
                    response+="\n"
            else:
                response = data[response_key]
            
            # append end of text token if needed           
            if not response.endswith(tokenizer.eos_token):
                response = append_endoftext_if_needed(response, tokenizer.eos_token)

            # tokenize prompt and response
            prompt_token = tokenizer(
                prompt,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids = prompt_token["input_ids"][0]
            prompt_att_masks = prompt_token["attention_mask"][0]
            prompt_len = len(prompt_ids)
            
            response_token = tokenizer(
                response,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            response_ids = response_token["input_ids"][0]
            response_att_masks = response_token["attention_mask"][0]
            response_len = len(response_ids)
            
            # concatenate prompt and response
            full_ids = torch.cat([prompt_ids, response_ids], dim=0).to(torch.cuda.current_device()).unsqueeze(dim=0)
            full_att_mask = torch.cat([prompt_att_masks, response_att_masks], dim=0).to(torch.cuda.current_device()).unsqueeze(dim=0)
            
            assert full_ids.shape[-1] == prompt_len + response_len
            
            # get logprobs for response tokens only
            response_logprob = model(full_ids, attention_mask=full_att_mask, num_actions=response_len)
            response_logprob = response_logprob.squeeze()
            
            assert response_logprob.dim()==1 and response_logprob.shape[-1] == response_len
            # get just the logprobs list
            logprob_list = response_logprob.tolist()
            
            # validate by reconstructing response from tokens
            response_tokens = [tokenizer.decode(tok_id) for tok_id in response_ids.tolist()]
            restored_response = "".join(response_tokens)
            
            if restored_response != response:
                print(f"restored_response: {restored_response}\nresponse: {response}")
                print(f"response tokens: {response_tokens}")
                raise ValueError(f"restored_response: {restored_response}\nresponse: {response}")
            
            new_data = {
                "prompt": prompt,
                "response": response,
                "logprob": logprob_list,
                "answer_label": data[answer_key] if answer_key in data else None
            }
            new_dataset.append(new_data)
            if len(new_dataset) % 1000 == 1:
                print(f"\nlogged dataset sample at {len(new_dataset)} : \n{new_data}\n")

            # # make small dataset for debugging
            # if len(new_dataset) == 16:
            #     break
            
            # write data at save_file.jsonl
            json_line = json.dumps(new_data)
            f.write(json_line + '\n')

    return new_dataset
            





def build_RePO_dataset(args):
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
        # ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    # configure tokenizer
    #special_tokens = ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>', "<error_info>", "<end_of_error_info>", "<BACK>"]
    special_tokens = None
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer, special_token_list=special_tokens)
    strategy.print(model)
    # prepare models
    model = strategy.prepare(model)
    # # add "kn" at tokenizer & resize model
    # strategy.print(f"Before resizeing - Tokenizer size: {len(tokenizer)}") 
    # strategy.print(f"Before resizeing - Model size: {model.model.get_input_embeddings().weight.shape[0]}")
    # strategy.print(f"Before resizeing - Model size: {model.model.get_output_embeddings().weight.shape[0]}")

    # # tokenizer.add_special_tokens({"additional_special_tokens": ["¿"]})
    # # model.model.resize_token_embeddings(len(tokenizer))

    # # tokenizer.add_tokens(["ки"], special_tokens=True)
    
    # strategy.print(f"Tokenizer size: {len(tokenizer)}") 
    # strategy.print(f"Model size: {model.model.get_input_embeddings().weight.shape[0]}")
    # strategy.print(f"Model size: {model.model.get_output_embeddings().weight.shape[0]}")

    # # gradient_checkpointing
    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable(
    #         gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
    #     )

    # # configure optimizer
    # optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

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
    


    os.makedirs(args.save_path, exist_ok=True)

    # logging dataset
    train_dataset = generate_execute_logprobs(
        train_data,
        os.path.join(args.save_path, "RePO_train.jsonl"),
        model,
        tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        input_template=args.input_template,
    )
    strategy.print(f"Train dataset size: {len(train_dataset)}")
    strategy.print(f"excluded train dataset: {len(train_data) - len(train_dataset)}")

    eval_dataset = generate_execute_logprobs(
        eval_data,
        os.path.join(args.save_path, "RePO_test.jsonl"),
        model,
        tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        input_template=args.input_template,
    )
    strategy.print(f"Eval dataset size: {len(eval_dataset)}")
    strategy.print(f"excluded eval dataset: {len(eval_data) - len(eval_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # main parameters for build_RePO_dataset
    parser.add_argument("--pretrain", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./RePO_datasets")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON dataset prompt key")
    parser.add_argument("--response_key", type=str, default="response", help="JSON dataset response key")
    
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


    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--aRePOy_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")


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



    build_RePO_dataset(args)
