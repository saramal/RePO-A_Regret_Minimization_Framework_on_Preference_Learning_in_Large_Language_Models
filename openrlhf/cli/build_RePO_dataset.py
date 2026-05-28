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
    model_A,
    tokenizer_A,
    model_B=None,
    tokenizer_B=None,
    prompt_key="prompt",
    chosen_key="chosen",
    rejected_key="rejected",
    input_template=None,
    answer_key="answer",
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
    """
    import json
    
    if model_B is None:
        model_B = model_A
        tokenizer_B = tokenizer_A
    new_dataset = []
    with open(save_path, 'w') as f:
        
        for data in tqdm(dataset, desc="logging rollout logprobs"):
            chosen_model = data["chosen_model"] if "chosen_model" in data else None
            rejected_model = data["rejected_model"] if "rejected_model" in data else None

            # identify rollout policy of each trajectories
            if chosen_model is not None and "A" in chosen_model:
                chosen_model = model_A
                chosen_tokenizer = tokenizer_A
                rejected_model = model_B
                rejected_tokenizer = tokenizer_B
                
            elif chosen_model is not None and "B" in chosen_model:
                chosen_model = model_B
                chosen_tokenizer = tokenizer_B
                rejected_model = model_A
                rejected_tokenizer = tokenizer_A
                
            else:
                chosen_model = rejected_model = model_A
                chosen_tokenizer = rejected_tokenizer = tokenizer_A
            
            
            
            
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
            
            
            if chosen_key.split(",") and len(chosen_key.split(",")) > 1:
                chosen=""
                for _chosen_key in chosen_key.split(","):
                    if input_template is not None and _chosen_key.strip() == "prompt":
                        chosen += input_template.format(data[_prompt_key.strip()])
                    else:
                        chosen += data[_chosen_key.strip()]
                    chosen+="\n"
            else:
                chosen = data[chosen_key]
            
            
            if rejected_key.split(",") and len(rejected_key.split(",")) > 1:
                rejected=""
                for _rejected_key in rejected_key.split(","):
                    if input_template is not None and _rejected_key.strip() == "prompt":
                        rejected += input_template.format(data[_rejected_key.strip()])
                    else:
                        rejected += data[_rejected_key.strip()]
                    rejected+="\n"
            else:
                rejected = data[rejected_key]
                
            # prompt = data[prompt_key]
            # chosen = data[chosen_key]
            # rejected = data[rejected_key]
            # answer = data[answer_key] if answer_key in data else None
            
            
            # append end of text token if needed           
            if not chosen.endswith(chosen_tokenizer.eos_token):
                chosen = append_endoftext_if_needed(chosen, chosen_tokenizer.eos_token)

            if not rejected.endswith(rejected_tokenizer.eos_token):
                rejected = append_endoftext_if_needed(rejected, rejected_tokenizer.eos_token)


            # TODO: delete below codes after validate(assert)
            chosen_prompt_only_token = chosen_tokenizer(
                prompt,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            chosen_prompt_ids = chosen_prompt_only_token["input_ids"][0]
            chosen_prompt_att_masks = chosen_prompt_only_token["attention_mask"][0]
            chosen_prompt_len = len(chosen_prompt_ids)
            
            chosen_only_token = chosen_tokenizer(
                chosen,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            chosen_only_ids = chosen_only_token["input_ids"][0]
            chosen_only_att_masks = chosen_only_token["attention_mask"][0]
            chosen_only_len = len(chosen_only_ids)
            

            # assert chosen_only_ids[-1] == chosen_tokenizer.eos_token_id, f"chosen_only_ids[-1]: {chosen_only_ids[-1]}, chosen_tokenizer.eos_token_id: {chosen_tokenizer.eos_token_id}"
            
            
            rejected_prompt_only_token = rejected_tokenizer(
                prompt,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            rejected_prompt_ids = rejected_prompt_only_token["input_ids"][0]
            rejected_prompt_att_masks = rejected_prompt_only_token["attention_mask"][0]
            rejected_prompt_len = len(rejected_prompt_ids)
            
            rejected_only_token = rejected_tokenizer(
                rejected,
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            rejected_only_ids = rejected_only_token["input_ids"][0]
            rejected_only_att_masks = rejected_only_token["attention_mask"][0]
            rejected_only_len = len(rejected_only_ids)
            
            # assert rejected_only_ids[-1] == rejected_tokenizer.eos_token_id, f"rejected_only_ids[-1]: {rejected_only_ids[-1]}, rejected_tokenizer.eos_token_id: {rejected_tokenizer.eos_token_id}"
            
            
            
            # chosen_full = prompt + chosen
            # chosen_token = chosen_tokenizer(
            #     chosen_full,
            #     padding=False,
            #     return_tensors="pt",
            #     add_special_tokens=False,
            # )
            # chosen_ids = chosen_token["input_ids"][0]
            # chosen_att_mask = chosen_token["attention_mask"][0]
            
            chosen_ids = torch.cat([chosen_prompt_ids, chosen_only_ids], dim=0).to(torch.cuda.current_device()).unsqueeze(dim=0)
            chosen_att_mask = torch.cat([chosen_prompt_att_masks, chosen_only_att_masks], dim=0).to(torch.cuda.current_device()).unsqueeze(dim=0)
            
            
            # # for debug
            # chosen_compare_tokens = [(tokenizer_A.decode(full_id, skip_special_tokens = False), tokenizer_A.decode(concat_id, skip_special_tokens = False)) for full_id, concat_id in zip(chosen_ids.tolist(), chosen_prompt_ids.tolist()+chosen_only_ids.tolist())]
            # chosen_compare_token_ids = [(full_id, concat_id) for full_id, concat_id in zip(chosen_ids.tolist(), chosen_prompt_ids.tolist()+chosen_only_ids.tolist())]
            # print(f"chosen_ids: {len(chosen_ids)} chosen_prompt_len: {chosen_prompt_len}, chosen_only_len : {chosen_only_len}")
            # print(f"chosen_ids: {chosen_ids}, chosen_prompt_len: {chosen_prompt_ids}, chosen_only_len: {chosen_only_ids}")
            # print(f"chosen_compare_token_ids: {chosen_compare_token_ids}")
            # print(f"chosen_compare_tokens: {chosen_compare_tokens}")
            # print(f"chosen_ids: {tokenizer_A.decode(chosen_ids, skip_special_tokens = False)}, chosen_prompt: {tokenizer_A.decode(chosen_prompt_ids, skip_special_tokens = False)}, chosen_only_len: {tokenizer_A.decode(chosen_only_ids, skip_special_tokens = False)}")
            # import pdb
            # pdb.set_trace()
            assert chosen_ids.shape[-1] == chosen_prompt_len + chosen_only_len
            
            
                
            # rejected_full = prompt + rejected
            # rejected_token = rejected_tokenizer(
            #     rejected_full,
            #     padding=False,
            #     return_tensors="pt",
            #     add_special_tokens=False,
            # )
            # rejected_ids = rejected_token["input_ids"][0]
            # rejected_att_mask = rejected_token["attention_mask"][0]
            rejected_ids = torch.cat([rejected_prompt_ids, rejected_only_ids], dim=0).to(torch.cuda.current_device()).unsqueeze(dim=0)
            rejected_att_mask = torch.cat([rejected_prompt_att_masks, rejected_only_att_masks], dim=0).to(torch.cuda.current_device()).unsqueeze(dim=0)
            
            assert rejected_ids.shape[-1] == rejected_prompt_len + rejected_only_len
            
            chosen_only_logprob = chosen_model(chosen_ids, attention_mask=chosen_att_mask, num_actions=chosen_only_len)
            rejected_only_logprob = rejected_model(rejected_ids, attention_mask=rejected_att_mask, num_actions=rejected_only_len)
            
            chosen_only_logprob = chosen_only_logprob.squeeze()
            rejected_only_logprob = rejected_only_logprob.squeeze()
            
            assert chosen_only_logprob.dim()==1 and chosen_only_logprob.shape[-1] == chosen_only_len
            assert rejected_only_logprob.dim()==1 and rejected_only_logprob.shape[-1] == rejected_only_len
        

            # chosen_logprob_with_token = [{chosen_tokenizer.decode(tok_id): logprob} for tok_id, logprob in zip(chosen_only_ids.tolist(), chosen_only_logprob.tolist())]
            chosen_logprob_with_token = {"tokens": [chosen_tokenizer.decode(tok_id) for tok_id in chosen_only_ids.tolist()], "logprobs":chosen_only_logprob.tolist()}
            # rejected_logprob_with_token = [{rejected_tokenizer.decode(tok_id): logprob} for tok_id, logprob in zip(rejected_only_ids.tolist(), rejected_only_logprob.tolist())]
            rejected_logprob_with_token = {"tokens": [rejected_tokenizer.decode(tok_id) for tok_id in rejected_only_ids.tolist()], "logprobs":rejected_only_logprob.tolist()}

            assert len(chosen_logprob_with_token["tokens"]) == chosen_only_len
            assert len(chosen_logprob_with_token["logprobs"]) == chosen_only_len
            
            assert len(rejected_logprob_with_token["tokens"]) == rejected_only_len
            assert len(rejected_logprob_with_token["logprobs"]) == rejected_only_len            
            

            restored_chosen  = ""
            for token, logp in zip(chosen_logprob_with_token["tokens"], chosen_logprob_with_token["logprobs"]):
                restored_chosen += token
            if restored_chosen != chosen:
                print(f"restored_chosen: {restored_chosen}\nchosen: {chosen}")
                print(f"chosen tokens: {chosen_logprob_with_token['tokens']}")
                print(f"chosen only tokens: {[chosen_tokenizer.decode(tok_id) for tok_id in chosen_only_ids.tolist()]}")

                continue
            # assert restored_chosen == chosen
            
            restored_rejected = ""
            for token, logp in zip(rejected_logprob_with_token["tokens"], rejected_logprob_with_token["logprobs"]):
                restored_rejected += token
            if restored_rejected != rejected:
                print(f"restored_rejected: {restored_rejected}\nrejected: {rejected}")
                print(f"rejected tokens: {rejected_logprob_with_token['tokens']}")
                print(f"rejected only tokens: {[rejected_tokenizer.decode(tok_id) for tok_id in rejected_only_ids.tolist()]}")
                continue
            # assert restored_rejected == rejected
            
            new_data = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_logprob_with_token": chosen_logprob_with_token,
                "rejected_logprob_with_token": rejected_logprob_with_token,
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
    model_A = Actor(
        args.pretrain_A,
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
    tokenizer_A = get_tokenizer(args.pretrain_A, model_A.model, "right", strategy, use_fast=not args.disable_fast_tokenizer, special_token_list=special_tokens)
    strategy.print(model_A)
    # print(model_A)
    # prepare models
    model_A = strategy.prepare(model_A)
    
    if args.pretrain_B is not None:
        model_B = Actor(
            args.pretrain_B,
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
        tokenizer_B = get_tokenizer(args.pretrain_B, model_B.model, "right", strategy, use_fast=not args.disable_fast_tokenizer, special_token_list=special_tokens)
        # strategy.print(model_B)
        # print(model_B)
        # prepare models
        model_B = strategy.prepare(model_B)

    else:
        model_B = None
        tokenizer_B = None
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
        model_A,
        tokenizer_A,
        model_B,
        tokenizer_B,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        input_template=args.input_template,
    )
    strategy.print(f"Train dataset size: {len(train_dataset)}")
    strategy.print(f"excluded train dataset: {len(train_data) - len(train_dataset)}")

    eval_dataset = generate_execute_logprobs(
        eval_data,
        os.path.join(args.save_path, "RePO_test.jsonl"),
        model_A,
        tokenizer_A,
        model_B,
        tokenizer_B,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        input_template=args.input_template,
    )
    strategy.print(f"Eval dataset size: {len(eval_dataset)}")
    strategy.print(f"excluded eval dataset: {len(eval_data) - len(eval_dataset)}")


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
