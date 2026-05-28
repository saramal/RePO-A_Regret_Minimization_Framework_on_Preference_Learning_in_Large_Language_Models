from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import exist_and_not_none, zero_pad_sequences
from datasets import interleave_datasets, load_dataset, load_from_disk
import os
from tqdm import tqdm
import re
#######TODO#########
#   fix is_dpo -> alter with other option
#   
####################


def RePO_datasets(
    train_data_path,
    eval_data_path=None,
    ):
    
    ext = os.path.splitext(train_data_path)[-1]
    train_dataset = load_dataset(ext, data_files=train_data_path)
    
    if eval_data_path:
        ext = os.path.splitext(eval_data_path)[-1]
        eval_dataset = load_dataset(ext, data_files=eval_data_path)
    else:
        eval_dataset = None
    
    return train_dataset, eval_dataset
    



def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    chosen_logprob_key="chosen_logprob_with_token",
    rejected_logprob_key="rejected_logprob_with_token",
    aRePOy_chat_template=None,
    system_prompt=None,
    # is_dpo=False,
) -> str:
    if aRePOy_chat_template:
        if prompt_key:

            prompt = aRePOy_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = aRePOy_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = aRePOy_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = aRePOy_chat_template(data[chosen_key], tokenize=False)
            rejected = aRePOy_chat_template(data[rejected_key], tokenize=False)

            # Not compatible with chosen/rejected format data! RePO needs actions
            
            # if is_dpo:
            #     prompt = aRePOy_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
            #     chosen = chosen[len(prompt) :]
            #     rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                # consider this is llama3 format
                
                # remove original prompt
                prefix = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
                suffix = "<|im_end|>\n<|im_start|>assistant\n"
                pattern = re.escape(prefix) + r"(.*?)" + re.escape(suffix)
                match = re.search(pattern, prompt, re.DOTALL)
                if match:
                    clean_prompt = match.group(1)
                else:
                    raise ValueError(f"Maybe not Qwen prompt format or parsing error:: Could not find prompt in {prompt}")
                
                prompt = input_template.format(input=clean_prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]
        
    # # logprob with tokens: [(token, logprob), ...]
    # chosen_logprob_with_token = data[chosen_logprob_key]
    # rejected_logprob_with_token = data[rejected_logprob_key]
    
    # answer label for evaluation
    answer_label = getattr(data, "answer_label", 0)

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, answer_label, margin


class RePODataset_Deterministic_fast(Dataset):
    """
    Dataset for RePO framework for deterministic generation

    Args:
        dataset: dataset for RePO model for deterministic generation
            dataset: {prompt, chosen, rejected, chosen_logprob_with_token, rejected_logprob_with_token, answer_label}
        self.tokenizer: self.tokenizer for RePO target model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        # is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        # self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        
        # dummy value for logprob label
        self.dummy_value = getattr(self.strategy.args, "dummy_value", 100)
        
        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)

        
        #Do not consider chat template case
        self.aRePOy_chat_template = False
        # self.aRePOy_chat_template = getattr(self.strategy.args, "aRePOy_chat_template", False)
        
        if self.aRePOy_chat_template:
            self.aRePOy_chat_template = self.tokenizer.aRePOy_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        if "llama" in self.strategy.args.pretrain:
            self.input_template = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{{answer}}$.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {input}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            self.input_template = input_template
        
        self._map_kwargs = dict(
            input_template=self.input_template,
            prompt_key=self.prompt_key,
            chosen_key=self.chosen_key,
            rejected_key=self.rejected_key,
            aRePOy_chat_template=self.aRePOy_chat_template,
        )

        # batched map+filter
        processed_dataset = dataset.map(
            self._process_and_filter_batch,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )

        # Don't make python list objects for the processed dataset;
        # just use the dataset object directly.
        self.dataset = processed_dataset

    def _process_and_filter_batch(self, batch):
        """
        HF dataset's batched map function.
        - preprocess_data call
        - skip samples with None prompt (filter role)
        """
        outputs = {
            "prompt": [],
            "chosen": [],
            "reject": [],
            "answer_label": [],
            "extra": [],
        }

        # batch size
        # (any key can be used, but chosen_key is used as reference)
        n = len(batch[self.chosen_key])

        for i in range(n):
            # restore each row (dict)
            row = {k: v[i] for k, v in batch.items()}

            (
                prompt,
                chosen,
                rejected,
                answer_label,
                margin,
            ) = preprocess_data(
                row,
                **self._map_kwargs,
            )

            # 🔥 Here, invalid sample filtering (filter role)
            #   original filter: x["prompt"] is not None
            if prompt is None:
                continue
            
            # skip sanitization check for fast datasetd
            
            # # Check if the token is not a single token
            # missmatch_token = False
            # for token in chosen_logprob_with_token["tokens"]:
            #     token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            #     if token_id.size(0) != 1:
            #         prompt = None
            #         missmatch_token = True
            #         self.strategy.print(f"Missmatch token: {token}")
            #         break
            # for token in rejected_logprob_with_token["tokens"]:
            #     token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            #     if token_id.size(0) != 1:
            #         prompt = None
            #         missmatch_token = True
            #         self.strategy.print(f"Missmatch token: {token}")
            #         break

            # if missmatch_token:
            #     continue

            outputs["prompt"].append(prompt)
            outputs["chosen"].append(chosen)
            outputs["reject"].append(rejected)
            outputs["answer_label"].append(answer_label)
            outputs["extra"].append(margin)

        return outputs

    def process_data(self, data):
        prompt, chosen, reject, answer_label, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.aRePOy_chat_template,
            # self.is_dpo,
        )
        

        # Check if the token is not a single token
        


        # calculate prompt len directly from prompt attention mask
        
        # if self.is_dpo:
        #     prompt_token = self.tokenizer(
        #         prompt,
        #         max_length=self.max_length,
        #         padding=False,
        #         truncation=True,
        #         return_tensors="pt",
        #         add_special_tokens=False,
        #     )
        #     prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        #     # Filter the sample whose length is greater than max_length (2 for answer length)
        #     if prompt_ids_len >= self.max_length - 2:
        #         prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "answer_label": answer_label,
            "extra": margin,
        }

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        # get the row from the dataset
        row = self.dataset[idx]
        prompt = row["prompt"]
        chosen = row["chosen"]
        reject = row["reject"]
        answer_label = row["answer_label"]
        extra = row["extra"]

        # Iteration method(for tuple list: [(token, logprob),...])
        
        prompt_tokenized = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_token_ids = prompt_tokenized["input_ids"][0]
        prompt_att_masks = prompt_tokenized["attention_mask"][0]
        prompt_ids_len = prompt_att_masks.int().sum().item()
        
        chosen_tokenized = self.tokenizer(chosen, add_special_tokens=False, return_tensors="pt")
        chosen_token_ids = chosen_tokenized["input_ids"][0]
        chosen_att_masks = chosen_tokenized["attention_mask"][0]
        chosen_ids_len = chosen_att_masks.int().sum().item()
        
        rejected_tokenized = self.tokenizer(reject, add_special_tokens=False, return_tensors="pt")
        rejected_token_ids = rejected_tokenized["input_ids"][0]
        rejected_att_masks = rejected_tokenized["attention_mask"][0]
        # chosen_token_ids=[]
        # chosen_att_masks=[]
        # chosen_logprobs=[]
        # for token, logprob in zip(chosen_logprob_with_token["tokens"], chosen_logprob_with_token["logprobs"]):
        #     token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        #     assert token_id.size(0) == 1
        #     chosen_token_ids.append(token_id)
        #     chosen_att_masks.append(1)
        #     chosen_logprobs.append(logprob)
        # chosen_token_ids = torch.cat(chosen_token_ids, dim=0).to(dtype=torch.int)
        # chosen_att_masks = torch.tensor(chosen_att_masks, dtype=torch.bool)
        # chosen_logprobs = torch.tensor(chosen_logprobs, dtype=torch.float32)
        
        # rejected_token_ids=[]
        # rejected_att_masks=[]
        # rejected_logprobs=[]
        # for token, logprob in zip(rejected_logprob_with_token["tokens"], rejected_logprob_with_token["logprobs"]):
        #     token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        #     assert token_id.size(0) == 1
        #     rejected_token_ids.append(token_id)
        #     rejected_att_masks.append(1)
        #     rejected_logprobs.append(logprob)
        # rejected_token_ids = torch.cat(rejected_token_ids, dim=0).to(dtype=torch.int)
        # rejected_att_masks = torch.tensor(rejected_att_masks, dtype=torch.bool)
        # rejected_logprobs = torch.tensor(rejected_logprobs, dtype=torch.float32)
        
        
        chosen_input_ids = torch.cat([prompt_token_ids, chosen_token_ids], dim=0).unsqueeze(dim=0).to(dtype=torch.int)
        chosen_full_att_masks = torch.cat([prompt_att_masks, chosen_att_masks]).unsqueeze(dim=0)
        chosen_label_mask = torch.tensor([0] * prompt_ids_len + [1] * chosen_token_ids.shape[-1], dtype=torch.bool).unsqueeze(dim=0)
        
        rejected_input_ids = torch.cat([prompt_token_ids, rejected_token_ids], dim=0).unsqueeze(dim=0).to(dtype=torch.int)
        rejected_full_att_masks = torch.cat([prompt_att_masks, rejected_att_masks]).unsqueeze(dim=0)
        rejected_label_mask = torch.tensor([0] * prompt_ids_len + [1] * rejected_token_ids.shape[-1], dtype=torch.bool).unsqueeze(dim=0)
        
        assert(chosen_input_ids.dim() == chosen_full_att_masks.dim() == chosen_label_mask.dim()==2)
        assert(rejected_input_ids.dim() == rejected_full_att_masks.dim() == rejected_label_mask.dim()==2)
        assert(chosen_input_ids.shape[-1] == chosen_full_att_masks.shape[-1] == chosen_label_mask.shape[-1])
        assert(rejected_input_ids.shape[-1] == rejected_full_att_masks.shape[-1] == rejected_label_mask.shape[-1])
        
        
        return (
            chosen_input_ids,
            chosen_full_att_masks,
            chosen_label_mask,
            
            rejected_input_ids,
            rejected_full_att_masks,
            rejected_label_mask,
            
            prompt_tokenized["input_ids"],
            prompt_tokenized["attention_mask"],
            answer_label,
            extra
        )
        
        
        ## Origin Methods... using string with full sentence + eos
        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        chosen_label_masks = []
        reject_ids = []
        rejects_masks = []
        rejected_label_masks = []
        prompt_ids=[]
        prompt_masks=[]
        answers = []
        extras = []
        
        for chosen_id, chosen_mask, chosen_label_mask, reject_id, rejects_mask, rejected_label_mask, prompt_id, prompt_mask, answer, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            chosen_label_masks.append(chosen_label_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            rejected_label_masks.append(rejected_label_mask)
            prompt_ids.append(prompt_id)
            prompt_masks.append(prompt_mask)
            answers.append(answer)
            extras.append(extra)

        ## TODO: Why padding side is left while dpo use right? should RePO use right too?
        # if self.is_dpo:
        #     padding_side = "right"
        # else:
        #     padding_side = "left"
        padding_side = "right"
            
        RePO_output = {}
        RePO_output["chosen_ids"] = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        RePO_output["chosen_masks"] = zero_pad_sequences(chosen_masks, side=padding_side)
        RePO_output["chosen_label_masks"] = zero_pad_sequences(chosen_label_masks, side=padding_side, value=False)
        
        RePO_output["reject_ids"] = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        RePO_output["rejects_masks"] = zero_pad_sequences(rejects_masks, side=padding_side)
        RePO_output["rejected_label_masks"] = zero_pad_sequences(rejected_label_masks, side=padding_side, value=False)
        
        RePO_output["prompt_ids"] = zero_pad_sequences(prompt_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        RePO_output["prompt_masks"] = zero_pad_sequences(prompt_masks, side=padding_side)
        
        RePO_output["answers"] = answers
        RePO_output["extras"] = extras
        
        return RePO_output

    def packing_collate_fn(self, item_list):
        raise ValueError("Packing collate_fn is not implemented for RePO dataset")
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras
