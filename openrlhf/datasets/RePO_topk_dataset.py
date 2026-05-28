from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import exist_and_not_none, zero_pad_sequences, zero_pad_sequences_for_topk
from datasets import interleave_datasets, load_dataset, load_from_disk
import os
from tqdm import tqdm


def RePO_datasets_topk(
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


def preprocess_data_topk(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    chosen_logprob_key="chosen_logprob_with_token",
    rejected_logprob_key="rejected_logprob_with_token",
    chosen_top_k_key="chosen_top_k_data",
    rejected_top_k_key="rejected_top_k_data",
    aRePOy_chat_template=None,
    system_prompt=None,
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
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]
        
    # logprob with tokens
    chosen_logprob_with_token = data[chosen_logprob_key]
    rejected_logprob_with_token = data[rejected_logprob_key]
    
    # top-k data
    chosen_top_k_data = data.get(chosen_top_k_key, None)
    rejected_top_k_data = data.get(rejected_top_k_key, None)
    
    # answer label for evaluation
    answer_label = data.get("answer_label", None)

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, chosen_logprob_with_token, rejected_logprob_with_token, chosen_top_k_data, rejected_top_k_data, answer_label, margin


class RePODataset_topk(Dataset):
    """
    Dataset for RePO framework with top-k support

    Args:
        dataset: dataset for RePO model
            dataset: {prompt, chosen, rejected, chosen_logprob_with_token, rejected_logprob_with_token, answer_label, chosen_top_k_data, rejected_top_k_data}
        tokenizer: tokenizer for RePO target model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        
        # dummy value for logprob label
        self.dummy_value = getattr(self.strategy.args, "dummy_value", 100)
        
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", "prompt")
        self.chosen_key = getattr(self.strategy.args, "chosen_key", "chosen")
        self.rejected_key = getattr(self.strategy.args, "rejected_key", "rejected")
        self.chosen_logprob_key = getattr(self.strategy.args, "chosen_logprob_key", "chosen_logprob_with_token")
        self.rejected_logprob_key = getattr(self.strategy.args, "rejected_logprob_key", "rejected_logprob_with_token")
        self.chosen_top_k_key = getattr(self.strategy.args, "chosen_top_k_key", "chosen_top_k_data")
        self.rejected_top_k_key = getattr(self.strategy.args, "rejected_top_k_key", "rejected_top_k_data")
        
        self.topk_k = getattr(self.strategy.args, "top_k", 10)
        # Do not consider chat template case
        self.aRePOy_chat_template = False
        
        if self.aRePOy_chat_template:
            self.aRePOy_chat_template = self.tokenizer.aRePOy_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.chosen_logprob_with_tokens = processed_dataset["chosen_logprob_with_token"]
        self.reject_logprob_with_tokens = processed_dataset["rejected_logprob_with_token"]
        self.chosen_top_k_datas = processed_dataset["chosen_top_k_data"]
        self.rejected_top_k_datas = processed_dataset["rejected_top_k_data"]
        self.answer_labels = processed_dataset["answer_label"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, chosen_logprob_with_token, rejected_logprob_with_token, chosen_top_k_data, rejected_top_k_data, answer_label, margin = preprocess_data_topk(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.chosen_logprob_key,
            self.rejected_logprob_key,
            self.chosen_top_k_key,
            self.rejected_top_k_key,
            self.aRePOy_chat_template,
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "chosen_logprob_with_token": chosen_logprob_with_token,
            "rejected_logprob_with_token": rejected_logprob_with_token,
            "chosen_top_k_data": chosen_top_k_data,
            "rejected_top_k_data": rejected_top_k_data,
            "answer_label": answer_label,
            "extra": margin,
        }

    def __len__(self):
        return len(self.chosens)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosens[idx]
        reject = self.rejects[idx]
        chosen_logprob_with_token = self.chosen_logprob_with_tokens[idx]
        rejected_logprob_with_token = self.reject_logprob_with_tokens[idx]
        chosen_top_k_data = self.chosen_top_k_datas[idx]
        rejected_top_k_data = self.rejected_top_k_datas[idx]
        answer_label = self.answer_labels[idx]
        extra = self.extras[idx]

        # Process prompt
        prompt_tokenized = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_token_ids = prompt_tokenized["input_ids"][0]
        prompt_att_masks = prompt_tokenized["attention_mask"][0]
        prompt_ids_len = prompt_att_masks.int().sum().item()
        
        # Process chosen tokens and logprobs
        chosen_token_ids = []
        chosen_att_masks = []
        chosen_logprobs = []
        for token, logprob in zip(chosen_logprob_with_token["tokens"], chosen_logprob_with_token["logprobs"]):
            token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            assert token_id.size(0) == 1
            chosen_token_ids.append(token_id)
            chosen_att_masks.append(1)
            chosen_logprobs.append(logprob)
        chosen_token_ids = torch.cat(chosen_token_ids, dim=0).to(dtype=torch.int)
        chosen_att_masks = torch.tensor(chosen_att_masks, dtype=torch.bool)
        chosen_logprobs = torch.tensor(chosen_logprobs, dtype=torch.float32)
        
        # Process rejected tokens and logprobs
        rejected_token_ids = []
        rejected_att_masks = []
        rejected_logprobs = []
        for token, logprob in zip(rejected_logprob_with_token["tokens"], rejected_logprob_with_token["logprobs"]):
            token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            assert token_id.size(0) == 1
            rejected_token_ids.append(token_id)
            rejected_att_masks.append(1)
            rejected_logprobs.append(logprob)
        rejected_token_ids = torch.cat(rejected_token_ids, dim=0).to(dtype=torch.int)
        rejected_att_masks = torch.tensor(rejected_att_masks, dtype=torch.bool)
        rejected_logprobs = torch.tensor(rejected_logprobs, dtype=torch.float32)
        
        # Process chosen top-k data
        chosen_topk_logprobs = []
        chosen_topk_indices = []
        chosen_topk_masks = []
        
        if chosen_top_k_data["top_k_logprobs"] and chosen_top_k_data["top_k_tokens"]:
            if len(chosen_top_k_data["top_k_logprobs"][0]) > self.topk_k:
                raise ValueError(f"Top-k length is greater than {self.topk_k}")
            for pos_logprobs, pos_tokens in zip(chosen_top_k_data["top_k_logprobs"], chosen_top_k_data["top_k_tokens"]):
                # Convert token strings to indices
                pos_indices = []
                for token in pos_tokens:
                    token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    assert token_id.size(0) == 1
                    pos_indices.append(token_id.item())
                
                chosen_topk_logprobs.append(pos_logprobs[:self.topk_k])
                chosen_topk_indices.append(pos_indices[:self.topk_k])
                chosen_topk_masks.append([1] * self.topk_k)  # All top-k positions are valid
        else:
            raise ValueError(f"No top-k data available")
            # Create dummy data if no top-k data available
            k = 10  # default k
            for _ in range(len(chosen_token_ids)):
                chosen_topk_logprobs.append([self.dummy_value] * k)
                chosen_topk_indices.append([0] * k)
                chosen_topk_masks.append([0] * k)
        
        # Process rejected top-k data
        rejected_topk_logprobs = []
        rejected_topk_indices = []
        rejected_topk_masks = []
        
        if rejected_top_k_data["top_k_logprobs"] and rejected_top_k_data["top_k_tokens"]:
            if len(rejected_top_k_data["top_k_logprobs"][0]) > self.topk_k:
                raise ValueError(f"Top-k length is greater than {self.topk_k}")
            for pos_logprobs, pos_tokens in zip(rejected_top_k_data["top_k_logprobs"], rejected_top_k_data["top_k_tokens"]):
                # Convert token strings to indices
                pos_indices = []
                for token in pos_tokens:
                    token_id = self.tokenizer(token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    assert token_id.size(0) == 1
                    pos_indices.append(token_id.item())
                
                rejected_topk_logprobs.append(pos_logprobs[:self.topk_k])
                rejected_topk_indices.append(pos_indices[:self.topk_k])
                rejected_topk_masks.append([1] * self.topk_k)  # All top-k positions are valid
        else:
            raise ValueError(f"No top-k data available")
        
        # Create input tensors
        chosen_input_ids = torch.cat([prompt_token_ids, chosen_token_ids], dim=0).unsqueeze(dim=0).to(dtype=torch.int)
        chosen_full_att_masks = torch.cat([prompt_att_masks, chosen_att_masks]).unsqueeze(dim=0)
        chosen_full_logprob_labels = torch.cat([torch.tensor([self.dummy_value] * prompt_ids_len), chosen_logprobs]).unsqueeze(dim=0)
        chosen_label_mask = torch.tensor([0] * prompt_ids_len + [1] * chosen_token_ids.shape[-1], dtype=torch.bool).unsqueeze(dim=0)
        
        # Create chosen top-k tensors
        chosen_topk_logprob_labels = torch.cat([
            torch.tensor([[self.dummy_value] * self.topk_k] * prompt_ids_len), 
            torch.tensor(chosen_topk_logprobs)
        ]).unsqueeze(dim=0).to(dtype=torch.float32)
        chosen_topk_logprob_masks = torch.cat([
            torch.tensor([[0] * self.topk_k] * prompt_ids_len), 
            torch.tensor(chosen_topk_masks)
        ]).unsqueeze(dim=0).to(dtype=torch.bool)
        chosen_topk_logprob_indices = torch.cat([
            torch.tensor([[0] * self.topk_k] * prompt_ids_len), 
            torch.tensor(chosen_topk_indices)
        ]).unsqueeze(dim=0).to(dtype=torch.int)
        
        rejected_input_ids = torch.cat([prompt_token_ids, rejected_token_ids], dim=0).unsqueeze(dim=0).to(dtype=torch.int)
        rejected_full_att_masks = torch.cat([prompt_att_masks, rejected_att_masks]).unsqueeze(dim=0)
        rejected_full_logprob_labels = torch.cat([torch.tensor([self.dummy_value] * prompt_ids_len), rejected_logprobs]).unsqueeze(dim=0)
        rejected_label_mask = torch.tensor([0] * prompt_ids_len + [1] * rejected_token_ids.shape[-1], dtype=torch.bool).unsqueeze(dim=0)
        
        # Create rejected top-k tensors
        rejected_topk_logprob_labels = torch.cat([
            torch.tensor([[self.dummy_value] * self.topk_k] * prompt_ids_len), 
            torch.tensor(rejected_topk_logprobs)
        ]).unsqueeze(dim=0).to(dtype=torch.float32)
        rejected_topk_logprob_masks = torch.cat([
            torch.tensor([[0] * self.topk_k] * prompt_ids_len), 
            torch.tensor(rejected_topk_masks)
        ]).unsqueeze(dim=0).to(dtype=torch.bool)
        rejected_topk_logprob_indices = torch.cat([
            torch.tensor([[0] * self.topk_k] * prompt_ids_len), 
            torch.tensor(rejected_topk_indices)
        ]).unsqueeze(dim=0).to(dtype=torch.int)
        
        
        assert(chosen_input_ids.dim() == chosen_full_att_masks.dim() == chosen_full_logprob_labels.dim() == chosen_label_mask.dim()==2)
        assert(rejected_input_ids.dim() == rejected_full_att_masks.dim() == rejected_full_logprob_labels.dim() == rejected_label_mask.dim()==2)
        assert(chosen_input_ids.shape[-1] == chosen_full_att_masks.shape[-1] == chosen_full_logprob_labels.shape[-1] == chosen_label_mask.shape[-1])
        assert(rejected_input_ids.shape[-1] == rejected_full_att_masks.shape[-1] == rejected_full_logprob_labels.shape[-1] == rejected_label_mask.shape[-1])
        
        assert(chosen_topk_logprob_labels.dim() == chosen_topk_logprob_masks.dim() == chosen_topk_logprob_indices.dim() == 3)
        assert(rejected_topk_logprob_labels.dim() == rejected_topk_logprob_masks.dim() == rejected_topk_logprob_indices.dim() == 3)
        assert(chosen_topk_logprob_labels.shape[-2] == chosen_topk_logprob_masks.shape[-2] == chosen_topk_logprob_indices.shape[-2])
        assert(rejected_topk_logprob_labels.shape[-2] == rejected_topk_logprob_masks.shape[-2] == rejected_topk_logprob_indices.shape[-2])

        return (
            chosen_input_ids,
            chosen_full_att_masks,
            chosen_full_logprob_labels,
            chosen_label_mask,
            chosen_topk_logprob_labels,
            chosen_topk_logprob_masks,
            chosen_topk_logprob_indices,
            
            rejected_input_ids,
            rejected_full_att_masks,
            rejected_full_logprob_labels,
            rejected_label_mask,
            rejected_topk_logprob_labels,
            rejected_topk_logprob_masks,
            rejected_topk_logprob_indices,
            
            prompt_tokenized["input_ids"],
            prompt_tokenized["attention_mask"],
            answer_label,
            extra
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        chosen_logprob_labels = []
        chosen_logprob_masks = []
        chosen_topk_logprob_labels = []
        chosen_topk_logprob_masks = []
        chosen_topk_logprob_indices = []
        
        reject_ids = []
        rejects_masks = []
        rejected_logprob_labels = []
        rejected_logprob_masks = []
        rejected_topk_logprob_labels = []
        rejected_topk_logprob_masks = []
        rejected_topk_logprob_indices = []
        
        prompt_ids = []
        prompt_masks = []
        answers = []
        extras = []
        
        for (chosen_id, chosen_mask, chosen_logprob, chosen_logprob_mask, chosen_topk_logprob, chosen_topk_mask, chosen_topk_indices,
             reject_id, rejects_mask, rejected_logprob, rejected_logprob_mask, rejected_topk_logprob, rejected_topk_mask, rejected_topk_indices,
             prompt_id, prompt_mask, answer, extra) in item_list:
            
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            chosen_logprob_labels.append(chosen_logprob)
            chosen_logprob_masks.append(chosen_logprob_mask)
            chosen_topk_logprob_labels.append(chosen_topk_logprob)
            chosen_topk_logprob_masks.append(chosen_topk_mask)
            chosen_topk_logprob_indices.append(chosen_topk_indices)
            
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            rejected_logprob_labels.append(rejected_logprob)
            rejected_logprob_masks.append(rejected_logprob_mask)
            rejected_topk_logprob_labels.append(rejected_topk_logprob)
            rejected_topk_logprob_masks.append(rejected_topk_mask)
            rejected_topk_logprob_indices.append(rejected_topk_indices)
            
            prompt_ids.append(prompt_id)
            prompt_masks.append(prompt_mask)
            answers.append(answer)
            extras.append(extra)

        padding_side = "right"
            
        RePO_output = {}
        RePO_output["chosen_ids"] = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        RePO_output["chosen_masks"] = zero_pad_sequences(chosen_masks, side=padding_side)
        RePO_output["chosen_logprob_labels"] = zero_pad_sequences(chosen_logprob_labels, side=padding_side, value=self.dummy_value)
        RePO_output["chosen_logprob_masks"] = zero_pad_sequences(chosen_logprob_masks, side=padding_side, value=False)
        RePO_output["chosen_topk_logprob_labels"] = zero_pad_sequences_for_topk(chosen_topk_logprob_labels, side=padding_side, value=self.dummy_value, pad_dim=-2)
        RePO_output["chosen_topk_logprob_masks"] = zero_pad_sequences_for_topk(chosen_topk_logprob_masks, side=padding_side, value=False, pad_dim=-2)
        RePO_output["chosen_topk_logprob_indices"] = zero_pad_sequences_for_topk(chosen_topk_logprob_indices, side=padding_side, value=0, pad_dim=-2)
        
        RePO_output["reject_ids"] = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        RePO_output["rejects_masks"] = zero_pad_sequences(rejects_masks, side=padding_side)
        RePO_output["rejected_logprob_labels"] = zero_pad_sequences(rejected_logprob_labels, side=padding_side, value=self.dummy_value)
        RePO_output["rejected_logprob_masks"] = zero_pad_sequences(rejected_logprob_masks, side=padding_side, value=False)
        RePO_output["rejected_topk_logprob_labels"] = zero_pad_sequences_for_topk(rejected_topk_logprob_labels, side=padding_side, value=self.dummy_value, pad_dim=-2)
        RePO_output["rejected_topk_logprob_masks"] = zero_pad_sequences_for_topk(rejected_topk_logprob_masks, side=padding_side, value=False, pad_dim=-2)
        RePO_output["rejected_topk_logprob_indices"] = zero_pad_sequences_for_topk(rejected_topk_logprob_indices, side=padding_side, value=0, pad_dim=-2)
        
        RePO_output["prompt_ids"] = zero_pad_sequences(prompt_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        RePO_output["prompt_masks"] = zero_pad_sequences(prompt_masks, side=padding_side)
        
        RePO_output["answers"] = answers
        RePO_output["extras"] = extras
        
        return RePO_output
