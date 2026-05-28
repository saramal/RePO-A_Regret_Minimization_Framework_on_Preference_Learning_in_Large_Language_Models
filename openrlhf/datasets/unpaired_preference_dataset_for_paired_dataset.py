from typing import Callable

import torch
from torch.utils.data import Dataset

from .utils import zero_pad_sequences


def preprocess_data(
    data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None, aRePOy_chat_template=None
):
    """
    Preprocess data from raw dataset to prompt, chosen, rejected

    Args:
        data: raw data from dataset
    """

    if aRePOy_chat_template:
        if prompt_key:
            prompt = aRePOy_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = aRePOy_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = aRePOy_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = aRePOy_chat_template(data[prompt_key][:-1], tokenize=False, add_generation_prompt=True)
            chosen = aRePOy_chat_template(data[prompt_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[prompt_key]
        if input_template:
            prompt = input_template.format(prompt)
        chosen = data[chosen_key]
        rejected = data[rejected_key]
    return prompt, chosen, rejected


class UnpairedPreferenceDatasetForPairedDataset(Dataset):
    """
    Unpaired preference dataset for algorithm, like KTO

    Args:
        dataset: raw dataset
        self.tokenizer: self.tokenizer for model
        self.max_length: max length of input
    """

    def __init__(
        self, dataset, tokenizer: Callable, max_length: int, strategy, input_template=None, num_processors=8
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.aRePOy_chat_template = getattr(self.strategy.args, "aRePOy_chat_template", False)

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
        self.prompts = processed_dataset["prompt"] + processed_dataset["prompt"]
        self.responses = processed_dataset["chosen"] + processed_dataset["rejected"]
        self.labels = [1] * len(processed_dataset["chosen"]) + [0] * len(processed_dataset["rejected"])
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"] + processed_dataset["prompt_ids_len"]

    def process_data(self, data):
        prompt, chosen, rejected = preprocess_data(
            data, self.input_template, self.prompt_key, self.chosen_key, self.rejected_key, self.aRePOy_chat_template
        )
        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        # filter the sample whose length is greater than max_length (2 for answer length)
        if prompt_ids_len >= self.max_length - 2:
            prompt = None

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "prompt_ids_len": prompt_ids_len}

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index], self.responses[index], self.labels[index], self.prompt_ids_lens[index]

    def collate_fn(self, item_list):
        def tokenizer(prompt, response):
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
            inputs["attention_mask"][0][-1] = True
            return inputs["input_ids"], inputs["attention_mask"]

        tot_ids, tot_masks, tot_labels, prompt_ids_lens = [], [], [], []
        for prompt, response, label, prompt_ids_len in item_list:
            input_ids, attention_mask = tokenizer(prompt, response)
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(label)
            prompt_ids_lens.append(prompt_ids_len)

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        for idx in range(len(item_list)):
            next_idx = (idx + 1) % len(item_list)
            input_ids, attention_mask = tokenizer(item_list[idx][0], item_list[next_idx][1])
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(-1)
            prompt_ids_lens.append(item_list[idx][3])

        input_ids = zero_pad_sequences(tot_ids, side="right", value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(tot_masks, side="right")
        return input_ids, attention_mask, torch.LongTensor(tot_labels), prompt_ids_lens
