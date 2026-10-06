
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(
            0,
            len(token_ids) - max_length,
            stride
        ):

            input_chunk = token_ids[
                i : i + max_length
            ]

            target_chunk = token_ids[
                i + 1 : i + max_length + 1
            ]


            self.input_ids.append(
                torch.tensor(input_chunk)
            )

            self.target_ids.append(
                torch.tensor(target_chunk)
            )


    def __len__(self):

        return len(self.input_ids)

    def __getitem__(self, index):   
        return (
            self.input_ids[index],
            self.target_ids[index] )


def create_dataloader_v1(raw_text1,batch_size1=32,max_length=128,last_drop1=True,stride1=64,shuffle1=True,num_worker1=0):
    tokenizer = tiktoken.get_encoding("gpt2")


    dataset = GPTDataset(
        text=raw_text1,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride1
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size1,
        shuffle=shuffle1,
        drop_last=last_drop1,
        num_workers=num_worker1

    )

    return dataloader


