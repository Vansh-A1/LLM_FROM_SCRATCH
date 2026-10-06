
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


if __name__ == "__main__":
    with open(
        "/data/vansh/LLM_FROM_SCRATCH/LLM_CODE/data/training_text/pilot_v1/train.txt",
        "r",
        encoding="utf-8"
    ) as file:

        raw_text = file.read()

    DataLoader1=create_dataloader_v1(raw_text1=raw_text,batch_size1=1,stride1=1,max_length=6)

    data_iter=iter(DataLoader1)
    next1=next(data_iter)
    print(data_iter)
    print(next1)

    embeding_dim=256
    vocab_size=tiktoken.get_encoding("gpt2").n_vocab
    embeding_matrix=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embeding_dim)

    dataloader=create_dataloader_v1(raw_text1=raw_text)
    for input_tokens, target_tokens in dataloader:

        input_embeddings = embeding_matrix(input_tokens)

        print("Input IDs shape:", input_tokens.shape)
        print("Embedding shape:", input_embeddings.shape)
        print("Target IDs shape:", target_tokens.shape)
        break
