from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDataset(Dataset):
    """Build context windows and their one-token-shifted targets."""

    def __init__(self, text, tokenizer, max_length, stride):
        if max_length <= 0 or stride <= 0:
            raise ValueError("max_length and stride must be positive")

        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(token_ids[i:i + max_length], dtype=torch.long)
            )
            self.target_ids.append(
                torch.tensor(token_ids[i + 1:i + max_length + 1], dtype=torch.long)
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    raw_text1,
    batch_size1=4,
    max_length=128,
    last_drop1=True,
    stride1=64,
    shuffle1=True,
    num_worker1=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(raw_text1, tokenizer, max_length, stride1)
    return DataLoader(
        dataset,
        batch_size=batch_size1,
        shuffle=shuffle1,
        drop_last=last_drop1,
        num_workers=num_worker1,
    )


if __name__ == "__main__":
    corpus_path = Path(__file__).resolve().parents[1] / "archive" / "the-verdict.txt"
    raw_text = corpus_path.read_text(encoding="utf-8")
    dataloader = create_dataloader_v1(
        raw_text1=raw_text, batch_size1=1, stride1=1, max_length=3
    )
    input_ids, target_ids = next(iter(dataloader))
    print("Input IDs:", input_ids)
    print("Target IDs:", target_ids)

