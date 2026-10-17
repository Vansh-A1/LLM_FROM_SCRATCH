import json
import os
import random
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import tiktoken


class GPTDataset(IterableDataset):

    def __init__(
        self,
        data_file,
        context_length,
        shuffle_buffer_size=1024,
        seed=42,
        documents_file=None,
        max_tokens=None
    ):
        super().__init__()

        if context_length < 1:
            raise ValueError("context_length must be positive")

        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size cannot be negative")

        if max_tokens is not None and max_tokens < context_length:
            raise ValueError("max_tokens must allow at least one complete window")

        self.data_file = Path(data_file)

        if documents_file is None:
            self.documents_file = self.data_file.with_suffix(".documents.jsonl")
        else:
            self.documents_file = Path(documents_file)

        for file_path in (self.data_file, self.documents_file):

            if not file_path.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.stat().st_size == 0:
                raise ValueError(f"File is empty: {file_path}")

        self.context_length = context_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0
        self.max_tokens = max_tokens

    def set_epoch(self, epoch):

        self.epoch = epoch

    def read_token_windows(self):

        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = []

        file_size = self.data_file.stat().st_size
        previous_end = 0

        # The document index stores byte positions, so use binary mode.
        with self.data_file.open("rb") as text_file:

            with self.documents_file.open("r", encoding="utf-8") as index_file:

                for line_number, line in enumerate(index_file, start=1):

                    document = json.loads(line)
                    start = document["byte_start"]
                    end = document["byte_end"]

                    if start < previous_end or end <= start or end > file_size:
                        raise ValueError(
                            f"Invalid document offsets at index line {line_number}"
                        )

                    previous_end = end

                    # Read and tokenize only one document at a time.
                    text_file.seek(start)
                    document_bytes = text_file.read(end - start)
                    document_text = document_bytes.decode("utf-8")

                    document_tokens = tokenizer.encode(document_text)
                    token_ids.extend(document_tokens)
                    token_ids.append(tokenizer.eot_token)

                    offset = 0

                    while len(token_ids) - offset >= self.context_length + 1:

                        window = torch.tensor(
                            token_ids[
                                offset : offset + self.context_length + 1
                            ],
                            dtype=torch.long
                        )

                        input_chunk = window[:-1]
                        target_chunk = window[1:]

                        yield input_chunk, target_chunk

                        offset += self.context_length

                    # Carry unused tokens into the next document's windows.
                    token_ids = token_ids[offset:]

        # The final short window is dropped instead of padded.

    def __iter__(self):

        if get_worker_info() is not None:
            raise RuntimeError("Use num_workers=0 for this loader")

        random_generator = random.Random(self.seed + self.epoch)
        shuffle_buffer = []

        # Added: limit the text before shuffling so every epoch uses the same subset.
        windows = self.read_token_windows()
        if self.max_tokens is not None:
            windows = islice(windows, self.max_tokens // self.context_length)

        for input_ids, target_ids in windows:

            sample = (input_ids, target_ids)

            if self.shuffle_buffer_size <= 1:
                yield sample

            elif len(shuffle_buffer) < self.shuffle_buffer_size:
                shuffle_buffer.append(sample)

            else:
                random_index = random_generator.randrange(len(shuffle_buffer))

                yield shuffle_buffer[random_index]

                shuffle_buffer[random_index] = sample

        random_generator.shuffle(shuffle_buffer)

        for sample in shuffle_buffer:
            yield sample


def create_dataloader_from_file(
    data_file,
    batch_size=2,
    context_length=256,
    shuffle=True,
    shuffle_buffer_size=1024,
    drop_last=True,
    seed=42,
    documents_file=None,
    max_tokens=None
):

    if not shuffle:
        shuffle_buffer_size = 0

    dataset = GPTDataset(
        data_file=data_file,
        context_length=context_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        documents_file=documents_file,
        max_tokens=max_tokens
    )

    # Shuffling happens inside GPTDataset, so DataLoader only makes batches.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=0
    )

    return dataloader


if __name__ == "__main__":

    data_directory = Path(os.environ.get(
        "LLM_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data"
    ))
    data_file = data_directory / "train.txt"

    dataloader = create_dataloader_from_file(
        data_file=data_file,
        batch_size=2,
        context_length=256,
        shuffle=False
    )

    embeding_dim = 256
    vocab_size = tiktoken.get_encoding("gpt2").n_vocab

    embeding_matrix = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embeding_dim
    )

    for input_tokens, target_tokens in dataloader:

        input_embeddings = embeding_matrix(input_tokens)

        print("First 10 input IDs:", input_tokens[0, :10])
        print("First 10 target IDs:", target_tokens[0, :10])
        print("Input IDs shape:", input_tokens.shape)
        print("Embedding shape:", input_embeddings.shape)
        print("Target IDs shape:", target_tokens.shape)

        break
