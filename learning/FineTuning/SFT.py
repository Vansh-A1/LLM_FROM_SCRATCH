import json
import torch
import torch.nn as nn
import tiktoken

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

from gpt_model import GPTModel, GPT_CONFIG


DATA_PATH = "train_sft.jsonl"
PRETRAINED_PATH = "best_pretrained_model.pth"
SAVE_PATH = "best_sft_model.pth"

CONTEXT_LENGTH = 256
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SFTDataset(Dataset):

    def __init__(self, file_path, context_length=256):
        super().__init__()

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_id = self.tokenizer.eot_token
        self.dataset = []

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                item = json.loads(line)

                prompt = (
                    f"### User:\n{item['instruction']}\n"
                    f"### Assistant:\n"
                )

                response = f" {item['response']}"

                prompt_ids = self.tokenizer.encode_ordinary(prompt)

                response_ids = (
                    self.tokenizer.encode_ordinary(response)
                    + [self.eot_id]
                )

                full_ids = prompt_ids + response_ids

                if len(full_ids) - 1 > context_length:
                    continue

                input_ids = full_ids[:-1]
                target_ids = full_ids[1:]

                target_ids[:len(prompt_ids) - 1] = (
                    [-100] * (len(prompt_ids) - 1)
                )

                padding_length = context_length - len(input_ids)

                input_ids += [self.eot_id] * padding_length
                target_ids += [-100] * padding_length

                self.dataset.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long)
                ))

        if len(self.dataset) < 2:
            raise ValueError("Need at least 2 valid SFT examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


dataset = SFTDataset(
    DATA_PATH,
    context_length=CONTEXT_LENGTH
)

val_size = max(1, int(0.1 * len(dataset)))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = GPTModel(GPT_CONFIG)

checkpoint = torch.load(
    PRETRAINED_PATH,
    map_location="cpu",
    weights_only=True
)

if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint, strict=True)
model = model.to(DEVICE)

print("Pretrained model loaded successfully!")


criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0
    train_tokens = 0

    for input_ids, target_ids in train_loader:

        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)

        predictions = model(input_ids)

        loss = criterion(
            predictions.reshape(-1, predictions.size(-1)),
            target_ids.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        valid_tokens = (target_ids != -100).sum().item()

        train_loss_sum += loss.item() * valid_tokens
        train_tokens += valid_tokens

    avg_train_loss = train_loss_sum / train_tokens

    model.eval()

    val_loss_sum = 0
    val_tokens = 0

    with torch.no_grad():

        for input_ids, target_ids in val_loader:

            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            predictions = model(input_ids)

            loss = criterion(
                predictions.reshape(-1, predictions.size(-1)),
                target_ids.reshape(-1)
            )

            valid_tokens = (target_ids != -100).sum().item()

            val_loss_sum += loss.item() * valid_tokens
            val_tokens += valid_tokens

    avg_val_loss = val_loss_sum / val_tokens

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": avg_val_loss
        }, SAVE_PATH)

        print("Best SFT model saved!")


checkpoint = torch.load(
    SAVE_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nSFT training completed!")


tokenizer = tiktoken.get_encoding("gpt2")

prompt = (
    "### User:\nWhat is gradient descent?\n"
    "### Assistant:\n"
)

input_ids = tokenizer.encode_ordinary(prompt)

input_tensor = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=DEVICE
)

generated_ids = []

with torch.no_grad():

    for _ in range(100):

        context = input_tensor[:, -CONTEXT_LENGTH:]

        logits = model(context)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        token_id = next_token.item()

        if token_id == tokenizer.eot_token:
            break

        generated_ids.append(token_id)

        input_tensor = torch.cat(
            [input_tensor, next_token],
            dim=1
        )

print("\nUser: What is gradient descent?")
print("Assistant:", tokenizer.decode(generated_ids))
