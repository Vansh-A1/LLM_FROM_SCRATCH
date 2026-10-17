from model import Model
from Dateset_Dataloader import create_dataloader_from_file
from pathlib import Path

import torch
import torch.nn as nn
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)

d_in = 768
d_out = 768

context_length = 256  
num_heads = 12
d_head = d_out // num_heads

vocab_size = 50257
batch_size = 32

epochs = 50
learning_rate = 3e-4

# Added: simple pilot settings. Set warmup_steps to 0 to disable warmup.
max_steps = 200
save_every = 50
warmup_steps = 20
use_mixed_precision = True
resume_training = False
validation_batches = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# BF16 is optional; use normal precision on CPU or unsupported GPUs.
amp_enabled = (
    use_mixed_precision
    and device.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

output_directory = Path(__file__).resolve().parent / "pilot_run"
output_directory.mkdir(parents=True, exist_ok=True)
checkpoint_file = output_directory / "checkpoint.pt"
temporary_checkpoint = output_directory / "checkpoint.tmp"
history_file = output_directory / "loss_history.csv"
plot_file = output_directory / "loss_plot.png"

if checkpoint_file.exists() and not resume_training:
    raise FileExistsError("Use a new output_directory or set resume_training = True")

pilot_directory = (
    Path(__file__).resolve().parents[1]
    / "data" / "training_text" / "pilot_v1"
)

data_file = pilot_directory / "train.txt"

train_dataloader = create_dataloader_from_file(
    data_file=data_file,
    documents_file=pilot_directory / "train_document_index.jsonl",
    batch_size=batch_size,
    context_length=context_length,
    shuffle=True
)

validation_file = data_file.parent / "validation.txt"

validation_dataloader = create_dataloader_from_file(
    data_file=validation_file,
    documents_file=pilot_directory / "validation_document_index.jsonl",
    batch_size=batch_size,
    context_length=context_length,
    shuffle=False,
    drop_last=False
)

# Keep loader randomness separate from model randomness when resuming.
train_dataloader.generator = torch.Generator().manual_seed(42)
validation_dataloader.generator = torch.Generator().manual_seed(43)

token_embedding = nn.Embedding(
    vocab_size,
    d_in
).to(device)

position_embedding = nn.Embedding(
    context_length,
    d_in
).to(device)



model = Model(
    d_in=d_in,
    d_out=d_out,
    d_head=d_head,
    context_length=context_length,
    vocab_size=vocab_size
).to(device)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    list(model.parameters())
    + list(token_embedding.parameters())
    + list(position_embedding.parameters()),
    lr=learning_rate
)


# Added: saved progress lets a run continue from its last checkpoint.
global_step = 0
start_epoch = 0
start_step = 0
loss_history = []
history_columns = ["epoch", "step", "train_loss", "validation_loss", "learning_rate"]

training_settings = {
    "data_file": str(data_file),
    "batch_size": batch_size,
    "context_length": context_length,
    "d_head": d_head,
    "learning_rate": learning_rate,
    "warmup_steps": warmup_steps,
    "mixed_precision": amp_enabled
}

if resume_training:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

    if checkpoint["settings"] != training_settings:
        raise ValueError("Keep the data and training settings unchanged when resuming")

    model.load_state_dict(checkpoint["model"])
    token_embedding.load_state_dict(checkpoint["token_embedding"])
    position_embedding.load_state_dict(checkpoint["position_embedding"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    global_step = checkpoint["global_step"]
    start_epoch = checkpoint["epoch"]
    start_step = checkpoint["next_batch"]
    loss_history = checkpoint["loss_history"]

    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    if global_step >= max_steps or start_epoch >= epochs:
        raise ValueError("Increase max_steps or epochs to continue this completed run")

    print(f"Resuming from step {global_step}")

print(f"Device: {device} | BF16 mixed precision: {amp_enabled}")

model.train()
token_embedding.train()
position_embedding.train()

for epoch in range(start_epoch, epochs):
    train_dataloader.dataset.set_epoch(epoch)
    epoch_finished = True
    next_batch = 0

    for step, (input_ids, target_ids) in enumerate(train_dataloader):

        # Re-read and skip completed batches when resuming within an epoch.
        if epoch == start_epoch and step < start_step:
            next_batch = step + 1
            continue

        if global_step >= max_steps:
            epoch_finished = False
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Added: gradually increase the learning rate during the first updates.
        current_lr = learning_rate
        if warmup_steps > 0:
            current_lr = learning_rate * min(1.0, (global_step + 1) / warmup_steps)

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr

        optimizer.zero_grad()

        # Added: optional mixed precision covers the forward pass and loss.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled
        ):

            # Token embeddings
            token_emb = token_embedding(input_ids)

            # Positional embeddings
            seq_len = input_ids.shape[1]
            positions = torch.arange(seq_len, device=device)
            pos_emb = position_embedding(positions)

            # Combine embeddings and predict the next tokens.
            x = token_emb + pos_emb
            prediction = model(x)

            loss = criterion(
                prediction.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )

        if not torch.isfinite(loss).item():
            raise ValueError(f"Non-finite training loss at step {global_step + 1}")

        # Backpropagation stays outside autocast.
        loss.backward()

        # Added: clip gradients for the model and both embeddings.
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + list(token_embedding.parameters())
            + list(position_embedding.parameters()),
            max_norm=1.0
        )

        optimizer.step()
        global_step += 1
        next_batch = step + 1

        loss_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": loss.item(),
            "validation_loss": None,
            "learning_rate": current_lr
        })

        if step % 10 == 0:
            print(
                f"Epoch: {epoch + 1} | "
                f"Step: {global_step} | "
                f"Train loss: {loss.item():.4f} | LR: {current_lr:.6f}"
            )

        # Added: periodically save progress. Replace the old file only after writing.
        if global_step % save_every == 0 and global_step < max_steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "token_embedding": token_embedding.state_dict(),
                    "position_embedding": position_embedding.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "next_batch": next_batch,
                    "global_step": global_step,
                    "loss_history": loss_history,
                    "settings": training_settings,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                },
                temporary_checkpoint
            )
            temporary_checkpoint.replace(checkpoint_file)

            with history_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=history_columns)
                writer.writeheader()
                writer.writerows(loss_history)

    # Validation runs once after each epoch.
    model.eval()
    token_embedding.eval()
    position_embedding.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for step, (input_ids, target_ids) in enumerate(validation_dataloader):

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled
            ):
                token_emb = token_embedding(input_ids)

                seq_len = input_ids.shape[1]
                positions = torch.arange(seq_len, device=device)
                pos_emb = position_embedding(positions)

                x = token_emb + pos_emb
                prediction = model(x)

                loss = criterion(
                    prediction.reshape(-1, vocab_size),
                    target_ids.reshape(-1)
                )

            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite validation loss")

            # Count tokens so a smaller final batch gets the correct weight.
            number_of_tokens = target_ids.numel()
            total_loss += loss.item() * number_of_tokens
            total_tokens += number_of_tokens

            if step + 1 >= validation_batches:
                break

    if total_tokens == 0:
        raise ValueError("Validation loader produced no complete token windows")

    validation_loss = total_loss / total_tokens

    loss_history.append({
        "epoch": epoch + 1,
        "step": global_step,
        "train_loss": None,
        "validation_loss": validation_loss,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })

    print(
        f"Epoch: {epoch + 1} | Step: {global_step} | "
        f"Validation loss (up to {validation_batches} batches): "
        f"{validation_loss:.4f}"
    )

    model.train()
    token_embedding.train()
    position_embedding.train()

    # Save after validation too, including progress through a shortened pilot epoch.
    if epoch_finished:
        next_epoch = epoch + 1
        next_batch = 0
    else:
        next_epoch = epoch

    torch.save(
        {
            "model": model.state_dict(),
            "token_embedding": token_embedding.state_dict(),
            "position_embedding": position_embedding.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "loss_history": loss_history,
            "settings": training_settings,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        },
        temporary_checkpoint
    )
    temporary_checkpoint.replace(checkpoint_file)

    with history_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history_columns)
        writer.writeheader()
        writer.writerows(loss_history)

    start_step = 0
    if global_step >= max_steps:
        break


# Added: plot the losses already saved in the history.
train_steps = []
train_losses = []
validation_steps = []
validation_losses = []

for record in loss_history:
    if record["train_loss"] is not None:
        train_steps.append(record["step"])
        train_losses.append(record["train_loss"])

    if record["validation_loss"] is not None:
        validation_steps.append(record["step"])
        validation_losses.append(record["validation_loss"])

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="Training loss")
plt.plot(validation_steps, validation_losses, marker="o", label="Validation loss (sample)")
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)
plt.close()

print(f"Pilot finished at step {global_step}. Outputs: {output_directory}")
