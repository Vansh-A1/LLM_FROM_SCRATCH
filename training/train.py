from llm.model import Model
from training.dataset import create_dataloader_from_file
from pathlib import Path

import torch
import torch.nn as nn
import csv
import matplotlib
import shutil
import os
from itertools import chain, islice

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)

d_in = 768
d_out = 768

context_length = 1024  # Start the new run with the requested context length.
num_heads = 12
d_head = d_out // num_heads

vocab_size = 50257
batch_size = 4  # Checked on the GPU at context length 1024.
gradient_accumulation_steps = 4

epochs = 3
train_tokens = 300_000_000  # Added: use the same 300M-token subset for all three epochs.
learning_rate = 3e-4

# Full training: None removes the pilot cap. Set warmup_steps to 0 to disable warmup.
max_steps = None
warmup_steps = 20
use_mixed_precision = True
resume_training = False  # Start fresh; previous outputs have been preserved separately.
validation_batches = 10
validation_every = 10  # Added: validate and refresh the graph every 10 optimizer updates.

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# BF16 is optional; use normal precision on CPU or unsupported GPUs.
amp_enabled = (
    use_mixed_precision
    and device.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

# Local output paths can be selected without changing the model or training settings.
output_directory = Path(os.environ.get(
    "LLM_OUTPUT_DIR",
    Path(__file__).resolve().parents[1] / "runs" / "run_300m_3epochs"
))
checkpoint_directory = output_directory / "checkpoints"
checkpoint_directory.mkdir(parents=True, exist_ok=True)
checkpoint_file = checkpoint_directory / "last.pt"
temporary_checkpoint = checkpoint_directory / "checkpoint.tmp"
history_file = output_directory / "loss_history.csv"
plot_file = output_directory / "loss_plot.png"

if (checkpoint_file.exists() or any(checkpoint_directory.glob("epoch_*.pt"))) and not resume_training:
    raise FileExistsError("Use a new output_directory or set resume_training = True")

data_directory = Path(os.environ.get(
    "LLM_DATA_DIR",
    Path(__file__).resolve().parents[1] / "data"
))

data_file = data_directory / "train.txt"

train_dataloader = create_dataloader_from_file(
    data_file=data_file,
    documents_file=data_directory / "train.documents.jsonl",
    batch_size=batch_size,
    context_length=context_length,
    shuffle=True,
    max_tokens=train_tokens
)

validation_file = data_file.parent / "validation.txt"

validation_dataloader = create_dataloader_from_file(
    data_file=validation_file,
    documents_file=data_directory / "validation.documents.jsonl",
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
start_sample = 0
loss_history = []
best_validation_loss = float("inf")
history_columns = ["epoch", "step", "train_loss", "validation_loss", "learning_rate"]

training_settings = {
    "data_file": str(data_file),
    "train_tokens": train_tokens,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "context_length": context_length,
    "d_head": d_head,
    "learning_rate": learning_rate,
    "warmup_steps": warmup_steps,
    "mixed_precision": amp_enabled
}

if resume_training:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

    # Allow a new batch size while keeping the other training settings unchanged.
    saved_settings = checkpoint["settings"].copy()
    saved_settings["batch_size"] = batch_size
    if saved_settings != training_settings:
        raise ValueError("Keep settings other than batch_size unchanged when resuming")

    model.load_state_dict(checkpoint["model"])
    token_embedding.load_state_dict(checkpoint["token_embedding"])
    position_embedding.load_state_dict(checkpoint["position_embedding"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    global_step = checkpoint["global_step"]
    start_epoch = checkpoint["epoch"]
    # Older checkpoints counted batches; count samples to allow a new batch size.
    if "next_sample" in checkpoint:
        start_sample = checkpoint["next_sample"]
    else:
        start_sample = checkpoint["next_batch"] * checkpoint["settings"]["batch_size"]
    loss_history = checkpoint["loss_history"]
    best_validation_loss = checkpoint["best_validation_loss"]

    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    if (max_steps is not None and global_step >= max_steps) or start_epoch >= epochs:
        raise ValueError("Increase max_steps or epochs to continue this completed run")

    print(f"Resuming from step {global_step}")

print(f"Device: {device} | BF16 mixed precision: {amp_enabled}")
print(f"Epochs: {epochs} | Training token limit per epoch: {train_tokens:,}")
print(f"Validation every {validation_every} updates, using {validation_batches} batches")

model.train()
token_embedding.train()
position_embedding.train()

for epoch in range(start_epoch, epochs):
    train_dataloader.dataset.set_epoch(epoch)
    epoch_finished = True
    next_sample = 0
    train_iterator = iter(train_dataloader)

    # Skip exactly the completed samples, even when the batch size has changed.
    if epoch == start_epoch:
        for _ in range(start_sample // batch_size):
            next(train_iterator)
        remaining = start_sample % batch_size
        if remaining:
            inputs, targets = next(train_iterator)
            train_iterator = chain([(inputs[remaining:], targets[remaining:])], train_iterator)
        next_sample = start_sample

    # Take up to four CPU batches, then process them one at a time on the GPU.
    micro_batches = list(islice(train_iterator, gradient_accumulation_steps))
    if not micro_batches:
        raise ValueError("Training loader produced no complete token windows")

    while micro_batches:
        if max_steps is not None and global_step >= max_steps:
            epoch_finished = False
            break

        # Added: gradually increase the learning rate during the first updates.
        current_lr = learning_rate
        if warmup_steps > 0:
            current_lr = learning_rate * min(1.0, (global_step + 1) / warmup_steps)

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr

        optimizer.zero_grad()
        update_loss = 0.0
        batch_tokens = sum(target.numel() for inputs, target in micro_batches)

        for input_ids, target_ids in micro_batches:

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

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

            # Average the gradients, including a smaller group at the epoch end.
            loss_weight = target_ids.numel() / batch_tokens
            (loss * loss_weight).backward()
            update_loss += loss.item() * loss_weight

        # Clip and update once after accumulating the micro-batches.
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + list(token_embedding.parameters())
            + list(position_embedding.parameters()),
            max_norm=1.0
        )

        optimizer.step()
        global_step += 1
        next_sample += sum(inputs.shape[0] for inputs, targets in micro_batches)

        loss_history.append({
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": update_loss,
            "validation_loss": None,
            "learning_rate": current_lr
        })

        if global_step == 1 or global_step % 10 == 0:
            print(
                f"Epoch: {epoch + 1} | "
                f"Step: {global_step} | "
                f"Train loss: {update_loss:.4f} | LR: {current_lr:.6f}"
            )

        # Look ahead on the CPU so the final update also gets validation.
        next_micro_batches = list(islice(train_iterator, gradient_accumulation_steps))
        epoch_finished = not next_micro_batches

        if (global_step % validation_every == 0 or epoch_finished
                or (max_steps is not None and global_step >= max_steps)):
            # Added: inline validation every 10 updates and at the end of each epoch.
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
            is_best = validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss

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

            with history_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=history_columns)
                writer.writeheader()
                writer.writerows(loss_history)

            # Added: refresh the loss graph after each validation check.
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

        micro_batches = next_micro_batches

    # Added: keep a checkpoint for each completed epoch, plus last.pt for resuming.
    if epoch_finished:
        next_epoch = epoch + 1
        next_sample = 0
    else:
        next_epoch = epoch

    torch.save(
        {
            "model": model.state_dict(),
            "token_embedding": token_embedding.state_dict(),
            "position_embedding": position_embedding.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": next_epoch,
            "next_sample": next_sample,
            "global_step": global_step,
            "loss_history": loss_history,
            "best_validation_loss": best_validation_loss,
            "settings": training_settings,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        },
        temporary_checkpoint
    )
    temporary_checkpoint.replace(checkpoint_file)

    if epoch_finished:
        shutil.copy2(checkpoint_file, checkpoint_directory / f"epoch_{epoch + 1}.pt")

    start_sample = 0
    if max_steps is not None and global_step >= max_steps:
        break


print(f"Training finished at step {global_step}. Outputs: {output_directory}")
