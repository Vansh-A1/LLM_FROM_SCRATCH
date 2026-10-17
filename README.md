# LLM From Scratch

**Building a language model one component at a time, with Python and PyTorch.**

This is my implementation journal for understanding how text becomes tokens,
how training examples are constructed, and how a Transformer learns to predict
the next token. The code is kept readable so each part can be studied separately.

> **Current stage:** a GPT-style model and a short pilot training loop, including
> held-out validation, gradient clipping, checkpoints, loss history, and plots.

## Project structure

| Directory | Contents |
| --- | --- |
| [`learning/Basics/`](learning/Basics/) | Tokenization, input/target pairs, embeddings, and attention exercises |
| [`LLM_CODE/codes/`](LLM_CODE/codes/) | The current model, streaming data loader, and training script |
| [`LLM_CODE/data/`](LLM_CODE/data/) | Pilot acquisition, standardization, cleaning, and text preparation scripts |
| [`LLM_CODE/actual_data/`](LLM_CODE/actual_data/) | Larger-corpus acquisition and preparation code, manifests, and documentation |

Dataset contents, model weights, checkpoints, and generated training outputs stay
local and are excluded from Git. Source manifests and raw-directory placeholders
remain in the repository.

## Current model and training path

Text → GPT-2 token IDs → input/target windows → token + positional embeddings →
causal Transformer blocks → vocabulary scores → next-token loss.

The main files are:

- [`Dateset_Dataloader.py`](LLM_CODE/codes/Dateset_Dataloader.py): reads indexed
  documents progressively, adds end-of-text separators, packs token windows,
  and shuffles a bounded buffer.
- [`Multi_head_attention.py`](LLM_CODE/codes/Multi_head_attention.py): causal
  multi-head self-attention.
- [`transformer_block.py`](LLM_CODE/codes/transformer_block.py): pre-normalization,
  attention, feed-forward layers, dropout, and residual connections.
- [`model.py`](LLM_CODE/codes/model.py): 12 Transformer blocks, final normalization,
  and a vocabulary output layer.
- [`train.py`](LLM_CODE/codes/train.py): token and learned positional embeddings,
  cross-entropy training, and pilot-run tracking.

The current configuration uses width 768, 12 heads, context length 256, and batch
size 2. Input embeddings and the output layer have separate weights, bringing the
complete model to **162,447,360 parameters**. The original 125M-scale target is a
size reference; this configuration includes both separate vocabulary matrices.

## Setup

Use Python 3.10 or later. The training script has been checked with PyTorch 2.10.
Install a PyTorch build compatible with your hardware, then install the project
dependencies:

```bash
git clone https://github.com/Vansh-A1/LLM_FROM_SCRATCH.git
cd LLM_FROM_SCRATCH
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Prepare local data before training

The small test pilot is separate from the larger `actual_data` corpus. The
trainer expects these files under `LLM_CODE/data/training_text/pilot_v1/`:

- `train.txt` and `train_document_index.jsonl`.
- `validation.txt` and `validation_document_index.jsonl`.

The pilot contains 367 training documents and 43 held-out validation documents.
The files are not included in Git. See [`data/README.md`](LLM_CODE/data/README.md),
[`data/SOURCES.md`](LLM_CODE/data/SOURCES.md), and the acquisition/preparation
scripts for their provenance and workflow. Raw data should remain unchanged.
The larger corpus has separate instructions in
[`actual_data/README.md`](LLM_CODE/actual_data/README.md).

## Run the short pilot

After preparing the local data:

```bash
python LLM_CODE/codes/train.py
```

Settings are near the top of `train.py`:

- Fixed seed: 42.
- Maximum training updates: 200, or the configured epochs if they finish sooner.
- Checkpoint interval: 50 updates.
- Learning-rate warmup: 20 updates; set `warmup_steps` to 0 to disable it.
- Optional BF16 mixed precision on supported CUDA GPUs; otherwise normal precision.
- Validation after each epoch or the shortened pilot segment, using the first
  10 validation batches. This is a sample metric, not a full validation-set score.

Outputs are written under `LLM_CODE/codes/pilot_run/`:

- `checkpoint.pt`: model, embeddings, optimizer, random states, and progress.
- `loss_history.csv`: training losses, sampled validation losses, and learning rates.
- `loss_plot.png`: the recorded training and validation curves.

To continue an interrupted run, set `resume_training = True`. Keep the data and
training settings the same. Increase `max_steps` if the saved run reached the
limit, or `epochs` if it completed all configured epochs. A fresh run refuses to
overwrite an existing checkpoint; choose a new output directory for a separate
experiment. Resuming within an epoch re-reads and skips completed batches.

The small pilot is for checking the training workflow. It is not enough to
establish a fully pretrained model's capabilities. Historical learning examples
also contain local file paths; check those paths before running them elsewhere.

## Next work

- Run and inspect the full-size pilot's loss curves and memory use.
- Add text generation and inspect generated samples.
- Document larger-corpus experiments and evaluation results.

## Contributing

Focused fixes and clear explanations are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md).

Built and maintained by [Vansh Joshi](https://github.com/Vansh-A1).
