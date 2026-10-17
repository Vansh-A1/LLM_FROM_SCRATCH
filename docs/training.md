# Training guide

The model lives in `llm/`. The active input pipeline and training loop live in
`training/`. Run commands from the repository root.

## Local input files

Use an existing prepared UTF-8 corpus. Dataset acquisition and cleaning tools are
not part of this repository.

```text
/path/to/prepared/corpus/
├── train.txt
├── train.documents.jsonl
├── validation.txt
└── validation.documents.jsonl
```

Each JSONL index has one object per document. The loader uses these two fields:

```json
{"byte_start": 0, "byte_end": 1000}
```

Those numbers are an example of the schema, not a bundled document. Offsets refer
to bytes in the corresponding TXT file, and `byte_end` is exclusive. Documents
must be nonempty, ordered, nonoverlapping, within the file, and valid UTF-8. Extra
provenance fields may remain in the index. Keep whole documents in separate
training and validation splits.

The loader reads one document at a time, tokenizes it with GPT-2, adds an
end-of-text token, and packs contiguous next-token windows. It does not load the
entire corpus into memory. Use `num_workers=0`; this loader rejects worker copies
that would duplicate the data stream.

## Choose paths and start

```bash
export LLM_DATA_DIR="/absolute/path/to/prepared/corpus"
export LLM_OUTPUT_DIR="/absolute/path/to/training/run"
python -m training.train
```

If the environment variables are absent, inputs default to `data/` and outputs
to `runs/run_300m_3epochs/`, both relative to the repository root and ignored by Git.
The input files must already exist. The trainer does not download or clean data.

## Run settings

The settings are explicit variables at the top of `training/train.py`:

- `epochs = 3` and `train_tokens = 300_000_000`.
- `context_length = 1024`, `batch_size = 4`, and `gradient_accumulation_steps = 4`.
- `learning_rate = 3e-4`, with 20 warmup updates and gradient-norm clipping at 1.0.
- `validation_every = 10` optimizer updates and `validation_batches = 10`.
- `use_mixed_precision = True`: BF16 is used on supported CUDA hardware.
- `resume_training = False`: the default starts with fresh weights.

The 300M-token cap selects the same prefix of the prepared training stream each
epoch, before its bounded shuffle. A new epoch changes the shuffle seed. The
validation file is separate and has no training-token cap. Validation checks read
its first 10 batches consistently, rather than traversing the full split.

One full update normally processes 16 sequences, or 16,384 target tokens. The last
update can be smaller; its micro-batch losses are weighted by their token counts.
At these defaults, each epoch has 299,999,232 usable target tokens and 18,311
optimizer updates. Three epochs expose the model to that same subset three times.

These are experiment settings, not universal hardware recommendations. Batch size
4 at context 1024 was checked on an NVIDIA RTX 2000 Ada Generation with 16 GB VRAM.

## Validation and graphs

Every 10 optimizer updates, and after the final update of each epoch, the loop:

1. Switches the model and both embedding layers to evaluation mode.
2. Computes token-weighted validation loss without gradients.
3. Restores training mode.
4. Writes the loss history and refreshes the plot.

`loss_history.csv` records training loss per optimizer update, sampled validation
loss, epoch, step, and learning rate. `loss_plot.png` shows both loss curves and
is refreshed after each check. It is a periodically updated image, not a dashboard.

## Output files

```text
/path/to/training/run/
├── training.log                 # When you redirect terminal output here
├── loss_history.csv
├── loss_plot.png
└── checkpoints/
    ├── epoch_1.pt
    ├── epoch_2.pt
    ├── epoch_3.pt
    └── last.pt
```

The log file is optional: normal launches print to the terminal. Checkpoints are
written at epoch ends, so an unfinished first epoch does not yet have a checkpoint.
Each snapshot includes model and embedding weights, optimizer state, progress,
loss history, run settings, and random-number-generator states. A temporary file
is written before `last.pt` is replaced. The three epoch snapshots are retained.

Keep the machine and training process running for the duration of a run. A normal
foreground launch ends when its terminal session closes; use your preferred
persistent terminal or process manager for a long run.

## Resume

Set `resume_training = True` in `training/train.py`, use the same data and output
paths, and launch the same command. The trainer loads `checkpoints/last.pt`.

Keep the data, token cap, architecture, and other training settings unchanged.
Batch-size changes are supported by tracking the next sequence position. A resume
within an epoch replays the input stream and skips completed sequences, which can
take time for a late checkpoint. Completed epoch checkpoints start at the next
epoch. To repeat a completed experiment, use a new output directory with
`resume_training = False`.

Weights and generated outputs remain local. Do not commit the corpus, credentials,
or checkpoint files.
