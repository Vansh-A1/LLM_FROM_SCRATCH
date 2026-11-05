# Completed GPT-2 Small experiment

- `prepare.py`: prepare and verify the pinned OpenWebText corpus.
- `train.py`: model, distributed training, checkpointing, and CPU/GPU smoke tests.
- `inference.py`: CPU continuation from an explicit trusted local checkpoint.

Run the documented script entry points from the repository root. Start with the [training guide](../docs/training.md), [model card](../docs/model-card.md), and [experiment record](../configs/gpt2-small-epoch1.json). The source files are unchanged copies from the evaluated run; see [hashes](../results/source_provenance.json).

The trainer's default schedule is 600,000 updates. The completed experiment used `--stop-at 17234` for its fixed one-epoch-equivalent limit. Checkpoints, token binaries, local recovery configuration, and active-run records are not bundled.
