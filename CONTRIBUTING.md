# Contributing

Keep changes focused, reproducible, and clear enough to follow from the math to the code.

## Layout

- `gpt2/`: completed 124M experiment, including data preparation and inference.
- `evaluation/`: evaluation implementations and frozen task protocol.
- `configs/` and `results/`: experiment records and measured evidence.
- `docs/`: explanations, model card, methodology, and figures.
- `llm/`, `training/`, `learning/`: historical experiments and exercises.

## Checks

```bash
python scripts/check_repository.py
python gpt2/train.py --smoke-test --device cpu
```

For changes to model, loss, batch allocation, checkpointing, or evaluation, explain the intended behavior and include a focused numerical or round-trip check. Record the command, dependencies, expected result, and observed result. Use separate output directories and never overwrite the published experiment's evidence with a new run.

The training source hash is part of checkpoint provenance. Editing that file creates a new implementation revision: document the new hash and test resume compatibility rather than rewriting the historical source record. Do not edit recorded results to make a new revision appear to reproduce an old run.

Keep weights, datasets, caches, credentials, host addresses, private SSH configurations, and bulky generated logs outside Git. Dataset acquisition code is welcome; dataset redistribution requires separate rights review. Benchmark claims must identify the checkpoint, split, scoring rule, precision, sample count, and limitations.

A project-wide license has not yet been declared; existing third-party license notices must remain intact.
