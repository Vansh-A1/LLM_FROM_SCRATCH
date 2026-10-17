# Contributing

Focused fixes and clear explanations are welcome. This is a learning project;
readability and understanding the math matter more than adding abstractions.

## Where changes belong

- `llm/`: the attention layers, Transformer blocks, and current model.
- `training/`: the streaming loader, optimizer loop, validation, and run outputs.
- `learning/`: isolated exercises and explicitly labeled historical drafts.
- `docs/`: setup, input-format, and training documentation.

## Before proposing a change

1. Keep the change focused on one concept or problem.
2. Preserve the author's explicit coding style where practical.
3. For model or training changes, explain the behavior and show a small check.
4. Report the command, environment, expected behavior, and observed result.
5. Keep local datasets, acquisition/cleaning tools, checkpoints, and credentials
   outside the published code tree.

A quick syntax check is:

```bash
python -m compileall -q llm training learning
```

Run a training check only with your own local corpus and a separate output folder.
Do not present an early training-loss value as a model-quality benchmark. When
reporting an evaluation, identify the dataset, metric, run settings, and limits.
