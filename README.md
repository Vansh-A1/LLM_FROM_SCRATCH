# LLM From Scratch

**Building a language model one component at a time, with Python and PyTorch.**

This is my ongoing implementation journal for understanding how text becomes tokens, how training examples are constructed, and how neural networks learn to predict the next token. Each stage connects a small, readable implementation to the idea behind it.

> **Current stage:** language-model foundations — tokenization, sliding-window input/target pairs, and token embeddings. Attention, the Transformer model, and end-to-end training are the next milestones.

## What is implemented

| Component | Implementation | What to look for |
| --- | --- | --- |
| Basic tokenizer | [`Basics/tokenizer.py`](Basics/tokenizer.py) | Vocabulary construction, token-to-ID mapping, and decoding |
| GPT-2 tokenization | [`Basics/bytePair_enoding.py`](Basics/bytePair_enoding.py) | Encoding and decoding with `tiktoken` |
| Training pairs | [`Basics/input_output_pairs.py`](Basics/input_output_pairs.py) | Overlapping context windows, one-token-shifted targets, and PyTorch data loading |
| Token embeddings | [`Basics/embeding.py`](Basics/embeding.py) | Mapping token IDs to learned 256-dimensional vectors |
| Example corpus | [`archive/the-verdict.txt`](archive/the-verdict.txt) | Text used by the foundation examples |

The simple tokenizer is implemented directly. The GPT-2 example uses the existing BPE tokenizer supplied by `tiktoken`; training a BPE tokenizer is a future extension.

## The learning path

```text
Text → Token IDs → Context / Target Pairs → Token Embeddings
                                             ↓
                          Causal Attention → Transformer → Next-Token Training
```

The first four components are present. The attention and model-training stages are planned.

A next-token training pair shifts the target by one position:

```text
Input:   [t0, t1, t2]
Target:  [t1, t2, t3]
```

With batch size `B`, context length `T`, and embedding dimension `D`, token IDs have shape `[B, T]` and embeddings have shape `[B, T, D]`. The embedding demonstration uses `D = 256`.

## Getting started

Use Python 3.10+ and a virtual environment:

```bash
git clone https://github.com/Vansh-A1/LLM_FROM_SCRATCH.git
cd LLM_FROM_SCRATCH
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

Run the examples from the repository root, in this order:

```bash
python Basics/tokenizer.py
python Basics/bytePair_enoding.py
python Basics/input_output_pairs.py
python Basics/embeding.py
```

The BPE example prompts for text. The dataset example prints input/target token tensors; the embedding example prints their shapes. `tiktoken` may download its encoding data on first use.

The basic tokenizer intentionally uses a small corpus-derived vocabulary. Unseen words raise a `KeyError`, and punctuation is discarded by its regular expression. These make useful next exercises in unknown-token handling and reversible tokenization.

## Development roadmap

- [x] Build and inspect a simple vocabulary-based tokenizer.
- [x] Explore GPT-2 BPE encoding and decoding.
- [x] Construct sliding-window next-token training pairs.
- [x] Map tokens into embedding vectors.
- [ ] Add positional embeddings and explain tensor shapes.
- [ ] Implement self-attention, causal masking, and multi-head attention.
- [ ] Assemble a GPT-style Transformer block and model.
- [ ] Add a reproducible training loop, validation split, and checkpointing.
- [ ] Generate text with temperature, top-k, and top-p sampling.
- [ ] Publish learning curves, sample outputs, and evaluation notes.

These are development milestones, with each completed stage intended to include an explanation and a runnable example.

## How to study the code

1. Change the input text and inspect how the token IDs change.
2. Vary context length and stride; compare overlapping training examples.
3. Inspect the embedding shapes before adding attention.
4. Record the expected behavior, observed output, and open questions for each experiment.

## Contributing

Explanations, small reproducible examples, and focused fixes are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the format.

Built and maintained by [Vansh Joshi](https://github.com/Vansh-A1).

