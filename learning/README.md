# Learning exercises

These are earlier steps in building the project. The active model imports code
from `llm/` and `training/`; it does not import these exercises.

| Topic | File | What it explores |
| --- | --- | --- |
| Tokenization | [`tokenization/regex_tokenizer.py`](tokenization/regex_tokenizer.py) | A small vocabulary built from regex-split text |
| Tokenization | [`tokenization/gpt2_encoding.py`](tokenization/gpt2_encoding.py) | Encode and decode text with the GPT-2 tokenizer |
| Training examples | [`data_loading/sliding_window_dataset.py`](data_loading/sliding_window_dataset.py) | Input/target pairs from overlapping token windows |
| Embeddings | [`embeddings/token_embeddings.py`](embeddings/token_embeddings.py) | Convert input IDs into embedding vectors |
| Attention | [`attention/causal_attention.py`](attention/causal_attention.py) | Single-head attention and a causal mask |
| Historical draft | [`attention/causal_attention_draft.py`](attention/causal_attention_draft.py) | An unfinished attention/dropout experiment |

## Run an exercise

Run from the repository root after installing `requirements.txt`:

```bash
python -m learning.tokenization.gpt2_encoding
python -m learning.attention.causal_attention
```

Text-based exercises require your own local sample. No books or text datasets
are included. The default sample path is the ignored `data/sample.txt`; set
`LLM_SAMPLE_TEXT` to select another file:

```bash
export LLM_SAMPLE_TEXT="/absolute/path/to/sample.txt"
python -m learning.data_loading.sliding_window_dataset
python -m learning.embeddings.token_embeddings
```

The regex tokenizer is an early demonstration with a fixed example sentence;
its sample corpus must contain that sentence's words. These exercises retain
their original learning logic and may have limitations. In particular,
`causal_attention_draft.py` is unfinished and is not a supported runnable example.

For the model actually used for training, start with [`../llm/model.py`](../llm/model.py)
and [`../training/train.py`](../training/train.py).
