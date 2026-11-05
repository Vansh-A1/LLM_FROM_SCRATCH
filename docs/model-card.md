# Model card: GPT-2 Small trained from scratch

## Model overview

A decoder-only GPT-2 Small base language model trained from scratch on an OpenWebText token corpus. It predicts the next token and continues prompts. It is **not instruction-tuned** and should not be treated as a reliable question-answering assistant.

| Property | Value |
|---|---|
| Unique parameters | 124,439,808 |
| Transformer blocks | 12 |
| Attention heads | 12 |
| Embedding width | 768 |
| Context length | 1,024 tokens |
| Tokenizer | GPT-2 byte-pair encoding, vocabulary 50,257 |
| Embedding/output head | Tied |
| Activation | Tanh-approximation GELU |
| Training dropout | 0.0 |
| Completed optimizer updates | 17,234 |
| Training-token exposures | 9,035,579,392 |
| Training limit | One epoch-equivalent token budget |
| Distributed run | Three ranks across three machines |

The training used BF16, AdamW (β₁=0.9, β₂=0.95, ε=1e−8, weight decay 0.1), gradient clipping at 1.0 and 524,288 tokens per optimizer update. The configured learning-rate schedule had **2,000 warmup updates, a peak of 6e−4, a floor of 6e−5 and a 600,000-update cosine-decay horizon**. The separate one epoch-equivalent token limit stopped the actual run at update 17,234, when the learning rate remained approximately 5.991e−4. This mismatch is documented context; these observations do not establish it as the cause of a generation weakness.

## Training data and provenance

The prepared training split contained 9,035,582,489 tokens from 8,009,762 documents; the validation split contained 4,434,606 tokens from 4,007 documents. Token exposures are the amount consumed by the actual run and are distinct from the full file length. The token loader samples overlapping random windows. “One epoch-equivalent” describes the exposure budget, not a guaranteed traversal of every training token or document.

- Training token-file SHA256: `68a352b5dbd52e85b7a8a1556c1ccae65ce14b02e4913788b0883b89ceddc1d1`.
- Validation token-file SHA256: `009ecdd6da34f1ae915ff1f0f3aeb0416b02f2252356505b32ce44166dd5cfcc`.
- Final checkpoint SHA256: `6fe9cc53a9b3890d682c54208a54ce696ed8be98fbff414d1fa19b212aa3b1fe`.
- Original model-definition SHA256: `9e1416e3a2fe393097b05c51575ae434bfea464bbc2d8c82c9ee909115945a40`.

Weights, training/validation token files, raw corpus data and private machine recovery configuration are not distributed through this Git repository. Dataset/model use remains subject to the respective upstream terms; this card does not grant rights to upstream content.

## Evaluation

The final sampled validation loss was **3.1298661**, with perplexity **22.8709**. Exact saved-sample replay reproduced this result within small backend/precision differences. A separate full-file FP32 protocol measured cross-entropy **3.0556138** and perplexity **21.2342** over 4,434,605 targets. These measurements have different sampling/context coverage and are not an improvement-over-time comparison.

Full zero-shot raw accuracies were LAMBADA **29.2451%**, HellaSwag **27.7236%**, PIQA **60.9358%** and ARC-Easy **42.1296%**. The external pretrained GPT-2 reference scored higher on all four point estimates, but its training data/history differ. Paired intervals and normalized scores are in the [evaluation report](evaluation.md); the ARC-Easy interval does not establish a tie or equivalence.

## Intended uses and limitations

Suitable uses include education, inspection of an end-to-end language-model training pipeline, controlled next-token experiments and research on small-model generation. The model has learned readable English continuation patterns, but sustained coherence and factual consistency remain unreliable. It can drift off topic, repeat text or produce incorrect facts. Base-model continuations should not be interpreted as instruction-following capability.

The fixed generation study contained 600 total outputs across the trained model and reference. Both models showed greedy exact-loop flags on 16/20 ordinary-prose prompts. The trained model produced whitespace-only outputs on 35/45 sampled continuations of five indented code prompts, compared with 0/45 for the reference. No ordinary-prose prompt produced a whitespace-only output. Code was not executed; the prompt suite does not measure general coding accuracy.

Only 120 primary-model outputs received AI semantic review; no matched reference semantic review or human-rater study was performed. No comprehensive bias, toxicity, privacy, memorization, adversarial robustness or training/benchmark decontamination study was completed. Web-corpus risks therefore remain unquantified. The model is unsuitable as a verified source for consequential decisions without independent validation.

## Reproducibility

See [evaluation instructions](../evaluation/README.md), [benchmark results](../results/benchmarks.json) and [validation results](../results/validation.json). The checkpoint was verified unchanged after evaluation. Publication reorganized source and curated existing results; it did not retrain the model or modify its learned weights.
