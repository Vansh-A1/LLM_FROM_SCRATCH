# Reproduce evaluation

These inference-only tools evaluate the audited GPT-2 Small checkpoint and an optional external pretrained GPT-2 reference. They do not create an optimizer or invoke a training entrypoint. The checkpoint and datasets are external inputs; the repository contains source, compact measured results and figures.

## Environment

Run from the repository root with Python 3.13.11 for the recorded environment:

```bash
python -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r evaluation/requirements.txt
```

Install a PyTorch 2.10.0 build suitable for your CUDA driver if using a GPU. CPU execution is supported but full evaluation is slow. Tokenizer files are fetched by tiktoken when absent. Benchmark data can be downloaded at pinned Hugging Face revisions. Model weights are never fetched automatically by these scripts.

## Required inputs and integrity

The default model loader accepts only the audited checkpoint SHA256 `6fe9cc53a9b3890d682c54208a54ce696ed8be98fbff414d1fa19b212aa3b1fe`. It verifies the original model-definition hash, GPT-2 Small configuration, update 17,234, finite weights and tied embedding/output weights. Loading uses PyTorch's restricted `weights_only=True` path with a narrow allowlist for NumPy RNG state.

Pass an explicit checkpoint path. The default architecture is `gpt2/train.py`, a byte-identical copy of the evaluated source. `--source` can select another location of that same source. `--expected-checkpoint-sha256` can identify a different checkpoint with the supported final-step format/configuration, but **its results are not the published audited results**. Do not use these options to relabel an unverified artifact.

For the external reference, obtain the `openai-community/gpt2` snapshot at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and pass its local directory. The exact safetensors hash is enforced. Its training data/history differ from this project.

## Lightweight tests

These tests use synthetic tensors and tiny models; they do not load the 124M checkpoint or train it.

```bash
python evaluation/benchmark_code/test_adapter.py
python evaluation/generation_code/test_generation.py
python evaluation/generation_code/test_metrics.py
python evaluation/test_full_validation.py
```

The adapter tests cover causal shifting, known likelihoods, padding, truncation, cache reuse, token boundaries and invalid modes. Generation tests cover nucleus/top-k selection, isolated RNG, frozen prompt counts and cropped final-position logits. Metric tests check repetition and whitespace cases. Validation tests check complete, non-overlapping target coverage, including the final partial window.

## Standardized benchmarks

Prepare pinned datasets and inspect provenance without loading model weights:

```bash
python evaluation/benchmark_code/run.py --prepare-only --output-dir outputs/benchmarks
```

Smoke-check the first ten documents of each task, then evaluate all four full splits. Smoke results are separate and must not be reported as full benchmark results.

```bash
python evaluation/benchmark_code/run.py \
  --checkpoint /path/to/final.pt --device cuda \
  --smoke --batch-size 8 --max-batch-tokens 2048 \
  --output-dir outputs/benchmarks

python evaluation/benchmark_code/run.py \
  --checkpoint /path/to/final.pt --device cuda \
  --batch-size 32 --max-batch-tokens 4096 \
  --output-dir outputs/benchmarks

python evaluation/benchmark_code/run.py \
  --model reference --reference-snapshot /path/to/gpt2-snapshot \
  --device cuda --batch-size 32 --max-batch-tokens 4096 \
  --output-dir outputs/benchmarks
```

Omit `--device cuda` for CPU. `--tasks lambada_openai hellaswag piqa arc_easy` selects tasks. Final results and request caches live under the requested output directory; repeat an identical command to resume. Use a new output directory when changing checkpoint, code, data or protocol. The runner refuses inconsistent caches and preserves completed result files. Do not run concurrent writers against the same output directory.

For byte-identical historical Arrow exports, add `--arrow-root /path/to/arrow-exports`. Use a subdirectory for each task and the filenames in [dataset_provenance.json](benchmark_code/dataset_provenance.json), for example `arrow-exports/hellaswag/hellaswag-validation.arrow`. All recorded splits and their hashes are checked. Without this flag, the runner loads the pinned Hugging Face revision and verifies expected row counts; archive its newly recorded provenance.

## Generation and automatic analysis

The runner reconstructs the frozen 30-prompt schedule with the same four decoding modes and seeds, saving full text, token IDs and probability traces. Outputs are not edited or filtered.

```bash
python evaluation/generation_code/generation.py \
  --checkpoint /path/to/final.pt --device cuda \
  --output-root outputs/generation-primary
python evaluation/generation_code/analyze.py \
  --output-root outputs/generation-primary

python evaluation/generation_code/generation.py \
  --reference-snapshot /path/to/gpt2-snapshot --device cuda \
  --output-root outputs/generation-reference
python evaluation/generation_code/analyze.py \
  --output-root outputs/generation-reference
```

Generation uses a lossless mapping of the reference's Conv1D tensors into the equivalent model modules; benchmarking uses the Hugging Face reference directly. `--freeze-only --output-root outputs/prompts` records the generation protocol without loading a model. AI semantic reviews are a separate analysis and are not automatically recreated by these scripts.

## Full held-out perplexity

Supply the original validation file: contiguous little-endian uint16 GPT-2 tokens, with SHA256 `009ecdd6da34f1ae915ff1f0f3aeb0416b02f2252356505b32ce44166dd5cfcc`.

```bash
python evaluation/full_validation.py \
  --checkpoint /path/to/final.pt --data /path/to/val.bin \
  --device cuda --output-dir outputs/full-validation
```

This portable runner uses one window per batch and the same exact window/target protocol: context 1,024, stride 512, every target from position 1 through N−1 exactly once, no document-boundary reset. The historical result used batches 4 and 8; roundoff can differ. It does not recreate the distinct sampled training-validation measurement. `--expected-data-sha256` permits an explicitly identified alternate file; that run is a different experiment.

## Provenance

[Publication provenance](../results/evaluation_source_provenance.json) records original and public file hashes and the nature of portability edits. [Measured results](../docs/evaluation.md) explain numerical differences, uncertainty, data overlap limitations, AI review scope and the distinction between sampled validation and full-file perplexity. The published utilities were checked with synthetic tests and CLI/import checks. The expensive model evaluation was not rerun during publication.
