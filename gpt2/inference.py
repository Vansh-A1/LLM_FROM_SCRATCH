#!/usr/bin/env python3
"""CPU text completion for checkpoints produced by this project's train.py.

Example (automatically reads checkpoints/active_run.json and its latest.pt):
  /opt/conda/bin/python inference.py --prompt "The future of artificial intelligence is" \
      --max-new-tokens 128 --output samples/cpu_sample.json

Use --checkpoint PATH to select a specific *trusted local* training checkpoint.
Training checkpoints contain Python/NumPy RNG state and require weights_only=False.
The checkpoint is opened read-only once, so an atomic replacement of latest.pt
while sampling cannot change the weights being used. No checkpoint is copied.

Reference: Andrej Karpathy's official nanoGPT sample.py and GPT.generate:
  https://github.com/karpathy/nanoGPT/blob/master/sample.py
  https://github.com/karpathy/nanoGPT/blob/master/model.py
Uses this project's GPT architecture/tokenizer, eval mode, disabled autograd,
last-position logits, context cropping, temperature, top-k and multinomial sampling.
CPU float32, no autocast/compile, and no CUDA or distributed initialization.
This pretrained base model completes text; it has no chat/instruction template.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import tiktoken
import torch

from train import GPT, GPTConfig, ROOT


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve(strict=True)
    record = json.loads((ROOT / 'checkpoints/active_run.json').read_text())
    run = Path(record['run'])
    if not run.is_absolute():
        run = ROOT / run
    # Never silently fall back to random initialization or a different run.
    return (run / 'latest.pt').resolve(strict=True)


def load_model(path: Path) -> tuple[GPT, dict]:
    # Hash and deserialize the same open inode, even if latest.pt is replaced.
    with path.open('rb') as handle:
        stat = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
        handle.seek(0)
        checkpoint = torch.load(handle, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Expected this project\'s format_version=1 checkpoint.')
    config = GPTConfig(**checkpoint['model_config'])
    if config.vocab_size != 50257:
        raise ValueError('This sampler requires the training run\'s GPT-2 tokenizer (50257 tokens).')
    step = checkpoint['next_step']
    if not isinstance(step, int) or step < 1:
        raise ValueError('Expected a checkpoint containing completed training updates.')
    state = checkpoint['model']
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite model weights: {name}')
    # Both names refer to a single shared parameter in GPT.
    if not torch.equal(state['transformer.wte.weight'], state['lm_head.weight']):
        raise ValueError('Checkpoint token embedding and tied output weights differ.')
    model = GPT(config).to(device='cpu', dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    metadata = {
        'path': str(path), 'sha256': digest.hexdigest(), 'bytes': stat.st_size,
        'saved_at_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        'step': step,
        'tokens_processed': step * checkpoint['train_config']['total_batch_size'],
        'model_config': checkpoint['model_config'],
    }
    del state, checkpoint  # Release optimizer/scaler/RNG state; inference uses weights only.
    gc.collect()
    return model, metadata


def next_token_logits(model: GPT, token_ids: torch.Tensor) -> torch.Tensor:
    """Same transformer as GPT.forward, projecting only the final position."""
    context = token_ids[:, -model.config.block_size:]
    positions = torch.arange(context.size(1), device='cpu', dtype=torch.long)
    x = model.transformer.drop(model.transformer.wte(context) + model.transformer.wpe(positions))
    for block in model.transformer.h:
        x = block(x)
    return model.lm_head(model.transformer.ln_f(x[:, -1:, :]))[:, 0, :]


@torch.inference_mode()
def generate(model: GPT, prompt_ids: list[int], *, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: int = 200, seed: int = 1337,
             eos_token: int | None = 50256) -> tuple[list[int], str]:
    if not prompt_ids:
        raise ValueError('The encoded prompt must contain at least one token.')
    if max_new_tokens < 1 or top_k < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Require max_new_tokens >= 1, top_k >= 0, and finite temperature >= 0.')
    if model.training or any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Generation requires model.eval() on CPU.')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device='cpu')
    reason = 'max_new_tokens'
    for _ in range(max_new_tokens):
        logits = next_token_logits(model, token_ids)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite next-token logits.')
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -float('inf'))
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1, generator=generator)
        token_ids = torch.cat((token_ids, next_id), dim=1)
        if eos_token is not None and next_id.item() == eos_token:
            reason = 'endoftext'
            break
    return token_ids[0, len(prompt_ids):].tolist(), reason


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, help='Trusted checkpoint; default: active run latest.pt')
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt', action='append', help='Text prefix; repeat for multiple prompts')
    prompts.add_argument('--prompt-file', type=Path, help='Use the entire UTF-8 file as one prompt')
    parser.add_argument('--num-samples', type=int, default=1, help='Samples per prompt')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8, help='0 = greedy; default 0.8')
    parser.add_argument('--top-k', type=int, default=200, help='0 = unrestricted; default 200')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--threads', type=int, default=2, help='CPU intra-op threads; default 2')
    parser.add_argument('--ignore-eos', action='store_true', help='Continue past endoftext')
    parser.add_argument('--output', type=Path, help='Save JSON with full text, token IDs and provenance')
    args = parser.parse_args(argv)
    for key in ('num_samples', 'max_new_tokens', 'threads'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if not math.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0:
        parser.error('--temperature must be finite and nonnegative; --top-k must be nonnegative')
    if not 0 <= args.seed < 2**63:
        parser.error('--seed must be in [0, 2**63)')
    if args.output:
        args.output = args.output.expanduser().resolve()
        if args.output.suffix.lower() != '.json':
            parser.error('--output must end with .json')
        # Keep outputs separate from checkpoint and recovery control files.
        if any(args.output.is_relative_to(ROOT / d) for d in ('checkpoints', 'recovery', 'data')):
            parser.error('--output must be outside checkpoints/, recovery/ and data/')
        if args.output.exists():
            parser.error('--output already exists; choose a new filename')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cache = ROOT / '.cache/openwebtext/tiktoken'
    if cache.is_dir():
        os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(cache))
    encoder = tiktoken.get_encoding('gpt2')
    prompts = ([args.prompt_file.read_text(encoding='utf-8')] if args.prompt_file else
               args.prompt if args.prompt is not None else ['<|endoftext|>'])
    encoded = [encoder.encode(p, allowed_special={'<|endoftext|>'}) or [encoder.eot_token]
               for p in prompts]
    path = resolve_checkpoint(args.checkpoint)
    model, checkpoint = load_model(path)
    print(f'CPU float32 | checkpoint update {checkpoint["step"]:,} | '
          f'{checkpoint["tokens_processed"]:,} training tokens | {args.threads} threads',
          file=sys.stderr, flush=True)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': checkpoint,
        'settings': {'device': 'cpu', 'dtype': 'float32', 'threads': args.threads,
                     'torch_version': str(torch.__version__), 'tokenizer': 'gpt2',
                     'temperature': args.temperature, 'top_k': args.top_k,
                     'max_new_tokens': args.max_new_tokens, 'ignore_eos': args.ignore_eos},
        'samples': [],
    }
    for prompt, ids in zip(prompts, encoded):
        for _ in range(args.num_samples):
            seed = (args.seed + len(report['samples'])) % 2**63
            start = time.perf_counter()
            generated, reason = generate(model, ids, max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature, top_k=args.top_k, seed=seed,
                                         eos_token=None if args.ignore_eos else encoder.eot_token)
            elapsed = time.perf_counter() - start
            text = encoder.decode(ids + generated)
            sample = {'prompt': prompt, 'prompt_token_ids': ids, 'seed': seed,
                      'generated_token_ids': generated, 'generated_tokens': len(generated),
                      'completion': encoder.decode(generated), 'text': text,
                      'finish_reason': reason, 'seconds': elapsed,
                      'tokens_per_second': len(generated) / elapsed}
            report['samples'].append(sample)
            print(f'\n--- Sample {len(report["samples"])} (seed {seed}) ---\n{text}', flush=True)
            print(f'{len(generated)} tokens in {elapsed:.2f}s ({len(generated) / elapsed:.2f} tok/s)',
                  file=sys.stderr, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        print(f'Saved {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
