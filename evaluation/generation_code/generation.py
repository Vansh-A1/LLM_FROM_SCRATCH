"""Frozen, resumable inference-only GPT-2 generation evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT
EXPERIMENT = ROOT.parent
EOS = 50256
METHODS = {
    'greedy': {'temperature': None, 'top_k': None, 'top_p': None, 'seeds': [None]},
    'top_k': {'temperature': 0.8, 'top_k': 40, 'top_p': None, 'seeds': [42, 123, 2026]},
    'top_p': {'temperature': 0.8, 'top_k': None, 'top_p': 0.9, 'seeds': [42, 123, 2026]},
    'combined': {'temperature': 0.8, 'top_k': 40, 'top_p': 0.9, 'seeds': [42, 123, 2026]},
}
PROMPTS = {
    'sentence_continuation': [
        'The little boy walked into the forest and',
        'Every morning, the scientist opened her notebook and',
        'The train arrived at the station just as',
        'The history of artificial intelligence began with',
        'When Maria found the old photograph, she remembered',
        'After months of practice, the young musician finally',
        'The power went out during dinner, so the family',
        'At the edge of the village stood a house that',
    ],
    'encyclopedic_continuation': [
        'The solar system consists of',
        'The Earth is the third planet from the Sun and',
        'A dog is a domesticated mammal that',
        'The human brain is responsible for',
        'The Pacific Ocean is',
        'The printing press changed society by',
    ],
    'explanatory_continuation': [
        'Photosynthesis is the process by which plants',
        'Ice floats on water because',
        'Rain forms when water vapor',
        'Exercise can improve cardiovascular health by',
        'A bicycle stays balanced while moving because',
        'When a metal is heated, it usually expands because',
    ],
    'direct_question': [
        'What is the capital of France?',
        'Why is the sky blue?',
        'What is machine learning?',
        'How many days are there in a leap year?',
        'What causes the seasons on Earth?',
    ],
    'code_continuation': [
        'def add(a, b):\n    ',
        'def fibonacci(n):\n    ',
        'def is_even(n):\n    ',
        'def factorial(n):\n    ',
        'def reverse_string(text):\n    ',
    ],
}

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def atomic_new(path, payload):
    """Publish one complete file without replacing an existing file, even in races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('x', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()

def encoder():
    return tiktoken.get_encoding('gpt2')

def output_id(prompt_id, method, seed):
    return f'{prompt_id}__{method}__' + ('none' if seed is None else str(seed))

def freeze():
    enc = encoder()
    prompts = []
    for cat_index, (category, texts) in enumerate(PROMPTS.items()):
        for index, prompt in enumerate(texts, 1):
            prompts.append({'id': f'{chr(65 + cat_index)}{index:02d}', 'category': category,
                            'text': prompt, 'sha256_utf8': sha(prompt),
                            'token_ids': enc.encode(prompt, disallowed_special=())})
    assert [len(v) for v in PROMPTS.values()] == [8, 6, 6, 5, 5]
    spec = {'schema_version': 1, 'prompts': prompts, 'tokenizer': 'tiktoken:gpt2',
            'vocab_size': enc.n_vocab, 'eot_token_id': enc.eot_token,
            'max_new_tokens': 128, 'methods': METHODS,
            'eot_handling': 'Stop after generating EOT; save all IDs and raw decoded text; visible completion excludes terminal EOT.',
            'review_selection_rule': 'All 30 greedy outputs plus seed 42 for every prompt and each sampled decoder: 120 outputs.',
            'review_output_ids': [output_id(p['id'], m, s) for p in prompts for m in METHODS
                                  for s in ([None] if m == 'greedy' else [42])],
            'metric_conventions': {'unusual_whitespace_run_chars': 16,
                'loop_min_period_tokens': 2, 'loop_max_period_tokens': 32,
                'loop_min_repeats': 3, 'loop_min_span_tokens': 12}}
    spec['protocol_sha256'] = sha(canonical(spec))
    spec['frozen_at_utc'] = now()
    path = ROOT / 'GENERATION_PROMPTS.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['protocol_sha256'] != spec['protocol_sha256']:
            raise RuntimeError('Existing frozen protocol differs; refusing to alter it.')
        return old
    atomic_new(path, json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    return spec

def filtered_logits(logits, method):
    if method == 'greedy':
        return logits
    settings = METHODS[method]
    logits = logits / settings['temperature']
    if settings['top_k']:
        k = min(settings['top_k'], logits.numel())
        cutoff = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if settings['top_p']:
        ordered, indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative > settings['top_p']
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        mask = torch.zeros_like(remove).scatter(0, indices, remove)
        logits = logits.masked_fill(mask, -torch.inf)
    return logits

def next_logits(model, ids):
    """Project the actual final context position through the unchanged model modules."""
    ids = ids[:, -model.config.block_size:]
    positions = torch.arange(ids.shape[1], dtype=torch.long, device=ids.device)
    x = model.transformer.drop(model.transformer.wte(ids) + model.transformer.wpe(positions))
    for block in model.transformer.h:
        x = block(x)
    return model.lm_head(model.transformer.ln_f(x[:, -1:, :]))[0, 0, :]

@torch.inference_mode()
def generate(model, prompt_ids, method, seed, *, max_new_tokens=128, eos=EOS, timeout_seconds=600):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Generation requires eval mode and frozen parameters.')
    if not prompt_ids or method not in METHODS or max_new_tokens < 1:
        raise ValueError('Invalid generation configuration.')
    device = next(model.parameters()).device
    rng = None if method == 'greedy' else torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated, traces = [], []
    started = time.monotonic()
    reason = 'max_new_tokens'
    for position in range(max_new_tokens):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f'Output exceeded {timeout_seconds}s; incomplete output is not published.')
        logits = next_logits(model, ids).float()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite logits.')
        raw_probs = logits.softmax(dim=-1)
        top_probs, top_ids = torch.topk(raw_probs, 10)
        if method == 'greedy':
            chosen = logits.argmax()
            filtered_probability = 1.0
        else:
            sample_probs = filtered_logits(logits, method).softmax(dim=-1)
            chosen = torch.multinomial(sample_probs, 1, generator=rng)[0]
            filtered_probability = sample_probs[chosen].item()
        token = chosen.item()
        generated.append(token)
        traces.append({'generated_position_zero_based': position, 'token_id': token,
                       'raw_probability': raw_probs[chosen].item(),
                       'sampling_probability': filtered_probability,
                       'space_220_probability': raw_probs[220].item(),
                       'entropy_nats': (-(raw_probs * logits.log_softmax(dim=-1)).sum()).item(),
                       'top_two_probability_margin': (top_probs[0] - top_probs[1]).item(),
                       'top10_token_ids': top_ids.tolist(), 'top10_probabilities': top_probs.tolist()})
        ids = torch.cat([ids, chosen.reshape(1, 1)], dim=1)
        if token == eos:
            reason = 'endoftext'
            break
    return generated, reason, traces

def run(args):
    protocol = freeze()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Root supplies a safely deserialized, strictly verified loader. Importing it must not run training.
    sys.path.insert(0, str(CODE_ROOT))
    from safe_model import configure_paths, load_model
    configure_paths(args.checkpoint,args.source,args.expected_checkpoint_sha256)
    if args.reference_snapshot:
        import reference_model
        reference_model.configure_paths(args.reference_snapshot)
        model, enc, identity = reference_model.load_model(device=args.device)
    else:
        model, enc, identity = load_model(device=args.device)
    model.eval().requires_grad_(False)
    model = model.to(dtype=torch.float32)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Expected FP32 parameters.')
    settings = {'device': args.device, 'precision': 'float32', 'tf32': False,
                'attention': 'torch scaled_dot_product_attention, math backend',
                'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                'no_grad': True, 'eval_mode': True, 'max_new_tokens': 128,
                'tokenizer': 'tiktoken:gpt2', 'eot_token_id': EOS,
                'top_k_ties': 'retain all tied logits at kth cutoff',
                'combined_filter_order': 'temperature, top-k, then top-p on renormalized retained distribution',
                'rng': 'independent torch.Generator on model device; reset per output to its specified seed'}
    if args.device.startswith('cuda'):
        settings['gpu'] = torch.cuda.get_device_name()
    jobs = [(p, method, seed) for method, params in METHODS.items()
            for seed in params['seeds'] for p in protocol['prompts']]
    assert len(jobs) == 300
    records = []
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        for index, (prompt, method, seed) in enumerate(jobs, 1):
            oid = output_id(prompt['id'], method, seed)
            out = ROOT / 'generation_outputs' / f'{oid}.json'
            if out.exists():
                record = json.loads(out.read_text())
                if record['protocol_sha256'] != protocol['protocol_sha256'] or record['checkpoint'] != identity:
                    raise ValueError(f'Resume identity mismatch: {out}')
                records.append(record)
                continue
            begin = time.perf_counter()
            ids, reason, traces = generate(model, prompt['token_ids'], method, seed,
                                          timeout_seconds=args.timeout_seconds)
            elapsed = time.perf_counter() - begin
            content_ids = ids[:-1] if ids and ids[-1] == EOS else ids
            record = {'output_id': oid, 'prompt_id': prompt['id'], 'category': prompt['category'],
                      'prompt': prompt['text'], 'prompt_sha256_utf8': prompt['sha256_utf8'],
                      'prompt_token_ids': prompt['token_ids'], 'method': method, 'seed': seed,
                      'sampling': {k: v for k, v in METHODS[method].items() if k != 'seeds'},
                      'protocol_sha256': protocol['protocol_sha256'], 'checkpoint': identity,
                      'execution': settings, 'created_at_utc': now(), 'generated_token_ids': ids,
                      'generated_length_including_eot': len(ids), 'content_token_ids': content_ids,
                      'content_token_length': len(content_ids), 'raw_decoded_output': enc.decode(ids),
                      'completion': enc.decode(content_ids), 'full_text': prompt['text'] + enc.decode(content_ids),
                      'stopping_reason': reason, 'elapsed_seconds': elapsed,
                      'generated_tokens_per_second': len(ids) / elapsed, 'probability_trace': traces}
            atomic_new(out, json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
            print(f'{index}/300 {oid} tokens={len(ids)} stop={reason} seconds={elapsed:.2f}', flush=True)
    final = ROOT / 'GENERATION_RESULTS.jsonl'
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    if final.exists():
        if final.read_text() != data:
            raise ValueError('Existing generation aggregate differs; refusing overwrite.')
    else:
        atomic_new(final, data)
    print(f'COMPLETE: {len(records)} generation records; protocol {protocol["protocol_sha256"]}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--freeze-only', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--source', type=Path, default=CODE_ROOT.parent/'gpt2/train.py')
    p.add_argument('--expected-checkpoint-sha256')
    p.add_argument('--reference-snapshot', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--timeout-seconds', type=float, default=600)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    ROOT = args.output_root.resolve()
    if args.freeze_only:
        print(json.dumps(freeze(), indent=2))
    else:
        run(args)
