#!/usr/bin/env python3
# Portions adapted from nanoGPT, MIT License.
# Copyright (c) 2022 Andrej Karpathy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""GPT-2 small / OpenWebText controlled reproduction. Python >=3.10, PyTorch >=2.4.

Quick start (from this directory; install a CUDA-enabled torch wheel):
  python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install numpy==2.4.2 datasets==4.8.4 tiktoken==0.14.0 tqdm
  python prepare.py --check-only
  python prepare.py
  python train.py --smoke-test --device cpu
  torchrun --standalone --nproc_per_node=3 train.py --smoke-test
  torchrun --standalone --nproc_per_node=3 train.py
  torchrun --standalone --nproc_per_node=3 train.py --resume checkpoints/tokens_2000000000.pt

Target: one 32 GiB GPU and two 16 GiB GPUs, one DDP process per GPU.
For two hosts with 2+1 GPUs, set MASTER_ADDR to the reachable two-GPU host
on BOTH hosts, then launch with the same global WORLD_SIZE=3:
  # Two-GPU host (local devices 0 and 1):
  export MASTER_ADDR=<two-gpu-host-address> MASTER_PORT=29500 WORLD_SIZE=3
  RANK=0 LOCAL_RANK=0 python train.py &
  RANK=1 LOCAL_RANK=1 python train.py &
  wait
  # One-GPU host (local device 0):
  export MASTER_ADDR=<two-gpu-host-address> MASTER_PORT=29500 WORLD_SIZE=3
  RANK=2 LOCAL_RANK=0 python train.py
The installation above matches the locally tested CUDA 12.8 stack; use a wheel
compatible with the server's NVIDIA driver (https://pytorch.org/get-started/previous-versions/).
Both nodes need identical prepared data and source; checkpoint paths must be shared
or copied to each node before resume. MASTER_ADDR is the reachable rank-0 host.

References inspected 2026-11-01:
  https://github.com/karpathy/nanoGPT/blob/master/model.py
    blob c698f8b60129d793494de058b0dd4e318c0dcb5e
  https://github.com/karpathy/nanoGPT/blob/master/train.py
    blob de5785051050d64bbda37a4964c2c079f0739b2b
  https://github.com/karpathy/nanoGPT/blob/master/config/train_gpt2.py
  https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  https://arxiv.org/html/2005.14165v4 (Table 2.1 and Appendix B)

Explicit differences, required by this experiment:
* nanoGPT defaults: bias=False, padded vocab=50304, exact GELU; here GPT-2
  bias=True, vocab=50257, tanh GELU (as in build-nanogpt).
* nanoGPT batch=491520; here build-nanogpt's exact 524288 tokens/update.
* GPT-3 uses context 2048, batch ramp, 375M-token warmup, 260B-token cosine.
  Here context1024, constant global batch, nanoGPT's 2000/600000-step schedule.
* Current nanoGPT warmup is (step+1)/(warmup_iters+1), not the older zero-LR
  first step or build-nanogpt's denominator. We execute indices 0..599999,
  exactly 600000 updates (avoiding nanoGPT's inclusive-loop extra update).
* Matrix parameters INCLUDING tied token/position embeddings decay; 1D
  parameters do not. Cosine changes the LR, never the decay coefficient.
* Random windows use disjoint rank-specific start ranges (same uniform-window
  objective as nanoGPT) to prevent identical sampled starts across ranks.
* Three ranks cannot divide 512 sequences equally. Each update assigns
  171/171/170 sequences, rotating extra sequences by optimizer-step index.
  With microbatch capacity 8: 21 full microsteps plus tails 3/3/2. Multiply
  each mean loss by (actual microbatch sequences * world_size / 512), so
  DDP's rank average yields the exact mean over 524288 tokens. This reduces
  to loss/accumulation_steps for the original equal-size/divisible case.
  A zero-sized tail (e.g. capacity 1 or 2) runs one zero-weight placeholder
  sequence to participate in DDP; it never contributes tokens or gradients.

No full run starts with --smoke-test. CPU smoke uses tiny test models plus the
full architecture's parameter count; CUDA smoke additionally checks the actual
124M model at context1024, selected precision, compiled forward/backward and
DDP gradient synchronization. Test tokens are never written as OpenWebText.
Training always runs these gates before its first optimizer update.

--micro-batch-size auto benchmarks powers of two (plus the rank quota cap) including
AdamW state with 20% memory headroom; the minimum across ranks wins. Compiled
DDP then checks full and tail shapes before training. If that fails for memory, relaunch
with a smaller explicit --micro-batch-size (global tokens stay invariant).
--no-compile is an explicit, logged environmental fallback. Compilation errors
are otherwise fatal with the original diagnostic; there is no silent fallback.
--allow-sdpa-fallback explicitly permits non-Flash SDPA if forced Flash fails.
Compilation specializes the finite set of full/tail batch shapes (dynamic=False).
PyTorch 2.10's DDP graph partitioner failed on automatically dynamic batch graphs
in our three-host check ('int' object has no attribute 'meta'); explicit shape
specialization preserves model/gradient semantics and avoids that compiler path.
Checkpoint resume requires the same world size, microbatch, dtype and kernels
for reproducibility; numerical bitwise identity across CUDA stacks is not promised.
Milestone checkpoints are retained at 2B, 4B and 6B tokens and the final update.
A separate latest.pt recovery checkpoint is saved at initialization and every five
minutes at a completed update. Atomic replacement keeps the previous file intact
until the new file is fully written and fsynced. SIGINT/SIGTERM/SIGHUP request a
collective save and clean exit at the next healthy update boundary; a hard power
loss, SIGKILL, or failed NCCL collective relies on the most recent periodic save.
Touch SAVE_NOW or STOP_AFTER_CHECKPOINT in the output directory to request a save
or a save-and-stop. The supervisor respects PAUSE_AUTORESTART in its config folder.
Use --stop-at for an absolute training limit that automatic resume cannot extend.
Each file includes model, optimizer, scaler, and all ranks' random/data-loader state.
Only load checkpoints created by this script (torch.load uses pickle).
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = True


@dataclass(frozen=True)
class TrainConfig:
    total_batch_size: int = 524288
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eps: float = 1e-8
    warmup_iters: int = 2000
    max_iters: int = 600000
    lr_decay_iters: int = 600000
    use_flash_attention: bool = True
    use_compile: bool = True
    use_fused_adamw: bool = True
    backend: str = 'nccl'
    seed: int = 1337


CFG = TrainConfig()


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        # named_parameters deduplicates the tied embedding. Each parameter gets
        # exactly ONE deliberate final initialization, including residuals.
        for name, p in self.named_parameters():
            if name.endswith('bias'):
                nn.init.zeros_(p)
            elif p.ndim < 2:
                nn.init.ones_(p)
            else:
                std = 0.02 / math.sqrt(2 * config.n_layer) if name.endswith('c_proj.weight') else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.config.block_size
        pos = torch.arange(t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def configure_optimizer(model, device, fused=False):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [dict(params=[p for p in params if p.ndim >= 2], weight_decay=CFG.weight_decay),
              dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0)]
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(params)
    assert set(ids) == {id(p) for p in params}
    kwargs = dict(lr=CFG.learning_rate, betas=(CFG.beta1, CFG.beta2), eps=CFG.eps)
    if 'fused' in inspect.signature(torch.optim.AdamW).parameters:
        kwargs['fused'] = fused
    elif fused:
        raise RuntimeError('This PyTorch version has no fused AdamW.')
    return torch.optim.AdamW(groups, **kwargs)


def get_lr(step):
    assert step >= 0
    if step < CFG.warmup_iters:
        return CFG.learning_rate * (step + 1) / (CFG.warmup_iters + 1)
    if step > CFG.lr_decay_iters:
        return CFG.min_lr
    ratio = (step - CFG.warmup_iters) / (CFG.lr_decay_iters - CFG.warmup_iters)
    return CFG.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (CFG.learning_rate - CFG.min_lr)


def accumulation_steps(micro_batch_size, world_size):
    """Common DDP rounds; partial final microbatches preserve the exact token budget."""
    sequences, remainder = divmod(CFG.total_batch_size, GPTConfig().block_size)
    assert remainder == 0
    if not 1 <= world_size <= sequences:
        raise ValueError(f'world_size must be between 1 and {sequences}')
    largest_quota = (sequences + world_size - 1) // world_size
    if not 1 <= micro_batch_size <= largest_quota:
        raise ValueError(f'Microbatch capacity must be between 1 and {largest_quota}')
    return (largest_quota + micro_batch_size - 1) // micro_batch_size


def accumulation_plan(micro_batch_size, world_size, rank, step=0):
    """Actual sequence counts, including zero-weight slots needed for DDP collectives.

    All ranks execute the same number of rounds and synchronize only the last.
    Rotate the remainder by successful update index, restored from checkpoints.
    """
    rounds = accumulation_steps(micro_batch_size, world_size)
    if not 0 <= rank < world_size or step < 0:
        raise ValueError('Invalid rank or optimizer-step index')
    sequences = CFG.total_batch_size // GPTConfig().block_size
    base, remainder = divmod(sequences, world_size)
    quota = base + int((rank - step) % world_size < remainder)
    plan = tuple(min(micro_batch_size, max(0, quota - i * micro_batch_size))
                 for i in range(rounds))
    assert sum(plan) == quota
    return plan


def microbatch_shapes(micro_batch_size, world_size):
    # Every rotation is a permutation of step zero's rank quotas. Include a
    # one-sequence zero-weight placeholder shape if a rank has an empty tail.
    return sorted({max(1, size) for rank in range(world_size)
                   for size in accumulation_plan(micro_batch_size, world_size, rank)}, reverse=True)


@torch.no_grad()
def verify_initialization(model, verbose=True):
    assert model.lm_head.weight is model.transformer.wte.weight
    report = []
    for name, p in model.named_parameters():
        if name.endswith('bias'):
            assert torch.count_nonzero(p) == 0, name
        elif p.ndim < 2:
            assert torch.equal(p, torch.ones_like(p)), name
        else:
            expected = 0.02 / math.sqrt(2 * model.config.n_layer) if name.endswith('c_proj.weight') else 0.02
            mean, std = p.float().mean().item(), p.float().std().item()
            # Six standard errors; tiny tests get a modest sampling allowance.
            assert abs(mean) < 6 * expected / math.sqrt(p.numel()), (name, mean)
            tolerance = max(0.02, 6 / math.sqrt(2 * (p.numel() - 1)))
            assert abs(std / expected - 1) < tolerance, (name, std, expected)
            if name in {'transformer.wte.weight', 'transformer.wpe.weight',
                        'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_proj.weight',
                        'transformer.h.0.mlp.c_proj.weight'}:
                report.append(dict(parameter=name, mean=mean, std=std, expected_std=expected))
    if verbose:
        print('Initialization:', json.dumps(report), flush=True)
    return report


class TokenLoader:
    """nanoGPT random contiguous windows, with disjoint legal starts per rank.

    Reopen memmap per batch so long runs do not retain an expanding mapped RSS.
    CPU generator state is checkpointed separately for train and validation.
    """
    def __init__(self, path, block_size, rank, world_size, seed):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size % 2 or size <= 2 * block_size:
            raise ValueError(f'Invalid uint16 token file: {self.path}')
        self.block_size = block_size
        starts = size // 2 - block_size
        self.low = starts * rank // world_size
        self.high = starts * (rank + 1) // world_size
        if self.low >= self.high:
            raise ValueError('Too few tokens for distributed windows')
        self.generator = torch.Generator().manual_seed(seed)

    def batch(self, batch_size, device):
        data = np.memmap(self.path, dtype='<u2', mode='r')
        starts = torch.randint(self.low, self.high, (batch_size,), generator=self.generator).tolist()
        tokens = torch.from_numpy(np.stack([
            np.array(data[s:s + self.block_size + 1], dtype=np.int64) for s in starts]))
        del data
        if tokens.min() < 0 or tokens.max() >= 50257:
            raise ValueError(f'Out-of-vocabulary ID in {self.path}; run prepare.py --verify-only')
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        if torch.device(device).type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def read_manifest(directory):
    directory = Path(directory)
    path = directory / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing. Run python prepare.py first; no synthetic substitute is allowed.')
    manifest = json.loads(path.read_text())
    for key, value in dict(schema_version=1, complete=True, dataset='Skylion007/openwebtext',
                           tokenizer='gpt2', vocab_size=50257, dtype='uint16', byte_order='little', eot_token=50256).items():
        if manifest.get(key) != value:
            raise ValueError(f'Invalid data manifest {key}: {manifest.get(key)!r}')
    assert manifest['split'] == dict(validation_fraction=0.0005, seed=2357, shuffle=True)
    assert manifest['source']['revision'] == 'b4325f019c648b1641a1784748667e8b74e5e064'
    hashes = []
    for split in ('train', 'val'):
        info = manifest['files'][split]
        p = directory / f'{split}.bin'
        assert info['filename'] == p.name
        assert info['documents'] == (8009762 if split == 'train' else 4007)
        assert p.stat().st_size == info['bytes'] == info['tokens'] * 2
        assert 0 <= info['min_token'] <= info['max_token'] < 50257
        assert info['tokens'] > GPTConfig().block_size
        hashes.append(info['sha256'])
    assert hashes[0] != hashes[1], 'Train and validation cannot be the same dataset'
    # Full file range/hash validation is prepare.py --verify-only. On launch,
    # check manifest/lengths and each sampled batch to avoid rereading 18GB/rank.
    return manifest


def capture_rng(device):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.device(device).type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(os, 'O_DIRECTORY'):
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class Runtime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self):
        return dist.is_initialized()

    def log(self, message):
        if self.rank == 0:
            print(message, flush=True)

    def all(self, value):
        flag = torch.tensor(int(value), device=self.device)
        if self.distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def mean(self, tensor):
        tensor = tensor.detach().clone()
        if self.distributed:
            dist.all_reduce(tensor)
            tensor /= self.world_size
        return tensor


def setup_runtime(args):
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    if args.device == 'cuda' and not cuda:
        raise RuntimeError('CUDA requested but unavailable')
    if not cuda and not args.smoke_test:
        raise RuntimeError('Full training requires CUDA. CPU checks: --smoke-test --device cpu')
    device = torch.device(f'cuda:{local}' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    if 'RANK' in os.environ:
        # gloo is ONLY a CPU test backend, never the experiment's backend.
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(minutes=30))
    return Runtime(rank, local, world, device)


def autocast_context(runtime, dtype):
    return torch.autocast('cuda', dtype=dtype) if runtime.device.type == 'cuda' else nullcontext()


def check_hardware(runtime, args):
    if runtime.device.type == 'cpu':
        runtime.log('CPU smoke: CUDA FlashAttention, fused AdamW, CUDA compile and NCCL skipped.')
        return torch.float32, False, False
    prop = torch.cuda.get_device_properties(runtime.device)
    free, total = torch.cuda.mem_get_info(runtime.device)
    info = dict(rank=runtime.rank, name=prop.name, capability=list(torch.cuda.get_device_capability()),
                total_GiB=total / 2**30, free_GiB=free / 2**30,
                bf16=torch.cuda.is_bf16_supported(), torch=torch.__version__, cuda=torch.version.cuda)
    runtime.log('Hardware: ' + json.dumps(runtime.gather(info)))
    dtype = torch.bfloat16 if runtime.all(info['bf16']) else torch.float16
    runtime.log(f'Precision: {dtype}; FP16 GradScaler enabled={dtype == torch.float16}')
    flash, error = True, None
    try:
        tensors = [torch.randn(1, 12, 128, 64, device=runtime.device, dtype=dtype,
                               requires_grad=True) for _ in range(3)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(*tensors, is_causal=True)
            gradients = torch.autograd.grad(out.float().square().mean(), tensors)
        references = [t.detach().float().requires_grad_(True) for t in tensors]
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(*references, is_causal=True)
            expected_grads = torch.autograd.grad(expected.square().mean(), references)
        torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
        for actual, target in zip(gradients, expected_grads):
            torch.testing.assert_close(actual.float(), target, atol=1e-5, rtol=0.1)
        # A single forced backend cannot silently select math. Compare both
        # outputs and derivatives against a separate FP32 math-kernel run.
        del tensors, out, gradients, references, expected, expected_grads
    except (RuntimeError, AssertionError) as exc:
        flash, error = False, repr(exc)
    errors = runtime.gather(error)
    flash = runtime.all(flash)
    if not flash and not args.allow_sdpa_fallback:
        raise RuntimeError(f'Forced CUDA FlashAttention failed: {errors}. Explicit fallback: --allow-sdpa-fallback')
    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_math_sdp(not flash)
    torch.backends.cuda.enable_mem_efficient_sdp(not flash)
    torch.backends.cuda.enable_cudnn_sdp(False)
    runtime.log(f'Attention: {"FLASH_ATTENTION forced, forward/backward compared with FP32 math" if flash else "SDPA fallback: " + str(errors)}')
    fused, error = True, None
    try:
        p = nn.Parameter(torch.ones(4, device=runtime.device))
        opt = torch.optim.AdamW([p], lr=CFG.learning_rate, betas=(CFG.beta1, CFG.beta2),
                               eps=CFG.eps, fused=True)
        p.grad = torch.ones_like(p)
        opt.step()
        assert torch.isfinite(p).all()
        del p, opt
    except (RuntimeError, TypeError, AssertionError) as exc:
        fused, error = False, repr(exc)
    errors = runtime.gather(error)
    fused = runtime.all(fused)
    runtime.log(f'Fused AdamW: {fused}' + (f'; standard AdamW fallback: {errors}' if not fused else ' (real optimizer step verified)'))
    torch.cuda.empty_cache()
    return dtype, fused, flash


def probe_microbatch(raw, runtime, dtype, fused, requested):
    """No model update: lr=0 materializes Adam moments, then state is discarded."""
    if requested != 'auto':
        batch = int(requested)
        accumulation_steps(batch, runtime.world_size)
        return batch
    sequences = CFG.total_batch_size // GPTConfig().block_size
    max_batch = (sequences + runtime.world_size - 1) // runtime.world_size
    candidates = sorted({2**power for power in range(max_batch.bit_length())} | {max_batch})
    chosen = 0
    saved_rng = capture_rng(runtime.device)
    for batch in candidates:
        success, error, fatal = True, None, False
        optimizer = configure_optimizer(raw, runtime.device, fused=fused)
        for group in optimizer.param_groups:
            group['lr'] = 0.0
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(runtime.device)
        budget = torch.cuda.memory_allocated(runtime.device) + int(0.80 * free)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        x = y = logits = loss = None
        try:
            x = torch.randint(50257, (batch, 1024), device=runtime.device)
            y = torch.randint(50257, x.shape, device=runtime.device)
            with autocast_context(runtime, dtype):
                logits, loss = raw(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), CFG.grad_clip, error_if_nonfinite=True)
            optimizer.step()  # lr=0: weights unchanged, no decay either.
            torch.cuda.synchronize(runtime.device)
            peak = torch.cuda.max_memory_allocated(runtime.device)
            success = peak <= budget
            error = None if success else f'peak={peak / 2**30:.2f}GiB exceeds 80%-free-memory budget'
        except torch.cuda.OutOfMemoryError as exc:
            success, error = False, str(exc).splitlines()[0]
        except Exception as exc:
            success, error, fatal = False, repr(exc), True
        finally:
            raw.zero_grad(set_to_none=True)
            del optimizer, x, y, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
        accepted = runtime.all(success)
        errors = runtime.gather(error)
        if not runtime.all(not fatal):
            raise RuntimeError(f'Microbatch probe numerical/kernel failure: {errors}')
        runtime.log(f'Microbatch probe B={batch}: {"accepted" if accepted else "stop: " + str(errors)}')
        if not accepted:
            break
        chosen = batch
    restore_rng(saved_rng, runtime.device)
    if not chosen:
        raise RuntimeError('Even microbatch=1 failed the memory gate; free GPU memory before launch.')
    return chosen


def check_gradients(raw):
    norms = []
    for name, p in raw.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise AssertionError(f'No gradient: {name}')
            norms.append(torch.linalg.vector_norm(p.grad.detach().float()))
    norm = torch.linalg.vector_norm(torch.stack(norms))
    assert torch.isfinite(norm) and norm > 0, f'Invalid gradient norm: {norm}'
    return norm


def check_ddp_gradients(raw, runtime):
    if not runtime.distributed:
        return
    # Test the entire gradient vector in bounded chunks, not just its norm.
    for parameter in raw.parameters():
        flat = parameter.grad.detach().flatten()
        for start in range(0, flat.numel(), 1_048_576):
            actual = flat[start:start + 1_048_576]
            reference = actual.clone()
            dist.broadcast(reference, src=0)
            if not runtime.all(torch.allclose(actual, reference, atol=1e-7, rtol=1e-6)):
                raise AssertionError('DDP gradients disagree across ranks')


def forward_backward_gate(raw, model, optimizer, runtime, dtype, batch, sample=None, initial=True, scale=65536.0, attempt=0):
    """Actual-shape two-microstep test, no optimizer update or data consumption."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16), init_scale=scale)
    losses = []
    for micro in range(2):
        if sample is None:
            tokens = torch.randint(50257, (batch, 1025), device=runtime.device)
            x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        else:
            x, y = sample
        assert x.shape == y.shape == (batch, 1024)
        assert torch.equal(x[:, 1:], y[:, :-1]), 'Next-token shift failed'
        assert x.min() >= 0 and y.min() >= 0 and x.max() < 50257 and y.max() < 50257
        sync = model.no_sync() if isinstance(model, DDP) and micro == 0 else nullcontext()
        with sync:
            with autocast_context(runtime, dtype):
                logits, loss = model(x, y)
            assert logits.shape == (batch, 1024, 50257)
            finite = bool(torch.isfinite(logits).all() & torch.isfinite(loss))
            if not runtime.all(finite):
                raise FloatingPointError('Nonfinite logits/loss in preflight')
            losses.append(loss.detach())
            scaler.scale(loss / 2).backward()
        del logits, loss, x, y
    scaler.unscale_(optimizer)
    finite = all(p.grad is not None for p in raw.parameters())
    if finite:
        norms = torch.stack([p.grad.detach().float().norm() for p in raw.parameters()])
        finite = bool(torch.isfinite(norms).all())
    if not runtime.all(finite):
        optimizer.zero_grad(set_to_none=True)
        if dtype == torch.float16 and attempt < 10:
            runtime.log(f'FP16 preflight overflow: retrying with common loss scale {scale / 2}')
            return forward_backward_gate(raw, model, optimizer, runtime, dtype, batch, sample,
                                         initial, scale / 2, attempt + 1)
        raise FloatingPointError('Nonfinite/missing gradients in full-model preflight')
    norm = check_gradients(raw)
    check_ddp_gradients(raw, runtime)
    before = torch.nn.utils.clip_grad_norm_(raw.parameters(), CFG.grad_clip, error_if_nonfinite=True)
    after = check_gradients(raw)
    assert after <= 1.00001 and before > 0
    measured = runtime.mean(torch.stack(losses).mean()).item()
    baseline = math.log(50257)
    runtime.log(f'Full GPT-2 preflight: loss={measured:.6f}; uniform baseline={baseline:.6f}; '
                f'unclipped_norm={norm.item():.6f}; clipped_norm={after.item():.6f}')
    if initial and abs(measured - baseline) > 2.0:
        raise AssertionError(f'Initial loss {measured} unexpectedly far from log(vocab)={baseline}; investigate before training')
    optimizer.zero_grad(set_to_none=True)
    return measured



def prime_optimizer_memory(raw, optimizer):
    """Allocate real Adam states before compiled memory checks, without changing weights."""
    learning_rates = [group['lr'] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group['lr'] = 0.0
        for p in group['params']:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for group, lr in zip(optimizer.param_groups, learning_rates):
        group['lr'] = lr


def compile_and_wrap(raw, runtime, dtype, batch, enabled):
    model = raw
    error = None
    if enabled:
        runtime.log('Compiling the full GPT-2 model and checking its actual training shape...')
        try:
            torch._dynamo.config.suppress_errors = False
            # The schedule has a small, known set of batch shapes. Specialize
            # each one to avoid PyTorch2.10 DDP partitioning of symbolic outputs.
            model = torch.compile(raw, dynamic=False)
            for shape in microbatch_shapes(batch, runtime.world_size):
                x = torch.randint(50257, (shape, 1024), device=runtime.device)
                with autocast_context(runtime, dtype):
                    logits, loss = model(x, x)
                loss.backward()
                del logits, loss, x
                raw.zero_grad(set_to_none=True)
                runtime.log(f'torch.compile training shape passed: B={shape}, T=1024')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            raw.zero_grad(set_to_none=True)
        errors = runtime.gather(error)
        if not runtime.all(error is None):
            raise RuntimeError(f'torch.compile actual forward/backward failed: {errors}. '
                               'For an environmental compiler issue, relaunch with --no-compile. '
                               'For OOM, lower --micro-batch-size. The experiment settings stay fixed.')
        runtime.log('torch.compile: all full/tail shapes passed (compile before DDP, as nanoGPT).')
    else:
        runtime.log('torch.compile disabled explicitly by --no-compile.')
    if runtime.distributed:
        model = DDP(model, device_ids=[runtime.local_rank] if runtime.device.type == 'cuda' else None,
                    broadcast_buffers=False)
    return model


def local_training_state(runtime, train_loader, val_loader):
    return dict(rng=capture_rng(runtime.device), train_generator=train_loader.generator.get_state(),
                val_generator=val_loader.generator.get_state())


def restore_training_state(state, runtime, train_loader, val_loader):
    restore_rng(state['rng'], runtime.device)
    train_loader.generator.set_state(state['train_generator'].cpu())
    val_loader.generator.set_state(state['val_generator'].cpu())


def save_checkpoint(path, raw, optimizer, scaler, step, best, runtime, loaders, metadata):
    rank_states = runtime.gather(local_training_state(runtime, *loaders))
    error = None
    if runtime.rank == 0:
        try:
            atomic_save(dict(format_version=1, model=raw.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), next_step=step, best_val_loss=best,
                             model_config=asdict(raw.config), train_config=asdict(CFG),
                             rank_states=rank_states, metadata=metadata), path)
        except Exception as exc:
            error = repr(exc)
    errors = runtime.gather(error)
    if any(errors):
        raise RuntimeError(f'Checkpoint write failed; atomic write did not complete: {errors}')
    runtime.log(f'Checkpoint saved: {path} (next update={step})')


def atomic_checkpoint_alias(source, destination, runtime):
    """Point latest.pt at a completed milestone without duplicating its disk blocks."""
    error = None
    if runtime.rank == 0:
        temporary = None
        try:
            fd, name = tempfile.mkstemp(prefix='latest-link.', suffix='.tmp', dir=Path(destination).parent)
            os.close(fd)
            temporary = Path(name)
            temporary.unlink()
            os.link(source, temporary)
            os.replace(temporary, destination)
            directory_fd = os.open(Path(destination).parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as exc:
            error = repr(exc)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    errors = runtime.gather(error)
    if any(errors):
        raise RuntimeError(f'Could not update recovery checkpoint alias: {errors}')


class GracefulStop:
    """Signal handlers only set a flag; DDP saves at a common completed update."""
    def __init__(self, enabled=True):
        self.enabled, self.requested, self.previous = enabled, False, {}

    def __enter__(self):
        if self.enabled:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                self.previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._request)
        return self

    def _request(self, signum, frame):
        self.requested = True

    def __exit__(self, *unused):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def recovery_decision(runtime, out_dir, interval, last_saved, stop_requested):
    """All ranks agree whether to save/stop, even if only one receives a signal."""
    if interval <= 0:
        return False, False
    due = False
    stopping = bool(stop_requested)
    if runtime.rank == 0:
        due = (time.monotonic() - last_saved >= interval
               or (Path(out_dir) / 'SAVE_NOW').exists())
        stopping |= (Path(out_dir) / 'STOP_AFTER_CHECKPOINT').exists()
    flags = torch.tensor([int(due or stopping), int(stopping)], device=runtime.device)
    if runtime.distributed:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    return bool(flags[0].item()), bool(flags[1].item())


def training_stop(step, args):
    absolute = getattr(args, 'stop_at', None)
    if absolute is not None:
        return min(CFG.max_iters, absolute)
    return min(CFG.max_iters, step + args.stop_after) if args.stop_after else CFG.max_iters


def checkpoint_paths(out_dir, step, stop, token_milestones):
    """Save only when a successful update crosses a token target, plus final.

    Use absolute completed-update counts, including updates restored on resume.
    Every DDP rank computes the same paths and participates in state gathering.
    Checkpoints are atomic full-update snapshots, so a milestone can exceed its
    target by fewer than one global batch (524288 tokens).
    """
    if step == stop:
        # A final snapshot also covers any token milestone crossed at this update;
        # avoid writing the identical model and Adam state twice.
        return [Path(out_dir) / 'final.pt']
    previous_tokens = (step - 1) * CFG.total_batch_size
    current_tokens = step * CFG.total_batch_size
    paths = [Path(out_dir) / f'tokens_{target}.pt' for target in token_milestones
             if previous_tokens < target <= current_tokens]
    return paths


@torch.no_grad()
def evaluate(raw, runtime, dtype, loader, batch, iters):
    # Raw eager model avoids compilation of a separate evaluation graph and
    # DDP forward collectives. Identical weights; each rank evaluates disjoint starts.
    was_training = raw.training
    raw.eval()
    total = torch.zeros((), device=runtime.device)
    try:
        for _ in range(iters):
            x, y = loader.batch(batch, runtime.device)
            with autocast_context(runtime, dtype):
                logits, loss = raw(x, y)
            del logits
            total += loss.detach()
        value = runtime.mean(total / iters).item()
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite validation loss')
        return value
    finally:
        raw.train(was_training)


def emit_metrics(runtime, path, record):
    if runtime.rank == 0:
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with Path(path).open('a') as handle:
            handle.write(line + '\n')


def train_loop(raw, model, optimizer, scaler, runtime, dtype, batch, loaders, args,
               step, best, metadata):
    train_loader, val_loader = loaders
    accum = accumulation_steps(batch, runtime.world_size)
    optimizer.zero_grad(set_to_none=True)
    # --stop-after only limits this invocation; it never rescales the LR schedule.
    stop = training_stop(step, args)
    skipped_tokens = 0
    overflow_streak = 0
    model.train()
    recovery_seconds = getattr(args, 'recovery_seconds', 0)
    last_saved = time.monotonic()
    with GracefulStop(enabled=recovery_seconds > 0) as termination:
        while step < stop:
            torch.cuda.synchronize(runtime.device)
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(runtime.device)
            loss_sum = torch.zeros((), device=runtime.device)
            plan = accumulation_plan(batch, runtime.world_size, runtime.rank, step)
            for micro, sequences in enumerate(plan):
                if sequences:
                    x, y = train_loader.batch(sequences, runtime.device)
                else:
                    # Keep the final DDP forward/backward collective on every rank.
                    # This placeholder does not advance the data RNG or add tokens.
                    x = y = torch.zeros((1, GPTConfig().block_size), device=runtime.device, dtype=torch.long)
                sync = model.no_sync() if isinstance(model, DDP) and micro < accum - 1 else nullcontext()
                with sync:
                    with autocast_context(runtime, dtype):
                        logits, loss = model(x, y)
                    del logits
                    weight = sequences * GPTConfig().block_size * runtime.world_size / CFG.total_batch_size
                    loss_sum += loss.detach() * weight
                    # DDP averages ranks. Token weighting also handles unequal tails;
                    # when all microbatches are equal this is exactly 1/accum.
                    scaler.scale(loss * weight).backward()
            mean_loss = runtime.mean(loss_sum)
            if not runtime.all(torch.isfinite(mean_loss)):
                raise FloatingPointError(f'Nonfinite loss at step {step}; no optimizer update performed')
            scaler.unscale_(optimizer)  # no-op for BF16; always before clipping.
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), CFG.grad_clip, error_if_nonfinite=False)
            finite = runtime.all(torch.isfinite(norm))
            if not finite:
                if not scaler.is_enabled():
                    raise FloatingPointError(f'Nonfinite gradient at step {step}; optimizer not advanced')
                # All DDP ranks have the same overflow after gradient averaging.
                # Explicit common backoff prevents rank-local scaler drift.
                scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
                optimizer.zero_grad(set_to_none=True)
                overflow_streak += 1
                skipped_tokens += CFG.total_batch_size
                runtime.log(f'FP16 overflow: skipped update {step}; new scale={scaler.get_scale()}; retrying same LR index')
                if overflow_streak >= 10:
                    raise FloatingPointError('10 consecutive FP16 overflows; inspect numerical stability')
                continue
            overflow_streak = 0
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < old_scale:
                raise RuntimeError('Unexpected scaler skip despite finite unscaled gradients')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(runtime.device)
            elapsed = torch.tensor(time.perf_counter() - start, device=runtime.device, dtype=torch.float64)
            if runtime.distributed:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            step += 1
            if step % args.log_interval == 0 or step == 1 or step == stop:
                memory = runtime.gather(dict(rank=runtime.rank,
                                             allocated_GiB=torch.cuda.memory_allocated() / 2**30,
                                             peak_GiB=torch.cuda.max_memory_allocated() / 2**30))
                emit_metrics(runtime, args.out_dir / 'metrics.jsonl', dict(
                    event='train', step=step, lr_step=step - 1, train_loss=mean_loss.item(),
                    lr=lr, grad_norm=runtime.mean(norm).item(), tokens_processed=step * CFG.total_batch_size,
                    overflow_tokens_this_invocation=skipped_tokens,
                    tokens_per_second=CFG.total_batch_size / elapsed.item(),
                    iteration_seconds=elapsed.item(), gpu_memory=memory))
            validation = step % args.eval_interval == 0 or step == stop
            if validation:
                value = evaluate(raw, runtime, dtype, val_loader, batch, args.eval_iters)
                emit_metrics(runtime, args.out_dir / 'metrics.jsonl', dict(event='validation', step=step, val_loss=value))
                best = min(best, value)
            milestones = checkpoint_paths(args.out_dir, step, stop, args.checkpoint_tokens)
            for checkpoint_path in milestones:
                save_checkpoint(checkpoint_path, raw, optimizer, scaler, step, best,
                                runtime, loaders, metadata)
            save_recovery, stop_requested = recovery_decision(
                runtime, args.out_dir, recovery_seconds, last_saved, termination.requested)
            if recovery_seconds > 0 and (save_recovery or milestones):
                recovery_path = args.out_dir / 'latest.pt'
                if milestones:
                    atomic_checkpoint_alias(milestones[-1], recovery_path, runtime)
                else:
                    save_checkpoint(recovery_path, raw, optimizer, scaler, step, best,
                                    runtime, loaders, metadata)
                last_saved = time.monotonic()
                emit_metrics(runtime, args.out_dir / 'metrics.jsonl', dict(
                    event='recovery_checkpoint', step=step, tokens_processed=step * CFG.total_batch_size,
                    path=str(recovery_path), graceful_stop=stop_requested))
                if runtime.rank == 0:
                    (args.out_dir / 'SAVE_NOW').unlink(missing_ok=True)
            if stop_requested:
                runtime.log(f'Graceful interruption: saved recovery state at update {step}')
                break
    return step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--micro-batch-size', default='auto', help='auto or positive per-GPU sequence capacity; final microbatches may be smaller')
    parser.add_argument('--no-compile', action='store_true', help='Explicit, logged eager fallback')
    parser.add_argument('--allow-sdpa-fallback', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'openwebtext')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--eval-iters', type=int, default=200)
    parser.add_argument('--checkpoint-tokens', type=int, nargs='+',
                        default=[2_000_000_000, 4_000_000_000, 6_000_000_000],
                        help='Absolute trained-token milestones; save after crossing each, plus final')
    parser.add_argument('--log-interval', type=int, default=10)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--stop-after', type=int, help='Stop/save after N additional updates; schedule unchanged')
    limits.add_argument('--stop-at', type=int, help='Absolute completed-update limit, preserved across restarts')
    parser.add_argument('--recovery-seconds', type=float, default=300,
                        help='Atomically replace latest.pt at completed updates every N seconds (default 300)')
    parser.add_argument('--supervise-once', type=Path, help='Check/recover the run described by this supervisor JSON')
    args = parser.parse_args()
    for key in ('eval_interval', 'eval_iters', 'log_interval'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if any(value < 1 for value in args.checkpoint_tokens):
        parser.error('--checkpoint-tokens values must be positive')
    if args.checkpoint_tokens != sorted(set(args.checkpoint_tokens)):
        parser.error('--checkpoint-tokens must be unique and increasing')
    if not math.isfinite(args.recovery_seconds) or args.recovery_seconds <= 0:
        parser.error('--recovery-seconds must be finite and positive')
    if args.stop_at is not None and not 1 <= args.stop_at <= CFG.max_iters:
        parser.error('--stop-at must be between 1 and 600000')
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    if args.micro_batch_size != 'auto':
        try:
            assert int(args.micro_batch_size) > 0
        except (ValueError, AssertionError):
            parser.error('--micro-batch-size must be auto or a positive integer')
    if args.smoke_test and args.resume:
        parser.error('--resume is for training; smoke tests create and round-trip their own temporary checkpoint')
    return args


def main():
    args = parse_args()
    if args.supervise_once:
        return supervise_once(args.supervise_once)
    runtime = setup_runtime(args)
    try:
        accumulation_steps(1, runtime.world_size)  # validate rank count before allocation.
        # All ranks run CPU mathematical tests before choosing fused CUDA kernels.
        run_cpu_smoke()
        run_ddp_smoke(runtime)
        if runtime.device.type == 'cpu':
            runtime.log('CPU smoke checks passed. Full-size CUDA and NCCL checks not run.')
            return
        dtype, fused, flash = check_hardware(runtime, args)
        if not args.smoke_test:
            manifest = read_manifest(args.data_dir)
            dataset_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            loaders = (TokenLoader(args.data_dir / 'train.bin', 1024, runtime.rank, runtime.world_size, CFG.seed + runtime.rank),
                       TokenLoader(args.data_dir / 'val.bin', 1024, runtime.rank, runtime.world_size, CFG.seed + 10000 + runtime.rank))
            manifests = runtime.gather(dataset_id)
            if len(set(manifests)) != 1:
                raise RuntimeError('All nodes must use the same prepared dataset manifest')
            args.out_dir.mkdir(parents=True, exist_ok=True)
            if args.resume is None and any(args.out_dir.glob('*.pt')):
                raise FileExistsError('Existing checkpoint: use --resume or a fresh --out-dir; refusing to overwrite a run.')
        else:
            dataset_id, loaders = None, None
        seed_all(CFG.seed)  # identical initialization before DDP broadcasts.
        raw = GPT()
        assert sum(p.numel() for p in raw.parameters()) == 124_439_808
        verify_initialization(raw, verbose=runtime.rank == 0)
        runtime.log('Parameters: 124,439,808 (tied weights counted once, including position embeddings)')
        raw.to(runtime.device)
        checkpoint = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if checkpoint['format_version'] != 1 or checkpoint['model_config'] != asdict(GPTConfig()) or checkpoint['train_config'] != asdict(CFG):
                raise ValueError('Checkpoint configuration differs from this controlled experiment')
            raw.load_state_dict(checkpoint['model'], strict=True)
            if args.micro_batch_size == 'auto':
                args.micro_batch_size = str(checkpoint['metadata']['micro_batch_size'])
        batch = probe_microbatch(raw, runtime, dtype, fused, args.micro_batch_size)
        accum = accumulation_steps(batch, runtime.world_size)
        seed_all(CFG.seed + runtime.rank)
        before_probes = capture_rng(runtime.device)
        optimizer = configure_optimizer(raw, runtime.device, fused=fused)
        prime_optimizer_memory(raw, optimizer)
        model = compile_and_wrap(raw, runtime, dtype, batch, enabled=not args.no_compile)
        scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
        metadata = dict(world_size=runtime.world_size, micro_batch_size=batch, accumulation_steps=accum,
                        batch_policy='rotating_sequence_quotas_v1', global_sequences=512,
                        dtype=str(dtype), flash=flash, fused_adamw=fused, compiled=not args.no_compile,
                        compile_dynamic=False,
                        dataset_id=dataset_id, torch=str(torch.__version__), cuda=torch.version.cuda)
        runtime.log('Training configuration: ' + json.dumps(dict(model=asdict(GPTConfig()), train=asdict(CFG), runtime=metadata)))
        quotas = [sum(accumulation_plan(batch, runtime.world_size, rank)) for rank in range(runtime.world_size)]
        assert sum(quotas) * 1024 == CFG.total_batch_size
        runtime.log(f'Effective batch: per-rank sequences {quotas} * 1024 = 524288 tokens/update; '
                    f'{accum} common accumulation rounds, capacity={batch}; extra sequences rotate each update')
        for shape in microbatch_shapes(batch, runtime.world_size):
            sample = None
            if loaders:
                loader_state = loaders[0].generator.get_state()
                sample = loaders[0].batch(shape, runtime.device)
                loaders[0].generator.set_state(loader_state)
            runtime.log(f'Full-model/DDP preflight for microbatch shape {shape}')
            forward_backward_gate(raw, model, optimizer, runtime, dtype, shape, sample,
                                  initial=checkpoint is None)
            del sample
        runtime.log('Preflight GPU memory: ' + json.dumps(runtime.gather(dict(
            rank=runtime.rank, allocated_GiB=torch.cuda.memory_allocated() / 2**30,
            peak_GiB=torch.cuda.max_memory_allocated() / 2**30))))
        optimizer.state.clear()  # discard lr=0 probe moments/step; real update zero remains zero.
        restore_rng(before_probes, runtime.device)
        if args.smoke_test:
            runtime.log('CUDA smoke passed: actual GPT-2 forward/backward, clipping, precision, compile and gradients. '
                        + (f'DDP synchronization checked across {runtime.world_size} GPUs.' if runtime.world_size > 1 else
                           'Single-GPU NCCL DDP wrapper verified; multi-GPU synchronization remains untested.'
                           if runtime.distributed else 'Multi-GPU DDP not tested in this invocation.'))
            return
        step, best = 0, math.inf
        if checkpoint:
            if checkpoint['metadata'] != metadata:
                raise ValueError(f'Resume runtime/data mismatch. Saved={checkpoint["metadata"]}, current={metadata}')
            if len(checkpoint['rank_states']) != runtime.world_size:
                raise ValueError('Resume requires unchanged world size')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            step, best = checkpoint['next_step'], checkpoint['best_val_loss']
            assert 0 <= step <= CFG.max_iters and math.isfinite(best)
            restore_training_state(checkpoint['rank_states'][runtime.rank], runtime, *loaders)
            del checkpoint
        invocation_stop = training_stop(step, args)
        runtime.log(f'Ready: next optimizer update={step}; this invocation stops at update={invocation_stop}; '
                    f'invocation token budget={(invocation_stop - step) * CFG.total_batch_size}; '
                    'unchanged LR schedule horizon=600000 updates')
        runtime.log('Checkpoint policy: absolute token milestones='
                    f'{args.checkpoint_tokens}; final update={invocation_stop}; '
                    f'rolling recovery latest.pt every {args.recovery_seconds:g} seconds; '
                    'atomic replacement retains the old file until the new file is complete')
        # Baseline validation on scratch; it uses only the validation generator.
        if step == 0:
            best = evaluate(raw, runtime, dtype, loaders[1], batch, args.eval_iters)
            emit_metrics(runtime, args.out_dir / 'metrics.jsonl', dict(event='validation', step=0, val_loss=best))
            save_checkpoint(args.out_dir / 'latest.pt', raw, optimizer, scaler, step, best,
                            runtime, loaders, metadata)
            emit_metrics(runtime, args.out_dir / 'metrics.jsonl', dict(
                event='recovery_checkpoint', step=step, tokens_processed=0,
                path=str(args.out_dir / 'latest.pt'), graceful_stop=False))
        train_loop(raw, model, optimizer, scaler, runtime, dtype, batch, loaders, args, step, best, metadata)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

# Smoke checks are below so the two-file project carries its own verification.

def run_recovery_smoke():
    """Fault-injection tests run entirely on CPU, without SSH or GPU workers."""
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from unittest.mock import patch
    cpu_runtime = SimpleNamespace(device=torch.device('cpu'), rank=0, world_size=1,
        distributed=False, gather=lambda value: [value], all=lambda value: bool(value),
        mean=lambda value: value, log=lambda value: None)
    with tempfile.TemporaryDirectory(prefix='gpt2-recovery-smoke-') as name:
        folder = Path(name)
        latest = folder/'latest.pt'
        atomic_save({'step': 10}, latest)
        original = latest.read_bytes()
        real_save = torch.save
        def interrupted_save(payload, handle):
            handle.write(b'incomplete snapshot')
            raise OSError('injected interrupted write')
        with patch.object(torch, 'save', side_effect=interrupted_save):
            try:
                atomic_save({'step': 11}, latest)
            except OSError:
                pass
            else:
                raise AssertionError('Injected write failure did not propagate')
        assert latest.read_bytes() == original
        assert not list(folder.glob('*.tmp'))
        atomic_save({'step': 11}, latest)
        assert torch.load(latest, weights_only=True)['step'] == 11
        milestone = folder/'tokens_test.pt'
        atomic_save({'step': 20}, milestone)
        atomic_checkpoint_alias(milestone, latest, cpu_runtime)
        assert os.stat(latest).st_ino == os.stat(milestone).st_ino
        atomic_save({'step': 21}, latest)
        assert torch.load(milestone, weights_only=True)['step'] == 20
        assert torch.load(latest, weights_only=True)['step'] == 21
        with patch.object(os, 'link', side_effect=OSError('injected link failure')):
            try:
                atomic_checkpoint_alias(milestone, latest, cpu_runtime)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Failed alias update was accepted')
        assert torch.load(latest, weights_only=True)['step'] == 21
        assert not list(folder.glob('*.tmp'))
        with patch.object(time, 'monotonic', return_value=1000):
            assert recovery_decision(cpu_runtime, folder, 300, 800, False) == (False, False)
            assert recovery_decision(cpu_runtime, folder, 300, 700, False) == (True, False)
            assert recovery_decision(cpu_runtime, folder, 300, 800, True) == (True, True)
            (folder/'SAVE_NOW').touch()
            assert recovery_decision(cpu_runtime, folder, 300, 800, False) == (True, False)
            (folder/'SAVE_NOW').unlink()
            (folder/'STOP_AFTER_CHECKPOINT').touch()
            assert recovery_decision(cpu_runtime, folder, 300, 800, False) == (True, True)
            (folder/'STOP_AFTER_CHECKPOINT').unlink()
        old_handler = signal.getsignal(signal.SIGTERM)
        with GracefulStop() as graceful:
            graceful._request(signal.SIGTERM, None)
            assert graceful.requested
        assert signal.getsignal(signal.SIGTERM) == old_handler
        assert training_stop(100, SimpleNamespace(stop_at=17234, stop_after=None)) == 17234
        assert training_stop(100, SimpleNamespace(stop_at=None, stop_after=20)) == 120
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
            def forward(self, x, y):
                return torch.empty(0), self.weight.square().sum()
        model = Toy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        loader = SimpleNamespace(batch=lambda size, device: (torch.zeros(1), torch.zeros(1)))
        args = SimpleNamespace(out_dir=folder,stop_after=None,stop_at=30,recovery_seconds=300,
                               log_interval=1,eval_interval=1000,eval_iters=1,checkpoint_tokens=[2000000000])
        saved = []
        def capture(path, raw, optimizer, scaler, step, *unused):
            saved.append((Path(path).name, step))
        with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
             patch.object(torch.cuda, 'memory_allocated', return_value=0), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
             patch.dict(globals(), save_checkpoint=capture, emit_metrics=lambda *a: None), \
             patch(__name__+'.recovery_decision', side_effect=[(True,False),(False,False),(True,True)]):
            step = train_loop(model, model, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                              cpu_runtime, torch.float32, 512, (loader,loader), args,10,1.,{})
        assert step == 13 and saved == [('latest.pt',11),('latest.pt',13)], saved

        # Verify supervision decisions without running SSH or launching processes.
        fake_root = folder/'project'
        fake_root.mkdir()
        (fake_root/'train.py').write_text('fixture')
        source_hash = hashlib.sha256(b'fixture').hexdigest()
        config_dir = fake_root/'recovery'
        config_dir.mkdir()
        output = fake_root/'checkpoints/test'
        output.mkdir(parents=True)
        config = dict(relative_out='checkpoints/test',source_sha256=source_hash,
            target_updates=100,recovery_seconds=300,stall_seconds=900,max_restarts_per_hour=3,
            allow_initial_start=True,identity_file='/unused/test/key',manifest_name='manifest.json',
            manifest_sha256='manifest',dataset_sizes={'train.bin':100,'val.bin':10},
            hosts=[dict(rank=i,host=f'host{i}',root=str(fake_root),python=sys.executable,user='test') for i in range(3)])
        config_path = config_dir/'supervisor.json'
        config_path.write_text(json.dumps(config))
        health = []
        calls = []
        def reset(active=False):
            health[:] = [dict(rank=i,reachable=True,processes=[dict(worker=True,pid=i+100)] if active else [],
                gpu_ok=True,free_MiB=16000,compute_pids=[],source_sha256=source_hash,
                manifest_sha256='manifest',sizes=config['dataset_sizes']) for i in range(3)]
            calls.clear()
            for filename in ('supervisor_state.json','events.jsonl','PAUSE_AUTORESTART'):
                (config_dir/filename).unlink(missing_ok=True)
            for filename in ('metrics.jsonl','latest.pt','final.pt'):
                (output/filename).unlink(missing_ok=True)
        def rpc(c,h,action,**extra):
            calls.append((action,h['rank'],extra))
            if action == 'inspect':
                return dict(health[h['rank']])
            if action == 'stop':
                health[h['rank']]['processes'] = []
                return {'stopped':True}
            if action == 'checkpoint_hash':
                return {'sha256':'digest','bytes':8}
            if action == 'launch':
                return dict(rank=h['rank'],**extra)
            raise AssertionError(action)
        def check():
            with redirect_stdout(io.StringIO()):
                return supervise_once(config_path)
        import subprocess
        with patch.dict(globals(), ROOT=fake_root, recovery_remote=rpc), \
             patch.object(subprocess,'run',return_value=SimpleNamespace(returncode=0,stderr='')), \
             patch(__name__+'.inspect_recovery_checkpoint',return_value={'step':10,'sha256':'digest','bytes':8}):
            reset(active=True)
            (output/'metrics.jsonl').write_text(json.dumps(dict(event='train',step=12))+'\n')
            assert check()['status']=='running'
            assert all(c[0]=='inspect' for c in calls)
            event = check()['event_id']
            assert check()['event_id']==event, 'Unchanged health must not emit duplicate events'
            reset(active=True)
            health[1]={'rank':1,'reachable':False,'error':'injected offline host'}
            assert check()['status']=='waiting_for_hosts'
            assert len([c for c in calls if c[0]=='stop'])==2
            assert not any(c[0]=='launch' for c in calls)
            reset()
            (output/'metrics.jsonl').write_text(json.dumps(dict(event='train',step=12))+'\n')
            assert check()['status']=='blocked'
            assert not any(c[0]=='launch' for c in calls)
            reset()
            health[2]['compute_pids']=[9999]
            assert check()['status']=='waiting_for_gpu'
            assert not any(c[0]=='launch' for c in calls)
            reset()
            health[1]['source_sha256']='different source'
            assert check()['status']=='blocked'
            reset()
            (config_dir/'PAUSE_AUTORESTART').touch()
            assert check()['status']=='paused' and not calls
            reset()
            (output/'latest.pt').write_bytes(b'fixture!')
            (output/'metrics.jsonl').write_text(json.dumps(dict(event='train',step=12))+'\n')
            result=check()
            assert result['status']=='starting' and result['resume_step']==10
            launches=[c for c in calls if c[0]=='launch']
            assert len(launches)==3 and all(c[2]['resume_step']==10 for c in launches)
            reset()
            (output/'final.pt').write_bytes(b'fixture!')
            with patch(__name__+'.inspect_recovery_checkpoint',return_value={'step':100,'sha256':'digest','bytes':8}):
                assert check()['status']=='complete' and not calls


def run_cpu_smoke():
    """Executable mathematical checks; does not change the experiment configuration."""
    import copy
    import dataclasses
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np
    import torch
    import torch.nn.functional as F

    cpu = torch.device("cpu")
    saved_rng = capture_rng(cpu)
    saved_threads = torch.get_num_threads()
    torch.set_num_threads(min(saved_threads, 2))
    checks = []

    def passed(message):
        checks.append(message)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[CPU smoke PASS] {message}", flush=True)

    def close(actual, expected, label, *, atol=1e-10, rtol=1e-8):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)

    try:
        # CPU-only tests must not perturb any already-initialized CUDA RNG streams.
        torch.random.default_generator.manual_seed(8675309)
        with torch.device("meta"):
            full = GPT(GPTConfig())
        count = sum(p.numel() for p in full.parameters())
        assert count == 124_439_808, f"GPT-2 parameter count is {count:,}, expected 124,439,808"
        assert full.transformer.wte.weight is full.lm_head.weight
        assert len(full.transformer.h) == 12
        assert full.transformer.wte.weight.shape == (50257, 768)
        assert full.transformer.wpe.weight.shape == (1024, 768)
        del full
        passed("GPT-2 dimensions, 124,439,808 unique parameters, and embedding tying")

        cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=97)
        model = GPT(cfg)
        verify_initialization(model, verbose=False)
        gelus = [module for module in model.modules() if isinstance(module, torch.nn.GELU)]
        assert len(gelus) == cfg.n_layer and all(module.approximate == "tanh" for module in gelus)
        passed("standard/residual initialization, biases, LayerNorm, and tanh GELU")

        x = torch.randint(cfg.vocab_size, (2, cfg.block_size))
        y = torch.randint(cfg.vocab_size, x.shape)
        logits, loss = model(x, y)
        assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        baseline = math.log(cfg.vocab_size)
        assert abs(loss.item() - baseline) < 1.0, (loss.item(), baseline)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            altered = x.clone()
            altered[:, 7:] = (altered[:, 7:] + 1) % cfg.vocab_size
            changed_logits, _ = model(altered)
        close(logits[:, :7].detach(), changed_logits[:, :7], "future-token leakage", atol=2e-6, rtol=2e-6)
        passed(f"forward/backward shapes, finite gradients, causal logits, initial loss {loss.item():.4f} vs log(vocab) {baseline:.4f}")

        # Compare both outputs and derivatives with explicit masked attention.
        attention = copy.deepcopy(model.transformer.h[0].attn).double()
        reference = copy.deepcopy(attention)
        hidden = torch.randn(2, 9, cfg.n_embd, dtype=torch.float64, requires_grad=True)
        ref_hidden = hidden.detach().clone().requires_grad_(True)
        actual = attention(hidden)
        q, k, v = reference.c_attn(ref_hidden).split(cfg.n_embd, dim=2)
        def heads(value):
            return value.view(2, 9, cfg.n_head, cfg.n_embd // cfg.n_head).transpose(1, 2)
        q, k, v = map(heads, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(cfg.n_embd // cfg.n_head)
        mask = torch.ones(9, 9, dtype=torch.bool).tril()
        weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        expected = reference.c_proj((weights @ v).transpose(1, 2).contiguous().view(2, 9, cfg.n_embd))
        close(actual, expected, "SDPA vs masked reference forward")
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        close(hidden.grad, ref_hidden.grad, "SDPA vs masked reference input derivative")
        for (name, actual_parameter), (_, expected_parameter) in zip(attention.named_parameters(), reference.named_parameters()):
            close(actual_parameter.grad, expected_parameter.grad, f"SDPA derivative: {name}")
        passed("SDPA equals explicit causal attention in forward and backward")

        accumulated = copy.deepcopy(model).double()
        concatenated = copy.deepcopy(accumulated)
        inputs = torch.randint(cfg.vocab_size, (6, cfg.block_size))
        targets = torch.randint(cfg.vocab_size, inputs.shape)
        for micro in range(3):
            _, micro_loss = accumulated(inputs[micro * 2:(micro + 1) * 2], targets[micro * 2:(micro + 1) * 2])
            (micro_loss / 3).backward()
        _, full_loss = concatenated(inputs, targets)
        full_loss.backward()
        for (name, actual_parameter), (_, expected_parameter) in zip(accumulated.named_parameters(), concatenated.named_parameters()):
            close(actual_parameter.grad, expected_parameter.grad, f"mean accumulated gradient: {name}")
        for parameter in accumulated.parameters():
            parameter.grad.mul_(100.0)
        unclipped = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in accumulated.parameters()]))
        assert unclipped.item() > 1.0
        reported = torch.nn.utils.clip_grad_norm_(accumulated.parameters(), max_norm=1.0)
        clipped = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in accumulated.parameters()]))
        close(reported, unclipped, "unclipped norm reported by gradient clipping")
        assert 0.999 < clipped.item() <= 1.0 + 1e-10
        passed("mean gradient accumulation equals a concatenated batch; final clipping works")

        # Three ranks cannot divide 512 sequences evenly. Simulate DDP's
        # final rank mean explicitly, including unequal final microbatches.
        weighted = copy.deepcopy(model).double()
        weighted_reference = copy.deepcopy(weighted)
        weighted_x = torch.randint(cfg.vocab_size, (512, cfg.block_size))
        weighted_y = torch.randint(cfg.vocab_size, weighted_x.shape)
        weighted_loss = torch.zeros((), dtype=torch.float64)
        offset = 0
        for rank in range(3):
            plan = accumulation_plan(8, 3, rank, step=0)
            for count in plan:
                _, micro_loss = weighted(weighted_x[offset:offset + count], weighted_y[offset:offset + count])
                # Production multiplies by world_size, then DDP divides by it.
                (micro_loss * (count * 3 / 512) / 3).backward()
                weighted_loss += micro_loss.detach() * count / 512
                offset += count
        assert offset == 512
        _, weighted_reference_loss = weighted_reference(weighted_x, weighted_y)
        weighted_reference_loss.backward()
        close(weighted_loss, weighted_reference_loss.detach(), "unequal-microbatch global mean loss")
        for (name, parameter), (_, reference_parameter) in zip(weighted.named_parameters(), weighted_reference.named_parameters()):
            close(parameter.grad, reference_parameter.grad, f"unequal-microbatch global mean gradient: {name}")
        del weighted, weighted_reference
        passed("three-rank weighted tail batches equal the mean loss/gradients of exactly 512 sequences")

        optimizer = configure_optimizer(model, cpu, fused=False)
        parameters = list(model.parameters())
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(grouped) == len(parameters)
        assert len({id(p) for p in grouped}) == len(grouped)
        assert {id(p) for p in grouped} == {id(p) for p in parameters}
        for group in optimizer.param_groups:
            assert group["betas"] == (0.9, 0.95)
            assert group["eps"] == 1e-8 and group["lr"] == 6e-4
            assert not group.get("fused", False)
            for parameter in group["params"]:
                assert group["weight_decay"] == (0.1 if parameter.ndim >= 2 else 0.0)
        before = {id(p): p.detach().clone() for p in parameters}
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for parameter in parameters:
            factor = 1.0 - 6e-4 * 0.1 if parameter.ndim >= 2 else 1.0
            close(parameter, before[id(parameter)] * factor, "decoupled weight decay group application", atol=1e-8, rtol=1e-6)
        optimizer.zero_grad(set_to_none=True)
        passed("AdamW hyperparameters, unique grouping, and actual matrix-only weight decay")

        scheduler_values = {step: get_lr(step) for step in (0, 1999, 2000, 301000, 599999, 600000, 600001)}
        assert math.isclose(scheduler_values[0], 6e-4 / 2001, rel_tol=1e-12)
        assert math.isclose(scheduler_values[1999], 6e-4 * 2000 / 2001, rel_tol=1e-12)
        assert scheduler_values[2000] == 6e-4
        assert math.isclose(scheduler_values[301000], (6e-4 + 6e-5) / 2, rel_tol=1e-12)
        assert 6e-5 < scheduler_values[599999] < 6e-5 + 1e-12
        assert scheduler_values[600000] == scheduler_values[600001] == 6e-5
        for world_size in (1, 2, 3, 4, 8):
            for micro_batch in (1, 2, 4, 8):
                actual_steps = accumulation_steps(micro_batch, world_size)
                for step in range(world_size):
                    plans = [accumulation_plan(micro_batch, world_size, rank, step=step)
                             for rank in range(world_size)]
                    assert all(len(plan) == actual_steps for plan in plans)
                    assert all(0 <= count <= micro_batch for plan in plans for count in plan)
                    assert sum(sum(plan) for plan in plans) * 1024 == 524288
                    totals = [sum(plan) for plan in plans]
                    assert max(totals) - min(totals) <= 1
                if 512 % (micro_batch * world_size) == 0:
                    assert micro_batch * 1024 * world_size * actual_steps == 524288
        three_rank_plans = [[accumulation_plan(8, 3, rank, step=step) for rank in range(3)]
                            for step in range(3)]
        assert [sum(plan) for plan in three_rank_plans[0]] == [171, 171, 170]
        assert [plan[-1] for plan in three_rank_plans[0]] == [3, 3, 2]
        assert all(len(plan) == 22 and plan[:-1] == (8,) * 21 for plan in three_rank_plans[0])
        assert all(sum(sum(step_plans[rank]) for step_plans in three_rank_plans) == 512
                   for rank in range(3)), "Unequal rank quotas did not rotate fairly across updates"
        assert three_rank_plans[0] != three_rank_plans[1] != three_rank_plans[2]
        for micro_batch in (1, 2):
            plans = [accumulation_plan(micro_batch, 3, rank) for rank in range(3)]
            assert plans[2][-1] == 0 and all(plan[-1] > 0 for plan in plans[:2])
            assert len({len(plan) for plan in plans}) == 1
        for invalid_micro, invalid_world in ((0, 4), (1, 0), (1, 513), (172, 3)):
            try:
                accumulation_steps(invalid_micro, invalid_world)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid accumulation plan was accepted")
        for invalid_rank, invalid_step in ((-1, 0), (3, 0), (0, -1)):
            try:
                accumulation_plan(8, 3, invalid_rank, step=invalid_step)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid accumulation rank or step was accepted")
        passed("nanoGPT scheduler boundaries and exact 524,288-token batch plans, including three-rank rotation")

        token_milestones = (2_000_000_000, 4_000_000_000, 6_000_000_000)
        epoch_stop = 17_234
        checkpoint_root = Path("checkpoint-policy-smoke")
        expected_steps = (3815, 7630, 11445)
        for target, target_step in zip(token_milestones, expected_steps):
            assert (target_step - 1) * CFG.total_batch_size < target <= target_step * CFG.total_batch_size
            assert checkpoint_paths(checkpoint_root, target_step - 1, epoch_stop, token_milestones) == []
            assert checkpoint_paths(checkpoint_root, target_step, epoch_stop, token_milestones) == [
                checkpoint_root / f"tokens_{target}.pt"]
            assert checkpoint_paths(checkpoint_root, target_step + 1, epoch_stop, token_milestones) == []
            # Resuming from the saved next_step must not save this milestone again.
            assert all(not checkpoint_paths(checkpoint_root, update, epoch_stop, token_milestones)
                       for update in range(target_step + 1, target_step + 4))
            assert checkpoint_paths(checkpoint_root, target_step, target_step, token_milestones) == [
                checkpoint_root / "final.pt"]
        selected = [(update, path.name) for update in range(1, epoch_stop + 1)
                    for path in checkpoint_paths(checkpoint_root, update, epoch_stop, token_milestones)]
        assert selected == [(3815, "tokens_2000000000.pt"), (7630, "tokens_4000000000.pt"),
                            (11445, "tokens_6000000000.pt"), (17234, "final.pt")]
        assert epoch_stop * CFG.total_batch_size == 9_035_579_392
        passed("checkpoint token boundaries, resumed absolute counts, exactly four epoch saves, and final deduplication")

        # Exercise the production loop on CPU, replacing only GPU bookkeeping,
        # validation, metrics I/O, and checkpoint I/O. The real loop still performs
        # loss/backward, clipping, optimizer updates, and checkpoint selection.
        class CheckpointPolicyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, inputs, targets):
                return torch.empty(0), self.weight.square().sum()

        for start_step in (998, 1001):
            loop_model = CheckpointPolicyModel()
            loop_optimizer = torch.optim.SGD(loop_model.parameters(), lr=CFG.learning_rate)
            loop_scaler = torch.amp.GradScaler("cuda", enabled=False)
            saved_checkpoints, loop_metrics, evaluations = [], [], []
            loop_runtime = SimpleNamespace(device=cpu, rank=0, world_size=1, distributed=False,
                                           all=lambda value: bool(value), mean=lambda value: value,
                                           gather=lambda value: [value], log=lambda value: None)
            loop_loader = SimpleNamespace(batch=lambda size, device: (
                torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)))
            loop_args = SimpleNamespace(out_dir=checkpoint_root, stop_after=1004 - start_step,
                                        log_interval=1, eval_interval=1000, eval_iters=1,
                                        checkpoint_tokens=[1001 * CFG.total_batch_size])

            def fake_evaluate(*unused_args):
                evaluations.append(True)
                return 1.0 / len(evaluations)  # Every validation improves the best loss.

            def capture_checkpoint(path, raw, optimizer, scaler, step, best, runtime, loaders, metadata):
                saved_checkpoints.append((step, Path(path).name))

            replacements = dict(evaluate=fake_evaluate, save_checkpoint=capture_checkpoint,
                                emit_metrics=lambda runtime, path, record: loop_metrics.append(record))
            with patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "reset_peak_memory_stats"), \
                    patch.object(torch.cuda, "memory_allocated", return_value=0), \
                    patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                    patch.dict(globals(), replacements):
                final_step = train_loop(loop_model, loop_model, loop_optimizer, loop_scaler,
                                        loop_runtime, torch.float32, 512, (loop_loader, loop_loader),
                                        loop_args, start_step, math.inf, {})
            assert final_step == 1004
            expected_saves = ([(1001, f"tokens_{1001 * CFG.total_batch_size}.pt")]
                              if start_step < 1001 else []) + [(1004, "final.pt")]
            assert saved_checkpoints == expected_saves, saved_checkpoints
            validation_steps = [record["step"] for record in loop_metrics if record["event"] == "validation"]
            assert validation_steps == ([1000, 1004] if start_step < 1000 else [1004])
            assert all(name not in ("best.pt", "latest.pt") for _, name in saved_checkpoints)
        passed("production training loop saves only milestones/final, skips improved-validation saves, and resumes without duplicate saves")

        run_recovery_smoke()
        passed("atomic recovery replacement survives interrupted writes; timed/graceful saves and guarded automatic resume")

        # Tiny overfit test uses the specified maximum LR, without mutating TrainConfig.
        overfit = GPT(cfg)
        overfit_optimizer = configure_optimizer(overfit, cpu, fused=False)
        with torch.no_grad():
            _, starting_loss = overfit(x, y)
        for _ in range(25):
            overfit_optimizer.zero_grad(set_to_none=True)
            _, batch_loss = overfit(x, y)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(overfit.parameters(), 1.0)
            overfit_optimizer.step()
        with torch.no_grad():
            _, final_loss = overfit(x, y)
        assert final_loss.item() < starting_loss.item() - 0.05, (starting_loss.item(), final_loss.item())
        passed(f"fixed-batch loss decreases ({starting_loss.item():.4f} -> {final_loss.item():.4f})")

        with tempfile.TemporaryDirectory(prefix="gpt2-smoke-") as directory:
            root = Path(directory)
            token_path = root / "tokens.bin"
            np.arange(4096, dtype=np.uint16).tofile(token_path)
            loader0 = TokenLoader(token_path, cfg.block_size, rank=0, world_size=2, seed=17)
            loader1 = TokenLoader(token_path, cfg.block_size, rank=1, world_size=2, seed=17)
            x0, y0 = loader0.batch(32, cpu)
            x1, y1 = loader1.batch(32, cpu)
            for batch_x, batch_y in ((x0, y0), (x1, y1)):
                assert batch_x.shape == batch_y.shape == (32, cfg.block_size)
                assert batch_x.dtype == batch_y.dtype == torch.long
                assert torch.equal(batch_x[:, 1:], batch_y[:, :-1])
                assert torch.equal(batch_y, batch_x + 1)
            assert x0[:, 0].max() < x1[:, 0].min(), "DDP rank start-position partitions overlap"
            loader_state = loader0.generator.get_state()
            next_x, next_y = loader0.batch(2, cpu)
            loader0.generator.set_state(loader_state)
            replay_x, replay_y = loader0.batch(2, cpu)
            assert torch.equal(next_x, replay_x) and torch.equal(next_y, replay_y)
            boundary_path = root / "shortest-legal.bin"
            np.arange(cfg.block_size + 1, dtype=np.uint16).tofile(boundary_path)
            boundary_loader = TokenLoader(boundary_path, cfg.block_size, rank=0, world_size=1, seed=17)
            boundary_x, boundary_y = boundary_loader.batch(1, cpu)
            assert torch.equal(boundary_x[0], torch.arange(cfg.block_size))
            assert torch.equal(boundary_y[0], torch.arange(1, cfg.block_size + 1))
            passed("memmap shapes, next-token shift, rank partitioning, and loader replay")

            resume_model = GPT(cfg)
            resume_optimizer = configure_optimizer(resume_model, cpu, fused=False)
            def advance(active_model, active_optimizer, step):
                # Exercise every RNG source saved by capture_rng, as well as Adam moments.
                import random
                batch_x = torch.randint(cfg.vocab_size, (2, cfg.block_size))
                batch_y = torch.randint(cfg.vocab_size, batch_x.shape)
                extra_rng_values = (random.random(), float(np.random.random()))
                active_optimizer.zero_grad(set_to_none=True)
                _, active_loss = active_model(batch_x, batch_y)
                active_loss.backward()
                torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
                for group in active_optimizer.param_groups:
                    group["lr"] = get_lr(step)
                active_optimizer.step()
                return active_loss.detach(), batch_x, batch_y, extra_rng_values

            for step in range(3):
                advance(resume_model, resume_optimizer, step)
            checkpoint_path = root / "resume.pt"
            # Exercise the production checkpoint schema and both loader generators.
            local_runtime = SimpleNamespace(device=cpu, rank=0, world_size=1,
                                            gather=lambda value: [value], log=lambda message: None)
            disabled_scaler = torch.amp.GradScaler("cuda", enabled=False)
            smoke_metadata = {"world_size": 1, "micro_batch_size": 2, "dtype": "torch.float32"}
            save_checkpoint(checkpoint_path, resume_model, resume_optimizer, disabled_scaler,
                            3, 4.0, local_runtime, (loader0, loader1), smoke_metadata)
            assert checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
            expected_result = advance(resume_model, resume_optimizer, 3)
            expected_loader_batches = [loader.batch(2, cpu) for loader in (loader0, loader1)]
            # This is a locally-created, trusted checkpoint; RNG state needs full loading.
            restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            replay_model = GPT(GPTConfig(**restored["model_config"]))
            replay_optimizer = configure_optimizer(replay_model, cpu, fused=False)
            replay_model.load_state_dict(restored["model"])
            replay_optimizer.load_state_dict(restored["optimizer"])
            replay_scaler = torch.amp.GradScaler("cuda", enabled=False)
            replay_scaler.load_state_dict(restored["scaler"])
            assert replay_scaler.state_dict() == disabled_scaler.state_dict() == {}
            assert restored["format_version"] == 1 and restored["metadata"] == smoke_metadata
            assert restored["train_config"] == dataclasses.asdict(TrainConfig())
            assert restored["next_step"] == 3 and restored["best_val_loss"] == 4.0
            assert len(restored["rank_states"]) == 1
            restore_training_state(restored["rank_states"][0], local_runtime, loader0, loader1)
            replay_result = advance(replay_model, replay_optimizer, restored["next_step"])
            replay_loader_batches = [loader.batch(2, cpu) for loader in (loader0, loader1)]
            for expected_tensor, replay_tensor in zip(expected_result[:3], replay_result[:3]):
                assert torch.equal(expected_tensor, replay_tensor), "Resumed next batch/loss differs"
            assert expected_result[3] == replay_result[3], "Python/NumPy RNG was not restored"
            for expected_batch, replay_batch in zip(expected_loader_batches, replay_loader_batches):
                assert all(torch.equal(a, b) for a, b in zip(expected_batch, replay_batch))
            for name, value in resume_model.state_dict().items():
                assert torch.equal(value, replay_model.state_dict()[name]), f"Resumed parameter mismatch: {name}"
            expected_state = resume_optimizer.state_dict()
            replay_state = replay_optimizer.state_dict()
            assert expected_state["param_groups"] == replay_state["param_groups"]
            for parameter_id, state in expected_state["state"].items():
                for key, expected_value in state.items():
                    replay_value = replay_state["state"][parameter_id][key]
                    if isinstance(expected_value, torch.Tensor):
                        assert torch.equal(expected_value, replay_value), f"Resumed optimizer mismatch: {key}"
                    else:
                        assert expected_value == replay_value
            passed("production checkpoint schema, exact resumed update, optimizer/LR/scaler state, and all RNG states")
    finally:
        restore_rng(saved_rng, cpu)
        torch.set_num_threads(saved_threads)
    return checks


def run_ddp_smoke(runtime):
    """Compare accumulated DDP gradients with a true global batch on an existing group."""
    import copy
    import random
    import tempfile
    from contextlib import nullcontext
    from pathlib import Path

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_initialized() or runtime.world_size < 2:
        if runtime.rank == 0:
            print("[DDP smoke SKIP] launch with torchrun and at least two ranks to verify collective gradients", flush=True)
        return
    device = runtime.device
    saved_rng = capture_rng(device)
    try:
        with tempfile.TemporaryDirectory(prefix='gpt2-ddp-recovery-') as control_directory:
            decision = recovery_decision(runtime, Path(control_directory), 300,
                                         time.monotonic(), runtime.rank == 1)
            assert decision == (True, True), 'Rank-local stop request did not reach every rank'
        runtime.log('[DDP smoke PASS] rank-local interruption requests a common recovery save on every rank')
        torch.random.default_generator.manual_seed(24680)
        cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=97)
        model = GPT(cfg).to(device)
        device_ids = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else None
        distributed_model = DDP(model, device_ids=device_ids)
        reference = copy.deepcopy(model)
        generator = torch.Generator(device="cpu").manual_seed(12345)
        all_x = torch.randint(cfg.vocab_size, (512, cfg.block_size), generator=generator).to(device)
        all_y = torch.randint(cfg.vocab_size, all_x.shape, generator=generator, device="cpu").to(device)
        batch = min(8, math.ceil(512 / runtime.world_size))
        test_batches = [batch]
        if 512 % runtime.world_size:
            # B2 on three ranks produces a zero-weight last slot on one rank;
            # that rank must still take part in the final backward collective.
            test_batches.append(2 if runtime.world_size == 3 else 1)
        _, reference_loss = reference(all_x, all_y)
        reference_loss.backward()
        expected = torch.cat([p.grad.flatten() for p in reference.parameters()])
        for test_batch in test_batches:
            model.zero_grad(set_to_none=True)
            plans = [accumulation_plan(test_batch, runtime.world_size, rank) for rank in range(runtime.world_size)]
            plan = plans[runtime.rank]
            offset = sum(sum(rank_plan) for rank_plan in plans[:runtime.rank])
            mean_loss = torch.zeros((), device=device)
            for micro, count in enumerate(plan):
                if count:
                    x, y = all_x[offset:offset + count], all_y[offset:offset + count]
                    offset += count
                else:
                    x = y = torch.zeros((1, cfg.block_size), dtype=torch.long, device=device)
                with distributed_model.no_sync() if micro < len(plan) - 1 else nullcontext():
                    _, loss = distributed_model(x, y)
                    weight = count * runtime.world_size / 512
                    (loss * weight).backward()
                    mean_loss += loss.detach() * weight
            actual = torch.cat([p.grad.flatten() for p in model.parameters()])
            assert torch.isfinite(actual).all(), "DDP gradients are not finite"
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-4,
                                       msg="Weighted DDP accumulation differs from global 512-sequence batch")
            all_gradients = [torch.empty_like(actual) for _ in range(runtime.world_size)]
            dist.all_gather(all_gradients, actual)
            assert all(torch.equal(actual, other) for other in all_gradients), "DDP ranks have different synchronized gradients"
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= runtime.world_size
            torch.testing.assert_close(mean_loss, reference_loss.detach(), atol=2e-6, rtol=1e-6)
            if runtime.rank == 0:
                zero_slots = sum(count == 0 for rank_plan in plans for count in rank_plan)
                print(f"[DDP smoke PASS] {runtime.world_size} ranks, microbatch {test_batch}, {zero_slots} zero-weight slots: "
                      "exact 512-sequence mean gradients/loss and identical synchronized gradients", flush=True)

        # Use a local temporary data directory on each rank, so this test also
        # works when /tmp is not shared between machines. Only rank zero writes
        # the checkpoint; its small trusted payload is broadcast for restoration.
        with tempfile.TemporaryDirectory(prefix="gpt2-ddp-smoke-") as directory:
            token_path = Path(directory) / "tokens.bin"
            (np.arange(8192, dtype=np.uint16) % cfg.vocab_size).tofile(token_path)
            loaders = tuple(TokenLoader(token_path, cfg.block_size, runtime.rank,
                                        runtime.world_size, 811 + runtime.rank + offset)
                            for offset in (0, 10000))
            random.seed(951 + runtime.rank)
            np.random.seed(951 + runtime.rank)
            torch.random.default_generator.manual_seed(951 + runtime.rank)
            optimizer = configure_optimizer(model, device, fused=False)
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            def advance_ddp(step):
                optimizer.zero_grad(set_to_none=True)
                batches = []
                local_loss = torch.zeros((), device=device)
                plan = accumulation_plan(batch, runtime.world_size, runtime.rank, step=step)
                for micro, count in enumerate(plan):
                    if count:
                        x, y = loaders[0].batch(count, device)
                    else:
                        x = y = torch.zeros((1, cfg.block_size), dtype=torch.long, device=device)
                    batches.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
                    with distributed_model.no_sync() if micro < len(plan) - 1 else nullcontext():
                        _, loss = distributed_model(x, y)
                        weight = count * runtime.world_size / 512
                        scaler.scale(loss * weight).backward()
                        local_loss += loss.detach() * weight
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                for group in optimizer.param_groups:
                    group["lr"] = get_lr(step)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                validation_batch = tuple(t.cpu().clone() for t in loaders[1].batch(1, device))
                rng_values = (random.random(), float(np.random.random()),
                              torch.rand((), device=device).item(), torch.rand(()).item())
                return batches, validation_batch, local_loss.detach().cpu(), norm.detach().cpu(), rng_values, plan

            for step in range(2):
                advance_ddp(step)
            path = Path(directory) / "ddp.pt"
            metadata = {"world_size": runtime.world_size, "micro_batch_size": batch,
                        "test": "distributed-checkpoint-rotating-quota"}
            save_checkpoint(path, model, optimizer, scaler, 2, 4.0, runtime, loaders, metadata)
            written = runtime.gather(path.exists())
            assert written == [True] + [False] * (runtime.world_size - 1), "Nonzero rank wrote a checkpoint"
            expected_result = advance_ddp(2)
            expected_weights = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            payload = [torch.load(path, map_location="cpu", weights_only=False) if runtime.rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            checkpoint = payload[0]
            assert checkpoint["format_version"] == 1 and checkpoint["next_step"] == 2
            assert checkpoint["metadata"] == metadata and len(checkpoint["rank_states"]) == runtime.world_size
            assert checkpoint["scaler"] == {} and checkpoint["best_val_loss"] == 4.0
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            restore_training_state(checkpoint["rank_states"][runtime.rank], runtime, *loaders)
            actual_result = advance_ddp(checkpoint["next_step"])
            assert expected_result[5] == actual_result[5], "DDP resumed accumulation plan differs"
            assert len(expected_result[0]) == len(actual_result[0]) == accumulation_steps(batch, runtime.world_size)
            def resumed_close(expected_tensor, actual_tensor, label, *, atol=1e-9):
                if device.type == "cuda" and expected_tensor.is_floating_point():
                    # CUDA attention backward can differ at the last few bits.
                    torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=1e-5, msg=label)
                else:
                    assert torch.equal(expected_tensor, actual_tensor), label
            for expected_batch, actual_batch in zip(expected_result[0], actual_result[0]):
                assert all(torch.equal(a, b) for a, b in zip(expected_batch, actual_batch)), "DDP resumed train samples differ"
            assert all(torch.equal(a, b) for a, b in zip(expected_result[1], actual_result[1])), "DDP resumed validation samples differ"
            resumed_close(expected_result[2], actual_result[2], "DDP resumed loss differs", atol=1e-6)
            resumed_close(expected_result[3], actual_result[3], "DDP resumed gradient norm differs", atol=1e-6)
            assert expected_result[4] == actual_result[4], "DDP per-rank RNG restoration failed"
            for name, value in model.state_dict().items():
                resumed_close(expected_weights[name], value, f"DDP resumed weight differs: {name}")
            actual_optimizer = optimizer.state_dict()
            assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
            for parameter_id, state in expected_optimizer["state"].items():
                for key, value in state.items():
                    actual_value = actual_optimizer["state"][parameter_id][key]
                    if isinstance(value, torch.Tensor):
                        resumed_close(value, actual_value, f"DDP resumed optimizer differs: {key}")
                    else:
                        assert value == actual_value
            synchronized_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
            rank0_weights = synchronized_weights.clone()
            dist.broadcast(rank0_weights, src=0)
            assert torch.equal(synchronized_weights, rank0_weights), "DDP resumed weights differ between ranks"
            agreement = "numerically consistent" if device.type == "cuda" else "exact"
            runtime.log(f"[DDP smoke PASS] production rank-zero checkpoint, rotating rank quotas, gathered per-rank RNG/loaders, "
                        f"{agreement} resumed DDP update/optimizer/scaler")
    finally:
        restore_rng(saved_rng, device)


# One-shot supervisor used by the scheduled recovery check. It never trains in
# this process and never uses CUDA; all GPU work stays in the three workers.
REMOTE_RECOVERY_CONTROL = r'''
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
c=json.loads(sys.argv[1]); action=c['action']; root=Path(c['root']); out=root/c['relative_out']
def identity(pid):
    try:
        s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=s[0],ppid=int(s[1]),start=s[19])
    except (OSError,IndexError,ValueError):return None

def processes():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().decode().split('\0')
            # Exact argv boundaries prevent touching other projects or runs.
            if str(root/'train.py') in a and '--out-dir' in a and a[a.index('--out-dir')+1]==str(out):
                item=identity(int(p.name))
                if item and item['state']!='Z':
                    item['worker']='torch.distributed.run' not in a
                    found.append(item)
        except (OSError,ValueError,IndexError,UnicodeDecodeError):pass
    return found

if action=='inspect':
    gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=name,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    apps=subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    gpu_fields=gpu.stdout.strip().split(',')
    manifest=root/'data/openwebtext/manifest.json'
    # The manifest filename is supplied from the validated local data folder.
    manifest=root/'data/openwebtext'/c['manifest_name']
    sizes={n:(root/'data/openwebtext'/n).stat().st_size for n in ('train.bin','val.bin')}
    records=processes()
    print(json.dumps(dict(reachable=True,rank=c['rank'],processes=records,
        gpu_ok=gpu.returncode==0,gpu=gpu.stdout.strip(),free_MiB=int(gpu_fields[1]) if gpu.returncode==0 else 0,
        compute_pids=[int(v) for v in apps.stdout.splitlines() if v.strip().isdigit()],
        source_sha256=hashlib.sha256((root/'train.py').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),sizes=sizes,
        exit=json.loads((out/'exit.json').read_text()) if (out/'exit.json').exists() else None)))
elif action=='stop':
    owned=processes()
    for item in owned:
        if item['worker']:
            current=identity(item['pid'])
            if current and current['start']==item['start']:os.kill(item['pid'],signal.SIGTERM)
    deadline=time.monotonic()+35
    while time.monotonic()<deadline and processes():time.sleep(.2)
    for item in processes():
        current=identity(item['pid'])
        if current and current['start']==item['start']:os.kill(item['pid'],signal.SIGKILL)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and processes():time.sleep(.2)
    assert not processes(),'Owned training workers did not exit'
    print(json.dumps(dict(rank=c['rank'],stopped=True)))
elif action=='checkpoint_hash':
    p=out/'latest.pt'
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps(dict(rank=c['rank'],sha256=digest,bytes=p.stat().st_size)))
elif action=='launch':
    assert not processes(),'Refusing duplicate workers'
    assert hashlib.sha256((root/'train.py').read_bytes()).hexdigest()==c['source_sha256']
    out.mkdir(parents=True,exist_ok=True)
    attempt=out/'attempts'/f"{c['attempt']:04d}"
    attempt.mkdir(parents=True,exist_ok=False)
    if (out/'exit.json').exists():(out/'exit.json').rename(attempt/'previous_exit.json')
    if (out/'run.json').exists():(out/'run.json').rename(attempt/'previous_run.json')
    # This marker was consumed by the previous completed graceful stop.
    (out/'STOP_AFTER_CHECKPOINT').unlink(missing_ok=True)
    command=[c['python'],'-u','-m','torch.distributed.run','--nnodes=3','--nproc_per_node=1',
        '--node_rank='+str(c['rank']),'--master_addr='+c['master_addr'],'--master_port='+str(c['master_port']),
        str(root/'train.py'),'--out-dir',str(out),'--stop-at',str(c['target_updates']),
        '--micro-batch-size','8','--log-interval','1','--recovery-seconds',str(c['recovery_seconds']),
        '--checkpoint-tokens']+[str(v) for v in c['checkpoint_token_milestones']]
    if c.get('resume_step') is not None:command+=['--resume',str(out/'latest.pt')]
    wrapper="import json,subprocess,sys\nfrom pathlib import Path\nfrom datetime import datetime,timezone\nroot=Path(sys.argv[1]);out=Path(sys.argv[2]);attempt=Path(sys.argv[3]);cmd=json.loads(sys.argv[4])\nrc=subprocess.call(cmd,cwd=root)\nrecord=json.dumps(dict(exit_code=rc,finished_at_utc=datetime.now(timezone.utc).isoformat(),attempt=str(attempt)),indent=2)+'\\n'\n(attempt/'exit.json').write_text(record)\n(out/'exit.json').write_text(record)\nraise SystemExit(rc)\n"
    env=dict(os.environ,OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        NCCL_SOCKET_IFNAME=c['interface'],GLOO_SOCKET_IFNAME=c['interface'],NCCL_IB_DISABLE='1',
        TORCH_NCCL_TRACE_BUFFER_SIZE='2000',TORCH_NCCL_DUMP_ON_TIMEOUT='1')
    (attempt/'train.py.snapshot').write_bytes((root/'train.py').read_bytes())
    with (out/'train.log').open('ab',buffering=0) as log:
        log.write(('\n=== Recovery-managed attempt '+str(c['attempt'])+'; resume step '+str(c.get('resume_step'))+' ===\n').encode())
        p=subprocess.Popen([c['python'],'-u','-c',wrapper,str(root),str(out),str(attempt),json.dumps(command)],
            cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(rank=c['rank'],host=c['host'],supervisor_pid=p.pid,command=command,attempt=c['attempt'],
        source_sha256=c['source_sha256'],resume_step=c.get('resume_step'),target_updates=c['target_updates'],
        started_at_utc=datetime.now(timezone.utc).isoformat(),status='launched')
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    (attempt/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
else:raise ValueError('Unknown recovery control action')
'''


def recovery_json(path, value):
    """Small supervisor records use the same write/fsync/replace discipline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def recovery_remote(config, host, action, **extra):
    import shlex
    import subprocess
    c = dict(config, **host, action=action, **extra)
    # No credentials are placed in arguments, logs, or the JSON configuration.
    c.pop('hosts', None)
    command = [host['python'], '-c', REMOTE_RECOVERY_CONTROL, json.dumps(c)]
    if host['rank'] != 0:
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10',
               '-o', 'ServerAliveCountMax=2', '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
               '-i', config['identity_file']]
        if host.get('socket'):
            ssh += ['-S', host['socket']]
        command = ssh + [host['user'] + '@' + host['host'], shlex.join(command)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{host['host']} {action}: {result.stderr[-1800:]}")
    return json.loads(result.stdout)


def inspect_recovery_checkpoint(path, config):
    """Only locally-created trusted snapshots; validate before any restart."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (checkpoint['format_version'] != 1 or checkpoint['model_config'] != asdict(GPTConfig())
                or checkpoint['train_config'] != asdict(CFG)):
            raise ValueError('Recovery checkpoint configuration mismatch')
        step = checkpoint['next_step']
        if not isinstance(step, int) or not 0 <= step <= config['target_updates']:
            raise ValueError('Invalid recovery step')
        if len(checkpoint['rank_states']) != 3 or checkpoint['metadata']['world_size'] != 3:
            raise ValueError('Recovery checkpoint must contain all three rank states')
        expected_dataset_id = hashlib.sha256(json.dumps(read_manifest(ROOT/'data/openwebtext'), sort_keys=True).encode()).hexdigest()
        if checkpoint['metadata']['dataset_id'] != expected_dataset_id:
            raise ValueError('Checkpoint dataset differs from the prepared dataset')
        if not math.isfinite(checkpoint['best_val_loss']):
            raise ValueError('Nonfinite validation state in checkpoint')
        for tensor in checkpoint['model'].values():
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite model weights in recovery checkpoint')
        for state in checkpoint['optimizer']['state'].values():
            if int(state['step'].item()) != step:
                raise ValueError('Optimizer/model step mismatch')
            if not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all():
                raise ValueError('Nonfinite optimizer state in checkpoint')
        if step and not checkpoint['optimizer']['state']:
            raise ValueError('Missing optimizer state in trained checkpoint')
        with Path(path).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        return dict(step=step, sha256=digest, bytes=Path(path).stat().st_size)
    finally:
        torch.set_num_threads(previous_threads)


def supervise_once(config_path):
    """Idempotent, locked check; retry only this run and never silently reset it."""
    import concurrent.futures
    import fcntl
    import shlex
    import subprocess
    from datetime import datetime, timezone
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    folder = config_path.parent
    state_path = folder/'supervisor_state.json'
    lock = (folder/'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(status='busy',message='Another recovery check owns the lock')))
        return
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(attempt=0, restarts=[], event_id=0)
    now = time.time()
    stamp = datetime.now(timezone.utc).isoformat()
    out = ROOT/config['relative_out']

    def finish(status, message, **details):
        previous = state.get('status')
        state.update(status=status, message=message, checked_at_utc=stamp, **details)
        if previous != status or status == 'restarting':
            state['event_id'] = state.get('event_id', 0) + 1
            event = dict(event_id=state['event_id'], status=status, message=message, at_utc=stamp, **details)
            with (folder/'events.jsonl').open('a') as handle:
                handle.write(json.dumps(event, allow_nan=False)+'\n')
        recovery_json(state_path, state)
        print(json.dumps(state, allow_nan=False))
        return state

    try:
        if (folder/'PAUSE_AUTORESTART').exists():
            return finish('paused', 'Automatic recovery explicitly paused; no workers changed')
        if hashlib.sha256((ROOT/'train.py').read_bytes()).hexdigest() != config['source_sha256']:
            return finish('blocked', 'Local training source changed; refusing an unreviewed automatic restart')
        metrics = []
        metrics_path = out/'metrics.jsonl'
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if row.get('event') == 'train':
                        metrics.append(row)
                except json.JSONDecodeError:
                    pass  # A concurrently appended last line can be incomplete.
        last_step = metrics[-1]['step'] if metrics else 0
        checkpoint_path = out/'latest.pt'
        if (out/'final.pt').exists():
            verified = inspect_recovery_checkpoint(out/'final.pt', config)
            if verified['step'] == config['target_updates']:
                return finish('complete', 'The requested one-epoch run has a verified final checkpoint', checkpoint=verified, last_step=last_step)
        def inspect(host):
            try:
                return recovery_remote(config, host, 'inspect')
            except Exception as exc:
                return dict(rank=host['rank'], reachable=False, error=str(exc))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            health = list(pool.map(inspect, config['hosts']))
        active = [h.get('reachable') and any(p['worker'] for p in h['processes']) for h in health]
        all_reachable = all(h['reachable'] for h in health)
        if all(active):
            activity_time = max(metrics_path.stat().st_mtime if metrics_path.exists() else 0, state.get('launched_at', now))
            if now - activity_time < config['stall_seconds']:
                status = ('running' if metrics and metrics_path.stat().st_mtime >= state.get('launched_at', 0)
                          and last_step > state.get('resume_step', -1) else 'starting')
                return finish(status, 'All three training workers are alive', health=health, last_step=last_step,
                              tokens_processed=last_step*CFG.total_batch_size,
                              recovery_checkpoint_exists=checkpoint_path.exists())
        # A failed member invalidates the DDP group. Stop only this run's remaining
        # workers, never an unrelated GPU process. Existing durable state survives.
        if any(h.get('processes') for h in health):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(recovery_remote, config, host, 'stop')
                           for host, h in zip(config['hosts'], health) if h['reachable'] and h['processes']]
                for future in futures:
                    future.result()
        if not all_reachable:
            return finish('waiting_for_hosts', 'Training stopped; waiting for authenticated SSH access to all three machines', health=health, last_step=last_step)
        # Reinspect after stopping to avoid treating owned GPU memory as other work.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            health = list(pool.map(inspect, config['hosts']))
        for h in health:
            if not h['reachable']:
                return finish('waiting_for_hosts', 'A host became unreachable during recovery', health=health)
            if (h['source_sha256'] != config['source_sha256'] or h['manifest_sha256'] != config['manifest_sha256']
                    or h['sizes'] != config['dataset_sizes']):
                return finish('blocked', 'Code or prepared dataset differs between hosts; no restart performed', health=health)
            if not h['gpu_ok'] or h['free_MiB'] < 10000 or h['compute_pids']:
                return finish('waiting_for_gpu', 'A GPU is unavailable or occupied; unrelated jobs were left untouched', health=health)
        verified = None
        if checkpoint_path.exists():
            verified = inspect_recovery_checkpoint(checkpoint_path, config)
            state['ever_had_checkpoint'] = True
        elif last_step or state.get('ever_had_checkpoint') or not config.get('allow_initial_start'):
            return finish('blocked', 'No recoverable checkpoint; refusing to restart learned progress from scratch', last_step=last_step)
        if verified and verified['step'] >= config['target_updates']:
            return finish('blocked', 'Recovery state reached target but final.pt is absent; review required', checkpoint=verified)
        recent_restarts = [t for t in state.get('restarts', []) if now-t < 3600]
        if len(recent_restarts) >= config['max_restarts_per_hour']:
            return finish('backoff', 'Three launches within an hour; waiting before another automatic attempt', last_step=last_step)
        if verified:
            # Ship the one authoritative checkpoint to both workers. rsync uses a
            # temporary destination; hash-check both copies before launching any rank.
            def transfer(host):
                ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','IdentitiesOnly=yes',
                       '-o','IdentityAgent=none','-i',config['identity_file']]
                if host.get('socket'):
                    ssh += ['-S',host['socket']]
                destination = host['user']+'@'+host['host']+':'+str(Path(host['root'])/config['relative_out']/'latest.pt')
                result = subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(checkpoint_path),destination],capture_output=True,text=True,timeout=300)
                if result.returncode:
                    raise RuntimeError(result.stderr[-1500:])
                check = recovery_remote(config,host,'checkpoint_hash')
                if check['sha256'] != verified['sha256'] or check['bytes'] != verified['bytes']:
                    raise ValueError('Remote checkpoint hash mismatch')
                return check
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(transfer, config['hosts'][1:]))
        state.update(attempt=state.get('attempt',0)+1, restarts=recent_restarts+[now],
                     launched_at=now, resume_step=verified['step'] if verified else 0)
        finish('restarting' if verified else 'starting',
               f"Launching three GPUs from update {state['resume_step']}", checkpoint=verified)
        launched = []
        for host in config['hosts']:
            launched.append(recovery_remote(config,host,'launch',attempt=state['attempt'],
                                            resume_step=verified['step'] if verified else None))
        return finish('starting', 'All three ranks launched; awaiting a completed optimizer update', launches=launched)
    except Exception as exc:
        return finish('error', str(exc))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

if __name__ == '__main__':
    main()
