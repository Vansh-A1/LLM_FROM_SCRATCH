"""Inference-only access to the immutable final checkpoint; no training calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
CHECKPOINT = None
SOURCE = ROOT / 'gpt2/train.py'
EXPECTED_SHA256 = '6fe9cc53a9b3890d682c54208a54ce696ed8be98fbff414d1fa19b212aa3b1fe'
SOURCE_SHA256 = '9e1416e3a2fe393097b05c51575ae434bfea464bbc2d8c82c9ee909115945a40'
ALLOWED_GLOBALS = {'numpy._core.multiarray._reconstruct', 'numpy.ndarray', 'numpy.dtype'}


def configure_precision():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')

def configure_paths(checkpoint=None, source=None, expected_sha256=None):
    """Set explicit inputs; the historical source hash remains mandatory."""
    global CHECKPOINT, SOURCE, EXPECTED_SHA256
    if checkpoint is not None: CHECKPOINT = Path(checkpoint).resolve()
    if source is not None: SOURCE = Path(source).resolve()
    if expected_sha256 is not None: EXPECTED_SHA256 = expected_sha256

def safe_checkpoint():
    """Restricted unpickler, only known NumPy RNG-array constructors allowlisted."""
    if CHECKPOINT is None:
        raise ValueError("Pass --checkpoint to select the audited local checkpoint")
    allowed = ALLOWED_GLOBALS
    actual = set(torch.serialization.get_unsafe_globals_in_checkpoint(CHECKPOINT))
    if not actual <= allowed:
        raise ValueError(f'Unexpected checkpoint globals: {actual - allowed}')
    with CHECKPOINT.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != EXPECTED_SHA256:
            raise ValueError('Checkpoint SHA256 mismatch')
        handle.seek(0)
        with torch.serialization.safe_globals([
            np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType
        ]):
            checkpoint = torch.load(handle, map_location='cpu', weights_only=True)
    return checkpoint

def architecture():
    source = SOURCE
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Training model-definition source changed')
    # Import only class definitions: __name__ is not __main__, so no entrypoint runs.
    name = '_immutable_gpt2_architecture_for_evaluation'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

def load_model(device='cpu'):
    configure_precision()
    checkpoint = safe_checkpoint()
    module = architecture()
    expected = dict(n_layer=12,n_head=12,n_embd=768,block_size=1024,vocab_size=50257,dropout=0.0,bias=True)
    assert checkpoint['model_config'] == expected
    assert checkpoint['format_version'] == 1 and checkpoint['next_step'] == 17234
    state = checkpoint['model']
    assert all(torch.isfinite(t).all().item() for t in state.values())
    assert torch.equal(state['transformer.wte.weight'], state['lm_head.weight'])
    # Temporary constructor values are completely replaced by strict loading.
    with torch.random.fork_rng(devices=[]):
        model = module.GPT(module.GPTConfig(**expected)).float()
    model.load_state_dict(state, strict=True)
    assert model.lm_head.weight is model.transformer.wte.weight
    assert sum(p.numel() for p in model.parameters()) == 124439808
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    tokenizer = tiktoken.get_encoding('gpt2')
    assert tokenizer.n_vocab == 50257 and tokenizer.eot_token == 50256
    assert tokenizer.decode([220]) == ' '
    identity = dict(path=str(CHECKPOINT),sha256=EXPECTED_SHA256,source_sha256=SOURCE_SHA256,
                    step=checkpoint['next_step'],tokens_processed=9035579392,parameters=124439808,
                    model_config=expected,train_config=checkpoint['train_config'],runtime=checkpoint['metadata'],
                    dtype='float32',tf32=False,weights_only=True,safe_globals=sorted(ALLOWED_GLOBALS)+['numpy.dtypes.UInt32DType'],
                    context_length=1024,eot_token_id=50256,space_token_id=220)
    del state, checkpoint
    gc.collect()
    return model, tokenizer, identity

@torch.inference_mode()
def next_token_logits(model, token_ids):
    """Mathematically same forward; project just the last context position."""
    token_ids = token_ids[:, -model.config.block_size:]
    positions = torch.arange(token_ids.shape[1], device=token_ids.device)
    x = model.transformer.drop(model.transformer.wte(token_ids) + model.transformer.wpe(positions))
    for block in model.transformer.h:
        x = block(x)
    return model.lm_head(model.transformer.ln_f(x[:, -1:, :]))[:, -1, :]
