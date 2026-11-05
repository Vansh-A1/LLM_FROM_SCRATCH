"""Offline safe published GPT-2 reference, mapped losslessly into equivalent GPT modules."""
from pathlib import Path
import gc
import hashlib
import json
import torch
import tiktoken
from safetensors.torch import load_file
from safe_model import architecture, configure_precision

SNAPSHOT = None
WEIGHTS = None
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
EXPECTED_REFERENCE_SHA256 = '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'

def configure_paths(snapshot):
    global SNAPSHOT, WEIGHTS
    SNAPSHOT = Path(snapshot).resolve()
    WEIGHTS = SNAPSHOT / 'model.safetensors'

def load_model(device='cpu'):
    configure_precision()
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    assert (config['n_layer'], config['n_head'], config['n_embd'], config['n_positions'], config['vocab_size']) == (12, 12, 768, 1024, 50257)
    assert config['activation_function'] == 'gelu_new' and config['layer_norm_epsilon'] == 1e-5
    assert config['attn_pdrop'] == config['resid_pdrop'] == config['embd_pdrop'] == 0.1
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if digest != EXPECTED_REFERENCE_SHA256:
        raise ValueError('Cached reference safetensors identity changed.')
    source = load_file(str(WEIGHTS), device='cpu')
    mapped = {}
    transposed = []
    ignored = []
    for key, value in source.items():
        if key.endswith('.attn.bias'):
            # The registered causal mask is represented by is_causal=True in SDPA.
            expected = torch.tril(torch.ones_like(value, dtype=torch.bool))
            assert torch.equal(value.bool(), expected)
            ignored.append(key)
            continue
        target_key = 'transformer.' + key
        if key.endswith(('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')):
            value = value.t().contiguous()
            transposed.append(key)
        mapped[target_key] = value
    mapped['lm_head.weight'] = mapped['transformer.wte.weight']
    assert all(torch.isfinite(x).all().item() for x in mapped.values())
    module = architecture()
    cfg = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50257, dropout=0.1, bias=True)
    with torch.random.fork_rng(devices=[]):
        model = module.GPT(module.GPTConfig(**cfg)).float()
    model.load_state_dict(mapped, strict=True)
    assert model.lm_head.weight is model.transformer.wte.weight
    assert sum(p.numel() for p in model.parameters()) == 124439808
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    tokenizer = tiktoken.get_encoding('gpt2')
    identity = {'model_id': 'openai-community/gpt2', 'revision': REVISION, 'path': str(WEIGHTS),
                'sha256': digest, 'parameters': 124439808, 'model_config': cfg,
                'source_config': config, 'dtype': 'float32', 'tf32': False,
                'safe_loading': 'safetensors.torch.load_file, CPU; no pickle deserialization',
                'mapping': 'HF Conv1D weight matrices transposed into equivalent torch Linear matrices; tied output head uses wte; causal mask buffers verified and represented by SDPA is_causal.',
                'transposed_tensor_names': transposed, 'verified_causal_mask_buffers': ignored,
                'comparison_label': 'External pretrained reference, different data and training history; not equal compute.'}
    del source, mapped
    gc.collect()
    return model, tokenizer, identity
