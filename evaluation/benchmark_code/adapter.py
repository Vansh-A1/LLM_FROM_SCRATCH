"""Inference-only nanoGPT adapter for pinned lm_eval 0.4.13 likelihood tasks.

TemplateLM supplies the original context/continuation token boundary handling.
Truncation/causal shifting match HFLM._loglikelihood_tokens. No train(), backward,
optimizer, model mutation, or mixed precision is used anywhere in this module.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from lm_eval.api.model import TemplateLM
from tqdm import tqdm


class NanoGPTLM(TemplateLM):
    def __init__(self, model, tokenizer, identity, batch_size=8, cache_path=None,
                 max_batch_tokens=4096, deadline=None, progress_path=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.identity = identity
        self._device = next(model.parameters()).device
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.deadline = deadline
        self.progress_path = Path(progress_path) if progress_path else None
        self.backend = "causal"
        self._max_length = model.config.block_size
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Model must already be eval(), requires_grad_(False)")
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Standard scoring requires FP32 parameters")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError("TF32 must be disabled")
        self.cache = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self.cache = sqlite3.connect(cache_path, timeout=60)
            self.cache.execute("PRAGMA journal_mode=WAL")
            self.cache.execute("PRAGMA synchronous=FULL")
            self.cache.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self.cache.execute("CREATE TABLE IF NOT EXISTS likelihood (key TEXT PRIMARY KEY,args TEXT NOT NULL,logprob REAL NOT NULL,greedy INTEGER NOT NULL,targets INTEGER NOT NULL)")
            provenance = json.dumps({"checkpoint":identity["sha256"],"source":identity["source_sha256"],
                "adapter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch":torch.__version__,"dtype":"float32","tf32":False,"attention":"math","max_length":self.max_length}, sort_keys=True)
            row = self.cache.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()
            if row and row[0] != provenance:
                raise ValueError("Cache provenance does not match the model and adapter")
            self.cache.execute("INSERT OR IGNORE INTO metadata VALUES ('provenance',?)",(provenance,))
            self.cache.commit()

    @property
    def eot_token_id(self): return self.tokenizer.eot_token

    @property
    def max_length(self): return self._max_length

    @property
    def max_gen_toks(self): return 128

    @property
    def tokenizer_name(self): return "tiktoken-gpt2"

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.tokenizer.encode(string, allowed_special={"<|endoftext|>"})

    @staticmethod
    def cache_key(context, continuation):
        return hashlib.sha256(json.dumps([context,continuation],separators=(",",":")).encode()).hexdigest()

    def _progress(self, done, total, cached):
        if self.progress_path:
            value={"done_requests":done,"total_requests":total,"cache_hits":cached,"updated_unix":time.time()}
            temp=self.progress_path.with_suffix(".tmp")
            temp.write_text(json.dumps(value,indent=2)+"\n")
            temp.replace(self.progress_path)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        outputs = [None] * len(requests)
        jobs = []
        cached = 0
        for index,(args,context,continuation) in enumerate(requests):
            context = context or [self.eot_token_id]
            if not continuation or len(continuation) > self.max_length:
                raise ValueError("Original harness scoring requires 1..context_length continuation tokens")
            key = self.cache_key(context, continuation)
            if self.cache:
                row = self.cache.execute("SELECT logprob,greedy FROM likelihood WHERE key=?",(key,)).fetchone()
                if row:
                    outputs[index] = (row[0], bool(row[1])); cached += 1; continue
            inputs = (context + continuation)[-(self.max_length + 1):][:-1]
            if len(inputs) < len(continuation):
                raise ValueError("Insufficient causal context")
            jobs.append((index,key,args,inputs,continuation))
        # Longest first detects OOM early, and length sorting avoids padding waste.
        jobs.sort(key=lambda j:(-len(j[3]),j[3]))
        bar=tqdm(total=len(requests),initial=cached,desc="FP32 likelihood requests",disable=disable_tqdm)
        start=0
        while start < len(jobs):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise TimeoutError("Evaluation time budget reached; committed request cache is resumable")
            length=len(jobs[start][3])
            bs=max(1,min(self.batch_size,self.max_batch_tokens//length))
            batch=jobs[start:start+bs]
            x=torch.full((len(batch),length),self.eot_token_id,dtype=torch.long,device=self.device)
            for i,job in enumerate(batch): x[i,:len(job[3])]=torch.tensor(job[3],device=self.device)
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                returned=self.model(x)
                logits=returned[0] if isinstance(returned,tuple) else returned
                if logits.dtype != torch.float32:
                    raise ValueError("Unexpected scoring precision")
                rows=[]
                for i,(index,key,args,inputs,continuation) in enumerate(batch):
                    relevant=logits[i,len(inputs)-len(continuation):len(inputs)]
                    logp=torch.log_softmax(relevant,dim=-1,dtype=torch.float32)
                    target=torch.tensor(continuation,device=self.device)
                    selected=logp.gather(-1,target[:,None]).squeeze(-1)
                    if not torch.isfinite(selected).all(): raise ValueError("Non-finite likelihood")
                    # Match the official HFLM FP32 sum, rather than changing metric numerics.
                    score=float(selected.sum().item())
                    greedy=bool(torch.equal(logp.argmax(-1),target))
                    outputs[index]=(score,greedy)
                    rows.append((key,json.dumps(args,ensure_ascii=False),score,int(greedy),len(continuation)))
                    if args is not None: self.cache_hook.add_partial("loglikelihood",args,(score,greedy))
            if self.cache:
                self.cache.executemany("INSERT OR REPLACE INTO likelihood VALUES (?,?,?,?,?)",rows)
                self.cache.commit()
            start += len(batch); bar.update(len(batch)); self._progress(cached+start,len(requests),cached)
            del returned,logits,x,logp,relevant,selected,target
        bar.close()
        assert all(x is not None for x in outputs)
        return outputs

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("This adapter is deliberately restricted to the four requested likelihood benchmarks")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("Generation has a separate frozen-prompt evaluation protocol")

    def close(self):
        if self.cache: self.cache.close()
