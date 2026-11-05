"""Inference-only, resumable FP32 held-out validation with each target scored once.

Portable implementation of the published 1024-context/512-stride window protocol.
Historical results used batches 4/8; this implementation defaults to one window.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from safe_model import configure_paths, load_model


def full_windows(file_tokens):
    windows = []; target = 1
    while target < file_tokens:
        end = min(file_tokens-1, 1024 if target == 1 else target+511)
        begin = max(0, end-1024)
        windows.append((begin, end, target)); target = end+1
    return windows


def save_new(path, value):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with path.open('x') as handle: handle.write(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'gpt2/train.py')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--data',type=Path,required=True,help='Contiguous little-endian uint16 GPT-2 tokens')
    parser.add_argument('--expected-data-sha256',default='009ecdd6da34f1ae915ff1f0f3aeb0416b02f2252356505b32ce44166dd5cfcc')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.data.stat().st_size % 2: raise ValueError('Incomplete uint16 token file')
    with args.data.open('rb') as handle: digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=args.expected_data_sha256: raise ValueError('Validation data SHA256 mismatch')
    data=np.memmap(args.data,dtype='<u2',mode='r')
    if len(data)<2 or int(data.max())>=50257: raise ValueError('Invalid token file')
    configure_paths(args.checkpoint,args.source,args.expected_checkpoint_sha256)
    model, _, identity=load_model(args.device)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    manifest={'checkpoint_sha256':identity['sha256'],'source_sha256':identity['source_sha256'],
              'data_sha256':digest,'file_tokens':len(data),'context':1024,'stride':512,
              'dtype':'float32','attention':'SDPA MATH','tf32':False,'batch_size':1}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest: raise ValueError('Resume provenance mismatch')
    else: save_new(out/'manifest.json',manifest)
    records=[]
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for i,(begin,end,target) in enumerate(full_windows(len(data))):
            path=out/f'window{i:05d}.json'
            if path.exists():
                value=json.loads(path.read_text())
                if [value['begin'],value['end'],value['first_target']]!=[begin,end,target]:
                    raise ValueError('Saved target coverage mismatch')
            else:
                tokens=torch.from_numpy(np.array(data[begin:end+1],dtype=np.int64)).unsqueeze(0).to(args.device)
                logits=model(tokens[:,:-1])[0]; offset=target-begin-1
                nll=F.cross_entropy(logits[0,offset:,:],tokens[0,offset+1:],reduction='sum').item()
                value=dict(begin=begin,end=end,first_target=target,tokens=end-target+1,nll_sum=nll)
                save_new(path,value)
            if value['tokens']!=end-target+1 or not math.isfinite(value['nll_sum']): raise ValueError('Invalid saved score')
            records.append(value)
            if i%100==0: print(f'Completed {i+1} windows',flush=True)
    targets=sum(r['tokens'] for r in records)
    assert targets==len(data)-1
    loss=sum(r['nll_sum'] for r in records)/targets
    result={**manifest,'loss':loss,'perplexity':math.exp(loss),'scored_targets':targets,'windows':len(records),
            'protocol':'Targets 1..N-1 exactly once; no document-boundary resets'}
    if not (out/'summary.json').exists(): save_new(out/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
