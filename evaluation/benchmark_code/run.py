"""Pinned, resumable EleutherAI benchmark execution, strictly inference only."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parent
OUT=CODE.parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
sys.path.insert(0,str(OUT))
from safe_model import configure_precision, configure_paths, load_model
import torch
from adapter import NanoGPTLM

TASKS={"lambada_openai":5153,"hellaswag":10042,"piqa":1838,"arc_easy":2376}

def sha(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()

def json_default(value):
    if hasattr(value,"item"):return value.item()
    if hasattr(value,"tolist"):return value.tolist()
    if isinstance(value,Path):return str(value)
    if isinstance(value,set):return sorted(value)
    if callable(value):return f"{value.__module__}.{value.__name__}"
    raise TypeError(f"Cannot serialize {type(value)}")

def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,default=json_default,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text()==text:return
        raise FileExistsError(f"Preserving existing artifact: {path}")
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(text);os.replace(tmp,path)

def task_object(name, arrow_root=None):
    """Retain original harness semantics; load pinned data or verified Arrow exports."""
    from datasets import Dataset, DatasetDict, load_dataset
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    pins={v['task']:v for v in json.loads((CODE/'dataset_pins.json').read_text())}
    old=json.loads((CODE/'dataset_provenance.json').read_text())[name]
    manager=TaskManager(); yaml=Path(manager.task_index[name].yaml_path)
    if sha(yaml)!=old['task_yaml_sha256']: raise ValueError(f'Task definition changed: {name}')
    config=load_yaml(yaml,resolve_func=True)
    if config['dataset_path']!=pins[name]['dataset']: raise ValueError('Dataset mismatch')
    provenance={**old,'source_yaml_sha256':sha(yaml),'loading':'pinned Hugging Face revision'}
    if arrow_root is not None:
        split_data={}
        for split, values in old['splits'].items():
            item=values['cache_files'][0]
            path=Path(arrow_root)/name/item['path']
            if sha(path)!=item['sha256']: raise ValueError(f'Arrow hash mismatch: {path}')
            split_data[split]=Dataset.from_file(str(path))
        datasets=DatasetDict(split_data)
        provenance['loading']='hash-verified historical Arrow files'
    else:
        datasets=load_dataset(config['dataset_path'],config.get('dataset_name'),revision=pins[name]['revision'])
    for split, values in old['splits'].items():
        if len(datasets[split])!=values['documents']: raise ValueError('Dataset count mismatch')
    class PinnedTask(ConfigurableTask):
        def download(self,dataset_kwargs=None,**kwargs): self.dataset=datasets
    task=PinnedTask(config=config)
    if len(task.eval_docs)!=TASKS[name]: raise ValueError('Unexpected evaluation split size')
    provenance['eval_split']=task.config.test_split if task.has_test_docs() else task.config.validation_split
    provenance['eval_items']=len(task.eval_docs)
    if name!='lambada_openai':
        counts=Counter(len(task.doc_to_choice(doc)) for doc in task.eval_docs)
        provenance['choice_count_distribution']={str(k):v for k,v in sorted(counts.items())}
        provenance['uniform_random_choice_reference']=sum(v/k for k,v in counts.items())/len(task.eval_docs)
    return task,provenance

def environment():
    import lm_eval
    core=Path(lm_eval.__file__).parent
    files=['evaluator.py','api/model.py','api/task.py','api/metrics.py','models/huggingface.py']
    return {'python':sys.version,'executable':sys.executable,
        'packages':{p:importlib.metadata.version(p) for p in ['torch','lm_eval','datasets','transformers','tiktoken','numpy','huggingface_hub']},
        'harness_root':str(core),'core_source_hashes':{p:sha(core/p) for p in files},
        'adapter_sha256':sha(CODE/'adapter.py'),'runner_sha256':sha(__file__),
        'dtype':'float32','autocast':False,'tf32':False,'deterministic_algorithms':True,'attention_backend':'math for primary; eager for Hugging Face reference',
        'scoring':'Original HFLM-compatible FP32 log_softmax and FP32 continuation sum, same-weight nanoGPT architecture',
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}

def actual_adapter_smoke(adapter):
    from lm_eval.api.instance import Instance
    pairs=[('A dog is',' an animal'),('','The ocean is blue.'),('Hello ','world.'),('The capital of France is',' Paris')]
    requests=[Instance('loglikelihood',{},p,i) for i,p in enumerate(pairs)]
    observed=adapter.loglikelihood(requests,disable_tqdm=True)
    records=[]
    for pair,got in zip(pairs,observed):
        if pair[0]:context,continuation=adapter._encode_pair(*pair)
        else:context,continuation=[adapter.eot_token_id],adapter.tok_encode(pair[1])
        allids=context+continuation
        x=torch.tensor([allids[:-1]],device=adapter.device)
        with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            returned=adapter.model(x);logits=returned[0] if isinstance(returned,tuple) else returned
            relevant=logits[0,len(context)-1:]
            target=torch.tensor(continuation,device=adapter.device)
            expected=float(torch.log_softmax(relevant,dim=-1).gather(-1,target[:,None]).sum())
            expected_greedy=bool(torch.equal(relevant.argmax(-1),target))
        if abs(expected-got[0])>2e-4 or expected_greedy!=got[1]:raise AssertionError('Direct/adapter scoring mismatch')
        records.append({'context':pair[0],'continuation':pair[1],'context_ids':context,'continuation_ids':continuation,
            'adapter_loglikelihood':got[0],'direct_loglikelihood':expected,'absolute_difference':abs(expected-got[0]),'greedy':got[1]})
    return {'status':'passed','type':'actual checkpoint direct-forward checks; not a standardized benchmark score','records':records}

def wilson(correct,n,z=1.959963984540054):
    p=correct/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def summarize(result,name,identity,provenance,limit):
    samples=result['samples'][name];metrics=result['results'][name];n=len(samples)
    if not limit and n!=TASKS[name]:raise AssertionError(f'Incomplete {name}: {n}')
    ids=[s['doc_id'] for s in samples]
    if len(set(ids))!=n:raise AssertionError('Duplicate item IDs')
    counts={}
    for metric in ['acc','acc_norm']:
        if f'{metric},none' in metrics:
            values=[float(s[metric]) for s in samples]
            if any(v not in (0.,1.) for v in values):raise ValueError('Unexpected nonbinary item accuracy')
            correct=sum(map(int,values))
            if abs(correct/n-metrics[f'{metric},none'])>1e-12:raise AssertionError('Aggregate mismatch')
            counts[metric]={'correct':correct,'total':n,'accuracy':correct/n,'wilson_95_ci':wilson(correct,n)}
    return {'status':'smoke_subset' if limit else 'complete','task':name,'model_identity':identity,
        'n':n,'zero_shot':True,'limit':limit,'metrics':metrics,'correct_counts':counts,
        'task_config':result['configs'][name],'task_version':result['versions'][name],
        'dataset':provenance,'item_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'perplexity_definition':'LAMBADA exp(mean negative loglikelihood of entire target last word), not full-text token perplexity' if name=='lambada_openai' else None,
        'length_normalization':'Original harness continuation character length' if name!='lambada_openai' else None}

def load_reference(device, snapshot):
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM
    import tiktoken
    path=Path(snapshot)
    expected='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
    if sha(path/'model.safetensors')!=expected:raise ValueError('Reference weights hash mismatch')
    configure_precision()
    ref=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,use_safetensors=True,dtype=torch.float32,attn_implementation='eager')
    ref.eval().requires_grad_(False).to(device)
    assert ref.get_input_embeddings().weight is ref.get_output_embeddings().weight
    assert sum(p.numel() for p in ref.parameters())==124439808
    class ReferenceWrapper(torch.nn.Module):
        def __init__(self):super().__init__();self.reference=ref;self.config=SimpleNamespace(block_size=1024);self.eval()
        def forward(self,x):return self.reference(input_ids=x,use_cache=False).logits,None
    identity={'model':'openai-community/gpt2','revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e','sha256':expected,'source_sha256':sha(Path(inspect.getsourcefile(ref.__class__))),
        'dtype':'float32','tf32':False,'safe_serialization':'safetensors','comparison_type':'external pretrained reference, not equal compute'}
    return ReferenceWrapper(),tiktoken.get_encoding('gpt2'),identity

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-batch-tokens',type=int,default=2048)
    parser.add_argument('--timeout-seconds',type=int,default=14400)
    parser.add_argument('--model',choices=['final','reference'],default='final')
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path, default=CODE.parents[1]/"gpt2/train.py")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--reference-snapshot", type=Path)
    parser.add_argument("--arrow-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    configure_paths(args.checkpoint,args.source,args.expected_checkpoint_sha256)
    if not args.prepare_only and args.model=="reference" and args.reference_snapshot is None: parser.error("--reference-snapshot is required for the reference")
    if importlib.metadata.version('lm_eval')!='0.4.13':raise ValueError('Wrong harness version')
    run_dir=OUT/'benchmark_outputs'/args.model/('smoke' if args.smoke else 'full')
    run_dir.mkdir(parents=True,exist_ok=True)
    tasks={};provenance={}
    for name in args.tasks:
        tasks[name],provenance[name]=task_object(name,args.arrow_root)
        save_new(OUT/'benchmark_code'/'pinned_tasks'/f'{name}.json',provenance[name])
    env=environment()
    if args.prepare_only:
        save_new(OUT/'benchmark_code'/'environment.json',env)
        print(json.dumps({'prepared':{k:len(v.eval_docs) for k,v in tasks.items()},'environment':env},indent=2));return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    deadline=time.monotonic()+args.timeout_seconds
    model,tokenizer,identity=load_model(args.device) if args.model=='final' else load_reference(args.device,args.reference_snapshot)
    adapter=NanoGPTLM(model,tokenizer,identity,batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,deadline=deadline,
        cache_path=OUT/'benchmark_outputs'/args.model/'request_cache.sqlite',progress_path=run_dir/'progress.json')
    save_new(run_dir/f'run_environment_{stamp}.json',{'environment':env,'identity':identity,'arguments':vars(args),'started_utc':stamp})
    try:
        smoke=actual_adapter_smoke(adapter)
        save_new(run_dir/f'adapter_checks_{stamp}.json',smoke)
        from lm_eval import simple_evaluate
        for name,task in tasks.items():
            result_path=run_dir/f'{name}.json';summary_path=run_dir/f'{name}_summary.json'
            if result_path.exists():
                if not summary_path.exists():
                    preserved=json.loads(result_path.read_text())
                    saved_identity=preserved['evaluation_provenance']['model_identity']
                    if saved_identity['sha256']!=identity['sha256']:raise ValueError('Existing raw result identity mismatch')
                    recovered=summarize(preserved,name,saved_identity,provenance[name],10 if args.smoke else None)
                    recovered['elapsed_seconds']=preserved['evaluation_provenance']['elapsed_seconds']
                    recovered['summary_reconstructed_from_preserved_raw_result']=True
                    save_new(summary_path,recovered)
                old=json.loads(summary_path.read_text())
                if old['model_identity']['sha256']!=identity['sha256']:raise ValueError('Existing result identity mismatch')
                print(f'Already completed: {name}; preserved existing results',flush=True);continue
            started=time.monotonic()
            result=simple_evaluate(model=adapter,tasks=[task],num_fewshot=0,limit=10 if args.smoke else None,
                bootstrap_iters=1000,log_samples=True,random_seed=42,numpy_random_seed=42,
                torch_random_seed=42,fewshot_random_seed=42,apply_chat_template=False,
                metadata={'checkpoint_sha256':identity['sha256'],'precision':'FP32 TF32 disabled'})
            if result is None:raise ValueError('Harness produced no results')
            summary=summarize(result,name,identity,provenance[name],10 if args.smoke else None)
            summary['elapsed_seconds']=time.monotonic()-started
            result['evaluation_provenance']={'model_identity':identity,'elapsed_seconds':summary['elapsed_seconds'],
                'dataset_revision':provenance[name]['revision'],'adapter_sha256':env['adapter_sha256'],'runner_sha256':env['runner_sha256']}
            save_new(result_path,result);save_new(summary_path,summary)
            print(json.dumps({'task':name,'status':summary['status'],'n':summary['n'],'metrics':summary['metrics'],'elapsed_seconds':summary['elapsed_seconds']},default=json_default),flush=True)
    except BaseException as error:
        save_new(run_dir/f'failure_{stamp}.json',{'error':repr(error),'traceback':traceback.format_exc(),'resumable_cache':str(OUT/'benchmark_outputs'/args.model/'request_cache.sqlite')})
        raise
    finally:adapter.close()

if __name__=='__main__':main()
