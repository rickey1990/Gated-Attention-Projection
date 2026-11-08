"""Portable full-model torch.compile CPU microbenchmark.

This reproduces the IDEA of original benchmark_extended.py and lets the
architecture settings vary. Runs here are new measurements, not the originally
reported 19.93% result. Test eager and compiled in SEPARATE processes.
"""
# Required Notice: Copyright (c) 2026 GAP20 original author, rickeyernest1990@gmail.com
import argparse, json, copy, statistics, time, os
from pathlib import Path
import torch
from torch.nn import functional as F
from models import LanguageModel,count,PRESETS

def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.variant=='custom':
        cfg={'gap_width':args.gap_width,'gap_qk_head':args.gap_qk_head}
    else:
        cfg=PRESETS[args.variant].copy()
        if args.gap_width is not None:cfg['gap_width']=args.gap_width
        if args.gap_qk_head is not None:cfg['gap_qk_head']=args.gap_qk_head
    cfg.update(context=args.context,vocab=args.vocab,depth=args.depth,heads=args.heads)
    m=LanguageModel(args.kind,**cfg)
    checkpoint_used=None
    if args.checkpoint:
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        # strict loading detects architecture mismatch. Never quietly compare a different variant.
        m.load_state_dict(ck['model'],strict=True)
        if 'metadata' in ck and ck['metadata'].get('params')!=count(m):
            raise ValueError('checkpoint/model parameter metadata mismatch')
        checkpoint_used=str(args.checkpoint)
    reference=copy.deepcopy(m).eval()
    if args.mode=='compiled':
        m=torch.compile(m,backend='inductor',fullgraph=True)
    x=torch.randint(0,args.vocab,(args.batch,args.context))
    y=torch.randint(0,args.vocab,(args.batch,args.context))
    t0=time.perf_counter()
    with torch.no_grad():
        ref=reference(x)
        got=m.eval()(x)
    start_seconds=time.perf_counter()-t0
    err=float((ref-got).abs().max().item())
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    def step():
        m.train()
        opt.zero_grad(set_to_none=True)
        logits=m(x)
        loss=F.cross_entropy(logits.reshape(-1,args.vocab),y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step()
    t0=time.perf_counter();step();first_step_sec=time.perf_counter()-t0
    for _ in range(args.warmup):step()
    times=[]
    for _ in range(args.groups):
        t0=time.perf_counter()
        for __ in range(args.steps):step()
        times.append((time.perf_counter()-t0)/args.steps)
    sec=statistics.median(times)
    return {'kind':args.kind,'variant':args.variant,'model_config':cfg,
            'checkpoint':checkpoint_used or 'randomly initialized (not a 100M checkpoint)',
            'parameters':count(reference),'mode':args.mode,
            'batch':args.batch,'context':args.context,'threads':args.threads,
            'compile_and_first_forward_seconds':start_seconds,
            'first_training_step_seconds':first_step_sec,
            'max_absolute_forward_difference':err,
            'group_seconds_per_step':times,'median_seconds_per_step':sec,
            'training_tokens_per_second':args.batch*args.context/sec}

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--mode',required=True,choices=['eager','compiled'])
    pa.add_argument('--variant',choices=['paper100m','compact34','custom'],default='paper100m')
    pa.add_argument('--kind',choices=['gap20','standard'],default='gap20')
    pa.add_argument('--gap-width',type=int,default=None)
    pa.add_argument('--gap-qk-head',type=int,default=None)
    pa.add_argument('--context',type=int,default=96)
    pa.add_argument('--vocab',type=int,default=2048)
    pa.add_argument('--depth',type=int,default=4)
    pa.add_argument('--heads',type=int,default=4)
    pa.add_argument('--batch',type=int,default=16)
    pa.add_argument('--threads',type=int,default=4)
    pa.add_argument('--groups',type=int,default=6)
    pa.add_argument('--steps',type=int,default=30)
    pa.add_argument('--warmup',type=int,default=3)
    pa.add_argument('--seed',type=int,default=1234)
    pa.add_argument('--checkpoint',type=Path,default=None)
    pa.add_argument('--out',type=Path,default=None)
    a=pa.parse_args()
    if a.variant=='custom' and (a.gap_width is None or a.gap_qk_head is None):
        pa.error('custom requires --gap-width and --gap-qk-head')
    result=run(a);print(json.dumps(result,indent=2))
    if a.out:a.out.write_text(json.dumps(result,indent=2))