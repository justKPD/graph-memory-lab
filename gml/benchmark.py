"""Isolated repeated-worker benchmark. Each worker randomizes configurations."""
import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import uuid
import psutil
import torch
from .graph import generate
from .kernel import extension
from .model import GraphModel, prepare_sparse

BACKENDS=('index_add','pyg','sparse','cuda_csr')
POLICIES=('resident','checkpoint','offload')
WORKLOADS=[
 {'name':'uniform_train','family':'uniform','nodes':4096,'degree':16,'width':64,'depth':4,'split':'train'},
 {'name':'skew_train','family':'skewed','nodes':4096,'degree':16,'width':64,'depth':4,'split':'train'},
 {'name':'ring_holdout','family':'ring','nodes':8192,'degree':12,'width':96,'depth':5,'split':'holdout'},
 {'name':'uniform_large_holdout','family':'uniform','nodes':16384,'degree':24,'width':96,'depth':6,'split':'holdout'},
 {'name':'karate_public','family':'karate','nodes':34,'degree':0,'width':64,'depth':4,'split':'public-reference'},
]


def cleanup():
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()


class RSSPeak:
    def __enter__(self):
        self.process=psutil.Process(); self.start=self.peak=self.process.memory_info().rss
        self.stop=threading.Event()
        def sample():
            while not self.stop.wait(0.005):
                self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.peak=max(self.peak,self.process.memory_info().rss)


def validation(model,x,target,graph,backend,policy):
    reference=copy.deepcopy(model)
    candidate=copy.deepcopy(model)
    a=x.detach().clone().requires_grad_();b=x.detach().clone().requires_grad_()
    yr=reference(a,graph,'index_add','resident');yc=candidate(b,graph,backend,policy)
    lr=(yr-target).square().mean();lc=(yc-target).square().mean()
    lr.backward();lc.backward()
    actual=[yc,lc,b.grad]+[p.grad for p in candidate.parameters()]
    wanted=[yr,lr,a.grad]+[p.grad for p in reference.parameters()]
    errors=[]
    for c,r in zip(actual,wanted):
        if c is None or not torch.isfinite(c).all():raise AssertionError('Missing/nonfinite gradient')
        torch.testing.assert_close(c,r,rtol=1e-4,atol=1e-5)
        errors.append(float((c-r).detach().abs().max()))
    return max(errors)


def measure(spec,backend,policy,repeat,steps=7,warmup=3):
    cleanup()
    # Conservative preflight bound, not an artificial OOM experiment.
    free,total=torch.cuda.mem_get_info()
    estimate=spec['nodes']*max(spec['degree'],1)*spec['width']*4*4
    estimate+=spec['nodes']*spec['width']*4*spec['depth']*12+512*1024**2
    if estimate>free*0.7:
        return {'status':'SKIPPED_RESOURCE_BOUND','spec':spec,'backend':backend,'policy':policy,'repeat':repeat,
                'estimate_bytes':estimate,'free_bytes':free}
    setup_start=time.perf_counter()
    graph=generate(spec['family'],spec['nodes'],spec['degree'],123,prepare_csr=backend in ('cuda_csr','sparse'))
    if backend=='sparse':
        # Transpose storage is not required by the library baseline.
        prepare_sparse(graph,release_structures=True)
    torch.manual_seed(101)
    model=GraphModel(spec['width'],spec['depth']).cuda()
    x=torch.randn(graph.nodes,spec['width'],device='cuda',requires_grad=True)
    target=torch.randn_like(x)
    torch.cuda.synchronize()
    setup_ms=(time.perf_counter()-setup_start)*1000
    error=validation(model,x,target,graph,backend,policy)
    # Validation creates transient allocations, removed before any measured step.
    cleanup()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
    def step():
        optimizer.zero_grad(set_to_none=True);x.grad=None
        out=model(x,graph,backend,policy)
        loss=(out-target).square().mean();loss.backward();optimizer.step()
        return float(loss.detach())
    torch.cuda.synchronize();start=time.perf_counter();first_loss=step();torch.cuda.synchronize()
    cold_step_ms=(time.perf_counter()-start)*1000
    for _ in range(warmup):step()
    optimizer.zero_grad(set_to_none=True);x.grad=None
    gc.collect()
    torch.cuda.synchronize()
    base_allocated=torch.cuda.memory_allocated()
    base_reserved=torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    try:torch.cuda.memory.reset_peak_host_memory_stats()
    except (AttributeError,RuntimeError):pass
    wall=[];forward=[];backward=[];update=[];losses=[];offload=[]
    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for event in events:event.record()
    torch.cuda.synchronize()
    with RSSPeak() as rss:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True);x.grad=None
            torch.cuda.synchronize();start=time.perf_counter()
            events[0].record()
            out=model(x,graph,backend,policy);loss=(out-target).square().mean()
            events[1].record();loss.backward();events[2].record();optimizer.step();events[3].record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter()-start)*1000)
            forward.append(events[0].elapsed_time(events[1]))
            backward.append(events[1].elapsed_time(events[2]))
            update.append(events[2].elapsed_time(events[3]))
            losses.append(float(loss.detach()))
            offload.append(asdict(model.last_offload_stats))
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    import pynvml
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        temp=pynvml.nvmlDeviceGetTemperature(handle,0)
        power=pynvml.nvmlDeviceGetPowerUsage(handle)
        device_used=pynvml.nvmlDeviceGetMemoryInfo(handle).used
    except pynvml.NVMLError:temp=power=device_used=None
    finally:pynvml.nvmlShutdown()
    host_stats=dict(torch.cuda.memory.host_memory_stats())
    row={'status':'PASS','run_id':str(uuid.uuid4()),'spec':spec,'backend':backend,'policy':policy,'repeat':repeat,
         'graph_hash':graph.digest,'edges':graph.edges.shape[1],'nodes':graph.nodes,
         'structural_bytes':graph.structural_bytes,'setup_ms':setup_ms,'conversion_ms':graph.conversion_ms,
         'cold_step_ms':cold_step_ms,'base_allocated_bytes':base_allocated,'base_reserved_bytes':base_reserved,
         'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
         'incremental_peak_bytes':max(0,allocated-base_allocated),'wall_ms':wall,
         'forward_gpu_ms':forward,'backward_gpu_ms':backward,'optimizer_gpu_ms':update,
         'first_loss':first_loss,'losses':losses,'max_validation_abs_error':error,
         'host_rss_start_bytes':rss.start,'host_rss_peak_bytes':rss.peak,
         'offload':offload,'host_allocator_stats':host_stats,
         'temperature_c_after':temp,'power_mw_after':power,'device_used_bytes_after':device_used,
         'host_rss_sampling_interval_ms':5,'numerical_rtol':1e-4,'numerical_atol':1e-5}
    del model,optimizer,x,target,out,loss,graph
    cleanup()
    return row


def worker(args):
    extension()
    # Load library backend before timing. No import cost hidden in a warm-up label.
    from torch_geometric.utils import scatter
    config=list(itertools.product(range(len(WORKLOADS)),BACKENDS,POLICIES))
    if args.pilot:config=[c for c in config if c[0] in (0,1)]
    random.Random(900+args.repeat).shuffle(config)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for index,(wi,backend,policy) in enumerate(config):
            spec=WORKLOADS[wi]
            try:
                row=measure(spec,backend,policy,args.repeat,args.steps,args.warmup)
            except torch.OutOfMemoryError:
                row={'status':'OOM','spec':spec,'backend':backend,'policy':policy,'repeat':args.repeat}
                f.write(json.dumps(row)+'\n');f.flush()
                raise  # Leave worker; never continue from partially updated state.
            except Exception as e:
                row={'status':'INVALID','spec':spec,'backend':backend,'policy':policy,'repeat':args.repeat,
                     'error':type(e).__name__+': '+str(e)}
            f.write(json.dumps(row)+'\n');f.flush()
            print(f'{index+1}/{len(config)} {spec["name"]} {backend}/{policy} {row["status"]}',flush=True)


def suite(args):
    root=Path(args.output)
    if root.exists() and any(root.glob('repeat_*.jsonl')):
        raise FileExistsError('Preserve measured evidence: choose a fresh --output folder')
    root.mkdir(parents=True,exist_ok=True)
    manifest={'created_unix':time.time(),'client_date':'2026-10-24','python':sys.version,'platform':platform.platform(),
              'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'device_properties':str(torch.cuda.get_device_properties(0)),
              'repeats':args.repeats,'warmup_steps':args.warmup,'measured_steps_per_config':args.steps,
              'workloads':WORKLOADS,'protocol':'Independent sequential processes by repeat; random config order within worker; clean model and allocator between configs; synchronized wall and GPU events; no discarded samples.',
              'limitations':['One laptop GPU; generated mechanism workloads plus tiny public graph',
                             'Host RSS sampled; transient peaks may be missed',
                             'Pinned bytes are logical payload; allocator stats separately reported',
                             'Setup timing includes hashing/generation/CSR construction as applicable',
                             'Dense floating activations offloaded; integer/sparse structure retained'],
              'source_hashes':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ('gml','csrc') for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cpp')}}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for repeat in range(args.repeats):
        command=[sys.executable,'-m','gml.benchmark','worker','--repeat',str(repeat),
                 '--output',str(root/f'repeat_{repeat}.jsonl'),'--steps',str(args.steps),'--warmup',str(args.warmup)]
        if args.pilot:command.append('--pilot')
        result=subprocess.run(command,check=False)
        if result.returncode:raise RuntimeError(f'Worker {repeat} exited {result.returncode}; inspect raw data')
    print('SUITE COMPLETE',root)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('worker','suite'))
    parser.add_argument('--output',default='results/benchmark')
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--steps',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.steps<1 or args.warmup<0 or args.repeats<1:parser.error('Invalid run bounds')
    worker(args) if args.command=='worker' else suite(args)
