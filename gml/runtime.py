"""Real GPU training controlled at step boundaries under a declared budget replay."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
import psutil
import torch
from .agents import deliberate
from .evidence import Ledger
from .graph import generate
from .kernel import extension
from .model import GraphModel,prepare_sparse


def run(registry_path='results/registry.json',output='results/runtime.json',workload='uniform_train'):
    from .assistant import load_assistant
    registry=load_assistant(registry_path).registry
    profiles=[p for p in registry['profiles'] if p['workload']==workload]
    if len(profiles)!=12:raise ValueError('Complete 12-configuration workload registry required')
    spec=profiles[0]['spec'];extension()
    graph=generate(spec['family'],spec['nodes'],spec['degree'],123)
    prepare_sparse(graph)
    if graph.digest!=profiles[0]['graph_hash']:raise ValueError('Graph evidence hash mismatch')
    headroom=1024**2
    minimum=min(p['extra_working_set_bytes'] for p in profiles)
    maximum=max(p['extra_working_set_bytes'] for p in profiles)
    middle=int(statistics.median(p['extra_working_set_bytes'] for p in profiles))
    schedule=[maximum+headroom*8]*2+[middle+headroom*2]*2+[minimum+headroom*2]*2+[0]+[maximum+headroom*8]*3
    torch.manual_seed(101)
    base=GraphModel(spec['width'],spec['depth']).cuda()
    x=torch.randn(graph.nodes,spec['width'],device='cuda')
    target=torch.randn_like(x)
    # Warm library/runtime paths once, outside timed controller trials, without updates.
    for p in profiles:
        base.zero_grad(set_to_none=True)
        base(x,graph,p['backend'],p['policy']).square().mean().backward()
    base.zero_grad(set_to_none=True);torch.cuda.synchronize()
    import pynvml
    from telemetry import observe
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    ledger_path=Path(output).with_suffix('.ledger.jsonl')
    if ledger_path.exists():raise FileExistsError('Choose a fresh output path; do not overwrite an evidence chain')
    ledger=Ledger(ledger_path)
    ledger.append('runtime-start',{'registry_manifest':registry['manifest_sha256'],'workload':workload,
                                 'graph_hash':graph.digest,'extra_budget_schedule':schedule,
                                 'scope':'Logical extra-working-set budget replay + bounded own-process GPU reservation'})
    trials=[]
    try:
        for strategy in ('fixed','threshold','greedy_validated','council'):
            model=copy.deepcopy(base);reference=copy.deepcopy(base)
            optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
            ref_optimizer=torch.optim.SGD(reference.parameters(),lr=0.001)
            fixed=min((p for p in profiles if p['policy']=='resident'),key=lambda p:p['step_ms'])
            current=fixed['id'];age=100;records=[];pressure=None
            for index,budget in enumerate(schedule):
                if index==4:
                    free,_=torch.cuda.mem_get_info()
                    pressure_bytes=min(64*1024**2,int(free*0.05))
                    pressure=torch.empty(pressure_bytes//4,device='cuda',dtype=torch.float32).fill_(0)
                if index==7:
                    pressure=None
                    # Deliberately yield inactive cached storage after pressure phase.
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                observation=observe(handle)
                reusable=max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
                physical_available=observation['device_free_bytes']+reusable
                effective_budget=min(budget,max(0,physical_available-128*1024**2))
                host_available=observation['host_available_bytes']
                start=time.perf_counter()
                eligible=[p for p in profiles if p['numerical_valid'] and p['graph_hash']==graph.digest
                          and p['extra_working_set_bytes']+headroom<=effective_budget
                          and p['host_offload_payload_peak_bytes']+headroom<=host_available]
                opinions=[]
                if strategy=='council':
                    review=deliberate(profiles,graph.digest,effective_budget,host_available,current,
                                      age=age,horizon=len(schedule)-index,
                                      telemetry_age=max(0,time.monotonic()-observation['monotonic_seconds']),
                                      headroom=headroom)
                    action=review['decision']['action'];chosen=review['decision']['policy'];opinions=review['opinions']
                elif not eligible:
                    action,chosen='PAUSE',None
                elif strategy=='fixed':
                    chosen=fixed['id'] if fixed in eligible else None
                    action='KEEP' if chosen else 'PAUSE'
                elif strategy=='threshold':
                    resident=[p for p in eligible if p['policy']=='resident']
                    selected=min(resident,key=lambda p:p['step_ms']) if resident else min(eligible,key=lambda p:p['extra_working_set_bytes'])
                    chosen=selected['id'];action='KEEP' if chosen==current else 'PROPOSE_SWITCH'
                else:
                    chosen=min(eligible,key=lambda p:p['step_ms'])['id'];action='KEEP' if chosen==current else 'PROPOSE_SWITCH'
                decision_ms=(time.perf_counter()-start)*1000
                record={'step':index,'strategy':strategy,'logical_extra_budget_bytes':budget,
                        'effective_extra_budget_bytes':effective_budget,'telemetry':observation,
                        'inactive_allocator_bytes':reusable,'action':action,'chosen':chosen,
                        'decision_ms':decision_ms,'opinions':opinions,
                        'controlled_reservation_bytes':0 if pressure is None else pressure.numel()*pressure.element_size()}
                if chosen is None:
                    record.update({'trained':False,'numerical_valid':None})
                else:
                    p=next(p for p in profiles if p['id']==chosen)
                    if chosen!=current:age=0
                    current=chosen
                    value=x.detach().clone().requires_grad_()
                    before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize();start=time.perf_counter()
                    out=model(value,graph,p['backend'],p['policy'])
                    loss=(out-target).square().mean();loss.backward();optimizer.step()
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    actual_extra=max(0,torch.cuda.max_memory_allocated()-before)
                    # Independent shadow step is validation, outside timed policy execution.
                    ref_optimizer.zero_grad(set_to_none=True)
                    ref_value=x.detach().clone().requires_grad_()
                    ref_out=reference(ref_value,graph,'index_add','resident')
                    ref_loss=(ref_out-target).square().mean();ref_loss.backward();ref_optimizer.step()
                    torch.testing.assert_close(out,ref_out,rtol=1e-4,atol=1e-5)
                    torch.testing.assert_close(value.grad,ref_value.grad,rtol=1e-4,atol=1e-5)
                    for actual,wanted in zip(model.parameters(),reference.parameters()):
                        torch.testing.assert_close(actual,wanted,rtol=1e-4,atol=1e-5)
                    record.update({'trained':True,'numerical_valid':True,'training_ms':elapsed,
                                   'loss':float(loss.detach()),'actual_extra_allocated_bytes':actual_extra,
                                   'profile_extra_bytes':p['extra_working_set_bytes'],
                                   'logical_budget_exceeded':actual_extra>budget,
                                   'offload':asdict(model.last_offload_stats)})
                    del value,out,loss,ref_value,ref_out,ref_loss
                age+=1
                records.append(record);ledger.append('step',record)
            pressure=None
            trials.append({'strategy':strategy,'steps':records,
                           'trained_steps':sum(r['trained'] for r in records),
                           'pauses':sum(not r['trained'] for r in records),
                           'switches':sum(r['action']=='PROPOSE_SWITCH' for r in records),
                           'decision_ms_median':statistics.median(r['decision_ms'] for r in records),
                           'training_ms_total':sum(r.get('training_ms',0) for r in records),
                           'budget_violations':sum(r.get('logical_budget_exceeded',False) for r in records)})
            del model,reference,optimizer,ref_optimizer
            torch.cuda.empty_cache()
    finally:pynvml.nvmlShutdown()
    report={'workload':workload,'graph_hash':graph.digest,'trials':trials,
            'scope':'Actual GPU training, logical extra-working-set budget replay, real same-process bounded reservation; not an unrelated application or game.',
            'limitations':['Single replay per strategy; runtime timings descriptive, not a causal performance claim',
                           'All configurations profiled in advance; no claim of zero-cost online tuning',
                           'Extra budget excludes preallocated persistent graph/model/target resources',
                           'Shadow numerical validation runs outside timing and outside the logical budget',
                           'Device free plus inactive allocator is a planning estimate; fragmentation/races remain'],
            'registry_manifest':registry['manifest_sha256'],'ledger_root':ledger.previous}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    Ledger.verify(ledger_path)
    print(json.dumps({'output':output,'trials':[{k:t[k] for k in ('strategy','trained_steps','pauses','switches','budget_violations')} for t in trials]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registry',default='results/registry.json')
    parser.add_argument('--output',default='results/runtime.json');parser.add_argument('--workload',default='uniform_train')
    args=parser.parse_args();run(args.registry,args.output,args.workload)
