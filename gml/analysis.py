"""Aggregate independent-run units; never treat correlated training steps as runs."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np


def bootstrap(values,seed=17):
    x=np.asarray(values,dtype=float)
    if not len(x):return [None,None]
    rng=np.random.default_rng(seed)
    samples=np.median(rng.choice(x,size=(4000,len(x)),replace=True),axis=1)
    return [float(v) for v in np.quantile(samples,[0.025,0.975])]


def select(profiles,budget):
    feasible=[p for p in profiles if p.get('numerical_valid') and p.get('measured',True)
              and p['peak_allocated_bytes']<=budget]
    if not feasible:return None
    return min(feasible,key=lambda p:(p['step_ms'],p['id']))


def budget_comparisons(profiles):
    result=[]
    sizes=sorted({p['peak_allocated_bytes'] for p in profiles})
    budgets=[int((a+b)/2) for a,b in zip(sizes,sizes[1:])]+([sizes[-1]+1] if sizes else [])
    resident=[p for p in profiles if p['policy']=='resident']
    if not resident:return []
    a=min(resident,key=lambda p:p['step_ms'])
    for budget in budgets:
        b=select([p for p in profiles if p['backend']=='index_add'],budget)
        joint=select(profiles,budget)
        if b is None or joint is None:continue
        composed=next(p for p in profiles if p['backend']==a['backend'] and p['policy']==b['policy'])
        sequential=select([p for p in profiles if p['backend']==a['backend']],budget)
        feasible=composed['peak_allocated_bytes']<=budget
        ratios=[i/j for i,j in zip(composed['run_medians_ms'],joint['run_medians_ms'])]
        result.append({'workload':a['workload'],'budget_bytes':budget,'a':a['id'],'b':b['id'],
                       'independent':composed['id'],'independent_feasible':feasible,
                       'joint':joint['id'],'sequential':None if sequential is None else sequential['id'],
                       'independent_over_joint':composed['step_ms']/joint['step_ms'] if feasible else None,
                       'paired_ratio_ci95':bootstrap(ratios) if feasible else None,
                       'sequential_equals_joint':sequential is not None and sequential['id']==joint['id']})
    return result


def run(folder='results/benchmark',output='results/registry.json'):
    root=Path(folder)
    manifest=json.loads((root/'manifest.json').read_text())
    groups=defaultdict(list); rejected=[]
    for path in sorted(root.glob('repeat_*.jsonl')):
        for lineno,line in enumerate(path.read_text().splitlines(),1):
            row=json.loads(line)
            row['source']=str(path.as_posix())+':'+str(lineno)
            if row['status']!='PASS':rejected.append(row);continue
            groups[(row['spec']['name'],row['backend'],row['policy'])].append(row)
    profiles=[]
    for key,rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['repeat'])
        if len(rows)!=manifest['repeats']:
            raise ValueError(f'Incomplete repeats for {key}')
        if len({r['graph_hash'] for r in rows})!=1:raise ValueError('Graph identity drift')
        medians=[statistics.median(r['wall_ms']) for r in rows]
        p={'id':'/'.join(key),'workload':key[0],'backend':key[1],'policy':key[2],
           'spec':rows[0]['spec'],'graph_hash':rows[0]['graph_hash'],
           'step_ms':statistics.median(medians),'run_medians_ms':medians,'ci95_ms':bootstrap(medians),
           'iqr_ms':[float(x) for x in np.quantile(medians,[0.25,0.75])],
           'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
           'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows),
           'extra_working_set_bytes':max(r['incremental_peak_bytes'] for r in rows),
           'host_offload_payload_peak_bytes':max(s['host_peak_bytes'] for r in rows for s in r['offload']),
           'to_host_bytes_per_step':max(s['to_host_bytes'] for r in rows for s in r['offload']),
           'host_rss_peak_bytes':max(r['host_rss_peak_bytes'] for r in rows),
           'structural_bytes':max(r['structural_bytes'] for r in rows),
           'setup_ms':statistics.median(r['setup_ms'] for r in rows),
           'conversion_ms':statistics.median(r['conversion_ms'] for r in rows),
           'cold_step_ms':statistics.median(r['cold_step_ms'] for r in rows),
           'forward_gpu_ms':statistics.median(statistics.median(r['forward_gpu_ms']) for r in rows),
           'backward_gpu_ms':statistics.median(statistics.median(r['backward_gpu_ms']) for r in rows),
           'optimizer_gpu_ms':statistics.median(statistics.median(r['optimizer_gpu_ms']) for r in rows),
           'max_validation_abs_error':max(r['max_validation_abs_error'] for r in rows),
           'numerical_valid':True,'measured':True,'raw_sources':[r['source'] for r in rows],
           'raw_run_ids':[r['run_id'] for r in rows]}
        profiles.append(p)
    cases=[];comparisons=[];interactions=[]
    for workload in sorted({p['workload'] for p in profiles}):
        pp=[p for p in profiles if p['workload']==workload]
        resident={p['backend']:p for p in pp if p['policy']=='resident'}
        library=min((resident[b] for b in ('index_add','pyg','sparse') if b in resident),key=lambda p:p['step_ms'])
        custom=resident['cuda_csr'];baseline=resident['index_add']
        cases.append({'workload':workload,'custom':custom['id'],'best_library':library['id'],
                      'speedup_vs_best_library':library['step_ms']/custom['step_ms'],
                      'speedup_vs_index_add':baseline['step_ms']/custom['step_ms'],
                      'allocated_reduction_vs_best_library':1-custom['peak_allocated_bytes']/library['peak_allocated_bytes'],
                      'library_conversion_ms':library['conversion_ms'],'custom_conversion_ms':custom['conversion_ms'],
                      'library_step_ms':library['step_ms'],'custom_step_ms':custom['step_ms'],
                      'custom_setup_ms':custom['setup_ms'],'library_setup_ms':library['setup_ms'],
                      'break_even_reuse':max(0,(custom['setup_ms']-library['setup_ms'])/(library['step_ms']-custom['step_ms']))
                       if library['step_ms']>custom['step_ms'] else None})
        comparisons+=budget_comparisons(pp)
        def find(b,p):return next(x for x in pp if x['backend']==b and x['policy']==p)
        for policy in ('checkpoint','offload'):
            base=find('index_add','resident');a=find('cuda_csr','resident')
            b=find('index_add',policy);ab=find('cuda_csr',policy)
            differences=[wab-wa-wb+w0 for wab,wa,wb,w0 in zip(ab['run_medians_ms'],a['run_medians_ms'],b['run_medians_ms'],base['run_medians_ms'])]
            interactions.append({'workload':workload,'policy':policy,'interaction_ms':statistics.median(differences),
                                 'ci95_ms':bootstrap(differences),'definition':'T_AB-T_A-T_B+T_0; same graph and precision'})
    robust=[c for c in comparisons if c['independent_feasible'] and c['paired_ratio_ci95'][0]>1.05]
    registry={'version':1,'manifest':manifest,'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
              'manifest_path':(root/'manifest.json').as_posix(),
              'raw_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('repeat_*.jsonl')},
              'profiles':profiles,'rejected':rejected,'case_comparisons':cases,'budget_comparisons':comparisons,
              'interactions':interactions,'robust_budget_counterexamples':robust,
              'statistical_note':'Percentile bootstrap of independent-run medians, five run units; exploratory selection, not a confirmatory population claim.',
              'selection_note':'Budget comparisons are offline selections from a fully measured registry. They are not zero-cost online tuning or learned generalization.'}
    registry['setup_note']='Amortization estimate uses total recorded graph/data/model setup, including sparse preparation. conversion_ms is only the Graph.create component, not complete sparse preparation. Setup is not a fully cold process/compiler/optimizer benchmark; sparse setup constructs temporary CSR resources that could be avoided.'
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    print(json.dumps({'profiles':len(profiles),'rejected':len(rejected),'robust_budget_counterexamples':len(robust),'output':str(path)}))
    return registry


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/benchmark')
    parser.add_argument('--output',default='results/registry.json');args=parser.parse_args()
    run(args.input,args.output)
