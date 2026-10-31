"""Confirm an exploratory budget counterexample, then bounded size reduction.

This does not claim global graph minimality, novel hardware, or universal policy.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from .analysis import bootstrap


def worker(spec_path,output,repeat):
    from .benchmark import measure
    from .kernel import extension
    extension()
    spec=json.loads(Path(spec_path).read_text())
    configs=spec['configurations'][:]
    random.Random(800+repeat).shuffle(configs)
    rows=[]
    for config in configs:
        row=measure(spec['workload_spec'],config['backend'],config['policy'],repeat,steps=9,warmup=3)
        row['role']=config['role'];rows.append(row)
    Path(output).write_text(json.dumps(rows,indent=2),encoding='utf-8')


def run(registry_path='results/registry.json',folder='results/counterexample'):
    registry=json.loads(Path(registry_path).read_text())
    root=Path(folder);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():
        raise FileExistsError('Choose a fresh counterexample folder; preserve the existing study')
    candidates=registry['robust_budget_counterexamples']
    if not candidates:
        result={'status':'NO_ROBUST_CANDIDATE','reason':'No exploratory independent/joint budget gap survived the declared CI criterion.'}
        (root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return result
    selected=max(candidates,key=lambda c:c['paired_ratio_ci95'][0])
    profiles={p['id']:p for p in registry['profiles']}
    independent=profiles[selected['independent']];joint=profiles[selected['joint']]
    spec={'workload_spec':independent['spec'],'configurations':[
        {'backend':independent['backend'],'policy':independent['policy'],'role':'independent'},
        {'backend':joint['backend'],'policy':joint['policy'],'role':'joint'}]}
    spec_path=root/'spec.json';spec_path.write_text(json.dumps(spec,indent=2))
    original=[]
    for repeat in range(5):
        path=root/f'confirm_{repeat}.json'
        subprocess.run([sys.executable,'-m','gml.counterexample','worker','--spec',str(spec_path),
                        '--output',str(path),'--repeat',str(repeat)],check=True)
        original+=json.loads(path.read_text())
    def summarize(rows):
        times={role:[] for role in ('independent','joint')}
        for row in sorted(rows,key=lambda r:r['repeat']):
            if row['status']!='PASS':return {'status':'INVALID_OR_RESOURCE_LIMIT'}
            times[row['role']].append(statistics.median(row['wall_ms']))
        ratios=[a/b for a,b in zip(times['independent'],times['joint'])]
        ci=bootstrap(ratios)
        return {'status':'CONFIRMED' if ci[0]>1.05 else 'NOT_CONFIRMED',
                'median_ratio':statistics.median(ratios),'ratio_ci95':ci,
                'run_medians_ms':times,
                'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows)}
    confirmation=summarize(original)
    confirmation['fits_original_budget']=confirmation.get('max_peak_allocated_bytes',float('inf'))<=selected['budget_bytes']
    reductions=[]
    if confirmation['status']=='CONFIRMED':
        # New independently launched worker per repeat/size; preserve every attempt.
        n=spec['workload_spec']['nodes']
        for nodes in sorted({max(128,n//2),max(128,n//4)},reverse=True):
            if nodes>=n:continue
            reduced=json.loads(json.dumps(spec));reduced['workload_spec']['nodes']=nodes
            reduced['workload_spec']['name']=spec['workload_spec']['name']+f'_reduced_{nodes}'
            path_spec=root/f'spec_{nodes}.json';path_spec.write_text(json.dumps(reduced,indent=2))
            rows=[]
            for repeat in range(5):
                path=root/f'reduced_{nodes}_{repeat}.json'
                subprocess.run([sys.executable,'-m','gml.counterexample','worker','--spec',str(path_spec),
                                '--output',str(path),'--repeat',str(repeat)],check=True)
                rows+=json.loads(path.read_text())
            reductions.append({'nodes':nodes,'summary':summarize(rows)})
    result={'status':confirmation['status'],'exploratory_certificate':selected,'confirmation':confirmation,
            'configurations':spec['configurations'],'workload':spec['workload_spec'],
            'size_reduction':reductions,
            'limitations':['Two bounded size reductions, not a globally minimal graph',
                           'Reduced cases compare the selected pair, not all configurations retuned at each size',
                           'Exploratory selection and fresh confirmation recorded separately',
                           'Sequential retuning remains a strong alternative to joint exhaustive search']}
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('worker','run'))
    parser.add_argument('--spec');parser.add_argument('--output');parser.add_argument('--repeat',type=int,default=0)
    args=parser.parse_args()
    worker(args.spec,args.output,args.repeat) if args.command=='worker' else run()
