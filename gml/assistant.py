"""Bounded evidence tools, with a small explicit command grammar, not an LLM."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from .analysis import select


class EvidenceAssistant:
    def __init__(self,registry):
        self.registry=registry

    def parse(self,text):
        if not isinstance(text,str) or len(text)>512:raise ValueError('Command must be at most 512 characters')
        if text.strip()=='verify':return {'action':'verify'}
        match=re.fullmatch(r'\s*(compare|propose)\s+([a-z_]+)(?:\s+budget\s+([0-9]+(?:\.[0-9]+)?))?\s*',text)
        if match:
            action,workload,budget=match.groups()
            return {'action':action,'workload':workload,'budget_mib':float(budget) if budget else 100000}
        match=re.fullmatch(r'\s*explain\s+([a-z_]+/[a-z_]+/[a-z_]+)\s*',text)
        if match:return {'action':'explain','profile_id':match.group(1)}
        raise ValueError('Supported: compare WORKLOAD budget MIB; propose WORKLOAD budget MIB; explain PROFILE_ID; verify')

    def execute(self,request):
        if not isinstance(request,dict):raise ValueError('Typed object required')
        allowed={'action','workload','budget_mib','profile_id','expected_graph_hash','precision'}
        if set(request)-allowed:raise ValueError('Unsupported fields; raw data and validation status cannot be edited')
        action=request.get('action')
        if action not in ('verify','compare','explain','propose'):raise ValueError('Unsupported tool')
        if request.get('precision','float32')!='float32':raise ValueError('No measured profile for requested precision')
        if action=='verify':
            return {'scope':'Registry data checks; not signed authenticity',
                    'profiles':len(self.registry['profiles']),
                    'all_numerically_valid':all(p['numerical_valid'] for p in self.registry['profiles']),
                    'manifest_sha256':self.registry['manifest_sha256'],
                    'source_count':sum(len(p['raw_sources']) for p in self.registry['profiles'])}
        if action=='explain':
            matches=[p for p in self.registry['profiles'] if p['id']==request.get('profile_id')]
            if len(matches)!=1:raise ValueError('Unknown measured profile')
            profile=matches[0]
            if request.get('expected_graph_hash',profile['graph_hash'])!=profile['graph_hash']:
                raise ValueError('Stale/mismatched graph evidence')
            return {'measured':True,'profile':profile,'limitations':self.registry['manifest']['limitations']}
        workload=request.get('workload')
        profiles=[p for p in self.registry['profiles'] if p['workload']==workload]
        if not profiles:raise ValueError('Unknown workload; a new graph needs a new experiment')
        if any(request.get('expected_graph_hash',p['graph_hash'])!=p['graph_hash'] for p in profiles):
            raise ValueError('Stale/mismatched graph evidence')
        budget=request.get('budget_mib',100000)
        if isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or not 0<=budget<=100000:
            raise ValueError('Finite budget between 0 and 100000 MiB required')
        chosen=select(profiles,int(budget*2**20))
        if action=='compare':
            return {'workload':workload,'budget_mib':budget,'budget_metric':'Peak tensor-allocated bytes, not total device memory',
                    'chosen':None if chosen is None else chosen['id'],
                    'profiles':[{'id':p['id'],'step_ms':p['step_ms'],
                                 'peak_mib':p['peak_allocated_bytes']/2**20,
                                 'feasible':p['peak_allocated_bytes']<=budget*2**20,
                                 'source_runs':p['raw_run_ids']} for p in profiles],
                    'claim':'Offline selection from measured configurations; not a new benchmark or unmeasured prediction'}
        plan={'workload':workload,'graph_hash':profiles[0]['graph_hash'],'dtype':'float32',
              'configurations':[p['id'] for p in profiles if p['peak_allocated_bytes']<=budget*2**20],
              'warmup_steps':3,'measured_steps':7,'independent_repeats':5,
              'numerical_rtol':1e-4,'numerical_atol':1e-5,'status':'PROPOSED_NOT_EXECUTED',
              'purpose':'Confirm measured selection under a declared allocated-memory budget',
              'budget_mib':budget,'network_access':False,'arbitrary_commands':False}
        plan['hash']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        return {'plan':plan,'note':'Proposal only. This read-only assistant cannot launch a GPU job or fabricate its result.'}


def load_assistant(path='results/registry.json'):
    registry=json.loads(Path(path).read_text(encoding='utf-8'))
    root=Path.cwd().resolve()
    def checked_path(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root/'results'):raise ValueError('Evidence path outside results')
        return p
    manifest=checked_path(registry['manifest_path'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=registry['manifest_sha256']:
        raise ValueError('Manifest changed since registry generation')
    raw={}
    for name,digest in registry['raw_sha256'].items():
        data=checked_path(name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Raw evidence changed')
        raw[name]=[json.loads(line) for line in data.decode().splitlines()]
    for profile in registry['profiles']:
        rows=[]
        for source in profile['raw_sources']:
            filename,line=source.rsplit(':',1)
            rows.append(raw[filename][int(line)-1])
        if any(r['status']!='PASS' or r['graph_hash']!=profile['graph_hash']
               or '/'.join((r['spec']['name'],r['backend'],r['policy']))!=profile['id'] for r in rows):
            raise ValueError('Profile identity/validation does not match raw evidence')
        expected=statistics.median(statistics.median(r['wall_ms']) for r in rows)
        if not math.isclose(profile['step_ms'],expected,rel_tol=1e-12):
            raise ValueError('Profile timing does not match raw evidence')
        if profile['peak_allocated_bytes']!=max(r['peak_allocated_bytes'] for r in rows):
            raise ValueError('Profile memory does not match raw evidence')
        if profile['extra_working_set_bytes']!=max(r['incremental_peak_bytes'] for r in rows):
            raise ValueError('Profile extra working set does not match raw evidence')
        if profile['host_offload_payload_peak_bytes']!=max(s['host_peak_bytes'] for r in rows for s in r['offload']):
            raise ValueError('Profile host payload does not match raw evidence')
    return EvidenceAssistant(registry)
