"""Small local ridge cost model; held-out evaluation, no API or language model."""
import json
from pathlib import Path
import numpy as np

BACKENDS=('index_add','pyg','sparse','cuda_csr')
POLICIES=('resident','checkpoint','offload')


def feature(p):
    s=p['spec'];n=34 if s['family']=='karate' else s['nodes']
    return [1.0]+[np.log1p(x) for x in (n,n*max(s['degree'],1),s['width'],s['depth'])]+[
       float(p['backend']==b and p['policy']==m) for b in BACKENDS for m in POLICIES]


def fit(profiles,alpha):
    x=np.array([feature(p) for p in profiles]);y=np.log([p['step_ms'] for p in profiles])
    center=x.mean(0);scale=x.std(0);scale[scale<1e-10]=1;center[0]=0
    xx=(x-center)/scale
    penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0
    weights=np.linalg.solve(xx.T@xx+penalty,xx.T@y)
    return {'center':center.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),'alpha':alpha}


def predict(model,profile):
    value=(np.array(feature(profile))-model['center'])/model['scale']
    return float(np.exp(np.clip(value@np.array(model['weights']),-20,20)))


def run(registry_path='results/registry.json',output='results/learned_selector.json'):
    registry=json.loads(Path(registry_path).read_text());profiles=registry['profiles']
    training=[p for p in profiles if p['spec']['split']=='train']
    validation=[]
    for alpha in (0.1,1.0,10.0):
        errors=[]
        for held in sorted({p['workload'] for p in training}):
            model=fit([p for p in training if p['workload']!=held],alpha)
            errors += [(np.log(predict(model,p))-np.log(p['step_ms']))**2 for p in training if p['workload']==held]
        validation.append({'alpha':alpha,'group_cv_log_mse':float(np.mean(errors))})
    alpha=min(validation,key=lambda r:r['group_cv_log_mse'])['alpha'];model=fit(training,alpha)
    configs=[(b,m) for b in BACKENDS for m in POLICIES]
    # A simple fixed configuration chosen only from training workloads.
    baseline=min(configs,key=lambda c:np.mean([np.log(p['step_ms']) for p in training if (p['backend'],p['policy'])==c]))
    trials=[]
    for workload in sorted({p['workload'] for p in profiles if p['spec']['split']!='train'}):
        pp=[p for p in profiles if p['workload']==workload]
        chosen=min(pp,key=lambda p:predict(model,p));fixed=next(p for p in pp if (p['backend'],p['policy'])==baseline)
        oracle=min(pp,key=lambda p:p['step_ms'])
        trials.append({'workload':workload,'learned_choice':chosen['id'],'fixed_choice':fixed['id'],
                       'oracle':oracle['id'],'learned_regret_ratio':chosen['step_ms']/oracle['step_ms'],
                       'fixed_regret_ratio':fixed['step_ms']/oracle['step_ms'],
                       'predicted_step_ms':predict(model,chosen),'actual_step_ms':chosen['step_ms']})
    improvement=float(np.mean([t['fixed_regret_ratio']-t['learned_regret_ratio'] for t in trials]))
    result={'model':'Local NumPy ridge regression on log latency','training_profiles':len(training),
            'train_ids':[p['id'] for p in training],'cv':validation,'parameters':model,'held_out_trials':trials,
            'fixed_training_choice':list(baseline),'mean_regret_advantage':improvement,
            'decision':'RETAIN_SIMPLE_RULE' if improvement<=0.02 else 'CANDIDATE_REQUIRES_MORE_DATA',
            'used_for_runtime_control':False,
            'limitations':['Only two training graph families with identical size/width/depth; inadequate evidence for generalization',
                           'Unconstrained latency selection; feasibility and correctness remain deterministic vetoes',
                           'Three held-out graph regimes are descriptive, not a statistical population claim',
                           'No LLM, external model/API, or autonomous change to numerical acceptance']}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'decision':result['decision'],'mean_regret_advantage':improvement,'held_out_trials':trials},indent=2))
    return result


if __name__=='__main__':run()
