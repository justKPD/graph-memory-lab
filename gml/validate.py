import copy
import json
from pathlib import Path
import torch
from .graph import Graph, generate
from .kernel import csr_sum, extension
from .model import GraphModel, aggregate, prepare_sparse


def run(output='results/validation.json'):
    extension()
    checks=[]
    def record(name, fn):
        fn()
        checks.append({'name':name,'status':'PASS'})
    torch.manual_seed(27)
    e=torch.tensor([[0,0,1,2,3,4,4],[1,1,2,2,4,0,4]],device='cuda')
    g=Graph.create(6,e)
    prepare_sparse(g,torch.float64)
    x=torch.randn(6,4,device='cuda',dtype=torch.float64,requires_grad=True)
    for backend in ('cuda_csr','pyg','sparse'):
        def check(backend=backend):
            a=x.detach().clone().requires_grad_()
            b=x.detach().clone().requires_grad_()
            y=aggregate(a,g,backend); ref=aggregate(b,g,'index_add')
            torch.testing.assert_close(y,ref,rtol=1e-10,atol=1e-12)
            y.square().sum().backward(); ref.square().sum().backward()
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-12)
        record('duplicate-self-isolated-gradient/'+backend,check)
    record('finite-difference-gradcheck',lambda:torch.autograd.gradcheck(lambda v:csr_sum(v,g),(x,),eps=1e-6,atol=1e-5))
    def empty():
        eg=Graph.create(6,torch.empty((2,0),device='cuda',dtype=torch.int64))
        value=x.detach().clone().requires_grad_(); out=csr_sum(value,eg)
        assert torch.count_nonzero(out)==0
        out.sum().backward(); assert torch.count_nonzero(value.grad)==0
    record('empty-edges',empty)
    def noncontiguous():
        value=torch.randn(4,6,device='cuda',dtype=torch.float64).t().requires_grad_()
        torch.testing.assert_close(csr_sum(value,g),aggregate(value,g,'index_add'))
    record('noncontiguous-input',noncontiguous)
    def permutation():
        p=torch.tensor([3,1,5,0,4,2],device='cuda')
        remap=torch.empty_like(p); remap[p]=torch.arange(6,device='cuda')
        other=Graph.create(6,remap[e])
        torch.testing.assert_close(csr_sum(x[p],other),csr_sum(x,g)[p])
        reordered=Graph.create(6,e[:,torch.randperm(e.shape[1],device='cuda')])
        torch.testing.assert_close(csr_sum(x,reordered),csr_sum(x,g))
    record('node-relabel-and-edge-order',permutation)
    for bad in (torch.tensor([[0],[6]]),torch.tensor([[-1],[0]])):
        def rejected(bad=bad):
            try: Graph.create(6,bad.to('cuda'))
            except ValueError: return
            raise AssertionError('Invalid graph accepted')
        record('index-range-rejection/'+str(bad.tolist()),rejected)
    # CPU double reference and multi-step stochastic training equivalence.
    for dropout in (0.0,0.2):
        base=GraphModel(width=4,depth=3,dropout=dropout).double()
        initial=torch.randn(6,4,dtype=torch.float64)
        target=torch.randn(6,4,dtype=torch.float64)
        cpu_g=Graph.create(6,e.cpu())
        ref_model=copy.deepcopy(base)
        ref_optimizer=torch.optim.SGD(ref_model.parameters(),lr=0.01)
        states=[]
        # CPU/GPU dropout generators differ, so CPU reference is deterministic only.
        if dropout==0:
            for step in range(3):
                ref_optimizer.zero_grad(set_to_none=True)
                v=initial.clone().requires_grad_()
                y=ref_model(v,cpu_g)
                loss=(y-target).square().mean();loss.backward();ref_optimizer.step()
                states.append([y.detach(),v.grad.detach()]+[p.detach().clone() for p in ref_model.parameters()])
        gpu_reference=None
        for backend in ('index_add','pyg','sparse','cuda_csr'):
            for policy in ('resident','checkpoint','offload'):
                def train(backend=backend,policy=policy):
                    nonlocal gpu_reference
                    model=copy.deepcopy(base).cuda()
                    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
                    trajectory=[]
                    torch.manual_seed(99)
                    for step in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        v=initial.cuda().requires_grad_()
                        y=model(v,g,backend,policy)
                        loss=(y-target.cuda()).square().mean();loss.backward()
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                        optimizer.step()
                        trajectory.append([y.detach().cpu(),v.grad.cpu()]+[p.detach().cpu().clone() for p in model.parameters()])
                    expected=states if dropout==0 else gpu_reference
                    if expected is None:
                        gpu_reference=trajectory
                    else:
                        for actual_step,expected_step in zip(trajectory,expected):
                            for actual,wanted in zip(actual_step,expected_step):
                                torch.testing.assert_close(actual,wanted,rtol=1e-8,atol=1e-10)
                record(f'train-3steps/dropout={dropout}/{backend}/{policy}',train)
    def mutant():
        # Dropping duplicate messages is an intentionally invalid 'fast' optimization.
        dedup=Graph.create(6,torch.unique(e,dim=1))
        assert not torch.allclose(csr_sum(torch.ones_like(x),g),csr_sum(torch.ones_like(x),dedup))
    record('detect-dropped-duplicate-mutant',mutant)
    for family in ('uniform','skewed','ring','karate'):
        def family_check(family=family):
            graph=generate(family,128,8,41)
            v=torch.randn(graph.nodes,16,device='cuda',dtype=torch.float64)
            torch.testing.assert_close(csr_sum(v,graph),aggregate(v,graph,'index_add'),rtol=1e-9,atol=1e-10)
        record('graph-family/'+family,family_check)
    report={'status':'PASS','checks':checks,'count':len(checks),'device':torch.cuda.get_device_name(),
            'limitations':'Float32 benchmarks require their own numerical gate; first-order gradients only; no formal proof.'}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':len(checks),'output':str(path)}))


if __name__=='__main__':run()
