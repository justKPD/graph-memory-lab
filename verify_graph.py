"""Graph-training numerical smoke check across resident/checkpoint/CPU offload."""
import copy
import json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint


class GraphStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = torch.nn.Linear(8, 8)
        self.node = torch.nn.Linear(8, 2)

    def forward(self, x, edges):
        src, dst = edges
        messages = self.edge(torch.cat((x[src], x[dst]), dim=-1)).relu()
        aggregate = x.new_zeros((x.shape[0], 8)).index_add(0, dst, messages)
        return self.node(aggregate)


torch.manual_seed(23)
base = GraphStep().double()
x_cpu = torch.randn(6, 4, dtype=torch.float64)
# Directed, duplicate edge, self-loop, and an isolated node.
edges_cpu = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 1, 2, 2, 4, 0]])
target_cpu = torch.randn(6, 2, dtype=torch.float64)
reference = None
results = {}
for mode in ('cpu_reference', 'resident', 'checkpoint', 'offload'):
    device = 'cpu' if mode == 'cpu_reference' else 'cuda'
    model = copy.deepcopy(base).to(device)
    x = x_cpu.to(device).clone().requires_grad_(True)
    edges, target = edges_cpu.to(device), target_cpu.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if mode == 'checkpoint':
        out = checkpoint(model, x, edges, use_reentrant=False)
    elif mode == 'offload':
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            out = model(x, edges)
    else:
        out = model(x, edges)
    loss = (out-target).square().mean()
    loss.backward()
    tensors = [out.detach().cpu(), x.grad.detach().cpu()]
    tensors += [p.grad.detach().cpu().clone() for p in model.parameters()]
    optimizer.step()
    tensors += [p.detach().cpu().clone() for p in model.parameters()]
    if reference is None:
        reference = tensors
    for actual, expected in zip(tensors, reference):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    results[mode] = {'passed':True,'loss':loss.item(),
                     'max_abs_error':max((a-b).abs().max().item()
                                         for a,b in zip(tensors,reference))}
report = {'checks':results,'dtype':'float64','nodes':6,'edges':6,
          'tests':['output','input_gradient','parameter_gradients','SGD_update'],
          'is_benchmark':False,'custom_cuda_aggregation':False,
          'limitations':'Tiny deterministic graph; no stochastic/multistep convergence guarantee'}
Path('graph_verified.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
