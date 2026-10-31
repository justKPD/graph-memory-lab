"""Additional stream/dtype and mutant checks after the main correctness suite."""
import json
from pathlib import Path
import torch
from gml.graph import Graph
from gml.kernel import csr_sum,extension
from gml.model import aggregate

extension()
graph=Graph.create(6,torch.tensor([[0,0,1,2,4],[1,1,2,2,0]],device='cuda'))
x=torch.randn(6,8,device='cuda',dtype=torch.float64,requires_grad=True)
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    value=csr_sum(x,graph);value.sum().backward()
stream.synchronize()
torch.testing.assert_close(value,aggregate(x,graph,'index_add'))
expected=torch.bincount(graph.edges[0],minlength=6).double().unsqueeze(1).expand_as(x)
torch.testing.assert_close(x.grad,expected)
try:csr_sum(x.half(),graph)
except ValueError:pass
else:raise AssertionError('Unsupported half precision was accepted')
detached=aggregate(x,graph,'index_add').detach()
try:torch.autograd.grad(detached.sum(),x)
except RuntimeError:pass
else:raise AssertionError('Detached mutant unexpectedly had a gradient path')
ones=torch.ones_like(x)
summed=aggregate(ones,graph,'index_add')
mean=summed/graph.indegree.clamp_min(1).unsqueeze(1)
assert not torch.allclose(summed,mean)
result={'status':'PASS','checks':['non-default-stream-forward-backward','unsupported-half-refused',
                                'detached-gradient-mutant-detected','sum-mean-mutant-detected'],
        'count':4,'is_benchmark':False}
Path('results/additional_cuda.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
