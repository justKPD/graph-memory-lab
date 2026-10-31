"""Actual graph training for Nsight Systems; not used for performance numbers."""
import torch
import os
import importlib.util
from gml.graph import generate
from gml.kernel import extension
from gml.model import GraphModel

if os.environ.get('GML_PROFILE_PREBUILT'):
    # Keep compiler/launcher processes out of the training trace. The path must
    # identify the extension already built and checked in this local environment.
    import gml.kernel as kernel
    spec=importlib.util.spec_from_file_location('gml_dual_csr_v1',os.environ['GML_PROFILE_PREBUILT'])
    kernel._extension=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel._extension)
else:
    extension()
torch.manual_seed(101)
graph=generate('uniform',4096,16,123)
model=GraphModel(64,4).cuda()
x=torch.randn(4096,64,device='cuda',requires_grad=True)
target=torch.randn_like(x)
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
for backend,policy in [('index_add','resident'),('cuda_csr','resident'),('cuda_csr','offload')]:
    for step in range(4):
        optimizer.zero_grad(set_to_none=True);x.grad=None
        torch.cuda.nvtx.range_push(f'{backend}/{policy}/step{step}')
        value=model(x,graph,backend,policy)
        loss=(value-target).square().mean();loss.backward();optimizer.step()
        torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
print('Graph training trace complete: index-add/resident, CUDA CSR/resident and CUDA CSR/offload')
