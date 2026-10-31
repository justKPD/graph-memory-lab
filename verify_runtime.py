"""Real CUDA execution and activation-policy gradient smoke checks, not a benchmark."""
import copy
import json
from pathlib import Path
import platform
import torch
from torch.utils.checkpoint import checkpoint

assert torch.cuda.is_available(), 'CUDA execution unavailable'
torch.manual_seed(17)
model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(),
                            torch.nn.Linear(128, 16)).cuda()
x = torch.randn(256, 64, device='cuda')
y = torch.randn(256, 16, device='cuda')
results = {}
reference = None
for mode in ('resident', 'checkpoint', 'offload'):
    m = copy.deepcopy(model)
    xx = x.detach().clone().requires_grad_(True)
    if mode == 'checkpoint':
        out = checkpoint(m, xx, use_reentrant=False)
    elif mode == 'offload':
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            out = m(xx)
    else:
        out = m(xx)
    loss = (out-y).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    state = [out.detach(), xx.grad.detach()] + [p.grad.detach() for p in m.parameters()]
    if reference is None:
        reference = [v.clone() for v in state]
    errors = [float((a-b).abs().max().item()) for a,b in zip(state, reference)]
    for a,b in zip(state, reference):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    results[mode] = {'passed':True,'loss':float(loss.item()),'maximum_abs_errors':errors}
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
report = {'python':platform.python_version(),'torch':torch.__version__,
          'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
          'capability':torch.cuda.get_device_capability(0),
          'compiled_architectures':torch.cuda.get_arch_list(),
          'nvml_total_bytes':memory.total,'nvml_free_bytes_at_check':memory.free,
          'checks':results,'is_performance_benchmark':False,
          'limitations':'Small MLP smoke check; not graph performance or council integration'}
pynvml.nvmlShutdown()
Path('runtime_verified.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
