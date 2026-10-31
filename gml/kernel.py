from pathlib import Path
import torch

_extension = None


def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        source = Path(__file__).resolve().parent.parent / 'csrc'
        _extension = load(name='gml_dual_csr_v1', sources=[str(source/'binding.cpp'), str(source/'aggregate.cu')],
                          extra_cuda_cflags=['-O3'], verbose=False)
    return _extension


class CSRSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, graph):
        # Immutable graph structure is a non-activation resource, retained by Graph.
        # Saving it through autograd hooks would redundantly offload it per layer.
        ctx.graph = graph
        return extension().csr_sum(x.contiguous(), graph.ptr, graph.col)

    @staticmethod
    def backward(ctx, grad):
        graph = ctx.graph
        return extension().csr_sum(grad.contiguous(), graph.tptr, graph.tcol), None


def csr_sum(x, graph):
    if x.ndim != 2 or x.shape[0] != graph.nodes or x.device != graph.ptr.device:
        raise ValueError('Node matrix shape/device does not match graph')
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError('Custom kernel supports float32/float64 only')
    if not x.is_cuda:
        raise ValueError('Custom kernel requires CUDA')
    return CSRSum.apply(x, graph)
