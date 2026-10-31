from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint
from .kernel import csr_sum
from .memory import activation_offload, OffloadStats


def aggregate(x, graph, backend):
    src,dst=graph.edges
    if backend=='index_add':
        return torch.zeros_like(x).index_add(0,dst,x[src])
    if backend=='pyg':
        from torch_geometric.utils import scatter
        return scatter(x[src],dst,dim=0,dim_size=graph.nodes,reduce='sum')
    if backend=='cuda_csr':
        return csr_sum(x,graph)
    if backend=='sparse':
        # Values and indices are fixed graph resources; callers prebuild outside timing.
        return torch.sparse.mm(graph.sparse,x)
    raise ValueError('Unknown aggregation backend')


def prepare_sparse(graph, dtype=torch.float32, release_structures=False):
    src,dst=graph.edges
    # CSR library invariants require unique sorted column entries. Coalescing
    # sums duplicate weights, preserving multigraph sum semantics exactly in reals.
    graph.sparse=torch.sparse_coo_tensor(torch.stack((dst,src)),
                     torch.ones(src.numel(),device=graph.edges.device,dtype=dtype),
                     size=(graph.nodes,graph.nodes),check_invariants=True).coalesce().to_sparse_csr()
    if release_structures:
        graph.ptr=graph.col=graph.tptr=graph.tcol=None
    return graph


class GraphModel(torch.nn.Module):
    def __init__(self, width=64, depth=4, dropout=0.0):
        super().__init__()
        self.layers=torch.nn.ModuleList([torch.nn.Linear(width,width) for _ in range(depth)])
        self.dropout=torch.nn.Dropout(dropout)

    def forward(self, x, graph, backend='index_add', policy='resident'):
        if policy not in ('resident','checkpoint','offload'):
            raise ValueError('Unsupported activation policy')
        degree=graph.indegree.clamp_min(1).to(x.dtype).unsqueeze(1)
        def block(value, layer):
            h=layer(value).relu()
            h=aggregate(h,graph,backend)/degree
            return value+0.1*self.dropout(h)
        self.last_offload_stats=OffloadStats()
        context=activation_offload(self.last_offload_stats) if policy=='offload' else nullcontext()
        with context:
            for layer in self.layers:
                if policy=='checkpoint':
                    # Bind each layer to its own closure; preserve stochastic RNG state.
                    x=checkpoint(lambda value, current=layer:block(value,current),x,
                                 use_reentrant=False,preserve_rng_state=True)
                else:
                    x=block(x,layer)
        return x
