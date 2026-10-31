from dataclasses import dataclass
import hashlib
import time
import torch


@dataclass
class Graph:
    """Validated immutable directed multigraph; mutate only by constructing a new Graph."""
    nodes: int
    edges: torch.Tensor
    ptr: torch.Tensor
    col: torch.Tensor
    tptr: torch.Tensor
    tcol: torch.Tensor
    digest: str
    conversion_ms: float
    family: str
    indegree: torch.Tensor

    @classmethod
    def create(cls, nodes, edges, family='provided', prepare_csr=True):
        if not isinstance(nodes, int) or nodes < 0:
            raise ValueError('Nonnegative integer node count required')
        if edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError('Edges must be int64 [2,E]')
        if edges.numel() and (int(edges.min())<0 or int(edges.max())>=nodes):
            raise ValueError('Edge index outside node range')
        edges=edges.contiguous().clone()
        digest=hashlib.sha256(str(nodes).encode()+edges.cpu().numpy().tobytes()).hexdigest()
        if edges.is_cuda:
            torch.cuda.synchronize()
        start=time.perf_counter()
        src,dst=edges
        def make(row, column):
            order=torch.argsort(row,stable=True)
            count=torch.bincount(row,minlength=nodes)
            ptr=torch.cat([count.new_zeros(1),count.cumsum(0)])
            return ptr.contiguous(),column[order].contiguous()
        indegree=torch.bincount(dst,minlength=nodes)
        if prepare_csr:
            ptr,col=make(dst,src)
            tptr,tcol=make(src,dst)
        else:
            ptr=col=tptr=tcol=None
        if edges.is_cuda:
            torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        return cls(nodes,edges,ptr,col,tptr,tcol,digest,elapsed,family,indegree)

    @property
    def structural_bytes(self):
        tensors=[self.edges,self.ptr,self.col,self.tptr,self.tcol,self.indegree]
        if hasattr(self,'sparse'):
            tensors += [self.sparse.crow_indices(),self.sparse.col_indices(),self.sparse.values()]
        return sum(t.numel()*t.element_size() for t in tensors if t is not None)


def generate(family, nodes, degree, seed, device='cuda',prepare_csr=True):
    gen=torch.Generator().manual_seed(seed)
    if family=='karate':
        import networkx as nx
        g=nx.karate_club_graph()
        pairs=list(g.edges())
        # Explicitly discard social interaction weights: unweighted, bidirectional sum.
        e=torch.tensor(pairs+[(v,u) for u,v in pairs],dtype=torch.int64).t()
        return Graph.create(len(g),e.to(device),'karate-unweighted-bidirectional',prepare_csr)
    if nodes<1 or degree<0:
        raise ValueError('Invalid generated workload')
    count=nodes*degree
    src=torch.randint(nodes,(count,),generator=gen)
    if family=='uniform':
        dst=torch.randint(nodes,(count,),generator=gen)
    elif family=='skewed':
        dst=(torch.rand(count,generator=gen).pow(3)*nodes).long().clamp_max(nodes-1)
    elif family=='ring':
        src=torch.arange(nodes).repeat_interleave(degree)
        dst=(src+torch.arange(1,degree+1).repeat(nodes))%nodes
    else:
        raise ValueError('Unknown graph family')
    return Graph.create(nodes,torch.stack((src,dst)).to(device),family,prepare_csr)
