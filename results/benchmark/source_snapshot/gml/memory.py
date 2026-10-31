from dataclasses import dataclass
import weakref
import torch


@dataclass
class OffloadStats:
    to_host_bytes: int=0
    to_device_bytes: int=0
    host_live_bytes: int=0
    host_peak_bytes: int=0
    packed_tensors: int=0

    def release(self,size):
        self.host_live_bytes-=size


def activation_offload(stats):
    """Offload dense floating saved tensors; fixed sparse/integer graph data stays.

    Uses PyTorch's pinned-copy implementation, including its stream semantics.
    Logical live payload counts do not equal allocator-reserved host memory.
    """
    base=torch.autograd.graph.save_on_cpu(pin_memory=True)
    def pack(tensor):
        if tensor.layout!=torch.strided or not tensor.is_floating_point() or not tensor.is_cuda:
            return ('resident',tensor)
        payload=base.pack_hook(tensor)
        size=payload[1].numel()*payload[1].element_size()
        stats.to_host_bytes+=size
        stats.host_live_bytes+=size
        stats.host_peak_bytes=max(stats.host_peak_bytes,stats.host_live_bytes)
        stats.packed_tensors+=1
        weakref.finalize(payload[1],stats.release,size)
        return ('offload',payload)
    def unpack(payload):
        tag,value=payload
        if tag=='resident':return value
        stats.to_device_bytes+=value[1].numel()*value[1].element_size()
        return base.unpack_hook(value)
    return torch.autograd.graph.saved_tensors_hooks(pack,unpack)
