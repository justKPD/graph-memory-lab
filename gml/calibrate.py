"""Bounded real-device transfer calibration; current-stream copies, no overlap claim."""
import json
from pathlib import Path
import statistics
import torch


def run(output='results/calibration.json'):
    records=[]
    for mib in (1,8,32):
        count=mib*2**20//4
        host=torch.ones(count,pin_memory=True)
        device=torch.ones(count,device='cuda')
        for direction in ('H2D','D2H'):
            source,target=(host,device) if direction=='H2D' else (device,host)
            for _ in range(3):target.copy_(source,non_blocking=True)
            torch.cuda.synchronize()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            samples=[]
            for _ in range(10):
                a.record();target.copy_(source,non_blocking=True);b.record();b.synchronize()
                samples.append(a.elapsed_time(b))
            median=statistics.median(samples)
            records.append({'payload_mib':mib,'direction':direction,'gpu_ms':samples,
                            'median_gpu_ms':median,'GB_per_second':mib*2**20/(median*1e6)})
        torch.testing.assert_close(device.cpu(),host)
        del host,device
    result={'device':torch.cuda.get_device_name(),'records':records,
            'scope':'Pinned host buffers, default/current stream, serialized copies; isolated sizes, no concurrent-transfer/compute model.',
            'limitations':'Do not extrapolate to NVLink-C2C or assume these bandwidths hold under contention.'}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
