"""Trace actual training with Kineto; timings here are not benchmark results."""
import csv
import json
import os
import runpy
from collections import defaultdict
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
out=root/'results/profiler'
out.mkdir(parents=True,exist_ok=True)
from torch.utils.cpp_extension import get_default_build_root
candidates=list(Path(get_default_build_root()).rglob('gml_dual_csr_v1.pyd'))
if not candidates:
    raise FileNotFoundError('Run the extension validation before this capture.')
os.environ['GML_PROFILE_PREBUILT']=str(max(candidates,key=lambda p:p.stat().st_mtime))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True,profile_memory=True) as prof:
    runpy.run_path(str(root/'profile_workload.py'),run_name='__main__')
prof.export_chrome_trace(str(out/'graph_training.torch.json'))
groups=defaultdict(list)
for event in prof.events():
    if event.device_type==torch.autograd.DeviceType.CUDA:
        groups[event.name].append(event.time_range.elapsed_us())
if not any('segmented_gather_sum' in name for name in groups):
    raise AssertionError('Trace did not contain the actual custom CUDA kernel')
with (out/'kernel_summary.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['Name','Instances','Total_GPU_us','Mean_GPU_us'])
    for name,times in sorted(groups.items(),key=lambda item:sum(item[1]),reverse=True):
        writer.writerow([name,len(times),sum(times),sum(times)/len(times)])
result={'status':'PASS','tool':'PyTorch profiler / Kineto CPU+CUDA',
        'kernel_groups':len(groups),'cuda_events':sum(map(len,groups.values())),
        'custom_kernel_groups':[name for name in groups if 'segmented_gather_sum' in name],
        'workloads':['index_add/resident','cuda_csr/resident','cuda_csr/offload'],
        'scope':'Four actual training steps per configuration; instrumentation overhead; not performance evidence',
        'nsight_systems':'Initial launcher captures failed. Prebuilt-extension capture contains custom kernels but target was terminated after a shutdown wait; raw logs retained.',
        'nsight_compute':'ERR_NVGPUCTRPERM; permission setting unchanged'}
(out/'status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
