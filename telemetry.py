"""Read-only device-wide GPU telemetry. No GPU context, kernels, or LLMs."""
import argparse
import json
from pathlib import Path
import time
import psutil
import pynvml


def observe(handle):
    start = time.perf_counter()
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    def optional(call):
        try:
            return call()
        except pynvml.NVMLError:
            return None
    util = optional(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
    snapshot = {
        'timestamp_unix':time.time(),
        'monotonic_seconds':time.monotonic(),
        'source':'NVML device-wide; not per-process tensor accounting',
        'device_uuid':pynvml.nvmlDeviceGetUUID(handle),
        'device_total_bytes':memory.total,
        'device_free_bytes':memory.free,
        'device_used_bytes':memory.used,
        'host_available_bytes':psutil.virtual_memory().available,
        'gpu_utilization_percent':None if util is None else util.gpu,
        'temperature_c':optional(lambda: pynvml.nvmlDeviceGetTemperature(handle,0)),
        'power_mw':optional(lambda: pynvml.nvmlDeviceGetPowerUsage(handle)),
    }
    snapshot['collection_ms']=(time.perf_counter()-start)*1000
    return snapshot


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--interval',type=float,default=0.2)
    parser.add_argument('--output',default='telemetry_setup.json')
    args=parser.parse_args()
    if not 1 <= args.samples <= 10000 or args.interval < 0:
        parser.error('Invalid bounded sampling request')
    pynvml.nvmlInit()
    try:
        handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        records=[]
        for i in range(args.samples):
            if i:
                time.sleep(args.interval)
            records.append(observe(handle))
        Path(args.output).write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(json.dumps({'samples':len(records),'output':args.output,
                          'last_free_bytes':records[-1]['device_free_bytes'],
                          'max_collection_ms':max(r['collection_ms'] for r in records)}))
    finally:
        pynvml.nvmlShutdown()
