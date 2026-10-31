"""Small deterministic specialists sharing evidence; no language model loaded."""
from dataclasses import asdict, dataclass
from council import Candidate, Council, Observation


@dataclass
class Opinion:
    specialist: str
    accepted: list
    rejected: dict


class CorrectnessAgent:
    def review(self,profiles,graph_hash):
        rejected={p['id']:'Missing compatible validated evidence' for p in profiles
                  if not p.get('numerical_valid') or not p.get('measured') or p['graph_hash']!=graph_hash}
        return Opinion('correctness',[p['id'] for p in profiles if p['id'] not in rejected],rejected)


class MemoryAgent:
    def review(self,profiles,extra_budget,host_budget,headroom):
        rejected={}
        for p in profiles:
            if p['extra_working_set_bytes']+headroom>extra_budget:
                rejected[p['id']]='Extra GPU working-set budget exceeded'
            elif p['host_offload_payload_peak_bytes']+headroom>host_budget:
                rejected[p['id']]='Host offload-payload budget exceeded'
        return Opinion('memory',[p['id'] for p in profiles if p['id'] not in rejected],rejected)


class PerformanceAgent:
    def review(self,profiles,current,horizon,switch_ms):
        costs={p['id']:p['step_ms']+(0 if p['id']==current else switch_ms/horizon) for p in profiles}
        return {'specialist':'performance','horizon_steps':horizon,'predicted_amortized_ms':costs,
                'basis':'Measured registry medians; switching estimate stated separately'}


def deliberate(profiles,graph_hash,extra_budget,host_budget,current,age=100,horizon=10,
               telemetry_age=0.0,switch_ms=0.0,headroom=1024**2):
    correct=CorrectnessAgent().review(profiles,graph_hash)
    memory=MemoryAgent().review(profiles,extra_budget,host_budget,headroom)
    performance=PerformanceAgent().review(profiles,current,horizon,switch_ms)
    candidates=[Candidate(p['id'],p['graph_hash'],p['extra_working_set_bytes'],
                          p['host_offload_payload_peak_bytes'],p['step_ms'],switch_ms,
                          p['id'] in correct.accepted,p.get('measured',False)) for p in profiles]
    council=Council(device_headroom=headroom,host_headroom=headroom,min_gain=0.05,cooldown_steps=2)
    decision=council.decide(Observation(graph_hash,extra_budget,host_budget,telemetry_age,True),
                            candidates,current,age,horizon)
    return {'decision':asdict(decision),'opinions':[asdict(correct),asdict(memory),performance]}
