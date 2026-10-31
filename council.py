"""CPU-side policy arbitration. No GPU operations or LLM calls occur here."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    workload: str
    device_free_bytes: int
    host_free_bytes: int
    age_seconds: float
    at_step_boundary: bool


@dataclass(frozen=True)
class Candidate:
    name: str
    workload: str
    incremental_device_bytes: int
    incremental_host_bytes: int
    step_ms: float
    switch_ms: float
    numerical_valid: bool
    measured: bool


@dataclass(frozen=True)
class Decision:
    action: str
    policy: str | None
    reasons: tuple[str, ...]


class Council:
    """Specialists supply constraints; numerical/feasibility vetoes beat speed.

    Memory estimates must be incremental from the same observation boundary,
    not total allocated/reserved peaks. Selection does NOT execute a switch.
    """
    def __init__(self, device_headroom=256 * 1024**2, host_headroom=256 * 1024**2,
                 max_age=2.0, min_gain=0.05, cooldown_steps=10):
        if min(device_headroom, host_headroom, max_age, min_gain, cooldown_steps) < 0:
            raise ValueError('Controller bounds must be nonnegative')
        if not all(math.isfinite(x) for x in (device_headroom,host_headroom,max_age,min_gain,cooldown_steps)):
            raise ValueError('Controller bounds must be finite')
        if min_gain >= 1:
            raise ValueError('min_gain must be below one')
        self.device_headroom = device_headroom
        self.host_headroom = host_headroom
        self.max_age = max_age
        self.min_gain = min_gain
        self.cooldown_steps = cooldown_steps

    def decide(self, observation, candidates, current, steps_since_switch,
               expected_remaining_steps):
        if not observation.at_step_boundary:
            return Decision('DEFER', None, ('Not at a supported step boundary',))
        if (not math.isfinite(observation.age_seconds)
                or not 0 <= observation.age_seconds <= self.max_age
                or not all(math.isfinite(v) for v in (observation.device_free_bytes,observation.host_free_bytes))
                or min(observation.device_free_bytes, observation.host_free_bytes) < 0):
            return Decision('DEFER', None, ('Stale or invalid telemetry',))
        if (not all(math.isfinite(v) for v in (expected_remaining_steps,steps_since_switch))
                or expected_remaining_steps <= 0 or steps_since_switch < 0):
            raise ValueError('Invalid planning horizon or switch age')
        candidates = tuple(candidates)
        if len({c.name for c in candidates}) != len(candidates):
            raise ValueError('Policy identifiers must be unique')
        accepted, reasons = [], []
        for c in candidates:
            # Correctness specialist: an evidence match, not majority voting.
            if c.workload != observation.workload or c.numerical_valid is not True or c.measured is not True:
                reasons.append(f'{c.name}: missing compatible validated measurements')
                continue
            if (not all(math.isfinite(v) for v in (c.incremental_device_bytes,c.incremental_host_bytes))
                    or min(c.incremental_device_bytes, c.incremental_host_bytes) < 0
                    or not math.isfinite(c.step_ms) or c.step_ms <= 0
                    or not math.isfinite(c.switch_ms) or c.switch_ms < 0):
                reasons.append(f'{c.name}: invalid resource/cost estimate')
                continue
            # Memory specialist: device savings must not exhaust host memory.
            if (c.incremental_device_bytes + self.device_headroom > observation.device_free_bytes
                    or c.incremental_host_bytes + self.host_headroom > observation.host_free_bytes):
                reasons.append(f'{c.name}: insufficient memory headroom')
                continue
            accepted.append(c)
        if not accepted:
            return Decision('PAUSE', None, tuple(reasons + ['No measured valid feasible policy']))
        live = next((c for c in accepted if c.name == current), None)
        # Performance specialist: include switching cost in the remaining horizon.
        def cost(c):
            return c.step_ms + (0 if c.name == current else c.switch_ms / expected_remaining_steps)
        best = min(accepted, key=lambda c: (cost(c), c.name))
        if live and best.name != current:
            gain = (cost(live) - cost(best)) / cost(live)
            if steps_since_switch < self.cooldown_steps or gain < self.min_gain:
                return Decision('KEEP', current, tuple(reasons + ['Hysteresis/cooldown prevents churn']))
        action = 'KEEP' if best.name == current else 'PROPOSE_SWITCH'
        return Decision(action, best.name, tuple(reasons + ['Selected by validated horizon cost']))
