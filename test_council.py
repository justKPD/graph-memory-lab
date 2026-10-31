import unittest
from dataclasses import replace
from council import Candidate, Council, Observation


class CouncilTests(unittest.TestCase):
    def setUp(self):
        self.council = Council(device_headroom=10, host_headroom=10)
        self.obs = Observation('graph-v1/fp32/loss-v1', 100, 200, 0.1, True)
        self.resident = Candidate('resident', self.obs.workload, 60, 0, 10, 0, True, True)
        self.offload = Candidate('offload', self.obs.workload, 20, 80, 12, 5, True, True)

    def choose(self, obs=None, cs=None, age=100, horizon=100):
        return self.council.decide(obs or self.obs, cs or [self.resident, self.offload],
                                   'resident', age, horizon)

    def test_keep_fast_resident(self):
        self.assertEqual(self.choose().policy, 'resident')

    def test_external_pressure_proposes_offload(self):
        d = self.choose(obs=replace(self.obs, device_free_bytes=40))
        self.assertEqual((d.action, d.policy), ('PROPOSE_SWITCH', 'offload'))

    def test_correctness_veto_beats_speed(self):
        bad = replace(self.offload, step_ms=1, numerical_valid=False)
        self.assertEqual(self.choose(cs=[self.resident, bad]).policy, 'resident')

    def test_host_pressure_pauses(self):
        d = self.choose(obs=replace(self.obs, device_free_bytes=40, host_free_bytes=50))
        self.assertEqual(d.action, 'PAUSE')

    def test_stale_telemetry_defers(self):
        self.assertEqual(self.choose(obs=replace(self.obs, age_seconds=5)).action, 'DEFER')

    def test_no_midstep_switch(self):
        self.assertEqual(self.choose(obs=replace(self.obs, at_step_boundary=False)).action, 'DEFER')

    def test_cooldown(self):
        self.assertEqual(self.choose(cs=[self.resident, replace(self.offload, step_ms=5)], age=1).action, 'KEEP')

    def test_switch_cost_changes_short_horizon_decision(self):
        fast = replace(self.offload, step_ms=5, switch_ms=100)
        self.assertEqual(self.choose(cs=[self.resident, fast], horizon=1).action, 'KEEP')
        self.assertEqual(self.choose(cs=[self.resident, fast], horizon=100).policy, 'offload')

    def test_graph_identity_mismatch_rejected(self):
        self.assertEqual(self.choose(cs=[replace(self.offload, workload='other')]).action, 'PAUSE')

    def test_prediction_not_promoted_to_measurement(self):
        self.assertEqual(self.choose(cs=[replace(self.offload, measured=False)]).action, 'PAUSE')

    def test_nan_latency_rejected(self):
        self.assertEqual(self.choose(cs=[replace(self.offload, step_ms=float('nan'))]).action, 'PAUSE')

    def test_memory_pressure_overrides_cooldown(self):
        d = self.choose(obs=replace(self.obs, device_free_bytes=40), age=0)
        self.assertEqual(d.action, 'PROPOSE_SWITCH')

    def test_duplicate_policy_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.choose(cs=[self.resident, self.resident])
    def test_nan_memory_estimate_rejected(self):
        self.assertEqual(self.choose(cs=[replace(self.offload,incremental_device_bytes=float('nan'))]).action,'PAUSE')
    def test_nan_free_memory_defers(self):
        self.assertEqual(self.choose(obs=replace(self.obs,device_free_bytes=float('nan'))).action,'DEFER')
    def test_non_boolean_validation_rejected(self):
        self.assertEqual(self.choose(cs=[replace(self.offload,numerical_valid='PASS')]).action,'PAUSE')
    def test_nonfinite_horizon_rejected(self):
        with self.assertRaises(ValueError):self.choose(horizon=float('inf'))


if __name__ == '__main__':
    unittest.main()
