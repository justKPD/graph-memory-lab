import copy
import json
from pathlib import Path
import tempfile
import unittest
from gml.agents import deliberate
from gml.evidence import Ledger


class EvidenceAgentTests(unittest.TestCase):
    def setUp(self):
        self.profiles=[{'id':'resident','graph_hash':'g1','extra_working_set_bytes':80,
                        'host_offload_payload_peak_bytes':0,'step_ms':10,'numerical_valid':True,'measured':True},
                       {'id':'offload','graph_hash':'g1','extra_working_set_bytes':20,
                        'host_offload_payload_peak_bytes':100,'step_ms':20,'numerical_valid':True,'measured':True}]
    def decide(self,profiles=None,budget=100,host=1000,age=0):
        return deliberate(profiles or self.profiles,'g1',budget,host,'resident',headroom=1,telemetry_age=age)
    def test_distinct_specialists_and_memory_veto(self):
        result=self.decide(budget=40)
        self.assertEqual(result['decision']['policy'],'offload')
        self.assertEqual([r['specialist'] for r in result['opinions']],['correctness','memory','performance'])
        self.assertIn('resident',result['opinions'][1]['rejected'])
    def test_wrong_graph_cannot_win(self):
        p=copy.deepcopy(self.profiles);p[0]['graph_hash']='other';p[0]['step_ms']=0.01
        self.assertEqual(self.decide(p)['decision']['policy'],'offload')
    def test_unknown_measurement_cannot_win(self):
        p=copy.deepcopy(self.profiles);p[0]['measured']=False
        self.assertEqual(self.decide(p)['decision']['policy'],'offload')
    def test_correctness_veto_even_if_cheap(self):
        p=copy.deepcopy(self.profiles);p[0]['numerical_valid']=False
        self.assertEqual(self.decide(p)['decision']['policy'],'offload')
    def test_host_exhaustion_pause(self):
        self.assertEqual(self.decide(budget=40,host=50)['decision']['action'],'PAUSE')
    def test_stale_observation_defers(self):
        self.assertEqual(self.decide(age=10)['decision']['action'],'DEFER')
    def test_chain_append_reopen_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl';ledger=Ledger(path)
            ledger.append('proposal',{'policy':'resident'});Ledger(path).append('execution',{'pass':True})
            self.assertEqual(len(Ledger.verify(path)),2)
    def test_changed_event_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl';ledger=Ledger(path)
            ledger.append('execution',{'pass':False})
            path.write_text(path.read_text().replace('false','true'))
            with self.assertRaises(ValueError):Ledger.verify(path)
    def test_truncated_prefix_has_different_external_root(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl';ledger=Ledger(path)
            ledger.append('a',{});ledger.append('b',{});original_root=ledger.previous
            path.write_text(path.read_text().splitlines()[0]+'\n')
            self.assertNotEqual(Ledger.verify(path)[-1]['hash'],original_root)


if __name__=='__main__':unittest.main()
