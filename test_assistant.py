import unittest
from gml.assistant import EvidenceAssistant


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.agent=EvidenceAssistant({'manifest_sha256':'x','manifest':{'limitations':[]},'profiles':[
          {'id':'uniform_train/cuda_csr/resident','workload':'uniform_train','graph_hash':'g1',
           'step_ms':2.0,'peak_allocated_bytes':64*2**20,'numerical_valid':True,'raw_sources':['run:1'],
           'raw_run_ids':['r1']}]})
    def test_compare_measured(self):
        result=self.agent.execute(self.agent.parse('compare uniform_train budget 100'))
        self.assertEqual(result['chosen'],'uniform_train/cuda_csr/resident')
    def test_no_feasible_candidate(self):
        self.assertIsNone(self.agent.execute({'action':'compare','workload':'uniform_train','budget_mib':1})['chosen'])
    def test_failed_validation_excluded(self):
        self.agent.registry['profiles'][0]['numerical_valid']=False
        self.assertIsNone(self.agent.execute({'action':'compare','workload':'uniform_train'})['chosen'])
    def test_stale_graph_refused(self):
        with self.assertRaises(ValueError):
            self.agent.execute({'action':'compare','workload':'uniform_train','expected_graph_hash':'other'})
    def test_wrong_precision_refused(self):
        with self.assertRaises(ValueError):self.agent.execute({'action':'compare','workload':'uniform_train','precision':'float16'})
    def test_tampering_field_refused(self):
        with self.assertRaises(ValueError):self.agent.execute({'action':'compare','workload':'uniform_train','numerical_valid':True})
    def test_arbitrary_shell_refused(self):
        with self.assertRaises(ValueError):self.agent.parse('run powershell delete things')
    def test_new_graph_not_claimed_measured(self):
        with self.assertRaises(ValueError):self.agent.execute({'action':'compare','workload':'unknown'})
    def test_proposed_is_not_executed(self):
        r=self.agent.execute(self.agent.parse('propose uniform_train budget 100'))
        self.assertEqual(r['plan']['status'],'PROPOSED_NOT_EXECUTED')
        self.assertFalse(r['plan']['arbitrary_commands'])
    def test_nan_budget_refused(self):
        with self.assertRaises(ValueError):self.agent.execute({'action':'compare','workload':'uniform_train','budget_mib':float('nan')})
    def test_unknown_profile_refused(self):
        with self.assertRaises(ValueError):self.agent.execute({'action':'explain','profile_id':'missing'})


if __name__=='__main__':unittest.main()
