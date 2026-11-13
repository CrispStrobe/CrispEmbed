"""Continuation admission and exact unchanged resource-bound controls."""
import copy,json,tempfile,unittest
from pathlib import Path
import worker
class Controls(unittest.TestCase):
 def test_frozen_inventory_only_scope(self):
  cfg,_=worker.admit(Path(__file__).with_name('protocol.json'));self.assertEqual([v['id']for v in cfg['cases']],['inventory-table']);self.assertEqual(cfg['cases'][0]['baseline_tokens'],248);self.assertTrue(cfg['cases'][0]['baseline_eos']);self.assertEqual(cfg['cases'][0]['arm_seconds'],1200);self.assertEqual((cfg['cap'],cfg['context'],cfg['threads']),(1536,2048,4));self.assertFalse(cfg['runtime_rebuild_allowed']);self.assertFalse(cfg['observed_forward_allowed']);self.assertFalse(cfg['receipt_forward_allowed']);self.assertFalse(cfg['book_forward_allowed']);self.assertFalse(cfg['resource_thresholds_changed']);self.assertEqual(len(cfg['reused_completed_arms']),6)
 def test_protocol_change_rejected(self):
  cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());cfg['cases'][0]['arm_seconds']=2400
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'protocol.json';p.write_text(json.dumps(cfg,indent=2)+'\n')
   with self.assertRaises(RuntimeError):worker.admit(p)
 def test_each_predicate_retains_original_boundary(self):
  self.assertEqual(worker.admission_flags(10<<30,2<<30,True),{'memory':True,'output':True,'seal':True})
  self.assertEqual(worker.admission_flags((10<<30)-1,2<<30,True),{'memory':False,'output':True,'seal':True})
  self.assertEqual(worker.admission_flags(10<<30,(2<<30)-1,True),{'memory':True,'output':False,'seal':True})
  self.assertEqual(worker.admission_flags(10<<30,2<<30,False),{'memory':True,'output':True,'seal':False})
if __name__=='__main__':unittest.main()
