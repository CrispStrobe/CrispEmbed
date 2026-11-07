import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('extended',ROOT/'tools/kaggle/lighton4b_physical_f16_table512.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Admission(unittest.TestCase):
 def test_closed_protocol(self):
  cfg=json.loads((ROOT/'docs/lighton4b-physical-f16-table512-v1.json').read_text());m.admit(cfg)
  for k,v in [('case','simple_form'),('cap',256),('threads',2),('context',4096),('form_receipt_sha256','0'*64),('physical_receipt_sha256','0'*64),('calibration_receipt_sha256','0'*64),('total_timeout_seconds',0),('per_arm_timeout_seconds',1500),('arms',['native']),('driver_source_sha256','0'*64)]:
   with self.subTest(key=k):
    c=copy.deepcopy(cfg);c[k]=v
    with self.assertRaises(RuntimeError):m.admit(c)
 def test_arm_and_global_deadline(self):
  self.assertEqual(m.remaining_budget(100,500,200),300)
  self.assertEqual(m.remaining_budget(100,None,200),3500)
  for arm,now in [(500,500),(500,501),(None,3700)]:
   with self.assertRaises(RuntimeError):m.remaining_budget(100,arm,now)
 def test_old_prefix_and_eos(self):
  old={'tokens':list(range(256)),'completion_probabilities':[{'p':1}]*256};r={'tokens':old['tokens']+[248046],'completion_probabilities':old['completion_probabilities']+[{'p':1}],'truncated':False,'tokens_evaluated':86,'stop':True,'stop_type':'eos'};m.validate_source(r,old)
  for key,val in [('tokens',[9]*257),('completion_probabilities',[]),('stop_type','limit'),('truncated',True),('tokens_evaluated',85),('stop',False)]:
   c=copy.deepcopy(r);c[key]=val
   with self.assertRaises(RuntimeError):m.validate_source(c,old)
if __name__=='__main__':unittest.main()
