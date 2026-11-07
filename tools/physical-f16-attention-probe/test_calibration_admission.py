import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('cal',ROOT/'tools/kaggle/lighton4b_physical_f16_calibration.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Admission(unittest.TestCase):
 def test_exact(self):m.admit(json.loads((ROOT/'docs/lighton4b-physical-f16-calibration-v1.json').read_text()))
 def test_mutations(self):
  cfg=json.loads((ROOT/'docs/lighton4b-physical-f16-calibration-v1.json').read_text())
  for k,v in [('case_ids',['simple_form']),('case_ids',list(reversed(m.CASE_IDS))),('form_receipt_sha256','0'*64),('physical_receipt_sha256','0'*64),('cap',257),('threads',2),('per_case_timeout_seconds',1900),('total_timeout_seconds',0)]:
   with self.subTest(key=k):
    c=copy.deepcopy(cfg);c[k]=v
    with self.assertRaises(RuntimeError):m.admit(c)
if __name__=='__main__':unittest.main()
