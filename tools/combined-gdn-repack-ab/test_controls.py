import ast,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
s=importlib.util.spec_from_file_location('gdn_full_ab',HERE/'worker.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class Controls(unittest.TestCase):
 def test_outer_receipts_remain_outside_child_inventory(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'outer.log').write_text('active');(p/'bootstrap.json').write_text('active');(p/'done.log').write_text('closed');(p/'CMakeCache.txt').write_text('closed');(p/'case').mkdir();(p/'case/generated.txt').write_text('closed')
   self.assertEqual(set(w.artifacts(p,{'done':{}})),{'done.log','CMakeCache.txt','case/generated.txt'})
 def test_transfer_retries_owned_partial_only_and_deadline_closed(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'owned';records=[]
   def fetch(*args):
    self.assertFalse(p.exists());p.write_bytes(b'x')
    if len(records)==1:raise TimeoutError('transport')
   w.transfer(fetch,'unused',p,1,'frozen',records);self.assertEqual([r['status']for r in records],['transport_failure','authenticated'])
   with self.assertRaises(RuntimeError):w.transfer(fetch,'unused',p,1,'frozen',[])
  for error in [w.WorkerDeadline('global'),ValueError('integrity')]:
   with tempfile.TemporaryDirectory()as d:
    p=Path(d)/'owned';records=[]
    def fail(*args):raise error
    with self.assertRaises(type(error)):w.transfer(fail,'unused',p,1,'frozen',records)
    self.assertEqual(len(records),1)
 def test_worker_syntax_and_frozen_protocol(self):
  ast.parse((HERE/'worker.py').read_text());w.admit_protocol(w.ROOT/'docs/lighton4b-combined-gdn-repack-ab-v1.json')
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'changed';p.write_bytes((w.ROOT/'docs/lighton4b-combined-gdn-repack-ab-v1.json').read_bytes()+b' ')
   with self.assertRaisesRegex(RuntimeError,'frozen protocol'):w.admit_protocol(p)
 def test_exact_two_mode_change(self):
  source={'CRISPEMBED_LIGHTON4B_HEAD_BF16':'0',w.GATE:'1','OTHER':'same'};off,on=w.gates(source)
  self.assertEqual(source[w.GATE],'1');self.assertEqual(off[w.GATE],'0');self.assertEqual(on[w.GATE],'1');self.assertEqual({k for k in off if off[k]!=on[k]},w.CHANGED)
 def test_complete_finite_argmax_and_first_row_identity(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);x=np.zeros(248320,dtype='<f4');x[27]=2
   for name in ['greedy.0.logits.f32','prefill.logits.f32','teacher.0.logits.f32']:x.tofile(p/name)
   w.rows(p,[27],np)
   with self.assertRaises(RuntimeError):w.rows(p,[28],np)
   y=x.copy();y[0]=np.nan;y.tofile(p/'greedy.0.logits.f32')
   with self.assertRaises(RuntimeError):w.rows(p,[27],np)
   x.tofile(p/'greedy.0.logits.f32');y=x.copy();y[1]=1;y.tofile(p/'teacher.0.logits.f32')
   with self.assertRaisesRegex(RuntimeError,'first vocabulary'):w.rows(p,[27],np)
if __name__=='__main__':unittest.main()
