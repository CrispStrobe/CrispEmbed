#!/usr/bin/env python3
import copy,importlib.util,json,sys,tempfile,unittest
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('paired',ROOT/'tools/kaggle/lighton4b_q4_repack_table512.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Admission(unittest.TestCase):
 def test_frozen(self):m.admit(json.loads((ROOT/'docs/lighton4b-q4-repack-table512-v1.json').read_text()))
 def test_mutations(self):
  cfg=json.loads((ROOT/'docs/lighton4b-q4-repack-table512-v1.json').read_text())
  for key,value in [('case','simple_form'),('arms',['on','off']),('cap',256),('threads',2),('context',4096),('total_timeout_seconds',0),('per_arm_timeout_seconds',-1),('controls_receipt_sha256',None),('controls_receipt_sha256','0'*64),('controls_source_sha','0'*40),('table_receipt_sha256','0'*64),('table_source_sha','0'*40),('expected_candidate_sha256','0'*64),('driver_source_sha256','0'*64),('minimum_available_memory_bytes',0),('model_revision','main')]:
   with self.subTest(key=key):
    bad=copy.deepcopy(cfg);bad[key]=value
    with self.assertRaises(RuntimeError):m.admit(bad)
 def test_actual_scope_contract(self):
  calls=[]
  def run(label,args,env,limit):
   calls.append((label,args,limit));cap=int(args[7]);threads=int(args[8]);mode=args[9];flag=args[-1]if len(args)==13 else None
   valid=(flag is None and cap<=256)or(flag=='--extended-table-cap512'and cap==512 and threads==4 and mode=='greedy-full')
   raise m.subprocess.CalledProcessError(4 if valid else 2,args)
  rows=m.cli_controls(Path('native'),Path('work'),Path('out'),run,{})
  self.assertEqual(len(rows),9);self.assertEqual(len(calls),9);self.assertEqual([r['returncode']for r in rows],[4,2,2,4,2,2,2,2,2]);self.assertTrue(all(x[2]==10 for x in calls))
 def test_firstscore_closed_identity(self):
  with tempfile.TemporaryDirectory()as folder:
   root=Path(folder);raw=np.zeros(248320,dtype='<f4')
   for name in ['prefill.logits.f32','teacher.0.logits.f32','greedy.0.logits.f32']:raw.tofile(root/name)
   self.assertEqual(m.first_score(root,np).size,248320)
   changed=raw.copy();changed[0]=1;changed.tofile(root/'greedy.0.logits.f32')
   with self.assertRaises(RuntimeError):m.first_score(root,np)
 def test_firstscore_invalid_rows(self):
  with tempfile.TemporaryDirectory()as folder:
   root=Path(folder)
   for raw in [np.zeros(248319,dtype='<f4'),np.full(248320,np.nan,dtype='<f4')]:
    raw.tofile(root/'bad.f32')
    with self.assertRaises(RuntimeError):m.score_row(root/'bad.f32',np)
 def test_cancel_raises_for_group_cleanup(self):
  with self.assertRaises(SystemExit)as caught:m.interrupted(m.signal.SIGTERM,None)
  self.assertEqual(caught.exception.code,143)
 def test_scope_wrong_code_rejects(self):
  with self.assertRaises(RuntimeError):m.cli_controls(Path('n'),Path('w'),Path('o'),lambda *args:None,{})
if __name__=='__main__':unittest.main()
