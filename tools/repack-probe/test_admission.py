import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];s=importlib.util.spec_from_file_location('operator',ROOT/'tools/repack-probe/worker.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Admission(unittest.TestCase):
 def test_exact_protocol(self):m.admit(json.loads((ROOT/'docs/lighton4b-source-repack-operator-v1.json').read_text()))
 def test_closed_mutations(self):
  c=json.loads((ROOT/'docs/lighton4b-source-repack-operator-v1.json').read_text())
  for k,v in [('actual_receipt_sha256','0'*64),('native_ggml_source','0'*40),('upstream_source','0'*40),('source_model_bytes',1),('source_model_url','https://example.com'),('maximum_transfer_bytes',1),('tokens',[16]),('threads',2),('worker_seconds',3600)]:
   with self.subTest(key=k):
    x=copy.deepcopy(c);x[k]=v
    with self.assertRaises(RuntimeError):m.admit(x)
  for k,v in [('source_offset',0),('source_packed_sha256','0'*64),('input','native/prefill.norm.f32'),('source_shape',[2560,1])]:
   with self.subTest(row=k):
    x=copy.deepcopy(c);x['matrices'][0][k]=v
    with self.assertRaises(RuntimeError):m.admit(x)
 def test_bijective_known_head_mapping(self):
  for role,rows in [('qkv',8192),('z',4096),('a',32),('b',32),('gate',9216)]:
   values=m.row_map(role,rows);self.assertEqual(sorted(values),list(range(rows)))
  self.assertEqual(m.row_map('a',32)[:4],[0,16,1,17]);self.assertEqual(m.row_map('qkv',8192)[:4096],list(range(4096)))
  with self.assertRaises(RuntimeError):m.row_map('unknown',32)
if __name__=='__main__':unittest.main()
