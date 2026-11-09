#!/usr/bin/env python3
"""Small protocol and real-source controls; no build or model."""
import copy,hashlib,json,unittest
from pathlib import Path
import worker
class Controls(unittest.TestCase):
 def setUp(self):self.root=Path(__file__).resolve().parents[2];self.cfg=json.loads((Path(__file__).parent/'protocol.json').read_text())
 def test_exact_protocol(self):
  worker.admit(self.cfg);self.assertEqual(sum(x['bytes']for x in self.cfg['tensors']),6030080);self.assertEqual(sum(len(c['runs'])for c in self.cfg['cases']),14)
 def test_fail_closed_mutations(self):
  mutations=[('threads',8),('tokens',[4,72]),('model_forward_allowed',True),('total_timeout_seconds',99999),('fused_gdn_ch',False),('recurrent_snapshots',32),('upstream_commit','main')]
  for k,v in mutations:
   with self.subTest(field=k):
    c=copy.deepcopy(self.cfg);c[k]=v
    with self.assertRaises(RuntimeError):worker.admit(c)
  for key in ['cases','tensors']:
   c=copy.deepcopy(self.cfg);c[key]=c[key][1:]
   with self.assertRaises(RuntimeError):worker.admit(c)
 def test_actual_native_source_pins(self):
  for name,digest in self.cfg['native_source_sha256'].items():self.assertEqual(hashlib.sha256((self.root/name).read_bytes()).hexdigest(),digest)
  for name,key in [('main.cpp','probe_cpp_sha256'),('CMakeLists.txt','probe_cmake_sha256')]:self.assertEqual(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest(),self.cfg[key])
 def test_exact_two_target_compile_profile(self):
  rows=[{'file':'/source/src/'+name,'command':'c++ -ffp-contract=off -o CMakeFiles/'+target+'.dir/test.o -c source.cpp'}for name in ['qwen35_delta.cpp','lighton4b_physical_gdn_source_ops.cpp']for target in ['gdn-hybrid-probe','gdn-hybrid-cache-controls']]
  self.assertEqual(len(worker.admit_compile(rows)),4)
  for mutation in [rows[:-1],rows+[rows[0]],[{**row,'command':row['command'].replace('-ffp-contract=off','')}if i==0 else row for i,row in enumerate(rows)],[{**row,'command':row['command'].replace('gdn-hybrid-probe','unknown')}if i==0 else row for i,row in enumerate(rows)]]:
   with self.assertRaises(RuntimeError):worker.admit_compile(mutation)
 def test_physical_row_maps(self):
  for role,width in [('qkv',8192),('z',4096),('a',32),('b',32)]:
   rows=worker.row_map(role,width);self.assertEqual(sorted(rows),list(range(width)))
  self.assertEqual(worker.row_map('qkv',8192)[:4096],list(range(4096)));self.assertEqual(worker.row_map('a',32)[:4],[0,16,1,17])
if __name__=='__main__':unittest.main()
