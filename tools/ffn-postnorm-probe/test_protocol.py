"""Small provenance and code-space controls; no build/model inference."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import worker
HERE=Path(__file__).parent
class Controls(unittest.TestCase):
 def setUp(self):self.c=json.loads((HERE/'protocol.json').read_text())
 def test_closed_protocol_source_and_prior(self):
  worker.admit(self.c)
  for path,key in [('main.cpp','adapter_cpp_sha256'),('CMakeLists.txt','adapter_cmake_sha256')]:self.assertEqual(hashlib.sha256((HERE/path).read_bytes()).hexdigest(),self.c[key])
  for path,key in [('protocol.json','prior_protocol_sha256'),('worker.py','prior_worker_sha256')]:self.assertEqual(hashlib.sha256((worker.ROOT/'tools/swiglu-source-probe'/path).read_bytes()).hexdigest(),self.c[key])
 def test_mutated_domains_are_rejected(self):
  for key,value in [('tokens',{'source':[72]}),('threads',8),('model_forward_allowed',True),('total_timeout_seconds',99999),('prior_receipt_sha256','unknown')]:
   c=copy.deepcopy(self.c);c[key]=value
   with self.subTest(key=key),self.assertRaises(RuntimeError):worker.admit(c)
  c=copy.deepcopy(self.c);c['weights'][0]['type']=14
  with self.assertRaises(RuntimeError):worker.admit(c)
 def test_coupled_q8_layout_and_independent_corners(self):
  dtype=np.dtype([('d','<f4'),('qs','i1',(256,)),('bsums','<i2',(16,))]);a=np.zeros(40,dtype=dtype);a['d']=1;a['qs'][0,0]=1;a['bsums'][0,0]=1;b=a.copy();b['d']=2;b['qs'][0,0]=2;b['bsums'][0,0]=2;snapshot=a.tobytes(),b.tobytes();v=worker.corners(a,b)
  self.assertEqual((a.tobytes(),b.tobytes()),snapshot);self.assertEqual(v['source'].tobytes(),a.tobytes());self.assertEqual(v['native'].tobytes(),b.tobytes());self.assertTrue(np.array_equal(v['source_qs_native_scale']['qs'],a['qs']));self.assertTrue(np.array_equal(v['source_qs_native_scale']['d'],b['d']));self.assertTrue(np.array_equal(v['native_qs_source_scale']['bsums'],b['bsums']))
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'codes.bin'
   for values in v.values():values.tofile(p);worker.code_array(np,p,4)
   nonfinite=a.copy();nonfinite['d'][0]=np.nan;wrong_sum=a.copy();wrong_sum['bsums'][0,0]=0
   for bad in [a[:1],nonfinite,wrong_sum]:
    bad.tofile(p)
    with self.assertRaises(RuntimeError):worker.code_array(np,p,4)
  diff=worker.code_diff(np,b,a);self.assertEqual(diff['different_qs'],1);self.assertEqual(diff['different_bsums'],1);self.assertEqual(diff['different_scales'],40);self.assertEqual(diff['scale_sign_changes'],0)
if __name__=='__main__':unittest.main()
