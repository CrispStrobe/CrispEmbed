"""Small closed-protocol and evidence controls; no build or inference."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import worker
HERE=Path(__file__).parent
class Controls(unittest.TestCase):
 def setUp(self):self.c=json.loads((HERE/'protocol.json').read_text())
 def test_closed_actual_types_and_source(self):
  worker.admit(self.c);self.assertEqual(self.c['parameter_bytes'],45895680)
  for path,digest in self.c['native_sources'].items():self.assertEqual(hashlib.sha256((worker.ROOT/path).read_bytes()).hexdigest(),digest)
  for path,key in [('main.cpp','adapter_cpp_sha256'),('CMakeLists.txt','adapter_cmake_sha256')]:self.assertEqual(hashlib.sha256((HERE/path).read_bytes()).hexdigest(),self.c[key])
 def test_fail_closed_geometry_domains_and_original_weights(self):
  for key,value in [('threads',8),('tokens',{'source':[72]}),('model_forward_allowed',True),('source_sha256','unverified'),('ab_receipt_sha256','wrong'),('total_timeout_seconds',99999)]:
   changed=copy.deepcopy(self.c);changed[key]=value
   with self.subTest(key=key),self.assertRaises(RuntimeError):worker.admit(changed)
  for field,value in [('type',12),('source_offset',0),('source_packed_sha256','wrong'),('whole_rows_permuted',32),('quant_blocks_unchanged',False)]:
   changed=copy.deepcopy(self.c);changed['weights'][2][field]=value
   with self.subTest(field=field),self.assertRaises(RuntimeError):worker.admit(changed)
 def test_numeric_magnitude_and_bits_are_distinct(self):
  a=np.array([0.0,2.0],dtype='<f4');b=np.array([-0.0,2.0],dtype='<f4');r=worker.stats(np,a,b);self.assertFalse(r['bit_exact']);self.assertEqual(r['rms'],0);self.assertEqual(r['different_f32_words'],1);self.assertAlmostEqual(r['reference_rms_magnitude'],2**.5)
  for bad in [np.array([np.nan,2],dtype='<f4'),np.array([2],dtype='<f4')]:
   with self.assertRaises(RuntimeError):worker.stats(np,a,bad)
 def test_only_closed_worker_owned_root_outputs(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'outer.log').write_text('active');(p/'bootstrap.json').write_text('active');(p/'closed.log').write_text('done');(p/'cases').mkdir();(p/'cases/output.f32').write_bytes(b'1234')
   self.assertEqual(set(worker.inventory(p,{'closed':{}})),{'closed.log','cases/output.f32'})
if __name__=='__main__':unittest.main()
