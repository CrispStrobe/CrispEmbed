import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(Path(__file__).parent))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('canonical',Path(__file__).with_name('canonical.py'))
class Controls(unittest.TestCase):
 def test_frozen_domain(self):m.admit(json.loads((ROOT/'docs/lighton4b-projector-canonical-v1.json').read_text()))
 def test_corrupt_manifest_rejected(self):
  cfg=json.loads((ROOT/'docs/lighton4b-projector-canonical-v1.json').read_text())
  for key,value in [('native_offset',0),('native_sha256','0'*64)]:
   c=copy.deepcopy(cfg);c['tensors'][0][key]=value
   with self.assertRaises(ValueError):m.admit(c)
 def test_source_provenance_mutations_rejected(self):
  cfg=json.loads((ROOT/'docs/lighton4b-projector-canonical-v1.json').read_text())
  for key in m.PINS:
   changed=copy.deepcopy(cfg);changed[key]=1 if isinstance(cfg[key],int) else 'mutated'
   with self.subTest(key=key):
    with self.assertRaises(ValueError):m.admit(changed)
 def test_temporal_order_and_precision(self):
  import numpy as np
  original=np.arange(2*3*2*2*2,dtype='<f4').reshape(2,3,2,2,2)
  joined=m.temporal_join([original[:,:,0].astype('<f2'),original[:,:,1].astype('<f2')]);self.assertTrue(np.array_equal(joined,original.reshape(-1)))
  a=np.array([1.,1.0001],dtype='<f4');b=a.astype('<f2').astype('<f4');r=m.compare(a,b);self.assertEqual(r['different_f32_bits'],1);self.assertEqual(r['different_after_native_f16_round'],0)
 def test_patch_spatial_temporal_layout(self):
  import numpy as np
  worker=load('projector_worker',Path(__file__).with_name('worker.py'))
  pixels=np.fromfunction(lambda y,x,c:10000*c+100*y+x,(64,64,3),dtype=int).astype('<f4')
  patches=worker.pixels_to_patches(pixels.reshape(-1),64,64).reshape(16,3,2,16,16)
  self.assertEqual(patches[:5,0,0,0,0].tolist(),[0,16,1600,1616,32])
  self.assertTrue(np.array_equal(patches[:,:,0],patches[:,:,1]))
 def test_nonfinite_rejected(self):
  import numpy as np
  with self.assertRaises(ValueError):m.compare(np.array([np.nan],dtype='<f4'),np.zeros(1,dtype='<f4'))
if __name__=='__main__':unittest.main()
