#!/usr/bin/env python3
"""Model-free controls for layout and uneven preferred projection input geometry."""
import importlib.util,json,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('receipt_layer0',HERE/'worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class Controls(unittest.TestCase):
 def test_actual_geometry_and_scope(self):
  cfg=json.loads((HERE/'protocol.json').read_text());self.assertEqual(cfg['source_prompt_rows'],[4,242,6,4]);self.assertFalse(any(cfg[k]for k in ['model_forward_allowed','source_capture_allowed','full_f32_model_allowed','runtime_changes_allowed']));self.assertEqual(cfg['projection_tokens'],[8,10,16,242])
 def test_physical_source_head_mapping(self):
  head=[0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23,8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31]
  self.assertEqual(w.permutation('a',32),head);self.assertEqual(w.permutation('qkv',8192)[:4096],list(range(4096)))
  for role,width in [('qkv',8192),('z',4096),('postgdn',4096),('a',32),('b',32)]:
   order=w.permutation(role,width);self.assertEqual(sorted(order),list(range(width)));source=np.arange(width,dtype='<f4');native=source[order];recovered=np.empty_like(native);recovered[order]=native;self.assertEqual(recovered.tobytes(),source.tobytes())
 def test_localizes_actual_tail_rows_without_values(self):
  x=np.zeros((10,32),dtype='<f4');y=x.copy();y[8,1]=1.;y[9,2]=-2.;r=w.compare(x,y);self.assertEqual(r['different_rows'],[8,9]);self.assertEqual(r['first_different_flat_index'],257);self.assertEqual(r['error_max_abs'],2.)
 def test_runtime_entry_has_full_capture_closure(self):
  source=(HERE/'component.cpp').read_text();self.assertIn('ALL256 actual native hidden0 byte closure',source);self.assertIn('duplicate GDN actual block/cache exact',source);self.assertIn('cache_owner.bytes()==0',source);self.assertNotIn('load_model',source)
if __name__=='__main__':unittest.main()
