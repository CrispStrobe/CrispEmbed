#!/usr/bin/env python3
"""Small admission controls; no model or inference."""
import hashlib,importlib.util,os,tempfile,unittest
from pathlib import Path
import source_patch,analysis
class Controls(unittest.TestCase):
 def test_worker_imports(self):
  spec=importlib.util.spec_from_file_location("text_trace_worker",Path(__file__).parent/"worker.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  self.assertEqual(module.TABLE,"36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3")
 def test_distinct_png_containers_same_rgb(self):
  import worker
  from PIL import Image
  from unittest.mock import patch
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);im=Image.new('RGB',(8,8),(31,127,253));im.save(root/'a.png',compress_level=0);im.save(root/'b.png',compress_level=9);(root/'rgb').write_bytes(im.tobytes())
   self.assertNotEqual((root/'a.png').read_bytes(),(root/'b.png').read_bytes())
   with patch.object(Image,'__version__','11.3.0'):
    worker.validate_png_inputs(root/'a.png',root/'b.png',root/'rgb')
    (root/'rgb').write_bytes(bytes(8*8*3))
    with self.assertRaisesRegex(RuntimeError,'RGB pixels'):worker.validate_png_inputs(root/'a.png',root/'b.png',root/'rgb')
 def test_phase_closed(self):
  self.assertIn('(n==0||n==126)',source_patch.HELPER)
  self.assertIn('ggml_backend_tensor_get(t,v.data(),0,bytes)',source_patch.HELPER)
  self.assertNotIn('GGML_TENSOR_FLAG_OUTPUT',source_patch.HELPER)
  self.assertEqual(len(source_patch.STAGES),len(set(source_patch.STAGES)))
 def test_permutations(self):
  for role,n in [('qkv',8192),('z',4096),('a',32),('b',32)]:
   rows=analysis.row_map(role,n);self.assertEqual(sorted(rows),list(range(n)))
  self.assertEqual(analysis.row_map('qkv',8192)[:4096],list(range(4096)))
  self.assertEqual(analysis.row_map('a',32)[:4],[0,16,1,17])
 def test_source_drift_rejected_before_write(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'src').mkdir();(root/'common').mkdir();p=root/'src/llama-context.cpp';p.write_text('wrong');(root/'common/sampling.cpp').write_text('wrong')
   with self.assertRaisesRegex(ValueError,'source drift'):source_patch.patch(root)
   self.assertEqual(p.read_text(),'wrong')
 def test_actual_primary_patch(self):
  primary=os.environ.get('TEXT_TRACE_PRIMARY_DIR')
  if not primary:self.skipTest('Optional pinned primary source directory absent')
  root_tools=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('observer',root_tools/'lighton4b_upstream_observer_patch.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
  with tempfile.TemporaryDirectory()as d:
   root=Path(d)
   for name in ['src/llama-context.cpp','common/sampling.cpp','tools/mtmd/mtmd-helper.cpp']:
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(primary)/('lighton4b-text-primary-'+p.name)).read_bytes())
   observer.patch(root);proof=source_patch.patch(root)
   self.assertEqual(len(proof),2);text=(root/'common/sampling.cpp').read_text();self.assertIn('f<<samples<<"\\n";',text)
   context=(root/'src/llama-context.cpp').read_text();self.assertLess(context.index('crisp_text_trace_begin(gf, sched.get())'),context.index('auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);'))
   self.assertIn('ggml_backend_sched_synchronize(sched.get());\n        crisp_text_trace_end()',context)
   with self.assertRaises(ValueError):source_patch.patch(root)
if __name__=='__main__':unittest.main()
