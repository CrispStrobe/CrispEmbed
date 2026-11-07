import importlib.util,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
resize=load('resize_patch_test',Path(__file__).with_name('resize_patch.py'))
class Controls(unittest.TestCase):
 def test_pinned_reference_bridge(self):
  source=Path('/tmp/lighton4b-projector-audit/mtmd-image.cpp')
  if not source.exists():self.skipTest('optional local primary-source cache')
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'tools/mtmd/mtmd-image.cpp';path.parent.mkdir(parents=True);path.write_bytes(source.read_bytes())
   result=resize.patch(d);self.assertNotEqual(result['before_sha256'],result['after_sha256']);self.assertIn('img_tool::resize(src,dst,{ow,oh},RESIZE_ALGO_BICUBIC,PAD_NONE)',path.read_text())
   with self.assertRaises(ValueError):resize.patch(d)
 def test_component_requires_measured_controls(self):
  import sys,copy
  sys.path.insert(0,str(Path(__file__).parent))
  module=load('vision_component_test',Path(__file__).with_name('component.py'))
  baseline=dict(status='passed_model_free_vision_controls',source_sha=module.CONTROL_SOURCE,model_acquisition=False,checkpoint_vision_forward_executed=False,actual_form_preprocessing_exact=True,synthetic_vision_component_executed=True)
  module.admit_control(baseline)
  for key,value in [('status','failed'),('source_sha','0'*40),('model_acquisition',True),('checkpoint_vision_forward_executed',True),('actual_form_preprocessing_exact',False),('synthetic_vision_component_executed',False)]:
   changed=copy.deepcopy(baseline);changed[key]=value
   with self.assertRaises(ValueError):module.admit_control(changed)
  self.assertEqual(len(module.CONTROL_SOURCE),40);self.assertEqual(len(module.CONTROL_RECEIPT),64)
  text=Path(__file__).with_name('component.py').read_text()
  self.assertIn("report['downloaded_weight_bytes']==1334056960",text)
 def test_canonical_uses_declared_precision(self):
  import sys,json,copy
  sys.path.insert(0,str(Path(__file__).parent));module=load('component_canonical_test',Path(__file__).with_name('component.py'))
  cfg=json.loads((ROOT/'docs/lighton4b-projector-canonical-v1.json').read_text());rows={r['native_name']:dict(native_sha256=r['native_sha256'],elements=r['native_bytes']//4,bit_exact=True,different_after_native_f16_round=0,zero_sign_only_after_f16_round=0)for r in cfg['tensors']};source={'canonical':{'tensors':rows}}
  module.admit_canonical(cfg,source)
  f32=next(r['native_name']for r in cfg['tensors']if r['upstream'][0]['type']==0);half=next(r['native_name']for r in cfg['tensors']if r['upstream'][0]['type']==1)
  rows[f32]['different_after_native_f16_round']=1;module.admit_canonical(cfg,source) # This cast must never be applied.
  for key,value,name in [('bit_exact',False,f32),('different_after_native_f16_round',1,half),('native_sha256','0'*64,half)]:
   bad=copy.deepcopy(source);bad['canonical']['tensors'][name][key]=value
   with self.assertRaises(ValueError):module.admit_canonical(cfg,bad)
 def test_explicit_model_free(self):
  text=Path(__file__).with_name('controls.py').read_text();self.assertIn('model_acquisition=False',text);self.assertNotIn('range_download(',text);self.assertNotIn('urlopen(',text)
 def test_reference_not_production(self):
  cm=(ROOT/'CMakeLists.txt').read_text();self.assertIn('lighton4b-vision-profile-probe EXCLUDE_FROM_ALL',cm);self.assertNotIn('resize_source.cpp',cm)
 def test_profile_owner_and_main_closure(self):
  text=(ROOT/'src/lighton4b_physical_f16_vision.cpp').read_text();self.assertIn('!lighton4b_physical_f16_vision_valid(physical)',text)
  cm=(ROOT/'CMakeLists.txt').read_text();self.assertEqual(cm.count('src/lighton4b_physical_f16_vision.cpp'),4);self.assertEqual(cm.count('src/lighton4b_pillow_resize.cpp'),4) # three sources plus compile option
if __name__=='__main__':unittest.main()
