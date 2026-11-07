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
 def test_component_blocked_until_measured_controls(self):
  text=Path(__file__).with_name('component.py').read_text()
  self.assertIn("CONTROL_RECEIPT='BLOCKED_UNTIL_HOSTED_CONTROLS_PASS'",text)
  self.assertIn("len(CONTROL_RECEIPT)==64 and len(CONTROL_SOURCE)==40",text)
  self.assertLess(text.index("report['legacy_anchor_exact']=True"),text.index("report['all_source_stages_byte_exact']="))
  self.assertIn("report['downloaded_weight_bytes']==1334056960",text)
 def test_explicit_model_free(self):
  text=Path(__file__).with_name('controls.py').read_text();self.assertIn('model_acquisition=False',text);self.assertNotIn('range_download(',text);self.assertNotIn('urlopen(',text)
 def test_reference_not_production(self):
  cm=(ROOT/'CMakeLists.txt').read_text();self.assertIn('lighton4b-vision-profile-probe EXCLUDE_FROM_ALL',cm);self.assertNotIn('resize_source.cpp',cm)
 def test_profile_owner_and_main_closure(self):
  text=(ROOT/'src/lighton4b_physical_f16_vision.cpp').read_text();self.assertIn('!lighton4b_physical_f16_vision_valid(physical)',text)
  cm=(ROOT/'CMakeLists.txt').read_text();self.assertEqual(cm.count('src/lighton4b_physical_f16_vision.cpp'),4);self.assertEqual(cm.count('src/lighton4b_pillow_resize.cpp'),4) # three sources plus compile option
if __name__=='__main__':unittest.main()
