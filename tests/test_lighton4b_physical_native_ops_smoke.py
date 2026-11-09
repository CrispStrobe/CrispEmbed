import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('physical_smoke',ROOT/'tools/kaggle/lighton4b_physical_native_ops_smoke.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Admission(unittest.TestCase):
 def setUp(self):self.cfg=json.loads((ROOT/'docs/lighton4b-physical-native-ops-smoke-v1.json').read_text())
 def test_frozen_three_arm_smoke(self):module.admit_protocol(self.cfg)
 def test_wrong_case_caps_and_sources_reject_before_any_transfer(self):
  mutations=[lambda c:c.update(cap=256),lambda c:c.update(arms=['off','on']),lambda c:c.update(arms=['reference','on','off']),lambda c:c['cases'][0].update(id='simple_table'),lambda c:c.update(actual_baseline_source_sha='latest'),lambda c:c.update(upstream_commit='main'),lambda c:c.update(expected_candidate_sha256='0'*64),lambda c:c['models']['Q4_K_M'].update(sha256='0'*64),lambda c:c['models']['Q4_K_M'].update(bytes=1),lambda c:c.update(threads=8),lambda c:c.update(image_max_tokens=512),lambda c:c.update(native_arm_timeout_seconds=5400),lambda c:c.update(total_model_timeout_seconds=21600)]
  for change in mutations:
   with self.subTest(change=repr(change)):
    cfg=copy.deepcopy(self.cfg);change(cfg)
    with self.assertRaises(RuntimeError):module.admit_protocol(cfg)
 def test_off_closes_reference_bytes_before_on(self):
  text=(ROOT/'tools/kaggle/lighton4b_physical_native_ops_smoke.py').read_text()
  self.assertIn("if arm in ['reference','off']",text)
  self.assertIn("close_stage_hashes(arm,current,baseline_files)",text)
  self.assertIn("args=[oldnative if arm=='reference' else native",text)
  self.assertIn("armenv['CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS']='1' if arm=='on' else '0'",text)
 def test_separate_reference_off_callbacks_share_all38_stages(self):
  reference={};actual={rel:format(i,'064x')for i,rel in enumerate(module.BASELINE_STAGE_FILES)}
  def callback(arm,current):
   def early_check():module.close_stage_hashes(arm,current,reference)
   return early_check
  callback('reference',actual)();self.assertEqual(len(reference),38)
  callback('off',dict(actual))();self.assertEqual(reference,actual)
  changed=dict(actual);changed['greedy.0.logits.f32']='f'*64
  with self.assertRaises(RuntimeError):callback('off',changed)()
  missing=dict(actual);missing.pop('prefill.layer.31.f32')
  with self.assertRaises(RuntimeError):callback('off',missing)()
if __name__=='__main__':unittest.main()
