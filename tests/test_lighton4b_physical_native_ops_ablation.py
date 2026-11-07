import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('physical_ablation',ROOT/'tools/kaggle/lighton4b_physical_native_ops_ablation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Admission(unittest.TestCase):
 def setUp(self):self.cfg=json.loads((ROOT/'docs/lighton4b-physical-native-ops-ablation-v1.json').read_text())
 def test_frozen_six_prefix_arms(self):module.admit_protocol(self.cfg)
 def test_wrong_scope_or_golden_source_rejects(self):
  changes=[lambda c:c.update(cap=256),lambda c:c.update(arms=['quant','rms']),lambda c:c.update(arms=list(reversed(c['arms']))),lambda c:c['cases'][0].update(id='simple_table'),lambda c:c.update(golden_smoke_report_sha256='0'*64),lambda c:c.update(golden_smoke_source_sha='latest'),lambda c:c.update(expected_candidate_sha256='0'*64),lambda c:c.update(threads=8),lambda c:c.update(native_arm_timeout_seconds=5400)]
  for change in changes:
   with self.subTest(change=repr(change)):
    cfg=copy.deepcopy(self.cfg);change(cfg)
    with self.assertRaises(RuntimeError):module.admit_protocol(cfg)
 def test_both_endpoint_anchors_require_all38_hashes(self):
  actual={rel:format(i,'064x')for i,rel in enumerate(module.STAGES)};gold={rel:{'sha256':value}for rel,value in actual.items()}
  for arm in ('off','all'):
   module.check_anchor(arm,actual,gold)
   changed=dict(actual);changed['greedy.0.logits.f32']='f'*64
   with self.assertRaises(RuntimeError):module.check_anchor(arm,changed,gold)
   changed=dict(actual);changed.pop('prefill.layer.31.f32')
   with self.assertRaises(RuntimeError):module.check_anchor(arm,changed,gold)
