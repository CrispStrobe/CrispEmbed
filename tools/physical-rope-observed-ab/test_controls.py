#!/usr/bin/env python3
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('observed',HERE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Controls(unittest.TestCase):
 def test_frozen_observed_protocol(self):
  cfg=worker.admit(HERE/'protocol.json');self.assertEqual(cfg['historical_off_tokens'],[218,167,6]);self.assertEqual(cfg['cases'],['simple_table','simple_form','arabic_printed_line']);self.assertEqual((cfg['cap'],cfg['threads'],cfg['arm_seconds']),(512,4,900))
  self.assertFalse(any(cfg[k]for k in ['model_f32_text_allowed','heldout_data_allowed','source_reference_forward_allowed','runtime_rebuild_allowed','quality_qualified']))
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'protocol.json';p.write_text((HERE/'protocol.json').read_text()+' ')
   with self.assertRaises(RuntimeError):worker.admit(p)
 def test_only_rope_changes_fused_stays_one(self):
  before={'CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU':'1','other':'unchanged'};off,on=worker.gates(before);self.assertEqual({k for k in off if off[k]!=on[k]},{worker.GATE});self.assertEqual(off['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU'],'1');self.assertNotIn(worker.GATE,before)
  with self.assertRaises(RuntimeError):worker.gates({'CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU':'0'})
 def test_no_model_admission_before_supported_source_component(self):
  candidate='a8a22749ab0b757ae1f317b6fa5b84ffd148b939';units=['test-lighton4b-physical-rope','test-lighton4b-source-prefill-chunks','test-lighton4b-physical-swiglu','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-native-ops','test-lighton4b-physical-f16-attention','physical-rope-unsupported','physical-rope-wasm-guard'];good=dict(status='measured_source_fed_rope_controls',passed=True,candidate_source=candidate,checkpoint_loaded=False,all_source_stages_exact=True,actual_cache_toggle_recovery_pass=True,unit_controls={k:'PASS'for k in units});worker.admit_controls(good,candidate)
  for key in ['passed','all_source_stages_exact','actual_cache_toggle_recovery_pass']:
   bad=copy.deepcopy(good);bad[key]=False
   with self.assertRaises(RuntimeError):worker.admit_controls(bad,candidate)
  bad=copy.deepcopy(good);bad['unit_controls'][units[0]]='SKIP PASS'
  with self.assertRaises(RuntimeError):worker.admit_controls(bad,candidate)
  bad=copy.deepcopy(good);bad['candidate_source']='wrong'
  with self.assertRaises(RuntimeError):worker.admit_controls(bad,candidate)

if __name__=='__main__':unittest.main()
