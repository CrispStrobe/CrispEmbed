#!/usr/bin/env python3
"""Model-free admission/stop controls for the preserved decoded panels."""
import importlib.util,json,unittest
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('aligned_observed',HERE/'observed.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_only_one_gate_changes_and_both_working_operators_stay_on(self):
  before={'CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU':'1','CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE':'1'};off,on=m.gates(before);self.assertEqual({k for k in off if off[k]!=on[k]},{m.GATE});self.assertEqual(off[m.GATE],'0');self.assertEqual(on[m.GATE],'1')
 def test_eos_and_explicit_cap_both_admitted(self):
  m.stop_state(dict(generated_tokens=2,cap=512,upstream_physical=True,eos=True),[1,248046]);m.stop_state(dict(generated_tokens=512,cap=512,upstream_physical=True,eos=False),[1]*512)
 def test_earlier_eos_or_silent_truncation_rejected(self):
  for ids,summary in [([248046,1,248046],dict(generated_tokens=3,cap=512,upstream_physical=True,eos=True)),([1]*511,dict(generated_tokens=511,cap=512,upstream_physical=True,eos=False)),([1,248046],dict(generated_tokens=2,cap=1536,upstream_physical=True,eos=True)),([1,248046],dict(generated_tokens=2,cap=512,upstream_physical=False,eos=True))]:
   with self.assertRaises((RuntimeError,ValueError)):m.stop_state(summary,ids)
 def test_original_fixed_floors_and_receipt_first_order(self):
  p=json.loads((HERE/'fixed-protocol.json').read_text());self.assertEqual(p['case_order'],['historical-receipt','two-column-page','inventory-table']);self.assertEqual(p['arm_order'],[name+'/'+arm for name in p['case_order']for arm in ['off','on']]);self.assertTrue(p['continue_after_decoded_negative']);self.assertEqual([c['baseline_tokens']for c in p['cases']],[468,654,248]);self.assertEqual(p['cap'],1536);self.assertFalse(p['model_acquisition_allowed'])
 def test_candidate_and_control_bundle_frozen(self):
  p=m.admit(HERE/'observed-protocol.json');self.assertEqual(p['candidate_source'],'f8bffd1217fa41a958047d88dafb782a4b1cd989');self.assertEqual(p['candidate_controls_sha256'],'2d0d4131b12c6392bb92e494644e9f7d4900b17ba6c7389beaa96baae2a91305');self.assertFalse(p['runtime_rebuild_allowed']);self.assertFalse(p['source_reference_forward_allowed']);self.assertEqual(p['historical_off_tokens'],[282,170,6])
if __name__=='__main__':unittest.main()
