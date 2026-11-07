import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('followup',ROOT/'tools/kaggle/lighton4b_composed_form_vision.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_frozen_protocol(self):m.admit(json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text()))
 def test_mutations(self):
  cfg=json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text())
  for key,val in [('case','simple_table'),('arms',['fixed','old']),('component_receipt_sha256',None),('component_receipt_sha256','0'*64),('component_source_sha','0'*40),('cap',256),('threads',2),('worker_seconds',0),('changed_gates',[]),('candidate_sha256','0'*64),('source_feature_sha256','0'*64),('driver_sha256','0'*64),('old_ids',1),('original_reference_ids',1),('quality_qualified',True)]:
   with self.subTest(key=key):
    bad=copy.deepcopy(cfg);bad[key]=val
    with self.assertRaises(RuntimeError):m.admit(bad)
 def test_only_two_vision_flags(self):
  old,fixed=m.gate_pair({'CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION':'1','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK':'1','CRISPEMBED_LIGHTON4B_HEAD_BF16':'1','sentinel':'unchanged'})
  self.assertEqual({x for x in old if old[x]!=fixed[x]},set(m.GATES));self.assertEqual(old['CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK'],'0');self.assertEqual(fixed['CRISPEMBED_LIGHTON4B_HEAD_BF16'],'0');self.assertEqual(fixed['sentinel'],'unchanged')
 def test_component_all28_required(self):
  proof=dict(status='measured_same_input_projector_component',source_sha=m.PRODUCTION,legacy_anchor_exact=True,metadata_controls_before_weights=True,all_source_stages_byte_exact=True,first_differing_stage=None,source_metrics={x:dict(bit_exact=True,candidate_sha256='a',reference_sha256='a')for x in m.VISION})
  m.admit_component(proof)
  for mutate in ['missing','different','legacy','partial']:
   bad=copy.deepcopy(proof)
   if mutate=='missing':bad['source_metrics'].pop(m.VISION[0])
   if mutate=='different':bad['source_metrics'][m.VISION[0]]['reference_sha256']='b'
   if mutate=='legacy':bad['legacy_anchor_exact']=False
   if mutate=='partial':bad['all_source_stages_byte_exact']=False
   with self.assertRaises(RuntimeError):m.admit_component(bad)
if __name__=='__main__':unittest.main()
