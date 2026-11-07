import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('calibration',ROOT/'tools/kaggle/lighton4b_composed_vision_calibration.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Admission(unittest.TestCase):
 def test_frozen(self):m.admit(json.loads((ROOT/'docs/lighton4b-composed-vision-calibration-v1.json').read_text()))
 def test_mutations(self):
  cfg=json.loads((ROOT/'docs/lighton4b-composed-vision-calibration-v1.json').read_text())
  for key,value in [('cases',list(reversed(m.CASES))),('arms',list(reversed(m.ARMS))),('cap',256),('threads',2),('worker_seconds',0),('arm_seconds',-1),('form_receipt_sha256','0'*64),('form_source_sha','0'*40),('extended_table_receipt_sha256','0'*64),('physical_receipt_sha256','0'*64),('calibration_receipt_sha256','0'*64),('frozen_source_feature_ids',{}),('candidate_sha256','0'*64),('quality_qualified',True)]:
   with self.subTest(key=key):
    c=copy.deepcopy(cfg);c[key]=value
    with self.assertRaises(RuntimeError):m.admit(c)
 def test_cpu_profile_closed(self):
  m.cpu_admit('Linux','x86_64',['fma','avx2','f16c'])
  for args in [('Darwin','x86_64',['fma','avx2','f16c']),('Linux','aarch64',['fma','avx2','f16c']),('Linux','x86_64',['fma','avx2'])]:
   with self.assertRaises(RuntimeError):m.cpu_admit(*args)
 def test_actual_cli_shapes(self):
  for arm,argc in [('source-features',13),('composed',11)]:
   a=m.args('native','candidate','rgb',317,452,'teacher','dest',arm,'feature');self.assertEqual(len(a),argc);self.assertEqual(a[7:10],[512,4,'greedy-full']);self.assertEqual(a[-1],'--observed-composed-cap512');self.assertEqual('--upstream-vision'in a,arm=='source-features')
  with self.assertRaises(RuntimeError):m.args('n','c','r',1,1,'t','d','other','f')
 def test_actual_form_admission(self):
  gatekeys1=['CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION'];gatekeys0=['CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_HEAD_BF16','CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS'];gates={k:'1'for k in gatekeys1}|{k:'0'for k in gatekeys0}
  form=dict(status='measured_composed_form_vision',source_sha=m.SOURCE,all_component_owned_members_rehashed=73,arms={k:dict(expected_text_and_rows_exact=True,all_vision_processor_anchors_exact=True,summary=dict(eos=True,source_fed_vision=False),generated_ids=list(range(n)),gates=gates.copy())for k,n in [('old',194),('fixed',167)]})
  m.admit_form(form)
  for key in gatekeys1+gatekeys0:
   bad=copy.deepcopy(form);bad['arms']['fixed']['gates'][key]='0'if key in gatekeys1 else'1'
   with self.assertRaises(RuntimeError):m.admit_form(bad)
if __name__=='__main__':unittest.main()
