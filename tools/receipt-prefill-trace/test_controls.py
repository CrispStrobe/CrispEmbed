#!/usr/bin/env python3
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def load(name):
 s=importlib.util.spec_from_file_location('receipt_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
worker=load('worker');source=load('source_worker');patch=load('source_patch')
class Controls(unittest.TestCase):
 def test_scope_and_geometry(self):
  c=json.loads((HERE/'protocol.json').read_text());self.assertFalse(any(c[k]for k in ['full_decode_allowed','native_model_forward_allowed','full_f32_model_allowed']));self.assertEqual(c['source_prompt_rows'],[4,242,6,4]);self.assertEqual(sum(c['source_prompt_rows']),256);self.assertEqual(c['source_generated_tokens'],465);self.assertEqual(c['fixed_receipt_sha256'],'8001a29fab5abc3481e601168e10ca4383d10060ad7e4356a9110eb345c0b56a')
 def test_first_sample_frozen_neutrality(self):
  frozen={'tokens':[1,2],'completion_probabilities':[{'x':1},{'x':2}]};r=dict(tokens=[1],completion_probabilities=[{'x':1}],stop_type='limit',tokens_evaluated=256,truncated=False);source.validate_response(r,frozen)
  for key,value in [('tokens',[2]),('tokens_evaluated',266),('stop_type','eos')]:
   bad=dict(r);bad[key]=value
   with self.assertRaises(RuntimeError):source.validate_response(bad,frozen)
 def test_observer_names_and_bounded_prefill(self):
  self.assertEqual(len(patch.STAGES),len(set(patch.STAGES)));m=patch.configured();self.assertIn('(n==0)',m.HELPER);self.assertNotIn('n==31',m.HELPER);self.assertIn('512u<<20',m.HELPER)
  for layer in range(3):self.assertIn('linear_attn_qkv_mixed-'+str(layer),patch.STAGES)
  anchor='    cb(Qcur_full, "Qcur_full", il);\n    cb(Kcur, "Kcur", il);\n    cb(Vcur, "Vcur", il);';original=anchor+'\ncb(Kcur, "Kcur", il);cb(Vcur, "Vcur", il);';actual=patch.disambiguate(original);self.assertEqual(actual.replace('"Kcur_raw"','"Kcur"').replace('"Vcur_raw"','"Vcur"'),original)
 def test_exact_and_nonexact_magnitudes(self):
  x=np.array([[1.,2.]],dtype='<f4');same=worker.comparison(x,x.copy());self.assertTrue(same['bit_exact']);self.assertIsNone(same['first_different_flat_index']);y=x.copy();y[0,1]=3.;r=worker.comparison(x,y);self.assertEqual(r['first_different_flat_index'],1);self.assertEqual(r['error_max_abs'],1.)
if __name__=='__main__':unittest.main()
