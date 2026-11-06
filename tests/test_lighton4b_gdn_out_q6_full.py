"""Bounded full-calibration consumer controls; no framework/model execution."""
import ast,hashlib,importlib.util,json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full',ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_full.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_unicode_and_reference_denominators(self):
  ref={'generated_ids':[27,5,248046],'decoded':'é a'};x=m.agreement([27,6,248046],'e a',ref);self.assertEqual(x['matching_prefix_ids'],1);self.assertEqual(x['cer'],{'edits':1,'reference_codepoints':3,'rate':1/3});self.assertEqual(x['wer']['rate'],.5);self.assertFalse(x['ids_exact']);self.assertTrue(m.agreement(ref['generated_ids'],ref['decoded'],ref)['decoded_exact'])
 def test_fixed_retained_candidate_and_runtime(self):
  p=json.loads((ROOT/'docs/lighton4b-gdn-out-q6-full-protocol-v1.json').read_text());self.assertEqual(p['cases'],['simple_table','arabic_printed_line']);self.assertEqual(p['heldout_cases'],[]);self.assertEqual(p['max_new_tokens'],256);self.assertEqual(p['model']['sha256'],'83e065c472aa12d1c75be997c22d5038350540798f6ff7dbf6ca0240765c1c7a');self.assertEqual(p['artifact']['id'],13503268909);self.assertTrue(all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in p['runtime_paths_sha256'].items()));self.assertEqual(p['fixed_precision_gates']['HEAD_BF16'],1)
 def test_no_reconversion_dispatch_or_generic_batch(self):
  code=(ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_full.py').read_text();tree=ast.parse(code);self.assertNotIn('gh workflow run',code);self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr=='stage_originals'for n in ast.walk(tree)));self.assertNotIn('conversion.convert(',code);self.assertIn("[('smoke',1),('full',256)]",code);self.assertIn("len(admitted)==2",code);self.assertIn("ids[-1]==248046",code);self.assertIn("int(values.argmax())==token_id",code)
if __name__=='__main__':unittest.main()
