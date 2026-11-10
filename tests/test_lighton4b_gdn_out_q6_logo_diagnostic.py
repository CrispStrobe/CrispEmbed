"""Closed known-logo post-selection scope and exact retained Q6 policy."""
import subprocess
import ast,hashlib,importlib.util,json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('logo',ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_logo_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def _require_commit(case, commit):
    # Shallow clones (CI/Kaggle --depth 1) lack historical commits; only that
    # genuinely-absent case is skipped. Any present blob must still match exactly.
    if subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], cwd=ROOT, capture_output=True).returncode:
        case.skipTest('pinned commit ' + commit + ' absent from this (shallow) clone; hash pins need its history')

class Controls(unittest.TestCase):
 def test_only_anchor_and_named_logo_transfers(self):
  for case in('simple_table','handwritten_letter'):self.assertTrue(m.source_member(case,'capture.json').endswith(case+'/capture.json'))
  for case,name in[('arabic_printed_line','capture.json'),('receipt_example','capture.json'),('handwritten_letter','../batch.json')]:
   with self.assertRaises(ValueError):m.source_member(case,name)
 def test_fixed_candidate_and_diagnostic_label(self):
  p=json.loads((ROOT/'docs/lighton4b-gdn-out-q6-logo-protocol-v1.json').read_text());self.assertEqual(p['model']['sha256'],'83e065c472aa12d1c75be997c22d5038350540798f6ff7dbf6ca0240765c1c7a');self.assertEqual(p['diagnostic_cases'],['handwritten_letter']);self.assertFalse(p['clean_heldout_evaluation']);self.assertTrue(p['post_selection_diagnostic']);self.assertTrue(p['policy_selection_unchanged']);_require_commit(self,p['runtime_reference_commit']);self.assertTrue(all(hashlib.sha256(subprocess.check_output(['git','show',p['runtime_reference_commit']+':'+k],cwd=ROOT)).hexdigest()==v for k,v in p['runtime_paths_sha256'].items()))
 def test_exact_schedule_no_requantization(self):
  code=(ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_logo_diagnostic.py').read_text();ast.parse(code);self.assertIn("[('smoke','simple_table',1),('full','handwritten_letter',256)]",code);self.assertNotIn('conversion.convert(',code);self.assertNotIn('stage_originals(',code);self.assertNotIn('gh workflow run',code);self.assertIn('len(logo_protected)==36',code);self.assertIn("admitted.get('simple_table')is True",code);self.assertIn('signal.alarm(60*60)',code)
if __name__=='__main__':unittest.main()
