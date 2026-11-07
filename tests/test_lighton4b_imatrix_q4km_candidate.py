import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('candidate',ROOT/'tools/ci-heavy/lighton4b_imatrix_q4km_candidate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_predeclared_budget_and_source(self):
  p=json.loads((ROOT/'docs/lighton4b-imatrix-q4km-candidate-protocol-v1.json').read_text());self.assertEqual(p['source']['sha256'],m.profile.SOURCE_F16_SHA);self.assertEqual(p['recipe']['tag'],m.profile.RECIPE);self.assertEqual(p['recipe']['types'],{'F32':474,'Q4_K':216,'Q6_K':33});self.assertEqual(p['calibration_only']['cases'],['simple_table','arabic_printed_line']);self.assertFalse(p['execution']['full_generation_in_this_worker']);self.assertGreaterEqual(p['limits']['decoded_followup']['table_child_seconds'],75*60)
 def test_wrong_or_duplicate_capture_rejected(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);folder=root/'one'/'simple_table';folder.mkdir(parents=True);path=folder/'capture.json';path.write_bytes(b'authentic')
   self.assertEqual(m.unique_capture(root,'simple_table',m.sha(path)),path)
   with self.assertRaises(ValueError):m.unique_capture(root,'simple_table','0'*64)
   other=root/'two'/'simple_table';other.mkdir(parents=True);(other/'capture.json').write_bytes(b'authentic')
   with self.assertRaises(ValueError):m.unique_capture(root,'simple_table',m.sha(path))
 def test_non_model_cannot_be_admitted(self):
  with tempfile.TemporaryDirectory()as directory:
   path=Path(directory)/'bad.gguf';path.write_bytes(b'GGUF'+b'\0'*60)
   with self.assertRaises(ValueError):m.audit_q4km(path,{})
if __name__=='__main__':unittest.main()
