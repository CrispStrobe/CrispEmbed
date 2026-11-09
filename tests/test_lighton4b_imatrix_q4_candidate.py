#!/usr/bin/env python3
"""Small offline admission controls; no checkpoint, quantization or inference."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate',ROOT/'tools/ci-heavy/lighton4b_imatrix_q4_candidate.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
class Controls(unittest.TestCase):
 def test_immutable_capture_admission(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);case=root/'mount'/'simple_table';case.mkdir(parents=True);path=case/'capture.json';path.write_bytes(b'authenticated capture')
   self.assertEqual(candidate.unique_capture(root,'simple_table',candidate.sha(path)),path)
   with self.assertRaises(ValueError):candidate.unique_capture(root,'simple_table','0'*64)
   other=root/'other'/'simple_table';other.mkdir(parents=True);(other/'capture.json').write_bytes(path.read_bytes())
   with self.assertRaises(ValueError):candidate.unique_capture(root,'simple_table',candidate.sha(path))
 def test_missing_capture_rejected(self):
  with tempfile.TemporaryDirectory()as directory:
   with self.assertRaises(ValueError):candidate.unique_capture(Path(directory),'arabic_printed_line','0'*64)
 def test_publisher_payload_cannot_be_native_candidate(self):
  with tempfile.TemporaryDirectory()as directory:
   path=Path(directory)/'not-a-model.gguf';path.write_bytes(b'GGUF'+b'\0'*60)
   with self.assertRaises(ValueError):candidate.audit_candidate(path,None)
 def test_protocol_separation_and_resource_floor(self):
  p=json.loads((ROOT/'docs/lighton4b-imatrix-q4-calibration-protocol-v1.json').read_text())
  self.assertEqual(p['calibration_only']['cases'],['simple_table','arabic_printed_line'])
  self.assertEqual(p['recipe']['counts'],{'F32':474,'Q4_K':248,'Q8_0':1})
  self.assertEqual(p['statistics']['applied_count_required'],248)
  self.assertEqual(p['source']['sha256'],'757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606')
  self.assertGreaterEqual(p['execution']['minimum_free_disk_bytes'],18*2**30)
  self.assertFalse(p['statistics']['corpus_license_authenticated'])
  self.assertFalse(p['execution']['github_dispatch_allowed'])
if __name__=='__main__':unittest.main()
