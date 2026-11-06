"""Closed calibration transfer and first-score-only worker controls; no model."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('q6_candidate',ROOT/'tools/ci-heavy/lighton4b_q6_candidate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_only_calibration_members(self):
  for case in ('simple_table','arabic_printed_line'):self.assertTrue(m.calibration_member(case,'capture.json').endswith(case+'/capture.json'))
  for case,name in [('receipt_example','capture.json'),('logo','capture.json'),('simple_table','../batch.json'),('simple_table','a/b')]:
   with self.assertRaises(ValueError):m.calibration_member(case,name)
 def test_no_batch_or_generic_stage_originals(self):
  code=(ROOT/'tools/ci-heavy/lighton4b_q6_candidate.py').read_text();tree=ast.parse(code)
  self.assertNotIn("'batch.json'",code);self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr=='stage_originals'for n in ast.walk(tree)))
 def test_model_job_not_dispatched_by_worker(self):
  code=(ROOT/'tools/ci-heavy/lighton4b_q6_candidate.py').read_text();self.assertNotIn('gh workflow run',code);self.assertIn("teacher,native,1,4,'smoke'",code)
if __name__=='__main__':unittest.main()
