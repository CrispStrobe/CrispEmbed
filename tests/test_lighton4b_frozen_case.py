#!/usr/bin/env python3
"""Fail-closed fixed-case contract controls; no model data or forwards."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class FixedCaseControls(unittest.TestCase):
 def test_exact_cases_and_pngs(self):
  for name in ('simple_table','arabic_printed_line','receipt_example','handwritten_letter'):
   protocol,case,frozen=module.load_contract(name)
   self.assertEqual(case['expected_grid'],frozen['expected_grid'])
   self.assertEqual(case['expected_input_tokens'],frozen['expected_input_tokens'])
   self.assertEqual(module.sha(ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file']),frozen['png_sha256'])
   self.assertEqual(protocol['candidate_recipe'],module.MIXED)
 def test_outside_case_rejected(self):
  for name in ('simple_form','../simple_table','new_case',''):
   with self.assertRaises(ValueError):module.load_contract(name)
 def test_manifest_digest_required(self):
  original=module.FROZEN_SHA
  try:
   module.FROZEN_SHA='0'*64
   with self.assertRaises(ValueError):module.load_contract('simple_table')
  finally:module.FROZEN_SHA=original
if __name__=='__main__':unittest.main()
