"""Distinct-profile admission never changes frozen mixed/Q4 gates."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_bf16_plain_q4_amd_diagnostic as m
# Tools set a shared base.PROTOCOL at import; capture this tool's own path
# so later-collected sibling tools in one pytest session cannot redirect it.
OWN_PROTOCOL=m.base.PROTOCOL
class Controls(unittest.TestCase):
 def test_distinct_explicit_profile(self):
  p=json.loads(OWN_PROTOCOL.read_text());self.assertTrue(p['distinct_backend_diagnostic']);self.assertFalse(p['generation_or_quality_qualified'])
  self.assertNotEqual(p['original_first_score_sha256'],p['frozen_official_first_score_sha256']);self.assertEqual(p['candidate_order'],['q4_k'])
 def test_frozen_guard_rejects_changed_q4(self):
  o=object.__new__(m.plain.full.FullOracle);o.p={'mixed_first_score_sha256':'old'}
  with self.assertRaises(ValueError):o.check_first_score({'sha256':'new'})
  o.check_first_score({'sha256':'old'})
 def test_diagnostic_preserves_negative(self):
  o=object.__new__(m.AMDOracle);o.p={'mixed_first_score_sha256':'old'};o.report={};o.save=lambda:None
  o.check_first_score({'sha256':'new'});r=o.report['first_q4_vs_frozen_profile'];self.assertFalse(r['bit_exact']);self.assertEqual(r['frozen_sha256'],'old');self.assertEqual(r['actual_sha256'],'new')
 def test_both_attempts_repeated_exact(self):
  root=ROOT/'tests/results/qwen35-ocr'
  a=json.loads((root/'lightonocr3-4b-bf16-plain-q4-source-initial/oracle.json').read_text());b=json.loads((root/'lightonocr3-4b-bf16-plain-q4-source-avx2-retry/oracle.json').read_text())
  self.assertEqual(a['arms']['original']['sha256'],b['arms']['original']['sha256']);self.assertEqual(a['original_bf16_tensor_sha256'],b['original_bf16_tensor_sha256']);self.assertNotIn('q4_k-weights',b)
if __name__=='__main__':unittest.main()
