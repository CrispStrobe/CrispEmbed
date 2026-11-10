"""Closed hardware selection cannot choose a profile from observed scores."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_bf16_plain_q4_dual_profile as m
# Tools set a shared base.PROTOCOL at import; capture this tool's own path
# so later-collected sibling tools in one pytest session cannot redirect it.
OWN_PROTOCOL=m.base.PROTOCOL
class Controls(unittest.TestCase):
 def setUp(self):self.p=json.loads(OWN_PROTOCOL.read_text())
 def test_intel_exact_profile(self):
  key,p=m.select_profile(self.p,{'vendor_id':'GenuineIntel','cpu family':'6','model':'79'});self.assertTrue(p['require_frozen_q4_anchor']);self.assertTrue(p['original_first_score_sha256'].startswith('a4bece'))
 def test_amd_distinct_profile(self):
  key,p=m.select_profile(self.p,{'vendor_id':'AuthenticAMD','cpu family':'23','model':'49'});self.assertFalse(p['require_frozen_q4_anchor']);self.assertTrue(p['original_first_score_sha256'].startswith('02ca'))
 def test_unknown_hardware_rejected(self):
  for d in ({},{'vendor_id':'GenuineIntel','cpu family':'6','model':'85'},{'vendor_id':'AuthenticAMD','cpu family':'6','model':'79'}):
   with self.assertRaises(ValueError):m.select_profile(self.p,d)
 def test_three_negative_receipts_remain(self):
  r=ROOT/'tests/results/qwen35-ocr'
  a=json.loads((r/'lightonocr3-4b-bf16-plain-q4-source-avx2-retry/oracle.json').read_text());b=json.loads((r/'lightonocr3-4b-bf16-plain-q4-amd-profile-placement/oracle.json').read_text())
  self.assertNotEqual(a['arms']['original']['sha256'],b['arms']['original']['sha256']);self.assertEqual(a['original_bf16_tensor_sha256'],b['original_bf16_tensor_sha256']);self.assertNotIn('q4_k-weights',b)
if __name__=='__main__':unittest.main()
