#!/usr/bin/env python3
"""Offline exact published type schedule and closed fresh-source recipe controls."""
import copy,hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_imatrix_q4km_recipe as m
from lighton4b_q6_recipe import plan as old_plan
class Controls(unittest.TestCase):
 def test_publisher_actual33_named_types(self):
  raw=json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/publisher-q4km-type-census.json').read_text());self.assertTrue(m.validate_publisher_header(raw));self.assertEqual(raw['declared_source_name'].lower(),'b71010f095dc3735de7ee6b9969bff8f8f50b39a')
  for name in m.plan()['upstream_q6_names']:
   wrong=copy.deepcopy(raw);wrong['tensors'][name]['type']=12
   with self.assertRaises(ValueError):m.validate_publisher_header(wrong)
 def test_old32_q8head_unchanged(self):
  old=old_plan();new=m.plan();self.assertEqual(old['types'],{'0':474,'8':1,'12':216,'14':32});self.assertEqual(new['types'],{'0':474,'12':216,'14':33});self.assertEqual(set(new['q6_names'])-set(old['q6_names']),{'l.embed_tokens.weight'});self.assertEqual(new['q6_names_sha256'],hashlib.sha256(('\n'.join(new['q6_names'])+'\n').encode()).hexdigest());self.assertFalse(new['publisher_byte_equivalence']);self.assertFalse(new['quality_qualified'])
 def test_closed_cli_no_override_or_requantization(self):
  args=m.quantizer_arguments('original-f16.gguf','new.gguf','adapted.gguf');self.assertEqual(args,['original-f16.gguf','new.gguf','q4_k','--lighton4b-imatrix-q4km-style','--imatrix','adapted.gguf']);p=m.plan();self.assertEqual(p['expected_imatrix_applications'],p['q6_body_imatrix_applications']+p['q4_body_imatrix_applications']);self.assertEqual(p['extra_payload_bytes_vs_plain_q4_q8head'],10946560)
if __name__=='__main__':unittest.main()
