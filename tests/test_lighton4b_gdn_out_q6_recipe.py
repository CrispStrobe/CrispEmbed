"""Separate24-output Q6 selection preserves original32 recipe and strict lineage."""
from pathlib import Path
import sys
import unittest
import tempfile
import test_lighton4b_q6_convert as fixture
import lighton4b_q6_convert as conversion
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_gdn_out_q6_recipe as out
import lighton4b_q6_recipe as old
class Controls(unittest.TestCase):
 def test_exact24_and_no_other_precision_changes(self):
  p=out.plan();self.assertEqual(p['q6_names'],sorted(f'l.layers.{i}.linear_attn.out_proj.weight'for i in range(32)if i%4!=3));self.assertEqual(p['types'],{'0':474,'8':1,'12':224,'14':24});self.assertEqual(p['extra_payload_bytes_vs_plain_q4'],64880640)
 def test_custom_metadata_serializes_with_typed_guards(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d)/'base.gguf';p=fixture.fixture(base);actual=out.plan();p.update(recipe=actual['recipe'],q6_names_sha256=actual['q6_names_sha256'],metadata_strings=actual['metadata_strings'],metadata_uint32=actual['metadata_uint32']);head,rows,size=conversion.header(conversion.parse(base),base,p);target=Path(d)/'candidate.gguf';target.write_bytes(head+b'\0'*(size-len(head)));meta=conversion.parse(target,conversion.TYPES)['metadata'];self.assertEqual(meta['qwen35_ocr.mixed_quant_recipe']['value'],out.RECIPE)
   for k,v in actual['metadata_uint32'].items():self.assertEqual(meta[k]['value'],v)
   self.assertEqual(meta['qwen35_ocr.q6_names_sha256']['value'],actual['q6_names_sha256'])
 def test_original32_preserved(self):
  a=old.plan();b=out.plan();self.assertEqual(len(a['q6_names']),32);self.assertNotEqual(a['recipe'],b['recipe']);self.assertEqual(len(a['q6_names']),32);self.assertEqual(a['types']['14'],32)
 def test_direct_source_and_tied_head(self):
  p=out.plan();self.assertEqual(next(r['type']for r in p['all_tensors']if r['name']=='l.embed_tokens.weight'),8);self.assertIn('quantized Q8/Q4 inputs prohibited',p['source_policy']);self.assertFalse(p['quality_qualified'])
if __name__=='__main__':unittest.main()
