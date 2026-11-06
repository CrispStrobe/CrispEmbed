"""Exact selective names and no quantized-input laundering; no framework/model."""
import copy
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_q6_recipe as m
class Controls(unittest.TestCase):
 def test_exact_upstream_layers_and_names(self):
  p=m.plan();self.assertEqual([i for i in range(32)if m.use_more_bits(i)],[0,1,2,3,6,9,12,15,18,21,24,27,28,29,30,31]);self.assertEqual(len(p['q6_names']),32);self.assertEqual(p['types'],{'0':474,'8':1,'12':216,'14':32});self.assertEqual(p['extra_payload_bytes_vs_plain_q4'],164904960)
 def test_head_protected_and_widths(self):
  p=m.plan();self.assertEqual(next(r['type']for r in p['all_tensors']if r['name']=='l.embed_tokens.weight'),8)
  for r in p['all_tensors']:
   if r['type']==14:self.assertEqual(r['source_shape'][-1]%256,0)
 def test_unquantized_input_guard(self):
  p=m.plan();d={}
  for r in p['all_tensors']:
   s=r['source_shape'];s=[s[0],math.prod(s[1:])]if r['name']=='v.patch_embed.proj.weight'else s;d[r['name']]={'stored_shape':list(reversed(s)),'type':0 if r['type']==0 else 1}
  self.assertEqual(d['v.patch_embed.proj.weight']['stored_shape'],[1536,1024])
  m.validate_source_inventory(d,p)
  bad=copy.deepcopy(d);bad['v.patch_embed.proj.weight']['stored_shape']=[96,1024]
  with self.assertRaises(ValueError):m.validate_source_inventory(bad,p)
  for typ in (8,12,14):
   bad=copy.deepcopy(d);bad['l.layers.0.mlp.down_proj.weight']['type']=typ
   with self.assertRaises(ValueError):m.validate_source_inventory(bad,p)
 def test_no_heldout_or_quality_claim(self):
  p=m.plan();self.assertFalse(p['held_out_selection']);self.assertFalse(p['quality_qualified']);self.assertEqual(p['general_file_type'],15)
if __name__=='__main__':unittest.main()
