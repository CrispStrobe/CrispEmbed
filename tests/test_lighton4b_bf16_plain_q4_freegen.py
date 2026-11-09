"""Model-free unchanged-Q4 oracle admission and corruption controls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('plain_q4',ROOT/'tools/lighton4b_bf16_plain_q4_freegen.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((ROOT/'docs/lighton4b-bf16-plain-q4-freegen-v1.json').read_text())
        self.plan=m.base.tensor_plan(json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_text()))
    def test_complete_unchanged_policy(self):
        selected=m.base.mixed_selection(self.p,self.plan)
        self.assertEqual(len(selected),249);self.assertEqual(set(selected.values()),{'q4_k'})
        self.assertEqual(self.p['candidate_order'],['q4_k']);self.assertEqual(self.p['max_new_tokens'],256)
        self.assertEqual(self.p['maximum_score_bytes'],256*993280)
    def test_bad_map_or_digest_rejected(self):
        for field,value in [('plain_q4_names',self.p['plain_q4_names'][:-1]),('plain_q4_names_sha256','0'*64),('candidate_order',['q8_0','q4_k']),('max_chunk_elements',1048576),('held_out_calibration',True),('architecture_quality_selection',True)]:
            p=copy.deepcopy(self.p);p[field]=value
            with self.assertRaises(ValueError):m.base.mixed_selection(p,self.plan)
    def test_no_q8_file(self):
        self.assertEqual(set(self.p['candidates']),{'q4_k'})
        self.assertEqual(self.p['candidates']['q4_k']['sha256'],'f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91')
        self.assertEqual(self.p['mixed_first_score_sha256'],'e59d6054c85e81fb1be7101c9899287fa3a1cdd23a3110c8e03d874992387a0d')
    def test_closure_eos_or_cap(self):
        m.full.validate_generation([20353,248046],2,{'root':2,'text':2,'vision':1})
        m.full.validate_generation([20353]*256,256,{'root':256,'text':256,'vision':1})
if __name__=='__main__':unittest.main()
