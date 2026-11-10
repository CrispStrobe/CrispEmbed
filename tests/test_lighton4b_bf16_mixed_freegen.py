"""Pure-stdlib full-generation contract checks; no framework/checkpoint import."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('full_oracle',ROOT/'tools/lighton4b_bf16_mixed_freegen.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def test_frozen_policy_and_bounds(self):
        p=json.loads((ROOT/'docs/lighton4b-bf16-mixed-freegen-v1.json').read_text());plan=m.base.tensor_plan(json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_text()))
        self.assertEqual(len(m.base.mixed_selection(p,plan)),249)
        self.assertEqual((p['max_new_tokens'],p['maximum_score_bytes'],p['worker_timeout_seconds']),(256,256*993280,5400))
        self.assertTrue(p['full_free_generation']);self.assertFalse(p['teacher_forced_replay']);self.assertFalse(p['held_out_calibration'])
        self.assertEqual(hashlib.sha256((ROOT/'docs/lighton4b-bf16-quant-v1.json').read_bytes()).hexdigest(),p['original_full_protocol_sha256'])
    def test_genuine_generate_call(self):
        tree=ast.parse((ROOT/'tools/lighton4b_bf16_mixed_freegen.py').read_text());calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='generate']
        self.assertEqual(len(calls),1);kw={k.arg:ast.literal_eval(k.value)for k in calls[0].keywords if k.arg and k.arg!='max_new_tokens'}
        self.assertEqual(kw,{'do_sample':False,'use_cache':True,'return_dict_in_generate':True,'output_scores':True})
        self.assertEqual(next(k.value.id for k in calls[0].keywords if k.arg=='max_new_tokens'),'cap')
    def test_eos_or_cap_without_eos_both_valid(self):
        m.validate_generation([27,248046],2,{'root':2,'text':2,'vision':1})
        m.validate_generation([27]*256,256,{'root':256,'text':256,'vision':1})
    def test_bad_scope_stop_counts_reject(self):
        for ids,scores,counts in [([],0,{'root':0,'text':0,'vision':1}),([27],1,{'root':1,'text':1,'vision':1}),([248046,27],2,{'root':2,'text':2,'vision':1}),([27]*257,257,{'root':257,'text':257,'vision':1}),([27,248046],1,{'root':2,'text':2,'vision':1}),([27,248046],2,{'root':2,'text':2,'vision':2}),([27,248046],2,{'root':1,'text':2,'vision':1})]:
            with self.assertRaises(ValueError):m.validate_generation(ids,scores,counts)
    def test_raw_decoded_agreement(self):
        # Use strings for raw text; no normalization of accents/newlines/case.
        r=m.decoded_agreement([1,2],'é\nA',[1,3],'e\nA')
        self.assertEqual((r['matching_prefix_ids'],r['first_differing_id'],r['cer']['edits']),(1,1,1))
        self.assertEqual(r['wer']['edits'],1)
if __name__=='__main__':unittest.main()
