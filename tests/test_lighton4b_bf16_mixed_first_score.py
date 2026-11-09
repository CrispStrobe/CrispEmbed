"""Predeclared mixed partition controls; no framework/model import."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mixed_contract',ROOT/'tools/lighton4b_bf16_quant_first_score.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MixedContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=m.tensor_plan(json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_text()))
        cls.protocol=json.loads((ROOT/'docs/lighton4b-bf16-mixed-first-score-v1.json').read_text())
    def test_complete_architectural_partition(self):
        selected=m.mixed_selection(self.protocol,self.plan)
        self.assertEqual(len(selected),249)
        self.assertEqual(sum(v=='q8_0'for v in selected.values()),153)
        self.assertEqual(sum(v=='q4_k'for v in selected.values()),96)
        self.assertEqual(selected['l.embed_tokens.weight'],'q8_0')
        self.assertTrue(all(v==('q4_k'if'.mlp.'in k else'q8_0')for k,v in selected.items()))
    def test_tamper_duplicate_missing_digest_reject(self):
        for kind in ('missing','duplicate','digest'):
            p=copy.deepcopy(self.protocol);row=p['mixed_policy']['q8_0']
            if kind=='missing':row['names'].pop()
            elif kind=='duplicate':row['names'].append(row['names'][0])
            else:row['names_sha256']='0'*64
            with self.assertRaises(ValueError):m.mixed_selection(p,self.plan)
    def test_rehashed_wrong_architecture_reject(self):
        p=copy.deepcopy(self.protocol);a=p['mixed_policy']['q8_0'];b=p['mixed_policy']['q4_k'];a['names'][1],b['names'][0]=b['names'][0],a['names'][1]
        for row in (a,b):row['names'].sort();row['names_sha256']=hashlib.sha256(('\n'.join(row['names'])+'\n').encode()).hexdigest()
        with self.assertRaises(ValueError):m.mixed_selection(p,self.plan)
    def test_order_scope_chunk_and_call_controls(self):
        for key,value in [('candidate_order',['q4_k','q8_0']),('expected_forward_arms',3),('max_chunk_elements',262144),('held_out_calibration',True),('architecture_quality_selection',True)]:
            p=copy.deepcopy(self.protocol);p[key]=value
            with self.assertRaises(ValueError):m.mixed_selection(p,self.plan)
    def test_old_three_arm_protocol_stays_unmixed(self):
        self.assertIsNone(m.mixed_selection(json.loads(m.PROTOCOL.read_text()),self.plan))
    def test_bounded_cache_digest_exact(self):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'owned';p.write_bytes(b'bounded cache proof'*4096)
            self.assertEqual(m.digest_with_bounded_cache(p),m.digest(p))
if __name__=='__main__':unittest.main()
