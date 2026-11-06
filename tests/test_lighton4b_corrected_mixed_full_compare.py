"""Bounded model-free admission controls; fixtures never admit real execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/ci-heavy'),str(ROOT/'tools')]
spec=importlib.util.spec_from_file_location('mixed_full_compare',ROOT/'tools/ci-heavy/lighton4b_corrected_mixed_full_compare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def fixture(self):
        r=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-form/native-report.json').read_text())
        # Explicit structural fixture only; no synthetic receipt enters an admission registry.
        r.update(source_sha=m.SOURCE,full_completion_executed=True,requested_teacher_positions=177,requested_generation_cap=256,full_teacher_score_comparison_executed=False)
        r['native_summary'].update(teacher_positions=177,cap=256)
        r['decoded']['native_eos']=r['native_summary']['eos']
        for i in range(177):r['artifacts'][f'teacher.{i}.logits.f32']=dict(bytes=993280,sha256='0'*64)
        return r
    def test_normalized_receipt_explicit_and_full_geometry(self):
        p=m.compare.protocol();r=m.normalize(self.fixture(),p)
        self.assertEqual(len(r['scores']),177)
        self.assertEqual([x['position']for x in r['scores']],list(range(177)))
        self.assertEqual(r['precision'],m.RECIPE)
        self.assertIn('comparer-derived',r['normalization'])
        self.assertEqual(r['teacher_forced_ids'],p['original_generated_ids'])
    def test_smoke_wrong_source_weight_gate_reference_and_partial_reject(self):
        for field,value in [('full_completion_executed',False),('source_sha','0'*40),('recipe','q4_k'),('requested_teacher_positions',1),('requested_generation_cap',2)]:
            r=self.fixture();r[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.normalize(r,m.compare.protocol())
        for kind in ('weights','gates','source','vision','score'):
            r=self.fixture()
            if kind=='weights':r['model']['sha256']='0'*64
            elif kind=='gates':r['disabled_gates']['CRISPEMBED_LIGHTON4B_BF16_LINEAR']='1'
            elif kind=='source':r['original_bf16']['capture_sha256']='0'*64
            elif kind=='vision':r['raw_vision_controls']['vision.patch']['bit_exact']=False
            else:r['artifacts']['teacher.176.logits.f32']['bytes']=4
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.normalize(r,m.compare.protocol())
    def test_original_smoke_not_admitted(self):
        r=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-form/native-report.json').read_text())
        with self.assertRaises(ValueError):m.normalize(r,m.compare.protocol())
    def test_frozen_budget_override_rejected_before_payload_or_binary(self):
        p=copy.deepcopy(m.compare.protocol());p['precisions'].append(m.RECIPE);p['native_producers'].append({});p['max_abs']=1.0
        with tempfile.TemporaryDirectory()as d,patch.dict('os.environ',{'GITHUB_ACTIONS':'true'}):
            out=Path(d)/'blocked'
            self.assertEqual(m.compare.compare(Path(d)/'absent',Path(d)/'absent','absent',out,admitted_protocol=p),2)
            r=json.loads((out/'comparison.json').read_text());self.assertIn('reference or budgets',r['error']);self.assertFalse(r['native_execution_admitted'])
if __name__=='__main__':unittest.main()
