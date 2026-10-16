"""Model-free rejection of drift, wrong precision and incomplete quantized evidence."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
from qwen35_quality_quant import frozen_baseline, frozen_original, summary, tensor_audit


def bootstrap():
    spec = importlib.util.spec_from_file_location('quant_quality_bootstrap', ROOT / 'tools/ci-heavy/qwen35_ocr_bootstrap.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class QualityQuant(unittest.TestCase):
    def fixture(self):
        return next(c for c in json.loads((ROOT / 'tests/results/qwen35-ocr/quality.json').read_text())['cases']
                    if c['id'] == 'inventory-table')

    def test_preselected_recipe_document_requests(self):
        module = bootstrap()
        self.assertIsNone(module.quality_quant_request(['--quality']))
        for profile, case in itertools.product(('q8', 'q8-q6', 'q4'),
                ('inventory-table', 'historical-receipt', 'two-column-page')):
            self.assertEqual(module.quality_quant_request(['--quality-quant', profile, case]), (profile, case))
        for args in ([], ['q8'], ['q8', 'unknown'], ['unknown', 'inventory-table'],
                     ['q8', '../inventory-table'], ['q8', 'inventory-table', '--relax']):
            with self.assertRaises(ValueError): module.quality_quant_request(['--quality-quant', *args])

    def test_missing_review_and_render_drift_rejected_before_heavy_imports(self):
        for method in ('require_reviewed_preparation', 'preflight'):
            module = bootstrap(); quality = Mock()
            getattr(quality, method).side_effect = ValueError('frozen evidence changed')
            with patch.object(module, 'module', return_value=quality), patch.dict('os.environ', {'HEAVY_OUT':'unused'}), \
                    patch.object(sys, 'argv', ['bootstrap', '--quality-quant', 'q8', 'inventory-table']), \
                    patch('builtins.__import__', side_effect=AssertionError('heavy import reached')):
                with self.assertRaisesRegex(ValueError, 'frozen evidence changed'): module.main()

    def test_original_exact_except_observed_seconds(self):
        old = self.fixture()['reference']; new = copy.deepcopy(old); new['generation_seconds'] += 1
        self.assertTrue(frozen_original(new, old))
        for mutate in (lambda r:r.update(generated_ids=[]), lambda r:r.update(selected_token_confidences=[]),
                       lambda r:r.update(grid=[]), lambda r:r['cpu_controls'].pop('backend_off')):
            new = copy.deepcopy(old); mutate(new)
            with self.assertRaises(ValueError): frozen_original(new, old)

    def test_full_f16_replay_except_observed_seconds(self):
        old = self.fixture(); new = copy.deepcopy(old); new['decode']['native_recognize_ms'] += 1
        self.assertTrue(frozen_baseline(new, old))
        for mutate in (lambda r:r['decode'].update(native_ids=[]),
                       lambda r:r['decode']['metrics'][0].update(max_abs=10),
                       lambda r:r['prefill'].update(qualified_failed_stages=['new']),
                       lambda r:r['teacher_quality'].update(qualified_failures=99),
                       lambda r:r['native_quality'].update(parse_ok=False)):
            new = copy.deepcopy(old); mutate(new)
            with self.assertRaises(ValueError): frozen_baseline(new, old)

    def tensors(self):
        make = lambda shape, kind:dict(shape=shape, type=kind, bytes=4, sha256='old')
        before={'v.matrix':make([2,2],'F32'), 'l.norm':make([2],'F32'), 'l.conv':make([2,2,2],'F16'),
                'l.embed_tokens.weight':make([2,2],'F16'), 'l.proj':make([2,2],'F16')}
        after=copy.deepcopy(before)
        for name in ('l.embed_tokens.weight','l.proj'):after[name].update(type='Q8_0',sha256='quantized')
        return before, after

    def test_protected_bytes_and_all_logical_shapes(self):
        before,after=self.tensors();self.assertTrue(tensor_audit(before,after,['v.*=f32'],'q8_0'))
        for name in ('v.matrix','l.norm','l.conv'):
            new=copy.deepcopy(after);new[name]['sha256']='changed'
            with self.assertRaises(ValueError):tensor_audit(before,new,['v.*=f32'],'q8_0')
        for name in before:
            new=copy.deepcopy(after);new[name]['shape']=[9,9]
            with self.assertRaises(ValueError):tensor_audit(before,new,[],'q8_0')
        new=copy.deepcopy(after);new.pop('l.proj')
        with self.assertRaises(ValueError):tensor_audit(before,new,[],'q8_0')

    def test_effective_precision_and_rule_precedence(self):
        before,after=self.tensors();after['l.proj']['type']='Q6_K'
        self.assertTrue(tensor_audit(before,after,['l.*=q8_0','l.proj=q6_k'],'q4_k'))
        with self.assertRaises(ValueError):tensor_audit(before,after,['l.proj=q6_k','l.*=q8_0'],'q4_k')
        after['l.embed_tokens.weight']['type']='Q4_K'
        with self.assertRaises(ValueError):tensor_audit(before,after,['l.proj=q6_k'],'q4_k')

    def test_summary_retains_failures_and_rejects_incomplete_pairs(self):
        arm=dict(name='f16-baseline', decode=dict(complete_document=True,ids_exact=True,text_exact=True,
            same_stop=True,first_teacher_argmax_difference=None),prefill=dict(qualified_failed_stages=[]),
            teacher_quality=dict(qualified_failures=0),native_quality=dict(parse_ok=True))
        other=copy.deepcopy(arm);other['name']='q8-vision-f32'
        other['decode'].update(ids_exact=False,first_teacher_argmax_difference=4)
        other['prefill']['qualified_failed_stages']=['cache']
        other['teacher_quality']['qualified_failures']=3
        result=summary([arm,other]);self.assertTrue(result[1]['output_failure']);self.assertTrue(result[1]['teacher_argmax_failure'])
        self.assertEqual(result[1]['prefill_failures'],1);self.assertEqual(result[1]['score_failures'],3)
        for arms in ([arm], [arm,arm], [other,arm]):
            with self.assertRaises(ValueError):summary(arms)


if __name__ == '__main__': unittest.main()
