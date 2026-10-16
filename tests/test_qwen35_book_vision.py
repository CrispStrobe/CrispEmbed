"""Model-free frozen-input, observer-fidelity and translation-unit guards."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
from qwen35_book_vision import compile_isolation, frozen_reference


class BookVision(unittest.TestCase):
    def fixture(self):
        old = next(c['reference'] for c in json.loads((ROOT / 'tests/results/qwen35-ocr/quality.json').read_text())['cases'] if c['id'] == 'two-column-page')
        reference = copy.deepcopy(old)
        reference.update(max_new_tokens=1, generated_ids=old['generated_ids'][:1],
                         selected_token_confidences=old['selected_token_confidences'][:1])
        for rows in reference['cpu_controls'].values():
            rows.extend(dict(stage=f'prefill.vision.probe.5.trace.control{i}', max_abs=0) for i in range(12))
        reference['attention_trace'] = dict(original_calls={'prefill': 2, 'backend_off': 2, 'patch_chunked': 2},
            probability_and_output_exact_controls=12, projection_input_exact_controls=6,
            selected_blocks=[5, 11], query_positions=list(range(16)), full_key_axis=True)
        return reference, old

    def test_shared_original_control_records_and_prefix(self):
        self.assertTrue(frozen_reference(*self.fixture()))

    def test_changed_prefix_input_control_or_trace_rejected(self):
        for mutate in (lambda r:r.update(grid=[[1, 2, 3]]), lambda r:r.update(max_new_tokens=2),
                lambda r:r.update(generated_ids=[]), lambda r:r.update(selected_token_confidences=[0]),
                lambda r:r['cpu_controls'].pop('backend_off'),
                lambda r:r['cpu_controls']['backend_off'][0].update(max_abs=100),
                lambda r:r['attention_trace'].update(probability_and_output_exact_controls=11),
                lambda r:r['attention_trace'].update(full_key_axis=False)):
            reference, old = self.fixture(); mutate(reference)
            with self.assertRaises(ValueError): frozen_reference(reference, old)

    def commands(self):
        return [dict(file='/source/qwen35_vision.cpp', command='c++ production'),
                dict(file='/tests/test_qwen35_vision_ops.cpp', command='c++ -DCRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT=1'),
                dict(file='/tests/test_qwen35_attention_trace.cpp', command='c++ trace')]

    def test_production_and_attention_must_not_use_validation_copy(self):
        self.assertTrue(compile_isolation(self.commands()))
        for index in (0, 2):
            commands = self.commands(); commands[index]['command'] += ' -DCRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT=1'
            with self.assertRaises(ValueError): compile_isolation(commands)
        for index in range(3):
            commands = self.commands(); commands.pop(index)
            with self.assertRaises(ValueError): compile_isolation(commands)

    def test_missing_review_rejected_before_heavy_imports(self):
        spec = importlib.util.spec_from_file_location('book_vision_bootstrap', ROOT / 'tools/ci-heavy/qwen35_ocr_bootstrap.py')
        bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
        quality = Mock(); quality.require_reviewed_preparation.side_effect = ValueError('review absent')
        with patch.object(bootstrap, 'module', return_value=quality), patch.object(sys, 'argv', ['bootstrap', '--book-vision']):
            with patch('builtins.__import__', side_effect=AssertionError('heavy import reached')):
                with self.assertRaisesRegex(ValueError, 'review absent'): bootstrap.main()
        quality.preflight.assert_not_called()


if __name__ == '__main__': unittest.main()
