"""Model-free checks of the OvisOCR2 graph-decoder quality comparison and its recorded reference.

The graph-path reference (tests/results/qwen35-ocr/ovisocr2-graph-quality.json) is additive:
the frozen first-F16 receipt stays the decoder_cpu reference (CRISPEMBED_QWEN35_OCR_GRAPH_DECODER=0).
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('graph_check', ROOT / 'tools/ci-heavy/ovisocr2_graph_quality_check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
FROZEN = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']
REFERENCE = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-graph-quality.json').read_text())


def current_from_frozen():
    current = copy.deepcopy(FROZEN)
    current['cabi_smoke'] = {'f16': {'raw_text': FROZEN['native_cases']['f16']['dev-hello']['raw_text']}}
    return current


class Comparison(unittest.TestCase):
    def test_confidence_drift_passes_and_is_measured(self):
        current = current_from_frozen()
        row = current['native_cases']['f16']['two-column-page']
        row['confidences'] = [min(1.0, v + 1e-5) for v in row['confidences']]
        result = check.compare(current, FROZEN)
        self.assertTrue(result['passed'])
        self.assertTrue(result['graph_path_observed'])
        self.assertAlmostEqual(result['cases']['two-column-page']['max_selected_confidence_abs_drift'], 1e-5, places=7)
        self.assertEqual(result['token_count'], 1841)

    def test_identical_confidences_mean_gate_not_reached(self):
        self.assertFalse(check.compare(current_from_frozen(), FROZEN)['passed'])

    def test_any_id_or_text_difference_fails_and_is_listed(self):
        current = current_from_frozen()
        current['native_cases']['f16']['hold-code']['confidences'][0] -= 1e-6
        current['native_cases']['f16']['inventory-table']['token_ids'][3] += 1
        current['native_cases']['f16']['dev-menu']['raw_text'] += 'x'
        result = check.compare(current, FROZEN)
        self.assertFalse(result['passed'])
        self.assertEqual(result['differing_cases'], ['dev-menu', 'inventory-table'])
        self.assertEqual(result['cases']['inventory-table']['first_id_divergence'], 3)

    def test_missing_case_fails(self):
        current = current_from_frozen()
        current['native_cases']['f16']['hold-code']['confidences'][0] -= 1e-6
        del current['native_cases']['f16']['hold-date']
        result = check.compare(current, FROZEN)
        self.assertFalse(result['passed'])
        self.assertEqual(result['missing_cases'], ['hold-date'])


class Reference(unittest.TestCase):
    def test_reference_is_additive_and_complete(self):
        amendment = REFERENCE['amendment']
        self.assertEqual(amendment['old_reference']['path'], 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json')
        self.assertEqual(set(REFERENCE['arms']), {'native', 'f32', 'default'})
        for arm in REFERENCE['arms'].values():
            self.assertTrue(arm['passed'])
            self.assertEqual(arm['token_count'], 1841)
            self.assertEqual(set(arm['cases']), set(FROZEN['native_cases']['f16']))
            for name, case in arm['cases'].items():
                old = FROZEN['native_cases']['f16'][name]
                self.assertTrue(case['ids_text_eos_exact'])
                self.assertEqual(case['token_ids_sha256'], hashlib.sha256(json.dumps(old['token_ids']).encode()).hexdigest())
                self.assertEqual(len(case['confidences']), len(old['token_ids']))
                drift = max(abs(a - b) for a, b in zip(case['confidences'], old['confidences']))
                self.assertAlmostEqual(drift, case['max_selected_confidence_abs_drift'], places=12)
                self.assertLess(drift, 1e-4)

    def test_default_arm_equals_native(self):
        arms = REFERENCE['arms']
        for name in arms['native']['cases']:
            self.assertEqual(arms['default']['cases'][name]['confidences'], arms['native']['cases'][name]['confidences'])


if __name__ == '__main__':
    unittest.main()
