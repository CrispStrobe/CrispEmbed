"""Pure hosted-recipe guards; never import model or framework packages locally."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'tools/kaggle/ovisocr2-vllm-quality-reference'
with patch.dict(os.environ, {'OVIS_OUT': '/tmp/ovis-test-out', 'OVIS_SCRATCH': '/tmp/ovis-test-scratch'}):
    spec = importlib.util.spec_from_file_location('ovis_quality_gpu_worker', RECIPE / 'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
spec = importlib.util.spec_from_file_location('ovis_quality_gpu_runner',
                                               RECIPE / 'ovisocr2_vllm_quality_reference.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class QualityGPUControls(unittest.TestCase):
    def test_vllm_stop_sources_preserve_actual_ids(self):
        cap = 2
        eos = SimpleNamespace(token_ids=[42, 248046], finish_reason='stop', stop_reason=None)
        result = worker.checked_stop(eos, cap)
        self.assertTrue(result['complete'])
        self.assertTrue(result['terminal_eos_in_ids'])
        omitted = SimpleNamespace(token_ids=[42], finish_reason='stop', stop_reason=248044)
        result = worker.checked_stop(omitted, cap)
        self.assertTrue(result['complete'])
        self.assertTrue(result['special_stop_omitted_from_ids'])
        self.assertEqual(omitted.token_ids, [42])
        capped = SimpleNamespace(token_ids=[42, 43], finish_reason='length', stop_reason=None)
        self.assertFalse(worker.checked_stop(capped, cap)['complete'])
        for bad in (SimpleNamespace(token_ids=[42, 43, 44], finish_reason='length', stop_reason=None),
                    SimpleNamespace(token_ids=[42], finish_reason='length', stop_reason=None),
                    SimpleNamespace(token_ids=[42, 248046], finish_reason='length', stop_reason=None),
                    SimpleNamespace(token_ids=[248044, 42], finish_reason='stop', stop_reason=248044)):
            with self.assertRaises(ValueError):
                worker.checked_stop(bad, cap)

    def test_selected_logprob_is_actual_sampled_token(self):
        result = SimpleNamespace(token_ids=[42], logprobs=[{42: SimpleNamespace(logprob=-0.6931471805599453)}])
        scored = worker.selected_confidences(result)
        self.assertTrue(scored['available'])
        self.assertAlmostEqual(scored['values'][0], 0.5)
        self.assertFalse(worker.selected_confidences(
            SimpleNamespace(token_ids=[42], logprobs=[{}]))['available'])
        self.assertFalse(worker.selected_confidences(
            SimpleNamespace(token_ids=[42], logprobs=None))['available'])

    def test_reviewed_v6_receipt_gate(self):
        original = ROOT / 'tests/results/qwen35-ocr/ovisocr2-vllm-reference.json'
        if not original.is_file():
            self.skipTest('Sparse checkout omits the committed reviewed V6 receipt')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(runner, 'SCRATCH', output), patch.object(runner, 'OUT', output), \
                    patch.object(runner, 'fetch', side_effect=lambda _url, path, timeout: path.write_bytes(original.read_bytes())):
                report = {}
                runner.v6_positive_gate(report)
                self.assertTrue(report['reviewed_v6']['reviewed_genuine_forward'])
                changed = json.loads(original.read_text())
                changed['independent_review']['genuine_gpu_forward_completed'] = False
                with patch.object(runner, 'fetch', side_effect=lambda _url, path, timeout: path.write_text(json.dumps(changed))):
                    with self.assertRaises(ValueError):
                        runner.v6_positive_gate({})

    def test_no_model_framework_import(self):
        import sys
        for name in ('torch', 'vllm', 'transformers', 'PIL', 'numpy'):
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main()
