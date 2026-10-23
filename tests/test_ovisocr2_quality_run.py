"""Pure source and phase gates; no model or renderer imports."""

import importlib.util
import sys
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_quality_run', ROOT / 'tools/ci-heavy/ovisocr2_quality_run.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class QualityDriverControls(unittest.TestCase):
    def test_original_readme_ast_prompt_filter_and_cleanup(self):
        source = '''```python
class OvisOCR2Parser:
    def __init__(self):
        prompt = 'Original ' + 'prompt'
    def _clean_truncated_repeats(self, text: str) -> str:
        return text.removesuffix('!')
    def parse(self, filter_imgtags=True):
        text = 'placeholder'
        if filter_imgtags:
            text = "\\n\\n".join(block for block in text.split("\\n\\n")
                               if not block.strip().startswith('<img src="images/bbox_'))
```'''
        prompt, cleaner, filter_fn = driver.original_readme_functions(source)
        self.assertEqual(prompt, 'Original prompt')
        self.assertEqual(driver.process_readme_output(' hello! ', cleaner, filter_fn), 'hello')
        self.assertEqual(driver.process_readme_output('A\n\n<img src="images/bbox_1" />\n\nB',
                                                       cleaner, filter_fn), 'A\n\nB')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            driver.original_readme_functions('```python\nprompt = input()\n```')

    def test_missing_baseline_receipt_blocks_all_phases(self):
        missing = ROOT / 'tests/results/qwen35-ocr/ovisocr2-model-audit.json'
        if missing.exists():
            self.skipTest('A real receipt now exists; its content is reviewed by the hosted gate')
        for phase in driver.PHASES:
            with self.assertRaisesRegex(ValueError, 'baseline receipt absent'):
                driver.require_baseline(phase)
        with self.assertRaises(ValueError):
            driver.require_baseline('unsupported')

    def test_no_framework_import(self):
        for name in ('torch', 'transformers', 'gguf', 'PIL', 'vllm'):
            self.assertNotIn(name, sys.modules)

    def test_reviewed_original_reuse_requires_all_case_ids_and_caps(self):
        cases = [{'id': str(i), 'rgb_sha256': 'a' * 64, 'png_sha256': 'b' * 64,
                  'cap': 16} for i in range(11)]
        outputs = {case['id']: {'input_rgb_sha256': case['rgb_sha256'],
                                'input_png_sha256': case['png_sha256'], 'cap': case['cap'],
                                'token_ids': [42, 248046], 'confidences': [0.4, 0.8],
                                'mean_confidence': 0.6000000000000001,
                                'stop_id': 248046, 'finish_reason': 'stop'} for case in cases}
        current = {'inputs': cases, 'rendered_panel_sha256': 'c' * 64,
                   'reviewed_documents_sha256': 'd' * 64}
        profile = {'checkpoint_sha256': 'e' * 64,
                   'qualification': {'loaded_source_sha256': {'model': 'f' * 64}}}
        prior = {**current, 'source_commit': '1' * 40, 'requested_phase': 'original',
                 'status': 'completed original HF quality baseline; review pending',
                 'original_source': {'readme_sha256': '2' * 64,
                                     'checkpoint_sha256': profile['checkpoint_sha256'],
                                     'framework_source_sha256': profile['qualification']['loaded_source_sha256']},
                 'original_cases': outputs, 'scores': {'original': {'verified': True}}}
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'review.json'
            receipt.write_text(json.dumps({'source_commit': '1' * 40}))
            with patch.object(driver, 'ORIGINAL_REVIEW', receipt), \
                    patch.object(driver, 'score_outputs', return_value={'verified': True}):
                validated, scores = driver.validate_reused_original(
                    prior, current, cases, None, None, profile, '2' * 64)
                self.assertEqual(len(validated), 11)
                self.assertEqual(scores, {'verified': True})
                prior['original_cases']['10']['cap'] = 17
                with self.assertRaisesRegex(ValueError, 'record differs'):
                    driver.validate_reused_original(prior, current, cases, None, None, profile, '2' * 64)

    def test_selected_token_confidences_reject_nonfinite_and_mismatch(self):
        self.assertAlmostEqual(driver.validate_selected_confidences(
            [42, 248046], [0.25, 0.75], 16), 0.5)
        for ids, probabilities in (([42], []), ([42], [float('nan')]),
                                   ([42], [float('inf')]), ([42], [-0.1]),
                                   ([248320], [0.5]), ([], [])):
            with self.assertRaises(ValueError):
                driver.validate_selected_confidences(ids, probabilities, 16)

    def test_native_precision_env_has_explicit_calibrated_gate(self):
        self.assertEqual(driver.native_environment('f16')['CRISPEMBED_QWEN35_QUANT'], '0')
        for arm in ('q8', 'q4', 'q4-calibrated'):
            self.assertEqual(driver.native_environment(arm)['CRISPEMBED_QWEN35_QUANT'], '1')
        with self.assertRaises(ValueError):
            driver.native_environment('unreviewed')

    def test_wrapped_reviewed_model_receipt_binds_producer_and_artifact(self):
        producer = {'status': 'completed diagnostic audit', 'failures': [],
                    'source_commit': 'a' * 40,
                    'exports': {'f16': {'model_sha256': driver.F16_SHA}}}
        canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        receipt = {'run': 123, 'artifact': {'id': 456, 'zip_sha256': 'b' * 64},
                   'producer_report_sha256': 'c' * 64, 'producer': producer,
                   'independent_review': {'f16_positive': True,
                                          'artifact_zip_sha256': 'b' * 64,
                                          'producer_canonical_sha256': canonical}}
        self.assertEqual(driver.checked_model_receipt(receipt)['source_commit'], 'a' * 40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(receipt))
            with patch.object(driver, 'BASELINE', path):
                self.assertEqual(driver.require_baseline('original')['baseline_pins']['run'], 123)
        for change in ('bare', 'failed', 'f16', 'canonical', 'artifact', 'run', 'review'):
            altered = json.loads(json.dumps(receipt))
            if change == 'bare':
                altered = producer
            elif change == 'failed':
                altered['producer']['failures'] = ['context']
            elif change == 'f16':
                altered['producer']['exports']['f16']['model_sha256'] = 'd' * 64
            elif change == 'canonical':
                altered['independent_review']['producer_canonical_sha256'] = 'd' * 64
            elif change == 'artifact':
                altered['artifact']['zip_sha256'] = 'd' * 64
            elif change == 'run':
                altered['run'] = 0
            else:
                altered['independent_review']['f16_positive'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.checked_model_receipt(altered)

    def test_reviewed_model_service_run_head_and_artifact(self):
        pins = {'run': 123, 'source_commit': 'a' * 40,
                'artifact_id': 456, 'artifact_zip_sha256': 'b' * 64}
        run = {'id': 123, 'status': 'completed', 'conclusion': 'success',
               'head_sha': 'a' * 40, 'path': '.github/workflows/ovisocr2-metadata.yml'}
        artifact = {'id': 456, 'expired': False, 'workflow_run': {'id': 123},
                    'digest': 'sha256:' + 'b' * 64}
        def response(argv, **_):
            return json.dumps(artifact if 'artifacts/' in argv[-1] else run)
        with patch.object(driver.subprocess, 'check_output', side_effect=response):
            self.assertEqual(driver.verify_model_receipt_service(pins)['run'], 123)
            artifact['digest'] = 'sha256:' + 'd' * 64
            with self.assertRaisesRegex(ValueError, 'service proof'):
                driver.verify_model_receipt_service(pins)


if __name__ == '__main__':
    unittest.main()
