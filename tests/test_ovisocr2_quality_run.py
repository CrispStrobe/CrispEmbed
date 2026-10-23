"""Pure source and phase gates; no model or renderer imports."""

import importlib.util
import sys
import json
import hashlib
from pathlib import Path
import tempfile
import zipfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_quality_run', ROOT / 'tools/ci-heavy/ovisocr2_quality_run.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class QualityDriverControls(unittest.TestCase):
    def test_immutable_prepared_zip_members_and_path_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            archive = folder/'prepared.zip'
            cases = [{'id': name, 'file': name+'.png',
                      'png_sha256': hashlib.sha256(name.encode()).hexdigest()}
                     for name in ('historical-receipt', 'two-column-page', 'inventory-table')]
            prepared = json.dumps({'cases': cases}).encode()
            prior = {'artifact_report_sha256': hashlib.sha256(prepared).hexdigest()}
            def create(entries):
                with zipfile.ZipFile(archive, 'w') as zipped:
                    for name, value in entries:
                        zipped.writestr(name, value)
                return {'bytes': archive.stat().st_size, 'sha256': driver.sha(archive),
                        'source': {'case_inputs': cases}}
            entries = [('prepared.json', prepared)] + [
                (case['file'], case['id'].encode()) for case in cases]
            receipt = create(entries)
            actual, members = driver.verify_prepared_archive(archive, receipt, prior)
            self.assertEqual(actual['cases'], cases)
            self.assertEqual(set(members), {name for name, _ in entries})
            changed = {**receipt, 'sha256': '0'*64}
            with self.assertRaisesRegex(ValueError, 'ZIP size/SHA'):
                driver.verify_prepared_archive(archive, changed, prior)
            receipt = create(entries + [('../escape.png', b'bad')])
            with self.assertRaisesRegex(ValueError, 'members/path'):
                driver.verify_prepared_archive(archive, receipt, prior)
            traversal_cases = json.loads(json.dumps(cases))
            traversal_cases[-1]['file'] = '../inventory-table.png'
            traversal_prepared = json.dumps({'cases': traversal_cases}).encode()
            traversal_prior = {'artifact_report_sha256': hashlib.sha256(traversal_prepared).hexdigest()}
            receipt = create([('prepared.json', traversal_prepared),
                              *[(case['file'], case['id'].encode()) for case in traversal_cases]])
            receipt['source']['case_inputs'] = traversal_cases
            with self.assertRaisesRegex(ValueError, 'members/path'):
                driver.verify_prepared_archive(archive, receipt, traversal_prior)
            receipt = create(entries[:-1] + [(cases[-1]['file'], b'changed')])
            with self.assertRaisesRegex(ValueError, 'member SHA'):
                driver.verify_prepared_archive(archive, receipt, prior)
            receipt = create(entries)
            with self.assertRaisesRegex(ValueError, 'member SHA'):
                driver.verify_prepared_archive(archive, receipt,
                                               {'artifact_report_sha256': '0'*64})

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

    def test_realistic_document_completion_uses_eos_even_at_cap(self):
        contract = driver.load_module('ovis_quality_test_contract',
                                      ROOT / 'tools/ci-heavy/ovisocr2_quality_contract.py')
        panel = json.loads((ROOT / 'tests/regression/ocr-quality/panel.json').read_text())
        cases, outputs = [], {}
        for spec in panel['cases']:
            case = {'id': spec['id'], 'split': 'realistic-evaluation', 'kind': spec['kind'],
                    'license': spec['license'], 'gold': 'x', 'cap': 2,
                    'rgb_sha256': 'a' * 64, 'png_sha256': 'b' * 64}
            cases.append(case)
            outputs[case['id']] = {'input_rgb_sha256': case['rgb_sha256'],
                                   'input_png_sha256': case['png_sha256'],
                                   'raw_text': 'x', 'processed_text': 'x',
                                   'token_ids': [42, 248046], 'stop_id': 248046,
                                   'finish_reason': 'stop'}
        scorer = SimpleNamespace(score=lambda _gold, _text, *, complete: {'complete': complete})
        with patch.object(contract, 'score_panel', return_value={}), \
                patch.object(driver, 'load_module', return_value=scorer):
            result = driver.score_outputs(contract, None, cases, outputs)
            self.assertTrue(all(row['structure']['complete'] for row in result['realistic_documents']))
            outputs['inventory-table']['token_ids'] = [42, 43]
            outputs['inventory-table']['stop_id'] = None
            outputs['inventory-table']['finish_reason'] = 'length'
            result = driver.score_outputs(contract, None, cases, outputs)
            self.assertFalse(result['realistic_documents'][2]['structure']['complete'])

    def test_native_precision_env_has_explicit_calibrated_gate(self):
        self.assertEqual(driver.native_environment('f16')['CRISPEMBED_QWEN35_QUANT'], '0')
        for arm in ('q8', 'q4', 'q4-calibrated'):
            self.assertEqual(driver.native_environment(arm)['CRISPEMBED_QWEN35_QUANT'], '1')
        with self.assertRaises(ValueError):
            driver.native_environment('unreviewed')
        for arm in ('q8', 'q4', 'q4-calibrated'):
            env = driver.numerical_environment(arm)
            self.assertEqual(env['CRISPEMBED_QWEN35_QUANT'], '1')
            self.assertEqual(env['CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS'], '1')
            self.assertTrue(all(env[name] == '0' for name in driver.DIFF_FLOAT_ZERO))
        with self.assertRaises(ValueError):
            driver.numerical_environment('f16')

    def test_numerical_trace_records_earliest_drift_and_requires_full_coverage(self):
        line = lambda status, stage: (
            f'{status} {stage}: cos_min=0.999990000 cos_global=0.999990000 '
            'max_abs=0.002 mean_abs=0.0001 norms=12/12')
        passing = '\n'.join([line('PASS', f'{phase}.decoder.layer.{i}')
                             for phase in ('prefill', 'cached', 'full_prefix') for i in range(24)]
                            + [line('PASS', 'generated.scores.0'), 'Native generated IDs: 42 248046',
                               'PASS native decoded text exact bytes=5: hello',
                               'PASS composed 24-layer native decoder and greedy IDs'])
        self.assertTrue(driver.parse_numerical_diagnostic(passing, 0)['passed'])
        self.assertEqual(driver.parse_numerical_diagnostic(passing.encode(), 0)['decoded_text'], 'hello')
        changed = passing.replace(line('PASS', 'prefill.decoder.layer.3'),
                                  line('FAIL', 'prefill.decoder.layer.3'), 1)
        result = driver.parse_numerical_diagnostic(changed, 1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['first_failure']['stage'], 'prefill.decoder.layer.3')
        self.assertEqual(result['first_failure']['max_abs'], .002)
        self.assertEqual(result['generated_ids'], [42, 248046])
        self.assertFalse(driver.parse_numerical_diagnostic(passing.split('Native generated IDs:')[0], 0)['passed'])
        nonfinite = passing.replace('max_abs=0.002', 'max_abs=nan', 1)
        report = driver.parse_numerical_diagnostic(nonfinite, 1)
        self.assertFalse(report['passed'])
        self.assertIsNone(report['first_failure']['max_abs'])
        json.dumps(report, allow_nan=False)

    def test_reviewed_q8_failure_can_authorize_separate_q4_diagnostic(self):
        producer = {'requested_phase': 'q8', 'scores': {'q8': {'recorded': True}},
                    'quant_numerical': {'passed': False},
                    'models': {'q8': {'sha256': 'a' * 64}}}
        canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        receipt = {'producer': producer,
                   'independent_review': {'producer_canonical_sha256': canonical,
                                          'q8_diagnostic_complete': True,
                                          'allow_q4_diagnostic': True,
                                          'q8_positive_for_publication': False}}
        result = driver.checked_q8_diagnostic_review(receipt)
        self.assertFalse(result['q8_numerical_passed'])
        self.assertFalse(result['q8_positive_for_publication'])
        altered = json.loads(json.dumps(receipt))
        altered['independent_review']['allow_q4_diagnostic'] = False
        with self.assertRaises(ValueError):
            driver.checked_q8_diagnostic_review(altered)
        altered = json.loads(json.dumps(receipt))
        altered['producer']['models']['q8']['sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            driver.checked_q8_diagnostic_review(altered)

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
