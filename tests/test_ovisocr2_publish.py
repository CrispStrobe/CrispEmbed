"""Pure publication qualification tests. No hub client, auth or weights."""
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ovis_publish', ROOT/'tools/ci-heavy/ovisocr2_publish.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
HASH = 'a'*64
HEAD = 'b'*40
SELECTION = {'name': 'both', 'enabled_gates': list(subject.vision_contract().PRESETS['both']),
             'values': {flag: ('1' if flag in subject.vision_contract().PRESETS['both'] else '0')
                        for flag in subject.vision_contract().FLAGS}}


def write(folder, name, value):
    path = folder/name
    path.write_text(json.dumps(value))
    return subject.digest(path)


def fixtures(folder, precision='f16'):
    gpu = {'producer': {'status': 'completed bounded original vLLM T4 reference',
            'worker': {'status': 'completed bounded original vLLM reference',
                       'gpu_control': {'passed': True},
                       'runtime_stop_probe': {'all_stop_ids': [248044, 248046]}}},
           'independent_review': {'genuine_gpu_forward_completed': True}}
    gpu_sha = write(folder, 'ovisocr2-vllm-reference.json', gpu)
    model = {'run': 123,
             'artifact': {'id': 1001, 'zip_sha256': HASH,
                          'name': 'ovisocr2-model-reports-123'},
             'reference_artifact': {'id': 1002, 'zip_sha256': HASH,
                                    'name': 'ovisocr2-model-reference-123'},
             'producer': {'status': 'completed diagnostic audit', 'failures': [],
                          'source_commit': HEAD,
                          'selected_vision_preset': {**SELECTION,
                              'all_other_native_gates_unchanged': True,
                              'no_production_default_promoted': True},
                          'native_environment': SELECTION['values'],
                          'gpu_prerequisite': {'receipt_sha256': gpu_sha},
                          'exports': {'f16': {'model_sha256': subject.F16_SHA}},
                          'reference_artifacts': {'gguf': {'sha256': HASH}},
                          'original_capture': {'cached_vs_full_prefix_same_argmax': True,
                                               'cached_vs_full_prefix_max_abs': .001}},
             'independent_review': {'f16_positive': True, 'artifact_zip_sha256': HASH,
                                    'reference_artifact_zip_sha256': HASH}}
    model['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
        model['producer'], sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    model_sha = write(folder, 'ovisocr2-model-audit.json', model)
    public = {'producer': {'status': 'completed public audit', 'failures': [],
                           'source_commit': HEAD, 'selected_vision_preset': SELECTION,
                           'model_prerequisite': {'receipt_sha256': model_sha},
                           'artifacts': {'model': {'sha256': subject.F16_SHA}},
                           'public_confidence_comparison': {'passed': True},
                           'abi_layout': {'passed': True}},
              'independent_review': {'public_positive': True}}
    write(folder, 'ovisocr2-public-surfaces.json', public)
    original = {'report_sha256': HASH,
                'independent_review': {'original_11_complete': True}}
    original_sha = write(folder, 'ovisocr2-original-quality.json', original)
    candidate_sha = subject.F16_SHA if precision == 'f16' else HASH
    native = {f'case-{index}': {'token_ids': [1], 'confidences': [.9], 'cap': 12}
              for index in range(11)}
    quality = {'run': 456, 'producer': {
        'status': ('completed F16 quality observations; independent review pending'
                   if precision == 'f16' else 'completed quantized quality observations; independent review pending'),
        'requested_phase': precision, 'source_commit': HEAD,
        'selected_vision_preset': SELECTION,
        'prerequisite': {'baseline_sha256': model_sha, 'selected_vision_preset': SELECTION,
                         'original_review_sha256': original_sha},
        'models': {precision: {'sha256': candidate_sha, 'bytes': 1_717_222_848,
                               **({} if precision == 'f16' else {'recipe': subject.RECIPES[precision]})}},
        'scores': {'original': {}, 'f16': {}, precision: {}},
        'cabi_smoke': {precision: {'text': 'Hello'}},
        'f16_paired_synthetic': {'screen_passed': True},
        'f16_paired_documents': {'screen_passed': True},
        **({} if precision == 'f16' else {
            'tensor_audit': {'recipe': subject.RECIPES[precision], 'tensor_count': 473,
                             'protected_byte_exact': True, 'changed_matrix_count': 100},
            'quant_numerical': {'passed': True, 'exit_code': 0, 'comparison_coverage': True,
                                'first_failure': None, 'source_reference_sha256': HASH,
                                'log_sha256': HASH, 'metric_lines': ['PASS generated.scores.0']},
            'native_cases': {precision: native},
            'paired_synthetic': {'screen_passed': True},
            'paired_documents': {'original': {'screen_passed': True},
                                 'f16': {'screen_passed': True}}})},
        'independent_review': {'quality_positive': True,
                               **({} if precision == 'f16' else {'numerical_positive': True})}}
    if precision != 'f16':
        quality['candidate_artifact'] = {'id': 2002, 'zip_sha256': HASH}
        quality['independent_review']['candidate_artifact_zip_sha256'] = HASH
    quality_sha = write(folder, f'ovisocr2-{precision}-quality.json', quality)
    attempts = {name: {'status': 'rejected', 'report_sha256': HASH,
                       'run': 789, 'reason': 'Independent quality review rejected this attempt'}
                for name in ('q8', 'q4', 'q4-calibrated')}
    if precision != 'f16':
        attempts[precision] = {'status': 'qualified', 'report_sha256': quality_sha,
                               'run': 456, 'candidate_sha256': candidate_sha}
    decision = {'selected_precision': precision, 'attempts': attempts,
                'independent_review': {'selection_positive': True,
                                       'all_attempts_completed': True}}
    write(folder, 'ovisocr2-quant-attempts.json', decision)
    calibration = {}
    if precision == 'q4-calibrated':
        calibration = {'producer': {'precision': precision, 'candidate_sha256': candidate_sha,
                                    'recipe': subject.RECIPES[precision],
                                    'quality_receipt_sha256': quality_sha,
                                    'selection_frozen_before_heldout': True,
                                    'heldout_not_used_for_recipe': True,
                                    'imatrix_sha256': HASH},
                       'independent_review': {'calibration_positive': True}}
        write(folder, f'ovisocr2-{precision}-calibration.json', calibration)
        write(folder, 'ovisocr2-quality-panel.json', {'cases': [
            *({'id': f'dev-{n}', 'split': 'development'} for n in range(4)),
            *({'id': f'held-{n}', 'split': 'heldout'} for n in range(4))]})
    return model, public, quality, calibration, decision


class PublicationGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.old = subject.EVIDENCE
        subject.EVIDENCE = self.folder

    def tearDown(self):
        subject.EVIDENCE = self.old
        self.temp.cleanup()

    def test_pinned_three_model_license_chain_and_negative_control(self):
        chain = subject.require_license_chain()
        self.assertEqual(chain['models']['ATH-MaaS/OvisOCR2']['license_sha256'],
                         subject.LICENSE_SHA)
        self.assertEqual(chain['models']['Qwen/Qwen3.5-0.8B-Base']['license_sha256'],
                         subject.ROOT_LICENSE_SHA)
        self.assertTrue(chain['license_text_comparison']['exact_normalized_text_equal'])
        self.assertFalse(chain['license_text_comparison']['raw_license_sha256_equal'])
        altered = self.folder/'changed-license-chain.json'
        altered.write_text(json.dumps({**chain, 'models': {}}))
        with mock.patch.object(subject, 'LICENSE_CHAIN_PATH', altered):
            with self.assertRaisesRegex(ValueError, 'license chain'):
                subject.require_license_chain()
            with self.assertRaisesRegex(ValueError, 'license chain'):
                subject.require_receipts('f16')

    def test_native_review_is_bound_to_exact_original_gpu_receipt(self):
        fixtures(self.folder)
        file = self.folder/'ovisocr2-vllm-reference.json'
        altered = json.loads(file.read_text())
        altered['producer']['different_reference'] = True
        write(self.folder, file.name, altered)
        with self.assertRaises(ValueError):
            subject.require_receipts('f16')

    def test_f16_positive_exact_receipts(self):
        fixtures(self.folder)
        pins = subject.require_receipts('f16')
        self.assertEqual(pins['candidate_sha256'], subject.F16_SHA)
        self.assertEqual(pins['model_report_artifact_id'], 1001)
        self.assertEqual(pins['model_reference_artifact_id'], 1002)

    def test_model_report_and_reference_artifacts_are_distinct(self):
        model, _, _, _, _ = fixtures(self.folder)
        model['reference_artifact'] = model['artifact']
        write(self.folder, 'ovisocr2-model-audit.json', model)
        with self.assertRaises(ValueError): subject.require_receipts('f16')

    def test_nonfinite_cache_and_wrong_canonical_producer_are_rejected(self):
        self.assertFalse(subject.valid_cache_error(float('nan')))
        self.assertFalse(subject.valid_cache_error(float('inf')))
        self.assertFalse(subject.valid_cache_error(.0021))
        self.assertTrue(subject.valid_cache_error(.002))
        model, _, _, _, _ = fixtures(self.folder)
        model['independent_review']['producer_canonical_sha256'] = HASH
        write(self.folder, 'ovisocr2-model-audit.json', model)
        with self.assertRaises(ValueError): subject.require_receipts('f16')
        model['producer']['original_capture']['cached_vs_full_prefix_max_abs'] = float('nan')
        write(self.folder, 'ovisocr2-model-audit.json', model)
        with self.assertRaises(ValueError): subject.require_receipts('f16')

    def test_missing_and_negative_review_fail_before_any_artifact(self):
        with self.assertRaises(ValueError): subject.require_receipts('f16')
        _, _, quality, _, decision = fixtures(self.folder)
        decision['independent_review']['all_attempts_completed'] = False
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaises(ValueError): subject.require_receipts('f16')
        decision['independent_review']['all_attempts_completed'] = True
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        quality['independent_review']['quality_positive'] = False
        write(self.folder, 'ovisocr2-f16-quality.json', quality)
        with self.assertRaises(ValueError): subject.require_receipts('f16')

    def test_quant_fails_on_failed_screen_or_missing_candidate_artifact(self):
        _, _, quality, _, decision = fixtures(self.folder, 'q8')
        self.assertEqual(subject.require_receipts('q8')['candidate_artifact_id'], 2002)
        quality['producer']['paired_documents']['f16']['screen_passed'] = False
        quality_sha = write(self.folder, 'ovisocr2-q8-quality.json', quality)
        decision['attempts']['q8']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaises(ValueError): subject.require_receipts('q8')

    def test_quant_numerical_failure_blocks_same_quality_text(self):
        _, _, quality, _, decision = fixtures(self.folder, 'q8')
        quality['producer']['quant_numerical']['passed'] = False
        quality_sha = write(self.folder, 'ovisocr2-q8-quality.json', quality)
        decision['attempts']['q8']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaises(ValueError): subject.require_receipts('q8')

    def test_f16_can_select_after_all_three_quant_attempts_rejected(self):
        _, _, _, _, decision = fixtures(self.folder)
        self.assertEqual(subject.require_receipts('f16')['precision'], 'f16')
        decision['attempts'].pop('q4')
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaises(ValueError): subject.require_receipts('f16')

    def test_calibrated_q4_requires_actual_development_imatrix_and_prior_q8(self):
        _, _, quality, calibration, decision = fixtures(self.folder, 'q4-calibrated')
        quality['producer']['calibration'] = {'vectors': [{} for _ in range(72)],
                                              'imatrix_sha256': HASH, 'stats_sha256': HASH,
                                              'input_ids': [f'dev-{n}' for n in range(4)]}
        quality_sha = write(self.folder, 'ovisocr2-q4-calibrated-quality.json', quality)
        decision['attempts']['q4-calibrated']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        calibration['producer'].update(quality_receipt_sha256=quality_sha, imatrix_sha256=HASH)
        write(self.folder, 'ovisocr2-q4-calibrated-calibration.json', calibration)
        q8_sha = write(self.folder, 'ovisocr2-q8-quality.json',
              {'independent_review': {'q8_diagnostic_complete': True,
                                      'allow_q4_diagnostic': True}})
        decision['attempts']['q8']['report_sha256'] = q8_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        self.assertEqual(subject.require_receipts('q4-calibrated')['recipe'], 'q4-attention-q8')
        calibration['producer']['imatrix_sha256'] = 'c'*64
        write(self.folder, 'ovisocr2-q4-calibrated-calibration.json', calibration)
        with self.assertRaises(ValueError): subject.require_receipts('q4-calibrated')

    def test_remote_artifact_service_identity(self):
        row = {'id': 1001, 'name': 'ovisocr2-model-reference-123', 'expired': False,
               'workflow_run': {'id': 123}, 'digest': 'sha256:'+HASH}
        call = lambda _: json.dumps(row).encode()
        self.assertEqual(subject.service_artifact(1001, 123, row['name'], HASH, call)['id'], 1001)
        with self.assertRaises(ValueError):
            subject.service_artifact(1001, 124, row['name'], HASH, call)

    def test_only_exact_selected_artifact_bytes_are_accepted(self):
        candidate = self.folder/'ovisocr2-f16.gguf'
        candidate.write_bytes(b'qualified fixture')
        expected = subject.digest(candidate)
        self.assertEqual(subject.find_artifact(self.folder, candidate.name,
                                               expected, candidate.stat().st_size), candidate)
        with self.assertRaises(ValueError):
            subject.find_artifact(self.folder, candidate.name, HASH, candidate.stat().st_size)
        with self.assertRaises(ValueError):
            subject.find_artifact(self.folder, 'ovisocr2-f32.gguf', expected, candidate.stat().st_size)

    def test_quality_service_accepts_only_direct_or_registered_metadata_workflow(self):
        quality = '.github/workflows/ovisocr2-quality.yml'
        metadata = '.github/workflows/ovisocr2-metadata.yml'
        row = {'id': 123, 'status': 'completed', 'conclusion': 'success',
               'head_sha': 'a' * 40, 'path': quality}
        for actual in (quality, metadata):
            row['path'] = actual
            result = subject.service_run(123, 'a' * 40, quality,
                                         lambda _: json.dumps(row).encode())
            self.assertEqual(result['path'], actual)
        row['path'] = '.github/workflows/other.yml'
        with self.assertRaises(ValueError):
            subject.service_run(123, 'a' * 40, quality, lambda _: json.dumps(row).encode())
        row['path'] = quality
        with self.assertRaises(ValueError):
            subject.service_run(123, 'a' * 40, metadata, lambda _: json.dumps(row).encode())
        row['path'] = metadata
        row['head_sha'] = 'b' * 40
        with self.assertRaises(ValueError):
            subject.service_run(123, 'a' * 40, quality, lambda _: json.dumps(row).encode())

    def test_public_and_quality_must_use_reviewed_vision_math(self):
        for filename, slot in (('ovisocr2-public-surfaces.json', 'selected_vision_preset'),
                               ('ovisocr2-f16-quality.json', 'selected_vision_preset')):
            fixtures(self.folder)
            path = self.folder / filename
            receipt = json.loads(path.read_text())
            receipt['producer'][slot] = {**SELECTION, 'name': 'baseline'}
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'review/source|phase/model'):
                subject.require_receipts('f16')

    def test_source_readme_recovery_and_tamper_rejection(self):
        payload = b'pinned source card'
        with mock.patch.object(subject, 'UPSTREAM_README_SHA', hashlib.sha256(payload).hexdigest()):
            calls = []
            def opener(url, **kwargs):
                calls.append(url)
                return io.BytesIO(payload)
            path = subject.pinned_upstream_readme(self.folder / 'empty', self.folder / 'scratch', opener)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(calls, [subject.UPSTREAM_README_URL])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'provenance altered'):
                subject.pinned_upstream_readme(path.parent, self.folder, opener)
            with self.assertRaisesRegex(ValueError, 'payload SHA'):
                subject.pinned_upstream_readme(self.folder / 'empty', self.folder,
                                               lambda *args, **kwargs: io.BytesIO(b'changed'))

    def test_card_placeholders_and_allowlist(self):
        pins = {'public_source_commit': HEAD, 'candidate_sha256': HASH,
                'candidate_bytes': 1_717_222_848, 'precision': 'f16', 'recipe': 'original-f16',
                'selected_vision_preset': SELECTION}
        rendered = subject.card(pins, 'ovisocr2-f16.gguf')
        self.assertIn(HEAD, rendered)
        self.assertNotIn('${', rendered)
        for name, value in SELECTION['values'].items():
            self.assertIn(name + '=' + value, rendered)
        with self.assertRaises(ValueError): subject.require_receipts('f32')


if __name__ == '__main__':
    unittest.main()
