"""Pure guards for the pending engine 22 model-backed public audit."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'tools/ci-heavy/ovisocr3_public_surfaces.py'
spec = importlib.util.spec_from_file_location('ovis3_public', PATH)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class PublicControls(unittest.TestCase):
    def test_planned_native_and_quality_receipt_contracts(self):
        digest = 'a' * 64
        head = 'b' * 40
        names = ('f16-delta', 'f16-vision', 'f16-decoder-prepared',
                 'f16-decoder-native-vision', 'f16-decoder-native-positions',
                 'f16-decoder-native-frontend', 'f16-context')
        producer = {'source_commit': head, 'status': subject.NATIVE_STATUS,
                    'native14_positive': True, 'arm_count': 14, 'failures': [],
                    'commands': {name: {'exit_code': 0, 'passed': True,
                                        'comparison_coverage': True, 'log_sha256': digest}
                                 for name in names},
                    'exports': {'f16': {'model_sha256': digest, 'tensor_count': 473}},
                    'owned_readable': {'png_sha256': digest, 'rgb_sha256': digest},
                    'reference_artifacts': {'owned-readable.png': {'sha256': digest}},
                    'reference_original_json_sha256': digest,
                    'original_reference_producer_canonical_sha256': digest,
                    'hf_stop_runtime': {'actual_stops': [True, True, False]},
                    'native_environment': {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM': '1',
                                           'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN': '1',
                                           'CRISPEMBED_QWEN35_VISION_CASCADE_NORM': '0'},
                    'selected_vision_preset': {
                        'name': 'both',
                        'enabled_gates': ['CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                                          'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],
                        'all_other_native_gates_unchanged': True,
                        'no_production_default_promoted': True},
                    'original_capture': {'decoded': ['Hello OCR\n12345'],
                                         'generated_ids': [[10, 248046]],
                                         'generation_result': {'max_new_tokens': 12,
                                                               'eos_ids': [248044, 248046],
                                                               'generated_token_count': 2},
                                         'observational_control': {'prefill_logits_exact': True,
                                                                   'greedy_ids_exact': True},
                                         'cached_vs_full_prefix_same_argmax': True,
                                         'cached_vs_full_prefix_max_abs': .001}}
        canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        native = {'run': 123, 'head_sha': head, 'artifact_id': 456,
                  'artifact_zip_sha256': digest,
                  'f16_artifact': {'id': 789, 'name': 'ovisocr3-native14-f16-123', 'bytes': 1000,
                                   'zip_sha256': digest},
                  'producer': producer, 'independent_review': {
                      'all_fourteen_native_arms_positive': True,
                      'producer_canonical_sha256': canonical,
                      'f16_artifact_zip_sha256': digest}}
        quality_producer = {'status': 'completed diagnostic OvisOCR3 F16 eleven-case quality',
                            'qualified': True, 'f16_model_sha256': digest,
                            'native_cases': {str(i): {} for i in range(11)},
                            'scores': {'all_complete': True}, 'paired': {'all_exact': True}}
        quality_canonical = hashlib.sha256(json.dumps(quality_producer, sort_keys=True,
                                                      separators=(',', ':'), ensure_ascii=True,
                                                      allow_nan=False).encode()).hexdigest()
        quality = {'run': 234, 'head_sha': head, 'producer': quality_producer,
                   'independent_review': {'producer_canonical_sha256': quality_canonical,
                                          'f16_quality_positive': True}}
        overrides = {'NATIVE_RUN': 123, 'NATIVE_HEAD': head,
                     'NATIVE_ARTIFACT_NAME': 'ovisocr3-native14-f16-123',
                     'NATIVE_PRODUCER_SHA': canonical, 'F16_SHA': digest,
                     'QUALITY_RUN': 234, 'QUALITY_HEAD': head,
                     'QUALITY_PRODUCER_SHA': quality_canonical,
                     'QUALITY_REVIEW_KEY': 'f16_quality_positive'}
        with mock.patch.multiple(subject, **overrides):
            pins = subject.require_receipt(native)
            self.assertEqual(pins['model_sha256'], digest)
            self.assertEqual(pins['selected_vision_preset']['name'], 'both')
            self.assertTrue(subject.require_quality_receipt(quality, pins)['eleven_complete'])
            native['producer']['commands']['f16-vision']['comparison_coverage'] = False
            with self.assertRaises(ValueError): subject.require_receipt(native)
            native['producer']['commands']['f16-vision']['comparison_coverage'] = True
            quality['producer']['qualified'] = False
            with self.assertRaises(ValueError): subject.require_quality_receipt(quality, pins)

    def test_separate_large_and_small_artifact_service_identity(self):
        head = 'b' * 40
        pins = {'run': 123, 'source_commit': head, 'artifact_id': 789,
                'artifact_zip_sha256': 'a' * 64, 'artifact_bytes': 1000,
                'report_artifact_id': 456, 'report_artifact_zip_sha256': 'c' * 64}
        large = {'id': 789, 'name': 'ovisocr3-native14-f16-123', 'size_in_bytes': 1000,
                 'digest': 'sha256:' + 'a' * 64, 'expired': False,
                 'workflow_run': {'id': 123, 'head_sha': head}}
        small = {'id': 456, 'name': 'ovisocr3-native14-reports-123',
                 'digest': 'sha256:' + 'c' * 64, 'expired': False,
                 'workflow_run': {'id': 123, 'head_sha': head}}
        with mock.patch.object(subject, 'NATIVE_ARTIFACT_NAME', large['name']):
            self.assertEqual(subject.artifact_service_gate(
                pins, lambda _: json.dumps(large).encode())['id'], 789)
            self.assertEqual(subject.report_artifact_service_gate(
                pins, lambda _: json.dumps(small).encode())['id'], 456)
            for change in ({**large, 'size_in_bytes': 999},
                           {**large, 'digest': 'sha256:' + 'd' * 64},
                           {**large, 'expired': True}):
                with self.assertRaises(ValueError):
                    subject.artifact_service_gate(pins, lambda _, row=change: json.dumps(row).encode())

    def test_future_receipts_are_required_before_weights(self):
        self.assertEqual(subject.NATIVE_RUN, 0)
        self.assertEqual(subject.QUALITY_RUN, 0)
        with self.assertRaisesRegex(ValueError, 'native model pins are unresolved'):
            subject.require_receipt({})
        with self.assertRaisesRegex(ValueError, 'F16 quality pins are unresolved'):
            subject.require_quality_receipt({}, {'model_sha256': 'a' * 64})
        self.assertEqual(subject.RECEIPT.name, 'ovisocr3-native-model.json')
        self.assertEqual(subject.QUALITY_RECEIPT.name, 'ovisocr3-f16-quality.json')

    def test_three_separate_gates(self):
        self.assertEqual(subject.GATES, ('CRISPEMBED_QWEN35_OCR',
                                         'CRISPEMBED_OVISOCR3_FRONTEND',
                                         'CRISPEMBED_OVISOCR3_INFERENCE'))
        self.assertEqual(len(set(subject.GATES)), 3)
        self.assertNotIn('CRISPEMBED_OVISOCR2_FRONTEND', subject.GATES)
        base = {'CRISPEMBED_OVISOCR2_FRONTEND': '0',
                'CRISPEMBED_OVISOCR2_INFERENCE': '0',
                'CRISPEMBED_OVISOCR3_FRONTEND': '1'}
        for label in ('cli-stage21-ovis3-profile', 'http-start-ovis3-on-stage21'):
            probe = subject.profile_probe_env(label, base)
            self.assertEqual((probe['CRISPEMBED_OVISOCR2_FRONTEND'],
                              probe['CRISPEMBED_OVISOCR2_INFERENCE']), ('1', '1'))
            self.assertEqual(probe['CRISPEMBED_OVISOCR3_FRONTEND'], '1')
            self.assertEqual(base['CRISPEMBED_OVISOCR2_FRONTEND'], '0')
        for label in ('cli-stage22-lighton-profile', 'http-start-lighton-profile',
                      'cli-stage22-custom-prompt'):
            self.assertEqual(subject.profile_probe_env(label, base), base)

    def test_format3_truncation_is_header_complete(self):
        class Field:
            def __init__(self, value): self.value = value
            def contents(self): return self.value
        class Reader:
            def __init__(self, _):
                self.tensors = [None] * 473
                self.fields = {'qwen35_ocr.format_version': Field(3),
                               'qwen35_ocr.profile': Field('ovisocr3')}
                self.data_offset = 64
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'model.gguf'
            target = Path(directory) / 'truncated.gguf'
            source.write_bytes(bytes(range(256)) * 8)
            with mock.patch.dict('sys.modules', {'gguf': types.SimpleNamespace(GGUFReader=Reader)}):
                result = subject.metadata_complete_truncation(source, target)
            self.assertEqual(result['retained_tensor_bytes'], 64)
            self.assertEqual(target.read_bytes(), source.read_bytes()[:128])
            Reader.__init__ = lambda self, _: setattr(self, 'tensors', []) or setattr(self, 'fields', {}) or setattr(self, 'data_offset', 64)
            with mock.patch.dict('sys.modules', {'gguf': types.SimpleNamespace(GGUFReader=Reader)}):
                with self.assertRaises((ValueError, KeyError)):
                    subject.metadata_complete_truncation(source, Path(directory) / 'bad.gguf')

    def test_two_stage_cache_requires_second_load_failure(self):
        target = Path('/hosted/ovisocr3-truncated.gguf')
        trace = {'stages_tried': 2, 'metrics': [
            {'index': 1, 'engine': 'ovisocr3', 'accepted': False, 'text_chars': 3},
            {'index': 2, 'engine': 'ovisocr3', 'accepted': False, 'text_chars': 0}],
            'fallback_text': 'abc'}
        log = ('truncated/corrupt GGUF ' + str(target) + '\n'
               'ocr_orchestrator: qwen35/ovis load failed\n'
               'RESULT stage-cache ' + json.dumps(trace) + '\n')
        result = subject.validate_stage_cache_probe({'exit_code': 0}, log, 'abc', target)
        self.assertTrue(result['second_weight_load_failed'])
        with self.assertRaises(ValueError):
            subject.validate_stage_cache_probe({'exit_code': 0}, log.replace('"text_chars": 0', '"text_chars": 3'), 'abc', target)

    def test_confidence_and_abi_negative_controls(self):
        arms = {'c-abi': {'passed': True, 'confidence_count': 2, 'mean_confidence': .75},
                'python': {'passed': True, 'confidence_count': 2, 'mean_confidence': .75},
                'cli-stage22': {'passed': True, 'mean_confidence': .75},
                'http-stage22': {'passed': True, 'mean_confidence': .75},
                'rust-run': {'passed': True, 'metric_lines': [
                    'PASS Rust unified RGB cap=12 prompt-rejection text_bytes=3 confidence_count=2 mean=0.75']},
                'dart-run': {'passed': True, 'metric_lines': [
                    'PASS Dart unified RGB cap=12 prompt-rejection text_bytes=3 confidence_count=2 mean=0.75']}}
        self.assertEqual(subject.compare_confidences(arms)['confidence_count'], 2)
        arms['cli-stage22']['mean_confidence'] += .005
        with self.assertRaises(ValueError):
            subject.compare_confidences(arms)
        native = 'ABI stage_size=168 cleanup_size=40 params_size=88 vlm_max_tokens_offset=136 vlm_prompt_offset=144'
        python = {'stage_size': 168, 'cleanup_size': 40, 'params_size': 88,
                  'vlm_max_tokens_offset': 136, 'vlm_prompt_offset': 144}
        dart = ['ABI stage_size=168 cleanup_size=40 params_size=88']
        self.assertEqual(subject.compare_abi(native, dart, python)['dart']['stage_size'], 168)
        with self.assertRaises(ValueError):
            subject.compare_abi(native, ['ABI stage_size=160 cleanup_size=40 params_size=88'], python)

    def test_new_binding_sources_use_engine22(self):
        rust = (ROOT / 'tests/ovisocr3_public_bindings.rs').read_text()
        dart = (ROOT / 'tests/ovisocr3_public_bindings.dart').read_text()
        cache = (ROOT / 'tools/ci-heavy/ovisocr3_stage_cache_probe.py').read_text()
        self.assertIn('engine: 22', rust)
        self.assertIn('engine: 22', dart)
        self.assertIn('"engine": 22', cache)
        for source in (rust, dart, cache):
            self.assertNotIn('engine: 21', source)
        self.assertEqual(hashlib.sha256(PATH.read_bytes()).hexdigest(), subject.sha(PATH))


if __name__ == '__main__':
    unittest.main()
