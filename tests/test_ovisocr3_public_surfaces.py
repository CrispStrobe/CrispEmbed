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
