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
    def test_f16_quality_service_requires_parent_metadata_dispatch(self):
        run = 40187405735
        head = 'ee719aef2ac22eee67ab6c483a4d1d6633c901c8'
        result = {'id': run, 'status': 'completed', 'conclusion': 'success',
                  'head_sha': head, 'path': '.github/workflows/ovisocr2-metadata.yml'}
        def call(_, row=result):
            return json.dumps(row).encode()
        self.assertEqual(subject.quality_service_gate(run, head, call), result)
        for changed in (
            {'path': '.github/workflows/ovisocr3-quality-f16.yml'},
            {'status': 'in_progress'},
            {'conclusion': 'failure'},
            {'head_sha': '0' * 40},
            {'id': run + 1},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'service run'):
                subject.quality_service_gate(run, head, lambda _, row={**result, **changed}:
                                             json.dumps(row).encode())

    def test_source_quality_diagnostic_is_exact_and_never_qualifies_model(self):
        contract = subject.vision_contract()  # Shared gate remains source-only.
        self.assertIn('both', contract.PRESETS)
        policy_spec = importlib.util.spec_from_file_location('ovis3_diagnostic_test_policy',
            ROOT / 'tools/ci-heavy/ovisocr3_quality_contract.py')
        policy = importlib.util.module_from_spec(policy_spec)
        policy_spec.loader.exec_module(policy)
        panel = policy.load_panel()
        digest, head = 'a' * 64, 'b' * 40
        inputs, original_cases = [], {}
        for spec in panel['cases']:
            name, cap = spec['id'], spec['cap']
            rgb, png = spec.get('rgb_sha256', digest), spec.get('png_sha256', digest)
            inputs.append({'id': name, 'split': spec['split'], 'cap': cap,
                           'rgb_sha256': rgb, 'png_sha256': png})
            gold = spec['gold'] if 'gold' in spec else json.loads((ROOT / spec['gold_file']).read_text())['text']
            capped = name == 'two-column-page'
            ids = [42] * cap if capped else [42, 248046]
            original_cases[name] = {'rgb_sha256': rgb, 'png_sha256': png, 'cap': cap,
                                    'token_ids': ids, 'raw_text': 'repeat' if capped else gold,
                                    'stop_id': None if capped else 248046,
                                    'stop_kind': 'length' if capped else 'eos',
                                    'confidences': [.75] * len(ids), 'mean_confidence': .75}
        original_producer = {'source_commit': head,
            'status': 'completed original OvisOCR3 HF CPU/F32 eleven-case plain OCR quality',
            'checkpoint_sha256': 'aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600',
            'panel_canonical_sha256': policy.PANEL_CANONICAL_SHA256,
            'completed_case_count': 11, 'inputs': inputs, 'original_cases': original_cases,
            'scores': policy.score_panel(inputs, original_cases)}
        native_cases = json.loads(json.dumps(original_cases))
        quality_producer = {'source_commit': head,
            'status': 'completed diagnostic OvisOCR3 F16 eleven-case quality',
            'qualified': False, 'failures': [], 'f16_model_sha256': digest,
            'inputs': inputs, 'native_cases': native_cases,
            'commands': {name: {'exit_code': 0, 'log_sha256': digest} for name in original_cases},
            'scores': policy.score_panel(inputs, native_cases),
            'paired': policy.paired_screen(original_cases, native_cases)}
        def canonical(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        oc = canonical(original_producer)
        quality_producer['prerequisites'] = {'original_quality_canonical_sha256': oc,
                                             'native_model_canonical_sha256': digest}
        qc = canonical(quality_producer)
        original = {'run': 100, 'head_sha': head, 'artifact_id': 101,
            'artifact_zip_sha256': digest, 'producer': original_producer,
            'independent_review': {'producer_canonical_sha256': oc,
                'artifact_zip_sha256': digest, 'original_eleven_capture_complete': True,
                'source_completion_incomplete': True, 'original_eleven_complete': False,
                'all_eleven_reached_eos': False}}
        quality = {'run': 200, 'head_sha': head, 'artifact_id': 201,
            'artifact_zip_sha256': digest, 'producer': quality_producer,
            'independent_review': {'producer_canonical_sha256': qc,
                'artifact_zip_sha256': digest, 'f16_eleven_capture_complete': True,
                'paired_exact': True, 'source_completion_incomplete': True,
                'f16_quality_positive': False}}
        model = {'model_sha256': digest, 'producer_canonical_sha256': digest}
        pins = {'DIAGNOSTIC_ORIGINAL_RUN': 100, 'DIAGNOSTIC_ORIGINAL_HEAD': head,
                'DIAGNOSTIC_ORIGINAL_PRODUCER_SHA': oc, 'QUALITY_RUN': 200,
                'QUALITY_HEAD': head, 'QUALITY_PRODUCER_SHA': qc,
                'DIAGNOSTIC_QUALITY_REVIEW_KEY': 'f16_eleven_capture_complete'}
        with mock.patch.multiple(subject, **pins):
            result = subject.require_diagnostic_quality_receipts(original, quality, model)
            self.assertEqual(result['source_completion_debt'], ['two-column-page'])
            self.assertTrue(result['paired_ids_raw_stops_exact'])
            self.assertFalse(result['model_qualified'])
            self.assertFalse(result['publication_eligible'])
            def artifact_call(args):
                artifact_id = int(args[-1].split('/')[-1])
                run = 100 if artifact_id == 101 else 200
                label = 'original-quality' if artifact_id == 101 else 'f16-quality'
                return json.dumps({'id': artifact_id, 'name': f'ovisocr3-{label}-{run}',
                    'digest': 'sha256:' + digest, 'expired': False,
                    'workflow_run': {'id': run}}).encode()
            self.assertEqual(set(subject.diagnostic_artifact_service_gate(result, artifact_call)),
                             {'original', 'f16'})
            bad_pins = dict(result, quality_artifact_zip_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'artifact service'):
                subject.diagnostic_artifact_service_gate(bad_pins, artifact_call)
            with mock.patch.object(subject, 'QUALITY_REVIEW_KEY', 'f16_quality_positive'):
                with self.assertRaises(ValueError):
                    subject.require_quality_receipt(quality, model)
            changed = json.loads(json.dumps(quality))
            changed['producer']['native_cases']['dev-hello']['token_ids'] = [99, 248046]
            with self.assertRaises(ValueError):
                subject.require_diagnostic_quality_receipts(original, changed, model)
            changed = json.loads(json.dumps(quality))
            changed['producer']['native_cases']['two-column-page']['stop_id'] = 248046
            with self.assertRaises(ValueError):
                subject.require_diagnostic_quality_receipts(original, changed, model)
            changed = json.loads(json.dumps(original))
            changed['independent_review']['original_eleven_complete'] = True
            with self.assertRaises(ValueError):
                subject.require_diagnostic_quality_receipts(changed, quality, model)
            changed = json.loads(json.dumps(quality))
            changed['independent_review']['f16_quality_positive'] = True
            with self.assertRaises(ValueError):
                subject.require_diagnostic_quality_receipts(original, changed, model)

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
                    'native_environment': subject.reviewed_native_gate_values(),
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
        with mock.patch.multiple(subject, **overrides), \
             mock.patch.object(subject, 'native_source_closure',
                               return_value={'file_count': 29}) as closure:
            pins = subject.require_receipt(native)
            self.assertEqual(pins['model_sha256'], digest)
            self.assertEqual(pins['native_source_closure']['file_count'], 29)
            closure.assert_called_once_with(native['producer'], head)
            with mock.patch.object(subject, 'native_source_closure',
                                   side_effect=ValueError('changed native norm')):
                with self.assertRaisesRegex(ValueError, 'changed native norm'):
                    subject.require_receipt(native)
            self.assertEqual(pins['selected_vision_preset']['name'], 'both')
            self.assertTrue(subject.require_quality_receipt(quality, pins)['eleven_complete'])
            native['producer']['commands']['f16-vision']['comparison_coverage'] = False
            with self.assertRaises(ValueError): subject.require_receipt(native)
            native['producer']['commands']['f16-vision']['comparison_coverage'] = True
            native['producer']['native_environment'] = {
                **subject.reviewed_native_gate_values(), 'CRISPEMBED_QWEN35_QUANT': '1'}
            changed_canonical = hashlib.sha256(json.dumps(native['producer'], sort_keys=True,
                separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
            native['independent_review']['producer_canonical_sha256'] = changed_canonical
            with mock.patch.object(subject, 'NATIVE_PRODUCER_SHA', changed_canonical), \
                 self.assertRaisesRegex(ValueError, 'Model producer status/source differs'):
                subject.require_receipt(native)
            native['producer']['native_environment'] = subject.reviewed_native_gate_values()
            native['independent_review']['producer_canonical_sha256'] = canonical
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
        with mock.patch.object(subject, 'NATIVE_RUN', 0), \
             mock.patch.object(subject, 'QUALITY_RUN', 0):
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

    def test_public_child_and_inprocess_env_override_ambient_native_gates(self):
        reviewed = subject.reviewed_native_gate_values()
        poisoned = {name: '0' if value == '1' else '1' for name, value in reviewed.items()}
        poisoned['UNRELATED_TEST_SENTINEL'] = 'preserved'
        selection = {'name': 'both',
                     'enabled_gates': ['CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                                       'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],
                     'values': {name: reviewed[name] for name in (
                         'CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                         'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
                         'CRISPEMBED_QWEN35_VISION_CASCADE_NORM')},
                     'all_other_native_gates_unchanged': True,
                     'no_production_default_promoted': True}
        with mock.patch.dict(subject.os.environ, poisoned, clear=True):
            env = subject.env_with_gates(selection=selection)
            self.assertEqual({name: env[name] for name in reviewed}, reviewed)
            self.assertEqual(env['UNRELATED_TEST_SENTINEL'], 'preserved')
            self.assertEqual(env['CRISPEMBED_OVISOCR2_LARGE_IMAGES'], '0')
            self.assertEqual(env['CRISPEMBED_QWEN35_QUANT'], '0')
            subject.apply_inprocess_native_gates(env)
            self.assertEqual({name: subject.os.environ[name] for name in reviewed}, reviewed)
            disabled = subject.env_with_gates(enabled=False, selection=selection)
            self.assertEqual([disabled[name] for name in subject.GATES], ['0', '0', '0'])
            probe = subject.profile_probe_env('cli-stage21-ovis3-profile', env)
            self.assertEqual([probe[name] for name in ('CRISPEMBED_OVISOCR2_FRONTEND',
                                                        'CRISPEMBED_OVISOCR2_INFERENCE')], ['1', '1'])
            self.assertEqual([env[name] for name in ('CRISPEMBED_OVISOCR2_FRONTEND',
                                                     'CRISPEMBED_OVISOCR2_INFERENCE')], ['0', '0'])

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
