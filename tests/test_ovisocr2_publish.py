"""Pure publication qualification tests. No hub client, auth or weights."""
import importlib.util
import copy
import fnmatch
import io
import hashlib
import json
import os
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


def refresh_producer_review(receipt):
    receipt['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
        receipt['producer'], sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def tensor_proof(recipe):
    contract = subject.quant_tensor_contract()
    schema = contract.source_schema()
    kind, overrides = contract.RECIPES[recipe]
    before, after = {}, {}
    for name, shape in schema.items():
        old = {'shape': list(reversed(shape)), 'type': 'F16', 'sha256': HASH}
        before[name] = old
        if contract._protected(name, shape):
            after[name] = dict(old)
            continue
        chosen = kind.upper()
        for rule in overrides:
            pattern, precision = rule.rsplit('=', 1)
            if fnmatch.fnmatchcase(name, pattern):
                chosen = precision.upper()
        after[name] = {**old, 'type': chosen, 'sha256': 'c'*64}
    return {'before': before, 'after': after}, contract.validate_quant_tensors(
        before, after, recipe)


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
    inputs = [{'id': f'case-{index}'} for index in range(11)]
    quality = {'run': 456, 'source_commit': HEAD, 'artifact_id': 3001,
        'artifact_name': 'ovisocr2-quality-reports-456', 'artifact_zip_sha256': HASH,
        'report_sha256': HASH, 'producer': {
        'status': ('completed F16 quality observations; independent review pending'
                   if precision == 'f16' else 'completed quantized quality observations; independent review pending'),
        'requested_phase': precision, 'source_commit': HEAD,
        'selected_vision_preset': SELECTION,
        'inputs': inputs,
        'prerequisite': {'baseline_sha256': model_sha, 'selected_vision_preset': SELECTION,
                         'original_review_sha256': original_sha},
        'models': {precision: {'sha256': candidate_sha, 'bytes': 1_717_222_848,
                               **({} if precision == 'f16' else {'recipe': subject.RECIPES[precision]})}},
        'scores': {'original': {}, 'f16': {}, precision: {}},
        'native_cases': {'f16': native, **({} if precision == 'f16' else {precision: native})},
        'cabi_smoke': {precision: {'text': 'Hello'}},
        'f16_paired_synthetic': {'screen_passed': True},
        'f16_paired_documents': {'screen_passed': True},
        **({} if precision == 'f16' else {
            'tensor_audit': {'recipe': subject.RECIPES[precision], 'tensor_count': 473,
                             'protected_byte_exact': True, 'changed_matrix_count': 100},
            'quant_numerical': {'passed': True, 'exit_code': 0, 'comparison_coverage': True,
                                'first_failure': None, 'source_reference_sha256': HASH,
                                'log_sha256': HASH, 'metric_lines': ['PASS generated.scores.0']},
            'paired_synthetic': {'screen_passed': True},
            'paired_synthetic_original': {'screen_passed': True},
            'paired_documents': {'original': {'screen_passed': True},
                                 'f16': {'screen_passed': True}}})},
        'independent_review': {'quality_positive': True, 'f16_11_complete': precision == 'f16',
                               'artifact_zip_sha256': HASH,
                               **({} if precision == 'f16' else {'numerical_positive': True})}}
    if precision != 'f16':
        def probe(label, model_sha, gate):
            return {'passed': True, 'exit_code': 0, 'comparison_coverage': True,
                    'first_failure': None, 'model_sha256': model_sha,
                    'quant_gate': gate, 'source_reference_sha256': HASH,
                    'log': f'{precision}-{label}.log', 'log_sha256': HASH,
                    'raw_diff_manifest': f'{precision}-{label}-raw-manifest.json',
                    'raw_diff_manifest_sha256': HASH,
                    'raw_diff_coverage_complete': True,
                    'metric_lines': ['PASS prefill.linear.0.qkv']}
        quality['producer']['quant_first_linear_control'] = probe(
            'f16-first-linear-control', subject.F16_SHA, '0')
        quality['producer']['quant_first_linear'] = probe(
            'first-linear-diff', candidate_sha, '1')
        quality['producer']['quant_numerical'].update(probe(
            'official-numerical-diff', candidate_sha, '1'))
        quality['producer']['quant_numerical']['metric_lines'] = ['PASS generated.scores.0']
        quality['producer']['quant_raw_complete'] = True
        records, audit = tensor_proof(subject.RECIPES[precision])
        quality['producer']['tensor_records'] = records
        quality['producer']['tensor_audit'] = audit
        first_producer = json.loads(json.dumps(quality['producer']))
        first_producer['requested_phase'] = 'f16'
        first_producer['status'] = 'completed F16 quality observations; independent review pending'
        first_producer['models'] = {'f16': {'sha256': subject.F16_SHA, 'bytes': 1_717_222_848}}
        first_producer['native_cases'] = {'f16': native}
        first_producer['scores'] = {'original': {}, 'f16': {}}
        first_producer.pop('tensor_audit', None)
        first_producer.pop('tensor_records', None)
        first_producer.pop('quant_numerical', None)
        first_producer.pop('quant_first_linear_control', None)
        first_producer.pop('quant_first_linear', None)
        first_producer.pop('quant_raw_complete', None)
        first_producer.pop('paired_synthetic', None)
        first_producer.pop('paired_synthetic_original', None)
        first_producer.pop('paired_documents', None)
        first = {'run': 455, 'source_commit': HEAD, 'artifact_id': 3000,
                 'artifact_name': 'ovisocr2-quality-reports-455',
                 'artifact_zip_sha256': HASH, 'report_sha256': HASH,
                 'producer': first_producer,
                 'independent_review': {'quality_positive': True, 'f16_11_complete': True,
                                        'artifact_zip_sha256': HASH}}
        first['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
            first_producer, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
            allow_nan=False).encode()).hexdigest()
        first_sha = write(folder, 'ovisocr2-f16-quality.json', first)
        quality['producer']['first_f16_baseline'] = {'run': 455, 'source_commit': HEAD,
            'artifact_zip_sha256': HASH, 'report_sha256': HASH,
            'producer_canonical_sha256': first['independent_review']['producer_canonical_sha256'],
            'receipt_sha256': first_sha, 'case_count': 11}
        quality['producer']['prerequisite']['first_f16_review_sha256'] = first_sha
        quality['producer']['scores']['f16_host_control'] = {}
        quality['producer']['f16_host_control'] = {'native_case_count': 11,
            'paired_synthetic': {'screen_passed': True},
            'paired_documents': {'screen_passed': True}}
    quality['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
        quality['producer'], sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    if precision != 'f16':
        quality['candidate_artifact'] = {'id': 2002, 'zip_sha256': HASH}
        quality['independent_review']['candidate_artifact_zip_sha256'] = HASH
        quality['raw_diff_artifact'] = {
            'id': 2003, 'name': 'ovisocr2-quality-native-diff-456', 'zip_sha256': HASH}
        quality['independent_review']['raw_diff_verified'] = True
        quality['independent_review']['raw_diff_artifact_zip_sha256'] = HASH
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


def short_quant_fixtures(folder):
    _, _, quality, _, decision = fixtures(folder, 'q8')
    first_path = folder/'ovisocr2-f16-quality.json'
    first = json.loads(first_path.read_text())
    first_producer = first['producer']
    case = {'id': 'dev-hello', 'cap': 12, 'rgb_sha256': '1'*64,
            'png_sha256': '2'*64}
    ids = [9419, 77796, 271, 16, 17, 18, 19, 20, 248046]
    frozen = {'input_rgb_sha256': case['rgb_sha256'],
              'input_png_sha256': case['png_sha256'], 'cap': 12,
              'token_ids': ids, 'confidences': [.9]*9,
              'mean_confidence': .9, 'raw_text': 'Hello OCR\n\n12345',
              'processed_text': 'Hello OCR\n\n12345',
              'stop_id': 248046, 'finish_reason': 'stop'}
    first_producer['inputs'][0] = case
    first_producer['native_cases']['f16'].pop('case-0')
    first_producer['native_cases']['f16']['dev-hello'] = frozen
    native_source = {'src/qwen35_vision.cpp': '3'*64}
    first_producer['native_source_sha256'] = native_source
    first_producer['source_sha256'] = {'tools/ci-heavy/ovisocr2_quality_run.py': '4'*64}
    first_producer['versions'] = {'torch': '2.11.0', 'gguf': '0.19.0'}
    first_producer['cpu_runtime'] = {'machine': 'x86_64', 'cpu_model': 'first CPU',
                                     'thread_env': {'OMP_NUM_THREADS': '2',
                                                    'MKL_NUM_THREADS': '2',
                                                    'OPENBLAS_NUM_THREADS': '2'}}
    first['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
        first_producer, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    first_sha = write(folder, first_path.name, first)
    producer = quality['producer']
    producer['inputs'][0] = case
    producer['native_cases']['q8'].pop('case-0')
    producer['native_cases']['q8']['dev-hello'] = {'token_ids': ids,
        'confidences': [.9]*9, 'cap': 12}
    actual = {**frozen, 'confidences': [.91]*9, 'mean_confidence': .91,
              'wall_seconds': 2.5, 'child_peak_rss_kib_upper_bound': 500000}
    producer['native_cases']['f16'] = {'dev-hello': actual}
    producer['native_source_sha256'] = native_source
    producer['versions'] = dict(first_producer['versions'])
    producer['cpu_runtime'] = {**first_producer['cpu_runtime'], 'cpu_model': 'second CPU'}
    producer['prerequisite']['first_f16_review_sha256'] = first_sha
    producer['first_f16_baseline'].update(
        receipt_sha256=first_sha,
        producer_canonical_sha256=first['independent_review']['producer_canonical_sha256'])
    producer['scores'].pop('f16_host_control')
    producer['f16_paired_source'] = (
        'reviewed first F16 eleven-case receipt, recomputed on unchanged originals')
    driver = subject.quality_driver_contract()
    producer['f16_host_control'] = driver.validate_short_f16_host_control(actual, frozen, case)
    producer['f16_source_closure'] = {
        'first_f16_source_commit': HEAD, 'candidate_source_commit': HEAD,
        'paths': {name: {'first_oid': '5'*40, 'candidate_oid': '5'*40}
                  for name in driver.F16_SOURCE_CLOSURE_PATHS},
        'ast_contracts': {name: {'first_sha256': '6'*64, 'candidate_sha256': '6'*64}
                          for name in driver.F16_AST_CONTRACTS},
        'first_native_source_sha256': native_source,
        'first_runner_source_sha256': '4'*64}
    producer['f16_runtime_closure'] = driver.f16_runtime_closure(first_producer, producer)
    quality['independent_review']['short_host_control_positive'] = True
    quality['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
        producer, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    quality_sha = write(folder, 'ovisocr2-q8-quality.json', quality)
    decision['attempts']['q8']['report_sha256'] = quality_sha
    write(folder, 'ovisocr2-quant-attempts.json', decision)
    return quality, decision


class PublicationGate(unittest.TestCase):
    def test_short_host_control_requires_exact_source_review_and_frozen_case(self):
        quality, decision = short_quant_fixtures(self.folder)
        self.assertEqual(subject.require_receipts('q8')['precision'], 'q8')
        for change in ('src', 'ggml', 'cmake-ast', 'runtime-threads', 'wrong-ids', 'wrong-text',
                       'wrong-eos', 'frozen-score', 'fake-eleven-score', 'review'):
            altered = json.loads(json.dumps(quality))
            producer = altered['producer']
            if change in ('src', 'ggml'):
                producer['f16_source_closure']['paths'][change]['candidate_oid'] = '7'*40
            elif change == 'cmake-ast':
                producer['f16_source_closure']['ast_contracts']['cmake-configure'][
                    'candidate_sha256'] = '7'*64
            elif change == 'runtime-threads':
                producer['cpu_runtime']['thread_env']['OMP_NUM_THREADS'] = '8'
            elif change == 'wrong-ids':
                producer['native_cases']['f16']['dev-hello']['token_ids'][-1] = 248044
            elif change == 'wrong-text':
                producer['native_cases']['f16']['dev-hello']['raw_text'] = 'wrong'
            elif change == 'wrong-eos':
                producer['native_cases']['f16']['dev-hello']['stop_id'] = 248044
            elif change == 'frozen-score':
                producer['scores']['f16'] = {'changed': True}
            elif change == 'fake-eleven-score':
                producer['scores']['f16_host_control'] = {}
            else:
                altered['independent_review']['short_host_control_positive'] = False
            altered['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
                producer, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                allow_nan=False).encode()).hexdigest()
            changed_sha = write(self.folder, 'ovisocr2-q8-quality.json', altered)
            changed_decision = json.loads(json.dumps(decision))
            changed_decision['attempts']['q8']['report_sha256'] = changed_sha
            write(self.folder, 'ovisocr2-quant-attempts.json', changed_decision)
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.require_receipts('q8')

    def test_quant_cannot_substitute_host_f16_for_first_reviewed_f16(self):
        fixtures(self.folder, 'q8')
        self.assertEqual(subject.require_receipts('q8')['first_f16_run_id'], 455)
        quality_path = self.folder/'ovisocr2-q8-quality.json'
        original = quality_path.read_bytes()
        for mutate in (
                lambda p: p['first_f16_baseline'].update(receipt_sha256=HASH),
                lambda p: p['scores'].update(f16={'host-only': True}),
                lambda p: p['paired_synthetic_original'].update(screen_passed=False),
                lambda p: p.pop('f16_host_control')):
            quality_path.write_bytes(original)
            data = json.loads(original)
            mutate(data['producer'])
            quality_path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                subject.require_receipts('q8')
        quality_path.write_bytes(original)
        first_path = self.folder/'ovisocr2-f16-quality.json'
        first = json.loads(first_path.read_text())
        first['independent_review']['f16_11_complete'] = False
        first_path.write_text(json.dumps(first))
        with self.assertRaises(ValueError):
            subject.require_receipts('q8')

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

    def test_quant_raw_probes_fail_on_control_candidate_and_source_drift(self):
        model, _, quality, _, _ = fixtures(self.folder, 'q8')
        self.assertEqual(subject.checked_quant_raw_proofs(
            quality, model['producer'], 'q8', HASH)['id'], 2003)
        changes = (
            ('f16 negative', 'quant_first_linear_control', 'passed', False),
            ('candidate negative', 'quant_first_linear', 'passed', False),
            ('full decoder negative', 'quant_numerical', 'passed', False),
            ('f16 wrong model', 'quant_first_linear_control', 'model_sha256', HASH),
            ('candidate wrong model', 'quant_first_linear', 'model_sha256', subject.F16_SHA),
            ('f16 wrong gate', 'quant_first_linear_control', 'quant_gate', '1'),
            ('candidate wrong gate', 'quant_first_linear', 'quant_gate', '0'),
            ('missing raw', 'quant_first_linear', 'raw_diff_coverage_complete', False),
            ('wrong source', 'quant_numerical', 'source_reference_sha256', '0'*64),
            ('missing manifest SHA', 'quant_first_linear_control',
             'raw_diff_manifest_sha256', None),
            ('missing log SHA', 'quant_first_linear', 'log_sha256', None),
        )
        for label, probe, key, value in changes:
            altered = copy.deepcopy(quality)
            altered['producer'][probe][key] = value
            with self.subTest(label=label), self.assertRaisesRegex(ValueError,
                    'Reviewed quant raw probe differs'):
                subject.checked_quant_raw_proofs(altered, model['producer'], 'q8', HASH)
        incomplete = copy.deepcopy(quality)
        incomplete['producer']['quant_raw_complete'] = False
        with self.assertRaisesRegex(ValueError, 'raw artifact or completeness'):
            subject.checked_quant_raw_proofs(incomplete, model['producer'], 'q8', HASH)

    def test_quant_raw_artifact_requires_reviewed_exact_service_identity(self):
        model, _, quality, _, decision = fixtures(self.folder, 'q8')
        pins = subject.require_receipts('q8')
        self.assertEqual(pins['raw_diff_artifact_id'], 2003)
        self.assertEqual(pins['raw_diff_artifact_zip_sha256'], HASH)
        changes = (
            ('id', 0), ('id', True), ('name', 'ovisocr2-quality-native-diff-455'),
            ('zip_sha256', '0'*64), ('zip_sha256', None))
        for key, value in changes:
            altered = copy.deepcopy(quality)
            altered['raw_diff_artifact'][key] = value
            with self.subTest(field=key, value=value), self.assertRaisesRegex(
                    ValueError, 'raw artifact or completeness'):
                subject.checked_quant_raw_proofs(altered, model['producer'], 'q8', HASH)
        for field in ('raw_diff_verified', 'raw_diff_artifact_zip_sha256'):
            altered = copy.deepcopy(quality)
            altered['independent_review'][field] = False
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, 'raw artifact or completeness'):
                subject.checked_quant_raw_proofs(altered, model['producer'], 'q8', HASH)
        altered = copy.deepcopy(quality)
        altered['independent_review']['producer_canonical_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'does not bind the producer'):
            subject.checked_quant_raw_proofs(altered, model['producer'], 'q8', HASH)
        quality['raw_diff_artifact']['name'] = 'ovisocr2-quality-native-diff-455'
        quality_sha = write(self.folder, 'ovisocr2-q8-quality.json', quality)
        decision['attempts']['q8']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaisesRegex(ValueError, 'raw artifact or completeness'):
            subject.require_receipts('q8')

    def test_retained_473_tensor_records_recompute_protected_bytes(self):
        _, _, quality, _, decision = fixtures(self.folder, 'q8')
        self.assertEqual(len(quality['producer']['tensor_records']['before']), 473)
        self.assertEqual(subject.require_receipts('q8')['precision'], 'q8')
        name = 'v.patch_embed.proj.weight'
        quality['producer']['tensor_records']['after'][name]['sha256'] = '9'*64
        quality_sha = write(self.folder, 'ovisocr2-q8-quality.json', quality)
        decision['attempts']['q8']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaises(ValueError):
            subject.require_receipts('q8')

    def test_f16_can_select_after_all_three_quant_attempts_rejected(self):
        _, _, _, _, decision = fixtures(self.folder)
        f16_pins = subject.require_receipts('f16')
        self.assertEqual(f16_pins['precision'], 'f16')
        self.assertNotIn('raw_diff_artifact_id', f16_pins)
        decision['attempts'].pop('q4')
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        with self.assertRaises(ValueError): subject.require_receipts('f16')

    def test_calibrated_q4_requires_actual_development_imatrix_and_prior_q8(self):
        _, _, quality, calibration, decision = fixtures(self.folder, 'q4-calibrated')
        quality['producer']['calibration'] = {'vectors': [{} for _ in range(72)],
                                              'imatrix_sha256': HASH, 'stats_sha256': HASH,
                                              'imatrix_bytes': 200,
                                              'report_files': {
                                                  'development-imatrix-stats.tsv': {
                                                      'path': 'calibration/development-imatrix-stats.tsv',
                                                      'sha256': HASH, 'bytes': 100},
                                                  'development-imatrix.gguf': {
                                                      'path': 'calibration/development-imatrix.gguf',
                                                      'sha256': HASH, 'bytes': 200}},
                                              'input_ids': [f'dev-{n}' for n in range(4)]}
        quality['independent_review']['calibration_report_files_verified'] = True
        refresh_producer_review(quality)
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
        quality['producer']['calibration']['report_files']['development-imatrix.gguf']['bytes'] = 1
        refresh_producer_review(quality)
        quality_sha = write(self.folder, 'ovisocr2-q4-calibrated-quality.json', quality)
        decision['attempts']['q4-calibrated']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        calibration['producer']['quality_receipt_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-q4-calibrated-calibration.json', calibration)
        with self.assertRaises(ValueError): subject.require_receipts('q4-calibrated')
        quality['producer']['calibration']['report_files']['development-imatrix.gguf']['bytes'] = 200
        refresh_producer_review(quality)
        quality_sha = write(self.folder, 'ovisocr2-q4-calibrated-quality.json', quality)
        decision['attempts']['q4-calibrated']['report_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-quant-attempts.json', decision)
        calibration['producer']['quality_receipt_sha256'] = quality_sha
        write(self.folder, 'ovisocr2-q4-calibrated-calibration.json', calibration)
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
        raw = {'id': 2003, 'name': 'ovisocr2-quality-native-diff-456',
               'expired': False, 'workflow_run': {'id': 456},
               'digest': 'sha256:'+HASH}
        self.assertEqual(subject.service_artifact(
            2003, 456, raw['name'], HASH,
            lambda _: json.dumps(raw).encode())['id'], 2003)
        with self.assertRaises(ValueError):
            subject.service_artifact(2003, 456, raw['name'], '0'*64,
                                     lambda _: json.dumps(raw).encode())

    def test_quant_preflight_checks_raw_service_before_hub_access(self):
        fixtures(self.folder, 'q8')
        seen = []

        def artifact(*args):
            seen.append(args)
            return {'id': args[0], 'name': args[2]}

        with mock.patch.object(subject, 'service_run', return_value={}), \
                mock.patch.object(subject, 'service_artifact', side_effect=artifact), \
                mock.patch('sys.argv', ['publisher', '--precision', 'q8',
                    '--model-run-id', '123', '--quality-run-id', '456', '--preflight']), \
                mock.patch.dict(os.environ, {'HEAVY_OUT': str(self.folder/'out'),
                    'HEAVY_SCRATCH': str(self.folder/'scratch'), 'GITHUB_SHA': HEAD}, clear=False):
            subject.main()
        self.assertIn((2003, '456', 'ovisocr2-quality-native-diff-456', HASH), seen)
        report = json.loads((self.folder/'out/ovisocr2-publication.json').read_text())
        self.assertEqual(report['status'], 'preflight passed; no auth/packages/weights accessed')

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
            with self.assertRaisesRegex(ValueError, 'review/source|phase/model|First reviewed F16'):
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
        rendered = subject.card(pins, 'ovisocr2-f16.gguf', evidence_source_commit=HEAD)
        self.assertIn('/blob/' + HEAD + '/tests/results/qwen35-ocr/', rendered)
        for invalid in ('main', '', None):
            with self.assertRaises(ValueError):
                subject.card(pins, 'ovisocr2-f16.gguf', evidence_source_commit=invalid)
        self.assertIn(HEAD, rendered)
        self.assertNotIn('${', rendered)
        for name, value in SELECTION['values'].items():
            self.assertIn(name + '=' + value, rendered)
        with self.assertRaises(ValueError): subject.require_receipts('f32')


if __name__ == '__main__':
    unittest.main()
