"""Pure source and phase gates; no model or renderer imports."""

import importlib.util
import inspect
import copy
import subprocess
import sys
import json
import hashlib
from pathlib import Path
import tempfile
import zipfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_quality_run', ROOT / 'tools/ci-heavy/ovisocr2_quality_run.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class QualityDriverControls(unittest.TestCase):
    def test_first_linear_local_parser_requires_every_observed_operation(self):
        expected = driver.parse_first_linear_diagnostic('', 1)['local_expected_stages']
        def line(stage):
            return (f'PASS {stage}: cos_min=1 cos_global=1 '
                    'max_abs=0 mean_abs=0 norms=1/1')
        complete = '\n'.join(map(line, expected))
        positive = driver.parse_first_linear_diagnostic(complete, 0)
        self.assertTrue(positive['passed'])
        self.assertEqual(len(expected), 18)
        self.assertEqual(positive['decoder_layer_count'], 0)
        self.assertFalse(driver.parse_first_linear_diagnostic(
            '\n'.join(map(line, expected[:-1])), 0)['comparison_coverage'])
        self.assertFalse(driver.parse_first_linear_diagnostic(
            complete.replace('PASS ', 'FAIL ', 1), 1)['passed'])

    def test_raw_quant_manifest_inventories_metrics_traces_and_partial_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            metrics = ['PASS prefill.decoder.layer.0: cos_min=1 cos_global=1 '
                       'max_abs=0 mean_abs=0 norms=1/1']
            traces = ['TRACE prefill.attention.3.input max_abs=0 cos=1 norms=1/1']
            stages = {'prefill.decoder.layer.0': [2], 'prefill.attention.3.input': [1]}
            (folder/'prefill.decoder.layer.0.f32').write_bytes(b'12345678')
            (folder/'prefill.attention.3.input.f32').write_bytes(b'1234')
            manifest = driver.quant_raw_diff_manifest(
                folder, stages, metrics, traces, 'q8', 'decoder', 'a'*64)
            self.assertEqual(manifest['file_count'], 2)
            self.assertIn('prefill.attention.3.input', manifest['observed_metric_stages'])
            self.assertIn('cached.decoder.layer.0', manifest['missing_stages'])
            self.assertFalse(manifest['coverage_complete'])
            first = manifest['files'][0]
            self.assertEqual(first['sha256'], driver.sha(folder/(first['stage']+'.f32')))
            (folder/'prefill.attention.3.input.f32').write_bytes(b'bad')
            changed = driver.quant_raw_diff_manifest(folder, stages, metrics, traces,
                                                     'q8', 'decoder', 'a'*64)
            self.assertTrue(changed['invalid_files'])

    def test_first_linear_dump_and_weight_records_fail_closed(self):
        expected = driver.parse_first_linear_diagnostic('', 1)['local_expected_stages']
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            shapes = {stage: [1] for stage in expected}
            metrics = [f'PASS {stage}: cos_min=1 cos_global=1 max_abs=0 '
                       'mean_abs=0 norms=1/1' for stage in expected]
            for stage in expected:
                (folder/(stage+'.f32')).write_bytes(b'\0'*4)
            manifest = driver.quant_raw_diff_manifest(folder, shapes, metrics, [],
                                                      'q4', 'first-linear', 'a'*64)
            self.assertTrue(manifest['coverage_complete'])
            (folder/(expected[0]+'.f32')).unlink()
            self.assertFalse(driver.quant_raw_diff_manifest(folder, shapes, metrics, [],
                             'q4', 'first-linear', 'a'*64)['coverage_complete'])
        names = [f'l.layers.0.linear_attn.{name}' for name in (
            'in_proj_qkv.weight', 'in_proj_z.weight', 'in_proj_a.weight',
            'in_proj_b.weight', 'out_proj.weight', 'conv1d.weight',
            'A_log', 'dt_bias', 'norm.weight')]
        records = {'before': {name: {'type': 'F16'} for name in names},
                   'after': {name: {'type': 'Q8_0'} for name in names}}
        self.assertEqual(len(driver.first_linear_weight_records(records)['weights']['after']), 9)
        del records['after'][names[0]]
        with self.assertRaisesRegex(ValueError, 'First linear operation'):
            driver.first_linear_weight_records(records)

    def test_both_quant_probes_run_after_first_numeric_failure(self):
        calls = []

        def fake_probe(*args):
            probe = args[-1]
            calls.append(probe)
            return {'passed': False, 'raw_diff_coverage_complete': True}

        report = {}
        with patch.object(driver, 'frozen_numerical_shapes', return_value=({}, 'a'*64)), \
                patch.object(driver, 'quant_probe', side_effect=fake_probe):
            result = driver.quant_numerical_diagnostics(
                Path('/unused'), Path('/unused.gguf'), Path('/unused.ref'), 'q8',
                Path('/unused-out'), report, lambda: None, {}, Path('/unused-scratch'))
        self.assertEqual(calls, ['first-linear', 'decoder'])
        self.assertTrue(result['raw_complete'])
        self.assertTrue(report['quant_raw_complete'])
        self.assertFalse(result['decoder']['passed'])

    def test_frozen_stage_shapes_bind_exact_original_json_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reference = folder/'official-delta.gguf'
            original = folder/'ovisocr2-original.json'
            original.write_text(json.dumps({'stages': {f'stage.{i}': [1] for i in range(943)},
                                            'missing_stages': []}))
            producer = {'original_capture': {'sha256': driver.sha(original)}}
            shapes, digest = driver.frozen_numerical_shapes(reference, producer)
            self.assertEqual(len(shapes), 943)
            self.assertEqual(digest, driver.sha(original))
            original.write_text(original.read_text().replace('"stage.0": [1]', '"stage.0": [2]'))
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                driver.frozen_numerical_shapes(reference, producer)

    def test_quant_workflow_uploads_raw_separately(self):
        workflow = (ROOT/'.github/workflows/ovisocr2-quality.yml').read_text()
        self.assertIn('ovisocr2-quality-native-diff-${{ github.run_id }}', workflow)
        self.assertIn('/ovis-quality/native-diff/', workflow)
        source = (ROOT/'tests/test_qwen35_linear.cpp').read_text()
        self.assertIn('std::string(argv[3]) == "--first-layer"', source)
        self.assertIn('if (first_layer && index > 0) break;', source)
        original = subprocess.check_output(['git', 'show',
            '638380f6:tests/test_qwen35_linear.cpp'], cwd=ROOT, text=True)
        def compare_body(text):
            return text.split('static bool compare(', 1)[1].split('static bool include(', 1)[0]
        self.assertEqual(compare_body(source), compare_body(original))

    def test_short_control_is_explicit_and_full_driver_default_remains(self):
        self.assertEqual(inspect.signature(driver.run_main).parameters[
            'f16_host_control'].default, 'full')
        workflow = (ROOT/'.github/workflows/ovisocr2-quality.yml').read_text()
        self.assertIn('f16_host_control:', workflow)
        self.assertIn('default: dev-hello', workflow)
        self.assertEqual(driver.F16_HOST_CONTROLS, ('full', 'dev-hello'))
        with self.assertRaisesRegex(ValueError, 'Unknown F16 host control mode'):
            driver.run_main('q8', f16_host_control='invented')

    def test_exact_git_and_ast_source_closure_rejects_any_drift(self):
        first, candidate = 'a'*40, 'b'*40
        first_native = {'src/qwen35_vision.cpp': 'c'*64}
        paths = {name: {'first_oid': 'd'*40, 'candidate_oid': 'd'*40}
                 for name in driver.F16_SOURCE_CLOSURE_PATHS}
        asts = {name: {'first_sha256': 'e'*64, 'candidate_sha256': 'e'*64}
                for name in driver.F16_AST_CONTRACTS}
        record = {'first_f16_source_commit': first, 'candidate_source_commit': candidate,
                  'paths': paths, 'ast_contracts': asts,
                  'first_native_source_sha256': first_native}
        self.assertTrue(driver.validate_f16_source_closure(
            record, first, candidate, first_native, first_native))
        for change in ('src', 'ggml', 'ast', 'first_map', 'candidate_map', 'first_head'):
            altered = copy.deepcopy(record)
            first_map, candidate_map, old_head = first_native, first_native, first
            if change in ('src', 'ggml'):
                altered['paths'][change]['candidate_oid'] = 'f'*40
            elif change == 'ast':
                altered['ast_contracts']['cmake-configure']['candidate_sha256'] = 'f'*64
            elif change == 'first_map':
                altered['first_native_source_sha256'] = {}
            elif change == 'candidate_map':
                candidate_map = {}
            else:
                old_head = 'f'*40
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.validate_f16_source_closure(
                    altered, old_head, candidate, first_map, candidate_map)
        old_ast = driver.f16_runtime_ast_contracts((ROOT/'tools/ci-heavy/ovisocr2_quality_run.py').read_text())
        changed = (ROOT/'tools/ci-heavy/ovisocr2_quality_run.py').read_text().replace(
            "env['CRISPEMBED_QWEN35_QUANT'] = '0' if arm == 'f16' else '1'",
            "env['CRISPEMBED_QWEN35_QUANT'] = '1' if arm == 'f16' else '0'")
        self.assertNotEqual(old_ast['native_environment'],
                            driver.f16_runtime_ast_contracts(changed)['native_environment'])

    def test_actual_git_blob_closure_matches_same_source_without_model_access(self):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       text=True).strip()
        first_bytes = subprocess.check_output(['git', 'show',
            head+':tools/ci-heavy/ovisocr2_quality_run.py'], cwd=ROOT)
        native = {'tests/ovisocr2_quality_run.cpp': driver.sha(
            ROOT/'tests/ovisocr2_quality_run.cpp')}
        producer = {'source_commit': head, 'native_source_sha256': native,
                    'source_sha256': {'tools/ci-heavy/ovisocr2_quality_run.py':
                                      hashlib.sha256(first_bytes).hexdigest()}}
        proof = driver.f16_source_closure(producer, head, head)
        self.assertEqual(proof['paths']['ggml']['first_oid'],
                         proof['paths']['ggml']['candidate_oid'])
        producer['native_source_sha256'] = {'tests/ovisocr2_quality_run.cpp': '0'*64}
        with self.assertRaises(ValueError):
            driver.f16_source_closure(producer, head, head)

    def test_dev_hello_short_control_binds_pixels_ids_text_eos_and_actual_mean(self):
        case = {'id': 'dev-hello', 'cap': 256, 'rgb_sha256': 'a'*64,
                'png_sha256': 'b'*64}
        ids = [9419, 77796, 271, 16, 17, 18, 19, 20, 248046]
        frozen = {'token_ids': ids, 'raw_text': 'Hello OCR\n\n12345',
                  'processed_text': 'Hello OCR\n\n12345', 'mean_confidence': .9,
                  'stop_id': 248046}
        actual = {'input_rgb_sha256': case['rgb_sha256'],
                  'input_png_sha256': case['png_sha256'], 'cap': case['cap'],
                  'token_ids': ids, 'confidences': [.91]*9,
                  'mean_confidence': .91, 'raw_text': frozen['raw_text'],
                  'processed_text': frozen['processed_text'],
                  'stop_id': 248046, 'finish_reason': 'stop',
                  'wall_seconds': 2.5, 'child_peak_rss_kib_upper_bound': 500000}
        result = driver.validate_short_f16_host_control(actual, frozen, case)
        self.assertEqual(result['native_case_count'], 1)
        self.assertAlmostEqual(result['mean_confidence_abs_delta'], .01)
        for field, value in (('input_png_sha256', '0'*64), ('token_ids', ids[:-1]+[248044]),
                             ('raw_text', 'wrong'), ('stop_id', 248044),
                             ('mean_confidence', float('nan'))):
            altered = copy.deepcopy(actual)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.validate_short_f16_host_control(altered, frozen, case)

    def test_small_calibration_proof_copy_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats, matrix = root/'stats.tsv', root/'imatrix.gguf'
            stats.write_bytes(b'one development row\n')
            matrix.write_bytes(b'GGUF development vectors')
            report = {'calibration': {'stats_sha256': driver.sha(stats),
                                      'imatrix_sha256': driver.sha(matrix)}}
            copied = driver.preserve_calibration_proof(stats, matrix, root/'reports', report)
            self.assertEqual(set(copied), {'development-imatrix-stats.tsv',
                                           'development-imatrix.gguf'})
            self.assertEqual(copied['development-imatrix.gguf']['bytes'], matrix.stat().st_size)
            self.assertEqual((root/'reports/calibration/development-imatrix.gguf').read_bytes(),
                             matrix.read_bytes())
            stats.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                driver.preserve_calibration_proof(stats, matrix, root/'reports', report)

    def test_first_f16_receipt_requires_reviewed_eleven_case_source(self):
        selection = {'name': 'both', 'enabled_gates': ['CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                                                      'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],
                     'values': {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM': '1',
                                'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN': '1',
                                'CRISPEMBED_QWEN35_VISION_CASCADE_NORM': '0'}}
        producer = {'status': 'completed F16 quality observations; independent review pending',
                    'requested_phase': 'f16', 'source_commit': 'a' * 40,
                    'selected_vision_preset': selection,
                    'prerequisite': {'baseline_sha256': 'b' * 64,
                                     'original_review_sha256': 'c' * 64},
                    'models': {'f16': {'sha256': driver.F16_SHA}},
                    'inputs': [{'id': f'case-{i}'} for i in range(11)],
                    'native_cases': {'f16': {f'case-{i}': {} for i in range(11)}},
                    'scores': {'f16': {}},
                    'f16_paired_synthetic': {'screen_passed': True},
                    'f16_paired_documents': {'screen_passed': True}}
        canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        receipt = {'run': 123, 'source_commit': 'a' * 40, 'artifact_id': 456,
                   'artifact_name': 'ovisocr2-quality-reports-123',
                   'artifact_zip_sha256': 'd' * 64, 'report_sha256': 'e' * 64,
                   'producer': producer, 'independent_review': {
                       'quality_positive': True, 'f16_11_complete': True,
                       'producer_canonical_sha256': canonical,
                       'artifact_zip_sha256': 'd' * 64}}
        self.assertEqual(driver.checked_f16_quality_receipt(receipt, 'b' * 64, 'c' * 64,
                                                               selection)['run'], 123)
        for field in ('quality_positive', 'f16_11_complete'):
            changed = json.loads(json.dumps(receipt))
            changed['independent_review'][field] = False
            with self.assertRaises(ValueError):
                driver.checked_f16_quality_receipt(changed, 'b' * 64, 'c' * 64, selection)
        changed = json.loads(json.dumps(receipt))
        changed['producer']['inputs'][0]['id'] = 'changed'
        changed['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
            changed['producer'], sort_keys=True, separators=(',', ':'), ensure_ascii=True,
            allow_nan=False).encode()).hexdigest()
        with self.assertRaises(ValueError):
            driver.checked_f16_quality_receipt(changed, 'b' * 64, 'c' * 64, selection)
        changed = json.loads(json.dumps(receipt))
        changed['producer']['prerequisite']['original_review_sha256'] = 'f' * 64
        changed['independent_review']['producer_canonical_sha256'] = hashlib.sha256(json.dumps(
            changed['producer'], sort_keys=True, separators=(',', ':'), ensure_ascii=True,
            allow_nan=False).encode()).hexdigest()
        with self.assertRaises(ValueError):
            driver.checked_f16_quality_receipt(changed, 'b' * 64, 'c' * 64, selection)

    def test_first_f16_replay_recomputes_scores_and_rejects_case_drift(self):
        cases = [{'id': f'case-{i}', 'cap': 12, 'rgb_sha256': 'a' * 64,
                  'png_sha256': 'b' * 64} for i in range(11)]
        outputs = {row['id']: {'token_ids': [248046], 'confidences': [.9],
                   'mean_confidence': .9, 'input_rgb_sha256': row['rgb_sha256'],
                   'input_png_sha256': row['png_sha256'], 'cap': 12, 'stop_id': 248046,
                   'finish_reason': 'stop'} for row in cases}
        current = {'inputs': cases, 'original_source': {'readme_sha256': 'c' * 64},
                   'scores': {'original': {'unchanged': True}},
                   'selected_vision_preset': {'name': 'both'}}
        frozen = {'inputs': cases, 'original_source': current['original_source'],
                  'scores': {'original': current['scores']['original'],
                             'f16': {'frozen': True}},
                  'selected_vision_preset': current['selected_vision_preset'],
                  'native_cases': {'f16': outputs}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'f16.json'
            path.write_text(json.dumps({'producer': frozen}))
            with patch.object(driver, 'F16_REVIEW', path), \
                 patch.object(driver, 'score_outputs', return_value={'frozen': True}):
                self.assertEqual(driver.validate_reused_f16(
                    frozen, current, cases, None, None)[1], {'frozen': True})
                changed = json.loads(json.dumps(frozen))
                changed['native_cases']['f16']['case-0']['stop_id'] = None
                path.write_text(json.dumps({'producer': changed}))
                with self.assertRaisesRegex(ValueError, 'stop differs'):
                    driver.validate_reused_f16(changed, current, cases, None, None)

    def test_original_report_accepts_only_two_registered_workflow_paths(self):
        payload = json.dumps({'status': 'completed original HF quality baseline; review pending'}).encode()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            archive = folder / 'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('ovisocr2-quality.json', payload)
            receipt_path = folder / 'review.json'
            receipt_path.write_text(json.dumps({'source_commit': 'a' * 40, 'artifact_id': 456,
                'artifact_zip_sha256': driver.sha(archive),
                'report_sha256': hashlib.sha256(payload).hexdigest()}))
            artifact = {'name': 'ovisocr2-quality-reports-123', 'expired': False,
                        'workflow_run': {'id': 123}, 'digest': 'sha256:' + driver.sha(archive),
                        'size_in_bytes': archive.stat().st_size, 'id': 456}
            for workflow in (*driver.QUALITY_WORKFLOW_PATHS, '.github/workflows/other.yml'):
                run = {'id': 123, 'head_sha': 'a' * 40, 'status': 'completed',
                       'conclusion': 'success', 'path': workflow}
                def api(argv, **_):
                    value = run if argv[-1].endswith('/runs/123') else artifact
                    return json.dumps(value).encode()
                def download(argv, **kwargs):
                    kwargs['stdout'].write(archive.read_bytes())
                with patch.object(driver, 'ORIGINAL_REVIEW', receipt_path), \
                     patch.object(driver.subprocess, 'check_output', side_effect=api), \
                     patch.object(driver.subprocess, 'run', side_effect=download):
                    if workflow in driver.QUALITY_WORKFLOW_PATHS:
                        value, proof = driver.reviewed_original_artifact(123, folder)
                        self.assertEqual(value['status'], 'completed original HF quality baseline; review pending')
                        self.assertEqual(proof['report_sha256'], hashlib.sha256(payload).hexdigest())
                    else:
                        with self.assertRaisesRegex(ValueError, 'workflow differs'):
                            driver.reviewed_original_artifact(123, folder)

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

    def test_complete_frozen_activation_artifact_size_and_identity(self):
        pins = {'run': 40041958521}
        expected = {'gguf': {'bytes': 1454224192}, 'json': {'bytes': 86906}}
        artifact = {'id': 12540836880, 'name': 'ovisocr2-model-diff-reference-40041958521',
                    'expired': False, 'workflow_run': {'id': pins['run']},
                    'size_in_bytes': 1303553123,
                    'digest': 'sha256:54c823899bdde33448ff93137fbbf960981fa031877a6e29f76c91b0065d36a4'}
        def listing(row):
            return {'total_count': 1, 'artifacts': [row]}
        self.assertEqual(driver.numerical_reference_artifact(listing(artifact), pins, expected), artifact)
        for changed in ({'expired': True}, {'workflow_run': {'id': 1}},
                        {'size_in_bytes': 2_000_000_000}, {'digest': 'sha256:invalid'},
                        {'id': 0}, {'name': 'model-weights'}):
            with self.assertRaisesRegex(ValueError, 'bounded original'):
                driver.numerical_reference_artifact(listing({**artifact, **changed}), pins, expected)
        with self.assertRaisesRegex(ValueError, 'byte counts'):
            driver.numerical_reference_artifact(listing(artifact), pins, {'gguf': {'bytes': '1454224192'}})

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
        with patch.object(contract, 'score_panel', return_value={}):
            result = driver.score_outputs(contract, None, cases, outputs)
            self.assertTrue(all(row['structure']['complete'] for row in result['realistic_documents']))
            outputs['inventory-table']['token_ids'] = [42, 43]
            outputs['inventory-table']['stop_id'] = None
            outputs['inventory-table']['finish_reason'] = 'length'
            result = driver.score_outputs(contract, None, cases, outputs)
            self.assertFalse(result['realistic_documents'][2]['structure']['complete'])

    def test_ovis_document_border_policy_preserves_strict_and_raw_metrics(self):
        scorer = driver.load_module('ovis_quality_border_test_scorer',
                                    ROOT / 'tools/ocr_quality.py')
        contract = driver.load_module('ovis_quality_border_test_contract',
                                      ROOT / 'tools/ci-heavy/ovisocr2_quality_contract.py')
        cells = [[f'cell-{row}-{col}' for col in range(3)] for row in range(5)]
        body = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>'
                       for row in cells)
        gold = {'text': ' '.join(cell for row in cells for cell in row),
                'tables': [cells]}
        for opening in ('<table border=1>', '<table border="1">',
                        "<table border='12'>", '<table border = 0>'):
            with self.subTest(opening=opening):
                output = opening + body + '</table>'
                strict = scorer.score(gold, output, complete=True)
                scored = driver.score_ovis_document(
                    scorer, gold, output, complete=True, policy=contract.DOCUMENT_POLICY)
                self.assertFalse(strict['parse_ok'])
                self.assertEqual(scored['strict_v1'], strict)
                self.assertTrue(scored['parse_ok'])
                self.assertTrue(scored['exact_quality'])
                self.assertEqual(scored['table']['expected_cells'], 15)
                self.assertEqual(scored['table']['cell_errors'], 0)
                self.assertEqual(scored['border_attribute_count'], 1)
                self.assertEqual(scored['normalized_text'], '<table>' + body + '</table>')
                self.assertEqual(scored['normalized_sha256'], hashlib.sha256(
                    scored['normalized_text'].encode()).hexdigest())
                for key in ('raw_text', 'raw', 'characters', 'lines'):
                    self.assertEqual(scored[key], strict[key])

        unchanged = '<table>' + body + '</table>'
        scored = driver.score_ovis_document(
            scorer, gold, unchanged, complete=True, policy=contract.DOCUMENT_POLICY)
        self.assertEqual(scored['border_attribute_count'], 0)
        self.assertEqual(scored['normalized_text'], unchanged)

        changed = body.replace('cell-0-0', 'wrong', 1)
        reordered = body.replace('cell-0-0', 'temp', 1).replace('cell-0-1', 'cell-0-0', 1).replace('temp', 'cell-0-1', 1)
        added = body + '<tr><td>extra</td><td>extra</td><td>extra</td></tr>'
        for mutated in (changed, reordered, added):
            with self.subTest(mutated=mutated[:40]):
                scored = driver.score_ovis_document(
                    scorer, gold, '<table border=1>' + mutated + '</table>',
                    complete=True, policy=contract.DOCUMENT_POLICY)
                self.assertTrue(scored['parse_ok'])
                self.assertFalse(scored['exact_quality'])
                self.assertGreater(scored['table']['cell_errors'], 0)

        malformed = (
            '<table border=1 style="color:red">' + body + '</table>',
            '<table border=1>' + body.replace('<td>', '<td rowspan="2">', 1) + '</table>',
            '<table border=-1>' + body + '</table>',
            '<table border="1" border="2">' + body + '</table>',
            '<table border=1>' + body + '<tr><td>ragged</td></tr></table>',
            '<table border=1>' + body,
            '<table border=1><tr><td><table><tr><td>x</td></tr></table></td>'
            '<td>y</td><td>z</td></tr></table>',
            '<table border=1><!-- comment -->' + body + '</table>',
        )
        for output in malformed:
            with self.subTest(output=output[:60]):
                scored = driver.score_ovis_document(
                    scorer, gold, output, complete=True, policy=contract.DOCUMENT_POLICY)
                self.assertFalse(scored['parse_ok'])
                self.assertFalse(scored['exact_quality'])
        with self.assertRaisesRegex(ValueError, 'Unknown OvisOCR2'):
            driver.score_ovis_document(scorer, gold, unchanged, complete=True, policy='other')

    def test_native_precision_env_has_explicit_calibrated_gate(self):
        selection = {'name': 'cascade_precise_attn',
                     'enabled_gates': ['CRISPEMBED_QWEN35_VISION_CASCADE_NORM',
                                       'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],
                     'values': {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM': '0',
                                'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN': '1',
                                'CRISPEMBED_QWEN35_VISION_CASCADE_NORM': '1'}}
        self.assertEqual(driver.native_environment('f16', selection)['CRISPEMBED_QWEN35_QUANT'], '0')
        for arm in ('q8', 'q4', 'q4-calibrated'):
            env = driver.native_environment(arm, selection)
            self.assertEqual(env['CRISPEMBED_QWEN35_QUANT'], '1')
            self.assertEqual(env['CRISPEMBED_QWEN35_VISION_CASCADE_NORM'], '1')
        with self.assertRaises(ValueError):
            driver.native_environment('unreviewed', selection)
        for arm in ('q8', 'q4', 'q4-calibrated'):
            env = driver.numerical_environment(arm, selection)
            self.assertEqual(env['CRISPEMBED_QWEN35_QUANT'], '1')
            self.assertEqual(env['CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS'], '1')
            self.assertTrue(all(env[name] == '0' for name in driver.DIFF_FLOAT_ZERO))
            self.assertEqual(env['CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'], '1')
        with self.assertRaises(ValueError):
            driver.numerical_environment('f16', selection)
        with self.assertRaises(ValueError):
            driver.native_environment('f16', None)

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
                    'selected_vision_preset': {'name': 'cascade_precise_attn',
                        'enabled_gates': ['CRISPEMBED_QWEN35_VISION_CASCADE_NORM',
                                          'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],
                        'all_other_native_gates_unchanged': True,
                        'no_production_default_promoted': True},
                    'native_environment': {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM': '0',
                        'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN': '1',
                        'CRISPEMBED_QWEN35_VISION_CASCADE_NORM': '1'},
                    'exports': {'f16': {'model_sha256': driver.F16_SHA}}}
        canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        receipt = {'run': 123, 'artifact': {'id': 456, 'zip_sha256': 'b' * 64},
                   'producer_report_sha256': 'c' * 64, 'producer': producer,
                   'independent_review': {'f16_positive': True,
                                          'artifact_zip_sha256': 'b' * 64,
                                          'producer_canonical_sha256': canonical}}
        self.assertEqual(driver.checked_model_receipt(receipt)['source_commit'], 'a' * 40)
        self.assertEqual(driver.checked_model_receipt(receipt)['selected_vision_preset']['name'],
                         'cascade_precise_attn')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(receipt))
            with patch.object(driver, 'BASELINE', path):
                self.assertEqual(driver.require_baseline('original')['baseline_pins']['run'], 123)
        for change in ('bare', 'failed', 'f16', 'canonical', 'artifact', 'run', 'review',
                       'missing-vision', 'changed-vision'):
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
            elif change == 'missing-vision':
                altered['producer']['native_environment'].pop('CRISPEMBED_QWEN35_VISION_CASCADE_NORM')
            elif change == 'changed-vision':
                altered['producer']['native_environment']['CRISPEMBED_QWEN35_VISION_CASCADE_NORM'] = '0'
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
