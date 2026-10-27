"""Lightweight closure/acceptance controls; no arrays, models or builds."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import ovisocr2_final_integrated_v1 as c


class FinalContract(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / 'docs/ovisocr2-final-integrated-v1.json').read_text())

    def reviewed(self):
        value = copy.deepcopy(self.protocol)
        for d in value['dependencies'].values():
            d.update(reviewed=True, panel_completed=True, reviewed_outcome='candidate-rejected-default-off',
                     run=123, source='a'*40, conclusion='failure',
                     artifacts=[dict(id=1, name='review', bytes=100, sha256='b'*64,
                                     reports={'report.json':'c'*64})])
        return value

    def fixture(self):
        rows = {}
        for i in range(11):
            rows[str(i)] = dict(input_rgb_sha256='rgb'+str(i), input_png_sha256='png'+str(i),
                raw_text='text'+str(i), processed_text='text'+str(i), token_ids=[17,248046],
                stop_id=248046, stop_kind=1, finish_reason='stop', cap=32,
                confidences=[.5,.75], mean_confidence=.625)
        return dict(native_cases={'f16':rows}, scores={'f16':{'exact':11}},
                    f16_paired_synthetic={'screen_passed':True}, f16_paired_documents={'screen_passed':True})

    def pending(self):
        value = copy.deepcopy(self.protocol)
        for item in value['dependencies'].values():
            item.update(reviewed=False, panel_completed=False)
        return value

    def driver(self):
        spec = importlib.util.spec_from_file_location('final_driver', ROOT / 'tools/ci-heavy/ovisocr2_final_integrated_v1.py')
        driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
        return driver

    def test_pending_reviews_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'pending'):
            c.checked_protocol(self.pending())

    def test_pending_preflight_does_not_access_services_or_models(self):
        driver = self.driver()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict('os.environ', GITHUB_ACTIONS='true'):
            pending = Path(temporary) / 'pending.json';pending.write_text(json.dumps(self.pending()))
            with mock.patch.object(driver, 'SPEC', pending), mock.patch.object(driver, 'api', side_effect=AssertionError('Network must remain untouched')) as call:
                with self.assertRaisesRegex(ValueError, 'pending'):
                    driver.preflight(Path(temporary) / 'out')
                call.assert_not_called()

    def test_later_quality_check_retains_earlier_structured_failure(self):
        driver = self.driver()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            first = driver.save_rejection(out, 'preflight', None, ValueError('first failure'))
            before = first.read_bytes()
            second = driver.save_rejection(out, 'quality-check', 0, ValueError('later failure'))
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(json.loads(second.read_text())['phase'], 'quality-check')

    def test_direct_frozen_driver_requires_explicit_gate_before_access(self):
        result = subprocess.run([sys.executable, ROOT / 'tools/ci-heavy/ovisocr2_final_integrated_v1.py',
                                 '--phase', 'frozen'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--gate is required', result.stderr)

    def test_reviewed_negative_is_not_default_acceptance(self):
        value = self.reviewed()
        self.assertIs(c.checked_protocol(value), value)
        value['dependencies']['wasm_rounding_ab']['final_default'] = 'on'
        with self.assertRaisesRegex(ValueError, 'defaultOFF'):
            c.checked_protocol(value)

    def test_review_must_pin_actual_service_and_reports(self):
        for field, bad in [('panel_completed',False), ('run', None), ('source','latest'), ('artifacts',[])]:
            value = self.reviewed(); value['dependencies']['ovis_composed'][field] = bad
            with self.assertRaises(ValueError): c.checked_protocol(value)
        value = self.reviewed();value['dependencies']['ovis_composed']['artifacts'][0]['reports'] = {'../report.json':'c'*64}
        with self.assertRaises(ValueError): c.checked_protocol(value)

    def test_exact_final_objects_and_known_three_source_changes(self):
        proof = c.source_guard(ROOT, self.protocol, require_submodule=False)
        self.assertEqual(proof['ggml'], c.GGML)
        self.assertEqual(proof['declared_source_sha256'], c.DECLARED)
        for edit in ('omit', 'alter'):
            value = copy.deepcopy(self.protocol)
            if edit == 'omit': value['protected_git_objects'].pop('src')
            else: value['protected_git_objects']['src'] = 'a'*40
            with self.assertRaises(ValueError): c.source_guard(ROOT, value, require_submodule=False)

    def test_f16_helper_preparation_and_sixth_binary_source_are_closed(self):
        required = {'tools/ci-heavy/qwen35_quality.py', 'tools/ci-heavy/ocr_quality_prepare.py',
                    'tests/results/ocr-quality-prepared.json', 'tests/test_ovisocr2_image_bounds.cpp'}
        self.assertTrue(required.issubset(self.protocol['protected_git_objects']))
        first = json.loads((ROOT / 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']
        self.assertTrue(set(first['source_sha256']).issubset(c.PROTECTED_PATHS))
        for name in required:
            value = copy.deepcopy(self.protocol);value['protected_git_objects'].pop(name)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                c.source_guard(ROOT, value, require_submodule=False)

    def test_extra_production_source_is_rejected(self):
        path = ROOT / 'src/.final-integrated-owned-negative'
        self.assertFalse(path.exists())
        try:
            path.write_text('untracked source must reject\n')
            with self.assertRaisesRegex(ValueError, 'Dirty'):
                c.source_guard(ROOT, self.protocol, require_submodule=False)
        finally: path.unlink(missing_ok=True)

    def test_exact_six_binary_manifest_is_required(self):
        manifest = dict(source_commit='a'*40, protocol_sha256='b'*64, portable_cpu_baseline_checked=True,
                        binary_sha256={n:'c'*64 for n in c.BINARIES})
        self.assertIs(c.checked_binaries(manifest, 'a'*40, 'b'*64), manifest)
        for edit in ('missing', 'extra', 'badsha', 'wrongsource'):
            bad = copy.deepcopy(manifest)
            if edit == 'missing': bad['binary_sha256'].pop(c.BINARIES[0])
            elif edit == 'extra': bad['binary_sha256']['unexpected'] = 'c'*64
            elif edit == 'badsha': bad['binary_sha256'][c.BINARIES[0]] = True
            else: bad['source_commit'] = 'd'*40
            with self.assertRaisesRegex(ValueError, 'six final binaries'):
                c.checked_binaries(bad, 'a'*40, 'b'*64)

    def test_all_eleven_exact_cases_and_scores(self):
        first = self.fixture(); result = c.compare_quality(copy.deepcopy(first), first, 0)
        self.assertTrue(result['scope_passed']); self.assertFalse(result['release_qualified'])
        self.assertEqual(len(result['cases']), 11)

    def test_gate0_confidences_exact_gate1_drift_explicit(self):
        first = self.fixture(); current = copy.deepcopy(first)
        row = current['native_cases']['f16']['0'];row['confidences'][0] += .00001;row['mean_confidence'] += .000005
        with self.assertRaisesRegex(ValueError, 'Gate0'): c.compare_quality(current, first, 0)
        result = c.compare_quality(current, first, 1)
        self.assertGreater(result['cases']['0']['max_selected_confidence_abs_drift'], 0)

    def test_bad_ids_nonfinite_mean_scores_or_missing_case_reject(self):
        first = self.fixture()
        for edit in ('id','nan','mean','score','missing'):
            current = copy.deepcopy(first);row = current['native_cases']['f16']['0']
            if edit == 'id': row['token_ids'][0] += 1
            elif edit == 'nan': row['confidences'][0] = float('nan')
            elif edit == 'mean': row['mean_confidence'] = .9
            elif edit == 'score': current['scores']['f16']['exact'] = 10
            else: current['native_cases']['f16'].pop('0')
            with self.assertRaises(ValueError): c.compare_quality(current, first, 1)


if __name__ == '__main__': unittest.main()
