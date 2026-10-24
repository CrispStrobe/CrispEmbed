#!/usr/bin/env python3
"""Small, dependency-free integrity controls for hosted local operator replay."""

import hashlib
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovisocr2_vision_local_ops', ROOT / 'tools/ci-heavy/ovisocr2_vision_local_ops.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class VisionLocalOpsControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def records(self):
        stages = driver.SELECTED + tuple(f'prefill.vision.synthetic.{i}' for i in range(33))
        records, payload = {}, {}
        for precision in ('f32', 'f16'):
            for arm in ('baseline', 'precise_norm', 'precise_attn', 'both',
                        'cascade', 'cascade_precise_attn'):
                label = precision + '-' + arm
                records[label] = {}
                for index, stage in enumerate(stages):
                    name = label + '/' + stage + '.f32'
                    data = struct.pack('<f', float(index + 1))
                    payload[name] = data
                    records[label][stage] = {'bytes': len(data),
                                             'sha256': hashlib.sha256(data).hexdigest()}
        return records, payload

    def archive(self, payload):
        path = self.root / 'vectors.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, data in payload.items():
                archive.writestr(name, data)
        return path

    def test_unset_pins_fail_before_remote_access_or_weights(self):
        with mock.patch.dict(os.environ, {'HEAVY_OUT': str(self.root / 'out'),
                                       'HEAVY_SCRATCH': str(self.root / 'scratch')}), \
                mock.patch.multiple(driver, AB_RUN=None, AB_HEAD=None, AB_REPORT_ARTIFACT=None,
                                    AB_VECTOR_ARTIFACT=None, AB_REPORT_SHA256=None,
                                    AB_VECTOR_ARCHIVE_ROOT=None), \
                mock.patch.object(driver, 'api_json', side_effect=AssertionError('network called')):
            with self.assertRaisesRegex(ValueError, 'pins are unset'):
                driver.main(preflight_only=True)
        self.assertEqual(list((self.root / 'scratch' / 'ovis-vision-local-ops').iterdir()), [])

    def test_one_precision_passes_are_controls_but_common_preset_skips_fallback(self):
        helper = driver.load_ab_helper()
        labels = driver.required_labels(helper)
        report = {'arms': {label: {'exit_code': 1, 'timed_out': False,
                                  'metrics': {'first_failing_operation': 'prefill.vision.probe.11.up'}}
                           for label in labels}}
        report['arms']['f32-precise_norm']['exit_code'] = 0
        report['arms']['f32-precise_norm']['metrics']['first_failing_operation'] = None
        report['passing_arms'] = ['f32-precise_norm']
        report['all_arms_pass_existing_comparator'] = False
        self.assertEqual(driver.validate_ab_outcome(report, labels), ['f32-precise_norm'])
        report['arms']['f16-precise_norm']['exit_code'] = 0
        report['arms']['f16-precise_norm']['metrics']['first_failing_operation'] = None
        report['passing_arms'] = ['f32-precise_norm', 'f16-precise_norm']
        with self.assertRaisesRegex(ValueError, 'common passing'):
            driver.validate_ab_outcome(report, labels)
        report['arms']['f16-precise_norm']['timed_out'] = True
        with self.assertRaisesRegex(ValueError, 'complete comparator outcome'):
            driver.validate_ab_outcome(report, labels)

    def test_every_member_verified_and_only_four_per_arm_extracted(self):
        records, payload = self.records()
        path = self.archive(payload)
        with mock.patch.object(driver, 'AB_VECTOR_ARCHIVE_ROOT', ''):
            result = driver.verify_and_extract_vectors(path, self.root / 'scratch', records)
        self.assertEqual(result['verified_member_count'], 444)
        self.assertEqual(result['selected_member_count'], 48)
        extracted = list((self.root / 'scratch' / 'vision-vectors').rglob('*.f32'))
        self.assertEqual(len(extracted), 48)
        self.assertTrue(all(path.name.removesuffix('.f32') in driver.SELECTED for path in extracted))

    def test_tamper_extra_and_traversal_rejected(self):
        records, payload = self.records()
        first = next(iter(payload))
        altered = dict(payload)
        altered[first] = struct.pack('<f', 99.0)
        with mock.patch.object(driver, 'AB_VECTOR_ARCHIVE_ROOT', ''):
            with self.assertRaisesRegex(ValueError, 'SHA/size differs'):
                driver.verify_and_extract_vectors(self.archive(altered), self.root / 'sha', records)
            with self.assertRaisesRegex(ValueError, 'member set differs'):
                driver.verify_and_extract_vectors(self.archive(payload | {'f32-baseline/extra.f32': b'x'}),
                                                  self.root / 'extra', records)
            with self.assertRaisesRegex(ValueError, 'unsafe member'):
                driver.verify_and_extract_vectors(self.archive(payload | {'../escape.f32': b'x'}),
                                                  self.root / 'traversal', records)

    def test_raw_log_requires_both_controls_and_all_four_metrics(self):
        controls = ''.join(f'LOCAL_OP_CONTROL native.{stage} exact=1 n=8/8\n'
                           for stage in ('norm2', 'up'))
        input_line = ('LOCAL_OP_INPUT norm2 n=8 max_abs=0.1 mean_abs=0.01 worst_index=3 '
                      'row=0 native=1 original=0.9\n')
        metrics = ''.join(
            f'BUDGET prefill.vision.probe.11.{stage}: strict_absolute_pass=0 calibrated=1 '
            'self_max_abs=0.001 max_ratio=1.2\n'
            f'LOCAL_OP_METRIC {mode} prefill.vision.probe.11.{stage} n=8 max_abs=0.1 '
            'mean_abs=0.01 cos_min=0.99999 worst_index=3 row=0 candidate=1 '
            'original=0.9 existing_budget_pass=0\n'
            for mode in ('native_input', 'same_original_input') for stage in ('norm2', 'up'))
        completion = ('LOCAL_OP_DIAGNOSTIC_COMPLETE complete=1 qualification_claim=0 '
                      'precise_norm=0 cascade_norm=0 precise_attn=0\n')
        transcript = controls + input_line + metrics + completion
        self.assertEqual(len(driver.parse_local_log(transcript)['metrics']), 4)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            driver.parse_local_log(transcript.replace('native.up exact=1', 'native.up exact=0'))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            driver.parse_local_log(transcript.replace('same_original_input prefill.vision.probe.11.up',
                                                     'same_original_input prefill.vision.probe.11.down'))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            driver.parse_local_log(transcript.replace('candidate=1 original=0.9',
                                                     'candidate=nan original=0.9', 1))


if __name__ == '__main__':
    unittest.main()
