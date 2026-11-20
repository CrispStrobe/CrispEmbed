#!/usr/bin/env python3
"""Source-only controls: synthetic retained traces, no C++/model/network/CI execution."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pcs_capture', ROOT / 'tools/pcs_gate_capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def bits(value):
    return struct.pack('!f', value).hex()


class CaptureControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / 'reference.txt'
        self.reference.write_text('#LINE 0\n#TEXT ok next\n#TOKENS ▁ok ▁next\n#IDS 3664 1234\n'
                                  '#CAP 1110000000000000 0000000000000000\n#PUNC OK next\n')

    def trace(self, changed=False, split=False):
        rows = ['build\t201703\t' + 'synthetic compiler'.encode().hex(), 'isa\tAVX2',
                'request\t0\t2\t' + b'ok next'.hex(), 'token\t0\t0\t3664\t' + '▁ok'.encode().hex(),
                'token\t0\t1\t1234\t' + '▁next'.encode().hex(), 'chunk\t0\t0\t' + ('1' if split else '2')]
        if split:
            rows.append('chunk\t0\t1\t1')
        for t in range(2):
            # Exact threshold tie is false at 0.05f, despite being > Python 0.05.
            rows.append(f'seg\t0\t{t}\t00000000\t{bits(-3.0)}\t3d4ccccd\t0\t0')
            for c in range(16):
                value = 1.0 if t == 0 and c < 3 else -1.0
                if changed and t == 0 and c == 1:
                    value = -2e-7
                rows.append(f'cap\t0\t{t}\t{c}\t{bits(value)}\t{int(value > 0)}\t{int(t == 0 or split)}')
        return rows

    def analyze(self, rows):
        path = self.root / 'trace.tsv'
        path.write_text('\n'.join(rows) + '\n')
        return capture.analyze(path, self.reference)

    def test_original_source_is_exactly_recoverable_from_overlay(self):
        source = (ROOT / 'src/pcs.cpp').read_text()
        observed = capture.overlay(source)
        restored = observed.removeprefix(capture.PREFIX)
        for anchor, insertion in capture.HOOKS:
            restored = restored.replace(anchor + insertion, anchor)
        self.assertEqual(restored, source)
        for broken in (source.replace(capture.HOOKS[0][0], ''), source + capture.HOOKS[0][0]):
            with self.assertRaises(ValueError):
                capture.overlay(broken)

    def test_full_capture_and_f32_threshold_tie(self):
        report = self.analyze(self.trace())
        self.assertEqual((report['tokens'], report['casing_cells']), (2, 32))
        self.assertFalse(report['significant_casing_mismatches'])
        self.assertFalse(report['token_mapping_mismatches'])
        self.assertTrue(report['all_values_finite'])
        self.assertGreater(capture.f32('3d4ccccd'), 0.05)

    def test_real_character_slot_has_signed_margin_and_magnitude(self):
        report = self.analyze(self.trace(changed=True))
        row = report['first_known_casing_mismatch']
        self.assertEqual((row['token'], row['slot'], row['native_id'], row['native_piece']), (0, 1, 3664, '▁ok'))
        self.assertLess(row['margin_to_zero'], 0)
        self.assertEqual(row['absolute_magnitude'], -row['margin_to_zero'])
        self.assertEqual((row['actual_bit'], row['reference_bit']), (0, 1))

    def test_ignored_prefix_and_padding_stay_ignored(self):
        rows = self.trace()
        for slot in (0, 15):
            index = next(i for i, v in enumerate(rows) if v.startswith(f'cap\t0\t0\t{slot}\t'))
            rows[index] = f'cap\t0\t0\t{slot}\t{bits(-1 if slot == 0 else 1)}\t{int(slot == 15)}\t1'
        self.assertFalse(self.analyze(rows)['significant_casing_mismatches'])

    def test_native_identity_difference_is_not_mislabelled_as_only_a_casing_change(self):
        rows = self.trace()
        index = next(i for i, v in enumerate(rows) if v.startswith('token\t0\t0\t'))
        rows[index] = 'token\t0\t0\t999\t' + '▁other'.encode().hex()
        self.assertEqual(self.analyze(rows)['token_mapping_mismatches'][0]['native_id'], 999)

    def test_partial_duplicate_and_inconsistent_sign_traces_rejected(self):
        rows = self.trace()
        for invalid in (rows[:-1], rows + [rows[-1]], rows + ['unknown\tfield']):
            with self.assertRaises(ValueError):
                self.analyze(invalid)
        index = next(i for i, v in enumerate(rows) if v.startswith('cap\t0\t0\t1\t'))
        rows[index] = f'cap\t0\t0\t1\t{bits(1)}\t0\t1'
        with self.assertRaisesRegex(ValueError, 'sign or conditioning'):
            self.analyze(rows)

    def test_chunk_reset_and_actual_conditioning_are_verified(self):
        self.assertEqual(self.analyze(self.trace(split=True))['tokens'], 2)
        rows = self.trace()
        index = next(i for i, v in enumerate(rows) if v.startswith('cap\t0\t1\t0\t'))
        rows[index] = f'cap\t0\t1\t0\t{bits(-1)}\t0\t1'
        with self.assertRaisesRegex(ValueError, 'conditioning'):
            self.analyze(rows)

    def test_nan_cannot_be_relabelled_as_a_negative_margin(self):
        rows = self.trace()
        index = next(i for i, v in enumerate(rows) if v.startswith('cap\t0\t0\t1\t'))
        rows[index] = 'cap\t0\t0\t1\t7fc00000\t0\t1'
        report = self.analyze(rows)
        self.assertFalse(report['all_values_finite'])
        self.assertIsNone(report['first_known_casing_mismatch']['margin_to_zero'])
        json.dumps(report, allow_nan=False)

    def test_invalid_model_identity_stops_before_provider_import(self):
        with self.assertRaises(ValueError):
            capture.fetch_model('existing/repository', 'same-recipe.gguf', 'main', '0' * 64, self.root / 'model')

    def test_immutable_fetch_receipt_and_mismatched_hash_fail_closed(self):
        cached = self.root / 'cached-input'
        cached.write_bytes(b'synthetic bytes, no model')
        calls = []
        provider = types.SimpleNamespace(hf_hub_download=lambda **kw: calls.append(kw) or str(cached))
        with patch.dict('sys.modules', {'huggingface_hub': provider}):
            output = self.root / 'recorded-input'
            capture.fetch_model('existing/repository', 'same-recipe.gguf', 'a' * 40, capture.sha(cached), output)
            origin = json.loads(output.with_suffix('.origin.json').read_text())
            self.assertEqual(origin['sha256'], capture.sha(output))
            self.assertEqual(calls[0]['revision'], 'a' * 40)
            rejected = self.root / 'rejected-input'
            with self.assertRaises(ValueError):
                capture.fetch_model('existing/repository', 'same-recipe.gguf', 'a' * 40, '0' * 64, rejected)
            self.assertFalse(rejected.exists())

    def test_diagnostic_target_is_excluded_from_default_build_and_install(self):
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('option(CRISPEMBED_BUILD_PCS_CAPTURE "Expose test-only PCS provenance/casing observer" OFF)', cmake)
        self.assertIn('add_executable(punct-ab-capture EXCLUDE_FROM_ALL', cmake)
        self.assertNotRegex(cmake, r'install\([^)]*punct-ab-capture')

    def test_wrong_model_bytes_stop_before_compiler_or_build_inspection(self):
        model = self.root / 'wrong-input'
        model.write_bytes(b'synthetic bytes, no model')
        origin = self.root / 'wrong-origin.json'
        origin.write_text(json.dumps({'revision': 'a' * 40, 'sha256': '0' * 64}))
        with self.assertRaisesRegex(ValueError, 'immutable origin'):
            capture.provenance(model, origin, self.root / 'absent-binary', self.root / 'absent-build',
                               self.root / 'must-not-exist.json')

    def test_read_only_provenance_binds_inputs_and_observer_source(self):
        build = self.root / 'build'
        build.mkdir()
        model, binary, compiler = [self.root / n for n in ('model', 'binary', 'compiler')]
        for path in (model, binary, compiler):
            path.write_bytes(b'synthetic fixture, never executed')
        origin = self.root / 'origin.json'
        origin.write_text(json.dumps({'repository': 'existing/repository', 'filename': 'same-recipe.gguf',
                                      'revision': 'a' * 40, 'sha256': capture.sha(model)}))
        (build / 'CMakeCache.txt').write_text(f'CMAKE_CXX_COMPILER:FILEPATH={compiler}\n')
        (build / 'compile_commands.json').write_text(json.dumps([{'file': str(build / 'pcs_gate_observed.cpp'),
                                                                 'command': 'synthetic command, never executed'}]))
        observed = build / 'pcs_gate_observed.cpp'
        observed.write_text(capture.overlay((ROOT / 'src/pcs.cpp').read_text()))
        output = self.root / 'provenance.json'
        with patch.object(capture.subprocess, 'check_output', side_effect=lambda args, **kw:
                          'synthetic compiler' if '--version' in args else 'b' * 40):
            capture.provenance(model, origin, binary, build, output)
        report = json.loads(output.read_text())
        self.assertEqual(report['model']['sha256'], capture.sha(model))
        self.assertEqual(report['observed_source_sha256'], capture.sha(observed))
        self.assertEqual(report['binary']['sha256'], capture.sha(binary))
        observed.write_text('unexpected source')
        with self.assertRaisesRegex(ValueError, 'differs'):
            capture.provenance(model, origin, binary, build, self.root / 'rejected.json')


if __name__ == '__main__':
    unittest.main()
