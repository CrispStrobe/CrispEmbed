#!/usr/bin/env python3
"""Small source and raw-evidence controls for the frozen Ovis native replay."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/ci-heavy/ovisocr2_native_replay.py'
spec = importlib.util.spec_from_file_location('ovis_replay', SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class Sha:
    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReplayControls(unittest.TestCase):
    def test_frozen_helpers_and_trace_source(self):
        ab, model = replay.source_helpers()
        self.assertEqual(len(ab.ARM_GATES), 6)
        self.assertEqual(len(model.NATIVE_TARGETS), 5)
        self.assertEqual(Sha.sha(ROOT / 'tests/test_qwen35_delta.cpp'), replay.TRACE_DELTA_TEST_SHA)

    def test_unreviewed_or_different_preset_cannot_execute(self):
        ab, _ = replay.source_helpers()
        if replay.APPROVED_PRESET is None:
            with self.assertRaisesRegex(ValueError, 'reviewed A/B preset'):
                replay.approved_preset('baseline', ab)
        else:
            wrong = next(name for name in ab.ARM_GATES if name != replay.APPROVED_PRESET)
            with self.assertRaisesRegex(ValueError, 'differs from reviewed'):
                replay.approved_preset(wrong, ab)

    def test_delta_repeated_positive_variants_are_separate_and_hashed(self):
        stages = ('prefill.delta.output', 'prefill.delta.final_state',
                  'cached.delta.output', 'cached.delta.final_state')
        log = '\n'.join('PASS ' + stage + ': cos_min=1' for stage in stages)
        log += '\nPASS DELTA_RAW_DUMP_INTEGRITY stages=4 variants=linear-core,reset\n'
        shapes = {stage: [2] for stage in stages}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for stage in stages:
                (folder / (stage + '.f32')).write_bytes(b'\0' * 8)
            for variant, names in {
                    'linear-core': ('prefill.delta.final_state', 'cached.delta.final_state'),
                    'reset': ('prefill.delta.output', 'prefill.delta.final_state')}.items():
                (folder / variant).mkdir()
                for name in names:
                    (folder / variant / (name + '.f32')).write_bytes(b'\1' * 8)
            result = replay.raw_vector_manifest(folder, shapes, log, 'f32-delta', Sha)
            self.assertEqual(result['raw_vector_count'], 8)
            self.assertEqual(result['variant_stage_count'], 4)
            self.assertEqual({row['variant'] for row in result['files']},
                             {'primary', 'linear-core', 'reset'})
            (folder / 'reset' / 'unknown.f32').write_bytes(b'\0' * 8)
            with self.assertRaisesRegex(ValueError, 'variant dumps differ'):
                replay.raw_vector_manifest(folder, shapes, log, 'f32-delta', Sha)

    def test_missing_actual_dump_and_missing_marker_rejected(self):
        log = ('PASS prefill.delta.output: cos_min=1\n'
               'PASS cached.delta.output: cos_min=1\n')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'integrity marker'):
                replay.raw_vector_manifest(Path(tmp), {}, log, 'f16-delta', Sha)

    def test_context_requires_frozen_ids_text_and_eos(self):
        log = ('IDS raw:' + ''.join(' ' + str(x) for x in replay.FROZEN_IDS) + '\n'
               'UTF8_HEX raw:' + replay.FROZEN_TEXT.encode().hex() + '\n'
               'PASS OVIS_CONTEXT raw stop=1\nPASS OVIS_CONTEXT repeat stop=1\n')
        self.assertTrue(replay.verify_arm_log('f16-context', log, {}, Sha)['comparison_coverage'])
        with self.assertRaisesRegex(ValueError, 'context IDs/text'):
            replay.verify_arm_log('f16-context', log.replace(' 248046', ' 248044'), {}, Sha)


if __name__ == '__main__':
    unittest.main()
