"""Pure source and metric guards for hosted Ovis vision A/B; no model imports."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_vision_ab', ROOT / 'tools/ci-heavy/ovisocr2_vision_audit.py')
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)


def complete_log(fail=None, marker=True):
    names = ['prefill.vision.patch_embed', 'prefill.vision.position',
             'prefill.vision.embedded', 'prefill.vision.rotary_cos',
             'prefill.vision.rotary_sin']
    names += [f'prefill.vision.layer.{i}' for i in range(12)]
    names += [f'prefill.vision.probe.{i}.{field}' for i in (5, 11)
              for field in vision.VISION_PROBES]
    names += ['prefill.vision.merger.norm', 'prefill.vision.merger.up',
              'prefill.vision.merger.activation', 'prefill.vision.merger']
    lines = []
    for name in names:
        status = 'FAIL' if name == fail else 'PASS'
        ratio = '1.5' if name == fail else '0.1'
        lines.extend((f'{status} {name}: cos_min=1 cos_global=1 max_abs=0.001 '
                      f'mean_abs=0.0001 norms=1/1',
                      f'BUDGET {name}: strict_absolute_pass=1 calibrated=1 '
                      f'self_max_abs=0.0001 max_ratio={ratio}'))
    if marker:
        lines.append('PASS VISION_RAW_DUMP_INTEGRITY stages=2')
    return '\n'.join(lines) + '\n'


class OvisVisionAuditControls(unittest.TestCase):
    def test_actual_frozen_logs_identify_operation_before_summary_layer(self):
        for failed in ('prefill.vision.probe.11.norm2', 'prefill.vision.probe.11.up'):
            parsed = vision.parse_vision_log('unused', text=complete_log(failed),
                                             require_dump_integrity=True)
            self.assertEqual(parsed['stage_count'], 37)
            self.assertEqual(parsed['first_failing_operation'], failed)
            self.assertTrue(parsed['actual_dump_integrity_passed'])
        with self.assertRaisesRegex(ValueError, 'integrity marker'):
            vision.parse_vision_log('unused', text=complete_log(marker=False),
                                    require_dump_integrity=True)
        truncated = complete_log().replace(
            'PASS prefill.vision.merger.norm: cos_min=1 cos_global=1 max_abs=0.001 '
            'mean_abs=0.0001 norms=1/1\n'
            'BUDGET prefill.vision.merger.norm: strict_absolute_pass=1 calibrated=1 '
            'self_max_abs=0.0001 max_ratio=0.1\n', '')
        with self.assertRaisesRegex(ValueError, 'coverage incomplete'):
            vision.parse_vision_log('unused', text=truncated, require_dump_integrity=True)

    def test_all_37_actual_vectors_required_and_shape_checked(self):
        parsed = vision.parse_vision_log('unused', text=complete_log(), require_dump_integrity=True)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            shapes = {name: [1] for name in parsed['stages']}
            for name in shapes:
                (folder / (name + '.f32')).write_bytes(struct.pack('<f', 1.0))
            manifest = vision.vector_manifest(folder, shapes, parsed, 'f32-baseline')
            self.assertEqual(manifest['actual_primary_vector_count'], 37)
            self.assertTrue(all(row['provenance'] == 'actual_primary_comparison'
                                for row in manifest['files']))
            (folder / 'prefill.vision.layer.11.f32').unlink()
            with self.assertRaisesRegex(ValueError, 'coverage'):
                vision.vector_manifest(folder, shapes, parsed, 'f32-baseline')

    def test_only_three_vision_gates_vary(self):
        self.assertEqual(len(vision.ARM_GATES), 6)
        options = {name for values in vision.ARM_GATES.values() for name in values}
        self.assertEqual(options, {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
                                   'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
                                   'CRISPEMBED_QWEN35_VISION_CASCADE_NORM'})
        self.assertEqual(vision.BASE_ENV['CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS'], '1')
        self.assertTrue(all(vision.BASE_ENV[name] == '0' for name in options))
        self.assertEqual(vision.MODELS['f32'][2],
                         '7f04dbc597e122fc3f1db2d96a64e8d7759be9f62dca4bc4a472fe5f0db9bfd9')

    def test_no_framework_import(self):
        import sys
        for name in ('torch', 'numpy', 'gguf', 'PIL', 'transformers'):
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main()
