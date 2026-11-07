import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).parent))
from worker import validate_cache

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('attention_admission', ROOT / 'admission.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AttentionAdmission(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT / 'protocol-v1.json').read_text())

    def test_frozen_protocol(self):
        a.validate(self.p)
        self.assertEqual(sum(r['bytes'] for r in self.p['tensors']), 67237888)

    def test_mutations_rejected(self):
        changes = [('layer', 0), ('tokens', [16]), ('threads', 8), ('context', 16),
                   ('visible_kv_padded', 16), ('model_forward_allowed', True),
                   ('flash_attention', True), ('cache_k_type', 'f32'),
                   ('total_timeout_seconds', 0), ('physical_report_sha256', '0' * 64)]
        for field, value in changes:
            with self.subTest(field=field):
                p = copy.deepcopy(self.p)
                p[field] = value
                with self.assertRaises(ValueError):
                    a.validate(p)

    def test_actual_build_profiles(self):
        flags = {'GGML_LLAMAFILE': 'ON', 'GGML_AVX2': 'ON', 'GGML_NATIVE': 'OFF',
                 'GGML_AVX512': 'OFF', 'GGML_BLAS': 'OFF', 'CRISPEMBED_NATIVE': 'OFF'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'CMakeCache.txt'
            path.write_text('\n'.join(f'{k}:BOOL={v}' for k, v in flags.items()))
            validate_cache(path, runtime=True)
            for key in flags:
                changed = dict(flags)
                changed[key] = 'OFF' if changed[key] == 'ON' else 'ON'
                path.write_text('\n'.join(f'{k}:BOOL={v}' for k, v in changed.items()))
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    validate_cache(path, runtime=True)

    def test_range_mutations(self):
        for field, value in [('bytes', 1), ('type', 1), ('source_offset', -1),
                             ('name', 'l.layers.4.input_layernorm.weight')]:
            with self.subTest(field=field):
                p = copy.deepcopy(self.p)
                p['tensors'][0][field] = value
                with self.assertRaises(ValueError):
                    a.validate(p)


if __name__ == '__main__':
    unittest.main()
