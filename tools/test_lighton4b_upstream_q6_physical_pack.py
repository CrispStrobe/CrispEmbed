"""Model-free controls for lossless full-Q6 transport and source admission."""
import hashlib
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import lighton4b_upstream_q6_physical_pack as pack


class PhysicalQ6Controls(unittest.TestCase):
    def test_full_descriptor_identity(self):
        rows = {}
        fixture = pack.ROOT / 'tests/fixtures/lightonocr3-4b-source/upstream-q6-physical-schema.tsv'
        for line in fixture.read_text().splitlines():
            name, kind, *shape = line.split()
            shape = list(map(int, shape))
            while len(shape) > 1 and shape[-1] == 1:
                shape.pop()
            if name.endswith('.linear_attn.conv1d.weight'):
                self.assertEqual(shape, [4, 1, 8192])
                shape = [4, 8192]
            rows[pack.mapping.physical_name(name)] = {'native_name': name, 'type': int(kind), 'upstream_shape': shape}
        self.assertEqual(len(rows), 426)
        self.assertEqual(sum(r['type'] == 14 for r in rows.values()), 249)
        self.assertEqual(hashlib.sha256(pack.json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                         pack.metadata_strings()['qwen35_ocr.upstream_physical.mapping_sha256'])

    def test_small_source_values_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a', Path(d) / 'b'
            raw = bytes(range(177))
            a.write_bytes(raw)
            b.write_bytes(raw)
            small = {f'small{i}': {'type': 0, 'stored_shape': [1], 'bytes': 1, 'data_offset': i} for i in range(177)}
            quant = {f'quant{i}': {'type': 14, 'stored_shape': [256, 1], 'bytes': 210, 'data_offset': 0} for i in range(249)}
            up = {'tensors': {**small, **quant}}
            old = {'tensors': {k: dict(v) for k, v in up['tensors'].items()}}
            self.assertEqual(len(pack.validate_small_identity(a, up, b, old)), 177)
            b.write_bytes(raw[:91] + b'\xff' + raw[92:])
            with self.assertRaisesRegex(ValueError, 'actual small source parameter differs'):
                pack.validate_small_identity(a, up, b, old)
            b.write_bytes(raw)
            old['tensors']['small0']['type'] = 14
            with self.assertRaisesRegex(ValueError, 'small tensor type/size'):
                pack.validate_small_identity(a, up, b, old)
            old['tensors']['small0']['type'] = 0
            old['tensors']['quant0']['stored_shape'] = [128, 2]
            with self.assertRaisesRegex(ValueError, 'tensor geometry'):
                pack.validate_small_identity(a, up, b, old)

    def test_report_aliases_cannot_overwrite_models_or_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = [root / name for name in ('base', 'upstream', 'q4')]
            for i, path in enumerate(sources):
                path.write_bytes(bytes([i]) * 32)
            output = root / 'candidate'
            partial = root / 'candidate.partial'
            # Missing candidate paths matter too: report==output would otherwise
            # overwrite the fully converted model with JSON on success.
            for report in [*sources, output, partial, root / '.' / 'candidate']:
                with self.subTest(report=report.name), patch.object(pack, 'digest') as digest:
                    with self.assertRaisesRegex(ValueError, 'must not alias'):
                        pack.convert(*sources[:2], output, report, sources[2])
                    digest.assert_not_called()
                self.assertFalse(output.exists())
                self.assertFalse(partial.exists())
                for i, path in enumerate(sources):
                    self.assertEqual(path.read_bytes(), bytes([i]) * 32)
            for link_type in ('symlink', 'hardlink'):
                for i, source in enumerate(sources):
                    alias = root / f'{link_type}-{i}'
                    if link_type == 'symlink':
                        alias.symlink_to(source)
                    else:
                        alias.hardlink_to(source)
                    with self.subTest(link_type=link_type, source=source.name):
                        with self.assertRaisesRegex(ValueError, 'must not alias'):
                            pack.convert(*sources[:2], output, alias, sources[2])
                        self.assertEqual(source.read_bytes(), bytes([i]) * 32)
            # Keep ordinary failed-conversion diagnostics when paths are safe.
            report = root / 'report.json'
            with self.assertRaisesRegex(ValueError, 'whole protected source identity'):
                pack.convert(*sources[:2], output, report, sources[2])
            self.assertEqual(pack.json.loads(report.read_text())['status'], 'failed')

    def test_whole_q6_row_transport_and_inverse(self):
        raw = b''.join(bytes([i]) * 420 for i in range(4))
        source, target = io.BytesIO(raw), io.BytesIO()
        order = [2, 0, 3, 1]
        actual, permuted = pack.transport_tensor(source, target, 0, 0, len(raw), order, 14)
        expected = b''.join(raw[i * 420:(i + 1) * 420] for i in order)
        self.assertTrue(permuted)
        self.assertEqual(target.getvalue(), expected)
        self.assertEqual(actual, hashlib.sha256(expected).hexdigest())
        inverse = [order.index(i) for i in range(4)]
        restored = io.BytesIO()
        pack.transport_tensor(io.BytesIO(expected), restored, 0, 0, len(raw), inverse, 14)
        self.assertEqual(restored.getvalue(), raw)
        with self.assertRaisesRegex(ValueError, 'whole quantized blocks only'):
            pack.transport_tensor(io.BytesIO(b'x' * 1024), io.BytesIO(), 0, 0, 1024, order, 14)


if __name__ == '__main__':
    unittest.main()
