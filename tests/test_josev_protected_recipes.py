#!/usr/bin/env python3
"""Bounded corrupt-storage and immutable development-contract controls."""
import copy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import quantize_josev2b_protected as q
import josev2b_protected_contract as contract


def fixture(path, rows, identity=None):
    def text(s):
        return struct.pack('<Q', len(s.encode())) + s.encode()
    metadata = [('general.architecture', 8, 'josev2b_systemone'),
                ('general.license', 8, 'apache-2.0'),
                ('josev2b.packed.format', 8, q.FORMAT if identity else q.old.OLD_FORMAT),
                ('general.file_type', 4, (7 if identity.startswith('q8-') else 14) if identity else 0),
                ('general.quantization_version', 4, 2), ('tokenizer.fixture', 8, 'Ä [MASK]')]
    header = b'GGUF' + struct.pack('<IQQ', 3, len(rows), len(metadata))
    for name, kind, value in metadata:
        header += text(name) + struct.pack('<I', kind)
        header += text(value) if kind == 8 else struct.pack('<I', value)
    wanted = q.recipe(rows, identity) if identity else {n: 0 for n in rows}
    payload = bytearray()
    for name, shape in rows.items():
        offset = (len(payload) + 31) // 32 * 32
        payload.extend(bytes(offset - len(payload)))
        kind, n = wanted[name], math.prod(shape)
        if kind == 0:
            blob = struct.pack('<f', 1.) * n
        elif kind == 1:
            blob = struct.pack('<e', 1.) * n
        elif kind == 8:
            blob = (struct.pack('<e', 1.) + bytes([1]) * 32) * (n // 32)
        else:
            blob = (struct.pack('<ee', 1., 0.) + bytes(140)) * (n // 256)
        header += text(name) + struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *shape)
        header += struct.pack('<IQ', kind, offset)
        payload.extend(blob)
    path.write_bytes(header + bytes((-len(header)) % 32) + payload)


class Controls(unittest.TestCase):
    def authored(self, identity):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        base = Path(td.name)
        rows = {n: ([256, 1] if q.old.projection(n) else [1]) for n in q.schema()}
        source, output = base / 'source.gguf', base / 'candidate.gguf'
        fixture(source, rows)
        fixture(output, rows, identity)
        return rows, source, output

    def verify(self, rows, source, output, identity):
        with patch.object(q, 'PARAMETERS', sum(math.prod(s) for s in rows.values())):
            return q.verify_conversion(source, output, rows, identity)

    def test_four_plans_and_only_last_layer_q4(self):
        rows = q.schema()
        for identity, counts in q.COUNTS.items():
            plan = q.recipe(rows, identity)
            self.assertEqual({t: sum(v == t for v in plan.values()) for t in q.TYPES}, counts)
            self.assertTrue(all(plan[n] == 0 for n in rows if not q.old.projection(n)))
            if identity.startswith('q4-'):
                self.assertEqual({n for n, t in plan.items() if t == 12}, {
                    'l.layers.23.mlp.gate_proj.weight', 'l.layers.23.mlp.up_proj.weight'})
        with self.assertRaises(ValueError):
            q.recipe(rows, 'q8-guessed')

    def test_all_storage_variants_verify_without_mutation(self):
        for identity in q.COUNTS:
            rows, source, output = self.authored(identity)
            before = q.digest(output)
            result = self.verify(rows, source, output, identity)
            self.assertTrue(result['critical292_f32_byte_exact'])
            self.assertEqual(q.digest(output), before)
            self.assertEqual(result['counts'], {q.TYPES[t]: c for t, c in q.COUNTS[identity].items()})

    def test_critical_and_half_corruption_reject(self):
        for kind in (0, 1):
            identity = 'q4-last-f32-down'
            rows, source, output = self.authored(identity)
            parsed = q.parse(output)
            row = next(r for r in parsed['tensors'].values() if r['type'] == kind)
            with output.open('r+b') as f:
                f.seek(row['data_offset'])
                f.write(struct.pack('<f' if kind == 0 else '<e', 2.))
            with self.assertRaisesRegex(ValueError, 'F32 bytes|F16 rounding'):
                self.verify(rows, source, output, identity)

    def test_nonfinite_q8_q4_and_source_reject(self):
        for identity, target in [('q8-all', 'packed'), ('q4-last-f32-down', 'packed'),
                                 ('q8-all', 'source')]:
            rows, source, output = self.authored(identity)
            path = output if target == 'packed' else source
            parsed = q.parse(path)
            kind = 8 if identity == 'q8-all' else 12
            row = next(r for r in parsed['tensors'].values() if r['type'] == kind) if target == 'packed' else next(iter(parsed['tensors'].values()))
            with path.open('r+b') as f:
                f.seek(row['data_offset'])
                f.write(struct.pack('<e' if target == 'packed' else '<f', float('inf')))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.verify(rows, source, output, identity)

    def test_q8_mislabeled_q4_and_wrong_recipe_reject(self):
        rows, source, output = self.authored('q8-all')
        header = q.parse(output)
        with output.open('r+b') as f:
            f.seek(header['metadata']['general.file_type']['offset'])
            f.write(struct.pack('<I', 14))
        with self.assertRaisesRegex(ValueError, 'typed metadata'):
            self.verify(rows, source, output, 'q8-all')
        rows, source, output = self.authored('q8-f32-down')
        with self.assertRaisesRegex(ValueError, 'recipe type'):
            self.verify(rows, source, output, 'q8-all')

    def test_contract_rejects_budget_recipe_or_reference_changes(self):
        cfg = contract.validate(json.loads(contract.PATH.read_text()))
        for key in ('holdout_access', 'recipe_selected', 'publication_qualified'):
            bad = copy.deepcopy(cfg)
            bad[key] = True
            with self.assertRaises(ValueError):
                contract.validate(bad)
        bad = copy.deepcopy(cfg)
        bad['gates']['ordinal_expectation_absolute'] = .002
        with self.assertRaises(ValueError):
            contract.validate(bad)
        for name in ('tools/quantize.cpp', 'src/josev2b_packed_profile.h', 'tools/quantize_josev2b_mixed.py'):
            bad = copy.deepcopy(cfg)
            bad['protected_sha256'].pop(name)
            with self.assertRaisesRegex(ValueError, 'closure'):
                contract.validate(bad)


if __name__ == '__main__':
    unittest.main()
