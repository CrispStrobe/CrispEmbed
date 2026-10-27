#!/usr/bin/env python3
"""Stdlib controls for the MoEdern 27M/115M public dispatch. No model, no C++ build."""
from pathlib import Path
import importlib.util
import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


pack = module('moedern_family_pack', 'tools/pack_moedern_text_f32.py')
guard = module('moedern_family_guard', 'tools/moedern-public/family_contract.py')
REVISIONS = {'27M': 'd2b7b1feaaac1bf87510ca9525208c3daeabcc5a', '115M': '66380d2ade3b5027bba83bdcdba052d0bb86dcdb'}


def params(H, heads, kvh, hsh, E, X, conv):
    V, KV = 30002, kvh*(H//heads); n, t = hsh*V+hsh*H+2*H, 4
    for d in conv:
        if d: n += H+3*H*H+2*H+3*H+H*H; t += 6
        else: n += 2*H+2*H*H+2*H*KV+H*E+3*H*X*E+3*H*X; t += 13
    return n+H+128*H, t+2


def tiny(path, count):
    meta = pack.scalar('general.architecture', 'modernbert_hash_moe_v10')
    table = b''.join(pack.string('t%d' % i)+struct.pack('<IQIQ', 1, 1, 0, 32*i) for i in range(count))
    header = b'GGUF'+struct.pack('<IQQ', 3, count, 1)+meta+table; header += b'\0'*(-len(header) % 32)
    path.write_bytes(header+b''.join(struct.pack('<f', 1.0)+b'\0'*28 for _ in range(count))[:32*(count-1)+4])


class Controls(unittest.TestCase):
    def test_tokenizer_identical_across_sizes(self):
        files = {k: v['original']['files']['tokenizer.json'] for k, v in json.loads((ROOT/'docs/moedern-colbert-source-contract.json').read_text())['variants'].items()}
        self.assertEqual(set(files), {'1.5M', '27M', '115M'})
        self.assertEqual(len({(f['bytes'], f['git_blob_sha1'], f['sha256']) for f in files.values()}), 1)

    def test_family_pins_match_frozen_originals(self):
        shapes = {'27M': (640, 10, 5, 80, 8, 640, [0, 0]), '115M': (1024, 16, 8, 192, 8, 640, [1, 1, 0, 2, 1, 0, 1, 0, 2, 0])}
        header = (ROOT/'src/moedern_packed_profile.h').read_text()
        for variant, pin in pack.FAMILY.items():
            original = json.loads((ROOT/f'tests/results/moedern-colbert/original-{variant.lower()}.json').read_text())
            self.assertEqual((pin['revision'], original['revision']), (REVISIONS[variant], REVISIONS[variant]))
            self.assertEqual(pin['model'], {'bytes': original['conversion']['gguf_bytes'], 'sha256': original['conversion']['gguf_sha256']})
            self.assertEqual(pin['original_report_sha256'], original['original_report_sha256'])
            n, t = params(*shapes[variant])
            self.assertEqual((pin['parameters'], pin['tensors'], pin['bounds']['tensors']), (n, t, t))
            self.assertEqual(t, original['conversion']['tensor_count'])
            self.assertLess(n*4, pin['model']['bytes']); self.assertLessEqual(pin['model']['bytes']+pack.TOKENIZER['bytes'], pin['bounds']['file'])
            for value in (pin['revision'], pin['model']['sha256'], pin['original_report_sha256'], str(n)):
                self.assertIn(value, header)
        self.assertEqual(params(128, 2, 1, 16, 8, 128, [0, 0]), (1484448, 32))

    def test_one_five_packer_constants_unchanged(self):
        cfg = json.loads(guard.PATH.read_text())
        self.assertEqual(guard.packer_constants(), cfg['packer_one_five_constants'])
        self.assertEqual(pack.BOUNDS, {'file': 16*pack.CHUNK, 'tensors': 32, 'elements': 2*pack.CHUNK})

    def test_bounds_are_per_profile(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'m.gguf'; tiny(path, 33)
            with self.assertRaisesRegex(ValueError, 'metadata/tensor count'): pack.parse(path)
            self.assertEqual(len(pack.parse(path, pack.FAMILY['115M']['bounds'])['tensors']), 33)
            tiny(path, 2); self.assertEqual(len(pack.parse(path)['tensors']), 2)

    def test_family_pack_hosted_only_and_pinned(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'hosted'): pack.pack_family('27M', 'm', 't', 'r', 'o')
        with self.assertRaisesRegex(ValueError, 'unknown family'): pack.pack_family('1.5M', 'm', 't', 'r', 'o')
        with patch.dict(os.environ, {'KAGGLE_KERNEL_RUN_TYPE': 'Batch'}), tempfile.TemporaryDirectory() as d:
            report = Path(d)/'original.json'; report.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'frozen original report'): pack.pack_family('27M', 'm', 't', report, Path(d)/'o')

    def test_runtime_dispatch_is_gated(self):
        frontend = (ROOT/'src/moedern_text_frontend.cpp').read_text()
        text = (ROOT/'src/moedern_text.cpp').read_text()
        self.assertIn('packed && core_env::on("CRISPEMBED_MOEDERN_FAMILY")', frontend)
        self.assertIn('moedern_packed::file_bounds(path, error)) return false;', frontend)  # unchanged 1.5M default
        for r in REVISIONS.values(): self.assertIn(r, text)
        self.assertEqual(text.count('core_env::on("CRISPEMBED_MOEDERN_FAMILY")'), 2)
        self.assertIn('CRISPEMBED_MOEDERN_FAMILY_ENGINE', (ROOT/'CMakeLists.txt').read_text())

    def test_closure(self):
        cfg = guard.source(); self.assertFalse(cfg['default_enabled'])


if __name__ == '__main__': unittest.main()
