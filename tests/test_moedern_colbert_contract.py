#!/usr/bin/env python3
"""Small dependency-free controls; no tokenizer/model/framework loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/moedern-colbert'))
from contract import PANEL, read_contract, safetensors_inventory, tensor_schema, verify_file


class ContractTests(unittest.TestCase):
    def test_dynamic_schema(self):
        for name, declaration in read_contract()['variants'].items():
            rows = tensor_schema(declaration['config'])
            self.assertEqual(len(rows), 94 if name == '115M' else 32)
            self.assertEqual(len({r['source'] for r in rows}), len(rows))
            self.assertEqual(len({r['target'] for r in rows}), len(rows))
            self.assertEqual(next(r['shape'] for r in rows if r['target'] == 'dense_head.weight'), [128, declaration['config']['hidden_size']])
            self.assertEqual(sum(r['transform'] == 'expert_transpose' for r in rows), 12 if name == '115M' else 6)
            self.assertEqual(sum(r['transform'] == 'depthwise_transpose' for r in rows), 6 if name == '115M' else 0)

    def test_source_and_weight_pins(self):
        for declaration in read_contract()['variants'].values():
            for role in ['original', 'author_gguf']:
                model = declaration[role]
                self.assertRegex(model['revision'], r'^[a-f0-9]{40}$')
                self.assertEqual(model['files']['LICENSE']['sha256'], 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30')
                for name, identity in model['files'].items():
                    if name.endswith(('.safetensors', '.gguf')):
                        self.assertGreater(identity['bytes'], 0)
                        self.assertRegex(identity['sha256'], r'^[a-f0-9]{64}$')
                self.assertEqual(model['files']['tokenizer.json']['git_blob_sha1'], 'e1fc54fd0e0456a6c76106be4a338bfc08f29861')

    def test_payload_identity_not_just_name(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'sample'
            p.write_bytes(b'abc')
            identity = {'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest(),
                        'git_blob_sha1': hashlib.sha1(b'blob 3\0abc').hexdigest()}
            verify_file(p, identity)
            p.write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_file(p, identity)
            identity.pop('sha256')
            with self.assertRaisesRegex(ValueError, 'Git blob'):
                verify_file(p, identity)
            p.write_bytes(b'abcd')
            with self.assertRaisesRegex(ValueError, 'size'):
                verify_file(p, identity)

    def test_header_exact_f32_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'weights'
            def write(header, payload=b'\0'*8):
                data = json.dumps(header).encode()
                p.write_bytes(struct.pack('<Q', len(data))+data+payload)
            good = {'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}
            write(good)
            self.assertEqual(safetensors_inventory(p)['a']['shape'], [2])
            for header, payload in [({'a': {'dtype': 'F16', 'shape': [2], 'data_offsets': [0, 8]}}, b'\0'*8),
                                    ({'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [4, 12]}}, b'\0'*12),
                                    (good, b'\0'*9), (good, b'\0'*7)]:
                write(header, payload)
                with self.assertRaises(ValueError):
                    safetensors_inventory(p)

    def test_original_required_before_converter_dependency_imports(self):
        from convert_f32 import convert
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'receipt.json'
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt SHA'):
                convert(directory, '1.5M', Path(directory)/'out.gguf', p, '0'*64)
            with self.assertRaisesRegex(ValueError, 'identity/completion'):
                convert(directory, '1.5M', Path(directory)/'out.gguf', p, hashlib.sha256(p.read_bytes()).hexdigest())

    def test_owned_panel_before_any_candidate(self):
        panel = json.loads(PANEL.read_text())
        self.assertEqual(len(panel['cases']), 10)
        self.assertEqual(panel['full_capture'], ['q-de', 'd-de'])
        self.assertEqual(len({r['id'] for r in panel['cases']}), 10)
        self.assertTrue(any('[Q] [D] [MASK]' in r['text'] for r in panel['cases']))
        self.assertGreater(len(next(r['text'] for r in panel['cases'] if r['id'] == 'd-truncate')), 3000)


if __name__ == '__main__':
    unittest.main()
