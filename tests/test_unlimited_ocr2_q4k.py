"""Small stdlib controls only; no model, framework, tokenizer or compilation."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import http.server
import shutil
import socket
import subprocess
import threading
from unittest.mock import patch
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/unlimited-ocr2-q4k'))
import contract
import overlay
import adapter
import panel
import build_profile


def inventory():
    rows = {}
    for layer in range(1, 12):
        for part in ['gate', 'up']:
            rows[f'l.blk.{layer}.ffn_{part}_exps.weight'] = dict(type=1, shape=[1280, 896, 64],
                bytes=1280 * 896 * 64 * 2, sha256='1' * 64)
        rows[f'l.blk.{layer}.ffn_down_exps.weight'] = dict(type=1, shape=[896, 1280, 64],
            bytes=1280 * 896 * 64 * 2, sha256='2' * 64)
    rows['v.patch.weight'] = dict(type=1, shape=[16, 16, 3, 768], bytes=16 * 16 * 3 * 768 * 2, sha256='3' * 64)
    rows['l.output_norm.weight'] = dict(type=1, shape=[1280], bytes=1280 * 2, sha256='4' * 64)
    return rows


def candidate(source, recipe):
    out = copy.deepcopy(source)
    for name in contract.selected(source, recipe):
        out[name].update(type=12, bytes=1280 * 896 * 64 // 256 * 144, sha256='a' * 64)
    return out


class Q4Contract(unittest.TestCase):
    def test_retained_bootstrap_uses_proven_uv_without_models_or_builds(self):
        source = (ROOT / 'tools/kaggle/unlimited-ocr2-q4k/retained_kernel.py').read_text()
        self.assertIn('uv==0.8.17', source)
        self.assertIn("'--python', '3.12', '--seed'", source)
        self.assertNotIn('EnvBuilder', source)
        self.assertNotIn("['cmake'", source)
        self.assertIn("'native_forwards'", source.replace('native_forwards=0', "'native_forwards'"))

    def test_trailing_unit_rank_change_preserves_semantics_and_all_payload_bytes(self):
        source = inventory()
        source['v.pos_embed'] = dict(type=1, shape=[3145728, 1], bytes=6291456, sha256='5' * 64)
        target = candidate(source, 'last1')
        target['v.pos_embed']['shape'] = [3145728]
        proof = contract.protected(source, target, 'last1')
        self.assertTrue(proof['all_protected_byte_exact'])
        self.assertEqual(proof['physical_rank_canonicalizations'][0]['semantic_shape'], [3145728, 1, 1, 1])
        for key, value in [('shape', [1, 3145728]), ('shape', [1572864, 2]),
                           ('type', 0), ('bytes', 3145728), ('sha256', '6' * 64)]:
            bad = copy.deepcopy(target)
            bad['v.pos_embed'][key] = value
            with self.assertRaises(ValueError):
                contract.protected(source, bad, 'last1')

    def test_rank_expansion_and_invalid_axes_are_not_canonicalization(self):
        source = inventory()
        target = candidate(source, 'last1')
        target['l.output_norm.weight']['shape'] = [1280, 1]
        with self.assertRaisesRegex(ValueError, 'canonicalization'):
            contract.protected(source, target, 'last1')
        for shape in ([], [True], [0], [-1], [1, 1, 1, 1, 1]):
            with self.assertRaises(ValueError):
                contract.semantic_shape(shape)

    def test_actual_vocabulary_width_is_not_tokenizer_size_or_rounded_bound(self):
        self.assertEqual(panel.stage_axes('logits',[1,1,129280]),[1,1,129280])
        self.assertEqual(panel.stage_axes('layer_0_chunk_0',[2048,1,1280]),[2048,1,1280])
        for bad in ([1,1,129000],[1,1,128831],[1,1,129281],[129280,1,1],[1,1,True],[1,129280],[1,1,0]):
            with self.assertRaises(ValueError):panel.stage_axes('logits',bad)
        with self.assertRaises(ValueError):panel.stage_axes('layer_0_chunk_0',[2048,2048,1280])

    def test_frozen_original_ids_text_scope(self):
        refs, images = contract.references()
        self.assertEqual([len(refs[x]['ids']) for x in contract.CASES], [146, 138, 306, 164])
        self.assertEqual(set(refs), set(images))
        self.assertTrue(all(row['text'] for row in refs.values()))

    def test_actual_type_counts_and_expert_down_protection(self):
        source = inventory()
        for recipe, count in [('last1', 2), ('last4', 8), ('all11', 22)]:
            target = candidate(source, recipe)
            self.assertEqual(contract.protected(source, target, recipe)['actual_q4_k_stacked_tensors'], count)
            rules = contract.overrides(source, recipe)
            self.assertEqual(len(rules), len(source) - count)
            self.assertTrue(all('*' not in rule for rule in rules))
            self.assertTrue(all(not any(rule.startswith(name + '=') for rule in rules)
                                for name in contract.selected(source, recipe)))

    def test_rejects_down_fallback_or_critical_changes(self):
        source = inventory()
        for field, value in [('type', 2), ('shape', [896, 1280, 63]), ('sha256', 'b' * 64)]:
            target = candidate(source, 'last1')
            target['l.blk.1.ffn_down_exps.weight'][field] = value
            with self.assertRaises(ValueError):
                contract.protected(source, target, 'last1')
        for field, value in [('type', 2), ('shape', [1280, 896, 63]), ('bytes', 1)]:
            target = candidate(source, 'last1')
            target['l.blk.11.ffn_gate_exps.weight'][field] = value
            with self.assertRaises(ValueError):
                contract.protected(source, target, 'last1')

    def test_exact_names_and_four_axis_f32_are_fail_closed(self):
        source = inventory()
        target = candidate(source, 'last1')
        target.pop('l.output_norm.weight')
        with self.assertRaises(ValueError):
            contract.protected(source, target, 'last1')
        source['v.patch.weight']['type'] = 0
        with self.assertRaisesRegex(ValueError, 'flattened'):
            contract.overrides(source, 'last1')
        with self.assertRaises(ValueError):
            contract.selected({}, 'last1')

    def test_overlay_source_authentication_and_retained_connections(self):
        original = (ROOT / 'src/unlimited_ocr.cpp').read_bytes()
        patched = overlay.transform(original)
        self.assertEqual(original, (ROOT / 'src/unlimited_ocr.cpp').read_bytes())
        self.assertTrue(patched.startswith(b'#include <stdexcept>\n'))
        self.assertIn(b'for (ggml_tensor * retained', patched)
        self.assertIn(b'ggml_set_output(retained)', patched)
        self.assertIn(b'ggml_set_output(xn)', patched)
        self.assertNotIn(b'ggml_view_tensor(', overlay.HELPER.encode())
        with self.assertRaises(ValueError):
            overlay.transform(original + b'\n')
        with self.assertRaises(ValueError):
            overlay.replace_once('x x', 'x', 'y')

    def test_lossless_adapter_and_wrong_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = root / 'source.f32'
            p.write_bytes(struct.pack('<4f', -0.0, 1, 2, 3))
            item = dict(name='hn', path=p, shape=[2, 1, 2], sha256=contract.digest(p))
            receipt = adapter.write_gguf(root / 'a.gguf', [item])
            self.assertEqual(receipt['tensors']['hn']['sha256'], contract.digest(p))
            self.assertEqual(receipt['tensors']['hn']['shape'], [2, 1, 2])
            with self.assertRaises(ValueError):
                adapter.write_pair(root / 'b.gguf', root / 'c.gguf', [dict(name='hn', original=item, native=dict(item, shape=[1, 2, 2]))])
            p.write_bytes(struct.pack('<4f', float('nan'), 1, 2, 3))
            with self.assertRaises(ValueError):
                adapter.write_gguf(root / 'd.gguf', [dict(item, sha256=contract.digest(p))])

    def test_cli_trace_is_exact_and_cap_is_not_eos_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'stdout').write_text('raw\n\n')
            (path / 'stderr').write_text('[dbg] gen_ids (2): 1 2\n')
            parsed = panel.parsed(path)
            self.assertEqual(parsed['text'], 'raw\n')
            self.assertEqual(parsed['ids'], [1, 2])
            self.assertFalse(parsed['eos_observed'])
            (path / 'stderr').write_text('[dbg] gen_ids (2): 1 2\n[dbg] gen_ids (2): 1 2\n')
            with self.assertRaises(ValueError):
                panel.parsed(path)

    def test_quality_keeps_coordinate_and_numeric_gates_separate(self):
        reference = dict(ids=[1, 2], text='<PAGE><BLOCK>Hello[1,2,30,40]<FORK>')
        actual = dict(ids=[1, 3], text='<PAGE><BLOCK>Hello[2,2,30,40]<FORK>', token_cap_reached=False)
        quality = panel.task_quality(actual, reference)
        self.assertTrue(quality['task_completed'])
        self.assertFalse(quality['ids_exact'])
        self.assertFalse(quality['raw_text_exact'])
        actual['text'] = actual['text'].replace('Hello', 'Wrong')
        self.assertFalse(panel.task_quality(actual, reference)['task_completed'])
        actual['text'] = '<PAGE><BLOCK>Hello[5,2,30,40]<FORK>'
        self.assertFalse(panel.task_quality(actual, reference)['task_completed'])

    def test_hosted_real_ref_controls_and_first_f16_barrier(self):
        source = (ROOT / 'tools/unlimited-ocr2-q4k/panel.py').read_text()
        barrier = source.index("report['f16_all_four_strict_control_passed'] = True")
        self.assertLess(source.index('for case in CASES:'), barrier)
        self.assertGreater(source.index('for recipe, layers in RECIPES.items():'), barrier)
        kernel = (ROOT / 'tools/kaggle/unlimited-ocr2-q4k/kernel.py').read_text()
        self.assertLess(kernel.index('controls.py'), kernel.index("model = Path('/kaggle/temp/unlimited-f16.gguf')"))
        self.assertIn('accepted=(0, 1)', kernel)
        cpp = (ROOT / 'tests/unlimited-q4k/diff.cpp').read_text()
        self.assertIn('return failed ? 1 : 0;', cpp)
        self.assertIn('core_util::clean_exit', cpp)
        self.assertIn('0.002', cpp)
        self.assertIn('0.99999', cpp)

    def test_build_flags_reject_backend_drift(self):
        self.assertEqual(build_profile.flags(['cc', '-DGGML_AVX2', '-mavx2', '-O3', '-c', 'one.cpp']),
                         build_profile.flags(['cc', '-O3', '-mavx2', '-DGGML_AVX2', '-c', 'two.cpp']))
        with self.assertRaises(ValueError):
            build_profile.one([dict(file='one.cpp', arguments=['cc', '-c', 'one.cpp'])], 'ggml-cpu.cpp')
        with self.assertRaises(ValueError):
            build_profile.one([dict(file='one.cpp', arguments=['cc', '-c', 'one.cpp'])] * 2, 'one.cpp')

    def test_bounded_http1_transfer_keeps_full_authentication(self):
        args = contract.model_download_command('/hosted/task/model.gguf')
        self.assertIn('--http1.1', args)
        self.assertIn('--retry-all-errors', args)
        self.assertEqual(args[args.index('--retry') + 1], '2')
        self.assertEqual(args[args.index('--retry-max-time') + 1], '1200')
        self.assertEqual(args[args.index('--max-time') + 1], '600')
        self.assertEqual(args[args.index('--connect-timeout') + 1], '30')
        self.assertNotIn('--continue-at', args)
        self.assertIn(contract.F16_URL, args)
        self.assertEqual(args[-2:], ['--output', '/hosted/task/model.gguf'])
        kernel = (ROOT / 'tools/kaggle/unlimited-ocr2-q4k/kernel.py').read_text()
        transfer = kernel.index('run(model_download_command(model), 2400)')
        auth = kernel.index('model.stat().st_size == F16_BYTES and sha(model) == F16_SHA')
        self.assertLess(transfer, auth)
        self.assertLess(auth, kernel.index("'tools/unlimited-ocr2/replay_real_decode.py'"))
        self.assertLess(auth, kernel.index("'tools/unlimited-ocr2-q4k/panel.py'"))

    @unittest.skipUnless(shutil.which('curl'), 'curl unavailable')
    def test_real_small_http_retry_replaces_incomplete_output(self):
        payload = b'exact-small-transfer-fixture'
        calls = []
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_GET(self):
                calls.append(self.request_version)
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[:3] if len(calls) == 1 else payload)
                self.wfile.flush()
                if len(calls) == 1:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                destination = Path(folder) / 'bytes'
                url = 'http://127.0.0.1:' + str(server.server_port) + '/fixture'
                with patch.object(contract, 'F16_URL', url):
                    result = subprocess.run(contract.model_download_command(destination),
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(calls, ['HTTP/1.1', 'HTTP/1.1'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_deadline_rejects_before_child_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'never-created'
            panel.DEADLINE = time.monotonic() - 1
            try:
                with self.assertRaisesRegex(ValueError, 'deadline'):
                    panel.command(['this-must-never-launch'], path)
                self.assertFalse(path.exists())
            finally:
                panel.DEADLINE = None


if __name__ == '__main__':
    unittest.main()
