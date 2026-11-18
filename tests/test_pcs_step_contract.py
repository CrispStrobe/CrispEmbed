#!/usr/bin/env python3
"""New model-free controls for PCS step capture and the actual shared reader."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from pcs_step_capture import overlay, pack
from pcs_steps import inventory, write_archive
import numpy as np


class CaptureContract(unittest.TestCase):
    def test_production_expressions_remain_inside_wrappers(self):
        source = (ROOT / 'src/pcs.cpp').read_text()
        observed = overlay(source)
        self.assertIn('ggml_add(ctx0, tok_emb, pos_emb)', observed)
        self.assertIn('ggml_gelu_erf(ctx0, ffn)', observed)
        self.assertIn('ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f)', observed)
        self.assertNotIn('pcs_step_trace', source)
        self.assertEqual(observed.count('steps.set_inputs();'), 1)

    def test_ambiguous_source_refuses(self):
        source = (ROOT / 'src/pcs.cpp').read_text()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            overlay(source + '\n    ggml_tensor * tok_emb = ggml_get_rows(ctx0, ctx.tok_emb_w, inp_ids);')

    def test_missing_source_refuses(self):
        source = (ROOT / 'src/pcs.cpp').read_text().replace('ggml_gelu_erf(ctx0, ffn)', 'changed_gelu(ctx0, ffn)')
        with self.assertRaisesRegex(ValueError, 'anchor'):
            overlay(source)

    def test_inventory_has_unique_actual_operator_outputs(self):
        values = inventory()
        self.assertEqual(len(values), len(set(values.values())))
        self.assertIn('layer.11.ffn.gelu', values)
        self.assertIn('input.positions', values)
        self.assertNotIn('native.emb.token_position', values)

    def test_existing_conditioning_observer_composes_without_losing_anchors(self):
        from pcs_gate_capture import overlay as conditioning
        source = (ROOT / 'src/pcs.cpp').read_text()
        observed = overlay(conditioning(source))
        self.assertIn('pcs_gate_trace::casing', observed)
        self.assertIn('pcs_gate_trace::segmentation', observed)
        self.assertIn('steps.scalar("head.cap.output"', observed)


class ReaderContract(unittest.TestCase):
    root = Path(sys.argv[1]).resolve()
    counter = 0

    def compare(self, ref, actual, names=('stage',), budget='0.0001'):
        type(self).counter += 1
        directory = self.root / str(self.counter)
        directory.mkdir()
        write_archive(directory / 'reference.gguf', ref, {})
        write_archive(directory / 'actual.gguf', actual, {})
        (directory / 'names.txt').write_text('\n'.join(names) + '\n')
        return subprocess.run([str(self.root / 'crispembed-diff-pcs'), str(directory / 'reference.gguf'),
                               str(directory / 'actual.gguf'), str(directory / 'names.txt'), budget],
                              capture_output=True, text=True, timeout=5,
                              pass_fds=(int(os.environ['PCS_TEST_LOCK_FD']),) if 'PCS_TEST_LOCK_FD' in os.environ else ())

    def test_exact_nonzero_and_zero_rows(self):
        a = np.asarray([[0, 0], [1, 2]], dtype=np.float32)
        self.assertEqual(self.compare({'stage': a}, {'stage': a}).returncode, 0)

    def test_single_token_error_refuses(self):
        a = np.ones((4, 3), dtype=np.float32)
        b = a.copy(); b[2, 1] += 0.01
        self.assertEqual(self.compare({'stage': a}, {'stage': b}).returncode, 1)

    def test_shape_mismatch_refuses_same_element_count(self):
        a = np.ones((2, 3), dtype=np.float32)
        self.assertEqual(self.compare({'stage': a}, {'stage': a.reshape(3, 2)}).returncode, 1)

    def test_missing_stage_refuses(self):
        a = np.ones((2, 3), dtype=np.float32)
        self.assertEqual(self.compare({'stage': a}, {'other': a}).returncode, 1)

    def test_nonfinite_cannot_hide_behind_cosine(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = a.copy(); b[0, 0] = np.nan
        self.assertEqual(self.compare({'stage': a}, {'stage': b}).returncode, 1)

    def test_i32_ids_require_exact_equality(self):
        a = np.asarray([0, 3664, 2], dtype=np.int32)
        b = a.copy(); b[1] += 1
        self.assertEqual(self.compare({'input.ids': a}, {'input.ids': b}, names=('input.ids',)).returncode, 1)

    def test_invalid_budget_refuses(self):
        a = np.ones((1, 3), dtype=np.float32)
        self.assertEqual(self.compare({'stage': a}, {'stage': a}, budget='nan').returncode, 2)

    def test_pack_checks_exact_raw_byte_count(self):
        directory = self.root / 'malformed'
        directory.mkdir()
        (directory / 'manifest.tsv').write_text('stage\tf32\t3\t2\t1\t1\n')
        np.ones(5, dtype=np.float32).tofile(directory / 'stage.bin')
        with self.assertRaisesRegex(ValueError, 'bytes'):
            pack(directory, directory / 'bad.gguf')


if __name__ == '__main__':
    del sys.argv[1]
    unittest.main()
