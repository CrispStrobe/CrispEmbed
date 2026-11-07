#!/usr/bin/env python3
"""Offline controls for tools/ci-heavy/josev_quant_publish.py (no network, no weights)."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
import josev_quant_publish as jq


class Controls(unittest.TestCase):
    def tree(self, edit_proto=None, edit_receipt=None):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d)
        cfg = json.loads((ROOT / jq.PROTOCOL).read_text())
        rec = json.loads((ROOT / cfg['recipes']['Q4_K_MIX']['precision_receipt']).read_text())
        if edit_proto: edit_proto(cfg)
        if edit_receipt: edit_receipt(rec)
        for rel, obj in ((jq.PROTOCOL, cfg), (jq.F32_PROTOCOL, json.loads((ROOT / jq.F32_PROTOCOL).read_text())),
                         (cfg['recipes']['Q4_K_MIX']['precision_receipt'], rec)):
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(json.dumps(obj))
        return d

    def test_committed_protocol_passes_both_gates(self):
        cfg = jq.load()
        q4, q8 = jq.gate(cfg, 'Q4_K_MIX'), jq.gate(cfg, 'Q8_0')
        self.assertGreater(q4['head_logit_cos_min'], 0.94)
        self.assertGreater(q8['head_logit_cos_min'], 0.97)
        self.assertEqual(q4['typed_json_exact'], 0)
        self.assertEqual(q4['expected_sha256'][:8], 'adc7fa47')

    def test_unapproved_rejected(self):
        d = self.tree(lambda c: c.update(approved_by_owner=False))
        with self.assertRaisesRegex(ValueError, 'not approved'):
            jq.load(d)

    def test_below_bar_rejected(self):
        def low(r):
            r['recipes']['all-q4-down-q8']['cases'][0]['logits_cos'] = 0.9
        d = self.tree(edit_receipt=low)
        with self.assertRaisesRegex(ValueError, 'not above'):
            jq.gate(jq.load(d), 'Q4_K_MIX', d)

    def test_other_source_rejected(self):
        d = self.tree(edit_receipt=lambda r: r['model'].update(sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'different F32'):
            jq.gate(jq.load(d), 'Q4_K_MIX', d)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            jq.parse_variants('Q2_K', jq.load())

    def test_readme_keeps_f32_and_replaces_section(self):
        old = '# F32 card\n\nbody\n'
        once = jq.readme(old, jq.SECTION + '\nA\n')
        twice = jq.readme(once, jq.SECTION + '\nB\n')
        self.assertTrue(twice.startswith(old.rstrip()))
        self.assertIn('B', twice); self.assertNotIn('\nA\n', twice)


if __name__ == '__main__':
    unittest.main()
