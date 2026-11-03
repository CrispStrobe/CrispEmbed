#!/usr/bin/env python3
"""Stdlib controls for the default-off MoEdern mixed padded-Q4_K/Q8_0/F32 profile."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/moedern-colbert'))
import quantize_q4k_mixed as m  # noqa: E402


def names(layers, conv=()):
    out = ['token_embd.hash.weight', 'token_embd.proj.weight', 'output_norm.weight', 'dense_head.weight']
    for i in range(layers):
        b = f'blk.{i}.'
        if i in conv:
            out += [b + 'conv_in.weight', b + 'conv_out.weight', b + 'conv_dw.weight', b + 'conv_norm.weight']
            continue
        out += [b + s for s in ('attn_norm.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight',
                                'attn_output.weight', 'ffn_norm.weight', 'ffn_gate_inp.weight',
                                'ffn_up_exps.weight', 'ffn_gate_exps.weight', 'ffn_down_exps.weight',
                                'ffn_up_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_down_shexp.weight')]
    return out


class Controls(unittest.TestCase):
    def test_base_and_groups(self):
        n = names(2)
        t = m.assign('q8+attn_v', n, '27M')
        self.assertEqual({k for k, v in t.items() if v == 'Q4_K'}, {'blk.0.attn_v.weight', 'blk.1.attn_v.weight'})
        self.assertTrue(all(v == 'Q8_0' for k, v in t.items() if 'attn_v' not in k))
        # Router, norms, embeddings, conv kernel and dense head are never assigned.
        for critical in ('ffn_gate_inp', 'norm', 'token_embd', 'dense_head', 'conv_dw'):
            self.assertFalse(any(critical in k for k in t), critical)
        self.assertEqual(set(m.assign('q4', n, '27M').values()), {'Q4_K'})
        t = m.assign('q4-L1', n, '27M')
        self.assertTrue(all(v == ('F32' if k.startswith('blk.1.') else 'Q4_K') for k, v in t.items()))
        t = m.assign('q8+L0-1-attn_q', n, '27M')
        self.assertEqual(t['blk.0.attn_q.weight'], 'F32')
        self.assertEqual(t['blk.1.ffn_up_exps.weight'], 'Q4_K')
        t = m.assign('q4~attn_o~shexp', n, '27M')
        self.assertEqual(t['blk.0.attn_output.weight'], 'Q8_0')
        self.assertEqual(t['blk.1.ffn_down_shexp.weight'], 'Q8_0')
        self.assertEqual(t['blk.1.attn_q.weight'], 'Q4_K')

    def test_rejections(self):
        n = names(2)
        for bad in ('q8', 'q8+nothing', 'q6+attn', 'q8+L7', 'q8;rm'):
            with self.assertRaises(ValueError):
                m.assign(bad, n, '27M')

    def test_named_policies_are_valid_and_variant_bound(self):
        layout = {'1.5M': names(2), '27M': names(2), '115M': names(10, conv=(0, 1, 3, 4, 6, 8))}
        for name, policy in m.POLICIES.items():
            self.assertRegex(name, r'^moedern[0-9p]+m_q4k_[a-z0-9_]+$')
            self.assertIn(policy['variant'], layout)
            types = m.assign(name, layout[policy['variant']], policy['variant'])
            self.assertIn('Q4_K', types.values())
            other = next(v for v in layout if v != policy['variant'])
            with self.assertRaises(ValueError):
                m.assign(name, layout[other], other)

    def test_runtime_gate_and_recipe_string(self):
        header = (ROOT / 'src/moedern_q4k_mixed.h').read_text()
        self.assertIn('constexpr const char * RECIPE = "%s";' % m.RECIPE, header)
        self.assertIn('moedern.q4k.policy', header)
        self.assertIn('GGML_TYPE_Q8_0', header)
        for path in ('src/moedern_family_engine.cpp', 'src/moedern_text_frontend.cpp'):
            text = (ROOT / path).read_text()
            self.assertIn('core_env::on("CRISPEMBED_MOEDERN_Q4_K_MIXED")', text)
            self.assertNotIn('getenv("CRISPEMBED_MOEDERN_Q4_K_MIXED', text)
        engine = (ROOT / 'src/moedern_family_engine.cpp').read_text()
        # The plain padded and named Q8_0 entry points are unchanged.
        self.assertIn('return init_impl(path, threads, failure, true)', engine)
        self.assertIn('return init_impl(path, threads, failure, false, true)', engine)
        self.assertRegex(engine, r'const bool mixed = padded_q4k && core_env::on\("CRISPEMBED_MOEDERN_Q4_K_MIXED"\)')
        self.assertIn('Mixed: critical tensor must stay F32', engine)

    def test_publication_recipes_clear_the_cosine_bar_in_every_run(self):
        import json
        cfg = json.loads((ROOT / 'docs/moedern-q4k-publication-v1.json').read_text())
        self.assertEqual(cfg['status'], 'prepared, not published')
        for variant, recipes in cfg['recipes'].items():
            for name, rec in recipes.items():
                self.assertEqual((rec['precision_gate'], rec['precision_tier']), ('cosine', 'q4_k'))
                self.assertEqual(m.POLICIES[rec['policy']]['variant'], variant)
                receipt = json.loads((ROOT / rec['precision_receipt']).read_text())
                self.assertEqual(receipt['variant'], variant)
                arms = [r['arms'][rec['precision_dtype']] for r in receipt['runs'] if rec['precision_dtype'] in r['arms']]
                self.assertTrue(arms, name)
                for arm in arms:
                    self.assertTrue(arm['ids_exact'])
                    self.assertEqual(len(arm['cases']), 10)
                    for case in arm['cases'].values():
                        for kind in ('all', 'selected'):
                            self.assertGreater(case[kind]['min_row_cosine'], 0.94)
                runs = [r for r in receipt['runs'] if rec['policy'] in r['arms'] or
                        m.POLICIES[rec['policy']]['recipe'] in r['arms']]
                self.assertTrue({'portable' in r['build']['mode'] for r in runs} == {True})
                self.assertGreaterEqual(len({r['build']['cpu'] for r in runs}), 2, 'two CPU ISAs required')

    def test_driver_is_hosted_only_and_records_build_mode(self):
        text = (ROOT / 'tools/ci-heavy/moedern_q4k_lanes.py').read_text()
        self.assertIn("os.environ.get('GITHUB_ACTIONS') == 'true'", text)
        self.assertIn('-DGGML_NATIVE=OFF', text)
        self.assertIn("'GGML_NATIVE:BOOL=OFF' in cache", text)
        self.assertTrue(re.search(r'FIXTURES = \(.*\n\s+.3528982465132c835ea0fc7c0b8b4b90ea99ecae', text))


if __name__ == '__main__':
    unittest.main()
