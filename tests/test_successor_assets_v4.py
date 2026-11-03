#!/usr/bin/env python3
"""Successor assets v4: macOS-arm64 desktop tolerance, strict same-platform/other pairs, exact derivations."""
import copy, importlib.util, pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
spec = importlib.util.spec_from_file_location('v4', ROOT / 'tools/successor_assets_v4.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
P = V.load(); T = P['desktop_tolerance']
M = dict(max_abs=0.0025543533265590668, cosine=0.9998886284627958)


class Bound(unittest.TestCase):
    def test_rule(self):
        self.assertEqual((T['max_abs'], T['min_cosine']), (0.00307, 0.999866))
        self.assertEqual(V.check_tolerance(P), (0.00307, 0.999866))

    def test_macos_accepted_at_measured_and_bound_rejected_above(self):
        g = V.desktop_gate('macos-arm64', M, P)
        self.assertTrue(g['passed']); self.assertEqual(g['basis'], 'desktop-cross-platform-tolerance-v4')
        self.assertTrue(V.desktop_gate('macos-arm64', dict(max_abs=0.00307, cosine=0.999866), P)['passed'])
        self.assertFalse(V.desktop_gate('macos-arm64', dict(max_abs=0.00308, cosine=0.9999), P)['passed'])
        self.assertFalse(V.desktop_gate('macos-arm64', dict(max_abs=0.0025, cosine=0.999865), P)['passed'])

    def test_other_platforms_stay_strict(self):
        for platform in ['linux-arm64', 'windows-x86_64', 'linux-x86_64']:
            self.assertFalse(V.desktop_gate(platform, M, P)['passed'], platform)  # same drift elsewhere fails
            self.assertTrue(V.desktop_gate(platform, dict(max_abs=0.002, cosine=0.99999), P)['passed'])
            self.assertFalse(V.desktop_gate(platform, dict(max_abs=0.00201, cosine=0.99999), P)['passed'])
        self.assertEqual(V.desktop_gate('macos-arm64', dict(max_abs=1e-7, cosine=1.0), P)['basis'], 'strict')

    def test_contract_rejections(self):
        for k, v in [('platform', 'linux-arm64'), ('reference_platform', 'linux-arm64'), ('max_abs', 0.004), ('min_cosine', 0.9998),
                     ('margin', 0.3), ('attribution', 'NEON order'), ('model_sha256', '0' * 64)]:
            q = copy.deepcopy(P); q['desktop_tolerance'][k] = v
            with self.assertRaises(V.ProtocolError, msg=k): V.check_tolerance(q)
        q = copy.deepcopy(P); q['desktop_tolerance']['other_pairs_measured_strict'][0]['max_abs'] = 0.003
        with self.assertRaises(V.ProtocolError): V.check_tolerance(q)


class Frozen(unittest.TestCase):
    def test_declared_keeps_v3(self):
        V.check_declared(P)
        for k, v in [('strict_gates', dict(P['strict_gates'], max_abs=0.003)), ('wasm_tolerance', dict(P['wasm_tolerance'], max_abs=0.02)),
                     ('qualification', True)]:
            q = copy.deepcopy(P); q[k] = v
            with self.assertRaises(V.ProtocolError, msg=k): V.check_declared(q)

    def test_pins_include_v3(self):
        self.assertEqual(V.check_pins(P), 60)
        q = copy.deepcopy(P); q['read_only_blobs']['tools/successor_assets_v3.py'] = '0' * 40
        with self.assertRaises(V.ProtocolError): V.check_pins(q)

    def test_derivations_exact_and_strict_lines_kept(self):
        self.assertEqual(V.check_derived(), 4)
        r4 = (ROOT / 'tools/ci-heavy/successor_review_v4.py').read_text(); r1 = (ROOT / 'tools/ci-heavy/successor_review.py').read_text()
        for line in ["  'passed':maximum<=.002 and math.isfinite(cosine) and cosine>=.99999}",
                     "   and math.isfinite(v['cosine']) and 0<=v['max_abs']<=.002 and v['cosine']>=.99999)",
                     "   or not all(parity(v) for v in d['comparisons'].values())",
                     "  if v.get('passed') is not True or len(v.get('comparisons',[]))!=12 or not all(parity(x) for x in v['comparisons']):"]:
            self.assertIn(line, r1); self.assertIn(line, r4)  # same-platform and binding comparisons stay strict
        self.assertEqual(len(r4.splitlines()) - len(r1.splitlines()), 3)  # two imports + gate line
        with self.assertRaises(V.ProtocolError):
            V.derive('tools/ci-heavy/successor_review_v4.py', source_text=r1.replace("  if not measured['passed']:", "  if False:"))

    def test_review_module_imports(self):
        s = importlib.util.spec_from_file_location('rv4', ROOT / 'tools/ci-heavy/successor_review_v4.py')
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        ref = [0.1] * 384; vec = list(ref); vec[0] += 0.003
        self.assertFalse(m.compare_desktop_vectors(ref, vec)['passed'])

    def test_controls(self):
        out = V.controls()
        self.assertEqual(out['derived_files'], 4); self.assertEqual(out['desktop_bound'], (0.00307, 0.999866)); self.assertEqual(out['consumer_jobs'], 9)


if __name__ == '__main__':
    unittest.main()
