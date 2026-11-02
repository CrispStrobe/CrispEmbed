#!/usr/bin/env python3
"""Successor assets v3: WASM-only tolerance bounds, frozen strict gates, exact v1 derivations (stdlib only)."""
import copy, importlib.util, json, math, pathlib, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v3', ROOT / 'tools/successor_assets_v3.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
P = V.load()
T = P['wasm_tolerance']


def report_at(max_abs_target, cosine_target=None, **overrides):
    """A browser receipt whose recomputed max_abs (and optionally cosine) land on the requested values."""
    ref = [0.05 * math.sin(i + 1) for i in range(384)]
    def cos(v):
        return math.fsum(a * b for a, b in zip(v, ref)) / math.sqrt(math.fsum(a * a for a in v) * math.fsum(b * b for b in ref))
    if cosine_target is None:
        vec = list(ref); vec[0] = ref[0] + max_abs_target
    else:  # alternating +-d on every element (max_abs = d), d found by bisection on the cosine
        lo, hi = 0.0, max_abs_target
        for _ in range(200):
            mid = (lo + hi) / 2
            v = [r + (mid if i % 2 else -mid) for i, r in enumerate(ref)]
            (lo, hi) = (mid, hi) if cos(v) > cosine_target else (lo, mid)
        vec = [r + (lo if i % 2 else -lo) for i, r in enumerate(ref)]
    r = {'schema': 'crispembed-successor-wasm-consumer-v3', 'model_sha256': T['model_sha256'], 'text': T['text'],
         'page_errors': [], 'ocr': {'authored_black_blob_cleared_rgb_exact': True}, 'embed': {'vector': vec},
         'native_vector': ref, 'max_abs': max(abs(a - b) for a, b in zip(vec, ref)), 'cosine': cos(vec)}
    r.update(overrides)
    return r


class Bound(unittest.TestCase):
    def test_bound_is_rule_applied_to_measured(self):
        self.assertEqual((T['max_abs'], T['min_cosine']), (0.0101, 0.99837))
        self.assertEqual(V.check_tolerance(P), (0.0101, 0.99837))
        for m in T['measured']:
            self.assertEqual((m['max_abs'], m['cosine']), (0.00837269052863121, 0.9986498388732901))

    def test_accepted_at_measured_rejected_just_above(self):
        ok = report_at(0.00837269052863121)
        self.assertTrue(V.wasm_gate(ok, P)['passed'])
        self.assertFalse(V.wasm_gate(ok, P)['strict_native_gate_passed'])  # tolerance is what admits it
        edge = report_at(0.0101)
        self.assertTrue(V.wasm_gate(edge, P)['passed'])
        above = report_at(0.01011)
        g = V.wasm_gate(above, P)
        self.assertFalse(g['passed']); self.assertGreater(g['max_abs'], 0.0101)

    def test_cosine_side(self):
        g = V.wasm_gate(report_at(0.0099, 0.99838), P)
        self.assertTrue(g['structural']); self.assertGreaterEqual(g['cosine'], 0.99837); self.assertTrue(g['passed'])
        g = V.wasm_gate(report_at(0.0099, 0.99836), P)
        self.assertLess(g['cosine'], 0.99837); self.assertFalse(g['passed'])

    def test_structure_and_scope(self):
        base = report_at(0.001)
        self.assertTrue(V.wasm_gate(base, P)['passed'])
        for k, v in [('page_errors', ['x']), ('model_sha256', '0' * 64), ('text', 'other'), ('schema', 'crispembed-successor-wasm-consumer-v1'),
                     ('ocr', {'authored_black_blob_cleared_rgb_exact': False}), ('native_vector', [0.0] * 383)]:
            g = V.wasm_gate(dict(base, **{k: v}), P)
            self.assertFalse(g['passed'], k)
        lied = dict(base, max_abs=0.0)  # reported metric must agree with recomputation
        self.assertFalse(V.wasm_gate(lied, P)['passed'])
        bad = copy.deepcopy(base); bad['embed']['vector'][5] = float('nan')
        self.assertFalse(V.wasm_gate(bad, P)['passed'])

    def test_tolerance_contract_rejections(self):
        for path, value in [(('max_abs',), 0.02), (('min_cosine',), 0.99), (('margin',), 0.5), (('model_sha256',), '1' * 64),
                            (('applies_to', 'platform'), 'native')]:
            q = copy.deepcopy(P); t = q['wasm_tolerance']
            for k in path[:-1]: t = t[k]
            t[path[-1]] = value
            with self.assertRaises(V.ProtocolError, msg=repr(path)): V.check_tolerance(q)
        q = copy.deepcopy(P); q['wasm_tolerance']['measured'][1]['max_abs'] = 0.009
        with self.assertRaises(V.ProtocolError): V.check_tolerance(q)


class StrictUnchanged(unittest.TestCase):
    def test_declared_strict_gates_and_expectations(self):
        V.check_declared(P)
        for k, v in [('strict_gates', dict(P['strict_gates'], max_abs=0.003)), ('qualification', True)]:
            q = copy.deepcopy(P); q[k] = v
            with self.assertRaises(V.ProtocolError): V.check_declared(q)
        q = copy.deepcopy(P); q['expectations']['c']['expected'] = 'unproven'
        with self.assertRaises(V.ProtocolError): V.check_declared(q)

    def test_derived_files_exact_and_native_gates_byte_identical(self):
        self.assertEqual(V.check_derived(), 3)
        review = (ROOT / 'tools/ci-heavy/successor_review_v3.py').read_text()
        v1 = (ROOT / 'tools/ci-heavy/successor_review.py').read_text()
        for line in ["  'passed':maximum<=.002 and math.isfinite(cosine) and cosine>=.99999}",
                     "   and math.isfinite(v['cosine']) and 0<=v['max_abs']<=.002 and v['cosine']>=.99999)",
                     "   raise ValueError('Cross-platform desktop vector parity failed: '+platform+' '+repr(measured))"]:
            self.assertIn(line, v1); self.assertIn(line, review)
        self.assertEqual(len(review.splitlines()) - len(v1.splitlines()), 1)  # only the added import line
        wf = (ROOT / '.github/workflows/successor-assets-consumers-v3.yml').read_text()
        v1wf = (ROOT / '.github/workflows/successor-assets.yml').read_text()
        self.assertNotIn('test_successor_wasm.cjs', wf); self.assertIn('successor_assets_v3.py browser-gate', wf)
        self.assertEqual(len(wf.splitlines()), len(v1wf.splitlines()))
        self.assertEqual(sum(a != b for a, b in zip(wf.splitlines(), v1wf.splitlines())), 3)  # name, browser step, review step

    def test_derivation_rejects_moved_anchor(self):
        src = (ROOT / 'tests/test_successor_wasm.cjs').read_text()
        with self.assertRaises(V.ProtocolError):
            V.derive('tests/test_successor_wasm_v3.cjs', source_text=src.replace("maxAbs<=.002", "maxAbs<=.003"))
        with self.assertRaises(V.ProtocolError):
            V.derive('tests/test_successor_wasm_v3.cjs', source_text=src + src)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for target, (source, _) in V.DERIVATIONS.items():
                for p in (source, target):
                    (root / p).parent.mkdir(parents=True, exist_ok=True); (root / p).write_bytes((ROOT / p).read_bytes())
            (root / 'tools/ci-heavy/successor_review_v3.py').write_text((root / 'tools/ci-heavy/successor_review_v3.py').read_text().replace('<=.002', '<=.02'))
            with self.assertRaises(V.ProtocolError): V.check_derived(root)

    def test_pins_unchanged(self):
        self.assertGreaterEqual(V.check_pins(P), 50)
        q = copy.deepcopy(P); q['read_only_blobs']['tools/successor_assets_v2.py'] = '0' * 40
        with self.assertRaises(V.ProtocolError): V.check_pins(q)
        q = copy.deepcopy(P); q['read_only_blobs']['tests/results/packaging/does-not-exist.json'] = '0' * 40
        with self.assertRaises(V.ProtocolError): V.check_pins(q)

    def test_review_uses_gate_only_for_browser(self):
        sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
        spec = importlib.util.spec_from_file_location('rv3', ROOT / 'tools/ci-heavy/successor_review_v3.py')
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        ref = [0.1] * 384; vec = list(ref); vec[0] += 0.003  # 0.003: fails strict desktop parity
        self.assertFalse(m.compare_desktop_vectors(ref, vec)['passed'])

    def test_producer_check(self):
        run = dict(status='completed', conclusion='success', event='workflow_dispatch', path='.github/workflows/release-wasm.yml', head_sha='a' * 40)
        self.assertEqual(V.check_producer_run('wasm', run, lambda s: ['same'], P), 'a' * 40)
        with self.assertRaises(V.ProtocolError): V.check_producer_run('wasm', dict(run, conclusion='failure'), lambda s: ['same'], P)
        with self.assertRaises(V.ProtocolError): V.check_producer_run('wasm', run, lambda s: [s], P)

    def test_controls(self):
        out = V.controls()
        self.assertEqual(out['derived_files'], 3); self.assertEqual(out['wasm_bound'], (0.0101, 0.99837))


if __name__ == '__main__':
    unittest.main()
