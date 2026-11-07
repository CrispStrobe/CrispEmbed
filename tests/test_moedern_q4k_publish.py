"""Offline controls for tools/ci-heavy/moedern_q4k_publish.py (no network, no weights)."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pub', ROOT / 'tools/ci-heavy/moedern_q4k_publish.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class MoedernQ4kPublish(unittest.TestCase):
    def tree(self, mutate):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        for rel in (pub.PROTOCOL, 'tests/results/moedern-colbert/publication-27m.json',
                    'tests/results/moedern-colbert/publication-115m.json',
                    'tests/results/moedern-colbert/q4k-lane-27m.json', 'tests/results/moedern-colbert/q4k-lane-115m.json'):
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, d / rel)
        cfg = json.loads((d / pub.PROTOCOL).read_text())
        mutate(cfg)
        (d / pub.PROTOCOL).write_text(json.dumps(cfg))
        return d

    def test_both_variants_pass_and_render(self):
        for v in ('27M', '115M'):
            _, rec = pub.load(v)
            ev = pub.evidence(v, rec)
            text = pub.section(v, rec, ev)
            self.assertIn('**experimental**', text)
            self.assertIn('not measured', text)
            self.assertIn(rec['expected']['sha256'], text)
            self.assertIn('CRISPEMBED_MOEDERN_Q4_K_MIXED=1', text)
            merged = pub.card(v, rec, ev, '# F32 card\n\nbody\n' + text, 'a' * 64, 5)
            self.assertEqual(merged.count(pub.MARK), 1)
            self.assertTrue(merged.startswith('# F32 card'))

    def test_refusals(self):
        def unapproved(c): c['approved_by_owner'] = False
        def destination(c): c['recipes']['27M']['Q4_K_MIX']['repo'] = 'cstr/other'
        def sha(c): c['recipes']['115M']['Q4_K_MIX']['expected']['sha256'] = 'b' * 64
        for mutate, variant in ((unapproved, '27M'), (destination, '27M'), (sha, '115M')):
            d = self.tree(mutate)
            with self.assertRaises(ValueError):
                cfg, rec = pub.load(variant, d)
                pub.evidence(variant, rec, d)
        with self.assertRaises(ValueError):
            pub.load('1.5M')

    def test_publish_requires_hosted(self):
        from unittest import mock
        with mock.patch.dict('os.environ', {'GITHUB_ACTIONS': 'false'}), self.assertRaises(ValueError):
            pub.publish('27M', Path(tempfile.mkdtemp()))


if __name__ == '__main__':
    unittest.main()
