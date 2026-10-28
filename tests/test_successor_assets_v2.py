"""Stdlib controls for successor-asset protocol v2; no network, archives or models."""
import copy
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import successor_assets_v2 as v2


def count_refs(path, root=ROOT):
    refs = re.findall(r'uses:\s*\./(\.github/workflows/[\w.-]+\.ya?ml)', (root / path).read_text(encoding='utf-8'))
    return sum(1 + count_refs(r, root) for r in refs)


class SuccessorAssetsV2(unittest.TestCase):
    def setUp(self):
        self.p = v2.load()

    def test_controls_pass_on_declared_final_base(self):
        result = v2.controls(copy.deepcopy(self.p))
        self.assertEqual((result['native_archives'], result['wasm_archives']), (7, 2))
        self.assertEqual(self.p['final_base'], '7bfa060df57f29692c972b84acf7b73abeaa80e7')
        self.assertEqual(self.p['ggml'], 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb')

    def test_pending_base_blocks(self):
        for key in ('final_base', 'ggml'):
            p = copy.deepcopy(self.p); p[key] = 'PENDING_FINAL'
            with self.assertRaisesRegex(v2.ProtocolError, 'pending'):
                v2.check_declared(p)

    def test_base_must_match_final_declaration(self):
        p = copy.deepcopy(self.p); p['final_base'] = '0' * 40
        with self.assertRaisesRegex(v2.ProtocolError, 'differ'):
            v2.check_declared(p)

    def test_validation_pending_and_no_claims(self):
        self.assertEqual(v2.check_declared(self.p), 'pending')
        for key in ('qualification', 'release_alias_allowed', 'app_upgrade_allowed'):
            p = copy.deepcopy(self.p); p[key] = True
            with self.assertRaises(v2.ProtocolError):
                v2.check_declared(p)
        p = copy.deepcopy(self.p); p['final_source_validation']['run'] = 1
        with self.assertRaises(v2.ProtocolError):
            v2.check_declared(p)

    def test_v1_files_unchanged(self):
        self.assertGreater(v2.check_v1_unchanged(self.p), 30)
        self.assertIn('.github/workflows/successor-assets.yml', self.p['v1_read_only_blobs'])
        self.assertNotIn('.github/workflows/successor-assets-v2.yml', self.p['v1_read_only_blobs'])

    def test_v1_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for f in self.p['v1_read_only_blobs']:
                (root / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / f, root / f)
            v2.check_v1_unchanged(self.p, root)
            victim = root / '.github/workflows/successor-assets.yml'
            victim.write_text(victim.read_text(encoding='utf-8') + '#\n', encoding='utf-8')
            with self.assertRaisesRegex(v2.ProtocolError, 'changed'):
                v2.check_v1_unchanged(self.p, root)

    def test_matrix_completeness(self):
        for mutate in (lambda p: p['producers']['native']['archives'].pop(),
                       lambda p: p['producers']['wasm']['archives'].pop(),
                       lambda p: p['consumers']['v1_jobs'].remove('mobile-link'),
                       lambda p: p['consumers']['surfaces'].remove('windows-utf8-path'),
                       lambda p: p['expectations'].pop('dart')):
            p = copy.deepcopy(self.p); mutate(p)
            with self.assertRaises(v2.ProtocolError):
                v2.check_matrix(p)

    def test_wasm_never_claimed(self):
        p = copy.deepcopy(self.p); p['expectations']['browser'].update(expected='pass', may_claim=True)
        with self.assertRaisesRegex(v2.ProtocolError, 'WASM'):
            v2.check_matrix(p)

    def test_carried_evidence_complete(self):
        self.assertEqual(v2.check_evidence(self.p), len(list((ROOT / 'tests/results/packaging').glob('*.json'))))
        p = copy.deepcopy(self.p); p['carried_evidence'].popitem()
        with self.assertRaises(v2.ProtocolError):
            v2.check_evidence(p)

    def test_producer_gate(self):
        base = self.p['final_base']
        same = lambda sha: ['tree']
        run = {'status': 'completed', 'conclusion': 'success', 'event': 'workflow_dispatch',
               'path': '.github/workflows/release.yml', 'head_sha': 'a' * 40}
        self.assertEqual(v2.check_producer_run('native', run, same), 'a' * 40)
        with self.assertRaisesRegex(v2.ProtocolError, 'closure'):
            v2.check_producer_run('native', run, lambda sha: [sha == base])
        for k, bad in (('conclusion', 'failure'), ('event', 'push'), ('path', '.github/workflows/release-wasm.yml')):
            with self.assertRaises(v2.ProtocolError):
                v2.check_producer_run('native', dict(run, **{k: bad}), same)

    def test_dispatcher_reference_budget(self):
        self.assertLessEqual(count_refs('.github/workflows/ovisocr2-metadata.yml'), 50)
        self.assertNotIn('successor-assets-v2', (ROOT / '.github/workflows/ovisocr2-metadata.yml').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
