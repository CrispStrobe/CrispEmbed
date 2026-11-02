"""Reject tampered evidence without downloading checkpoints or running inference."""
import copy
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_qwen35_quality_quant import audit, verify_frozen, verify_quantizer
import qwen35_source_pins as pins


class QualityAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'tests/results/qwen35-ocr/quality-quant-q8-inventory-table.json'
        cls.original = json.loads(cls.path.read_text())

    def rejected(self, mutate):
        report = copy.deepcopy(self.original)
        mutate(report)
        with self.assertRaises(ValueError):
            verify_frozen(report, ROOT, self.original['source_commit'])

    def test_existing_preselected_frozen_controls(self):
        for profile in ('q8', 'q8-q6', 'q4'):
            path = ROOT / f'tests/results/qwen35-ocr/quality-quant-{profile}-inventory-table.json'
            report = json.loads(path.read_text())
            old, before = verify_frozen(report, ROOT, report['source_commit'])
            self.assertEqual(old['id'], 'inventory-table')
            self.assertEqual(len(before), 473)

    def test_wrong_source_incomplete_or_out_of_scope(self):
        for field, value in [('source_commit', '0' * 40), ('result', 'not completed'),
                ('production_changes', 'changed'), ('publication', 'published')]:
            self.rejected(lambda r: r.update({field: value}))

    def test_report_recipe_and_source_provenance(self):
        for mutate in (lambda r: r.update(prior_report_sha256='wrong'),
                lambda r: r.update(recipe_report_sha256='wrong'),
                lambda r: r.update(recipe_source='PLAN.md'),
                lambda r: r['runtime_sources'].clear(),
                lambda r: r['diagnostic_sources'].clear(),
                lambda r: r.update(revision='wrong'),
                lambda r: r['native_environment'].update(extra='1')):
            self.rejected(mutate)

    def test_original_and_f16_replay_records(self):
        for mutate in (lambda r: r['reference'].update(generated_ids=[]),
                lambda r: r['reference'].update(selected_token_confidences=[]),
                lambda r: r['arms'][0]['decode']['metrics'][0].update(max_abs=10),
                lambda r: r['arms'][1]['decode']['metrics'][0].update(max_abs=10),
                lambda r: r['arms'][0]['decode'].update(native_confidences=[]),
                lambda r: r.update(frozen_f16_exact=False)):
            self.rejected(mutate)

    def test_precision_claims_and_protected_bytes(self):
        def corrupt_payload(report):
            tensors = report['arms'][1]['tensor_audit']
            name = next(n for n in tensors if n.startswith('v.'))
            tensors[name]['sha256'] = 'wrong'
        for mutate in (lambda r: r['arms'][1].update(gguf_sha256='wrong'),
                lambda r: r['arms'][1].update(kind='q4_k'),
                lambda r: r['arms'][1].update(overrides=[]), corrupt_payload):
            self.rejected(mutate)

    def test_false_summary_and_changed_gold(self):
        self.rejected(lambda r: r['summary'][1].update(score_failures=0))
        self.rejected(lambda r: r['case']['gold'].clear())
        self.rejected(lambda r: r['reference_quality'].update(parse_ok=False))

    def quantizer_fixture(self):
        before = {'v.matrix': dict(type='F32', shape=[2, 2], bytes=16, sha256='vision'),
                  'l.proj': dict(type='F16', shape=[256, 2], bytes=1024, sha256='original')}
        after = copy.deepcopy(before)
        after['l.proj'].update(type='Q8_0', bytes=544, sha256='quantized')
        log = ('[ 1/2] v.matrix - f32, copy 0.0 MB\n'
               '[ 2/2] l.proj - f16, quantizing to q8_0... 0.0 MB -> 0.0 MB\n'
               '1 quantized, 1 kept\n')
        return before, after, log

    def test_quantizer_types_and_copied_payload(self):
        before, after, log = self.quantizer_fixture()
        self.assertEqual(verify_quantizer(log, before, after), dict(quantized=1, kept=1))
        for wrong in (log.replace('l.proj - f16', 'l.proj - f32'),
                      log.replace('to q8_0', 'to q4_k')):
            with self.assertRaises(ValueError): verify_quantizer(wrong, before, after)
        after['v.matrix']['sha256'] = 'changed'
        with self.assertRaises(ValueError): verify_quantizer(log, before, after)

    def test_quantizer_missing_duplicate_and_reordered_rows(self):
        before, after, log = self.quantizer_fixture()
        lines = log.splitlines(True)
        for wrong in (''.join(lines[1:]), ''.join([lines[1], lines[0], lines[2]]),
                log + lines[0], log + lines[-1], log.replace('/2]', '/3]')):
            with self.assertRaises(ValueError): verify_quantizer(wrong, before, after)

    def test_missing_native_logs_cannot_verify_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                audit(self.path, Path(directory), self.original['source_commit'])


class SourcePins(unittest.TestCase):
    """Frozen receipt pins change only through a recorded, reviewed re-pin chain."""

    def test_every_pin_is_current_or_recorded(self):
        self.assertEqual(pins.drift(), [])

    def test_unrecorded_or_stale_repins_are_rejected(self):
        path, frozen = 'src/qwen35_math.h', 'ae542fa73d48dc622fe0e5813ca49fc9616a343fc9d3ba2a927e2a82d7908357'
        recorded = pins.load()
        self.assertTrue(pins.source_matches(ROOT, path, frozen, recorded))
        self.assertFalse(pins.source_matches(ROOT, path, frozen, {}))
        self.assertFalse(pins.source_matches(ROOT, path, '0' * 64, recorded))
        stale = copy.deepcopy(recorded)
        stale[(path, frozen)]['repins'][-1]['to_sha256'] = '1' * 64
        stale[(path, frozen)]['repins'].append(dict(from_sha256='1' * 64, to_sha256='2' * 64))
        self.assertFalse(pins.source_matches(ROOT, path, frozen, stale))
        broken = copy.deepcopy(recorded)
        broken[(path, frozen)]['repins'][0]['from_sha256'] = '3' * 64
        with self.assertRaises(ValueError):
            pins.source_matches(ROOT, path, frozen, broken)

    def test_frozen_digests_are_the_receipt_commit_blobs(self):
        commits = {c for *_, c in pins.pins()}
        present = all(subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', c + '^{commit}'],
                                     capture_output=True).returncode == 0 for c in commits)
        if not present:
            if os.environ.get('OCR_PIN_REQUIRE_GIT') == '1':
                self.fail('Receipt commits are not fetched: ' + ', '.join(sorted(commits)))
            self.skipTest('receipt commits not available in this checkout')
        self.assertEqual(pins.verify_git(), 15)

    def test_workflow_runs_when_a_pinned_source_changes(self):
        workflow = (ROOT / '.github/workflows/ocr-quality.yml').read_text()
        for path in sorted({p for _, _, p, _, _ in pins.pins()} | {pins.AMENDMENTS, 'tools/qwen35_source_pins.py'}):
            self.assertEqual(workflow.count("- '%s'" % path), 2, path)


if __name__ == '__main__':
    unittest.main()
