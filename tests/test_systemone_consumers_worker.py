"""Model-free controls for the SystemOne public-consumer worker. No payloads."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('systemone_consumers', ROOT/'tools/ci-heavy/systemone_consumers.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


class Controls(unittest.TestCase):
    def setUp(self): self.cfg = w.load(ROOT/w.PROTOCOL)

    def test_fixed_scope_and_root_profile(self):
        self.assertEqual(self.cfg['expected']['compared_stages'], 227)
        self.assertEqual(self.cfg['expected']['consumers'], ['c', 'python', 'cli', 'http', 'rust', 'dart'])
        self.assertIn('-DGGML_LLAMAFILE=ON', self.cfg['root_profile']['cmake'])
        self.assertIn('-DCRISPEMBED_SYSTEMONE_AUDIT=ON', self.cfg['root_profile']['cmake'])
        self.assertFalse(any(self.cfg['qualification'].values()))
        self.assertEqual(len(self.cfg['gates']), 7)

    def test_reviewed_packing_pin(self):
        pin = self.cfg['packed_producer']; self.assertEqual(w.pending(self.cfg), [])
        self.assertEqual((pin['run'], pin['model_bytes']), (40456235976, 7659180448))
        self.assertEqual(pin['source'], '9772c89660f9eb4a0220ef46f5354493c2970c08')
        unreviewed = json.loads(json.dumps(self.cfg)); unreviewed['packed_producer']['reviewed'] = False
        self.assertIn('packed producer not independently reviewed', w.pending(unreviewed))

    def test_pending_rejects_before_any_service_or_payload_access(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(w, 'policy', return_value=dict(self.cfg, packed_producer=dict(self.cfg['packed_producer'], model_sha256=None))), \
             mock.patch.object(w, 'api', side_effect=AssertionError('service access')), \
             mock.patch.object(w, 'fetch', side_effect=AssertionError('payload access')), \
             mock.patch.object(w, 'root_build', side_effect=AssertionError('build')):
            out = Path(d)/'out'
            self.assertEqual(w.run(out, Path(d)/'scratch'), 1)
            report = json.loads((out/'review/hosted.json').read_text())
            self.assertTrue(report['rejected_before_payload_access'])
            self.assertFalse(report['completed'])

    def test_complete_pin_is_not_pending(self):
        cfg = json.loads(json.dumps(self.cfg)); pin = cfg['packed_producer']
        pin.update(reviewed=True, source='x', run=1, review_artifact={'id': 1}, review_receipt_sha256='a',
                   asset_artifact={'id': 2}, model_sha256='b', model_bytes=1, original_archive={'id': 3})
        self.assertEqual(w.pending(cfg), [])

    def test_ordinary_policy_rejects_off_host(self):
        with mock.patch.dict('os.environ', {'GITHUB_ACTIONS': ''}):
            with self.assertRaises(ValueError): w.policy()


if __name__ == '__main__': unittest.main()
