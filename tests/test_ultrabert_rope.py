"""Small fail-closed controls; actual operators, tensors and compilation stay hosted."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
import ultrabert_rope as audit


class Contract(unittest.TestCase):
    def log(self, failed=False):
        rows = []
        for i, key in enumerate(sorted(audit.expected_metrics())):
            mode, stage = key.split('/')
            fail = failed and i == 0
            rows.append('\t'.join(['ROPE', mode, stage, '1', '0.003' if fail else '0',
                                    '0', '0', '1', '1', '0' if fail else '1', '0' if fail else '1']))
        for i in range(24):
            for kind in ('observer', 'inputs'):
                rows.append(f'ROPEEXACT\t{kind}.layer_{i}\t1')
        rows.append('ROPEEND\t192\t48\t' + str(int(failed)))
        return '\n'.join(rows)

    def test_complete_metrics_and_bits(self):
        parsed = audit.parse_log(self.log())
        self.assertEqual(len(parsed['metrics']), 192)
        self.assertEqual(len(parsed['controls']), 48)
        self.assertEqual(parsed['failures'], 0)

    def test_real_numerical_failures_remain_failures(self):
        result = audit.parse_log(self.log(True))
        self.assertEqual(result['failures'], 1)
        self.assertFalse(next(iter(result['metrics'].values()))['passed'])

    def test_missing_duplicate_nonfinite_and_false_verdict_reject(self):
        good = self.log()
        for bad in [good.replace('ROPEEND\t192', 'ROPEEND\t191'),
                    '\n'.join(good.splitlines()[1:]), good + '\n' + good.splitlines()[0],
                    good.replace('\t1\t0\t0\t0\t1\t1\t1\t1', '\tnan\t0\t0\t0\t1\t1\t1\t1', 1),
                    good.replace('\t1\t0\t0\t0\t1\t1\t1\t1', '\t1\t0.1\t0\t0\t1\t1\t1\t0', 1)]:
            with self.subTest(tail=bad[-40:]), self.assertRaises(ValueError):
                audit.parse_log(bad)

    def test_observer_failure_cannot_be_waived(self):
        text = self.log().replace('ROPEEXACT\tobserver.layer_0\t1', 'ROPEEXACT\tobserver.layer_0\t0')
        with self.assertRaises(ValueError):
            audit.parse_log(text)  # Totals have not admitted the failure.
        result = audit.parse_log(text.replace('ROPEEND\t192\t48\t0', 'ROPEEND\t192\t48\t1'))
        self.assertEqual(result['failures'], 1)

    def test_archive_exact_members_size_and_paths(self):
        for names, inventory in [(['x'], {'x': {'bytes': 3}}),
                                  (['x'], {'x': {'bytes': 4}}),
                                  (['../x'], {'../x': {'bytes': 3}}),
                                  (['/x'], {'/x': {'bytes': 3}}),
                                  (['x', 'x'], {'x': {'bytes': 3}})]:
            data = io.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(data, 'w') as archive:
                    for name in names:
                        archive.writestr(name, b'123')
            data.seek(0)
            with zipfile.ZipFile(data) as archive:
                if names == ['x'] and inventory['x']['bytes'] == 3:
                    self.assertEqual(audit.checked_archive(archive, inventory), ['x'])
                else:
                    with self.assertRaises(ValueError):
                        audit.checked_archive(archive, inventory)

    def git(self, *args):
        if args == ('rev-parse', 'HEAD:ggml') or args[-2:] == ('rev-parse', 'HEAD') and args[0] == '-C':
            return audit.GGML
        if args == ('rev-parse', 'HEAD'):
            return '0' * 40
        return ''

    def test_scoped_source_pins_and_ggml_guard(self):
        with patch.object(audit, 'git', side_effect=self.git):
            result = audit.preflight()
        self.assertEqual(result['native_pins'], audit.PINS)
        with patch.object(audit, 'git', return_value='dirty'):
            with self.assertRaises(ValueError):
                audit.preflight()
        actual = audit.sha256
        with patch.object(audit, 'sha256', side_effect=lambda p: '0' * 64
                          if str(p).endswith('ultrabert_embed.cpp') else actual(p)):
            with self.assertRaisesRegex(ValueError, 'scoped native'):
                audit.preflight()


if __name__ == '__main__':
    unittest.main()
