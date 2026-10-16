"""Model-free malformed materialization report controls."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
from qwen35_materialization import parse_matrix_audit

SCHEMA = {'weight': dict(shape=[262145], dtype='float16')}
LOG = ('MATERIALIZE_MATRIX tensor=weight elements=262145 exact=1 finite=1 workers=4 '
       'max_abs=0 magnitude=2 norm=4 offset=0\nMATERIALIZE_COMPLETE tensors=1 bytes=1048580\n')

class Materialization(unittest.TestCase):
    def test_exact_complete(self):
        self.assertTrue(parse_matrix_audit(LOG, SCHEMA)['tensors'][0]['exact'])

    def reject(self, old, new):
        with self.assertRaises((ValueError, KeyError)):
            parse_matrix_audit(LOG.replace(old, new), SCHEMA)

    def test_changed_value(self): self.reject('exact=1', 'exact=0')
    def test_nonfinite(self): self.reject('finite=1', 'finite=0')
    def test_nan_magnitude(self): self.reject('magnitude=2', 'magnitude=nan')
    def test_error(self): self.reject('max_abs=0', 'max_abs=1e-7')
    def test_wrong_workers(self): self.reject('workers=4', 'workers=8')
    def test_wrong_offset(self): self.reject('offset=0', 'offset=4')
    def test_truncated(self): self.reject('bytes=1048580', 'bytes=1048576')
    def test_missing_completion(self): self.reject('MATERIALIZE_COMPLETE', 'INCOMPLETE')
    def test_duplicate(self):
        with self.assertRaises(ValueError): parse_matrix_audit(LOG + LOG, SCHEMA)

if __name__ == '__main__': unittest.main()
