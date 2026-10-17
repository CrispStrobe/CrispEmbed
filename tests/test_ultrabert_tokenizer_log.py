#!/usr/bin/env python3
"""Model-free controls for exact native tokenizer evidence."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
from ultrabert_tokenizer import parse

ROWS = [dict(case='empty', batch_row=0, ids=[0], mask=[1])]
GOOD = 'IDS\t0\t0\nMASK\t0\t1\nROW\t0\t1\nTOTAL\t1\t0'
BAD = 'IDS\t0\t0\t15\nMASK\t0\t1\t1\nROW\t0\t0\nTOTAL\t1\t1'


class TokenizerLogTest(unittest.TestCase):
    def test_complete_and_failed_ids_remain_visible(self):
        self.assertEqual(parse(GOOD, ROWS)['failures'], 0)
        failed = parse(BAD, ROWS)
        self.assertEqual(failed['failures'], 1)
        self.assertEqual(failed['rows'][0]['ids'], [0, 15])
        self.assertEqual(failed['rows'][0]['original_ids'], [0])

    def test_forged_green_or_total_rejected(self):
        for log in (BAD.replace('ROW\t0\t0', 'ROW\t0\t1'), BAD.replace('TOTAL\t1\t1', 'TOTAL\t1\t0')):
            with self.assertRaises(ValueError):
                parse(log, ROWS)

    def test_missing_duplicate_and_extra_rows_rejected(self):
        for log in (GOOD.split('\n', 1)[1], GOOD + '\nIDS\t0\t0',
                    GOOD + '\nTOTAL\t1\t0', GOOD + '\nROW\t1\t1'):
            with self.assertRaises(ValueError):
                parse(log, ROWS)

    def test_zero_id_bos_mask_is_a_failure(self):
        log = 'IDS\t0\t0\nMASK\t0\t0\nROW\t0\t0\nTOTAL\t1\t1'
        self.assertEqual(parse(log, ROWS)['failures'], 1)


if __name__ == '__main__':
    unittest.main()
