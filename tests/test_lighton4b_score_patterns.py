import struct
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from lighton4b_score_patterns import patterns

def raw(values):return struct.pack('<%df'%len(values),*values)
class Controls(unittest.TestCase):
    def test_affine_rank_preservation(self):
        p=patterns(raw([-2,-1,0,1,2]),raw([-1,1,3,5,7]));self.assertEqual(p['affine_fit']['slope'],2);self.assertEqual(p['affine_fit']['intercept'],3);self.assertEqual(p['affine_fit']['residual_rms'],0);self.assertEqual(p['native_top'][0]['id'],4)
    def test_rank_reversal_and_invalid_values(self):
        p=patterns(raw([0,1,2]),raw([3,1,0]));self.assertEqual(p['native_top'][0]['id'],0);self.assertEqual(p['original_top'][0]['id'],2);self.assertEqual(p['original_winner_native_rank'],3);self.assertEqual(p['native_minus_original_winner_gap'],{'original':-2,'native':3})
        with self.assertRaises(ValueError):patterns(raw([float('nan')]),raw([0]))
if __name__=='__main__':unittest.main()
