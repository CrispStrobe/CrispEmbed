"""Causal-position admission and immutable retention protocol, without framework."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/ci-heavy'))
import lighton4b_plain_q4_compare as m
class Controls(unittest.TestCase):
 def test_difference_row_valid_next_row_invalid(self):self.assertEqual(m.shared_positions([1,2,3,4],[1,2,9,4]),[0,1,2])
 def test_equal_and_length_boundary(self):self.assertEqual(m.shared_positions([1,2],[1,2]),[0,1]);self.assertEqual(m.shared_positions([1],[1,2]),[0])
 def test_unknown_empty_or_bad_ids(self):
  for a,b in [([],[]),([1],[]),([True],[1]),([-1],[1]),([248320],[1])]:
   with self.assertRaises(ValueError):m.shared_positions(a,b)
 def test_closed_artifact_and_source(self):
  self.assertEqual(m.NATIVE_RUN,42183620520);self.assertEqual(m.ARTIFACT,13488399938);self.assertEqual(len(m.SOURCE_SHA),64);self.assertEqual(m.SOURCE,'bd87593040c9a31cf5ca6ec2fd2369de6f34b995')
if __name__=='__main__':unittest.main()
