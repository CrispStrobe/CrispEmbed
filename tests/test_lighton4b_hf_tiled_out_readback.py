import importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('tiled_readback',ROOT/'tools/ci-heavy/lighton4b_hf_tiled_out_readback.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Tests(unittest.TestCase):
 def test_causal_rows_include_only_first_differing_selection(self):
  for a,b,n in [([],[],0),([],[1],0),([1,2],[1,2],2),([1,2,3],[1,2],2),([1,2],[9,2],1),([1,2,3],[1,9,3],2),([1,2],[1,9],2)]:self.assertEqual(module.causal_rows(a,b),n)
 def test_raw_unicode_edit_distance(self):
  for a,b,n in [('', 'abc',3),('ab','ba',2),('a\nb','ab',1),('أبجدية','أبجدية',0),('عربية','عربيه',1)]:self.assertEqual(module.distance(a,b),n)
  self.assertEqual(module.distance('one two'.split(),'one three'.split()),1)
if __name__=='__main__':unittest.main()
