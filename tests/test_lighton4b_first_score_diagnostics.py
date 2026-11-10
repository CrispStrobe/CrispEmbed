import importlib.util,math,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('d',Path(__file__).resolve().parents[1]/'tools/lighton4b_first_score_diagnostics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Diagnostics(unittest.TestCase):
 def test_shared_shift_changes_rms_but_not_distribution(self):
  x=m.compare_values([1.,2.,3.],[5.,6.,7.]);self.assertEqual(x['rms_error'],4);self.assertEqual(x['centered_rms_error'],0);self.assertAlmostEqual(x['source_to_native_next_token_kld'],0,places=14);self.assertAlmostEqual(x['probability_of_original_id'],x['original_probability'],places=14)
 def test_kld_direction_and_first_tie(self):
  x=m.compare_values([2.,2.,0.],[0.,1.,2.]);self.assertEqual(x['original_selected_id'],0);self.assertEqual(x['selected_id'],2);self.assertGreater(x['source_to_native_next_token_kld'],0);self.assertNotAlmostEqual(x['source_to_native_next_token_kld'],m.compare_values([0.,1.,2.],[2.,2.,0.])['source_to_native_next_token_kld'])
 def test_bad_geometry_or_nonfinite_rejected(self):
  for a,b in [([],[]),([1],[1,2]),([math.inf],[1])]:
   with self.assertRaises(ValueError):m.compare_values(a,b)
if __name__=='__main__':unittest.main()
