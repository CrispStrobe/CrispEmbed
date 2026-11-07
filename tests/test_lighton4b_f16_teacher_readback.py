import importlib.util,unittest,os,tempfile,subprocess
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'tools/ci-heavy/lighton4b_f16_teacher_readback.py';s=importlib.util.spec_from_file_location('reader',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_equal_tie_lower_id(self):
  a=np.array([1.,1.,0.],dtype='<f4');r=m.metrics(a,a,0);self.assertEqual(r['error_rms'],0);self.assertEqual(r['kld_reference_to_candidate'],0);self.assertEqual(r['candidate_teacher_rank_1based'],1);self.assertEqual(r['candidate_argmax'],0)
 def test_direction_and_rank(self):
  a=np.array([2.,0.,-1.],dtype='<f4');b=np.array([0.,2.,-1.],dtype='<f4');r=m.metrics(a,b,0);self.assertFalse(r['argmax_equal']);self.assertEqual(r['candidate_teacher_rank_1based'],2);self.assertGreater(r['kld_reference_to_candidate'],0);self.assertLess(r['candidate_probability_teacher'],r['reference_probability_teacher'])
 def test_source_revision_from_external_directory(self):
  expected=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.parents[2],text=True).strip();before=os.getcwd()
  with tempfile.TemporaryDirectory()as folder:
   try:os.chdir(folder);self.assertEqual(m.source_revision(),expected)
   finally:os.chdir(before)
 def test_fail_closed(self):
  with self.assertRaises(ValueError):m.metrics(np.array([1.],dtype='<f4'),np.array([np.nan],dtype='<f4'),0)
if __name__=='__main__':unittest.main()
