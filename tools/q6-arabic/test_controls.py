import copy
import ast,importlib.util,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('arabic',Path(__file__).with_name('worker.py'));w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class Controls(unittest.TestCase):
 def test_protocol_and_syntax(self):
  ast.parse(Path(w.__file__).read_text());w.admit(w.ROOT/'docs/lighton4b-q6-arabic-v1.json')
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'altered';p.write_bytes((w.ROOT/'docs/lighton4b-q6-arabic-v1.json').read_bytes()+b' ')
   with self.assertRaises(RuntimeError):w.admit(p)
 def test_only_declared_three_mechanisms(self):
  e={'CRISPEMBED_LIGHTON4B_HEAD_BF16':'0','other':'keep'};off,on=w.gates(e)
  self.assertEqual({k for k in on if on[k]!=off[k]},w.CHANGED);self.assertEqual(off['other'],on['other']);self.assertNotIn(next(iter(w.CHANGED)),e)
 def test_owned_executable(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'binary';p.write_bytes(b'actual fixture');p.chmod(0o444);w.stage(p);self.assertTrue(p.stat().st_mode&0o100)
 def test_retained_source_complete(self):
  r=dict(tokens=[1,2,3,4,5,248046],stop=True,stop_type='eos',truncated=False,tokens_predicted=6,completion_probabilities=[{'id':v}for v in [1,2,3,4,5,248046]]);w.upstream_complete(r)
  for key,value in [('stop',False),('stop_type','limit'),('truncated',True),('tokens_predicted',5),('completion_probabilities',[])]:
   bad=copy.deepcopy(r);bad[key]=value
   with self.assertRaises(RuntimeError):w.upstream_complete(bad)
 def test_profile_closed(self):
  self.assertEqual(w.PROFILE['GGML_OPENMP'],'OFF');self.assertEqual(w.PROFILE['GGML_LLAMAFILE'],'ON');self.assertEqual(len(w.PROFILE),13)
if __name__=='__main__':unittest.main()
