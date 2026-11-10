import ast,importlib.util,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('observed_ab',Path(__file__).with_name('worker.py'));w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class Controls(unittest.TestCase):
 def test_protocol_syntax_and_actual_bound_proof(self):
  ast.parse(Path(w.__file__).read_text());c=w.admit(w.ROOT/'docs/lighton4b-source-prefill-observed-ab-v1.json');w.bound_proof(c)
  for value in [None,'pending','X'*64,'0'*63]:
   with self.assertRaises(RuntimeError):w.bound_proof(dict(proof_receipt_sha256=value))
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'changed';p.write_bytes((w.ROOT/'docs/lighton4b-source-prefill-observed-ab-v1.json').read_bytes()+b' ')
   with self.assertRaises(RuntimeError):w.admit(p)
 def test_exact_single_flag(self):
  source={'CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED':'1','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK':'1','CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS':'1','other':'keep'};off,on=w.gates(source)
  self.assertEqual({k for k in off if off[k]!=on[k]},{w.GATE});self.assertNotIn(w.GATE,source)
  for key in source:self.assertEqual(off[key],on[key])
 def test_owned_binary_mode(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'binary';p.write_bytes(b'fixture');p.chmod(0o444);w.stage(p);self.assertTrue(p.stat().st_mode&0o100)
 def test_dependency_and_artifact_scope(self):
  text=Path(w.__file__).read_text();self.assertIn("cfg['cases']==['simple_table','simple_form','arabic_printed_line']",text);self.assertIn("not cfg['heldout_allowed']",text);self.assertNotIn('heldout-result.json',text);self.assertIn('shared.progress_count(dest)',text)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'outer-active.json').write_text('active');(p/'done.log').write_text('done');(p/'case').mkdir();(p/'case/generated.ids').write_text('1')
   self.assertEqual(set(w.shared.artifacts(p,{'done':{}})),{'done.log','case/generated.ids'})
if __name__=='__main__':unittest.main()
