import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('execution',Path(__file__).with_name('execution.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Contracts(unittest.TestCase):
 def config(self):return json.loads((m.ROOT/'docs/lighton4b-heldout-executor-v1.json').read_text())
 def test_frozen_and_mutations(self):
  cfg=self.config();m.admit(cfg)
  for key,value in [('cap',512),('context',4096),('threads',2),('case_worker_seconds',0),('arm_seconds',0),('qualified',True),('cases',list(reversed(m.prepare.CASE_IDS))),('observed_form_receipt_sha256','0'*64),('candidate_sha256','0'*64),('expected_media_prompt','other')]:
   bad=copy.deepcopy(cfg);bad[key]=value
   with self.assertRaises(ValueError):m.admit(bad)
 def test_frozen_reference_and_input_seal(self):
  cfg=self.config();case={'id':'synthetic-input-control','prepared':{'size':[32,32]}};ids=[1,2,3];mods=[0,1,0];positions=b'\0'*36
  with tempfile.TemporaryDirectory()as d:
   target=Path(d)/'seal';m.seal_inputs(target,case,ids,mods,[1,2,2],positions,ids,mods,positions,cfg['expected_media_prompt'],cfg)
   with self.assertRaises(FileExistsError):m.seal_inputs(target,case,ids,mods,[1,2,2],positions,ids,mods,positions,cfg['expected_media_prompt'],cfg)
   for changes in [('positions',positions[:-1]),('ids',[1,2]),('grid',[1,2,4])]:
    with self.assertRaises(ValueError):m.seal_inputs(Path(d)/changes[0],case,ids,mods,changes[1]if changes[0]=='grid'else[1,2,2],positions,changes[1]if changes[0]=='ids'else ids,mods,changes[1]if changes[0]=='positions'else positions,cfg['expected_media_prompt'],cfg)
 def test_source_complete_and_neutral(self):
  r=dict(tokens=[248046],tokens_evaluated=3,truncated=False,stop_type='eos',completion_probabilities=[{}],content='');self.assertTrue(m.source_response(r,3));m.neutral(r,copy.deepcopy(r))
  for key,value in [('truncated',True),('tokens_evaluated',4),('stop_type','limit'),('tokens',[]),('completion_probabilities',[])]:
   bad=copy.deepcopy(r);bad[key]=value
   with self.assertRaises(ValueError):m.source_response(bad,3)
  bad=copy.deepcopy(r);bad['content']='changed'
  with self.assertRaises(ValueError):m.neutral(r,bad)
 def test_actual_worker_syntax(self):
  import ast
  ast.parse((m.ROOT/'tools/heldout/worker.py').read_text())
 def test_observed_full_text_anchor_scope(self):
  self.assertEqual(len(m.OBSERVED_TEXT_ANCHORS),38);self.assertEqual(len(set(m.OBSERVED_TEXT_ANCHORS)),38)
  for key in ['prefill.norm.f32','teacher.0.logits.f32','greedy.0.logits.f32','prefill.layer.31.f32']:self.assertIn(key,m.OBSERVED_TEXT_ANCHORS)
 def test_actual_native_args(self):
  for arm,argc in [('native-composed',11),('native-source-features',13)]:
   args=m.native_args('binary','model','rgb',32,32,'one','out',arm,'features');self.assertEqual(len(args),argc);self.assertEqual(args[7:10],[1536,4,'greedy-full']);self.assertEqual(args[-1],'--provisional-heldout-cap1536')
 def test_parser_failure_and_table_shape_not_rescued(self):
  gold={'text':'a b','tables':[[['a','b']]]};valid=m.score(gold,'<table><tr><td>a</td><td>b</td></tr></table>',True);ragged=m.score(gold,'<table><tr><td>a b</td></tr></table>',True);self.assertTrue(valid['table_shape_valid']);self.assertFalse(ragged['table_shape_valid']);self.assertTrue(m.floor(valid,valid,True));self.assertFalse(m.floor(ragged,valid,True));self.assertFalse(m.floor(m.score(gold,'<figure>x</figure>',True),valid,True));self.assertFalse(m.floor(m.score(gold,'a b',False),valid))
 def test_raw_row_lossless_archive(self):
  import numpy as np,zipfile
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);x=np.zeros(248320,dtype='<f4');x[5]=1;x.tofile(root/'greedy.0.logits.f32');r=m.audit_rows(root,[5]);self.assertEqual(r['rows'][0]['argmax'],5);self.assertFalse((root/'greedy.0.logits.f32').exists());self.assertTrue((root/r['archive']['file']).is_file())
   with zipfile.ZipFile(root/r['archive']['file'])as z:self.assertEqual(z.read('greedy.0.logits.f32'),x.tobytes())
if __name__=='__main__':unittest.main()
