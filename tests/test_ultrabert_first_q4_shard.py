#!/usr/bin/env python3
"""Bounded source/pending activation/journal controls; no models or NumPy."""
import ast,copy,importlib.util,json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/portfolio'))
import ultrabert_first_q4_contract as c
s=importlib.util.spec_from_file_location('first_q4',ROOT/'tools/portfolio/ultrabert_first_q4_shard.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_model_phase_ready_shard0_pending_before_payload(self):
  cfg=c.declaration();c.phase_admission('model',cfg)
  self.assertIsNone(cfg['reviewed_dependencies']['model']);self.assertFalse(cfg['activation']['reviewed'])
  with self.assertRaisesRegex(ValueError,'pending reviewed model'):c.phase_admission('shard0',cfg)
  for bad in ['shard','aggregate','q8','shard1']:
   with self.assertRaises(ValueError):c.phase_admission(bad,cfg)
 def test_model_identity_and_explicit_shard_review_both_required(self):
  cfg=copy.deepcopy(c.declaration());cfg['reviewed_dependencies']['model']=dict(c.approval('freeze',cfg),model=dict(bytes=1,sha256='a'*64))
  with self.assertRaisesRegex(ValueError,'shard0'):c.phase_admission('shard0',cfg)
  cfg['activation']['reviewed']=True;c.phase_admission('shard0',cfg)
  cfg['reviewed_dependencies']['model'].pop('model')
  with self.assertRaisesRegex(ValueError,'model SHA'):c.phase_admission('shard0',cfg)
 def functions(self,path):
  text=path.read_text();return{n.name:ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 def test_original_conversion_body_literal_and_shard_only_identity_delta(self):
  before=self.functions(ROOT/'tools/portfolio/ultrabert_q4k_sharded.py');after=self.functions(ROOT/'tools/portfolio/ultrabert_first_q4_shard.py')
  self.assertEqual(before['prepare_model'],after['prepare_model'])
  wanted=before['shard'].replace("'ggml', 'controller_sha256']","'ggml', 'controller_sha256', 'new_controller_sha256']").replace("    checked_file(model, model_record['model'])","    checked_file(model, model_record['model'])\n    require(model_record['model']==pin['model'],'Explicit reviewed converted model SHA differs')")
  self.assertEqual(wanted,after['shard'])
 def test_old_sources_and_protocol_byte_exact(self):
  for p in ['tools/portfolio/ultrabert_q4k_sharded.py','docs/ultrabert-q4k-sharded-v1.json','tests/test_ultrabert_q4k_sharded.py','.github/workflows/ultrabert-q4k-sharded-v1.yml']:
   self.assertEqual((ROOT/p).read_bytes(),subprocess.check_output(['git','show','08b0a5727aa25412a1181db4695efb01be578b24:'+p],cwd=ROOT))
 def test_shard0_exact_original_striding(self):self.assertEqual(m.indices(0),list(range(0,5483,8)));self.assertEqual(len(m.indices(0)),686)
 def test_durable_prefix_torn_corrupt_or_wrong_binding_reject(self):
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp);binding=dict(model='a'*64,source='b'*40);rows=[dict(id='q:0',input_sha256='c'*64)]
   self.assertEqual(m.resume_prefix(p,binding,rows),0);raw=struct.pack('<768f',*([.25]*768));m.commit_row(p,dict(index=0,id='q:0',input_sha256='c'*64),raw,raw)
   self.assertEqual(m.resume_prefix(p,binding,rows),1)
   with self.assertRaisesRegex(ValueError,'binding'):m.resume_prefix(p,dict(binding,model='d'*64),rows)
   with(p/'rows.jsonl').open('ab')as f:f.write(b'{')
   with self.assertRaisesRegex(ValueError,'torn'):m.resume_prefix(p,binding,rows)
 def test_uncommitted_tail_truncated_committed_hash_still_strict(self):
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp);binding=dict(source='authored');rows=[dict(id='q:0',input_sha256='f'*64)];m.resume_prefix(p,binding,rows);raw=bytes(3072);m.commit_row(p,dict(index=0,id='q:0',input_sha256='f'*64),raw,raw)
   with(p/'raw.f32').open('ab')as f:f.write(b'uncommitted')
   self.assertEqual(m.resume_prefix(p,binding,rows),1);self.assertEqual((p/'raw.f32').read_bytes(),raw)
   (p/'normalized.f32').write_bytes(b'x'+raw[1:])
   with self.assertRaisesRegex(ValueError,'committed vector'):m.resume_prefix(p,binding,rows)
 def test_no_old_global_override_or_whole_panel(self):
  text=(ROOT/'tools/portfolio/ultrabert_first_q4_shard.py').read_text()
  for value in ['legacy.PROTOCOL=','legacy.ROOT=','old.run(','old.native(','aggregate(']:self.assertNotIn(value,text)
  self.assertIn("a.shard=0",text);self.assertIn("model_record['model']==pin['model']",text)
 def test_frozen_payload_all77_and_exact_receipt_required(self):
  text=(ROOT/'tools/portfolio/ultrabert_first_q4_shard.py').read_text()
  self.assertIn("legacy.validate_manifest(root)==review['payload_manifest']['members']",text);self.assertIn("receipt['source']['audit_source']==pin['source']",text)
  cfg=c.declaration();self.assertEqual(cfg['reviewed_dependencies']['freeze']['artifact_id'],12880973405);self.assertEqual(cfg['reviewed_dependencies']['freeze']['receipt_sha256'],'e70d3bc120a89854d569e748319f85467a587e0140db76e7db78a91c9f903b1a')
 def test_workflow_is_bounded_model_then_shard0_only(self):
  s=(ROOT/'.github/workflows/ultrabert-first-q4-shard-v1.yml').read_text()
  for token in ['options: [model, shard0]','timeout-minutes: 360','--parallel 2','c.legacy.admission(','m.symbol_audit(','-DGGML_LLAMAFILE=ON','-DGGML_NATIVE=OFF','LD_LIBRARY_PATH=']:self.assertIn(token,s)
  self.assertNotIn('matrix:',s);self.assertNotIn('KAGGLE_',s);self.assertNotIn('aggregate',s);self.assertEqual(s.count('if: always()'),3)
  self.assertLess(s.index('c.phase_admission'),s.index('git worktree add'))
 def test_new_exact_source_closure(self):
  cfg=c.verify_source(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());self.assertEqual(len(cfg['protected_git_objects']),11);self.assertEqual(cfg['gates'],c.read(c.legacy.PROTOCOL)['gates'])
 def test_isolated_help_and_no_optional_imports(self):
  p=subprocess.run([sys.executable,'-I',str(ROOT/'tools/portfolio/ultrabert_first_q4_shard.py'),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);self.assertEqual(p.returncode,0,p.stderr)
  for name in ['numpy','gguf','torch','transformers','huggingface_hub']:self.assertNotIn(name,sys.modules)
if __name__=='__main__':unittest.main()
