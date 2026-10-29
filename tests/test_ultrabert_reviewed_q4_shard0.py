"""Bounded activation/provenance/journal controls; no optional packages or data."""
import ast,copy,importlib.util,json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/portfolio'))
import ultrabert_reviewed_q4_contract as c
s=importlib.util.spec_from_file_location('reviewed_q4',ROOT/'tools/portfolio/ultrabert_reviewed_q4_shard0.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def functions(self,path):
  text=path.read_text();return{n.name:ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 def test_exact_model_and_only_shard0_activation(self):
  cfg=c.declaration();c.phase_admission('shard0',cfg)
  for phase in ['model','q8','shard1','shard7','aggregate','resume']:
   with self.assertRaises(ValueError):c.phase_admission(phase,cfg)
  self.assertEqual(len(m.indices(0)),686);self.assertEqual(m.indices(0),list(range(0,5483,8)))
 def test_changed_model_receipt_archive_or_scope_rejects(self):
  cfg=c.declaration()
  changes=[lambda p:p['reviewed_dependencies']['model'].__setitem__('sha256','0'*64),
   lambda p:p['reviewed_dependencies']['model']['model'].__setitem__('sha256','f'*64),
   lambda p:p['reviewed_dependencies']['model'].__setitem__('receipt_sha256','e'*64),
   lambda p:p['activation'].__setitem__('shards',[0,1]),lambda p:p.__setitem__('q8_activation_authorized',True),
   lambda p:p['reviewed_dependencies'].__setitem__('resume',{}),lambda p:p['gates'].__setitem__('f32_max_absolute',.003)]
  for change in changes:
   bad=copy.deepcopy(cfg);change(bad)
   with self.assertRaises(ValueError):c.validate(bad)
 def test_historical_c3d_files_stay_exact(self):
  paths=['tools/portfolio/ultrabert_first_q4_shard.py','tools/portfolio/ultrabert_first_q4_contract.py','docs/ultrabert-first-q4-shard-v1.json',
         'tests/test_ultrabert_first_q4_shard.py','.github/workflows/ultrabert-first-q4-shard-v1.yml']
  for path in paths:self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',c.CONVERSION+':'+path],cwd=ROOT))
 def test_forward_and_freeze_bodies_byte_literal(self):
  before=self.functions(ROOT/'tools/portfolio/ultrabert_first_q4_shard.py');after=self.functions(ROOT/'tools/portfolio/ultrabert_reviewed_q4_shard0.py')
  for name in ['frozen_inputs','shard']:self.assertEqual(before[name],after[name])
  self.assertNotIn('prepare_model',after)
 def test_conversion_review_is_scoped_and_complete(self):
  review=c.read(c.MODEL_REVIEW);producer=c.read(c.MODEL_RECEIPT)
  self.assertEqual(review['service']['id'],40843704011);self.assertEqual(len(review['small_members']),13)
  self.assertEqual(producer['model'],c.MODEL);self.assertEqual(len(producer['inventory']),218)
  self.assertEqual(producer['actual_stored_types'],{'F32':206,'Q4_K':12})
  self.assertFalse(review['qualification']);self.assertFalse(review['review']['raw_payload_hashes_rederived_locally'])
  self.assertTrue(review['legacy_header_from_frozen018_quantizer']);self.assertFalse(review['standard_header_publication_qualified'])
 def test_exact218_schema_and_critical_precision_reject_tamper(self):
  producer=c.read(c.MODEL_RECEIPT);self.assertEqual(c.model_inventory(producer),{'F32':206,'Q4_K':12})
  for mutate in [lambda p:p['inventory'].pop('output_norm.weight'),
   lambda p:p['inventory']['token_embd.weight'].__setitem__('type','Q4_K'),
   lambda p:p['inventory']['blk.20.ffn_gate.weight'].__setitem__('shape',[1024,768])]:
   bad=copy.deepcopy(producer);mutate(bad)
   with self.assertRaises(ValueError):c.model_inventory(bad)
 def test_durable_journal_committed_prefix_and_corruption(self):
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);binding=dict(model=c.MODEL);rows=[dict(id='q:0',input_sha256='a'*64)]
   self.assertEqual(m.resume_prefix(out,binding,rows),0);data=struct.pack('<768f',*([.25]*768));m.commit_row(out,dict(index=0,id='q:0',input_sha256='a'*64),data,data)
   self.assertEqual(m.resume_prefix(out,binding,rows),1)
   with(out/'raw.f32').open('ab')as f:f.write(b'uncommitted')
   self.assertEqual(m.resume_prefix(out,binding,rows),1);self.assertEqual((out/'raw.f32').read_bytes(),data)
   (out/'normalized.f32').write_bytes(b'x'+data[1:])
   with self.assertRaisesRegex(ValueError,'committed vector'):m.resume_prefix(out,binding,rows)
 def test_workflow_source_barrier_fresh_library_and_single_job(self):
  text=(ROOT/'.github/workflows/ultrabert-reviewed-q4-shard0-v1.yml').read_text()
  self.assertIn("c.phase_admission('shard0',cfg)",text);self.assertNotIn('matrix:',text);self.assertNotIn('options: [model, shard0]',text)
  self.assertIn('ultrabert_reviewed_q4_shard0.py shard0',text);self.assertNotIn('ultrabert_first_q4_shard.py',text)
  self.assertLess(text.index('c.phase_admission'),text.index('git worktree add'));self.assertIn('LD_LIBRARY_PATH=',text)
  self.assertEqual(text.count('if: always()'),3)
 def test_no_old_global_override_and_no_optional_import_on_help(self):
  text=(ROOT/'tools/portfolio/ultrabert_reviewed_q4_shard0.py').read_text()
  for item in ['legacy.PROTOCOL=','legacy.ROOT=','old.run(','old.native(','aggregate(','prepare_model(']:self.assertNotIn(item,text)
  self.assertIn('a.shard=0;a.resume_artifact=0',text)
  p=subprocess.run([sys.executable,'-I',str(ROOT/'tools/portfolio/ultrabert_reviewed_q4_shard0.py'),'--help'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  self.assertEqual(p.returncode,0,p.stderr)
  for name in ['torch','numpy','gguf','transformers','huggingface_hub']:self.assertNotIn(name,sys.modules)
if __name__=='__main__':unittest.main()
