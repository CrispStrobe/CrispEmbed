"""Bounded source/service/journal controls; no packages, model, build or inference."""
import ast,copy,hashlib,importlib.util,json,os,struct,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/portfolio'))
import ultrabert_remaining_q4_contract as c
import ultrabert_remaining_q4 as m
class Controls(unittest.TestCase):
 def functions(self,path):
  text=path.read_text();return {n.name:ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 def test_complete_disjoint_strided_coverage_and_call_caps(self):
  cfg=c.declaration();seen=[j for i in range(8)for j in m.indices(i)]
  self.assertEqual(sorted(seen),list(range(5483)));self.assertEqual(len(set(seen)),5483)
  self.assertEqual(2*sum(len(m.indices(i))for i in range(1,8)),9594)
  self.assertEqual(cfg['counts']['aggregate_public_calls'],9594+1372)
  self.assertEqual(cfg['activation']['shards'],list(range(1,8)))
 def test_exact_candidate_and_prior_scope_reject_changes(self):
  cfg=c.declaration()
  changes=[lambda p:p['activation'].__setitem__('shards',list(range(8))),lambda p:p.__setitem__('resume',True),
   lambda p:p['model'].__setitem__('sha256','0'*64),lambda p:p['counts'].__setitem__('original_forwards',1),
   lambda p:p['counts'].__setitem__('new_public_calls',9595),lambda p:p['gates'].__setitem__('top1_minimum_agreement',.98),
   lambda p:p['shard0']['artifact'].__setitem__('artifact_id',1),lambda p:p['shard0']['source_binding'].__setitem__('new_controller_sha256','f'*64),
   lambda p:p['service_dependencies']['shard0'].__setitem__('run_attempt',2),
   lambda p:p['shard0']['manifest']['binding.json'].__setitem__('sha256','0'*64)]
  for change in changes:
   bad=copy.deepcopy(cfg);change(bad)
   with self.assertRaises(ValueError):c.validate(bad)
 def test_full_closed_owned_and_inherited_sources(self):
  cfg=c.declaration();old=c.previous.declaration()
  expected=set(old['protected_sha256'])|set(old['owned_sha256'])|{str(c.previous.PATH.relative_to(ROOT)),str(c.FIRST_REVIEW.relative_to(ROOT)),'docs/ultrabert-first-q4-shard0-result.md'}
  self.assertEqual(set(cfg['protected_sha256']),expected);self.assertEqual(set(cfg['owned_sha256']),c.OWNED-{str(c.PATH.relative_to(ROOT))})
  for key in ('protected_sha256','owned_sha256'):
   for name,pin in cfg[key].items():self.assertEqual(c.sha(ROOT/name),pin,name)
 def test_historical_controllers_and_protocols_byte_exact(self):
  for name in ['tools/portfolio/ultrabert_reviewed_q4_shard0.py','tools/portfolio/ultrabert_reviewed_q4_contract.py','docs/ultrabert-reviewed-q4-shard0-v1.json','tools/portfolio/ultrabert_first_q4_shard.py','tools/portfolio/ultrabert_q4k_sharded.py','docs/ultrabert-q4k-sharded-v1.json']:
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show','a5feac83e177d7e08cfa23384089f7584f0736a9:'+name],cwd=ROOT),name)
 def test_exact_forward_body_except_explicit_attempt_accounting(self):
  before=self.functions(ROOT/'tools/portfolio/ultrabert_reviewed_q4_shard0.py');after=self.functions(ROOT/'tools/portfolio/ultrabert_remaining_q4.py')
  self.assertEqual(before['frozen_inputs'],after['frozen_inputs'])
  body=after['shard']
  body=body.replace('    require(start == 0, "This first continuation never resumes or repeats committed forwards")\n    report.update(raw_forward_attempts=0, normalized_forward_attempts=0)\n','')
  for key,why in [('raw_forward_attempts','Raw'),('normalized_forward_attempts','Normalized')]:
   body=body.replace("            report['"+key+"'] += 1\n            require(report['"+key+"'] <= len(rows), '"+why+" per-shard forward cap')\n            write(args.out / 'shard.json', report)\n",'')
  self.assertEqual(body,before['shard'])
  for forbidden in ['legacy.PROTOCOL=','legacy.ROOT=','previous.PATH=','old.run(','old.native(','prepare_model(']:self.assertNotIn(forbidden,(ROOT/'tools/portfolio/ultrabert_remaining_q4.py').read_text())
 def test_unchanged_ranking_implementation_and_task_thresholds(self):
  text=(ROOT/'tools/portfolio/ultrabert_remaining_q4.py').read_text()
  self.assertIn('before=old.rankings(data,original_norm);after=old.rankings(data,norm);quality=old.quality(before,after)',text)
  self.assertIn("return 0 if report['passed']else 1",text)
  cfg=c.declaration();self.assertEqual(cfg['gates']['ndcg_mrr_recall_maximum_degradation'],.001);self.assertEqual(cfg['gates']['top1_minimum_agreement'],.99)
 def test_wrong_prior_service_rejects_before_archive_transfer(self):
  cfg=c.declaration();pin=cfg['shard0']['artifact'];bad=dict(cfg['service_dependencies']['shard0'],head_sha='0'*40)
  with tempfile.TemporaryDirectory()as tmp,patch.object(m.legacy,'artifact')as transfer:
   with self.assertRaises(ValueError):m.artifact(types.SimpleNamespace(api=lambda _:bad),pin['artifact_id'],Path(tmp)/'payload',pin,128*2**20)
   transfer.assert_not_called()
 def test_shared_cache_retains_full_manifest_authentication(self):
  cfg=c.declaration();pin=cfg['shard0']['artifact'];service=cfg['service_dependencies']['shard0']
  def api(path):return service if path.startswith('actions/runs/')else dict(workflow_run=dict(id=service['id']))
  def transfer(old,ident,target,pin,maximum):
   target.mkdir();(target/'evidence').write_bytes(b'kept');m.write(target/'manifest.json',m.legacy.manifest(target));return target
  with tempfile.TemporaryDirectory()as tmp,patch.object(m.legacy,'artifact',side_effect=transfer)as fetch:
   target=Path(tmp)/'payload';old=types.SimpleNamespace(api=api)
   self.assertEqual(m.artifact(old,pin['artifact_id'],target,pin,128*2**20),target)
   self.assertEqual(m.artifact(old,pin['artifact_id'],target,pin,128*2**20),target);self.assertEqual(fetch.call_count,1)
   (target/'evidence').write_bytes(b'changed')
   with self.assertRaises(ValueError):m.artifact(old,pin['artifact_id'],target,pin,128*2**20)
 def fixture(self,base):
  frozen=base/'freeze';root=base/'shard';frozen.mkdir();root.mkdir();(root/'comparison').mkdir();cfg=c.declaration()
  m.write(frozen/'freeze.json',dict(completed=True));m.write(frozen/'dataset.json',dict(queries=[dict(id='q:0',text='first'),dict(id='q:1',text='second')],documents=[]))
  m.write(frozen/'lengths.json',[dict(id='q:0',ids=[1]),dict(id='q:1',ids=[2])])
  source=dict(audit_source='f'*40,runtime_source=m.RUNTIME,runtime_git_objects=m.read(m.legacy.PROTOCOL)['runtime_git_objects'],ggml=cfg['ggml'],controller_sha256=m.sha(ROOT/'tools/portfolio/ultrabert_q4k_sharded.py'),new_controller_sha256=m.sha(Path(m.__file__)),new_protocol_sha256=m.sha(m.PROTOCOL))
  lib=dict(bytes=1,sha256='a'*64);binary=dict(bytes=1,sha256='b'*64)
  row=dict(id='q:1',text='second');expected={'q:1':dict(ids=[2])};inputs=[dict(id='q:1',input_sha256=m.row_identity(row,expected))]
  binding=dict(source={k:source[k]for k in ['runtime_source','runtime_git_objects','ggml','controller_sha256','new_controller_sha256']},library=lib,comparator=binary,model=cfg['model'],model_receipt_sha256=cfg['reviewed_dependencies']['model']['receipt_sha256'],frozen_receipt_sha256=m.sha(frozen/'freeze.json'),shard=1,recipe=m.FIRST_RECIPE,inputs=inputs)
  data=struct.pack('<f',0.)*768
  m.legacy.resume_prefix(root,binding,inputs);m.legacy.commit_row(root,dict(index=0,global_index=1,actual_ids_exact=True,**inputs[0]),data,data)
  result=dict(max_absolute_budget=.002,minimum_row_cosine=.99999,status='passed',stages=[dict(name=k,ggml_shape=[768,1],row_axis=0,elements=768,found=True)for k in ['raw','normalized']])
  m.write(root/'comparison/diff.json',result)
  report=dict(source=source,build=dict(library=lib,comparator=binary),completed=True,recipe=m.FIRST_RECIPE,shard=1,completed_rows=1,total_rows=1,all_ids_exact=True,original_forwards=0,f32_forwards=0,quantized_qualified=False,strict_f32_vector_flags=True,streams={k:m.identity(root/k)for k in ['raw.f32','normalized.f32','rows.jsonl']},actual_ref=dict(exit=0,result=result))
  m.write(root/'shard.json',report);m.write(root/'manifest.json',m.legacy.manifest(root));return root,frozen,cfg
 def test_committed_row_identity_and_signed_zero_rejection(self):
  with tempfile.TemporaryDirectory()as tmp,patch.object(m,'indices',return_value=[1]):
   root,frozen,cfg=self.fixture(Path(tmp))
   self.assertTrue(m.authenticate_shard(root,1,cfg,frozen,'f'*40)['completed'])
   path=root/'raw.f32';data=bytearray(path.read_bytes());data[:4]=struct.pack('<f',-0.);path.write_bytes(data)
   # A changed saved file manifest alone cannot relabel a row's signed-zero hash.
   m.write(root/'manifest.json',m.legacy.manifest(root))
   with self.assertRaisesRegex(ValueError,'Committed row vector hash'):m.authenticate_shard(root,1,cfg,frozen,'f'*40)
 def test_partial_or_wrong_input_journal_rejects(self):
  with tempfile.TemporaryDirectory()as tmp,patch.object(m,'indices',return_value=[1]):
   root,frozen,cfg=self.fixture(Path(tmp));path=root/'rows.jsonl';saved=path.read_bytes();path.write_bytes(saved[:-1]);m.write(root/'manifest.json',m.legacy.manifest(root))
   with self.assertRaisesRegex(ValueError,'Torn journal'):m.authenticate_shard(root,1,cfg,frozen,'f'*40)
   record=json.loads(saved);record['global_index']=2;path.write_text(json.dumps(record)+'\n');m.write(root/'manifest.json',m.legacy.manifest(root))
   with self.assertRaisesRegex(ValueError,'Journal row/input'):m.authenticate_shard(root,1,cfg,frozen,'f'*40)
 def test_workflow_single_job_barriers_and_retention(self):
  text=(ROOT/'.github/workflows/ultrabert-remaining-q4-v1.yml').read_text()
  self.assertNotIn('matrix:',text);self.assertIn('timeout-minutes: 360',text);self.assertEqual(text.count('if: always()'),3)
  self.assertLess(text.index('c.verify_source'),text.index('git worktree add'));self.assertIn('LD_LIBRARY_PATH=',text)
  self.assertIn('tools/portfolio/ultrabert_remaining_q4.py --reviewed-source',text)
  self.assertIn('payload/shards/*/shard.json',text);self.assertIn('payload/query-summary.json',text)
  source=(ROOT/'tools/portfolio/ultrabert_remaining_q4.py').read_text()
  self.assertLess(source.index('reports[0]=authenticate_shard'),source.index('for index in range(1,8)'))
 def test_isolated_help_requires_no_optional_import(self):
  result=subprocess.run([sys.executable,'-I',str(ROOT/'tools/portfolio/ultrabert_remaining_q4.py'),'--help'],cwd=tempfile.gettempdir(),capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
  for name in ['numpy','gguf','torch','transformers']:self.assertNotIn(name,sys.modules)
if __name__=='__main__':unittest.main()
