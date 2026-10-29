"""Bounded source/stdlib controls. No arrays/models/frameworks/builds."""
import ast,copy,hashlib,importlib.util,json,pathlib,struct,subprocess,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/josev-new20-owner')]
import josev2b_task_quality_native_v1 as c
import provider
import canonical

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
worker=module('new20_worker','tools/ci-heavy/josev2b_task_quality_native_v1.py')
judge=module('new20_judge','tools/josev2b_task_quality_native_panel_v1.py')

def development():
 result=dict(completed=True,holdout_native_outputs=False,recipe_selected=False,f32=dict(strict_original_f32_numeric_and_typed_passed=True),recipes={})
 counts={'A':({'F32':292,'F16':78,'Q4_K':72},3613774304),'B':({'F32':316,'F16':78,'Q4_K':48},4651864544),'C':({'F32':316,'F16':118,'Q4_K':8},5375381984)}
 for recipe,(types,size)in counts.items():
  cases=[]
  for i in range(4):
   a=dict(model='model',usage={'input_tokens':1},answers={'field':dict(type='score',probabilities={'0':.4,'1':.6})})
   cases.append(dict(id=str(i),original_response=a,native_response_text=json.dumps(a),fields=[dict(id='field',type=2,original_argmax=1,native_argmax=1,ordinal_expectation=dict(absolute_error=.0009,budget=.001))]))
  conv=dict(completed=True,recipe=recipe,counts=types,output_bytes=size,tensors=[{}]*442,critical292_f32_byte_exact=True,independent_f16_rounding_exact=True,semantic_shapes_unchanged=True)
  result['recipes'][recipe]=dict(conversion=conv,panel=dict(structural_passed=True,compared_stages=227,strict_original_numeric_passed=False,quant_task_decisions=dict(cases=cases)))
 return result

def canonical_fixture():
 d=copy.deepcopy(c.config());dev=development();s=d['selected_candidate']
 s.update(recipe='A',checkout_source='a'*40,report_bytes=100,report_sha256='2'*64,legacy_model_bytes=3613774304,legacy_model_sha256='1'*64)
 m=d['canonicalization'];m['legacy_model']={'bytes':s['legacy_model_bytes'],'sha256':s['legacy_model_sha256']};m['canonical_model']={'bytes':s['legacy_model_bytes'],'sha256':'4'*64};m['proof']['sha256']='5'*64;m['descriptor']['sha256']='3'*64
 rows=[];inventory=[];proof_rows=[];offset=0
 for i in range(442):
  kind=0 if i<292 else 1 if i<370 else 12;label={0:'F32',1:'F16',12:'Q4_K'}[kind];size={0:1024,1:512,12:144}[kind];h={0:'a',1:'b',12:'c'}[kind]*64;name='tensor'+str(i)
  rows.append(dict(name=name,type=label,semantic_shape=[256],output_storage_rank=1,bytes=size,stored_sha256=h,source_f32_sha256=h if kind==0 else'd'*64,critical_f32=i<292))
  inventory.append(dict(name=name,type=kind,semantic_shape=[256],stored_shape=[256],bytes=size,sha256=h))
  proof_rows.append(dict(name=name,type=kind,semantic_shape=[256],stored_shape=[256],storage_rank=1,offset=offset,bytes=size,source_sha256=h,canonical_sha256=h));offset=(offset+size+31)//32*32
 dev['recipes']['A']['conversion']['tensors']=rows
 descriptor=dict(schema='crispembed-gguf-file-type-input-v1',reviewed=True,family='josev',recipe='A',producer_source=s['checkout_source'],producer_report={'bytes':100,'sha256':'2'*64},source_model=m['legacy_model'],architecture='josev2b_systemone',profile_markers={'josev2b.packed.format':'crispembed-josev-packed-q4k-v1','josev2b.packed.profile':'official-pytorch-text-batch1-512-no-extra-temperature'},inventory=inventory,report_model_sha256_path=['recipes','A','conversion','output_sha256'],report_model_bytes_path=['recipes','A','conversion','output_bytes'],report_recipe_path=['recipes','A','conversion','recipe'])
 m['descriptor']['canonical_json_sha256']=canonical.digest(json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode())
 proof=dict(schema='crispembed-gguf-file-type-canonicalization-v1',completed=True,family='josev',recipe='A',producer_source=s['checkout_source'],producer_report=descriptor['producer_report'],source_model=m['legacy_model'],canonical_model=m['canonical_model'],runtime_source=c.METADATA_RUNTIME,source_profile_sha256=c.METADATA_PROFILE_SHA,input_descriptor_file_sha256='3'*64,input_descriptor_sha256=m['descriptor']['canonical_json_sha256'],allowed_changes=[{'key':'general.file_type','kind':'UINT32','offset':128,'old':12,'new':14}],unchanged_prefix_sha256='6'*64,unchanged_suffix_sha256='7'*64,unchanged_metadata_sha256='8'*64,tensors=proof_rows,payloads_byte_exact=True,semantic_shapes_unchanged=True,source_preserved=True,metadata_only_bridge=True,numerical_qualified=False,publication_qualified=False,original_rerun=False)
 cases=[dict(id=str(i),modes=[dict(mode=mode,stages=57 if i<3 else 56,all_payload_sha_exact=True,response_bytes_exact=True)for mode in ['plain','observed','repeat']])for i in range(4)]
 response=[json.dumps({'case':i})for i in range(4)];arm=dict(response_texts=response,response_sha256=[canonical.digest(x.encode())for x in response],null_rejected=True,recovery_exact=True,gate_controls=[dict(gate=g,value=v,init_rejected=True,run_rejected=True)for g in ['CRISPEMBED_JOSEV_QUANT','CRISPEMBED_QWEN35_QUANT']for v in [None,'','0']])
 public=dict(completed=True,source=c.METADATA_RUNTIME,library_sha256='9'*64,cases=4,arms={'legacy':copy.deepcopy(arm),'canonical':copy.deepcopy(arm)},all_public_capi_response_bytes_exact=True,production_observers_excluded=True,publication_qualified=False)
 same=dict(schema='crispembed-gguf-file-type-same-payload-v1',completed=True,source=c.METADATA_RUNTIME,legacy={'structural_passed':True,'compared_stages':227,'strict_original_numeric_passed':False},canonical={'structural_passed':True,'compared_stages':227,'strict_original_numeric_passed':False},runtime_source=c.METADATA_RUNTIME,source_profile_sha256=c.METADATA_PROFILE_SHA,canonicalization_report_sha256='5'*64,source_model=m['legacy_model'],canonical_model=m['canonical_model'],four_cases=cases,all_mapped_stage_bytes_exact=True,all_typed_response_bytes_exact=True,all_public_capi_response_bytes_exact=True,public_capi=public,f32={'compared_stages':227,'strict_original_f32_numeric_and_typed_passed':True},old_numeric_results_replaced=False,numerical_qualified=False,publication_qualified=False)
 return d,descriptor,proof,same,dev

class Controls(unittest.TestCase):
 def test_pending_candidate_rejects_before_api_or_model(self):
  with mock.patch.object(provider,'api',side_effect=AssertionError('API forbidden')):
   with self.assertRaisesRegex(ValueError,'pending'):c.native_prerequisites()
 def test_original110_sources_and_contract_remain_byteexact(self):
  cfg=c.config()
  for name in cfg['protected_sha256']:
   if name=='tools/ci_process.py'or name=='tools/kaggle/josev-q4k/requirements.lock':continue
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',c.ORIGINAL+':'+name],cwd=ROOT),name)
 def test_copied_worker_has_only_declared_guard_timeout_import_changes(self):
  old=(ROOT/'tools/ci-heavy/josev2b_task_quality_v1.py').read_text();new=(ROOT/'tools/ci-heavy/josev2b_task_quality_native_v1.py').read_text()
  new=new.replace('from josev2b_task_quality_native_v1 import (','from josev2b_task_quality_v1 import (').replace("'tests','cmake','examples'","'tests','tools/cmake','examples'").replace('timeout=14400','timeout=7200').replace('from josev2b_task_quality_native_panel_v1 import compare','from josev2b_task_quality_panel_v1 import compare').replace('Separate NEW20 native caller. Frozen originals and selected mixed candidate only.','Hosted-only NEW20 public SystemOne F32/Q4 replay; explicit pending pins.')
  self.assertEqual(old,new)
 def test_panel_math_body_byteexact_and_single_failure_append(self):
  old=(ROOT/'tools/josev2b_task_quality_panel_v1.py').read_text();new=(ROOT/'tools/josev2b_task_quality_native_panel_v1.py').read_text().replace('from josev2b_task_quality_native_v1 import native_prerequisites','from josev2b_task_quality_v1 import native_prerequisites')
  self.assertEqual(old,new);self.assertEqual(new.count("result['failures'].append(record['id'])"),1)
 def test_actual_failure_reporting_one_entry_per_case(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=pathlib.Path(tmp);reference=root/'reference';reference.mkdir();(reference/'reference.json').write_bytes(b'authored fixture')
   records=[dict(id=str(i),official_response={})for i in range(20)];report=dict(arrays=[None]*1480,cases=records);outputs={}
   for mode in ['plain','observed','repeat']:
    path=root/mode;path.mkdir();(path/'native.jsonl').write_text(''.join(json.dumps(dict(id=str(i),completed=False,error='authored guard rejection'))+'\n'for i in range(20)));outputs[mode]=path
   with mock.patch.object(judge,'native_prerequisites',return_value={'original_reference_sha256':hashlib.sha256(b'authored fixture').hexdigest()}):
    result=judge.compare(reference,dict(records=records),report,outputs,root/'unused-diff',root/'comparison',precision='f32')
   self.assertEqual(result['failures'],[str(i)for i in range(20)]);self.assertFalse(result['qualified_for_owned_panel'])
 def test_actual_cmake_gitobject_exists_old_tools_cmake_does_not(self):
  self.assertEqual(subprocess.run(['git','cat-file','-e',c.RUNTIME+':cmake'],cwd=ROOT,capture_output=True).returncode,0)
  self.assertNotEqual(subprocess.run(['git','cat-file','-e',c.RUNTIME+':tools/cmake'],cwd=ROOT,capture_output=True).returncode,0)
 def test_runtime_guard_accepts_exact_objects_and_rejects_missing_or_changed(self):
  objects=c.config()['baseline_runtime_git_objects'];pins=dict(runtime_source=c.RUNTIME,runtime_protected_git_objects=objects)
  def git(args,**kwargs):
   if args[1:]==['rev-parse','HEAD']:return c.RUNTIME+'\n'
   if args[1]=='rev-parse':return objects[args[-1].split(':',1)[1]]+'\n'
   return ''
  with mock.patch.object(worker.subprocess,'check_output',side_effect=git):
   self.assertEqual(worker.runtime_guard(ROOT,pins)['source'],c.RUNTIME)
   bad=copy.deepcopy(pins);bad['runtime_protected_git_objects'].pop('cmake')
   with self.assertRaises(ValueError):worker.runtime_guard(ROOT,bad)
  with mock.patch.object(worker.subprocess,'check_output',side_effect=[c.RUNTIME+'\n','bad\n']):
   with self.assertRaises(ValueError):worker.runtime_guard(ROOT,pins)
 def test_predeclared_smallest_eligible_development_recipe(self):
  d=development();r=provider.eligible(d);self.assertEqual(r['selected'],'A');self.assertFalse(r['recipes']['A']['strict_f32_flags'])
  d['recipes']['A']['panel']['quant_task_decisions']['cases'][0]['fields'][0]['ordinal_expectation']['absolute_error']=.00101
  self.assertEqual(provider.eligible(d)['selected'],'B')
  for name in ('B','C'):d['recipes'][name]['conversion']['critical292_f32_byte_exact']=False
  self.assertIsNone(provider.eligible(d)['selected'])
 def test_development_probability_and_argmax_screen_are_separate(self):
  d=development();case=d['recipes']['A']['panel']['quant_task_decisions']['cases'][0];a=json.loads(case['native_response_text']);a['answers']['field']['probabilities']={'0':.409,'1':.591};case['native_response_text']=json.dumps(a)
  self.assertTrue(provider.eligible(d)['recipes']['A']['eligible']) # .009 is beyondold.0001 but inside separate.01 development screen.
  a['answers']['field']['probabilities']={'0':.411,'1':.589};case['native_response_text']=json.dumps(a);self.assertFalse(provider.eligible(d)['recipes']['A']['eligible'])
  d=development();d['recipes']['A']['panel']['quant_task_decisions']['cases'][0]['fields'][0]['native_argmax']=0;self.assertFalse(provider.eligible(d)['recipes']['A']['eligible'])
 def test_retained_zip_member_extent_hash_and_ambiguity_controls(self):
  import zipfile
  with tempfile.TemporaryDirectory()as tmp:
   root=pathlib.Path(tmp);archive=root/'a.zip';data=b'authored source fixture';pin=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   with zipfile.ZipFile(archive,'w')as z:z.writestr('raw/model',data)
   provider.member(archive,'raw/model',root/'copy',pin,100);self.assertEqual((root/'copy').read_bytes(),data)
   with self.assertRaises(ValueError):provider.member(archive,'../bad',root/'bad',pin,100)
   with self.assertRaises(ValueError):provider.member(archive,'raw/model',root/'wrong',dict(pin,sha256='0'*64),100)
 def test_frozen_original_model_and_development_pins(self):
  d=c.config();self.assertEqual(d['original']['version_label'],'v1');self.assertEqual(d['old_four_negative']['version_label'],'v4');self.assertEqual(d['original']['reports']['reference.json']['sha256'],'948673b0307cebcc15fdde3ea70b1163ed63b0d0bf3ef33c1488d6b89450f49b')
  self.assertEqual(d['f32_model']['bytes'],7659180448);self.assertFalse(d['selected_candidate']['reviewed']);self.assertFalse(d['publication_authorized'])
 def test_owned_and_protected_hashes(self):
  d=c.config();self.assertEqual(set(d['owned_sha256']),c.OWNED-{'PLAN.md',str(c.PATH.relative_to(ROOT))})
  for group in ['owned_sha256','protected_sha256']:
   for name,pin in d[group].items():self.assertEqual(c.sha256(ROOT/name),pin,name)
 def test_workflow_boundaries_and_output_retention(self):
  text=(ROOT/'.github/workflows/josev-new20-native-v1.yml').read_text();self.assertIn('timeout-minutes: 360',text);self.assertEqual(text.count('if: always()'),2)
  self.assertLess(text.index(' owner_v1.py')if' owner_v1.py'in text else text.index(' admit --scratch'),text.index('python -m venv'))
  self.assertIn('timeout=14400',(ROOT/'tools/ci-heavy/josev2b_task_quality_native_v1.py').read_text());self.assertNotIn('pip install torch',text)
 def test_checked_npy_actual_header_dtype_finite_policy(self):
  with tempfile.TemporaryDirectory()as tmp:
   path=pathlib.Path(tmp)/'authored.npy'
   def make(values,stage='head/memory_projection',negative=0):
    text=repr(dict(descr='<f4',fortran_order=False,shape=(1,2))).encode();header=text+b' '*(128-10-len(text)-1)+b'\n';data=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<2f',*values);path.write_bytes(data)
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),shape=[1,2],dtype='float32',stage=stage,negative_infinity=negative)
   row=make([1,2]);worker.checked_npy(path,row)
   path.write_bytes(path.read_bytes()[:-1]+b'X')
   with self.assertRaises(ValueError):worker.checked_npy(path,row)
   with self.assertRaises(ValueError):worker.checked_npy(path,make([float('nan'),2]))
   worker.checked_npy(path,make([float('-inf'),0],stage='backbone/layer/3/input/attention_mask',negative=1))
   with self.assertRaises(ValueError):worker.checked_npy(path,make([float('-inf'),0],negative=1))
 def test_exact_sdk_get_status_and_every_list_page_version(self):
  calls=[];script='source雪'
  class Request:
   def __str__(self):return json.dumps({'versionLabel':getattr(self,'version_label',None)})
  class Response:
   def __str__(self):return json.dumps({'blob':{'source':script}})
  class Obj:pass
  class Client:
   def get_kernel(self,q):calls.append(q);return Response()
   def get_kernel_session_status(self,q):calls.append(q);v=Obj();v.status='COMPLETE';return v
   def list_kernel_session_output(self,q):
    calls.append(q);v=Obj();f=Obj();f.file_name='member'+str(len(calls));f.url='https://example.invalid';v.files=[f];v.next_page_token='next'if len(calls)==3 else None;return v
  pin=dict(kernel_slug='public-slug',version_label='v1',source='source',submitted_script_bytes=len(script.encode()),submitted_script_sha256=hashlib.sha256(script.encode()).hexdigest())
  files,proof=provider.indexed_outputs(Client(),Request,Request,Request,'private-account',pin)
  self.assertEqual(len(calls),4);self.assertTrue(all(r.version_label=='v1'for r in calls));self.assertEqual(len(files),2);self.assertNotIn('private-account',json.dumps(proof))
  pin['submitted_script_bytes']=len(script)
  with self.assertRaises(ValueError):provider.indexed_outputs(Client(),Request,Request,Request,'private-account',pin)
 def test_metadata_runtime_and_canonical_proofs_cannot_activate_checkpoint(self):
  with self.assertRaisesRegex(ValueError,'pending'):c.runtime_prerequisites()
  with self.assertRaisesRegex(ValueError,'pending'):c.canonical_prerequisites()
 def test_exact56_runtime_source_and_preserved_six_baseline_files(self):
  r=c.runtime_source_proof();self.assertEqual(r['source'],c.METADATA_RUNTIME);self.assertEqual(len(r['baseline_633']['preserved_moedern_delta']),6);self.assertEqual(len(r['metadata_admission_delta']),7)
  self.assertNotEqual(r['git_objects']['src'],c.config()['baseline_runtime_git_objects']['src'])
  for category in ['baseline','metadata','math','objects']:
   d=copy.deepcopy(c.config());r=d['runtime_activation']
   if category=='baseline':r['baseline_633']['preserved_moedern_delta'].pop(next(iter(r['baseline_633']['preserved_moedern_delta'])))
   if category=='metadata':r['metadata_admission_delta'].pop('tools/quantize.cpp')
   if category=='math':r['linked_josev_math_paths'].pop('src/qwen35_decoder.cpp')
   if category=='objects':r['git_objects']['src']='0'*40
   with self.assertRaises(ValueError):c.runtime_source_proof(d)
 def test_complete_synthetic_canonical_evidence_positive(self):
  result=canonical.reports(*canonical_fixture());self.assertEqual(result['tensors'],442);self.assertEqual(result['old_four_stage_bytes'],dict(plain=227,observed=227,repeat=227));self.assertFalse(result['qualification'])
 def test_semantic_trailing_singleton_preserves_existing_storage_rank(self):
  args=canonical_fixture()
  for row in [args[1]['inventory'][0],args[2]['tensors'][0],args[4]['recipes']['A']['conversion']['tensors'][0]]:row['semantic_shape']=[256,1]
  h=canonical.digest(json.dumps(args[1],sort_keys=True,separators=(',',':')).encode());args[0]['canonicalization']['descriptor']['canonical_json_sha256']=h;args[2]['input_descriptor_sha256']=h
  self.assertEqual(canonical.reports(*args)['tensors'],442)
  args[2]['tensors'][0]['stored_shape']=[256,1];args[2]['tensors'][0]['storage_rank']=2
  with self.assertRaises(ValueError):canonical.reports(*args)
 def test_canonical_header_tensor_identity_and_no_qualification_negatives(self):
  for kind in ['header','source','canonical','payload','rank','shape','duplicate','qualification']:
   args=canonical_fixture();proof=args[2]
   if kind=='header':proof['allowed_changes'][0]['new']=12
   if kind=='source':proof['runtime_source']=c.RUNTIME
   if kind=='canonical':proof['canonical_model']=proof['source_model']
   if kind=='payload':proof['tensors'][0]['canonical_sha256']='0'*64
   if kind=='rank':proof['tensors'][0]['storage_rank']=2
   if kind=='shape':proof['tensors'][0]['semantic_shape']=[255]
   if kind=='duplicate':proof['tensors'][-1]['name']=proof['tensors'][0]['name']
   if kind=='qualification':proof['numerical_qualified']=True
   with self.subTest(kind=kind),self.assertRaises(ValueError):canonical.reports(*args)
 def test_same_payload_all_modes_all_four_public_gates_and_f32_control(self):
  for kind in ['mode','stage','response','public','gate','f32','source','numeric_flags']:
   args=canonical_fixture();same=args[3]
   if kind=='mode':same['four_cases'][0]['modes'].pop()
   if kind=='stage':same['four_cases'][0]['modes'][0]['stages']=56
   if kind=='response':same['four_cases'][0]['modes'][0]['response_bytes_exact']=False
   if kind=='public':same['public_capi']['arms']['canonical']['response_texts'][0]='changed'
   if kind=='gate':same['public_capi']['arms']['canonical']['gate_controls'][0]['run_rejected']=False
   if kind=='f32':same['f32']['strict_original_f32_numeric_and_typed_passed']=False
   if kind=='source':same['source']=c.RUNTIME
   if kind=='numeric_flags':same['canonical']['strict_original_numeric_passed']=True
   with self.subTest(kind=kind),self.assertRaises(ValueError):canonical.reports(*args)
 def test_canonical_provider_checks_reports_before_large_model_and_never_downloads_legacy(self):
  s=(ROOT/'tools/josev-new20-owner/provider.py').read_text();body=ast.get_source_segment(s,next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='candidate'))
  self.assertLess(body.index('canonical.reports('),body.index("raw=root/'canonical-raw.zip'"));self.assertLess(body.index('canonical.verify_model('),body.index('raw.unlink()'))
  self.assertNotIn("raw=root/'selected-raw.zip'",body);self.assertIn('legacy_model_downloaded=False',body)
 def test_canonical_payload_helper_rejects_unhosted_before_model_read(self):
  with mock.patch.dict('os.environ',{},clear=True),self.assertRaisesRegex(ValueError,'hosted'):
   canonical.verify_model(pathlib.Path('never-access-model'),c.config(),{})
 def test_new_cli_help_has_no_payload_imports(self):
  for name in ['tools/ci-heavy/josev2b_task_quality_owner_v1.py','tools/ci-heavy/josev2b_task_quality_native_v1.py']:
   result=subprocess.run([sys.executable,'-I','-S',str(ROOT/name),'--help'],capture_output=True,text=True,timeout=15);self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
