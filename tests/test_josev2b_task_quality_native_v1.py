"""Bounded source/stdlib controls. No arrays/models/frameworks/builds."""
import ast,copy,hashlib,importlib.util,json,pathlib,struct,subprocess,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/josev-new20-owner')]
import josev2b_task_quality_native_v1 as c
import provider

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
 def test_new_cli_help_has_no_payload_imports(self):
  for name in ['tools/ci-heavy/josev2b_task_quality_owner_v1.py','tools/ci-heavy/josev2b_task_quality_native_v1.py']:
   result=subprocess.run([sys.executable,'-I','-S',str(ROOT/name),'--help'],capture_output=True,text=True,timeout=15);self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
