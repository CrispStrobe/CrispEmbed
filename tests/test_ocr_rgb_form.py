"""Source/stdlib only; no image/tokenizer/model/package/compiler payload."""
import ast,hashlib,importlib.util,json,pathlib,subprocess,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools/ocr-rgb-input'),str(ROOT/'tools/ocr-real-scans')]
import form_contract as c
from retained_inputs import mounted_image,indexed_outputs

class Controls(unittest.TestCase):
 def test_no_mounted_candidate_preserves_url_fallback(self):
  with tempfile.TemporaryDirectory()as tmp:self.assertIsNone(mounted_image({'id':'page','image':{}},tmp))
 def test_exact_single_mounted_bytes_and_public_proof(self):
  data=b'authored encoded stand-in';page={'id':'page','image':dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())}
  with tempfile.TemporaryDirectory()as tmp:
   p=pathlib.Path(tmp)/'one/input-images/page.jpg';p.parent.mkdir(parents=True);p.write_bytes(data)
   value,proof=mounted_image(page,tmp);self.assertEqual(value,data);self.assertEqual(proof['member_suffix'],'input-images/page.jpg');self.assertNotIn(tmp,json.dumps(proof))
 def test_wrong_ambiguous_and_symlink_candidates_reject(self):
  data=b'abcd';page={'id':'page','image':dict(bytes=4,sha256=hashlib.sha256(data).hexdigest())}
  with tempfile.TemporaryDirectory()as tmp:
   p=pathlib.Path(tmp)/'input-images/page.jpg';p.parent.mkdir();p.write_bytes(b'bad!')
   with self.assertRaises(ValueError):mounted_image(page,tmp)
   p.write_bytes(data);q=p.with_suffix('.png');q.write_bytes(data)
   with self.assertRaises(ValueError):mounted_image(page,tmp)
   q.unlink();p.unlink();target=pathlib.Path(tmp)/'target';target.write_bytes(data);p.symlink_to(target)
   with self.assertRaises(ValueError):mounted_image(page,tmp)
 def test_all_decoder_functions_ast_identical_to813(self):
  a=ast.parse((ROOT/'tools/ocr-rgb-input/audit.py').read_text());b=ast.parse((ROOT/'tools/ocr-rgb-input/form_audit.py').read_text())
  defs=lambda x:{n.name:ast.dump(n,include_attributes=False)for n in x.body if isinstance(n,ast.FunctionDef)}
  a,b=defs(a),defs(b);self.assertEqual(set(a),set(b))
  for name in a:
   if name!='main':self.assertEqual(a[name],b[name],name)
 def test_kernel_only_inserts_mounted_provider(self):
  a=subprocess.check_output(['git','show','813e1ec9b813ebf0dc72c8faf5841659a024fbce:tools/kaggle/ocr-real-scans/kernel.py'],cwd=ROOT,text=True)
  b=(ROOT/'tools/kaggle/ocr-real-scans/kernel.py').read_text()
  b=b.replace("\nsys.path.insert(0,'tools/ocr-real-scans')\nfrom retained_inputs import mounted_image",'')
  b=b.replace("        retained=mounted_image(p,os.environ.get('CRISPEMBED_RETAINED_IMAGE_ROOT','/kaggle/input'))\n        if retained is not None:\n            data,rec['retained_input']=retained\n        elif 'hf_parquet' in im:","        if 'hf_parquet' in im:")
  self.assertEqual(a,b)
 def test_initial19_scope_and_missing_input_never_decoded(self):
  d=json.loads((ROOT/'tests/results/ocr-real-scans/rgb-input-initial.json').read_text());self.assertEqual(len(d['rows']),20);self.assertEqual(d['pages_compared'],19);self.assertEqual(d['rgb_exact'],5)
  bad=[r for r in d['rows']if not r['input_admitted']];self.assertEqual([r['id']for r in bad],[c.PAGE_ID]);self.assertFalse(bad[0]['pillow_decode_attempted']);self.assertFalse(bad[0]['stb_decode_attempted']);self.assertFalse(d['cause_established'])
 def test_exact_version_get_status_every_page_and_script_extent(self):
  calls=[];script='source '+('a'*40)
  class Request:
   def __str__(self):return json.dumps({'versionLabel':getattr(self,'version_label',None)})
  class Response:
   def __init__(self,value):self.value=value
   def __str__(self):return json.dumps(self.value)
  class Obj:pass
  class Client:
   def get_kernel(self,q):calls.append(q);return Response({'blob':{'source':script}})
   def get_kernel_session_status(self,q):calls.append(q);v=Obj();v.status='COMPLETE';return v
   def list_kernel_session_output(self,q):
    calls.append(q);v=Obj();f=Obj();f.file_name='row'+str(len(calls));f.url='https://example.invalid';v.files=[f];v.next_page_token='second'if len(calls)==3 else None;return v
  pin=dict(kernel_slug='original',version_label='v1',source='a'*40,submitted_script_bytes=len(script.encode()),submitted_script_sha256=hashlib.sha256(script.encode()).hexdigest())
  files,proof=indexed_outputs(Client(),Request,Request,Request,'private-owner',pin);self.assertEqual(len(files),2);self.assertEqual(len(calls),4);self.assertTrue(all(q.version_label=='v1'for q in calls));self.assertNotIn('private-owner',json.dumps(proof))
  pin['submitted_script_bytes']+=1
  with self.assertRaises(ValueError):indexed_outputs(Client(),Request,Request,Request,'private-owner',pin)
 def test_source_hashes_and_exact_protected_ownership(self):
  cfg=c.declaration();old=json.loads(c.old.PATH.read_text());required=(set(old['source_sha256'])-{'tools/kaggle/ocr-real-scans/kernel.py'})|{'docs/ocr-rgb-input-v1.json','tools/kaggle/ocr-real-scans/receipt.py'}
  self.assertEqual(set(cfg['protected_sha256']),required);self.assertEqual(set(cfg['owned_sha256']),c.OWNED-{'PLAN.md','docs/ocr-rgb-form-v1.json'})
  for group in ['protected_sha256','owned_sha256']:
   for name,pin in cfg[group].items():self.assertEqual(c.sha(ROOT/name),pin,name)
 def test_provider_no_latest_fallback_or_decoder_import(self):
  s=(ROOT/'tools/ocr-rgb-input/form_provider.py').read_text();self.assertNotIn('kernels_output',s);self.assertIn('indexed_outputs(',s);self.assertIn('Fresh request: no Authorization at storage',s)
  self.assertLess(s.index("stream(files[report_member]"),s.index("stream(files[pin['image_member']]"))
 def test_audit_scope_and_workflow_evidence(self):
  s=(ROOT/'tools/ocr-rgb-input/form_audit.py').read_text();self.assertIn("if p['id']==c.PAGE_ID",s);self.assertIn("report['compared_pages']==1",s)
  workflow=(ROOT/'.github/workflows/ocr-rgb-form-v1.yml').read_text();self.assertEqual(workflow.count('if: always()'),2);self.assertIn('timeout-minutes: 30',workflow)
  self.assertLess(workflow.index('Existing retained-read credentials required before tools'),workflow.index('python -m venv'))
 def test_indexed_helper_only_adds_exact_script_extent_and_proof(self):
  old=subprocess.check_output(['git','show','3f82073eea0ec7f14512e7d57ace161eb688c7a9:tools/ci-heavy/moedern_scifact_gha_v1.py'],cwd=ROOT,text=True)
  new=(ROOT/'tools/ocr-real-scans/retained_inputs.py').read_text()
  new=new.replace("    require(len(submitted.encode())==pin['submitted_script_bytes'],'Exact submitted script extent differs')\n",'').replace("submitted_script_bytes=len(submitted.encode()),",'')
  fun=lambda text:next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='indexed_outputs')
  self.assertEqual(ast.dump(fun(old),include_attributes=False),ast.dump(fun(new),include_attributes=False))
 def test_provider_receipt_and_image_both_required_beforedecode(self):
  data=b'abcd';pin=dict(source='a'*40,kernel_slug='original',submitted_script_bytes=10,submitted_script_sha256='b'*64,report_sha256='c'*64,image=dict(bytes=4,sha256=hashlib.sha256(data).hexdigest()))
  proof=dict(schema='crispembed-ocr-rgb-form-provider-v1',completed=True,model_execution=False,initial_report_sha256=c.INITIAL_REPORT,original_report_sha256='c'*64,original=dict(source='a'*40,kernel_slug='original',version_label='v1',status='COMPLETE',submitted_script_bytes=10,submitted_script_sha256='b'*64,actual_sdk_serialized_version_on_get_status_and_all_pages=True))
  with tempfile.TemporaryDirectory()as tmp,mock.patch.object(c,'declaration',return_value={'retained_original':pin}):
   path=pathlib.Path(tmp)/(c.PAGE_ID+'.jpg');path.write_bytes(data);c.verify_provider(path.parent,proof)
   path.write_bytes(b'bad!')
   with self.assertRaises(ValueError):c.verify_provider(path.parent,proof)
   path.write_bytes(data);proof['original']['version_label']='v2'
   with self.assertRaises(ValueError):c.verify_provider(path.parent,proof)
 def test_mounted_filesystem_error_does_not_leak_private_path(self):
  import retained_inputs as helper
  with mock.patch.object(helper.os,'walk',side_effect=PermissionError('/private/account/input')):
   with self.assertRaises(ValueError)as error:helper.mounted_image({'id':'page','image':{}},ROOT)
   self.assertNotIn('/private',str(error.exception))
 def test_source_syntax_only(self):
  for name in ['form_audit.py','form_provider.py','form_contract.py']:ast.parse((ROOT/'tools/ocr-rgb-input'/name).read_text())

if __name__=='__main__':unittest.main()
