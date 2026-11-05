#!/usr/bin/env python3
"""One hosted job: stage one authenticated mixed model, build once, run two frozen calibration cases sequentially."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
MODEL={'run':42006450732,'source':'e2beb641773c649394bd4e36526ff50c665f84cb','artifact':13405370923,
 'archive_bytes':3830524370,'archive_sha256':'6d7e72a423ae0e0b1ab5fd331bad240174c2b29421fe7b585130868cb8a0ee10',
 'receipt_sha256':'87cb309fd4bb3f9d988042bf66b9dc4fe10e6980f1e2248741404043a9096977',
 'file':'lightonocr3-4b-q4k-mixers-q8-v1.gguf','bytes':4683041248,
 'sha256':'d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474'}
ORIGINAL={'source':'42e5ce6f9bc6bd01865ac4f21d2157c98d3475f1','slug':'lighton4b-fourcase-bf16-v1','version':'v1',
 'bootstrap_sha256':'5be2b05e993791fe0aa88a61be6ca3997ba7f78b33021091149d6862ff7acffe',
 'batch_sha256':'bd8b4c4c095f17c7873c510d97b07f02868b62592251581d07c0362f3d762502'}
CASES=(('simple_table','6be5428f247b616e76d79513b62873cf0238842999be2fb0848997f8ced25763',222),
 ('arabic_printed_line','ff1a390370b702fabf8b5a25f0feec76888bb1ad8197cf08c3efd44adec1de84',6))
CASE_TIMEOUT_SECONDS=90*60
MAX_ORIGINAL_TRANSFER=512*1024**2

class CalibrationError(Exception):pass
def require(ok,message):
 if not ok:raise CalibrationError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=120))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def bounded_download(url,path,expected_sha,maximum,exact=None):
 count=0;h=hashlib.sha256()
 try:
  with urllib.request.urlopen(url,timeout=120) as response,path.open('xb') as stream:
   while block:=response.read(min(8*1024**2,maximum-count+1)):
    count+=len(block);require(count<=maximum,'original transfer byte bound exceeded');h.update(block);stream.write(block)
 except CalibrationError:raise
 except Exception:raise CalibrationError('authenticated original transfer failed') from None
 require(h.hexdigest()==expected_sha and (exact is None or count==exact),'authenticated original bytes differ')
 return count

def original_urls():
 from kaggle.api.kaggle_api_extended import KaggleApi
 client_api=KaggleApi();client_api.authenticate()
 Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
 owner=os.environ['KAGGLE_ACCOUNT'];urls={};token=None;pages=0
 with client_api.build_kaggle_client() as client:
  req=Get();req.user_name=owner;req.kernel_slug=ORIGINAL['slug'];req.version_label=ORIGINAL['version']
  metadata=json.loads(str(client.kernels.kernels_api_client.get_kernel(req)))
  require(hashlib.sha256(metadata['blob']['source'].encode()).hexdigest()==ORIGINAL['bootstrap_sha256'],'original submitted source differs')
  while True:
   pages+=1;require(pages<=32,'original file listing page bound exceeded')
   req=List();req.user_name=owner;req.kernel_slug=ORIGINAL['slug'];req.version_label=ORIGINAL['version'];req.page_size=100
   if token:req.page_token=token
   page=client.kernels.kernels_api_client.list_kernel_session_output(req)
   for item in page.files or []:
    require(item.file_name not in urls,'duplicate original output member');urls[item.file_name]=item.url
   token=page.next_page_token
   if not token:break
 return urls

def stage_model(scratch,out):
 run=api('actions/runs/'+str(MODEL['run']));artifact=api('actions/artifacts/'+str(MODEL['artifact']))
 require((run['head_sha'],run['conclusion'],run['run_attempt'])==(MODEL['source'],'success',1),'mixed producer run differs')
 require(not artifact['expired'] and artifact['workflow_run']['id']==MODEL['run'] and artifact['digest']=='sha256:'+MODEL['archive_sha256'] and
  artifact['size_in_bytes']==MODEL['archive_bytes'],'mixed artifact identity differs')
 archive=scratch/'mixed.zip'
 with archive.open('xb') as stream:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(MODEL['artifact'])+'/zip'],stdout=stream,check=True,timeout=900)
 require(archive.stat().st_size==MODEL['archive_bytes'] and sha(archive)==MODEL['archive_sha256'],'mixed archive hash differs')
 model=scratch/MODEL['file']
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();require(len(names)<=128 and len(names)==len(set(names)),'mixed archive member count differs')
  candidates=[n for n in names if Path(n).name==MODEL['file']];receipts=[n for n in names if Path(n).name=='mixed-candidate.json']
  require(len(candidates)==len(receipts)==1,'mixed candidate or receipt missing/ambiguous')
  require(z.getinfo(candidates[0]).file_size==MODEL['bytes'] and z.getinfo(receipts[0]).file_size<=2**20,'mixed member size differs')
  data=z.read(receipts[0]);require(hashlib.sha256(data).hexdigest()==MODEL['receipt_sha256'],'mixed pack receipt differs')
  (out/'mixed-candidate.json').write_bytes(data)
  with z.open(candidates[0]) as stream,model.open('xb') as dest:
   while block:=stream.read(8*1024**2):dest.write(block)
 require(model.stat().st_size==MODEL['bytes'] and sha(model)==MODEL['sha256'],'mixed model file differs')
 # Only this verified task-owned transport copy is transient; remote archive/model remain preserved.
 archive.unlink()
 return model

def stage_originals(scratch,out,worker):
 urls=original_urls();prefix='lighton4b-fourcase-bf16/';total=0;original=scratch/'original';original.mkdir()
 def get(member,path,digest,bound,exact=None):
  nonlocal total
  require(member in urls,'reviewed original member missing')
  size=bounded_download(urls[member],path,digest,bound,exact);total+=size
  require(total<=MAX_ORIGINAL_TRANSFER,'aggregate original transfer bound exceeded')
 get(prefix+'batch.json',original/'batch.json',ORIGINAL['batch_sha256'],4*1024**2)
 batch=json.loads((original/'batch.json').read_text())
 require(batch['status']=='captured' and batch['source_sha']==ORIGINAL['source'] and batch['model_dtype']=='bfloat16' and batch['model_loads']==1 and
  batch['protocol_sha256']==worker.PROTOCOL_SHA and batch['input_manifest_sha256']==worker.FROZEN_SHA,'original batch source/protocol differs')
 require(batch['cases']['receipt_example']['cap_hit'] is True and batch['cases']['receipt_example']['eos_completed'] is False,'retained heldout source failure differs')
 (out/'original-batch.json').write_bytes((original/'batch.json').read_bytes())
 for case_id,capture_sha,positions in CASES:
  _,case,_=worker.load_contract(case_id);folder=original/case_id;folder.mkdir()
  get(prefix+case_id+'/capture.json',folder/'capture.json',capture_sha,2**20)
  capture=json.loads((folder/'capture.json').read_text());worker.validate_original(capture,case)
  require(batch['cases'][case_id]['capture_sha256']==capture_sha and len(capture['generated_ids'])==positions and len(capture['scores'])==positions and
   capture['eos_completed'] is True and capture['cap_hit'] is False,'calibration original completion differs')
  for field,row in capture['prepared_inputs'].items():
   require(field in ('pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'),'undeclared original input field')
   require(Path(row['file']).name==row['file'],'original input path differs')
   get(prefix+case_id+'/'+row['file'],folder/row['file'],row['sha256'],8*1024**2)
  require(len(capture['prepared_inputs'])==5,'complete official input fields required')
  for i,row in enumerate(capture['scores']):
   require(row['position']==i and row['shape']==[1,248320] and row['bytes']==993280 and row['file']==f'scores-{i:03d}.f32','original full-score record differs')
   get(prefix+case_id+'/'+row['file'],folder/row['file'],row['sha256'],993280,993280)
 return original,total

def stop_group(process):
 os.killpg(process.pid,signal.SIGTERM)
 try:process.wait(timeout=15)
 except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def tiny_archive(path,data):
 name=b'generated.scores.0';header=struct.pack('<4sIQQQ',b'GGUF',3,1,0,len(name))+name
 header+=struct.pack('<IQQIQ',2,248320,1,0,0)
 with path.open('wb') as stream:stream.write(header);stream.write(b'\0'*((-len(header))%32));stream.write(data)

def compare_case(case_id,original,native_root,binary,scratch,out):
 capture=json.loads((original/'capture.json').read_text());native=json.loads((native_root/'native-case.json').read_text())
 require(native['status']=='completed' and native['recipe']=='lighton4b-q4k-mixers-q8-v1' and native['weights_sha256']==MODEL['sha256'] and
  native['original_capture_sha256']==sha(original/'capture.json') and native['teacher_forced_ids']==capture['generated_ids'] and
  native['independent_generation'] is True and native['actual_cap']==256 and native['input_admitted'] is True,'native case contract differs')
 require(len(native['scores'])==len(capture['scores']),'complete teacher score positions required')
 work=scratch/('compare-'+case_id);work.mkdir();order=work/'order.tsv';order.write_text('generated.scores.0 float 248320 1 1 1\n')
 rows=[];all_passed=True
 with (out/(case_id+'-actual-ref.log')).open('w') as log:
  for i,(a,b) in enumerate(zip(capture['scores'],native['scores'])):
   require(b['position']==i and b['shape']==[1,248320] and b['bytes']==993280,'native row geometry differs')
   path=native_root/b['file'];require(path.resolve().is_relative_to(native_root.resolve()) and path.stat().st_size==993280 and sha(path)==b['sha256'],'native score bytes differ')
   raw=(original/a['file']).read_bytes();require(len(raw)==993280 and hashlib.sha256(raw).hexdigest()==a['sha256'],'original row changed before comparison')
   tiny_archive(work/'original.gguf',raw);tiny_archive(work/'native.gguf',path.read_bytes())
   result=subprocess.run([str(binary),str(work/'original.gguf'),str(work/'native.gguf'),str(order)],capture_output=True,text=True,timeout=60)
   log.write('POSITION '+str(i)+'\n'+result.stdout+result.stderr);log.flush()
   stages=[x for x in result.stdout.splitlines() if x.startswith('STAGE ')];require(result.returncode in (0,1) and len(stages)==1,'complete actual Ref result required')
   metrics=dict(x.split('=',1) for x in stages[0].split()[2:] if '=' in x)
   require(metrics['finite']=='1' and metrics['count']=='248320','finite complete score comparison required')
   rows.append({'position':i,**metrics});all_passed &= result.returncode==0
 decoded=capture['generated_ids']==native['generated_ids'] and capture['decoded']==native['decoded']
 return {'case_id':case_id,'positions':len(rows),'metrics':rows,'numerical_passed':all_passed,'decoded_exact':decoded,
  'original_tokens':len(capture['generated_ids']),'native_tokens':len(native['generated_ids']),'native_eos_completed':native['eos_completed'],
  'native_cap_hit':native['cap_hit'],'native_receipt_sha256':sha(native_root/'native-case.json'),
  'passed':all_passed and decoded and native['eos_completed'] and not native['cap_hit']}

def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable calibration only')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);out.mkdir(exist_ok=True);scratch.mkdir(exist_ok=True)
 report={'schema':'lighton4b-sequential-calibration-native/v1','status':'preparing','quality_qualified':False,'scope':'two calibration cases only; no heldout qualification',
  'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model':MODEL,'original':ORIGINAL,'cases':[],
  'case_timeout_seconds':CASE_TIMEOUT_SECONDS,'original_transfer_bound':MAX_ORIGINAL_TRANSFER,'original_forward_executed':False}
 try:
  import shutil
  worker=module('frozen_case_worker',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py')
  common=module('native_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py')
  report['preflight']={'memory':common.memory(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
  require(report['preflight']['memory']['available_bytes']>=8*1024**3 and report['preflight']['scratch_free_bytes']>=12*1024**3 and
   report['preflight']['output_free_bytes']>=3*1024**3,'resource preflight failed before artifact acquisition')
  report['stage']='stage-authenticated-mixed-model';model=stage_model(scratch,out)
  report['stage']='stage-reviewed-originals';original,total=stage_originals(scratch,out,worker);report['original_transfer_bytes']=total
  report['heldout_receipt_source_completion_failed']=True
  report['stage']='build-once';build_out=out/'build';build_out.mkdir();build_env=os.environ.copy();build_env['HEAVY_OUT']=str(build_out)
  subprocess.run([sys.executable,str(ROOT/'tools/ci-heavy/lighton4b_frozen_case_build.py')],cwd=ROOT,env=build_env,check=True,timeout=20*60)
  build=scratch/'frozen-case-build';receipt=build_out/'frozen-case-build.json';receipt_sha=sha(receipt)
  worker.validate_build(build,receipt,receipt_sha,report['source_sha'])
  for case_id,capture_sha,_ in CASES:
   report['stage']='native-'+case_id;case_out=out/case_id;case_scratch=scratch/case_id;case_scratch.mkdir()
   env=os.environ.copy();env.update(HEAVY_OUT=str(out),HEAVY_SCRATCH=str(case_scratch))
   with (out/(case_id+'-worker.log')).open('w') as log:
    process=subprocess.Popen([sys.executable,str(ROOT/'tools/ci-heavy/lighton4b_frozen_case.py'),'--case',case_id,'--model',str(model),
     '--weights-sha256',MODEL['sha256'],'--recipe','lighton4b-q4k-mixers-q8-v1','--original-root',str(original/case_id),
     '--original-capture-sha256',capture_sha,'--mode','full','--threads','4','--prebuilt-build',str(build),
     '--build-receipt',str(receipt),'--build-receipt-sha256',receipt_sha],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=CASE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:stop_group(process);report['cases'].append({'case_id':case_id,'timed_out':True,'passed':False});continue
   if code:report['cases'].append({'case_id':case_id,'execution_failed':True,'returncode':code,'passed':False});continue
   report['cases'].append(compare_case(case_id,original/case_id,case_out,build/'crispembed-diff-lighton4b-document',scratch,out))
  report['status']='completed';report['calibration_passed']=len(report['cases'])==2 and all(x['passed'] for x in report['cases'])
  return 0 if report['calibration_passed'] else 1
 except Exception as error:
  report['status']='failed';report['error_type']=type(error).__name__
  report['error']=str(error)[:1000] if isinstance(error,CalibrationError) else 'hosted staging/build/calibration failed'
  return 1
 finally:(out/'calibration-native.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
