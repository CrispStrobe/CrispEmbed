#!/usr/bin/env python3
"""Independent full archive/model audit, then one predeclared A/logo heldout diagnostic."""
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def require(x,message):
 if not x:raise ValueError(message)
def streamhash(stream,limit):
 h=hashlib.sha256();count=0
 while block:=stream.read(8*1024**2):count+=len(block);require(count<=limit,'bounded stream overflow');h.update(block)
 return count,h.hexdigest()
def edits(a,b):
 d=[[0]*(len(b)+1)for _ in range(len(a)+1)]
 for i in range(len(a)+1):d[i][0]=i
 for j in range(len(b)+1):d[0][j]=j
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+int(a[i-1]!=b[j-1]))
 return d[-1][-1]
def spanhash(stream,count):
 h=hashlib.sha256();left=count
 while left:
  block=stream.read(min(left,8*1024**2));require(bool(block),'truncated tensor span');h.update(block);left-=len(block)
 return count,h.hexdigest()
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
 import numpy as np
 from PIL import Image
 require(np.__version__=='2.3.5' and Image.__version__=='11.3.0','serializer and RGB pins')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'logo-heldout';scratch.mkdir()
 common=load('native',ROOT/'tools/ci-heavy/lighton4b_native_document.py');transport=load('transport',ROOT/'tools/ci-heavy/lighton4b_calibration_native.py');worker=load('worker',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py');full=load('full',ROOT/'tools/ci-heavy/lighton4b_gdn_out_full_calibration.py');comparison=load('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py')
 sys.path.insert(0,str(ROOT/'tools'));import lighton4b_gdn_out_pack as pack
 protocolpath=ROOT/'docs/lighton4b-gdn-out-logo-heldout-v1.json';protocol=json.loads(protocolpath.read_text());case=protocol['case_id'];report={'schema':'lighton4b-gdn-out-logo-heldout-result/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol_sha256':common.sha(protocolpath),'source_engine_commit':protocol['source_engine_commit'],'case_id':case,'actual_content':protocol['actual_content'],'representative_ocr_claim':False,'quality_qualified':False,'candidate_unchanged':True,'other_heldout_cases_executed':[],'arms':{},'archive_audit':{}}
 active={}
 def terminate(signum,frame):
  process=active.get('process')
  if process is not None and process.poll()is None:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 def checkpoint(phase):
  report['phase']=phase;(out/'gdn-out-logo-heldout.json').write_text(json.dumps(report,indent=2)+'\n');print('phase:',phase,flush=True)
 def command(label,args,env=None,timeout=1200,stdoutpath=None,check=True):
  checkpoint(label)
  with(out/(label+'.log')).open('w')as log:
   dest=stdoutpath.open('xb')if stdoutpath else log
   try:
    process=subprocess.Popen(list(map(str,args)),stdout=dest,stderr=log,env=env,start_new_session=True);active['process']=process
    try:code=process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
     raise
    active.pop('process',None)
    if check and code:raise subprocess.CalledProcessError(code,args)
    return code
   finally:
    if stdoutpath:dest.close()
 try:
  report['preflight']={'memory':common.memory(),'load':os.getloadavg(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free,'swap':{k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in ('SwapTotal','SwapFree')}}
  require(report['preflight']['memory']['available_bytes']>=protocol['minimum_available_ram_bytes'] and report['preflight']['scratch_free_bytes']>=protocol['minimum_scratch_free_bytes'] and report['preflight']['output_free_bytes']>=protocol['minimum_output_free_bytes'],'cgroup/RAM/disk guards')
  require(case=='handwritten_letter' and protocol['other_heldout_cases']==[] and protocol['recipe']==pack.RECIPE,'fixed diagnostic scope')
  for path,sha in protocol['engine_file_sha256'].items():require(common.sha(ROOT/path)==sha,'frozen C++/worker engine file changed')
  require(subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==protocol['ggml_commit'],'pinned ggml commit')
  report['frozen_engine_file_hashes_checked']=len(protocol['engine_file_sha256'])
  run=transport.api('actions/runs/'+str(protocol['prior_run']));artifact=transport.api('actions/artifacts/'+str(protocol['prior_artifact']))
  require(run['head_sha']==protocol['source_engine_commit'] and run['conclusion']=='success' and run['run_attempt']==1,'prior producer run identity')
  require(not artifact['expired'] and artifact['workflow_run']['id']==protocol['prior_run'] and artifact['size_in_bytes']==protocol['prior_archive_bytes'] and artifact['digest']=='sha256:'+protocol['prior_archive_sha256'],'prior artifact identity')
  archive=scratch/'calibration.zip';command('download-prior-archive',['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(protocol['prior_artifact'])+'/zip'],stdoutpath=archive,timeout=900)
  require(archive.stat().st_size==protocol['prior_archive_bytes'] and common.sha(archive)==protocol['prior_archive_sha256'],'whole prior archive rehash')
  report['archive_audit']['whole_archive_sha256']=common.sha(archive);candidate=scratch/'candidate.gguf';rawhashes={};rawoutputs={}
  with zipfile.ZipFile(archive)as z:
   names=z.namelist();require(len(names)<=4096 and len(names)==len(set(names)),'ZIP inventory')
   require(z.getinfo(protocol['candidate_member']).file_size==protocol['candidate_bytes'],'complete candidate ZIP member size')
   with z.open(protocol['candidate_member'])as src,candidate.open('xb')as dest:
    count=0;h=hashlib.sha256()
    while b:=src.read(8*1024**2):count+=len(b);require(count<=protocol['candidate_bytes'],'candidate member bound');dest.write(b);h.update(b)
   require(count==protocol['candidate_bytes'] and h.hexdigest()==protocol['candidate_sha256'] and common.sha(candidate)==protocol['candidate_sha256'],'complete candidate rehash')
   prior=json.loads(z.read('gdn-out-full-calibration.json'));audit=json.loads(z.read('candidate-audit.json'));require(prior['source_sha']==protocol['source_engine_commit'] and audit['candidate_sha256']==protocol['candidate_sha256'],'producer receipt hashes')
   # Independently hash every archived F32 array, without materializing arrays on disk.
   for name in names:
    if name.endswith('.f32'):
     size=z.getinfo(name).file_size;require(size<=16*1024**2 and size%4==0,'array size bound')
     with z.open(name)as src:n,sha=streamhash(src,size)
     require(n==size,'complete array bytes');rawhashes[name]=sha
   for oldcase,arms in prior['cases'].items():
    for arm,row in arms.items():
     prefix=f'full/{arm}/{oldcase}/native/';require(rawhashes[prefix+'prefill.logits.f32']==row['first_score_sha256'],'raw original full firstscore inventory')
     for item in row['greedy_scores']:require(rawhashes[prefix+f'greedy.{item["position"]}.logits.f32']==item['sha256'],'raw greedy score inventory')
     idsraw=z.read(prefix+'generated.ids');textraw=z.read(prefix+'generated.txt');ids=[int(x)for x in idsraw.decode().split()];text=textraw.decode();require(ids==row['generated_ids'] and text==row['decoded'],'raw full ID/text inventory');rawoutputs.setdefault(oldcase,{})[arm]={'ids':ids,'text':text,'ids_sha256':hashlib.sha256(idsraw).hexdigest(),'text_sha256':hashlib.sha256(textraw).hexdigest()}
   controls=json.loads((ROOT/'docs/lighton4b-gdn-out-smoke-controls-v1.json').read_text());same_initial=0
   for oldcase,arms in controls['cases'].items():
    for arm,x in arms.items():
     prefix=f'smoke/{arm}/{oldcase}/native/';require(rawhashes[prefix+'prefill.logits.f32']==x['first_score_sha256'],'raw serialized smoke score')
     for i,sha in x['layer_sha256'].items():require(rawhashes[prefix+f'prefill.layer.{i}.f32']==sha,'raw serialized smoke block')
     for name,sha in rawhashes.items():
      if name.startswith(prefix):
       other=name.replace('smoke/','full/',1)
       if other in rawhashes:require(sha==rawhashes[other],'full/smoke initial array mismatch');same_initial+=1
  parsed=pack.parse(candidate);pins=json.loads((ROOT/'tests/fixtures/lighton4b-restoration-donor-freeze-v1.json').read_text());selected={x['name']for x in json.loads((ROOT/'docs/lighton4b-q4k-gdn-out-q8-v1.json').read_text())['selected_tensors']};require(len(parsed['tensors'])==723 and len(selected)==24,'complete persisted candidate census')
  payloads={};types={}
  with candidate.open('rb')as f:
   for name,tensor in parsed['tensors'].items():
    sha=pins['selected_payloads'][name]['q8_sha256'if name in selected else'q4_sha256']if name in pins['selected_payloads']else pins['protected_sha256'][name]
    f.seek(tensor['data_offset']);n,actual=spanhash(f,tensor['bytes'])
    require(n==tensor['bytes'] and actual==sha,'independent actual tensor payload hash');payloads[name]={'sha256':actual,'bytes':n,'type':tensor['type'],'shape':tensor['stored_shape']};types[str(tensor['type'])]=types.get(str(tensor['type']),0)+1
  require(types=={'0':474,'8':25,'12':224} and parsed['metadata']['general.file_type']['value']==14,'exact storage profile; not Q4_K_M')
  require(all(parsed['metadata'][k]['value']==v for k,v in pack.metadata_fields().items()),'exact recipe/source/manifest metadata')
  report['archive_audit'].update(candidate_sha256=common.sha(candidate),candidate_bytes=candidate.stat().st_size,tensor_hashes_checked=723,protected_payloads_checked=699,raw_float_arrays_checked=len(rawhashes),full_smoke_matching_array_pairs=same_initial,greedy_score_inventory_checked=sum(len(x['greedy_scores'])for c in prior['cases'].values()for x in c.values()),candidate_type_census=types,complete_candidate_verified=True)
  (out/'independent-payload-hashes.json').write_text(json.dumps(payloads,indent=2)+'\n');(out/'independent-array-hashes.json').write_text(json.dumps(rawhashes,indent=2)+'\n')
  # This is the verified task-owned transport ZIP; the immutable remote remains preserved.
  archive.unlink()
  buildout=out/'build';buildout.mkdir();buildenv=os.environ.copy();buildenv.update(HEAVY_OUT=str(buildout),HEAVY_SCRATCH=str(scratch));command('model-free-build',[sys.executable,ROOT/'tools/ci-heavy/lighton4b_frozen_case_build.py'],env=buildenv)
  build=scratch/'frozen-case-build';receipt=buildout/'frozen-case-build.json';receiptsha=common.sha(receipt);worker.validate_build(build,receipt,receiptsha,report['source_sha'])
  baseline=scratch/common.CANDIDATES['q4_k'][0]
  try:common.download(f'https://huggingface.co/{common.MODEL_REPO}/resolve/{common.MODEL_REVISION}/{baseline.name}',baseline)
  except Exception:raise ValueError('immutable baseline transfer failed')from None
  require(baseline.stat().st_size==common.CANDIDATES['q4_k'][2] and common.sha(baseline)==protocol['baseline_sha256'],'plain baseline fullfile hash')
  urls=transport.original_urls();prefix='lighton4b-fourcase-bf16/'
  def get(member,path,sha,bound=8*1024**2):transport.bounded_download(urls[prefix+member],path,sha,bound)
  get('batch.json',out/'original-batch.json',transport.ORIGINAL['batch_sha256']);batch=json.loads((out/'original-batch.json').read_text());require(batch['source_sha']==worker.ORIGINAL_SOURCE and batch['model_dtype']=='bfloat16' and batch['protocol_sha256']==worker.PROTOCOL_SHA and batch['input_manifest_sha256']==worker.FROZEN_SHA,'genuine original batch')
  original=scratch/'original';original.mkdir();originals={}
  for name in ('simple_table','arabic_printed_line',case):
   _,contract,_=worker.load_contract(name);folder=original/name;folder.mkdir();get(name+'/capture.json',folder/'capture.json',batch['cases'][name]['capture_sha256']);capture=json.loads((folder/'capture.json').read_text());worker.validate_original(capture,contract)
   for row in capture['prepared_inputs'].values():require(Path(row['file']).name==row['file'],'original input path');get(name+'/'+row['file'],folder/row['file'],row['sha256'])
   row=capture['scores'][0];require(row['shape']==[1,248320] and row['bytes']==993280 and row['position']==0,'source firstscore schema');get(name+'/'+row['file'],folder/row['file'],row['sha256'],993280);originals[name]=capture
  gold=originals[case];require(batch['cases'][case]['capture_sha256']==protocol['original_capture_sha256'] and gold['eos_completed'] and not gold['cap_hit'] and len(gold['generated_ids'])==len(gold['scores'])==65 and gold['generated_ids'][-1]==248046,'heldout complete genuine reference before native output')
  for i,item in enumerate(gold['scores']):require(item['position']==i and item['shape']==[1,248320] and item['bytes']==993280,'complete original score metadata')
  report['original_source_complete_before_native']=True;report['original']={'capture_sha256':protocol['original_capture_sha256'],'generated_tokens':65,'eos_completed':True,'cap_hit':False,'first_score':gold['scores'][0],'prepared_inputs':gold['prepared_inputs']}
  metric_audit={}
  for oldcase,arms in rawoutputs.items():
   metric_audit[oldcase]={}
   for arm,row in arms.items():
    ref=originals[oldcase];ce=edits(ref['decoded'],row['text']);we=edits(ref['decoded'].split(),row['text'].split());expected=prior['cases'][oldcase][arm]['against_genuine_bf16'];require(ce==expected['cer']['edits'] and we==expected['wer']['edits'],'independent raw calibration text metrics');metric_audit[oldcase][arm]={'character_edits':ce,'word_edits':we,'ids_sha256':row['ids_sha256'],'text_sha256':row['text_sha256']}
  report['archive_audit']['raw_calibration_metric_audit']=metric_audit
  env=os.environ.copy();env.update(prior['fixed_precision_gates']);env.update({x:'1'for x in common.GATES});env.update(CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4='0',CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8='0');report['fixed_precision_gates']=prior['fixed_precision_gates']
  command('actual-metadata',[build/'test-lighton4b-gdn-out-recipe','--metadata',candidate],env=env);command('actual-default-off',[build/'test-lighton4b-gdn-out-recipe','--closed-context',candidate],env=env)
  arms={'q4':(baseline,'q4_k',protocol['baseline_sha256']),'out':(candidate,pack.RECIPE,protocol['candidate_sha256'])}
  def execute(phase,arm,name):
   folder=out/phase/arm;folder.mkdir(parents=True,exist_ok=True);work=scratch/(phase+'-'+arm+'-'+name);work.mkdir();child=env.copy();child.update(HEAVY_OUT=str(folder),HEAVY_SCRATCH=str(work));model,recipe,sha=arms[arm]
   args=[sys.executable,ROOT/'tools/ci-heavy/lighton4b_frozen_case.py','--case',name,'--model',model,'--weights-sha256',sha,'--recipe',recipe,'--original-root',original/name,'--original-capture-sha256',batch['cases'][name]['capture_sha256'],'--mode','smoke'if phase=='smoke'else'greedy-full','--threads','4','--prebuilt-build',build,'--build-receipt',receipt,'--build-receipt-sha256',receiptsha]
   try:code=command(phase+'-'+arm+'-'+name,args,env=child,timeout=900 if phase=='smoke'else protocol['heldout_process_bound_seconds'],check=False)
   except subprocess.TimeoutExpired:return {'status':'timed_out'}
   path=folder/name/'native-case.json';require(path.is_file(),'case diagnostic receipt missing');row=json.loads(path.read_text())
   if code or row['status']!='completed':
    if phase=='smoke':raise ValueError('frozen calibration smoke failed')
    row['status']='execution_failed';row['process_returncode']=code
   return row
  report['calibration_smoke']={}
  for name in ('simple_table','arabic_printed_line'):
   report['calibration_smoke'][name]={}
   for arm in arms:
    row=execute('smoke',arm,name);require(row['status']=='completed','smoke timeout');folder=out/'smoke'/arm/name/'native';expected=controls['cases'][name][arm]
    checks={'firstscore_exact':common.sha(folder/'prefill.logits.f32')==expected['first_score_sha256'],'prefix_exact':row['generated_ids']==expected['prefix_ids'],'all32_blocks_exact':all(common.sha(folder/f'prefill.layer.{i}.f32')==expected['layer_sha256'][str(i)]for i in range(32))};require(all(checks.values()),'same pinned calibration controls before heldout');report['calibration_smoke'][name][arm]=checks
  report['all_audits_before_heldout_passed']=True
  for arm in arms:
   value=execute('heldout',arm,case);report['arms'][arm]=value
   if value['status']=='completed':
    folder=out/'heldout'/arm/case/'native';value['against_genuine_bf16']=full.agreement(value['generated_ids'],value['decoded'],gold);value['first_score_sha256']=common.sha(folder/'prefill.logits.f32');value['first_score_vs_genuine_bf16']=comparison.compare(np.fromfile(original/case/gold['scores'][0]['file'],dtype='<f4'),np.fromfile(folder/'prefill.logits.f32',dtype='<f4'));value['greedy_scores']=[{'position':i,'bytes':(folder/f'greedy.{i}.logits.f32').stat().st_size,'sha256':common.sha(folder/f'greedy.{i}.logits.f32')}for i in range(len(value['generated_ids']))]
   checkpoint('completed-'+arm)
  report['status']='completed';report['all_native_completed']=all(x['status']=='completed'for x in report['arms'].values())
 finally:(out/'gdn-out-logo-heldout.json').write_text(json.dumps(report,indent=2)+'\n');(out/'summary.md').write_text('Heldout A/logo diagnostic: '+report['status']+'. Not representative paragraph OCR; no broad qualification.\n')
if __name__=='__main__':main()
