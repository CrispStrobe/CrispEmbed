#!/usr/bin/env python3
"""One fixed exploratory recipe, actual serialized smoke controls then paired full greedy calibration."""
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def require(x,message):
 if not x:raise ValueError(message)
def levenshtein(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]
def agreement(ids,text,original):
 teacher=original['generated_ids'];first=next((i for i,(a,b)in enumerate(zip(ids,teacher))if a!=b),min(len(ids),len(teacher)))
 cer=levenshtein(list(original['decoded']),list(text));wer=levenshtein(original['decoded'].split(),text.split())
 return {'ids_exact':ids==teacher,'decoded_exact':text==original['decoded'],'first_differing_id':None if ids==teacher else first,'matching_prefix_ids':first,'cer':{'edits':cer,'reference_codepoints':len(original['decoded']),'rate':cer/max(1,len(original['decoded']))},'wer':{'edits':wer,'reference_words':len(original['decoded'].split()),'rate':wer/max(1,len(original['decoded'].split()))},'normalization':'none; Unicodecodepoints and whitespace words with case/punctuation retained; modelagreement, not human OCR accuracy'}
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'gdn-out-full';scratch.mkdir()
 common=load('native',ROOT/'tools/ci-heavy/lighton4b_native_document.py');transport=load('transport',ROOT/'tools/ci-heavy/lighton4b_calibration_native.py');worker=load('worker',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py');vision=load('vision',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py');comparison=load('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py')
 sys.path.insert(0,str(ROOT/'tools'));import lighton4b_gdn_out_pack as pack
 report={'schema':'lighton4b-gdn-out-full-calibration/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'recipe':pack.RECIPE,'calibration_cases':['simple_table','arabic_printed_line'],'heldout_cases_executed':[],'quality_qualified':False,'reference_forward_executed':False,'cases':{},'limits':['Calibration-selected policy; heldout remains untouched.','Independent greedy cap256 and one first-score row per case/arm, not full teacher score replay.','References are frozen genuine BF16 original outputs; no same-weight candidate source oracle runs here.','Experimental named recipe, not Q4_K_M or a public model alias.']}
 try:
  import numpy as np
  from PIL import Image
  require(np.__version__=='2.3.5' and Image.__version__=='11.3.0','serializer/RGB pins')
  report['preflight']={'memory':common.memory(),'load':os.getloadavg(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free,'swap':{k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in ('SwapTotal','SwapFree')}}
  require(report['preflight']['memory']['available_bytes']>=8*1024**3 and report['preflight']['scratch_free_bytes']>=20*1024**3 and report['preflight']['output_free_bytes']>=8*1024**3,'cgroup/RAM/disk guard')
  protocol=json.loads((ROOT/'docs/lighton4b-q4k-gdn-out-q8-v1.json').read_text());report['protocol_sha256']=common.sha(ROOT/'docs/lighton4b-q4k-gdn-out-q8-v1.json');require(protocol['recipe']==pack.RECIPE and protocol['calibration_cases']==report['calibration_cases'] and protocol['heldout_cases']==[],'fixed recipe/scope protocol')
  controls=json.loads((ROOT/'docs/lighton4b-gdn-out-smoke-controls-v1.json').read_text());report['smoke_controls_sha256']=common.sha(ROOT/'docs/lighton4b-gdn-out-smoke-controls-v1.json')
  buildout=out/'build';buildout.mkdir();buildenv=os.environ.copy();buildenv.update(HEAVY_OUT=str(buildout),HEAVY_SCRATCH=str(scratch))
  active={}
  def terminate(signum,frame):
   process=active.get('process')
   if process is not None and process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   raise SystemExit(128+signum)
  signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
  def run(label,args,env=None,timeout=1200,check=True):
   report['phase']=label;(out/'gdn-out-full-calibration.json').write_text(json.dumps(report,indent=2)+'\n');print('phase:',label,flush=True)
   with(out/(label+'.log')).open('w')as log:
    process=subprocess.Popen(list(map(str,args)),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active['process']=process
    try:code=process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
     raise
    active.pop('process',None)
    if check and code:raise subprocess.CalledProcessError(code,args)
    return process
  run('model-free-build',[sys.executable,ROOT/'tools/ci-heavy/lighton4b_frozen_case_build.py'],env=buildenv)
  build=scratch/'frozen-case-build';receipt=buildout/'frozen-case-build.json';receiptsha=common.sha(receipt);worker.validate_build(build,receipt,receiptsha,report['source_sha'])
  require('GDN_OUT_RECIPE passed=1' in (buildout/'gdn-out-recipe-controls.log').read_text(),'closed recipe/gate/metadata controls')
  models={}
  for recipe in ('q4_k','q8_0'):
   name,expected,size=common.CANDIDATES[recipe];path=scratch/name
   try:common.download(f'https://huggingface.co/{common.MODEL_REPO}/resolve/{common.MODEL_REVISION}/{name}',path)
   except Exception:raise ValueError('immutable model transfer failed')from None
   require(path.stat().st_size==size and common.sha(path)==expected,'immutable source file hash');models[recipe]=path
  q4,q8=pack.parse(models['q4_k']),pack.parse(models['q8_0']);pinpath=ROOT/'tests/fixtures/lighton4b-restoration-donor-freeze-v1.json';require(common.sha(pinpath)=='bc52de8180ef3b71d1d572add1b6e0998c8013697657388e44d16e599ad7aceb','fixed donor freeze')
  pins=json.loads(pinpath.read_text());expected={}
  for recipe,rows in [('q4_k',q4),('q8_0',q8)]:
   expected[recipe]={}
   for name,row in rows['tensors'].items():
    sha=pins['selected_payloads'][name]['q4_sha256'if recipe=='q4_k'else'q8_sha256']if name in pins['selected_payloads']else pins['protected_sha256'][name]
    expected[recipe][name]={'sha256':sha,'bytes':row['bytes'],'shape':row['stored_shape']}
  modeldir=out/'candidate';modeldir.mkdir();candidate=modeldir/(pack.RECIPE+'.gguf');audit=pack.compose(models['q4_k'],models['q8_0'],candidate,expected)
  audit.update(source_sha=report['source_sha'],base_file_sha256=pack.BASE_SHA,donor_file_sha256=pack.DONOR_SHA,candidate_sha256=common.sha(candidate),protocol_sha256=report['protocol_sha256'],manifest_sha256=pack.MANIFEST_SHA,metadata=dict(pack.metadata_fields()),protected_payloads_preserved=699,selected_whole_tensor_count=24,quality_qualified=False)
  (out/'candidate-audit.json').write_text(json.dumps(audit,indent=2)+'\n');report['candidate']={k:v for k,v in audit.items()if k!='records'}
  run('actual-candidate-metadata',[build/'test-lighton4b-gdn-out-recipe','--metadata',candidate])
  precision={k:'1'for k in vision.VISION_GATES};precision.update({k:'0'for k in ('CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2','CRISPEMBED_LIGHTON4B_DIRECT_QUANT','CRISPEMBED_QWEN35_WEIGHT_CACHE','CRISPEMBED_QWEN35_PRECISE_RMS','CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_LIGHTON4B_GREEDY_TRACE')});precision.update(CRISPEMBED_LIGHTON4B_HEAD_BF16='1',CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1');report['fixed_precision_gates']=precision
  env=os.environ.copy();env.update(precision);env.update({k:'1'for k in common.GATES});env.update(CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4='0',CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8='0')
  run('actual-default-off-context',[build/'test-lighton4b-gdn-out-recipe','--closed-context',candidate],env=env)
  both=env.copy();both.update(CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4='1',CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8='1');run('actual-conflicting-gates',[build/'test-lighton4b-gdn-out-recipe','--closed-context',candidate],env=both)
  wrong=env.copy();wrong['CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8']='1';run('actual-wrong-base-tag',[build/'test-lighton4b-gdn-out-recipe','--closed-context',models['q4_k']],env=wrong)
  urls=transport.original_urls();prefix='lighton4b-fourcase-bf16/'
  def get(member,path,sha,bound=8*1024**2):transport.bounded_download(urls[prefix+member],path,sha,bound)
  get('batch.json',out/'original-batch.json',transport.ORIGINAL['batch_sha256']);batch=json.loads((out/'original-batch.json').read_text());require(batch['source_sha']==worker.ORIGINAL_SOURCE and batch['protocol_sha256']==worker.PROTOCOL_SHA and batch['input_manifest_sha256']==worker.FROZEN_SHA and batch['model_dtype']=='bfloat16','genuine BF16 batch source')
  original=scratch/'original';original.mkdir();originals={}
  for case in report['calibration_cases']:
   _,contract,_=worker.load_contract(case);folder=original/case;folder.mkdir();get(case+'/capture.json',folder/'capture.json',batch['cases'][case]['capture_sha256']);capture=json.loads((folder/'capture.json').read_text());worker.validate_original(capture,contract);require(capture['eos_completed'] and not capture['cap_hit'],'calibration genuine BF16 complete output')
   for row in capture['prepared_inputs'].values():require(Path(row['file']).name==row['file'],'source input path');get(case+'/'+row['file'],folder/row['file'],row['sha256'])
   row=capture['scores'][0];get(case+'/'+row['file'],folder/row['file'],row['sha256'],993280);originals[case]=capture
  arms={'q4':(models['q4_k'],'q4_k',pack.BASE_SHA),'out':(candidate,pack.RECIPE,audit['candidate_sha256'])}
  def execute(phase,arm,case):
   folder=out/phase/arm;folder.mkdir(parents=True,exist_ok=True);work=scratch/(phase+'-'+arm+'-'+case);work.mkdir();child=env.copy();child.update(HEAVY_OUT=str(folder),HEAVY_SCRATCH=str(work));model,recipe,sha=arms[arm]
   argv=[sys.executable,ROOT/'tools/ci-heavy/lighton4b_frozen_case.py','--case',case,'--model',model,'--weights-sha256',sha,'--recipe',recipe,'--original-root',original/case,'--original-capture-sha256',batch['cases'][case]['capture_sha256'],'--mode','smoke'if phase=='smoke'else'greedy-full','--threads','4','--prebuilt-build',build,'--build-receipt',receipt,'--build-receipt-sha256',receiptsha]
   timeout=15*60 if phase=='smoke'else (90 if case=='simple_table' else 20)*60
   label=phase+'-'+arm+'-'+case
   try:result=run(label,argv,env=child,timeout=timeout,check=False)
   except subprocess.TimeoutExpired:return {'status':'timed_out','process_bound_seconds':timeout}
   path=folder/case/'native-case.json';require(path.is_file(),'case diagnostic receipt missing');resultrow=json.loads(path.read_text())
   if result.returncode!=0 or resultrow['status']!='completed':
    if phase=='smoke':raise ValueError('serialized smoke execution failed')
    resultrow['status']='execution_failed';resultrow['process_returncode']=result.returncode
   return resultrow
  report['smoke']={}
  # All four serialized-file smoke controls must pass before any new full output.
  for case in report['calibration_cases']:
   report['smoke'][case]={}
   for arm in arms:
    value=execute('smoke',arm,case);require(value['status']=='completed','smoke timeout');folder=out/'smoke'/arm/case/'native';expectedcontrol=controls['cases'][case][arm]
    checks={'firstscore_exact':common.sha(folder/'prefill.logits.f32')==expectedcontrol['first_score_sha256'],'prefix_exact':value['generated_ids']==expectedcontrol['prefix_ids'],'all32_blocks_exact':all(common.sha(folder/f'prefill.layer.{i}.f32')==expectedcontrol['layer_sha256'][str(i)]for i in range(32))}
    require(all(checks.values()),'serialized candidate/parallel conversion differs from measured projection arm');report['smoke'][case][arm]=checks
  report['serialized_smoke_controls_passed']=True
  for case in report['calibration_cases']:
   report['cases'][case]={}
   for arm in arms:
    value=execute('full',arm,case);report['cases'][case][arm]=value
    if value['status']=='completed':
     value['against_genuine_bf16']=agreement(value['generated_ids'],value['decoded'],originals[case]);folder=out/'full'/arm/case/'native';row=originals[case]['scores'][0];value['first_score_vs_genuine_bf16']=comparison.compare(np.fromfile(original/case/row['file'],dtype='<f4'),np.fromfile(folder/'prefill.logits.f32',dtype='<f4'));value['first_score_sha256']=common.sha(folder/'prefill.logits.f32');value['greedy_scores']=[{'position':i,'bytes':(folder/f'greedy.{i}.logits.f32').stat().st_size,'sha256':common.sha(folder/f'greedy.{i}.logits.f32')}for i in range(len(value['generated_ids']))]
    (out/'gdn-out-full-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
  report['status']='completed';report['all_native_executions_completed']=all(x['status']=='completed'for c in report['cases'].values()for x in c.values());report['all_full_outputs_exact']=report['all_native_executions_completed'] and all(x['against_genuine_bf16']['ids_exact'] and x['against_genuine_bf16']['decoded_exact']for c in report['cases'].values()for x in c.values())
 except Exception as e:report['failure_type']=type(e).__name__;raise
 finally:(out/'gdn-out-full-calibration.json').write_text(json.dumps(report,indent=2)+'\n');(out/'summary.md').write_text('Exploratory24GDNoutputQ8 full calibration: '+report['status']+'. Heldout untouched; no model/default qualification.\n')
if __name__=='__main__':main()
