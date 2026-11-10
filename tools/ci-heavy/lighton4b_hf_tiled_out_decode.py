#!/usr/bin/env python3
"""One closed HF-layout candidate: immutable packing and three observed full decodes."""
import hashlib,io,json,os,re,resource,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_hf_tiled_out_pack as packer
from gguf_file_type_canonicalize import sha
import importlib.util

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def require(ok,why):
 if not ok:raise ValueError(why)

def small_json(url,digest):
 with urllib.request.urlopen(url,timeout=60)as f:raw=f.read(2**20+1)
 require(len(raw)<=2**20 and hashlib.sha256(raw).hexdigest()==digest,'immutable small public receipt differs');return json.loads(raw)

def main():
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);build=Path(os.environ['HEAVY_BUILD']);r={'schema':'lighton4b-hf-tiled-out-observed-threecase/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'heldout_read':False,'quality_qualified':False,'shipping_approved':False,'original_forward_executed':False,'native_model_forward_attempted':False,'native_model_forward_executed':False,'full_f32_model_instantiated':False,'cases':{}};active=None;started=time.monotonic()
 def save():(out/'hf-tiled-out-complete-decode.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 def stop(*args):
  if active is not None and active.poll()is None:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
  r['partial_outputs_preserved']=True;save();raise TimeoutError('bounded threecase worker deadline')
 def run(label,args,env,timeout,progress=None,expected_code=0):
  nonlocal active
  before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();r['stage']=label;save();print('PHASE '+label,flush=True);code=None;seen=None
  try:
   with(out/(label+'.log')).open('w')as log:
    active=subprocess.Popen(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while True:
     remaining=timeout-(time.monotonic()-start)
     if remaining<=0:stop()
     try:code=active.wait(timeout=min(30,remaining));break
     except subprocess.TimeoutExpired:
      if progress is not None and progress.exists():
       try:
        p=json.loads(progress.read_text());n=p['generated_tokens']
        if isinstance(n,int)and 0<=n<=256 and n!=seen:print('PROGRESS '+label+' tokens='+str(n)+' elapsed='+str(round(time.monotonic()-start,1)),flush=True);seen=n
       except (ValueError,KeyError):pass
   require(code==expected_code,'bounded child failed: '+label)
  finally:
   active=None;after=resource.getrusage(resource.RUSAGE_CHILDREN);r.setdefault('child_resource',{})[label]={'seconds':time.monotonic()-start,'returncode':code,'user_seconds':after.ru_utime-before.ru_utime,'system_seconds':after.ru_stime-before.ru_stime,'cumulative_completed_children_peak_rss_bytes':int(after.ru_maxrss)*1024,'peak_scope':'cumulative completed children, not isolated per child'};save()
 try:
  require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only');signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(240*60)
  import numpy as np
  import PIL
  from PIL import Image
  require(np.__version__=='2.3.5'and PIL.__version__=='11.3.0','frozen image/array serializer versions required')
  common=load('tiled_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py');cases=load('tiled_cases',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py');helpers=load('tiled_metrics',ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_full.py');old=load('tiled_old_decode',ROOT/'tools/ci-heavy/lighton4b_imatrix_q4km_decode.py');producer_tools=load('tiled_producer',ROOT/'tools/ci-heavy/lighton4b_imatrix_q4km_candidate.py')
  protocol_path=ROOT/'docs/lighton4b-hf-tiled-out-observed-threecase-v1.json';protocol=json.loads(protocol_path.read_text());require(protocol['base_sha256']==packer.BASE_SHA and protocol['source_f16_proof_receipt_sha256']==packer.PAIRED_SHA and protocol['cases_order']==['simple_form','simple_table','arabic_printed_line']and protocol['worker_limits']['controller_seconds']==14400,'closed threecase protocol differs');r['protocol_sha256']=sha(protocol_path)
  r['cpu_profile']=old.cpu_profile();require(r['cpu_profile']['identity_fields_available'],'actual CPU identity required');memory=common.memory();require(memory['available_bytes']>=8*2**30,'8GiB available RAM required');r['memory_preflight']=memory
  paths=list(Path('/kaggle/input').rglob('q4km-candidate-first-score.json'));require(len(paths)==1,'single frozen old producer required');producer_path=paths[0];producer=old.admit_producer(producer_path,os.environ['LIGHTON4B_PRODUCER_RECEIPT_SHA256'],os.environ['LIGHTON4B_PRODUCER_SOURCE_SHA']);origin=producer_path.parent;base=origin/'lighton4b-weighted-q4km-style-q6head-v1.gguf';require(base.stat().st_size==packer.BASE_BYTES and sha(base)==packer.BASE_SHA,'immutable5d mounted baseline required');r['base_sha256']=packer.BASE_SHA
  paired=[p for p in Path('/kaggle/input').rglob('paired24-receipt.json')if p.stat().st_size<2**20 and sha(p)==packer.PAIRED_SHA];require(len(paired)==1,'single authenticated source-F16 proof required');paired=paired[0];require(json.loads(paired.read_text())['candidate_conversion_allowed'],'full original-F16 proof required')
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(producer['fixed_gates']);env.update(CRISPEMBED_LIGHTON4B_Q6_K='1',CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD='1',CRISPEMBED_LIGHTON4B_HF_TILED_OUT='0',CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL='0',CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS='0',CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE='0',LD_LIBRARY_PATH=os.environ['HEAVY_RUNTIME_LIB'],LC_ALL='C')
  for k,v in [('HEAD_BF16','1'),('BF16_QUANT_WEIGHTS','0'),('BF16_LINEAR','0'),('PARALLEL_DEQUANT','1')]:require(env['CRISPEMBED_LIGHTON4B_'+k]==v,'fixed precision changed')
  # Admit this exact new binary/host on the unchanged old model before packing.
  _,_,frozen=cases.load_contract('simple_table');png=ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file'];require(sha(png)==frozen['png_sha256'],'platform image differs');im=Image.open(png).convert('RGB');rgb=scratch/'platform.rgb';rgb.write_bytes(im.tobytes());require(hashlib.sha256(rgb.read_bytes()).hexdigest()==frozen['rgb_sha256'],'platform pixels differ');teacher=scratch/'platform.teacher.ids';teacher.write_text(str(producer['cases']['simple_table']['weighted_q4km']['generated_ids'][0])+'\n');platform=out/'old-model-platform';platform.mkdir()
  r['native_model_forward_attempted']=True;r['native_model_forward_executed']=None;save()
  run('old-model-platform',[build/'lighton4b-native-document',base,rgb,im.height,im.width,teacher,platform,1,4,'smoke'],env,10*60)
  expected=producer['cases']['simple_table']['weighted_q4km'];require(sha(platform/'prefill.logits.f32')==expected['first_score_sha256']==sha(platform/'teacher.0.logits.f32')==sha(platform/'greedy.0.logits.f32'),'old model whole-vocabulary platform anchor failed');stable={k:v['sha256']for k,v in expected['artifacts'].items()if k.startswith(('prefill.','vision.','processor.','text.'))or k in('prompt.ids','modalities.ids','rendered.txt')};require(all(sha(platform/k)==v for k,v in stable.items()),'old model actual stage anchor failed');r['native_model_forward_executed']=True;r['old_model_platform']={'first_score_sha256':expected['first_score_sha256'],'all_stage_hashes':stable};save()
  candidate=out/(packer.RECIPE+'.gguf');run('pack-original-f16-proven24',[sys.executable,ROOT/'tools/lighton4b_hf_tiled_out_pack.py','--base',base,'--paired-receipt',paired,'--tensors',paired.parent/'tensors','--output',candidate,'--receipt',out/'candidate-pack.json'],env,10*60);audit=json.loads((out/'candidate-pack.json').read_text());require(audit['status']=='packed'and audit['all699_protected_payloads_exact']and audit['all24_publisher_tiled_payloads_exact']and sha(candidate)==audit['candidate_sha256'],'full independently rehashed candidate required');r['candidate']={k:v for k,v in audit.items()if k!='tensors'}
  env.update(CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD='0',CRISPEMBED_LIGHTON4B_HF_TILED_OUT='1');require(all(env.get('CRISPEMBED_LIGHTON4B_'+k)==v for k,v in protocol['fixed_precision'].items()),'predeclared candidate precision gates differ');r['fixed_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')};save()
  # Exercise the actual preflight entry on a valid metadata-only header and a
  # single wrong proof field. Missing RGB must be reached only by valid metadata.
  header=packer.parse(candidate,packer.TYPES)
  with candidate.open('rb')as f:header_raw=f.read(header['data_offset'])
  valid_header=out/'valid-header-only.gguf';valid_header.write_bytes(header_raw);bad=bytearray(header_raw);position=header['metadata']['qwen35_ocr.hf_tiled_out.paired_receipt_sha256']['offset']+8;bad[position]=ord('0')if bad[position]!=ord('0')else ord('1');bad_header=out/'wrong-proof-header-only.gguf';bad_header.write_bytes(bad);missing=scratch/'missing.rgb'
  run('valid-preflight-reaches-input',[build/'lighton4b-native-case-preflight',valid_header,missing,1,1,out,packer.RECIPE],env,60,expected_code=5)
  run('wrong-proof-preflight-rejects',[build/'lighton4b-native-case-preflight',bad_header,missing,1,1,out,packer.RECIPE],env,60,expected_code=3)
  off=dict(env,CRISPEMBED_LIGHTON4B_HF_TILED_OUT='0');run('gate-off-preflight-rejects',[build/'lighton4b-native-case-preflight',valid_header,missing,1,1,out,packer.RECIPE],off,60,expected_code=3)
  r['actual_preflight_metadata_and_gate_controls_passed']=True;save()

  floor=json.loads((ROOT/'docs/lighton4b-imatrix-q4km-quality-floor-v1.json').read_text());form_floor=json.loads((ROOT/'docs/lighton4b-imatrix-q4km-form-floor-v1.json').read_text());require(set(floor['cases'])=={'simple_table','arabic_printed_line'}and form_floor['post_selection_diagnostic']and not form_floor['heldout_claim'],'predeclared observed scope required');floor['cases']['simple_form']=form_floor['case'];r['floor_sha256']={'calibration':sha(ROOT/'docs/lighton4b-imatrix-q4km-quality-floor-v1.json'),'form':sha(ROOT/'docs/lighton4b-imatrix-q4km-form-floor-v1.json')}
  oldroot='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/3f4b02d2c7b7531a815816512f61bf1fa4f93732/tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/'
  previous=small_json(oldroot+'full-calibration-v2/lighton-q4km-decode-results/q4km-complete-decode.json','1f296b54b74e2660f81f7fa7d9aae9850bb3d22fb19c5024c7a28908248090e7');previous_form=small_json(oldroot+'form-postselection-v1/lighton-q4km-form-results/q4km-complete-decode.json','a483fe4d56cfdde2c3835d2cf7ca1f322d09c4594664405e0faa8da073230643');require(previous['status']==previous_form['status']=='measured','frozen prior complete receipts required');old_cases={**previous['cases'],**previous_form['cases']}
  specs=[('simple_form',form_floor['original_capture_sha256']),('simple_table','6be5428f247b616e76d79513b62873cf0238842999be2fb0848997f8ced25763'),('arabic_printed_line','ff1a390370b702fabf8b5a25f0feec76888bb1ad8197cf08c3efd44adec1de84')]
  for case,digest in specs:
   is_form=case=='simple_form'
   if is_form:
    capture=small_json(form_floor['original_capture_url'],digest);require(capture['requested_dtype']=='bfloat16'and capture['parameter_dtype_counts']=={'torch.bfloat16':723}and capture['source']['revision']=='b71010f095dc3735de7ee6b9969bff8f8f50b39a'and capture['eos_completed']and len(capture['generated_ids'])==177,'exact genuine BF16 form required');capture_path=None;png=ROOT/'tests/regression/images/cc0/simple_form.png';require(sha(png)=='3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc','form pixels changed')
   else:
    capture_path=producer_tools.unique_capture(Path('/kaggle/input'),case,digest);capture=json.loads(capture_path.read_text());_,contract,frozen=cases.load_contract(case);cases.validate_original(capture,contract);require(contract['split']=='calibration'and capture['eos_completed']and not capture['cap_hit'],'complete BF16 calibration required');png=ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file'];require(sha(png)==frozen['png_sha256'],'calibration image changed')
   image=Image.open(png).convert('RGB');rgb=scratch/(case+'.rgb');rgb.write_bytes(image.tobytes());teacher=scratch/(case+'.teacher.ids');teacher.write_text(str(capture['generated_ids'][0])+'\n');folder=out/case;folder.mkdir();pre=folder/'preflight';pre.mkdir();native=folder/'native';native.mkdir()
   run(case+'-preflight',[build/'lighton4b-native-case-preflight',candidate,rgb,image.height,image.width,pre,packer.RECIPE],env,10*60)
   arrays={}
   for key,row in capture['prepared_inputs'].items():
    if is_form:
     if key=='pixel_values':value=np.fromfile(pre/'processor.patches.f32',dtype='<f4').reshape(row['shape'])
     elif key=='attention_mask':value=np.ones(row['shape'],dtype='<i8')
     else:value=np.array([int(x)for x in(pre/({'image_grid_thw':'grid','input_ids':'prompt','mm_token_type_ids':'modalities'}[key]+'.ids')).read_text().split()],dtype='<i8').reshape(row['shape'])
     stream=io.BytesIO();np.save(stream,value,allow_pickle=False);require(hashlib.sha256(stream.getvalue()).hexdigest()==row['sha256'],'form prepared source bytes changed: '+key)
    else:require(Path(row['file']).name==row['file']and sha(capture_path.parent/row['file'])==row['sha256'],'prepared source bytes changed');value=np.load(capture_path.parent/row['file'],allow_pickle=False)
    arrays[key]=value
   require(set(arrays)=={'pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'}and(pre/'processor.patches.f32').read_bytes()==arrays['pixel_values'].tobytes()and np.all(arrays['attention_mask']==1),'same actual source processor required')
   for key,name in [('image_grid_thw','grid'),('input_ids','prompt'),('mm_token_type_ids','modalities')]:require([int(x)for x in(pre/(name+'.ids')).read_text().split()]==arrays[key].reshape(-1).tolist(),'native/source context differs')
   require(common.memory()['available_bytes']>=8*2**30,'native child RAM unavailable');cap=256;run(case+'-full-greedy',[build/'lighton4b-native-document',candidate,rgb,image.height,image.width,teacher,native,cap,4,'smoke'],env,(20 if case=='arabic_printed_line'else 90)*60,native/'generated-progress.json')
   summary=json.loads((native/'native-summary.json').read_text());ids=[int(x)for x in(native/'generated.ids').read_text().split()];raw=(native/'generated.txt').read_bytes();valid=True
   try:text=raw.decode('utf-8')
   except UnicodeDecodeError:valid=False;text=raw.decode('utf-8',errors='replace')
   require(1<=len(ids)<=256 and all(0<=x<248320 for x in ids),'complete independent ID domain required');eos=ids[-1]==248046;require(summary['generated_tokens']==len(ids)and summary['eos']==eos and summary['cap']==256 and(eos or len(ids)==256),'independent EOS/cap mismatch');require(sha(native/'prefill.logits.f32')==sha(native/'teacher.0.logits.f32')==sha(native/'greedy.0.logits.f32'),'source causal row0 mismatch');require(len(list(native.glob('greedy.*.logits.f32')))==len(ids),'score-row count mismatch');rows=[]
   for index,token in enumerate(ids):
    path=native/('greedy.'+str(index)+'.logits.f32');require(path.stat().st_size==993280,'full vocabulary row required');v=np.fromfile(path,dtype='<f4');require(np.isfinite(v).all()and int(v.argmax())==token,'independent actual row argmax mismatch');rows.append({'position':index,'id':token,'sha256':sha(path)})
   score_metrics={'available':False,'reason':'Form BF16 vocabulary rows are not mounted; decoded receipt is authenticated.'}
   if not is_form:
    score=capture['scores'][0];require(Path(score['file']).name==score['file']and sha(capture_path.parent/score['file'])==score['sha256'],'source BF16 row changed');gold=np.fromfile(capture_path.parent/score['file'],dtype='<f4').astype(np.float64);current=np.fromfile(native/'prefill.logits.f32',dtype='<f4').astype(np.float64);require(gold.size==current.size==248320 and np.isfinite(gold).all(),'finite complete source BF16 row required');error=current-gold;prob=np.exp(current-current.max());prob/=prob.sum();score_metrics={'available':True,'source_sha256':score['sha256'],'source_max_abs_magnitude':float(np.abs(gold).max()),'native_max_abs_magnitude':float(np.abs(current).max()),'rms':float(np.sqrt(np.mean(error*error))),'max_abs':float(np.abs(error).max()),'cosine':float(current@gold/(np.linalg.norm(current)*np.linalg.norm(gold))),'probability_of_original_id':float(prob[capture['generated_ids'][0]]),'scope':'same causal input, quantized candidate versus original unquantized BF16; not same-weight runtime parity'}
   before=old_cases[case];agreement=helpers.agreement(ids,text,capture);limit=floor['cases'][case];require(agreement['cer']['reference_codepoints']==limit['reference_codepoints']and agreement['wer']['reference_words']==limit['reference_words'],'BF16 denominator differs');met=valid and agreement['cer']['edits']<=limit['character_edits']and agreement['wer']['edits']<=limit['word_edits']and eos
   r['cases'][case]={'ids':ids,'decoded':text,'utf8_valid':valid,'actual_eos':eos,'cap_hit':not eos,'cap':256,'source_capture_sha256':digest,'first_score_sha256':sha(native/'prefill.logits.f32'),'greedy_rows':rows,'against_genuine_bf16':agreement,'first_score_against_original_bf16':score_metrics,'against_pinned_llama_q4km':helpers.agreement(ids,text,{'generated_ids':limit['ids'],'decoded':limit['decoded']}),'against_old_hf_candidate':helpers.agreement(ids,text,{'generated_ids':before['generated_ids'],'decoded':before['decoded']}),'old_hf_candidate_against_bf16':before['against_genuine_bf16'],'observed_floor_met':met,'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in native.iterdir()if p.is_file()}};save();print('MEASURED '+case+' ids='+str(len(ids))+' eos='+str(eos)+' cer_edits='+str(agreement['cer']['edits'])+' wer_edits='+str(agreement['wer']['edits']),flush=True)
  require(sha(candidate)==audit['candidate_sha256'],'candidate changed during execution');r['all_three_observed_floors_met']=all(v['observed_floor_met']for v in r['cases'].values());r['status']='measured'
 finally:signal.alarm(0);r['seconds']=time.monotonic()-started;save()
if __name__=='__main__':main()
