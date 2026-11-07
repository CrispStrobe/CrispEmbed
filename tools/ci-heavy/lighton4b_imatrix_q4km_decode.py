#!/usr/bin/env python3
"""Complete calibration decode using the exact producer model and CPU runtime bundle."""
import io,resource,urllib.request,hashlib,importlib.metadata,importlib.util,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_imatrix_q4km_recipe as profile
from gguf_file_type_canonicalize import sha

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def require(ok,message):
 if not ok:raise ValueError(message)

def admit_producer(path,receipt_sha,source_sha):
 require(re.fullmatch('[0-9a-f]{64}',receipt_sha)is not None and re.fullmatch('[0-9a-f]{40}',source_sha)is not None,'sealed producer pins required')
 require(sha(path)==receipt_sha,'immutable producer receipt required')
 producer=json.loads(path.read_text())
 require(producer['source_sha']==source_sha and producer['status']=='measured'and producer['actual248_applied']and producer['both_frozen_null_scores_exact']and producer['post_merge_model_free_controls_passed'],'producer controls not admitted')
 require(producer['candidate']['types']=={'0':474,'12':216,'14':33}and producer['candidate']['all474_f32_exact']and producer['candidate']['actual_imatrix_applied_entries']==248,'candidate schedule/protected payload controls differ')
 require(set(producer['cases'])=={'simple_table','arabic_printed_line'},'calibration cases differ')
 return producer

def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only')
 form_only=os.environ.get('LIGHTON4B_Q4KM_FORM_ONLY')=='1'
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'q4km-complete-decode';scratch.mkdir();r={'schema':'lighton4b-imatrix-q4km-form-postselection/v1'if form_only else'lighton4b-imatrix-q4km-complete-calibration/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'quality_qualified':False,'current_q4_shipping_approved':False,'heldout_read':False,'model_forward_executed':False,'f32_model_instantiated':False,'cases':{},'post_selection_form_diagnostic':form_only};active=None;started=time.monotonic()
 def save():(out/'q4km-complete-decode.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 def stop(*args):
  r['deadline_kind']='global105minutes'if args and args[0]==signal.SIGALRM else'bounded_child_or_signal';r['partial_outputs_preserved']=True;save()
  if active is not None and active.poll()is None:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
  raise TimeoutError('bounded complete-decode deadline')
 def run(label,args,env,timeout):
  nonlocal active
  before_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
  r['stage']=label;print('PHASE '+label,flush=True);save();start=time.monotonic()
  with(out/(label+'.log')).open('w')as log:
   active=subprocess.Popen(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=active.wait(timeout=timeout)
   except subprocess.TimeoutExpired:stop()
   active=None
  after_usage=resource.getrusage(resource.RUSAGE_CHILDREN);r.setdefault('child_rusage',{})[label]={'user_seconds':after_usage.ru_utime-before_usage.ru_utime,'system_seconds':after_usage.ru_stime-before_usage.ru_stime,'cumulative_children_peak_rss_bytes':int(after_usage.ru_maxrss)*1024,'peak_scope':'cumulative completed children; not isolated per-child peak'};r.setdefault('child_seconds',{})[label]=time.monotonic()-start;save();require(code==0,'native child failed: '+label)
 try:
  import numpy as np
  import PIL
  from PIL import Image
  require(np.__version__=='2.3.5'and PIL.__version__=='11.3.0','frozen serializer versions differ');common=load('complete_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py');cases=load('complete_cases',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py');helpers=load('complete_helpers',ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_full.py');producer_helpers=load('complete_producer_helpers',ROOT/'tools/ci-heavy/lighton4b_imatrix_q4km_candidate.py')
  mem=common.memory();require(mem['available_bytes']>=8*2**30 and shutil.disk_usage(scratch).free>=8*2**30,'complete-decode RAM/disk unavailable');r['preflight']={'memory':mem,'free_disk_bytes':shutil.disk_usage(scratch).free,'load':os.getloadavg(),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip()if Path('/sys/fs/cgroup/cpu.max').exists()else None};signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(105*60)
  receipt_sha=os.environ.get('LIGHTON4B_PRODUCER_RECEIPT_SHA256','');source_sha=os.environ.get('LIGHTON4B_PRODUCER_SOURCE_SHA','');require(re.fullmatch('[0-9a-f]{64}',receipt_sha)is not None and re.fullmatch('[0-9a-f]{40}',source_sha)is not None,'sealed producer pins required')
  matches=list(Path('/kaggle/input').rglob('q4km-candidate-first-score.json'));require(len(matches)==1,'unique immutable producer receipt required');receipt_path=matches[0];producer=admit_producer(receipt_path,receipt_sha,source_sha);r['producer']={'source_sha':source_sha,'receipt_sha256':receipt_sha,'model':producer['candidate']}
  origin=receipt_path.parent;manifest_path=origin/'runtime-bundle.json';require(sha(manifest_path)==producer['runtime_bundle_manifest_sha256'],'runtime manifest changed');manifest=json.loads(manifest_path.read_text());require(manifest['source_sha']==source_sha,'runtime source changed');runtime=scratch/'runtime';runtime.mkdir();bundled={}
  for name,entry in manifest['files'].items():
   rel=Path(name);require(len(rel.parts)==2 and rel.parts[0]in('bin','lib')and all(re.fullmatch('[A-Za-z0-9_.+-]+',p)for p in rel.parts)and '..'not in rel.parts,'unsafe runtime path');src=origin/'runtime'/rel;require(src.stat().st_size==entry['bytes']and sha(src)==entry['sha256'],'runtime bytes changed');dst=runtime/rel;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
   if rel.parts[0]=='bin':dst.chmod(0o755)
   bundled[name]=entry
  r['runtime_bundle_manifest_sha256']=sha(manifest_path);r['same_producer_binaries']=True
  model=scratch/(profile.RECIPE+'.gguf');source_model=origin/model.name;require(source_model.stat().st_size==producer['candidate']['bytes']<=4050000000,'candidate byte count differs')
  with source_model.open('rb')as src,model.open('xb')as dst:shutil.copyfileobj(src,dst,4*2**20)
  require(sha(model)==producer['candidate']['sha256'],'full mounted model SHA differs');r['full_model_independently_rehashed']=True
  env=dict(os.environ);env.update(producer['fixed_gates']);env.update(CRISPEMBED_LIGHTON4B_Q6_K='1',CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD='1',CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE='0',LD_LIBRARY_PATH=str(runtime/'lib'),LC_ALL='C')
  for name,value in [('HEAD_BF16','1'),('BF16_QUANT_WEIGHTS','0'),('BF16_LINEAR','0'),('DIRECT_QUANT','0'),('PARALLEL_DEQUANT','1')]:require(env.get('CRISPEMBED_LIGHTON4B_'+name)==value,'fixed precision mode changed')
  r['fixed_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')};floor_path=ROOT/'docs/lighton4b-imatrix-q4km-quality-floor-v1.json';floor=json.loads(floor_path.read_text());r['floor_descriptor_sha256']=sha(floor_path);require(set(floor['cases'])=={'simple_table','arabic_printed_line'}and floor['heldout_data_included']is False,'calibration-only floor differs')
  case_specs=[('arabic_printed_line','ff1a390370b702fabf8b5a25f0feec76888bb1ad8197cf08c3efd44adec1de84'),('simple_table','6be5428f247b616e76d79513b62873cf0238842999be2fb0848997f8ced25763')]
  if form_only:
   form_floor_path=ROOT/'docs/lighton4b-imatrix-q4km-form-floor-v1.json';form_floor=json.loads(form_floor_path.read_text());require(form_floor['post_selection_diagnostic']and not form_floor['heldout_claim'],'form scope differs');r['form_floor_sha256']=sha(form_floor_path);floor={'cases':{'simple_form':form_floor['case']}};case_specs=[('simple_form',form_floor['original_capture_sha256'])]
  if form_only:
   # The producer has no form prefill; admit this worker's CPU arithmetic on its
   # existing full-vocabulary table row before undertaking the form diagnostic.
   _,_,platform_frozen=cases.load_contract('simple_table');platform_png=ROOT/'tests/regression/lighton4b-mixed-v1'/platform_frozen['png_file'];require(sha(platform_png)==platform_frozen['png_sha256'],'platform-control image changed');platform_image=Image.open(platform_png).convert('RGB');platform_rgb=platform_image.tobytes();require(hashlib.sha256(platform_rgb).hexdigest()==platform_frozen['rgb_sha256'],'platform-control RGB changed');platform_rgb_path=scratch/'platform.rgb';platform_rgb_path.write_bytes(platform_rgb);platform_teacher=scratch/'platform.teacher.ids';platform_teacher.write_text(str(producer['cases']['simple_table']['weighted_q4km']['generated_ids'][0])+'\n');platform_out=out/'platform-control';platform_out.mkdir();r['model_forward_attempted']=True;r['model_forward_executed']=None;save()
   run('same-producer-platform-control',[runtime/'bin/lighton4b-native-document',model,platform_rgb_path,platform_image.height,platform_image.width,platform_teacher,platform_out,1,4,'smoke'],env,10*60)
   platform_expected=producer['cases']['simple_table']['weighted_q4km'];require(sha(platform_out/'prefill.logits.f32')==platform_expected['first_score_sha256']==sha(platform_out/'teacher.0.logits.f32')==sha(platform_out/'greedy.0.logits.f32'),'producer-platform full-vocabulary arithmetic changed');platform_stable={name:row['sha256']for name,row in platform_expected['artifacts'].items()if name.startswith(('prefill.','vision.','processor.','text.'))or name in('prompt.ids','modalities.ids','rendered.txt')};require(all(sha(platform_out/name)==digest for name,digest in platform_stable.items()),'producer-platform stage changed');r['model_forward_executed']=True;r['platform_control']={'first_score_sha256':platform_expected['first_score_sha256'],'exact_stage_hashes':platform_stable};save()
  for case,digest in case_specs:
   if form_only:
    with urllib.request.urlopen(form_floor['original_capture_url'],timeout=60)as remote:raw=remote.read(2**20+1)
    require(len(raw)<=2**20 and hashlib.sha256(raw).hexdigest()==digest,'form original receipt changed');capture_path=scratch/'form-original.json';capture_path.write_bytes(raw);capture=json.loads(raw);require(capture['schema']=='lighton4b-bf16-freegen-capture/v1'and capture['status']=='captured'and capture['requested_dtype']=='bfloat16'and capture['parameter_dtype_counts']=={'torch.bfloat16':723}and capture['source']['revision']=='b71010f095dc3735de7ee6b9969bff8f8f50b39a'and len(capture['generated_ids'])==177 and capture['eos_completed']and not capture['cap_hit'],'form BF16 source differs');png=ROOT/'tests/regression/images/cc0/simple_form.png';require(sha(png)=='3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc','form image changed');image=Image.open(png).convert('RGB');rgb=image.tobytes()
   else:
    capture_path=producer_helpers.unique_capture(Path('/kaggle/input'),case,digest);capture=json.loads(capture_path.read_text());_,contract,frozen=cases.load_contract(case);cases.validate_original(capture,contract);require(contract['split']=='calibration'and capture['eos_completed']and not capture['cap_hit'],'complete source calibration required');png=ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file'];require(sha(png)==frozen['png_sha256'],'image changed');image=Image.open(png).convert('RGB');rgb=image.tobytes();require(image.size==(frozen['width'],frozen['height'])and hashlib.sha256(rgb).hexdigest()==frozen['rgb_sha256'],'RGB changed')
   rgb_path=scratch/(case+'.rgb');rgb_path.write_bytes(rgb);teacher=scratch/(case+'.teacher.ids');teacher.write_text(str(capture['generated_ids'][0])+'\n');folder=out/case;folder.mkdir();pre=folder/'preflight';pre.mkdir();native=folder/'native';native.mkdir()
   run(case+'-preflight',[runtime/'bin/lighton4b-native-case-preflight',model,rgb_path,image.height,image.width,pre,profile.RECIPE],env,10*60)
   arrays={}
   for key,row in capture['prepared_inputs'].items():
    if form_only:
     if key=='pixel_values':v=np.fromfile(pre/'processor.patches.f32',dtype='<f4').reshape(row['shape'])
     elif key=='attention_mask':v=np.ones(row['shape'],dtype='<i8')
     else:v=np.array([int(x)for x in(pre/({'image_grid_thw':'grid','input_ids':'prompt','mm_token_type_ids':'modalities'}[key]+'.ids')).read_text().split()],dtype='<i8').reshape(row['shape'])
     serialized=io.BytesIO();np.save(serialized,v,allow_pickle=False);require(hashlib.sha256(serialized.getvalue()).hexdigest()==row['sha256'],'form actual prepared bytes differ: '+key);arrays[key]=v
    else:require(Path(row['file']).name==row['file']and sha(capture_path.parent/row['file'])==row['sha256'],'original prepared bytes changed');arrays[key]=np.load(capture_path.parent/row['file'],allow_pickle=False)
   require(set(arrays)=={'pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'}and(pre/'processor.patches.f32').read_bytes()==arrays['pixel_values'].tobytes()and np.all(arrays['attention_mask']==1),'native/source processor differs')
   for key,name in [('image_grid_thw','grid'),('input_ids','prompt'),('mm_token_type_ids','modalities')]:require([int(x)for x in(pre/(name+'.ids')).read_text().split()]==arrays[key].reshape(-1).tolist(),'native/source input differs: '+key)
   require(common.memory()['available_bytes']>=8*2**30,'native child RAM unavailable');cap=256;deadline=90*60 if form_only else(75*60 if case=='simple_table'else 20*60)
   r['model_forward_attempted']=True
   if not r['model_forward_executed']:r['model_forward_executed']=None
   save();run(case+'-complete-greedy',[runtime/'bin/lighton4b-native-document',model,rgb_path,image.height,image.width,teacher,native,cap,4,'smoke'],env,deadline)
   summary=json.loads((native/'native-summary.json').read_text());ids=[int(x)for x in(native/'generated.ids').read_text().split()];raw_text=(native/'generated.txt').read_bytes();valid_utf8=True
   try:text=raw_text.decode('utf-8')
   except UnicodeDecodeError:valid_utf8=False;text=raw_text.decode('utf-8',errors='replace')
   require(1<=len(ids)<=cap and all(0<=i<248320 for i in ids),'complete independent IDs missing');eos=ids[-1]==248046;require(summary['generated_tokens']==len(ids)and summary['cap']==cap and summary['eos']==eos and(eos or len(ids)==cap),'independent EOS/cap incomplete');expected=producer['cases'][case]['weighted_q4km']if not form_only else None;first=native/'prefill.logits.f32';require(sha(first)==sha(native/'teacher.0.logits.f32')==sha(native/'greedy.0.logits.f32'),'prefill/teacher/independent row0 differs')
   if expected is not None:require(sha(first)==expected['first_score_sha256'],'same-binary producer/source-platform score0 changed')
   stable={name:row['sha256']for name,row in(expected['artifacts']if expected is not None else{}).items()if name.startswith(('prefill.','vision.','processor.','text.'))or name in('prompt.ids','modalities.ids','rendered.txt')};require(all(sha(native/name)==digest for name,digest in stable.items()),'same producer prefill stages changed');rows=[];require(len(list(native.glob('greedy.*.logits.f32')))==len(ids),'independent greedy row count differs')
   for index,token in enumerate(ids):
    path=native/('greedy.'+str(index)+'.logits.f32');require(path.stat().st_size==993280,'incomplete vocabulary row');v=np.fromfile(path,dtype='<f4');require(np.isfinite(v).all()and int(v.argmax())==token,'actual selected token not row argmax');rows.append({'position':index,'selected_id':token,'sha256':sha(path)})
   agreement=helpers.agreement(ids,text,capture);limit=floor['cases'][case];require(agreement['cer']['reference_codepoints']==limit['reference_codepoints']and agreement['wer']['reference_words']==limit['reference_words'],'quality denominator differs');floor_met=valid_utf8 and agreement['cer']['edits']<=limit['character_edits']and agreement['wer']['edits']<=limit['word_edits']and(eos or not limit['eos_completed'])
   r['cases'][case]={'generated_ids':ids,'decoded':text,'utf8_valid':valid_utf8,'actual_eos':eos,'cap_hit':not eos,'cap':cap,'native_summary':summary,'source_capture_sha256':digest,'same_producer_stage_hashes':stable,'producer_form_prefill_available':not form_only,'first_score_sha256':sha(first),'greedy_rows':rows,'against_genuine_bf16':agreement,'against_pinned_llama_q4km':helpers.agreement(ids,text,{'generated_ids':limit['ids'],'decoded':limit['decoded']}),'observed_upstream_q4km_floor_met':floor_met,'upstream_floor':{key:limit[key]for key in('character_edits','reference_codepoints','word_edits','reference_words','eos_completed','cap_hit')},'artifacts':{x.name:{'bytes':x.stat().st_size,'sha256':sha(x)}for x in native.iterdir()if x.is_file()}};r['model_forward_executed']=True;save()
  r['all_selected_observed_floors_met']=all(v['observed_upstream_q4km_floor_met']for v in r['cases'].values());r['all_two_calibration_floors_met']=r['all_selected_observed_floors_met']if not form_only else None;r['status']='measured'
 except Exception as e:r['error_type']=type(e).__name__;r['error']=str(e)if isinstance(e,ValueError)else'bounded complete-decode failed';raise
 finally:signal.alarm(0);r['seconds']=time.monotonic()-started;save()
if __name__=='__main__':main()
