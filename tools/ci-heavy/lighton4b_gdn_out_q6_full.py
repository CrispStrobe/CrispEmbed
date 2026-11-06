#!/usr/bin/env python3
"""Exact retained Q6-output candidate, calibration-only smoke then independent full greedy."""
import hashlib,importlib.util,json,os,re,shutil,signal,subprocess,sys,time,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def require(ok,message):
 if not ok:raise ValueError(message)
def levenshtein(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]
def agreement(ids,text,ref):
 teacher=ref['generated_ids'];prefix=next((i for i,(a,b)in enumerate(zip(ids,teacher))if a!=b),min(len(ids),len(teacher)));words=ref['decoded'].split();cer=levenshtein(ref['decoded'],text);wer=levenshtein(words,text.split())
 return {'ids_exact':ids==teacher,'decoded_exact':text==ref['decoded'],'matching_prefix_ids':prefix,'first_differing_id':None if ids==teacher else prefix,'cer':{'edits':cer,'reference_codepoints':len(ref['decoded']),'rate':cer/max(1,len(ref['decoded']))},'wer':{'edits':wer,'reference_words':len(words),'rate':wer/max(1,len(words))},'normalization':'none; Unicode codepoints and whitespace words; model agreement, not human OCR accuracy'}
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'q6-out-full';scratch.mkdir();p=json.loads((ROOT/'docs/lighton4b-gdn-out-q6-full-protocol-v1.json').read_text())
 helper=load('q6_helpers',ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_candidate.py');common=load('q6_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py');cases=load('q6_cases',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py');transport=load('q6_transport',ROOT/'tools/ci-heavy/lighton4b_calibration_native.py');vision=load('q6_vision',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py')
 report={'schema':'lighton4b-gdn-out-q6-full-calibration/v1','status':'blocked','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'runtime_reference_commit':p['runtime_reference_commit'],'protocol_sha256':common.sha(ROOT/'docs/lighton4b-gdn-out-q6-full-protocol-v1.json'),'calibration_cases':p['cases'],'heldout_captures_read':False,'original_batch_json_read':False,'reference_model_forward_executed':False,'f32_model_instantiated':False,'quality_qualified':False,'cases':{}}
 active={}
 def save():(out/'gdn-out-q6-full-calibration.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 def terminate(*args):
  report['deadline_reached']=True;report['deadline_kind']='global90minutes'if args and args[0]==signal.SIGALRM else'termination_signal'if args else'bounded_child_timeout';report['partial_native_outputs_preserved']=True;save()
  child=active.get('child')
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  raise TimeoutError('bounded full calibration deadline')
 def run(label,args,env=None,timeout=1200):
  report['stage']=label;report['active_child_timeout_seconds']=timeout;save()
  with(out/(label+'.log')).open('w')as log:
   child=subprocess.Popen(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active['child']=child
   try:code=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:terminate()
   active.pop('child',None);require(code==0,'bounded child failed: '+label)
 def get(url,path,digest,limit):
  h=hashlib.sha256();count=0
  with urllib.request.build_opener(helper.ArtifactRedirect()).open(url,timeout=120)as src,path.open('xb')as dst:
   while block:=src.read(4*2**20):count+=len(block);require(count<=limit,'authenticated transfer bound exceeded');h.update(block);dst.write(block)
  require(h.hexdigest()==digest,'authenticated transfer hash differs');return count
 def headroom():
  m=common.memory();require(m['available_bytes']>=8*2**30 and(m['cgroup_limit_bytes']is None or m['cgroup_limit_bytes']-m['cgroup_current_bytes']>=2*2**30),'native effective RAM/cgroup margin unavailable');return m
 try:
  import numpy as np
  from PIL import Image
  require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','frozen serializer/RGB versions differ');require(p['cases']==['simple_table','arabic_printed_line']and p['max_new_tokens']==256,'closed calibration scope differs')
  require(all(common.sha(ROOT/k)==v for k,v in p['runtime_paths_sha256'].items()),'compiled native source differs from first-score source');require(subprocess.check_output(['git','rev-parse','HEAD:ggml'],cwd=ROOT,text=True).strip()==p['ggml_commit'],'ggml source differs')
  require(common.sha(ROOT/'docs/lighton4b-mixed-calibration-v1.json')==p['calibration_protocol_sha256']and common.sha(ROOT/'tests/regression/lighton4b-mixed-v1/frozen-inputs.json')==p['input_manifest_sha256'],'frozen calibration/input manifests differ')
  a=shutil.disk_usage(scratch).free;b=shutil.disk_usage(out).free;shared=scratch.stat().st_dev==out.stat().st_dev;report['preflight']={'memory':headroom(),'load_average':os.getloadavg(),'scratch_free_bytes':a,'output_free_bytes':b,'shared_filesystem':shared,'swap':{k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('SwapTotal','SwapFree')}};require(a>=16*2**30 if shared else a>=12*2**30 and b>=4*2**30,'bounded scratch/output capacity unavailable');signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate);signal.signal(signal.SIGALRM,terminate);signal.alarm(90*60)
  def receipt(key,name):
   q=p[key];path=out/name;get(q['url'],path,q['sha256'],2**20);return json.loads(path.read_text())
  first=receipt('first_score_receipt','first-score-reference.json');prior=receipt('decoded_baseline_receipt','decoded-baseline-reference.json');require(first['status']=='measured'and first['candidate_sha256']==p['model']['sha256']and first['baseline_hash_controls_passed'],'first-score receipt is not admitted');require(prior['status']=='completed'and prior['source_sha']==p['decoded_baseline_receipt']['source'],'decoded baseline source differs')
  for case in p['cases']:
   for arm,old in[('plain_q4','q4'),('gdn_out_q8','out')]:require(first['cases'][case][arm]['first_score_sha256']==prior['cases'][case][old]['first_score_sha256'],'retained Q4/Q8 baseline row pins disagree')
   require(first['cases'][case]['selective_q6']['first_score_sha256']==p['first_score_sha256'][case],'candidate first-score pin differs')
  build=scratch/'build';run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);targets=['test-lighton4b-gdn-out-q6-recipe','lighton4b-native-document','lighton4b-native-case-preflight'];run('build',['cmake','--build',build,'--target',*targets,'--parallel','2']);report['binary_sha256']={x:common.sha(build/x)for x in targets};schema=out/'q6-out-schema.tsv';lines=[]
  for row in helper.plan()['all_tensors']:
   if row['name'].startswith('l.'):
    ne=list(reversed(row['source_shape']));ne += [1]*(4-len(ne));lines.append(' '.join([row['name'],str(row['type']),*map(str,ne)]))
  schema.write_text('\n'.join(lines)+'\n');run('native-q6-controls',[build/targets[0],schema])
  q=p['artifact'];token=os.environ.get('GH_TOKEN');require(bool(token),'read-only artifact authentication unavailable');archive=scratch/'candidate-transport.zip';request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(q['id'])+'/zip',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'});require(get(request,archive,q['sha256'],q['bytes'])==q['bytes'],'archive size differs');model=scratch/p['model']['file']
  with zipfile.ZipFile(archive)as z:
   members=[x for x in z.infolist()if Path(x.filename).name==p['model']['file']];require(len(members)==1 and members[0].file_size==p['model']['bytes'],'unique exact candidate member absent')
   with z.open(members[0])as src,model.open('xb')as dst:shutil.copyfileobj(src,dst,4*2**20)
  require(common.sha(model)==p['model']['sha256'],'exact candidate weight hash differs');archive.unlink();report['model_authenticated']=p['model'];urls=transport.original_urls();original=scratch/'original';original.mkdir();originals={}
  for case,capture_sha,_ in transport.CASES:
   require(case in p['cases'],'noncalibration source transfer prohibited');folder=original/case;folder.mkdir();get(urls[helper.calibration_member(case,'capture.json')],folder/'capture.json',capture_sha,2**20);capture=json.loads((folder/'capture.json').read_text());_,contract,frozen=cases.load_contract(case);cases.validate_original(capture,contract);require(contract['split']=='calibration'and capture['eos_completed']and not capture['cap_hit'],'original calibration is not complete');originals[case]=(capture,frozen)
   for arm,sha in [('q4','f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'),('out','40e54560ed349c077100c4e39276d29c02b98d6bec3611ac7b3a88d5a771c710')]:
    b=prior['cases'][case][arm];require(b['original_capture_sha256']==capture_sha and b['input_freeze_sha256']==p['input_manifest_sha256']and b['weights_sha256']==sha and b['actual_cap']==256 and b['input_admitted']and b['independent_generation'],'retained full baseline provenance differs')
   for record in capture['prepared_inputs'].values():get(urls[helper.calibration_member(case,record['file'])],folder/record['file'],record['sha256'],8*2**20)
  env=os.environ.copy();env.update({x:'1'for x in common.GATES});env.update({x:'1'for x in vision.VISION_GATES});env.update({x:'0'for x in ('CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2','CRISPEMBED_LIGHTON4B_DIRECT_QUANT','CRISPEMBED_QWEN35_WEIGHT_CACHE','CRISPEMBED_QWEN35_PRECISE_RMS','CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_LIGHTON4B_GREEDY_TRACE','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4','CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8')});env.update(CRISPEMBED_LIGHTON4B_Q6_K='1',CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6='1',CRISPEMBED_LIGHTON4B_HEAD_BF16='1',CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1',CRISPEMBED_LIGHTON4B_FROZEN_CASES='1');report['fixed_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')};admitted={};require(all(env.get('CRISPEMBED_LIGHTON4B_'+k)==str(v)for k,v in p['fixed_precision_gates'].items()),'fixed precision gate differs')
  for phase,cap in [('smoke',1),('full',256)]:
   for case in p['cases']:
    capture,frozen=originals[case];png=ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file'];require(common.sha(png)==frozen['png_sha256'],'frozen PNG differs');image=Image.open(png).convert('RGB');rgb=image.tobytes();require(image.size==(frozen['width'],frozen['height'])and hashlib.sha256(rgb).hexdigest()==frozen['rgb_sha256'],'frozen RGB differs');rgb_path=scratch/(case+'.rgb');rgb_path.write_bytes(rgb);teacher=scratch/(case+'.teacher.ids');teacher.write_text(str(capture['generated_ids'][0])+'\n');folder=out/phase/case;folder.mkdir(parents=True);native=folder/'native';native.mkdir()
    if phase=='smoke':
     pre=folder/'preflight';pre.mkdir();run(case+'-preflight',[build/'lighton4b-native-case-preflight',model,rgb_path,image.height,image.width,pre,p['recipe']],env);arrays={k:np.load(original/case/r['file'],allow_pickle=False)for k,r in capture['prepared_inputs'].items()};require(set(arrays)=={'pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'}and all(list(v.shape)==capture['prepared_inputs'][k]['shape']and v.dtype==np.dtype('<f4'if k=='pixel_values'else'<i8')for k,v in arrays.items()),'source prepared-input geometry differs');require((pre/'processor.patches.f32').read_bytes()==arrays['pixel_values'].tobytes(order='C')and np.all(arrays['attention_mask']==1),'actual processor/mask differs')
     for field,name in [('image_grid_thw','grid'),('input_ids','prompt'),('mm_token_type_ids','modalities')]:require([int(x)for x in(pre/(name+'.ids')).read_text().split()]==arrays[field].reshape(-1).tolist(),'actual native prepared input differs: '+field)
     admitted[case]=True
    else:require(len(admitted)==2 and all(admitted.values()),'both smokes required before full generation')
    with model.open('rb')as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    mem=headroom();started=time.monotonic();run(phase+'-'+case,['/usr/bin/time','-v','-o',folder/'native-time.txt',build/'lighton4b-native-document',model,rgb_path,image.height,image.width,teacher,native,cap,4,'smoke'],dict(env,LC_ALL='C'),timeout=45*60);elapsed=time.monotonic()-started
    score=native/'prefill.logits.f32';require(score.stat().st_size==993280 and common.sha(score)==p['first_score_sha256'][case]==common.sha(native/'teacher.0.logits.f32')==common.sha(native/'greedy.0.logits.f32'),'first score/teacher/independent greedy anchor differs');frozen_files=first['cases'][case]['selective_q6']['artifacts'];checked={name:row['sha256']for name,row in frozen_files.items()if name.startswith(('prefill.','vision.','processor.','text.'))or name in('prompt.ids','modalities.ids','rendered.txt')};require(all(common.sha(native/name)==digest for name,digest in checked.items()),'actual retained source stage differs')
    ids=[int(x)for x in(native/'generated.ids').read_text().split()];text=(native/'generated.txt').read_text();require(1<=len(ids)<=cap and all(0<=x<248320 for x in ids),'native independent IDs malformed');rows=[];require(len(list(native.glob('greedy.*.logits.f32')))==len(ids),'independent score row count differs')
    for i,token_id in enumerate(ids):
     row=native/f'greedy.{i}.logits.f32';require(row.stat().st_size==993280,'complete greedy score row missing');values=np.fromfile(row,dtype='<f4');require(np.isfinite(values).all()and int(values.argmax())==token_id,'actual independent greedy argmax differs');rows.append({'position':i,'sha256':common.sha(row),'selected_id':token_id})
    summary=json.loads((native/'native-summary.json').read_text());eos=ids[-1]==248046;require(summary['eos']==eos and summary['generated_tokens']==len(ids)and summary['cap']==cap,'independent EOS/cap summary differs');require(eos or len(ids)==cap,'native stopped before primary EOS/cap');value={'generated_ids':ids,'decoded':text,'eos_completed':eos,'cap_hit':not eos and len(ids)==cap,'cap':cap,'native_summary':summary,'seconds':elapsed,'native_peak_rss_bytes':int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',(folder/'native-time.txt').read_text()).group(1))*1024,'resource_preflight':mem,'input_admitted':True,'first_score_sha256':common.sha(score),'exact_retained_stage_sha256':checked,'greedy_scores':rows,'against_genuine_bf16':agreement(ids,text,capture),'against_retained_plain_q4':agreement(ids,text,prior['cases'][case]['q4']),'against_retained_q8_out':agreement(ids,text,prior['cases'][case]['out']),'artifacts':{x.name:{'bytes':x.stat().st_size,'sha256':common.sha(x)}for x in native.iterdir()if x.is_file()}}
    report.setdefault(phase,{})[case]=value
    if phase=='full':report['cases'][case]=value
    save()
  report.update(status='measured',all_prefill_hash_controls_passed=True,all_full_outputs_exact=all(x['against_genuine_bf16']['ids_exact']and x['against_genuine_bf16']['decoded_exact']for x in report['cases'].values()),all_eos_completed=all(x['eos_completed']for x in report['cases'].values()),full_generation_executed=True)
 except Exception as e:report['error_type']=type(e).__name__;report['error']=str(e)if isinstance(e,ValueError)else'closed bounded full calibration failed';raise
 finally:signal.alarm(0);save()
if __name__=='__main__':main()
