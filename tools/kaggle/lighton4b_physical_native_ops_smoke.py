#!/usr/bin/env python3
"""Reference-only same-physical-text upstream comparison; Kaggle CPU only."""
import base64,hashlib,io,json,os,signal,shutil,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
def memory():
 vals={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in ('MemAvailable','SwapTotal','SwapFree')}
 observed=0;bounds=[]
 for line in Path('/proc/self/cgroup').read_text().splitlines():
  hierarchy,controllers,relative=line.split(':',2)
  if hierarchy=='0'and controllers=='':base=Path('/sys/fs/cgroup');limitname='memory.max';usedname='memory.current'
  elif 'memory'in controllers.split(','):base=Path('/sys/fs/cgroup/memory');limitname='memory.limit_in_bytes';usedname='memory.usage_in_bytes'
  else:continue
  target=base/relative.lstrip('/')
  for node in (target,*target.parents):
   if node!=base and base not in node.parents:break
   try:
    raw=(node/limitname).read_text().strip();used=int((node/usedname).read_text());observed+=1
    if raw!='max'and int(raw)<2**60:bounds.append(max(0,int(raw)-used))
   except(OSError,ValueError):pass
 vals['cgroup_observed']=observed;vals['finite_cgroup_bounds']=len(bounds)
 if bounds:vals['MemAvailable']=min(vals['MemAvailable'],*bounds)
 require(observed>0,'actual memory cgroup inspection')
 return vals
def download(url,p,size,expected):
 require(not p.exists(),'task destination exists');h=hashlib.sha256();count=0
 with urllib.request.urlopen(url,timeout=120)as src,p.open('xb')as dst:
  while b:=src.read(8*1024**2):
   count+=len(b);require(count<=size,'download bound');h.update(b);dst.write(b)
 require(count==size and h.hexdigest()==expected,'complete immutable download hash')
def edits(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  n=[i]
  for j,y in enumerate(b,1):n.append(min(d[j]+1,n[-1]+1,d[j-1]+(x!=y)))
  d=n
 return d[-1]
def stop(p):
 if p is not None and p.poll()is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def admit_protocol(cfg):
 require(cfg['schema']=='lighton4b-physical-native-ops-smoke/v1','closed smoke schema')
 require(cfg['arms']==['reference','off','on'] and cfg['cap']==2 and len(cfg['cases'])==1 and cfg['cases'][0]['id']=='simple_form','closed two-token form/reference/OFF/ON scope')
 require(cfg['actual_baseline_source_sha']=='ac34f72ca89823a177c538ad598dfa00fc69f124','closed regenerated source')
 require(cfg['upstream_commit']=='6c01da6278ab6c69ed67d9124806b50d2b299ba7','closed upstream source')
 require(cfg['expected_candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43','closed candidate')
 require(cfg['models']['Q4_K_M']['sha256']=='b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26' and cfg['models']['Q4_K_M']['bytes']==2708804576,'closed physical weights')
 require(cfg['threads']==4 and cfg['context']==2048 and cfg['image_min_tokens']==64 and cfg['image_max_tokens']==256,'frozen CPU/input geometry')
 require(cfg['native_arm_timeout_seconds']<=900 and cfg['total_model_timeout_seconds']<=2400,'bounded smoke budgets')
def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir(),'Kaggle CPU only')
 import numpy as np
 from PIL import Image
 import sys,importlib.util
 require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','serializer pins')
 sys.path.insert(0,str(ROOT/'tools'))
 import lighton4b_upstream_observer_patch as observer
 import lighton4b_upstream_physical_pack as pack
 def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 common=module('native_document',ROOT/'tools/ci-heavy/lighton4b_native_document.py');vision=module('vision_smoke',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py')
 protocol=ROOT/'docs/lighton4b-physical-native-ops-smoke-v1.json';cfg=json.loads(protocol.read_text());admit_protocol(cfg);out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'physical-native-ops-smoke';scratch.mkdir();active=None
 report={'schema':cfg['schema'],'status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(protocol),'quality_qualified':False,'f32_model_instantiated':False,'full_completion_executed':False,'original_forward_executed':False,'preflight':{'memory':memory(),'scratch_free':shutil.disk_usage(scratch).free,'output_free':shutil.disk_usage(out).free,'load':os.getloadavg()},'cases':{},'process_resources':{}}
 def save(phase=None):
  if phase:report['phase']=phase;print('phase',phase,flush=True)
  (out/'physical-native-ops-smoke.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 def interrupted(signum,frame):report['interrupted_signal']=signum;stop(active);save();raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(cfg['total_model_timeout_seconds'])
 def command(label,args,env=None,timeout=1800,on_tick=None):
  nonlocal active
  save(label);start=time.monotonic();peak=0;failure=None;stage='spawn';process=None
  with(out/(label+'.log')).open('w')as f:
   try:
    active=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True);process=active
    while True:
     stage='deadline'
     if time.monotonic()-start>=timeout:raise TimeoutError(label+' deadline')
     if on_tick:stage='on_tick';on_tick()
     stage='memory_sample'
     try:
      rows=dict(line.split(':',1)for line in Path('/proc/'+str(active.pid)+'/status').read_text().splitlines());peak=max(peak,int(rows.get('VmHWM','0').split()[0])*1024)
     except(FileNotFoundError,ProcessLookupError):pass
     stage='wait'
     try:code=active.wait(timeout=1);break
     except subprocess.TimeoutExpired:pass
    stage='exit_status';require(code==0,label+' failed')
   except BaseException as error:
    failure={'error_type':type(error).__name__,'failure_stage':stage}
    if isinstance(error,(RuntimeError,ValueError,TimeoutError)):failure['reason']=str(error)
    raise
   finally:
    try:stop(process)
    finally:
     report['process_resources'][label]={'peak_rss_bytes_observed':peak,'elapsed_seconds':time.monotonic()-start,'returncode':process.returncode if process is not None else None,'status':'failed'if failure else'completed'}
     if failure:report['process_resources'][label].update(failure)
     active=None;save()
  return time.monotonic()-start
 def fetch(url,path,expected,limit=4*1024**2):
  with urllib.request.urlopen(url,timeout=120)as r:b=r.read(limit+1)
  require(len(b)<=limit and hashlib.sha256(b).hexdigest()==expected,'immutable bounded reference');path.write_bytes(b);return json.loads(b)
 def api(port,path,data=None,timeout=60):
  body=None if data is None else json.dumps(data).encode();req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=body,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=timeout)as r:b=r.read(8*1024**2+1)
  require(len(b)<=8*1024**2,'bounded server response');return json.loads(b)
 def metric(ids,text,refids,reftext):
  prefix=0
  for x,y in zip(ids,refids):
   if x!=y:break
   prefix+=1
  return {'raw_ids_exact':ids==refids,'decoded_exact':text==reftext,'matching_prefix_ids':prefix,'character_edits':edits(reftext,text),'reference_codepoints':len(reftext),'word_edits':edits(reftext.split(),text.split()),'reference_words':len(reftext.split())}
 def diff(a,b):
  a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);require(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'finite same-shape comparison');d=a-b
  return {'bit_exact':a.astype('<f4').tobytes()==b.astype('<f4').tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max_abs':float(np.abs(b).max()),'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))}
 try:
  require(report['preflight']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes'],'10GiB effective RAM before transfer');require(report['preflight']['scratch_free']>=cfg['minimum_scratch_free_bytes']and report['preflight']['output_free']>=cfg['minimum_output_free_bytes'],'24GiB scratch/8GiB output before transfer')
  baseline_files={};report['baseline_kind']=cfg['baseline_kind'];require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact smoke source')
  command('host-cpu',['lscpu']);command('submodules',['git','-C',ROOT,'submodule','update','--init','--depth','1','--recursive'])
  repo=scratch/'llama.cpp'
  for label,args in [('up-init',['git','init',repo]),('up-remote',['git','-C',repo,'remote','add','origin',cfg['upstream_repo']]),('up-fetch',['git','-C',repo,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])]:command(label,args)
  require(subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==observer.SOURCE==cfg['upstream_commit'],'pinned upstream code');sys.path.insert(0,str(repo/'gguf-py'));from gguf.quants import dequantize;from gguf import GGMLQuantizationType
  report['observer_patch']=observer.patch(repo)
  ub=scratch/'up-build';opts=['-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF','-DGGML_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'];command('up-configure',['cmake','-S',repo,'-B',ub,*opts]);command('up-build',['cmake','--build',ub,'--target','llama-server','--parallel','4']);shutil.copy(ub/'CMakeCache.txt',out/'upstream-CMakeCache.txt');server=ub/'bin/llama-server'
  nb=scratch/'native-build';command('native-configure',['cmake','-S',ROOT,'-B',nb,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);command('native-build',['cmake','--build',nb,'--target','lighton4b-native-document','--parallel','4']);shutil.copy(nb/'CMakeCache.txt',out/'native-CMakeCache.txt');native=nb/'lighton4b-native-document';report['binary_sha256']={'native':sha(native),'upstream':sha(server)}
  oldroot=scratch/'baseline-source'
  for label,args in [('old-init',['git','init',oldroot]),('old-remote',['git','-C',oldroot,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']),('old-fetch',['git','-C',oldroot,'fetch','--depth','1','origin',cfg['actual_baseline_source_sha']]),('old-checkout',['git','-C',oldroot,'checkout','--detach','FETCH_HEAD']),('old-submodule',['git','-C',oldroot,'submodule','update','--init','ggml'])]:command(label,args)
  require(subprocess.check_output(['git','-C',oldroot,'rev-parse','HEAD'],text=True).strip()==cfg['actual_baseline_source_sha'],'actual regenerated baseline source');require(sha(oldroot/'docs/lighton4b-upstream-physical-decode-v1.json')==cfg['baseline_protocol_sha256'],'unchanged old physical protocol bytes')
  oldbuild=scratch/'baseline-build';command('old-configure',['cmake','-S',oldroot,'-B',oldbuild,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);command('old-build',['cmake','--build',oldbuild,'--target','lighton4b-native-document','--parallel','4']);oldnative=oldbuild/'lighton4b-native-document';shutil.copy(oldbuild/'CMakeCache.txt',out/'reference-CMakeCache.txt');report['binary_sha256']['reference']=sha(oldnative);report['reference_source_sha']=cfg['actual_baseline_source_sha']
  for profile in ['upstream','native','reference']:require('GGML_LLAMAFILE:BOOL=ON' in (out/(profile+'-CMakeCache.txt')).read_text(),'explicit same LLAMAFILE profile')
  prefix='https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/';up=scratch/cfg['models']['Q4_K_M']['file'];proj=scratch/cfg['projector']['file']
  for path,row in [(up,cfg['models']['Q4_K_M']),(proj,cfg['projector'])]:save('download-'+path.name);download(prefix+path.name,path,row['bytes'],row['sha256'])
  base=cfg['base_model'];bp=scratch/base['file'];save('download-protected-base');download('https://huggingface.co/'+base['repository']+'/resolve/'+base['revision']+'/'+base['file'],bp,base['bytes'],base['sha256']);candidate=scratch/'physical-native.gguf';save('lossless-pack');audit=pack.convert(bp,up,candidate,out/'physical-pack.json');report['candidate_sha256']=audit['candidate_sha256'];report['candidate_bytes']=candidate.stat().st_size;require(report['candidate_sha256']==cfg['expected_candidate_sha256'],'candidate byte identity to first independently persisted transport');bp.unlink()
  header=pack.parse(up,pack.TYPES);embedding=header['tensors']['token_embd.weight'];require(embedding['type']==14 and embedding['stored_shape']==[2560,248320],'source tied Q6 embedding geometry')
  env=dict(os.environ);env.update({x:'0'for x in env if x.startswith('CRISPEMBED_')});env.update({x:'1'for x in common.GATES});env.update({x:'1'for x in vision.VISION_GATES});env.update(CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL='1',CRISPEMBED_LIGHTON4B_Q6_K='1',CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1',CRISPEMBED_LIGHTON4B_HEAD_BF16='0',CRISPEMBED_LIGHTON4B_BF16_LINEAR='0',CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS='0');report['native_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')}
  prepared=[]
  for case in cfg['cases']:
   name=case['id'];folder=out/name;folder.mkdir();original=fetch(case['reference_url'],folder/'original.json',case['reference_sha256']);retained=fetch(case['upstream_response_url'],folder/'retained-upstream-response.json',case['upstream_response_sha256']);imagepath=ROOT/case['image'];require(sha(imagepath)==case['image_sha256'],'frozen image');image=Image.open(imagepath).convert('RGB');canonical=folder/'input.png';Image.frombytes('RGB',image.size,image.tobytes()).save(canonical,compress_level=6);rgb=folder/'input.rgb';rgb.write_bytes(image.tobytes());teacher=folder/'unused-one-teacher.ids';teacher.write_text(str(original['generated_ids'][0])+'\n');responses=[];tokens=None
   report['cases'][name]={'arms':{}};save(name+'-source-control')
   for enabled in [False,True]:
    capture=folder/('observer-on'if enabled else'observer-off');capture.mkdir();child=dict(os.environ);child.pop(observer.ENV,None)
    if enabled:child[observer.ENV]=str(capture)
    with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    args=[server,'-m',up,'--mmproj',proj,'--host','127.0.0.1','--port',str(port),'-t','4','-tb','4','-c','2048','-np','1','-ngl','0','--no-mmproj-offload','--no-warmup','--flash-attn','off','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--image-min-tokens',str(cfg['image_min_tokens']),'--image-max-tokens',str(cfg['image_max_tokens'])]
    with(capture/'server.log').open('w')as log:
     active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=child,start_new_session=True)
     try:
      ready=False
      for _ in range(180):
       require(active.poll()is None,'source load failed')
       try:api(port,'/health');ready=True;break
       except Exception:time.sleep(1)
      require(ready,'source ready deadline');props=api(port,'/props');marker=props.get('media_marker');require(isinstance(marker,str)and 1<=len(marker)<=256,'actual transport marker');templ=api(port,'/apply-template',{'messages':[{'role':'user','content':marker}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt'];require(templ.count(marker)==1 and templ.replace(marker,'<__media__>')==cfg['expected_media_prompt'],'exact original wrapper');expanded=templ.replace(marker,'<|vision_start|>'+'<|image_pad|>'*case['image_tokens']+'<|vision_end|>');current=api(port,'/tokenize',{'content':expanded,'add_special':False,'parse_special':True})['tokens'];buf=io.BytesIO();np.save(buf,np.array([current],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(buf.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'original prompt ID bytes');tokens=current
      req={'prompt':{'prompt_string':templ,'multimodal_data':[base64.b64encode(canonical.read_bytes()).decode()]},'n_predict':2,'temperature':0,'seed':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'cache_prompt':False,'return_tokens':True,'n_probs':5,'stream':False};response=api(port,'/completion',req,300);(capture/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False)+'\n');require(not response.get('truncated')and response['tokens_evaluated']==case['expected_prompt_tokens'],'actual source prompt geometry');require(response['tokens']==retained['tokens'][:2],'source prefix differs from retained full baseline');require(response['completion_probabilities']==retained['completion_probabilities'][:2],'source prefix probabilities differ from retained full baseline');responses.append(response)
     finally:stop(active);active=None
   require(responses[0]['tokens']==responses[1]['tokens']and responses[0]['completion_probabilities']==responses[1]['completion_probabilities'],'observer OFF/ON exact neutral');source=folder/'observer-on';feature=source/'source.vision.0.f32';meta=json.loads((source/'source.vision.0.json').read_text());require(meta['tokens']==case['image_tokens']and meta['embedding']==2560 and feature.stat().st_size==case['image_tokens']*2560*4,'actual source features');require(len(list(source.glob('source.vision.*.f32')))==1 and len(list(source.glob('source.logits.*.f32')))==2,'bounded source captures');sourcefeatures=np.fromfile(feature,dtype='<f4').reshape(-1,2560);sourcepos=np.fromfile(source/'source.vision.0.positions.u32',dtype='<u4').reshape(-1,4);report['cases'][name]['source_observer_neutral']=True
   expected=[]
   with up.open('rb')as f:
    for token in tokens:
     f.seek(embedding['data_offset']+token*10*210);packed=f.read(10*210);require(len(packed)==2100,'Q6 embedding row');expected.append(dequantize(np.frombuffer(packed,dtype=np.uint8),GGMLQuantizationType.Q6_K).reshape(-1).astype('<f4'))
   expected=np.asarray(expected,dtype='<f4');require(expected.shape==(len(tokens),2560),'source embedding rows')
   prepared.append((case,folder,original,retained,image,rgb,teacher,tokens,source,feature,sourcefeatures,sourcepos,expected,meta))
  save('all-three-source-controls-exact')
  for case,folder,original,retained,image,rgb,teacher,tokens,source,feature,sourcefeatures,sourcepos,expected,meta in prepared:
   name=case['id']
   for arm in cfg['arms']:
    require(memory()['MemAvailable']>=cfg['minimum_available_memory_bytes'],'per-arm RAM guard');dest=folder/arm;dest.mkdir();args=[oldnative if arm=='reference' else native,candidate,rgb,image.height,image.width,teacher,dest,cfg['cap'],4,'greedy-full']
    args+=['--upstream-vision',feature];armenv=dict(env);armenv['CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS']='1' if arm=='on' else '0'
    early_done=[False]
    def early_check():
     if early_done[0]:return
     scorepath=dest/'greedy.0.logits.f32'
     if not scorepath.exists()or scorepath.stat().st_size!=248320*4:return
     emb=np.fromfile(dest/'text.embedded.f32',dtype='<f4').reshape(-1,2560);pre=np.fromfile(dest/'prefill.input.f32',dtype='<f4').reshape(-1,2560);features_now=np.fromfile(dest/'vision.merged.f32',dtype='<f4').reshape(-1,2560);mods_now=np.array(list(map(int,(dest/'modalities.ids').read_text().split())));axes_now=np.fromfile(dest/'prefill.positions.f32',dtype='<f4').reshape(3,-1);wanted_now=expected.copy();wanted_now[mods_now==1]=sourcefeatures
     require(emb.tobytes()==expected.tobytes()and pre.tobytes()==wanted_now.tobytes()and features_now.tobytes()==sourcefeatures.tobytes(),'early actual same-feature embedding/prefill bytes');image_rows=np.flatnonzero(mods_now==1);require(np.array_equal(image_rows,np.arange(image_rows[0],image_rows[-1]+1))and meta['past']==int(image_rows[0]),'source image chunk begins at actual native prefix');expected_axes=np.empty((3,len(tokens)),dtype='<f4');expected_axes[:,:image_rows[0]]=np.arange(image_rows[0],dtype='<f4');expected_axes[:,image_rows]=sourcepos[:,[0,2,1]].T;suffix_start=int(image_rows[-1])+1;expected_axes[:,suffix_start:]=np.arange(meta['past']+meta['position_count'],meta['past']+meta['position_count']+len(tokens)-suffix_start,dtype='<f4');require(axes_now.tobytes()==expected_axes.tobytes(),'early complete source/native text-image mrope positions')
     if arm in ['reference','off']:
      required=['text.embedded.f32','prefill.input.f32','prefill.positions.f32','prefill.norm.f32','prefill.logits.f32','greedy.0.logits.f32']+['prefill.layer.'+str(i)+'.f32'for i in range(32)]
      if arm=='reference':
       baseline_files={rel:sha(dest/rel)for rel in required};report['reference_actual_stage_sha256']=baseline_files
      else:
       for rel in required:require(sha(dest/rel)==baseline_files[rel],'OFF actual regenerated baseline stage drift: '+rel)
       report['off_actual_baseline_stage_sha256']={rel:sha(dest/rel)for rel in required}
     report['cases'][name]['arms'][arm]={'status':'decoding','actual_input_bytes_exact':True,'first_score_vs_source':diff(np.fromfile(scorepath,dtype='<f4'),np.fromfile(source/'source.logits.0.f32',dtype='<f4'))};early_done[0]=True;save(name+'-'+arm+'-early-controls-exact')
    elapsed=command(name+'-'+arm,args,armenv,cfg['native_arm_timeout_seconds'],early_check);early_check();require(early_done[0],'early source input controls missing');summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf-8');actualids=list(map(int,(dest/'prompt.ids').read_text().split()));require(actualids==tokens and summary['prompt_tokens']==len(tokens),'native exact prompt tokens');mods=np.array(list(map(int,(dest/'modalities.ids').read_text().split())));require((mods==1).sum()==case['image_tokens'],'native modality shape');embedded=np.fromfile(dest/'text.embedded.f32',dtype='<f4').reshape(-1,2560);require(embedded.tobytes()==expected.tobytes(),'native embedding all rows bit exact independent source Q6 decoder');prefill=np.fromfile(dest/'prefill.input.f32',dtype='<f4').reshape(-1,2560);features=np.fromfile(dest/'vision.merged.f32',dtype='<f4').reshape(-1,2560);wanted=expected.copy();wanted[mods==1]=features;require(prefill.tobytes()==wanted.tobytes(),'actual prefill embedding splice bit exact');axes=np.fromfile(dest/'prefill.positions.f32',dtype='<f4').reshape(3,-1);require(np.array_equal(axes[:,mods==1].T,sourcepos[:,[0,2,1]])and np.all(sourcepos[:,3]==0),'actual source/native image mrope positions exact');require(ids and len(ids)<=cfg['cap'] and summary['eos']==(ids[-1]in[248044,248046]) and (summary['eos'] or len(ids)==cfg['cap']),'native EOG and cap closure')
    require(features.tobytes()==sourcefeatures.tobytes(),'source-fed actual feature bytes exact')
    scores=np.fromfile(dest/'greedy.0.logits.f32',dtype='<f4');gold=np.fromfile(source/'source.logits.0.f32',dtype='<f4');require(scores.size==gold.size==248320 and ids[0]==int(scores.argmax()),'actual first score argmax');report['cases'][name]['arms'][arm]={'ids':ids,'decoded':text,'eog_completed':summary['eos'],'cap_hit':not summary['eos']and len(ids)==cfg['cap'],'elapsed_process_seconds':elapsed,'peak_rss_bytes_observed':report['process_resources'][name+'-'+arm]['peak_rss_bytes_observed'],'source_text_weights_same':True,'source_vision_features_same':True,'source_sha':cfg['actual_baseline_source_sha'] if arm=='reference' else report['source_sha'],'physical_native_ops':arm=='on','actual_embedding_bytes_exact':True,'actual_image_positions_exact':True,'merged_feature_vs_source':diff(features,sourcefeatures),'first_score_vs_source':diff(scores,gold),'vs_retained_upstream':metric(ids,text,retained['tokens'],retained['content']),'vs_original_bf16':metric(ids,text,original['generated_ids'],original['decoded']),'summary':summary};save(name+'-'+arm+'-measured')
   report['cases'][name]['artifacts']={str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in folder.rglob('*')if p.is_file()};save(name+'-complete')
  require(sha(up)==pack.UP_SHA and sha(candidate)==report['candidate_sha256'],'sources immutable after inference');report['status']='measured';report['generated_candidate_retained']=False;candidate.unlink()
 except BaseException as error:
  report['error_type']=type(error).__name__;report['error']='Worker phase failed; retain phase logs and partial captures'
  raise
 finally:stop(active);save()
if __name__=='__main__':main()
