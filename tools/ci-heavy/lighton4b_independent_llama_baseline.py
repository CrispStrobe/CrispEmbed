#!/usr/bin/env python3
"""Pinned independent upstream CPU baseline; no original-model forward or local model access."""
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
def main():
 provider='kaggle-cpu'if Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()else'github-actions'if os.environ.get('GITHUB_ACTIONS')=='true'else None
 require(provider is not None and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','authorized hosted portable only')
 import numpy as np
 from PIL import Image
 require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','input serializer pins')
 protocolpath=ROOT/'docs/lighton4b-independent-llama-baseline-v1.json';cfg=json.loads(protocolpath.read_text());out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'independent-llama';scratch.mkdir();report={'schema':'lighton4b-independent-llama-result/v1','status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(protocolpath),'quality_qualified':False,'original_forward_executed':False,'provider':provider,'preflight':{'memory':memory(),'load':os.getloadavg(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free},'arms':{},'limits':cfg['limits']};active=None;started=time.monotonic()
 def save(phase):report['phase']=phase;(out/'independent-llama.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print('phase',phase,flush=True)
 def interrupted(signum,frame):stop(active);raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(cfg['total_model_timeout_seconds'])
 def command(label,args,timeout=1200):
  nonlocal active
  save(label)
  with(out/(label+'.log')).open('w')as f:
   active=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=active.wait(timeout=timeout);require(code==0,label+' failed')
   finally:stop(active);active=None
 def api(port,path,data=None,timeout=60):
  body=None if data is None else json.dumps(data).encode();req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=body,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=timeout)as r:raw=r.read(8*1024**2+1)
  require(len(raw)<=8*1024**2,'bounded API response');return json.loads(raw)
 try:
  require(report['preflight']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes'],'available memory guard');require(report['preflight']['scratch_free_bytes']>=cfg['minimum_scratch_free_bytes']and report['preflight']['output_free_bytes']>=cfg['minimum_output_free_bytes'],'disk guards')
  refs={};images={}
  for case in cfg['cases']:
   p=ROOT/case['image'];require(sha(p)==case['image_sha256'],'frozen input bytes');image=Image.open(p).convert('RGB');canonical=Image.frombytes('RGB',image.size,image.tobytes());path=out/(case['id']+'.png');canonical.save(path,compress_level=6);images[case['id']]=path
   refpath=ROOT/case['reference_path']if case['reference_path']else out/(case['id']+'-original.json')
   if case['reference_url']:
    with urllib.request.urlopen(case['reference_url'],timeout=60)as r:raw=r.read(4*1024**2+1)
    require(len(raw)<=4*1024**2,'source receipt bound');refpath.write_bytes(raw)
   require(sha(refpath)==case['reference_sha256'],'original capture authentication');ref=json.loads(refpath.read_text());require(len(ref['generated_ids'])==case['reference_generated_ids']and ref['eos_completed']==case['reference_eos']and ref['cap_hit']==case['reference_cap'],'original completion identity');refs[case['id']]=ref
  repo=scratch/'llama.cpp';command('upstream-init',['git','init',repo]);command('upstream-remote',['git','-C',repo,'remote','add','origin',cfg['upstream_repo']]);command('upstream-fetch',['git','-C',repo,'fetch','--depth','1','origin',cfg['upstream_commit']]);command('upstream-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD']);require(subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'upstream SHA')
  build=scratch/'build';options=['-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF','-DGGML_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'];report['cmake_options']=options;command('configure',['cmake','-S',repo,'-B',build,*options]);command('build',['cmake','--build',build,'--target','llama-server','-j','4'],1800);binary=build/'bin/llama-server';report['binary_sha256']=sha(binary);shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt');command('upstream-version',[binary,'--version']);command('host-cpu',['lscpu']);command('host-packages',['python','-m','pip','freeze'])
  prefix='https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/';proj=scratch/cfg['projector']['file'];save('download-projector');download(prefix+proj.name,proj,cfg['projector']['bytes'],cfg['projector']['sha256'])
  for quant,model in cfg['models'].items():
   modelpath=scratch/model['file'];save('download-'+quant);download(prefix+modelpath.name,modelpath,model['bytes'],model['sha256']);report['arms'][quant]={}
   for case in cfg['cases']:
    require(time.monotonic()-started<cfg['total_model_timeout_seconds'],'total deadline');require(memory()['MemAvailable']>=7*1024**3,'per-arm available memory guard');name=case['id'];folder=out/(quant+'-'+name);folder.mkdir();save('run-'+quant+'-'+name)
    with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    args=[str(binary),'-m',str(modelpath),'--mmproj',str(proj),'--host','127.0.0.1','--port',str(port),'-t',str(cfg['threads']),'-tb',str(cfg['threads']),'-c',str(cfg['context']),'-np','1','-ngl','0','--no-mmproj-offload','--no-warmup','--flash-attn','off','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--image-min-tokens',str(cfg['image_min_tokens']),'--image-max-tokens',str(cfg['image_max_tokens'])]
    (folder/'server-command.json').write_text(json.dumps(args,indent=2)+'\n');start=time.monotonic()
    with(folder/'server.log').open('w')as log:
     active=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:
      ready=False
      for i in range(180):
       require(active.poll()is None,'server exited during load')
       try:api(port,'/health');ready=True;break
       except Exception:time.sleep(1)
      require(ready,'server ready deadline');models=api(port,'/models');require(any('multimodal'in x.get('capabilities',[])for x in models['models']),'actual multimodal model capability');(folder/'models.json').write_text(json.dumps(models,indent=2)+'\n');props=api(port,'/props');(folder/'props.json').write_text(json.dumps(props,indent=2)+'\n')
      marker=props.get('media_marker');require(isinstance(marker,str)and 1<=len(marker)<=256,'actual server transport marker');templ=api(port,'/apply-template',{'messages':[{'role':'user','content':marker}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}});(folder/'template.json').write_text(json.dumps(templ,indent=2)+'\n');require(templ['prompt'].count(marker)==1 and templ['prompt'].replace(marker,'<__media__>')==cfg['expected_media_prompt'],'upstream chat wrapper differs from frozen source')
      expanded=templ['prompt'].replace(marker,'<|vision_start|>'+'<|image_pad|>'*case['image_tokens']+'<|vision_end|>');tokens=api(port,'/tokenize',{'content':expanded,'add_special':False,'parse_special':True})['tokens'];idsbuf=io.BytesIO();np.save(idsbuf,np.array([tokens],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(idsbuf.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'frozen original prompt ID bytes');(folder/'expected-prompt.ids').write_text(''.join(str(x)+'\n'for x in tokens))
      request={'prompt':{'prompt_string':templ['prompt'],'multimodal_data':[base64.b64encode(images[name].read_bytes()).decode()]},'n_predict':cfg['cap'],'temperature':0,'seed':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'cache_prompt':False,'return_tokens':True,'n_probs':5,'stream':False};(folder/'request-settings.json').write_text(json.dumps({k:v for k,v in request.items()if k!='prompt'},indent=2)+'\n');response=api(port,'/completion',request,cfg['request_timeout_seconds']);(folder/'response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n');require(not response.get('truncated')and response['tokens_evaluated']==case['expected_prompt_tokens'],'actual upstream image/prompt token geometry differs');require(isinstance(response['tokens'],list)and len(response['tokens'])<=cfg['cap'],'raw generated IDs');require(response['stop_type']in ('eos','limit'),'explicit stopping classification');text=response['content'];ids=response['tokens'];(folder/'generated.txt').write_text(text);(folder/'generated.ids').write_text(''.join(str(x)+'\n'for x in ids));gold=refs[name];prefixids=0
      for x,y in zip(ids,gold['generated_ids']):
       if x!=y:break
       prefixids+=1
      process_memory={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/'+str(active.pid)+'/status').read_text().splitlines())if k in ('VmHWM','VmRSS')};report['arms'][quant][name]={'native_peak_rss_bytes':process_memory['VmHWM'],'native_final_rss_bytes':process_memory['VmRSS'],'status':'completed','decoded':text,'ids':ids,'stop_type':response['stop_type'],'eos_completed':response['stop_type']=='eos','cap_hit':response['stop_type']=='limit','eos_present_in_returned_ids':248046 in ids,'timings':response.get('timings'),'elapsed_with_load_seconds':time.monotonic()-start,'input_tokens':response['tokens_evaluated'],'prompt_ids_exact_original':True,'vision_bytes_parity_proven':False,'matching_reference_prefix_ids':prefixids,'raw_ids_exact_reference':ids==gold['generated_ids'],'decoded_exact_reference':text==gold['decoded'],'character_edits':edits(gold['decoded'],text),'reference_codepoints':len(gold['decoded']),'word_edits':edits(gold['decoded'].split(),text.split()),'reference_words':len(gold['decoded'].split()),'reference_eos':gold['eos_completed'],'reference_cap':gold['cap_hit'],'response_sha256':sha(folder/'response.json'),'text_sha256':sha(folder/'generated.txt'),'ids_sha256':sha(folder/'generated.ids')};save('completed-'+quant+'-'+name)
     finally:stop(active);active=None
   # Delete only this task-owned authenticated model after all three responses.
   modelpath.unlink()
  report['status']='completed'
 finally:stop(active);save(report.get('phase','failed'))
if __name__=='__main__':main()
