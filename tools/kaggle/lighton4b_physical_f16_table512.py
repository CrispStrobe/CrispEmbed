#!/usr/bin/env python3
"""Frozen extended table diagnostic. Cap256 evidence remains unchanged."""
import base64,csv,hashlib,io,json,os,resource,shutil,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
import importlib.util
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=module('full_form',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py');require=f.require;sha=f.sha
FORM='0a0c2084baa4e46aa80f09f7ae94fd4b1c56ccae2bbf2f9ef6210e81e8cf1d01';PHYSICAL='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2';CALIB='7639d6559da7cc58443ccfbe1dac518af0d70f3e7a83718d037a97bb147eb2b7'
def admit(cfg):
 require(cfg['schema']=='lighton4b-physical-f16-table512/v1'and cfg['case']=='simple_table'and cfg['cap']==512,'closed extended case/cap')
 require(cfg['form_receipt_sha256']==FORM and cfg['physical_receipt_sha256']==PHYSICAL and cfg['calibration_receipt_sha256']==CALIB,'immutable three receipts')
 require(cfg['threads']==4 and cfg['context']==2048 and cfg['total_timeout_seconds']==3600 and cfg['per_arm_timeout_seconds']==1200,'bounded execution')
 require(cfg['arms']==['upstream-off','upstream-on','native'],'fixed observer controls then native')
 require(cfg['driver_source_sha256']==sha(ROOT/'tests/lighton4b_native_document.cpp'),'explicit test-driver-only source change')
def remaining_budget(started,arm_deadline=None,now=None):
 now=time.monotonic()if now is None else now
 remaining=3600-(now-started)
 if arm_deadline is not None:remaining=min(remaining,arm_deadline-now)
 require(remaining>0,'source arm/whole-worker deadline')
 return remaining
def validate_source(response,old):
 require(not response['truncated']and response['tokens_evaluated']==86 and response['stop']and response['tokens'][:256]==old['tokens']and response['completion_probabilities'][:256]==old['completion_probabilities'],'exact old source256 prefix/probabilities')
 require(1<=len(response['tokens'])<=512 and len(response['completion_probabilities'])==len(response['tokens'])and response['stop_type']in['eos','limit']and(response['stop_type']=='eos')==(response['tokens'][-1]in[248044,248046])and(response['stop_type']!='limit'or len(response['tokens'])==512),'source EOS/cap closure')
def driver_scope_controls(native,work,out,loader):
 results=[];env=dict(os.environ);env['LD_LIBRARY_PATH']=loader
 base=[str(native),str(work/'absent.gguf'),str(work/'absent.rgb'),'1','1',str(work/'absent.ids'),str(out),'256','4','greedy-full','--upstream-vision',str(work/'absent.feature')]
 for label,cap,threads,mode,flag,expected in [('legacy256',256,4,'greedy-full',None,4),('legacy257',257,4,'greedy-full',None,2),('legacy512',512,4,'greedy-full',None,2),('extended512',512,4,'greedy-full','--extended-table-cap512',4),('extended513',513,4,'greedy-full','--extended-table-cap512',2),('wrongthreads',512,2,'greedy-full','--extended-table-cap512',2),('wrongmode',512,4,'full','--extended-table-cap512',2),('wrongflag',512,4,'greedy-full','--unknown',2),('extended256',256,4,'greedy-full','--extended-table-cap512',2)]:
  args=list(base);args[7]=str(cap);args[8]=str(threads);args[9]=mode
  if flag:args.append(flag)
  with(out/('scope-'+label+'.log')).open('w')as log:code=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=10).returncode
  require(code==expected,'actual reference CLI scope '+label);results.append({'case':label,'returncode':code})
 return results
def main():
 import numpy as np
 from PIL import Image
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only');require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','serializer versions')
 cfgpath=ROOT/'docs/lighton4b-physical-f16-table512-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'table512';work.mkdir();started=time.monotonic();active=None;source_deadline=[None];deadline=Deadline(3600);deadline.__enter__()
 base=module('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack;import lighton4b_upstream_observer_patch as observer;import lighton4b_upstream_dispatch_log as dispatch
 r={'schema':cfg['schema'],'status':'failed','source_sha':None,'arms':{},'commands':{},'quality_qualified':False,'original_bf16_forward_executed':False,'cap256_evidence_replaced':False}
 def save():(out/'physical-f16-table512.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def interrupted(signum,frame):base.stop(active);raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 def run(label,args,env=None,timeout=600,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(timeout,3600-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'one exact receipt '+name);return rows[0]
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'artifact '+p.name)
 def api(port,path,data=None,timeout=60):
  body=None if data is None else json.dumps(data).encode();req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=body,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=min(timeout,remaining_budget(started,source_deadline[0])))as response:b=response.read(8*1024**2+1)
  require(len(b)<=8<<20,'bounded source response');return json.loads(b)
 try:
  run('source-pin',['git','rev-parse','HEAD']);r['source_sha']=(out/'source-pin.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source');run('host-cpu',['lscpu'])
  r['preflight']={'memory':base.memory(),'scratch_free':shutil.disk_usage(work).free,'output_free':shutil.disk_usage(out).free};require(r['preflight']['memory']['MemAvailable']>=10<<30 and r['preflight']['scratch_free']>=25<<30 and r['preflight']['output_free']>=4<<30,'RAM/disk before transfer')
  fp=locate('physical-f16-full-form.json',FORM);form=json.loads(fp.read_text());pp=locate('physical-decode.json',PHYSICAL);physical=json.loads(pp.read_text());cp=locate('physical-f16-calibration.json',CALIB);calib=json.loads(cp.read_text());require(form['status']=='measured_full_form'and physical['status']=='measured'and calib['status']=='measured_two_case_calibration','complete predecessor receipts');require(calib['binary_sha256']==form['binary_sha256']and calib['gates']==form['arms']['f16-full']['gates']and calib['candidate_sha256']==form['candidate_sha256'],'identical preceding configuration')
  source_bundle=fp.parent/'runtime'
  for rel,row in form['runtime_bundle'].items():verify(source_bundle/rel,row)
  r['original_form_binary_sha256']=form['binary_sha256'];r['gates']=form['arms']['f16-full']['gates'];r['predecessor_receipts']={'form':FORM,'physical':PHYSICAL,'calibration':CALIB}
  run('fetch-form-source',['git','fetch','--depth','1','origin',form['source_sha']]);run('unchanged-production-source',['git','diff','--exit-code',form['source_sha'],'HEAD','--','src','ggml','CMakeLists.txt'])
  old_driver=subprocess.check_output(['git','show',form['source_sha']+':tests/lighton4b_native_document.cpp']);require(hashlib.sha256(old_driver).hexdigest()==cfg['original_driver_source_sha256'],'exact original cap256 driver')
  run('native-submodule',['git','submodule','update','--init','ggml']);nb=work/'native-build';nf={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
  run('native-configure',['cmake','-S',ROOT,'-B',nb,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in nf.items()]]);nc=(nb/'CMakeCache.txt').read_text().splitlines()
  for k,v in nf.items():require(f'{k}:BOOL={v}'in nc,'unchanged native profile '+k)
  shutil.copy(nb/'CMakeCache.txt',out/'native-CMakeCache.txt');run('native-build',['cmake','--build',nb,'--target','lighton4b-native-document','--parallel','2'],timeout=1200);native=nb/'lighton4b-native-document';r['native_binary_sha256']=sha(native);r['test_driver_source_sha256']=sha(ROOT/'tests/lighton4b_native_document.cpp')
  libraries=sorted(p for p in nb.rglob('libggml*.so*')if p.is_file());require(libraries,'actual native CPU libraries');native_loader=':'.join(sorted({str(p.parent)for p in libraries}));bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin'/native.name)
  for library in libraries:shutil.copy2(library,bundle/'lib'/library.name)
  r['rebuilt_runtime_bundle']={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in bundle.rglob('*')if p.is_file()}
  r['scope_controls']=driver_scope_controls(native,work,out,native_loader)
  protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());case=next(x for x in protocol['cases']if x['id']=='simple_table');folder=pp.parent/'simple_table';art=physical['cases']['simple_table']['artifacts']
  for rel in ['input.png','input.rgb','original.json','retained-upstream-response.json','upstream_features/prompt.ids','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32','observer-on/source.logits.1.f32']:verify(folder/rel,art[rel])
  original=json.loads((folder/'original.json').read_text());old_source=json.loads((folder/'retained-upstream-response.json').read_text());old_native=calib['cases']['simple_table'];require(len(old_source['tokens'])==len(old_native['generated_ids'])==256 and not old_native['summary']['eos']and old_source['stop_type']=='limit','retain actual old cap256 negative')
  require(sha(folder/'original.json')==case['reference_sha256']and sha(folder/'retained-upstream-response.json')==case['upstream_response_sha256'],'reference hashes');image=Image.open(ROOT/case['image']).convert('RGB');require(sha(ROOT/case['image'])==case['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'frozen image/RGB');prompt_ids=list(map(int,(folder/'upstream_features/prompt.ids').read_text().split()));b=io.BytesIO();np.save(b,np.array([prompt_ids],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(b.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'actual prompt NPY bytes')
  repo=work/'llama.cpp'
  for label,args in [('up-init',['git','init',repo]),('up-remote',['git','-C',repo,'remote','add','origin',protocol['upstream_repo']]),('up-fetch',['git','-C',repo,'fetch','--depth','1','origin',protocol['upstream_commit']]),('up-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==observer.SOURCE==protocol['upstream_commit'],'pinned upstream');r['observer_patch']=observer.patch(repo);r['dispatch_patch']=dispatch.patch(repo)
  build=work/'up-build';flags={'GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF','GGML_LLAMAFILE':'ON','GGML_OPENMP':'ON','GGML_BACKEND_DL':'OFF'}
  run('up-configure',['cmake','-S',repo,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF',*[f'-D{k}={v}'for k,v in flags.items()]]);cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in flags.items():require(f'{k}:BOOL={v}'in cache,'source profile '+k)
  shutil.copy(build/'CMakeCache.txt',out/'upstream-CMakeCache.txt');run('up-build',['cmake','--build',build,'--target','llama-server','--parallel','2'],timeout=1200);server=build/'bin/llama-server';r['upstream_binary_sha256']=sha(server)
  prefix='https://huggingface.co/'+protocol['model_repo']+'/resolve/'+protocol['model_revision']+'/';up=work/protocol['models']['Q4_K_M']['file'];proj=work/protocol['projector']['file']
  for path,row in [(up,protocol['models']['Q4_K_M']),(proj,protocol['projector'])]:r['phase']='download-'+path.name;save();base.download(prefix+path.name,path,row['bytes'],row['sha256'])
  row=protocol['base_model'];protected=work/row['file'];base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'physical-native.gguf';audit=pack.convert(protected,up,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==form['candidate_sha256'],'immutable same native weights');r['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
  responses=[]
  for enabled in [False,True]:
   arm='upstream-on'if enabled else'upstream-off';dest=out/arm;dest.mkdir();r['phase']=arm;save();child=dict(os.environ);child.pop(observer.ENV,None);child[dispatch.ENV]=str(dest)
   if enabled:child[observer.ENV]=str(dest)
   with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   args=[server,'-m',up,'--mmproj',proj,'--host','127.0.0.1','--port',str(port),'-t','4','-tb','4','-c','2048','-np','1','-ngl','0','--no-mmproj-offload','--no-warmup','--flash-attn','off','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--image-min-tokens',str(protocol['image_min_tokens']),'--image-max-tokens',str(protocol['image_max_tokens'])];arm_started=time.monotonic();source_deadline[0]=arm_started+1200;peak=0;before=resource.getrusage(resource.RUSAGE_CHILDREN);r['arms'][arm]={'status':'running'};save()
   with(dest/'server.log').open('w')as log:
    active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=child,start_new_session=True)
    try:
     for _ in range(180):
      require(active.poll()is None,'source process alive')
      try:api(port,'/health');break
      except Exception:time.sleep(1)
     else:raise RuntimeError('source ready deadline')
     marker=api(port,'/props').get('media_marker');require(isinstance(marker,str)and 1<=len(marker)<=256,'actual media marker');templ=api(port,'/apply-template',{'messages':[{'role':'user','content':marker}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt'];require(templ.count(marker)==1 and templ.replace(marker,'<__media__>')==protocol['expected_media_prompt'],'same wrapper');expanded=templ.replace(marker,'<|vision_start|>'+'<|image_pad|>'*case['image_tokens']+'<|vision_end|>');require(api(port,'/tokenize',{'content':expanded,'add_special':False,'parse_special':True})['tokens']==prompt_ids,'exact source prompt IDs')
     req={'prompt':{'prompt_string':templ,'multimodal_data':[base64.b64encode((folder/'input.png').read_bytes()).decode()]},'n_predict':512,'temperature':0,'seed':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'cache_prompt':False,'return_tokens':True,'n_probs':5,'stream':False};response=api(port,'/completion',req,1200);(dest/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False)+'\n');validate_source(response,old_source);remaining_budget(started,source_deadline[0]);(dest/'generated.ids').write_text(''.join(str(x)+'\n'for x in response['tokens']));(dest/'generated.txt').write_bytes(response['content'].encode('utf8'))
     try:peak=int(next(x.split()[1]for x in Path('/proc/'+str(active.pid)+'/status').read_text().splitlines()if x.startswith('VmHWM:')))*1024
     except(OSError,StopIteration):pass
     dispatch_rows=list(csv.DictReader((dest/dispatch.OUTPUT).read_text().splitlines(),delimiter='\t'));require(len(dispatch_rows)==9 and len({row['role']for row in dispatch_rows})==9 and all(row['extra']in['0','1']and row['host']in['0','1']for row in dispatch_rows),'actual loaded tensor dispatch rows');responses.append(response);r['arms'][arm]={'status':'completed','tensor_dispatch':dispatch_rows,'response_sha256':sha(dest/'response.json'),'generated_tokens':len(response['tokens']),'eos':response['stop_type']=='eos','cap_hit':response['stop_type']=='limit','elapsed_seconds':time.monotonic()-arm_started,'peak_rss_bytes_observed':peak,'vs_original_bf16':f.metric(base,response['tokens'],response['content'],original['generated_ids'],original['decoded'])};save()
    finally:
     base.stop(active);r['arms'][arm]['returncode']=active.returncode if active is not None else None;active=None;source_deadline[0]=None;r['arms'][arm]['elapsed_seconds']=time.monotonic()-arm_started;after=resource.getrusage(resource.RUSAGE_CHILDREN);r['arms'][arm]['child_user_seconds']=after.ru_utime-before.ru_utime;r['arms'][arm]['child_system_seconds']=after.ru_stime-before.ru_stime
     if r['arms'][arm]['status']=='running':r['arms'][arm]['status']='failed'
     save()
  require((out/'upstream-off'/dispatch.OUTPUT).read_bytes()==(out/'upstream-on'/dispatch.OUTPUT).read_bytes(),'same actual source dispatch with observer OFF/ON')
  require(responses[0]['tokens']==responses[1]['tokens']and responses[0]['content']==responses[1]['content']and responses[0]['completion_probabilities']==responses[1]['completion_probabilities']and responses[0]['stop_type']==responses[1]['stop_type'],'full source observer neutral')
  for rel in ['source.vision.0.f32','source.logits.0.f32','source.logits.1.f32']:require(sha(out/'upstream-on'/rel)==art['observer-on/'+rel]['sha256'],'source feature/first2 raw score anchor '+rel)
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(form['arms']['f16-full']['gates']);env['LD_LIBRARY_PATH']=native_loader;teacher=work/'one.ids';teacher.write_text(str(original['generated_ids'][0])+'\n');dest=out/'native';dest.mkdir();checked=[0]
  def native_tick():
   try:completed=int(json.loads((dest/'generated-progress.json').read_text())['generated_tokens'])
   except(OSError,ValueError,KeyError):return
   # Progress for token j+2 proves row j was closed before the next decode step.
   while checked[0]<min(256,completed-1):
    j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==calib['artifacts'][f'simple_table/greedy.{j}.logits.f32']['sha256'],'early old native raw score anchor');checked[0]+=1
   r['native_old256_raw_checked_during_run']=checked[0];save()
  run('native',[native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,512,4,'greedy-full','--upstream-vision',out/'upstream-on/source.vision.0.f32','--extended-table-cap512'],env,1200,native_tick)
  summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf8');require(summary['eos']==(ids[-1]in[248044,248046])and(len(ids)==512 or summary['eos'])and ids[:256]==old_native['generated_ids'],'native extended EOS/old256 prefix');
  for j in range(256):require(sha(dest/f'greedy.{j}.logits.f32')==calib['artifacts'][f'simple_table/greedy.{j}.logits.f32']['sha256'],'all256 native raw score anchors')
  for rel in f.STAGES:require(sha(dest/rel)==calib['artifacts']['simple_table/'+rel]['sha256'],'all38 native prefill stage anchors')
  require(sha(dest/'prefill.input.f32')==art['upstream_features/prefill.input.f32']['sha256']and sha(dest/'prefill.positions.f32')==art['upstream_features/prefill.positions.f32']['sha256'],'same actual inputs/positions')
  for j,token in enumerate(ids):
   row=np.fromfile(dest/f'greedy.{j}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and int(row.argmax())==token,'finite exact greedy argmax')
  r['arms']['native']={'summary':summary,'generated_ids':ids,'decoded':text,'vs_original_bf16':f.metric(base,ids,text,original['generated_ids'],original['decoded']),'vs_extended_upstream':f.metric(base,ids,text,responses[0]['tokens'],responses[0]['content']),'resources':r['commands']['native']};r['source_observer_full_neutral']=True;r['old_cap256_prefix_anchors_exact']=True;require(sha(candidate)==form['candidate_sha256'],'unchanged weights');candidate.unlink();r['status']='measured_extended_table'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  base.stop(active);deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for arm in cfg['arms']for p in(out/arm).rglob('*')if p.is_file()};save()
if __name__=='__main__':main()
