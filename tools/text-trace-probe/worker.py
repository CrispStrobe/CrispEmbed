#!/usr/bin/env python3
"""Pinned upstream-only neutral text trace; retained native arrays are read-only."""
import base64,csv,hashlib,importlib.util,json,os,platform,shutil,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command
sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_observer_patch as observer;import lighton4b_upstream_dispatch_log as dispatch
sys.path.insert(0,str(Path(__file__).parent));import source_patch,analysis
TABLE='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3'
CALIB='f69d23ffe6d65d9abf6c7c5c660723613ce84cc901383c5151e17d8f8dc5fe64'
PHYSICAL='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('physical',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');require=base.require;sha=base.sha
PROFILE={'BUILD_SHARED_LIBS':'ON','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_CPU_REPACK':'ON','GGML_OPENMP':'ON','GGML_LLAMAFILE':'ON','GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
def validate_response(response,frozen):
 require(response['tokens']==frozen['tokens']and response['content']==frozen['content']and response['completion_probabilities']==frozen['completion_probabilities'],'all actual282 source IDs/text/probability records unchanged')
 require(len(response['tokens'])==282 and response['tokens_evaluated']==86 and response['stop']and response['stop_type']=='eos'and not response['truncated']and response['tokens'][-1]==248046,'independent complete source EOS/prompt')
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64']and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU/array profile')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'text-trace';work.mkdir();started=time.monotonic();active=None;arm_end=[None];deadline=Deadline(3600);deadline.__enter__()
 report={'schema':'lighton4b-upstream-text-trace/v1','status':'failed','commands':{},'arms':{},'native_forward_executed':False,'original_bf16_forward_executed':False,'f32_model_instantiated':False,'quality_qualified':False}
 def save():(out/'text-trace.json').write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 def remaining():
  n=3600-(time.monotonic()-started)
  if arm_end[0]is not None:n=min(n,arm_end[0]-time.monotonic())
  require(n>0,'bounded whole/source arm deadline');return n
 def run(label,args,limit=900):
  report['phase']=label;report['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,remaining()),report['commands'][label],cwd=ROOT)
  finally:save()
 def stop_signal(sig,frame):base.stop(active);raise SystemExit(128+sig)
 prior={s:signal.signal(s,stop_signal)for s in[signal.SIGTERM,signal.SIGINT]}
 def locate(name,digest):
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(matches)==1,'one immutable '+name);return matches[0]
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'actual complete artifact '+p.name)
 def api(port,path,data=None):
  request=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=min(remaining(),900 if path=='/completion'else 30))as f:raw=f.read((8<<20)+1)
  require(len(raw)<=8<<20,'bounded source response');return json.loads(raw)
 try:
  run('source',['git','rev-parse','HEAD']);report['source_sha']=(out/'source.log').read_text().strip();require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source');run('cpu',['lscpu']);flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual private CPU features')
  report['capacity']={'memory':base.memory(),'scratch_free':shutil.disk_usage(work).free,'output_free':shutil.disk_usage(out).free};require(report['capacity']['memory']['MemAvailable']>=10<<30 and report['capacity']['scratch_free']>=12<<30 and report['capacity']['output_free']>=4<<30,'bounded reference-only capacity')
  tp=locate('physical-f16-table512.json',TABLE);table=json.loads(tp.read_text());require(table['status']=='measured_extended_table'and table['source_sha']=='f789609c2f1ff1a9527c8d02545d096ed10446b0'and table['source_observer_full_neutral']and table['old_cap256_prefix_anchors_exact'],'completed actual table')
  cp=locate('composed-vision-calibration.json',CALIB);calib=json.loads(cp.read_text());require(calib['status']=='measured_composed_vision_calibration'and calib['cases']['simple_table']['composed_matches_source_feature_parity'],'same actual corrected feature/text context');report['calibration_source_sha']=calib['source_sha'];require(calib['all_composed_merged_byte_exact']and all(calib['gates'][k]==v for k,v in {'CRISPEMBED_LIGHTON4B_HEAD_BF16':'0','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK':'0','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION':'1','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION':'1','CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE':'1'}.items()),'frozen corrected configuration')
  pp=locate('physical-decode.json',PHYSICAL);physical=json.loads(pp.read_text());require(physical['status']=='measured','actual raw source input');folder=pp.parent/'simple_table';art=physical['cases']['simple_table']['artifacts']
  for name in['input.png','input.rgb','upstream_features/prompt.ids']:verify(folder/name,art[name])
  response_path=tp.parent/'upstream-off/response.json';verify(response_path,table['artifacts']['upstream-off/response.json']);frozen=json.loads(response_path.read_text());validate_response(frozen,frozen);native_ids=table['arms']['native']['generated_ids'];require(native_ids[:126]==frozen['tokens'][:126]and native_ids[126]!=frozen['tokens'][126],'predeclared first actual same-prefix flip126')
  native=tp.parent/'native'
  for name,row in table['artifacts'].items():
   if name.startswith('native/prefill.')or name in ['native/greedy.126.logits.f32']or name.startswith('upstream-on/source.logits.')or name in ['upstream-on/source.vision.0.f32','upstream-on/'+dispatch.OUTPUT]:verify(tp.parent/name,row)
  protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());case=next(x for x in protocol['cases']if x['id']=='simple_table');prompt=[int(x)for x in(folder/'upstream_features/prompt.ids').read_text().split()];require(len(prompt)==86 and sha(ROOT/case['image'])==case['image_sha256']and sha(folder/'input.png')==case['image_sha256'],'same immutable PNG/prompt');require(protocol['upstream_commit']==source_patch.SOURCE==observer.SOURCE and protocol['model_revision']=='8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01','pinned source/checkpoint')
  repo=work/'upstream'
  for label,args in [('init',['git','init',repo]),('remote',['git','-C',repo,'remote','add','origin',protocol['upstream_repo']]),('fetch',['git','-C',repo,'fetch','--depth','1','origin',source_patch.SOURCE]),('checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==source_patch.SOURCE,'exact upstream');report['feature_sampler_patch']=observer.patch(repo);report['dispatch_patch']=dispatch.patch(repo);report['text_live_patch']=source_patch.patch(repo);save()
  build=work/'build';args=['cmake','-S',repo,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF']+[f'-D{k}={v}'for k,v in PROFILE.items()];run('configure',args);cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in PROFILE.items():require(f'{k}:BOOL={v}'in cache,'exact actual upstream profile '+k)
  shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt');run('build',['cmake','--build',build,'--target','llama-server','--parallel','2'],1200);server=build/'bin/llama-server';run('loader-version',[server,'--version'],30);report['source_binary_sha256']=sha(server)
  prefix='https://huggingface.co/'+protocol['model_repo']+'/resolve/'+protocol['model_revision']+'/';model=work/protocol['models']['Q4_K_M']['file'];projector=work/protocol['projector']['file']
  for path,row in[(model,protocol['models']['Q4_K_M']),(projector,protocol['projector'])]:report['phase']='acquire-'+path.name;save();base.download(prefix+path.name,path,row['bytes'],row['sha256'])
  responses=[]
  for arm in['off','on']:
   dest=out/arm;dest.mkdir();child=dict(os.environ);child[observer.ENV]=str(dest);child[dispatch.ENV]=str(dest);child.pop(source_patch.ENV,None);child.pop(source_patch.DONE,None)
   if arm=='on':child[source_patch.ENV]=str(dest)
   with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   args=[server,'-m',model,'--mmproj',projector,'--host','127.0.0.1','--port',str(port),'-t','4','-tb','4','-c','2048','-np','1','-ngl','0','--no-mmproj-offload','--no-warmup','--flash-attn','off','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--image-min-tokens',str(protocol['image_min_tokens']),'--image-max-tokens',str(protocol['image_max_tokens'])];report['phase']=arm;report['arms'][arm]={'status':'running'};save();arm_started=time.monotonic();arm_end[0]=arm_started+900
   with(dest/'server.log').open('w')as log:
    active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=child,start_new_session=True)
    try:
     for _ in range(180):
      remaining();require(active.poll()is None,'source alive')
      try:api(port,'/health');break
      except (OSError,urllib.error.URLError):time.sleep(min(1,remaining()))
     else:raise RuntimeError('source health deadline')
     marker=api(port,'/props').get('media_marker');require(isinstance(marker,str)and 1<=len(marker)<=256,'source media marker');template=api(port,'/apply-template',{'messages':[{'role':'user','content':marker}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt'];require(template.count(marker)==1 and template.replace(marker,'<__media__>')==protocol['expected_media_prompt'],'same wrapper');expanded=template.replace(marker,'<|vision_start|>'+'<|image_pad|>'*case['image_tokens']+'<|vision_end|>');require(api(port,'/tokenize',{'content':expanded,'add_special':False,'parse_special':True})['tokens']==prompt,'same actual source prompt')
     request={'prompt':{'prompt_string':template,'multimodal_data':[base64.b64encode((folder/'input.png').read_bytes()).decode()]},'n_predict':512,'temperature':0,'seed':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'cache_prompt':False,'return_tokens':True,'n_probs':5,'stream':False};response=api(port,'/completion',request);validate_response(response,frozen);responses.append(response);(dest/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False)+'\n');(dest/'generated.txt').write_bytes(response['content'].encode());(dest/'generated.ids').write_text(''.join(str(x)+'\n'for x in response['tokens']))
     for i in range(282):
      row_path=dest/f'source.logits.{i}.f32';verify(row_path,table['artifacts'][f'upstream-on/source.logits.{i}.f32']);values=np.fromfile(row_path,dtype='<f4');require(values.size==248320 and np.isfinite(values).all()and int(values.argmax())==response['tokens'][i],'finite actual full-vocabulary row/greedy argmax')
     verify(dest/'source.vision.0.f32',table['artifacts']['upstream-on/source.vision.0.f32']);require((dest/dispatch.OUTPUT).read_bytes()==(tp.parent/'upstream-on'/dispatch.OUTPUT).read_bytes(),'actual quant buffer dispatch unchanged');report['arms'][arm].update(status='source-frozen-exact',all282_raw_scores_exact=True,all282_ids_text_probability_records_exact=True,source_features_exact=True,generated_ids=response['tokens'],decoded=response['content'],eos=True)
    finally:
     if active is not None and Path(f'/proc/{active.pid}/status').is_file():report['arms'][arm]['process_status_before_stop']={x.split(':',1)[0]:x.split(':',1)[1].strip()for x in Path(f'/proc/{active.pid}/status').read_text().splitlines()if x.startswith(('VmHWM:','VmRSS:','Threads:'))}
     base.stop(active);report['arms'][arm]['returncode']=active.returncode if active is not None else None;active=None;arm_end[0]=None;report['arms'][arm]['seconds']=time.monotonic()-arm_started;save()
  require(responses[0]['tokens']==responses[1]['tokens']and responses[0]['completion_probabilities']==responses[1]['completion_probabilities'],'complete live observer neutrality');report['all282_scores_observer_neutral']=True;report['analysis']=analysis.analyze(out/'on',native,out/'on/source.logits.0.f32',out/'on/source.logits.126.f32',frozen['tokens']);report['status']='measured_neutral_text_trace';report['limits']=['One observed table and same-prefix step126; no heldout/general quality claim.','Stored-quantized actual upstream versus immutable native; no original BF16/F32 model instantiated.','Actual upstream batching/cropping retained; pure operator attribution requires matching stage inputs.','Only retained actual native prefill/first-block arrays exist; step126 native intermediates are not inferred.']
 except BaseException as e:report['error_type']=type(e).__name__;report['error']=str(e);raise
 finally:
  base.stop(active);deadline.close()
  for s,h in prior.items():signal.signal(s,h)
  report['seconds']=time.monotonic()-started;owned={x+'.log'for x in report['commands']}|{'CMakeCache.txt'};report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
