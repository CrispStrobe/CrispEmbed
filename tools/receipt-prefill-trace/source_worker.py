#!/usr/bin/env python3
"""Pinned upstream-only neutral text trace; retained native arrays are read-only."""
import base64,csv,hashlib,importlib.util,json,os,platform,shutil,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,WorkerDeadline,command
sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_observer_patch as observer;import lighton4b_upstream_dispatch_log as dispatch
sys.path.insert(0,str(Path(__file__).parent));import source_patch
REFERENCE='8bf1ebbdf12b9dd2b20603c11a9b7d4c990e97e7f37a2f5aca3fb3324b3c8dfc'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('physical',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');require=base.require;sha=base.sha
PROFILE={'BUILD_SHARED_LIBS':'ON','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_CPU_REPACK':'ON','GGML_OPENMP':'ON','GGML_LLAMAFILE':'ON','GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
def validate_response(response,frozen):
 require(response['tokens']==frozen['tokens'][:1] and response['completion_probabilities']==frozen['completion_probabilities'][:1], 'same frozen first sample')
 require(response['stop_type']=='limit' and response['tokens_evaluated']==256 and not response['truncated'],'one-token prefill-only request')
def validate_png_inputs(original,canonical,rgb):
 from PIL import Image
 a=Image.open(original).convert('RGB');b=Image.open(canonical).convert('RGB')
 require(Image.__version__=='11.3.0'and a.size==b.size and a.tobytes()==b.tobytes()==Path(rgb).read_bytes(),'same authenticated actual RGB pixels across PNG containers')
def main():
 import numpy as np
 from PIL import Image
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64']and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU/array profile')
 cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());require(cfg['schema']=='lighton4b-receipt-prefill/v1'and cfg['maximum_capture_bytes']==768<<20,'closed bounded stage capture')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'receipt-prefill-source';work.mkdir();started=time.monotonic();active=None;arm_end=[None];deadline=Deadline(3600);deadline.__enter__()
 report={'schema':'lighton4b-receipt-prefill-source/v1','status':'failed','commands':{},'arms':{},'native_forward_executed':False,'original_bf16_forward_executed':False,'f32_model_instantiated':False,'quality_qualified':False}
 def save():(out/'receipt-prefill-source.json').write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
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
  with urllib.request.urlopen(request,timeout=min(remaining(),1200 if path=='/completion'else 30))as f:raw=f.read((8<<20)+1)
  require(len(raw)<=8<<20,'bounded source response');return json.loads(raw)
 try:
  run('source',['git','rev-parse','HEAD']);report['source_sha']=(out/'source.log').read_text().strip();require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source');run('cpu',['lscpu']);flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual private CPU features')
  report['capacity']={'memory':base.memory(),'scratch_free':shutil.disk_usage(work).free,'output_free':shutil.disk_usage(out).free};require(report['capacity']['memory']['MemAvailable']>=10<<30 and report['capacity']['scratch_free']>=12<<30 and report['capacity']['output_free']>=4<<30,'bounded reference-only capacity')
  op=locate('heldout-result.json',REFERENCE);original=json.loads(op.read_text());require(original['case']=='historical-receipt' and original['source_observer_full_neutral'],'historical neutral receipt')
  folder=op.parent;art=original['artifacts'];report['historical_reference_sha256']=sha(op)
  def owned(name):
   path=folder/name;verify(path,art[name]);return path
  png=owned('inputs/input.png');rgb=owned('inputs/input.rgb');validate_png_inputs(png,png,rgb)
  sealpath=owned('pre-forward-inputs.json');require(sha(sealpath)==original['pre_forward_input_seal_sha256'],'original input seal')
  seal=json.loads(sealpath.read_text());prompt=seal['prompt_ids'];mods=seal['modalities'];require(seal['case']=='historical-receipt' and len(prompt)<=512 and mods.count(1)<=256,'bounded unchanged input seal')
  frozen=json.loads(owned('upstream-on/response.json').read_text());require(len(frozen['tokens'])==465 and frozen['stop_type']=='eos','frozen full source unchanged')
  rows=original['source_raw_rows']['rows'];require(len(rows)==465 and all(row['position']==i and row['argmax']==frozen['tokens'][i] for i,row in enumerate(rows)),'all sealed source rows (no decode repeat)')
  protocol=json.loads((ROOT/'docs/lighton4b-heldout-executor-v1.json').read_text());require(protocol['upstream_source']==source_patch.SOURCE==observer.SOURCE,'exact source')
  # Legacy aliases are deliberately explicit; no floating upstream or model revision.
  protocol['upstream_repo']='https://github.com/ggml-org/llama.cpp.git';protocol['upstream_commit']=protocol['upstream_source'];protocol['image_min_tokens']=64;protocol['image_max_tokens']=256
  for name in ['upstream-on/source.vision.0.f32','upstream-on/source.vision.0.positions.u32']:owned(name)
  repo=work/'upstream'
  for label,args in [('init',['git','init',repo]),('remote',['git','-C',repo,'remote','add','origin',protocol['upstream_repo']]),('fetch',['git','-C',repo,'fetch','--depth','1','origin',source_patch.SOURCE]),('checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==source_patch.SOURCE,'exact upstream')
  for name,digest in cfg['source_function_sha256'].items():require(sha(repo/name)==digest,'exact actual source attention blueprint')
  report['feature_sampler_patch']=observer.patch(repo);report['dispatch_patch']=dispatch.patch(repo);report['text_live_patch']=source_patch.patch(repo);save()
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
   args=[server,'-m',model,'--mmproj',projector,'--host','127.0.0.1','--port',str(port),'-t','4','-tb','4','-c','2048','-np','1','-ngl','0','--no-context-shift','--cache-type-k','f16','--cache-type-v','f16','--no-mmproj-offload','--no-warmup','--flash-attn','off','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--image-min-tokens',str(protocol['image_min_tokens']),'--image-max-tokens',str(protocol['image_max_tokens'])];report['phase']=arm;report['arms'][arm]={'status':'running'};save();arm_started=time.monotonic();arm_end[0]=arm_started+1200
   with(dest/'server.log').open('w')as log:
    active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=child,start_new_session=True)
    try:
     for _ in range(180):
      remaining();require(active.poll()is None,'source alive')
      try:api(port,'/health');break
      except WorkerDeadline:raise
      except (OSError,urllib.error.URLError):time.sleep(min(1,remaining()))
     else:raise RuntimeError('source health deadline')
     marker=api(port,'/props').get('media_marker');require(isinstance(marker,str)and 1<=len(marker)<=256,'source media marker');template=api(port,'/apply-template',{'messages':[{'role':'user','content':marker}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt'];require(template.count(marker)==1 and template.replace(marker,'<__media__>')==protocol['expected_media_prompt'],'same wrapper');expanded=template.replace(marker,'<|vision_start|>'+'<|image_pad|>'*mods.count(1)+'<|vision_end|>');require(api(port,'/tokenize',{'content':expanded,'add_special':False,'parse_special':True})['tokens']==prompt,'same actual source prompt')
     request={'prompt':{'prompt_string':template,'multimodal_data':[base64.b64encode(png.read_bytes()).decode()]},'n_predict':1,'temperature':0,'seed':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'cache_prompt':False,'return_tokens':True,'n_probs':5,'stream':False};response=api(port,'/completion',request);validate_response(response,frozen);responses.append(response);(dest/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False)+'\n');(dest/'generated.txt').write_bytes(response['content'].encode());(dest/'generated.ids').write_text(''.join(str(x)+'\n'for x in response['tokens']))
     for i in range(1):
      row_path=dest/f'source.logits.{i}.f32'
      verify(row_path,rows[i])
      values=np.fromfile(row_path,dtype='<f4');require(values.size==248320 and np.isfinite(values).all()and int(values.argmax())==response['tokens'][i],'finite actual full-vocabulary row/greedy argmax')
     verify(dest/'source.vision.0.f32',art['upstream-on/source.vision.0.f32']);verify(dest/'source.vision.0.positions.u32',art['upstream-on/source.vision.0.positions.u32']);require(len(list(dest.glob('source.vision.*.f32')))==1 and (dest/dispatch.OUTPUT).is_file(),'one actual vision forward/dispatch inventory');report['arms'][arm].update(status='source-frozen-exact',frozen_firstscore_exact=True,frozen_first_sample_exact=True,source_features_positions_exact=True,generated_ids=response['tokens'],decoded=response['content'],eos=False)
    finally:
     if active is not None and Path(f'/proc/{active.pid}/status').is_file():report['arms'][arm]['process_status_before_stop']={x.split(':',1)[0]:x.split(':',1)[1].strip()for x in Path(f'/proc/{active.pid}/status').read_text().splitlines()if x.startswith(('VmHWM:','VmRSS:','Threads:'))}
     base.stop(active);report['arms'][arm]['returncode']=active.returncode if active is not None else None;active=None;arm_end[0]=None;report['arms'][arm]['seconds']=time.monotonic()-arm_started;save()
  require(responses[0]['tokens']==responses[1]['tokens']and responses[0]['completion_probabilities']==responses[1]['completion_probabilities'],'complete live observer neutrality')
  for i in range(1):require(sha(out/'off'/f'source.logits.{i}.f32')==sha(out/'on'/f'source.logits.{i}.f32'),'actual raw firstscore observer-neutral')
  require((out/'off'/dispatch.OUTPUT).read_bytes()==(out/'on'/dispatch.OUTPUT).read_bytes(),'actual source quantized buffer dispatch observer-neutral');report['firstscore_observer_neutral']=True
  # Capture status is independent of a later source/native reader.
  indexes=[]
  for index in sorted((out/'on').glob('graph-*.tsv')):
   records=list(csv.DictReader(index.open(),delimiter='\t'));require(records,'nonempty graph index')
   indexes.append(dict(file=index.name,sample_count=int(records[0]['sample_count']),records=records))
  require(len(indexes)==5 and all(x['sample_count']==0 for x in indexes),'auxiliary plus four actual prefill graphs')
  report['graph_indexes']=indexes;report['status']='measured_neutral_receipt-prefill_source'
  report['limits']=['Actual quantized source prefill and first sample only; no full decode or original BF16/F32 model.', 'Native matched-input layer3 component replay remains independently required.']

 except BaseException as e:report['error_type']=type(e).__name__;report['error']=str(e);raise
 finally:
  base.stop(active);deadline.close()
  for s,h in prior.items():signal.signal(s,h)
  report['seconds']=time.monotonic()-started;owned={x+'.log'for x in report['commands']}|{'CMakeCache.txt'};report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
