#!/usr/bin/env python3
"""One frozen provisional held-out case; controls precede all held-out forwards."""
import argparse,base64,hashlib,importlib.util,io,json,os,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;sys.path.insert(0,str(HERE));import execution as ex
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command
from source_session import Session,api,port
from transfer import download as authenticated_transfer
require=ex.require;sha=ex.sha

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 import numpy as np
 from PIL import Image
 parser=argparse.ArgumentParser();parser.add_argument('--case',choices=ex.prepare.CASE_IDS);parser.add_argument('--controls-only',action='store_true');opt=parser.parse_args();require(opt.controls_only or opt.case is not None,'fixed case required')
 cfg=json.loads((ROOT/'docs/lighton4b-heldout-executor-v1.json').read_text());ex.admit(cfg);selection=json.loads(ex.prepare.PROTOCOL.read_text())
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and np.__version__=='2.3.5'and Image.__version__=='11.3.0','CPU/serializer profile')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'heldout';work.mkdir();started=time.monotonic();seconds=1800 if opt.controls_only else 6000
 deadline=Deadline(seconds);deadline.__enter__()
 def canceled(sig,_frame):raise SystemExit(128+sig)
 prior={s:signal.signal(s,canceled)for s in[signal.SIGTERM,signal.SIGINT]}
 r=dict(schema='lighton4b-provisional-heldout-result/v1',status='failed',case=opt.case,commands={},arms={},qualified=False,references_expert_adjudicated=False,heldout_forward_executed=False,model_acquisition=False,reference_limits=selection['reference'],source_observer_scope=cfg['source_raw_scope'])
 def save():(out/'heldout-result.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=900,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,seconds-(time.monotonic()-started)),r['commands'][label],cwd=ROOT,env=env,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'one frozen '+name);return rows[0]
 def verify(path,meta):require(path.stat().st_size==meta['bytes']and sha(path)==meta['sha256'],'authenticated '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  base=module('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');profile=module('vision_profile',ROOT/'tools/physical-f16-attention-probe/worker.py')
  r['capacity']=dict(memory=base.memory(),scratch_free=shutil.disk_usage(work).free,output_free=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch_free']>=25<<30 and r['capacity']['output_free']>=8<<30,'frozen capacity before models')
  run('fetch-production',['git','fetch','--depth','1','origin',cfg['production_source']]);run('production-bytes',['git','diff','--exit-code',cfg['production_source'],'HEAD','--','src','ggml','CMakeLists.txt'])
  run('ggml',['git','submodule','update','--init','ggml']);native_build=work/'native-build';flags=dict(profile.BUILD_FLAGS);flags['GGML_CPU_REPACK']='ON'
  run('native-configure',['cmake','-S',ROOT,'-B',native_build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);profile.validate_profile((native_build/'CMakeCache.txt').read_text().splitlines());shutil.copyfile(native_build/'CMakeCache.txt',out/'native-CMakeCache.txt')
  targets=['lighton4b-native-document','lighton4b-native-case-preflight','test-lighton4b-physical-f16-vision','test-lighton4b-physical-f16-attention'];run('native-build',['cmake','--build',native_build,'--target',*targets,'--parallel','2'])
  for target in targets[2:]:run(target,[native_build/target]);require(' PASS 'in(out/(target+'.log')).read_text(),'actual supported target')
  require('PHYSICAL_F16_VISION_SYNTHETIC PASS 'in(out/(targets[2]+'.log')).read_text()and'PHYSICAL_F16_ATTENTION PASS 'in(out/(targets[3]+'.log')).read_text(),'supported real private ABI')
  native=native_build/'lighton4b-native-document';env=dict(os.environ);env['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT']='1';dummy=work/'missing';teacher=work/'one.ids';teacher.write_text('248046\n')
  checks=[]
  for arm in ['native-composed','native-source-features']:checks.append((arm,ex.native_args(native,dummy,dummy,1,1,teacher,out,arm,dummy),4,env))
  canonical=ex.native_args(native,dummy,dummy,1,1,teacher,out,'native-composed')
  for name,index,value in [('cap-low',7,1535),('cap-high',7,1537),('threads',8,2),('mode',9,'full')]:
   args=list(canonical);args[index]=value;checks.append((name,args,2,env))
  disabled=dict(env);disabled['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT']='0';checks.append(('disabled',canonical,2,disabled));checks.append(('unflagged',canonical[:-1],2,env));checks.append(('old256',[native,dummy,dummy,1,1,teacher,out,256,4,'greedy-full'],4,disabled));checks.append(('old512',[native,dummy,dummy,1,1,teacher,out,512,4,'greedy-full','--observed-composed-cap512'],4,disabled))
  for label,args,expected,child in checks:
   try:run('cli-'+label,args,child,30)
   except subprocess.CalledProcessError as e:require(e.returncode==expected,'actual CLI status '+label)
   else:raise ValueError('missing model unexpectedly admitted')
  r['actual_cli_controls']=len(checks)
  upstream=work/'upstream'
  for name,args in [('init',['git','init',upstream]),('remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git']),('fetch',['git','-C',upstream,'fetch','--depth','1','origin',cfg['upstream_source']]),('checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD'])]:run('up-'+name,args)
  sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_observer_patch as observer;import lighton4b_upstream_physical_pack as pack
  require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==observer.SOURCE==cfg['upstream_source'],'pinned primary upstream');r['observer_patch']=observer.patch(upstream);shutil.copyfile(HERE/'source_preflight.cpp',upstream/'heldout-source-preflight.cpp')
  with(upstream/'CMakeLists.txt').open('a')as f:f.write('\nadd_executable(heldout-source-preflight heldout-source-preflight.cpp)\ntarget_include_directories(heldout-source-preflight PRIVATE tools/mtmd)\ntarget_link_libraries(heldout-source-preflight PRIVATE mtmd llama)\n')
  upflags={k:v for k,v in flags.items()if not k.startswith('CRISPEMBED_')};upflags['GGML_OPENMP']='ON';up_build=work/'up-build';run('up-configure',['cmake','-S',upstream,'-B',up_build,'-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF',*[f'-D{k}={v}'for k,v in upflags.items()]])
  cache=(up_build/'CMakeCache.txt').read_text().splitlines();require(all(f'{k}:BOOL={v}'in cache for k,v in upflags.items()),'actual complete source CPU profile');shutil.copyfile(up_build/'CMakeCache.txt',out/'upstream-CMakeCache.txt');run('up-build',['cmake','--build',up_build,'--target','llama-server','heldout-source-preflight','--parallel','2'],limit=1200)
  source_preflight=up_build/'bin/heldout-source-preflight';server=up_build/'bin/llama-server'
  try:run('source-preflight-noargs',[source_preflight],limit=30)
  except subprocess.CalledProcessError as e:require(e.returncode==1,'source preflight noargs closed')
  else:raise ValueError('source preflight unexpectedly admitted noargs')
  require('required'in(out/'source-preflight-noargs.log').read_text(),'actual source metadata CLI linked');run('server-help',[server,'--help'],limit=30)
  for flag in ['--cache-type-k','--cache-type-v','--no-context-shift','--no-warmup']:require(flag in(out/'server-help.log').read_text(),'pinned source CLI '+flag)
  r['native_binary_sha256']=sha(native);r['source_binary_sha256']=sha(server);r['source_preflight_binary_sha256']=sha(source_preflight)
  if opt.controls_only:r['status']='passed_model_free_heldout_controls';return
  require(len(cfg['controls_receipt_sha256'])==64,'hosted controls must be frozen before case execution');controlp=locate('heldout-result.json',cfg['controls_receipt_sha256']);control=json.loads(controlp.read_text());require(control['status']=='passed_model_free_heldout_controls'and control['model_acquisition']is False,'actual model-free controls passed')
  for name,meta in control['artifacts'].items():verify(controlp.parent/name,meta)
  def download(url,path,size,digest):
   r['phase']='download-'+Path(path).name;r.setdefault('downloads',{}).setdefault(Path(path).name,[]);save()
   try:authenticated_transfer(base.download,url,path,size,digest,r['downloads'][Path(path).name])
   finally:save()
  archive=work/'prepared.zip';download(selection['prepared_archive']['url'],archive,selection['prepared_archive']['bytes'],selection['prepared_archive']['sha256']);r['input_preparation']=ex.prepare.prepare(selection,archive);require(r['input_preparation']['candidate_outputs_read']is False,'all frozen inputs admitted before model');case=next(c for c in selection['cases']if c['id']==opt.case);case_out=out/'inputs';case_out.mkdir()
  with zipfile.ZipFile(archive)as z:
   names=[n for n in z.namelist()if Path(n).name==case['prepared']['file']];require(len(names)==1,'one immutable prepared PNG');png=case_out/'input.png';png.write_bytes(z.read(names[0]))
  image=Image.open(png).convert('RGB');rgb=case_out/'input.rgb';rgb.write_bytes(image.tobytes());require(sha(png)==case['prepared']['png_sha256']and sha(rgb)==case['prepared']['rgb_sha256'],'prepared case bytes')
  formp=locate('composed-form-vision.json',cfg['observed_form_receipt_sha256']);form=json.loads(formp.read_text());require(form['status']=='measured_composed_form_vision'and form['arms']['fixed']['expected_text_and_rows_exact'],'observed complete pixel form prerequisite')
  env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(form['arms']['fixed']['gates']);require(all(env[k]=='1'for k in['CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION'])and all(env[k]=='0'for k in['CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_HEAD_BF16','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS','CRISPEMBED_LIGHTON4B_BF16_LINEAR']),'frozen selected candidate gate tuple');env['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT']='1';env['CRISPEMBED_LIGHTON4B_FROZEN_CASES']='1';r['native_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')}
  r['model_acquisition']=True;save();prefix='https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/';up_model=work/cfg['models']['Q4_K_M']['file'];projector=work/cfg['projector']['file']
  for path,row in[(up_model,cfg['models']['Q4_K_M']),(projector,cfg['projector'])]:download(prefix+path.name,path,row['bytes'],row['sha256'])
  row=cfg['base_model'];protected=work/row['file'];download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,up_model,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['candidate_sha256'],'same immutable physical model');protected.unlink()
  # Before any held-out forward, require the ordinary512 driver path to reproduce
  # the previously observed167-token image decode and all retained raw scores.
  baseline=out/'observed-control';baseline.mkdir();physical_recipe='lighton4b-upstream-q4km-physical-v1'
  observed_rgb=formp.parent/'inputs/input.rgb'
  if not observed_rgb.exists():
   # The existing form worker records raw input in the completed physical source
   # dependency, so this model run requires that exact completed dependency too.
   physicalp=locate('physical-decode.json','712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2');observed_rgb=physicalp.parent/'simple_form/input.rgb';verify(observed_rgb,json.loads(physicalp.read_text())['cases']['simple_form']['artifacts']['input.rgb'])
  run('observed-control',[native,candidate,observed_rgb,317,452,teacher,baseline,512,4,'greedy-full','--observed-composed-cap512'],env,1200)
  old_ids=form['arms']['fixed']['generated_ids'];require(list(map(int,(baseline/'generated.ids').read_text().split()))==old_ids,'unchanged observed167 before heldout')
  for name in['prefill.input.f32','prefill.positions.f32','prefill.logits.f32']+[f'prefill.layer.{i}.f32'for i in range(32)]+[f'greedy.{i}.logits.f32'for i in range(167)]:require(sha(baseline/name)==form['artifacts']['fixed/'+name]['sha256'],'observed runtime byte guard '+name)
  r['observed167_runtime_byte_control']=True;r['observed_control_rows']=ex.audit_rows(baseline,old_ids);save()
  pre=out/'native-preflight';pre.mkdir();run('native-preflight',[native_build/'lighton4b-native-case-preflight',candidate,rgb,image.height,image.width,pre,physical_recipe],env)
  template=work/'template.txt';template.write_text(cfg['expected_media_prompt']);sp=out/'source-preflight';run('source-preflight',[source_preflight,up_model,projector,rgb,image.height,image.width,template,sp]);ids=list(map(int,(pre/'prompt.ids').read_text().split()));mods=list(map(int,(pre/'modalities.ids').read_text().split()));grid=list(map(int,(pre/'grid.ids').read_text().split()));source_meta=json.loads(Path(str(sp)+'.json').read_text());require(source_meta['model_forward_calls']==0 and source_meta['vocab_only']and source_meta['warmup']is False,'source input-only preflight');source_ids=list(map(int,Path(str(sp)+'.prompt.ids').read_text().split()));source_mods=list(map(int,Path(str(sp)+'.modalities.ids').read_text().split()))
  seal=out/'pre-forward-inputs.json';r['pre_forward_input_seal_sha256']=ex.seal_inputs(seal,case,ids,mods,grid,(pre/'prefill.positions.f32').read_bytes(),source_ids,source_mods,Path(str(sp)+'.positions.f32').read_bytes(),cfg['expected_media_prompt'],cfg);save()
  responses=[]
  for enabled in[False,True]:
   require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-source-arm resource admission')
   arm='upstream-on'if enabled else'upstream-off';dest=out/arm;dest.mkdir();child=dict(os.environ);child.pop(observer.ENV,None)
   if enabled:child[observer.ENV]=str(dest)
   endpoint=port();argv=[server,'-m',up_model,'--mmproj',projector,'--host','127.0.0.1','--port',endpoint,'-t',4,'-tb',4,'-c',2048,'-np',1,'-ngl',0,'--no-mmproj-offload','--no-warmup','--no-context-shift','--cache-type-k','f16','--cache-type-v','f16','--flash-attn','off','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--image-min-tokens',64,'--image-max-tokens',256]
   r['phase']=arm;save()
   with Session(argv,dest/'server.log',child,1200)as session:
    session.ready(endpoint);marker=api(endpoint,'/props',None,session.deadline)['media_marker'];templ=api(endpoint,'/apply-template',{'messages':[{'role':'user','content':marker}],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}},session.deadline)['prompt'];require(templ.count(marker)==1 and templ.replace(marker,'<__media__>')==cfg['expected_media_prompt'],'actual source frozen template');expanded=templ.replace(marker,'<|vision_start|>'+'<|image_pad|>'*mods.count(1)+'<|vision_end|>');require(api(endpoint,'/tokenize',{'content':expanded,'add_special':False,'parse_special':True},session.deadline)['tokens']==ids,'actual server prompt IDs');require(sha(seal)==r['pre_forward_input_seal_sha256'],'immutable pre-forward inputs')
    request={'prompt':{'prompt_string':templ,'multimodal_data':[base64.b64encode(png.read_bytes()).decode()]},'n_predict':1536,'temperature':0,'seed':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'cache_prompt':False,'return_tokens':True,'n_probs':5,'stream':False};r['heldout_forward_attempted']=True;save();response=api(endpoint,'/completion',request,session.deadline);complete=ex.source_response(response,len(ids));r['heldout_forward_executed']=True;(dest/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False)+'\n');(dest/'generated.ids').write_text(''.join(str(x)+'\n'for x in response['tokens']));(dest/'generated.txt').write_bytes(response['content'].encode());responses.append(response)
   r['arms'][arm]=dict(complete=complete,ids=response['tokens'],decoded=response['content'],score=ex.score(json.loads((ROOT/case['gold']).read_text()),response['content'],complete),resources=session.stats);save()
  ex.neutral(*responses);r['source_observer_full_neutral']=True;source_dir=out/'upstream-on';source_row=np.fromfile(source_dir/'source.logits.0.f32',dtype='<f4');require(source_row.size==248320 and np.isfinite(source_row).all(),'actual source first row');shutil.copyfile(source_dir/'source.logits.0.f32',source_dir/'first-score.f32');r['source_raw_rows']=ex.audit_rows(source_dir,responses[1]['tokens'],stem='source');feature=source_dir/'source.vision.0.f32';require(feature.stat().st_size==mods.count(1)*2560*4,'source actual merged geometry');gold=json.loads((ROOT/case['gold']).read_text())
  require(sha(up_model)==cfg['models']['Q4_K_M']['sha256']and sha(projector)==cfg['projector']['sha256'],'source weights unchanged');up_model.unlink();projector.unlink()
  for arm in['native-composed','native-source-features']:
   require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-native-arm resource admission')
   require(sha(seal)==r['pre_forward_input_seal_sha256'],'sealed unchanged inputs');dest=out/arm;dest.mkdir();run(arm,ex.native_args(native,candidate,rgb,image.height,image.width,teacher,dest,arm,feature),env,1200);summary=json.loads((dest/'native-summary.json').read_text());generated=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();require(generated and len(generated)<=1536 and summary['eos']==(generated[-1]in[248044,248046])and(summary['eos']or len(generated)==1536)and summary['source_fed_vision']==(arm=='native-source-features')and summary['teacher_positions']==1,'native actual EOS/cap/mode')
   require(list(map(int,(dest/'prompt.ids').read_text().split()))==ids and(dest/'prefill.positions.f32').read_bytes()==(pre/'prefill.positions.f32').read_bytes(),'native actual sealed positions/prompt');row=np.fromfile(dest/'prefill.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and sha(dest/'prefill.logits.f32')==sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32'),'actual same first row');difference=row.astype('f8')-source_row;merged=np.fromfile(dest/'vision.merged.f32',dtype='<f4');reference=np.fromfile(feature,dtype='<f4');require(merged.shape==reference.shape and np.isfinite(merged).all()and np.isfinite(reference).all(),'actual finite merged geometry');delta=merged.astype('f8')-reference
   r['arms'][arm]=dict(complete=summary['eos'],ids=generated,decoded=text,summary=summary,score=ex.score(gold,text,summary['eos']),raw_rows=ex.audit_rows(dest,generated),first_score_vs_source=dict(rms=float(np.sqrt(np.mean(difference*difference))),max_abs=float(abs(difference).max())),merged_vs_source=dict(bit_exact=merged.tobytes()==reference.tobytes(),rms=float(np.sqrt(np.mean(delta*delta))),max_abs=float(abs(delta).max()),source_max_abs=float(abs(reference).max())),resources=r['commands'][arm]);save()
  r['primary_meets_provisional_source_floor']=ex.floor(r['arms']['native-composed']['score'],r['arms']['upstream-off']['score'],case['id']=='inventory-table');r['all_primary_complete']=all(r['arms'][a]['complete']for a in['upstream-off','native-composed']);require(sha(candidate)==cfg['candidate_sha256'],'weights unchanged');r['status']='measured_provisional_heldout';r['qualified']=False;r['reference_adjudication_required']=True
 except BaseException as error:r['error_type']=type(error).__name__;r['error']=str(error);raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-started;owned={label+'.log'for label in r['commands']}|{'native-CMakeCache.txt','upstream-CMakeCache.txt','physical-pack.json','pre-forward-inputs.json','source-preflight.json','source-preflight.prompt.ids','source-preflight.modalities.ids','source-preflight.positions.f32'};r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
