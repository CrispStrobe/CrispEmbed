#!/usr/bin/env python3
"""Fixed physical Q4_K_M same-feature full form diagnostic, never qualification."""
import hashlib,importlib.util,io,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command as guarded_command

def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
STAGES=tuple(['text.embedded.f32','prefill.input.f32','prefill.positions.f32','prefill.norm.f32','prefill.logits.f32','greedy.0.logits.f32']+['prefill.layer.'+str(i)+'.f32'for i in range(32)])
def admit(cfg):
 require(cfg['schema']=='lighton4b-physical-f16-full-form/v1'and cfg['arms']==['baseline-prefix','f16-full'],'closed full form arms')
 require(cfg['cap']==256 and cfg['baseline_cap']==2 and cfg['teacher_positions']==177 and cfg['threads']==4 and cfg['context']==2048,'fixed score/generation scope')
 require(len(cfg['cases'])==1 and cfg['cases'][0]['id']=='simple_form'and cfg['expected_candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43','fixed input/weights')
 require(cfg['golden_smoke_report_sha256']=='1694d990aa2b23c17685025134b0c565aff844369ef823a46bfc6a8933a219a1','frozen actual same-feature smoke')
 for name,size in [('required_integration_source_sha',40),('required_integration_receipt_sha256',64),('reviewed_main_link_fix_sha',40),('reviewed_cmake_sha256',64)]:
  value=cfg.get(name);require(isinstance(value,str)and len(value)==size and all(c in'0123456789abcdef'for c in value),'reviewed integration receipt required: '+name)
 require(cfg['runtime_sources']['CMakeLists.txt']==cfg['reviewed_cmake_sha256'],'reviewed CMake-only exception hash')
 require(cfg['native_arm_timeout_seconds']==1800 and cfg['total_model_timeout_seconds']==2700,'bounded model budget')
 for name,digest in cfg['runtime_sources'].items():require(sha(ROOT/name)==digest,'tested runtime blob drift: '+name)

def metric(base,ids,text,reference_ids,reference_text):
 prefix=0
 for a,b in zip(ids,reference_ids):
  if a!=b:break
  prefix+=1
 return {'raw_ids_exact':ids==reference_ids,'decoded_exact':text==reference_text,'matching_prefix_ids':prefix,'character_edits':base.edits(reference_text,text),'reference_codepoints':len(reference_text),'word_edits':base.edits(reference_text.split(),text.split()),'reference_words':len(reference_text.split())}

def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only')
 import numpy as np
 from PIL import Image
 require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','serializer pins')
 cfgpath=ROOT/'docs/lighton4b-physical-f16-full-form-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'physical-f16-full-form';scratch.mkdir()
 base=module('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');common=module('native_base',ROOT/'tools/ci-heavy/lighton4b_native_document.py');vision=module('vision_gates',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py');sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
 report={'schema':cfg['schema'],'status':'failed','source_sha':None,'protocol_sha256':sha(cfgpath),'f32_model_instantiated':False,'upstream_forward_executed':False,'quality_qualified':False,'arms':{},'resources':{}}
 started=time.monotonic();deadline=Deadline(cfg['total_model_timeout_seconds']);deadline.__enter__()
 def save():(out/'physical-f16-full-form.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,timeout=600,on_tick=None):
  report['phase']=label;report['resources'][label]={};save();print('phase',label,flush=True)
  try:guarded_command(args,out/(label+'.log'),min(timeout,cfg['total_model_timeout_seconds']-(time.monotonic()-started)),report['resources'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one frozen receipt: '+name);return found[0]
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'actual input artifact: '+path.name)
 try:
  run('source-pin',['git','rev-parse','HEAD']);report['source_sha']=(out/'source-pin.log').read_text().strip();require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  proofpath=locate('physical-f16-controls.json',cfg['required_integration_receipt_sha256']);proof=json.loads(proofpath.read_text());require(proof['status']=='model_free_sourcefed_pass'and proof['source_sha']==cfg['required_integration_source_sha']and len(proof['sourcefed'])==18 and all(row['bit_exact']for row in proof['sourcefed'].values()),'complete tested runtime integration proof');report['integration_receipt_sha256']=sha(proofpath)
  run('fetch-tested-source',['git','fetch','--depth','1','origin',cfg['required_integration_source_sha']])
  run('tested-runtime-tree',['git','diff','--exit-code',cfg['required_integration_source_sha'],'HEAD','--','src','ggml','tests/lighton4b_native_document.cpp'])
  run('fetch-reviewed-main-link-fix',['git','fetch','--depth','1','origin',cfg['reviewed_main_link_fix_sha']])
  run('reviewed-cmake-only',['git','diff','--exit-code',cfg['reviewed_main_link_fix_sha'],'HEAD','--','CMakeLists.txt'])
  require(sha(ROOT/'CMakeLists.txt')==cfg['reviewed_cmake_sha256'],'reviewed main-stub-only CMake exception')
  # Protocol hashes and immutable integration-tree equality admit tools-only additions.
  run('host-cpu',['lscpu']);report['preflight']={'memory':base.memory(),'scratch_free':shutil.disk_usage(scratch).free,'output_free':shutil.disk_usage(out).free,'load':os.getloadavg()};require(report['preflight']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and report['preflight']['scratch_free']>=cfg['minimum_scratch_free_bytes']and report['preflight']['output_free']>=cfg['minimum_output_free_bytes'],'RAM/disk before weights')
  gp=locate('physical-native-ops-smoke.json',cfg['golden_smoke_report_sha256']);gold=json.loads(gp.read_text());require(gold['source_sha']==cfg['golden_smoke_source_sha']and gold['status']=='measured'and gold['candidate_sha256']==cfg['expected_candidate_sha256'],'frozen smoke source/weights')
  folder=gp.parent/'simple_form';art=gold['cases']['simple_form']['artifacts'];report['golden_report_sha256']=sha(gp)
  for rel in ['input.rgb','original.json','retained-upstream-response.json','off/prompt.ids','off/modalities.ids','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32']:verify(folder/rel,art[rel])
  case=cfg['cases'][0];require(sha(ROOT/case['image'])==case['image_sha256'],'immutable image');image=Image.open(ROOT/case['image']).convert('RGB');rgb=folder/'input.rgb';require(rgb.read_bytes()==image.tobytes(),'exact captured RGB')
  ids=list(map(int,(folder/'off/prompt.ids').read_text().split()));buf=io.BytesIO();np.save(buf,np.array([ids],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(buf.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'actual prompt IDs')
  original=json.loads((folder/'original.json').read_text());retained=json.loads((folder/'retained-upstream-response.json').read_text());require(len(original['generated_ids'])==177 and original['generated_ids'][-1]in[248044,248046],'frozen original teacher sequence');teacher=scratch/'original177.ids';teacher.write_text(''.join(str(x)+'\n'for x in original['generated_ids']))
  feature=folder/'observer-on/source.vision.0.f32';source_score=np.fromfile(folder/'observer-on/source.logits.0.f32',dtype='<f4');require(source_score.size==248320 and np.isfinite(source_score).all(),'full actual source row')
  run('submodule',['git','submodule','update','--init','ggml']);build=scratch/'build';flags={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in flags.items():require(f'{k}:BOOL={v}'in cache,'exact build profile '+k)
  run('build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-physical-f16-attention','--parallel','2']);run('host-supported-guard',[build/'test-lighton4b-physical-f16-attention']);require('PHYSICAL_F16_ATTENTION PASS 'in(out/'host-supported-guard.log').read_text(),'supported hardware/ABI required before model')
  shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt');native=build/'lighton4b-native-document';report['binary_sha256']=sha(native)
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin'/native.name)
  libraries=sorted(p for p in build.rglob('libggml*.so*')if p.is_file());require(libraries,'actual CPU backend libraries')
  for library in libraries:shutil.copy2(library,bundle/'lib'/library.name)
  report['runtime_bundle']={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in bundle.rglob('*')if p.is_file()}
  loader=':'.join(sorted({str(p.parent)for p in libraries}))+(':'+os.environ['LD_LIBRARY_PATH']if os.environ.get('LD_LIBRARY_PATH')else'')
  row=cfg['models']['Q4_K_M'];up=scratch/row['file'];report['phase']='acquire-source';save();base.download('https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/'+row['file'],up,row['bytes'],row['sha256'])
  row=cfg['base_model'];protected=scratch/row['file'];report['phase']='acquire-protected';save();base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256'])
  candidate=scratch/'physical-native.gguf';report['phase']='authenticated-lossless-pack';save();audit=pack.convert(protected,up,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['expected_candidate_sha256'],'immutable candidate');report['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update({k:'1'for k in common.GATES+vision.VISION_GATES});env.update(CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL='1',CRISPEMBED_LIGHTON4B_Q6_K='1',CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1',CRISPEMBED_LIGHTON4B_HEAD_BF16='0',CRISPEMBED_LIGHTON4B_BF16_LINEAR='0',CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS='0',CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS='1',CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION='all')
  env['LD_LIBRARY_PATH']=loader
  for arm in cfg['arms']:
   require(base.memory()['MemAvailable']>=cfg['minimum_available_memory_bytes'],'per-arm RAM');dest=out/arm;dest.mkdir();child=dict(env);child['CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION']='0'if arm=='baseline-prefix'else'1';cap=2 if arm=='baseline-prefix'else 256;mode='greedy-full'if arm=='baseline-prefix'else'full';last=[0.];greedy_observation={}
   def tick():
    progress={}
    try:progress=json.loads((dest/'generated-progress.json').read_text())
    except(OSError,ValueError):pass
    count=progress.get('generated_tokens')
    if isinstance(count,int)and count>0:
     if not greedy_observation:greedy_observation.update(first_count=count,first_observed_monotonic=time.monotonic())
     greedy_observation.update(last_count=count,last_observed_monotonic=time.monotonic())
    if time.monotonic()-last[0]<30:return
    last[0]=time.monotonic()
    report['active_progress']={'arm':arm,'teacher_rows':len(list(dest.glob('teacher.*.logits.f32'))),'generated_tokens':progress.get('generated_tokens'),'last_id':progress.get('last_id')};save();print('PHYSICAL_F16_HEARTBEAT '+json.dumps(report['active_progress']),flush=True)
   run(arm,[native,candidate,rgb,image.height,image.width,teacher,dest,cap,4,mode,'--upstream-vision',feature],child,cfg['native_arm_timeout_seconds'],tick)
   summary=json.loads((dest/'native-summary.json').read_text());generated=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf-8');require(generated and len(generated)<=cap and summary['eos']==(generated[-1]in[248044,248046])and(summary['eos']or len(generated)==cap),'actual independent greedy/EOS/cap closure')
   require((dest/'prompt.ids').read_bytes()==(folder/'off/prompt.ids').read_bytes()and sha(dest/'vision.merged.f32')==art['observer-on/source.vision.0.f32']['sha256']and sha(dest/'prefill.input.f32')==art['off/prefill.input.f32']['sha256']and sha(dest/'prefill.positions.f32')==art['off/prefill.positions.f32']['sha256'],'same actual prompt/features/positions')
   if arm=='baseline-prefix':
    for rel in STAGES:require(sha(dest/rel)==art['on/'+rel]['sha256'],'unchanged gate-OFF golden stage '+rel)
   teacher_count=1 if arm=='baseline-prefix'else 177;require(summary['teacher_positions']==teacher_count,'actual full teacher scope')
   for j in range(teacher_count):
    row=np.fromfile(dest/f'teacher.{j}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all(),'finite teacher row')
   for j,token in enumerate(generated):
    row=np.fromfile(dest/f'greedy.{j}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and int(row.argmax())==token,'every full greedy row/argmax')
   require(sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32')==sha(dest/'prefill.logits.f32'),'same prefill first row')
   row=np.fromfile(dest/'greedy.0.logits.f32',dtype='<f4');delta=row.astype('f8')-source_score
   result={'gates':{k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},'teacher_positions':teacher_count,'generated_ids':generated,'decoded':text,'eos':summary['eos'],'cap_hit':not summary['eos']and len(generated)==cap,'full_completion_executed':arm=='f16-full','source_first_score':{'bit_exact':row.tobytes()==source_score.tobytes(),'rms':float(np.sqrt(np.mean(delta*delta))),'max_abs':float(abs(delta).max()),'reference_max_abs':float(abs(source_score).max())},'resources':report['resources'][arm]}
   if greedy_observation:
    duration=time.monotonic()-greedy_observation['first_observed_monotonic'];tokens_after_first=len(generated)-greedy_observation['first_count']
    result['observed_greedy_throughput']={'tokens_after_first_observation':tokens_after_first,'window_seconds':duration,'tokens_per_second':tokens_after_first/duration if duration>0 else None,'scope':'1s polling window after first greedy observation to child completion, includes score writes/finalization; not a balanced speed comparison'}
   if arm=='f16-full':
    stages=[]
    for layer in range(32):
     x=np.fromfile(dest/f'prefill.layer.{layer}.f32',dtype='<f4');y=np.fromfile(out/'baseline-prefix'/f'prefill.layer.{layer}.f32',dtype='<f4');require(x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'same finite baseline stage')
     d=x.astype('f8')-y;stages.append({'layer':layer,'bit_exact':x.tobytes()==y.tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'baseline_max_abs':float(abs(y).max())})
    result['vs_native_baseline_prefill_stages']={'scope':'Native ALL-ops baseline F16-attention OFF vs ON, not actual upstream hidden','layers':stages,'first_different_layer':next((row['layer']for row in stages if not row['bit_exact']),None)}
   if arm=='f16-full':result.update(vs_retained_upstream=metric(base,generated,text,retained['tokens'],retained['content']),vs_original_bf16=metric(base,generated,text,original['generated_ids'],original['decoded']),retained_upstream_vs_original_bf16=metric(base,retained['tokens'],retained['content'],original['generated_ids'],original['decoded']),independent_greedy_verified=True)
   else:result['golden_38_stage_anchor_exact']=True
   report['arms'][arm]=result;save()
  require(sha(candidate)==cfg['expected_candidate_sha256']and sha(up)==cfg['models']['Q4_K_M']['sha256'],'weights unchanged');report['status']='measured_full_form';candidate.unlink()
 except BaseException as error:report['error_type']=type(error).__name__;report['error']=str(error);raise
 finally:
  deadline.close();report['elapsed_seconds']=time.monotonic()-started;report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for arm in cfg['arms']for p in(out/arm).rglob('*')if p.is_file()};save()

if __name__=='__main__':main()
