#!/usr/bin/env python3
"""Captured-feature arithmetic ablation; six bounded prefixes, no qualification."""
import hashlib,importlib.util,io,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def admit_protocol(cfg):
 require(cfg['schema']=='lighton4b-physical-native-ops-ablation/v1'and cfg['arms']==['off','all','quant','rms','layers','head']and cfg['cap']==2 and len(cfg['cases'])==1 and cfg['cases'][0]['id']=='simple_form','closed six-arm form prefix protocol')
 require(cfg['golden_smoke_report_sha256']=='1694d990aa2b23c17685025134b0c565aff844369ef823a46bfc6a8933a219a1'and cfg['golden_smoke_source_sha']=='285ab5a4e9d669746a3664ec7804783ca06cc6dd','exact actual golden smoke')
 require(cfg['expected_candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43'and cfg['threads']==4 and cfg['native_arm_timeout_seconds']<=900 and cfg['total_model_timeout_seconds']<=2400,'closed weights/CPU/budget')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
STAGES=tuple(['text.embedded.f32','prefill.input.f32','prefill.positions.f32','prefill.norm.f32','prefill.logits.f32','greedy.0.logits.f32']+['prefill.layer.'+str(i)+'.f32'for i in range(32)])
def check_anchor(arm,current,golden):
 require(arm in ['off','all']and set(current)==set(STAGES),'complete endpoint stage set')
 for rel in STAGES:require(current[rel]==golden[rel]['sha256'],'golden '+arm+' stage drift: '+rel)
def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir(),'Kaggle CPU only');require(not list(Path('/dev').glob('nvidia[0-9]*')),'CPU only')
 import numpy as np
 from PIL import Image
 require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','serializer pins')
 cfgpath=ROOT/'docs/lighton4b-physical-native-ops-ablation-v1.json';cfg=json.loads(cfgpath.read_text());admit_protocol(cfg);out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'native-ops-ablation';scratch.mkdir();active=None
 base=module('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');common=module('native_base',ROOT/'tools/ci-heavy/lighton4b_native_document.py');vision=module('vision_gates',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py');sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
 report={'schema':cfg['schema'],'status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(cfgpath),'full_completion_executed':False,'quality_qualified':False,'upstream_forward_executed':False,'arms':{},'resources':{}}
 def save(): (out/'native-ops-ablation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 def interrupted(signum,frame):base.stop(active);save();raise SystemExit(128+signum)
 for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(signum,interrupted)
 signal.alarm(cfg['total_model_timeout_seconds'])
 def command(label,args,env=None,timeout=1200):
  nonlocal active
  report['phase']=label;save();print('phase',label,flush=True);start=time.monotonic();peak=0;code=None
  with(out/(label+'.log')).open('w')as log:
   try:
    active=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    while True:
     require(time.monotonic()-start<timeout,label+' deadline')
     try:rows=dict(x.split(':',1)for x in Path('/proc/'+str(active.pid)+'/status').read_text().splitlines());peak=max(peak,int(rows.get('VmHWM','0').split()[0])*1024)
     except(FileNotFoundError,ProcessLookupError):pass
     try:code=active.wait(timeout=1);break
     except subprocess.TimeoutExpired:pass
    require(code==0,label+' nonzero')
   finally:
    base.stop(active);report['resources'][label]={'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes_observed':peak,'exit_code':None if active is None else active.poll()};active=None;save()
 def verify(path,row):require(path.is_file()and path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'captured input hash: '+path.name)
 try:
  require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact worker source');report['preflight']={'memory':base.memory(),'scratch_free':shutil.disk_usage(scratch).free,'output_free':shutil.disk_usage(out).free,'load':os.getloadavg()};require(report['preflight']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and report['preflight']['scratch_free']>=cfg['minimum_scratch_free_bytes']and report['preflight']['output_free']>=cfg['minimum_output_free_bytes'],'RAM/disk guard before acquisition')
  candidates=[p for p in Path('/kaggle/input').rglob('physical-native-ops-smoke.json')if p.stat().st_size<4*1024**2 and sha(p)==cfg['golden_smoke_report_sha256']];require(len(candidates)==1,'one exact completed smoke input');gp=candidates[0];gold=json.loads(gp.read_text());require(gold['source_sha']==cfg['golden_smoke_source_sha']and gold['status']=='measured'and gold['candidate_sha256']==cfg['expected_candidate_sha256'],'closed actual smoke source/weights/status');folder=gp.parent/'simple_form';art=gold['cases']['simple_form']['artifacts'];report['golden_report_sha256']=sha(gp)
  for rel in ['input.rgb','original.json','off/prompt.ids','off/modalities.ids','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32']:verify(folder/rel,art[rel])
  case=cfg['cases'][0];imagepath=ROOT/case['image'];require(sha(imagepath)==case['image_sha256'],'frozen original PNG');image=Image.open(imagepath).convert('RGB');rgb=folder/'input.rgb';require(rgb.read_bytes()==image.tobytes(),'actual captured RGB matches frozen image');ids=list(map(int,(folder/'off/prompt.ids').read_text().split()));buf=io.BytesIO();np.save(buf,np.asarray([ids],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(buf.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'complete original prompt IDs');original=json.loads((folder/'original.json').read_text());teacher=scratch/'one-teacher.ids';teacher.write_text(str(original['generated_ids'][0])+'\n');feature=folder/'observer-on/source.vision.0.f32';source_score=np.fromfile(folder/'observer-on/source.logits.0.f32',dtype='<f4');require(source_score.size==248320 and np.isfinite(source_score).all(),'actual source firstscore')
  command('host-cpu',['lscpu']);command('submodules',['git','-C',ROOT,'submodule','update','--init','ggml']);build=scratch/'build';command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);command('build',['cmake','--build',build,'--target','lighton4b-native-document','--parallel','4']);shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt');cache=(out/'CMakeCache.txt').read_text()
  for flag in ['GGML_LLAMAFILE:BOOL=ON','GGML_NATIVE:BOOL=OFF','GGML_AVX2:BOOL=ON','GGML_AVX512:BOOL=OFF','GGML_BLAS:BOOL=OFF']:require(flag in cache,'matched build profile: '+flag)
  native=build/'lighton4b-native-document';report['binary_sha256']=sha(native)
  row=cfg['models']['Q4_K_M'];up=scratch/row['file'];base.download('https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/'+row['file'],up,row['bytes'],row['sha256']);row=cfg['base_model'];protected=scratch/row['file'];base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=scratch/'physical-native.gguf';audit=pack.convert(protected,up,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['expected_candidate_sha256'],'exact lossless candidate');report['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update({k:'1'for k in common.GATES+vision.VISION_GATES});env.update(CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL='1',CRISPEMBED_LIGHTON4B_Q6_K='1',CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1',CRISPEMBED_LIGHTON4B_HEAD_BF16='0',CRISPEMBED_LIGHTON4B_BF16_LINEAR='0',CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS='0')
  for arm in cfg['arms']:
   require(base.memory()['MemAvailable']>=cfg['minimum_available_memory_bytes'],'per-arm RAM guard');dest=out/arm;dest.mkdir();child=dict(env);child['CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS']='0'if arm=='off'else'1';child['CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION']=arm;command(arm,[native,candidate,rgb,image.height,image.width,teacher,dest,2,4,'greedy-full','--upstream-vision',feature],child,cfg['native_arm_timeout_seconds'])
   current={rel:sha(dest/rel)for rel in STAGES};anchor=None
   if arm in ['off','all']:
    old='off'if arm=='off'else'on';endpoint={rel:art[old+'/'+rel]for rel in STAGES};check_anchor(arm,current,endpoint);anchor=True
   require((dest/'prompt.ids').read_bytes()==(folder/'off/prompt.ids').read_bytes()and sha(dest/'vision.merged.f32')==art['observer-on/source.vision.0.f32']['sha256'],'actual identical prompt/features');require(sha(dest/'prefill.input.f32')==art['off/prefill.input.f32']['sha256']and sha(dest/'prefill.positions.f32')==art['off/prefill.positions.f32']['sha256'],'actual identical hidden input/positions')
   scores=np.fromfile(dest/'greedy.0.logits.f32',dtype='<f4');require(scores.size==248320 and np.isfinite(scores).all(),'finite full firstscore');delta=scores.astype('float64')-source_score.astype('float64');generated=list(map(int,(dest/'generated.ids').read_text().split()));summary=json.loads((dest/'native-summary.json').read_text());require(generated and len(generated)<=2 and generated[0]==int(scores.argmax())and summary['eos']==(generated[-1]in[248044,248046])and(summary['eos']or len(generated)==2),'actual argmax/EOG/cap')
   report['arms'][arm]={'gates':{k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},'golden38stage_anchor_exact':anchor,'stage_sha256':current,'firstscore_sha256':sha(dest/'greedy.0.logits.f32'),'rms':float(np.sqrt(np.mean(delta*delta))),'max_abs':float(abs(delta).max()),'ids':generated,'decoded':(dest/'generated.txt').read_bytes().decode('utf8'),'eos':summary['eos'],'cap_hit':not summary['eos']and len(generated)==2,'resources':report['resources'][arm]};save()
  require(sha(candidate)==cfg['expected_candidate_sha256']and sha(up)==cfg['models']['Q4_K_M']['sha256'],'model bytes unchanged');report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for arm in cfg['arms']for p in(out/arm).rglob('*')if p.is_file()};report['status']='measured_prefix_ablations';candidate.unlink()
 except Exception as error:report['error_type']=type(error).__name__;report['error']=str(error);raise
 finally:base.stop(active);save()
if __name__=='__main__':main()
