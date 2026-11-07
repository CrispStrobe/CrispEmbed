#!/usr/bin/env python3
"""Closed source-fed GDN primitive comparison; no model loading or decode."""
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(x,s):
 if not x:raise RuntimeError(s)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def admit(cfg):
 require(hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='024361fbc8e40243c38e5119491aca03854b0b0ca128b79f721f41b519e2e10d','immutable predeclared protocol')
 require(cfg['schema']=='lighton4b-gdn-source-fed-primitives/v1'and cfg['tokens']==[4,16],'closed GDN input scope')
 require((cfg['key_heads'],cfg['value_heads'],cfg['width'],cfg['conv_kernel'],cfg['threads'])==(16,32,128,4,4),'exact model geometry')
 require(cfg['fused_gdn_ar']is True and cfg['fused_gdn_ch']is True and cfg['auto_fgdn']is False,'actual upstream fused defaults')
 require(not cfg['model_forward_allowed']and 0<cfg['total_timeout_seconds']<=1800,'operator-only budget')
 require(sum(x['bytes']for x in cfg['tensors'])==6030080 and len(cfg['tensors'])==5,'closed six-MB source tensor spans')
 require(set(x['physical_name']for x in cfg['tensors'])=={'blk.0.ssm_conv1d.weight','blk.0.ssm_a','blk.0.ssm_dt.bias','blk.0.ssm_norm.weight','blk.0.ssm_out.weight'},'fixed primitive tensor selection')
def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir(),'Kaggle CPU only')
 import numpy as np
 require(np.__version__=='2.3.5','numpy pin')
 cfgpath=ROOT/'tools/gdn-primitive-probe/protocol-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg)
 spec=importlib.util.spec_from_file_location('resources',ROOT/'tools/ci-heavy/lighton4b_upstream_physical_audit.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True);scratch=Path(os.environ['HEAVY_SCRATCH'])/'gdn-primitives';scratch.mkdir();active=None
 report={'schema':cfg['schema'],'status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(cfgpath),'actual_native_trace_compared':False,'actual_source_full_graph_captured':False,'model_forward_executed':False,'full_f32_model_created':False,'quality_qualified':False,'preflight':{'memory':helper.resources(),'disk_free':shutil.disk_usage(scratch).free,'load':os.getloadavg()},'runs':{},'comparisons':{},'processes':{}}
 def save(phase=None):
  if phase:report['phase']=phase;print('phase',phase,flush=True)
  (out/'gdn-primitives.json').write_text(json.dumps(report,indent=2)+'\n')
 def stop():
  nonlocal active
  if active is not None and active.poll()is None:
   try:os.killpg(active.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:active.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
  active=None
 def interrupted(signum,frame):report['signal']=signum;stop();save();raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(cfg['total_timeout_seconds'])
 def run(label,args,timeout=600):
  nonlocal active
  save(label);start=time.monotonic();process=None
  try:
   with(out/(label+'.log')).open('w')as f:
    active=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True);process=active;require(active.wait(timeout=timeout)==0,label+' failed')
  finally:
   stop();report['processes'][label]={'seconds':time.monotonic()-start,'returncode':process.returncode if process else None};save()
 def fetch(row,target):
  start=row['source_offset'];size=row['bytes'];req=urllib.request.Request(cfg['source_url'],headers={'Range':f'bytes={start}-{start+size-1}'})
  h=hashlib.sha256();count=0
  with urllib.request.urlopen(req,timeout=120)as r,target.open('xb')as f:
   require(r.status==206 and r.headers.get('Content-Range')==f"bytes {start}-{start+size-1}/{cfg['source_bytes']}",'exact authenticated source range')
   while b:=r.read(1024*1024):count+=len(b);require(count<=size,'range bound');h.update(b);f.write(b)
  require(count==size and h.hexdigest()==row['source_packed_sha256'],'whole source tensor SHA')
 def diff(a,b):
  x=np.fromfile(a,dtype='<f4');y=np.fromfile(b,dtype='<f4');require(x.shape==y.shape and x.size and np.isfinite(x).all()and np.isfinite(y).all(),'same finite stage shape');d=x.astype(np.float64)-y.astype(np.float64)
  return {'bit_exact':x.tobytes()==y.tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max_abs':float(np.abs(y).max()),'elements':x.size}
 try:
  require(report['source_sha']==os.environ['GDN_EXPECTED_SOURCE'],'exact worker source');require(report['preflight']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and report['preflight']['disk_free']>=cfg['minimum_scratch_bytes'],'2GiB effective RAM/4GiB disk')
  candidates=[p for p in Path('/kaggle/input').rglob('physical-decode.json')if p.stat().st_size<4*1024**2 and sha(p)==cfg['physical_result_sha256']];require(len(candidates)==1,'exact completed physical capture dependency');capture_root=candidates[0].parent;capture=json.loads(candidates[0].read_text());require(capture['status']=='measured'and capture['source_sha']==cfg['physical_source'],'actual physical source report')
  inputs=capture_root/'simple_form/upstream_features';report['admitted_capture_sha256']={}
  for name,meta in cfg['actual_captures'].items():
   p=capture_root/'simple_form'/name;require(p.stat().st_size==meta['bytes']and sha(p)==meta['sha256'],'actual captured input/attention SHA');report['admitted_capture_sha256'][name]=meta
  require(sha(ROOT/'src/qwen35_delta.cpp')==cfg['native_delta_sha256'],'unchanged actual native GDN helper source')
  weights=scratch/'weights';weights.mkdir();save('authenticate-five-small-parameter-spans')
  for row in cfg['tensors']:fetch(row,weights/row['file'])
  up=scratch/'upstream'
  for label,args in [('up-init',['git','init',up]),('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp']),('up-fetch',['git','-C',up,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'pinned upstream source')
  for name,h in cfg['source_function_sha256'].items():require(sha(up/name)==h,'source blueprint bytes')
  run('native-submodule',['git','-C',ROOT,'submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_commit'],'pinned native ggml')
  flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']
  native=scratch/'native-build';source=scratch/'source-build';run('native-configure',['cmake','-S',ROOT,'-B',native,*flags,'-DCRISPEMBED_NATIVE=OFF']);run('native-build',['cmake','--build',native,'--target','test-lighton4b-gdn-source-primitive','--parallel','2'],900)
  run('source-configure',['cmake','-S',ROOT/'tools/gdn-primitive-probe','-B',source,*flags,'-DGGML_SOURCE='+str(up/'ggml')]);run('source-build',['cmake','--build',source,'--target','gdn-source-probe','--parallel','2'],900)
  for label,path in [('native',native),('source',source)]:
   cache=(path/'CMakeCache.txt').read_text()
   for flag in ['GGML_NATIVE:BOOL=OFF','GGML_AVX2:BOOL=ON','GGML_AVX512:BOOL=OFF','GGML_LLAMAFILE:BOOL=ON','GGML_BLAS:BOOL=OFF']:require(flag in cache,'matched build flags')
   shutil.copyfile(path/'CMakeCache.txt',out/(label+'-CMakeCache.txt'))
  nb=native/'test-lighton4b-gdn-source-primitive';sb=source/'gdn-source-probe';report['binary_sha256']={'native':sha(nb),'upstream':sha(sb)}
  for n in cfg['tokens']:
   for name,binary,mode in [('native',nb,'native'),('source-native-ggml',nb,'source'),('source-upstream-ggml',sb,'source')]:
    dest=out/(name+'-'+str(n));dest.mkdir();run(dest.name,[binary,mode,n,inputs,weights,dest]);report['runs'][dest.name]={'control':json.loads((dest/'probe-control.json').read_text()),'stages':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in dest.glob('*.f32')}}
   legacy=out/('native-'+str(n));reference=out/('source-upstream-ggml-'+str(n));oldgraph=out/('source-native-ggml-'+str(n));report['comparisons'][str(n)]={'native_vs_source':{},'backend_control':{}}
   for p in reference.glob('*.f32'):
    if (legacy/p.name).exists():report['comparisons'][str(n)]['native_vs_source'][p.name]=diff(legacy/p.name,p)
    require((oldgraph/p.name).exists(),'same source graph version-control stages');report['comparisons'][str(n)]['backend_control'][p.name]=diff(oldgraph/p.name,p)
   # Projection control uses identical NATIVE pre-out values through upstream matmul.
   same=out/('same-native-values-upstream-projection-'+str(n));same.mkdir();run(same.name,[sb,'out',n,legacy,weights,same]);report['comparisons'][str(n)]['same_native_input_projection']={p.name:diff(legacy/p.name,p)for p in same.glob('*.f32')}
   if n==16:require(report['runs']['native-16']['control']['actual_main_attention_bit_exact'],'actual main full16 native GDN/out control');report['actual_native_trace_compared']=True
   save('measured-'+str(n))
  report['status']='measured';report['all_parameter_span_hashes']={x['file']:sha(weights/x['file'])for x in cfg['tensors']}
 except BaseException as error:report['error_type']=type(error).__name__;report['error']='Primitive phase failed; preserve exact controls and logs';raise
 finally:stop();save()
if __name__=='__main__':main()
