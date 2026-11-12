#!/usr/bin/env python3
"""Hosted model-free candidate build; retain exact tested runtime bundle for A/B."""
import argparse,hashlib,json,os,re,shutil,signal,subprocess,time
from pathlib import Path
TARGETS=['test-lighton4b-source-prefill-chunks','test-lighton4b-physical-swiglu','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-native-ops','test-lighton4b-physical-f16-attention']
FLAGS={'CMAKE_BUILD_TYPE':'Release','CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF','GGML_METAL':'OFF','CRISPEMBED_BUILD_SHARED':'OFF'}
GGML='3a27b6c665b191cb230be367db5c52d81dd3e2ac'
NEW_MARKER='physical SwiGLU: gate aliases, physical/BF16 scope, six GGML exact arms, finite/geometry controls pass'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);a=p.parse_args();require(re.fullmatch('[0-9a-f]{40}',a.source),'immutable candidate source');out=Path('/kaggle/working');require(out.is_dir(),'hosted worker only');work=Path('/tmp/lighton4b-swiglu-controls');work.mkdir();repo=work/'source';start=time.monotonic();active=None;r=dict(schema='lighton4b-physical-swiglu-controls/v1',source_sha=a.source,passed=False,checkpoint_loaded=False,gpu_model_inference_executed=False,commands=[],unit_control_outputs={})
 def save():(out/'swiglu-controls.json').write_text(json.dumps(r,indent=2)+'\n')
 def stop():
  nonlocal active
  if active is not None and active.poll()is None:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
  active=None
 def interrupted(sig,frame):stop();save();raise SystemExit(128+sig)
 for sig in[signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
 signal.alarm(3300);env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')};env.update(GIT_TERMINAL_PROMPT='0',GIT_CONFIG_NOSYSTEM='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run(label,args,cwd=None,seconds=600,expected=0):
  nonlocal active
  r['phase']=label;entry=dict(label=label,argv=list(map(str,args)));r['commands'].append(entry);save();log=out/(label+'.log');t=time.monotonic()
  with log.open('w')as f:
   try:active=subprocess.Popen(list(map(str,args)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rc=active.wait(timeout=min(seconds,3300-(time.monotonic()-start)-30))
   finally:stop()
  entry.update(returncode=rc,seconds=time.monotonic()-t,sha256=sha(log));save();require(rc==expected,'actual command '+label);return log.read_text(errors='replace')
 try:
  require(shutil.disk_usage(out).free>=12<<30,'bounded control disk');run('init',['git','init',repo]);run('remote',['git','-C',repo,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']);run('fetch',['git','-C',repo,'fetch','--depth','1','origin',a.source]);run('checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD']);require(run('source',['git','rev-parse','HEAD'],repo).strip()==a.source,'actual candidate');run('ggml',['git','submodule','update','--init','--depth','1','ggml'],repo);require(run('ggml-source',['git','rev-parse','HEAD'],repo/'ggml').strip()==GGML,'exact ggml')
  build=work/'build';run('configure',['cmake','-S',repo,'-B',build]+['-D'+k+'='+v for k,v in FLAGS.items()]);cache=build/'CMakeCache.txt';lines=cache.read_text().splitlines();require(all(k=='CMAKE_BUILD_TYPE'or f'{k}:BOOL={v}'in lines for k,v in FLAGS.items()),'actual profile flags');run('baseline',['python3',repo/'scripts/check-cpu-baseline.py',build]);run('build',['cmake','--build',build,'-j2','--target',*TARGETS,'lighton4b-native-document'],seconds=2400);abi=(build/'CMakeFiles/crispembed-qwen35-delta.dir/flags.make').read_text();require('CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI=1'in abi,'supported physical ABI')
  for target in TARGETS:
   text=run(target,[build/target],seconds=120);require('skip'not in text.lower()and 'pass'in text.lower(),'actual supported control marker');r['unit_control_outputs'][target]=text.strip()
  require(NEW_MARKER in r['unit_control_outputs']['test-lighton4b-physical-swiglu'],'six exact fused arms');native=build/'lighton4b-native-document';run('cli-observed512',[native,work/'missing.gguf',work/'missing.rgb',1,1,work/'missing.ids',out,512,4,'greedy-full','--observed-composed-cap512'],seconds=30,expected=4)
  shutil.copyfile(cache,out/'CMakeCache.txt');r['cmake_cache']=dict(bytes=cache.stat().st_size,sha256=sha(cache));bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin/lighton4b-native-document');libs=[f for f in(build/'bin').glob('*.so*')if f.is_file()];require(1<=len(libs)<=19,'runtime library census')
  for f in libs:shutil.copyfile(f,bundle/'lib'/f.name)
  files=[f for f in bundle.rglob('*')if f.is_file()];require(sum(f.stat().st_size for f in files)<=256<<20,'bounded runtime bundle');r['runtime_bundle']={str(f.relative_to(bundle)):dict(bytes=f.stat().st_size,sha256=sha(f))for f in files};r['native_binary_sha256']=sha(native);r['passed']=True
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:signal.alarm(0);stop();r['seconds']=time.monotonic()-start;save()
if __name__=='__main__':main()
