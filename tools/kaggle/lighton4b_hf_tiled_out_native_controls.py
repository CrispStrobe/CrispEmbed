#!/usr/bin/env python3
"""Kaggle CPU bootstrap template: synthetic controls only, no checkpoints."""
import hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE_SHA='__SOURCE_SHA__'
GGML_SHA='3a27b6c665b191cb230be367db5c52d81dd3e2ac'
base=Path('/kaggle/working');out=base/'lighton-hf-tiled-out-native-controls';out.mkdir(exist_ok=True);scratch=base/'lighton-hf-tiled-out-native-controls-scratch';source=base/'lighton-hf-tiled-out-native-controls-source';created=[];active=None
r={'schema':'lighton4b-hf-tiled-out-native-controls-bootstrap/v1','status':'failed','source_sha':SOURCE_SHA,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_downloaded':False,'model_inference_executed':False,'gpu_enabled':False};started=time.monotonic()
def interrupted(signum,frame):
 if active is not None and active.poll() is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 raise TimeoutError('bounded synthetic controls deadline')

def run(label,args,timeout=240,env=None):
 global active
 print('PHASE '+label,flush=True)
 with(out/(label+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=active.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
   raise
  finally:active=None
 if code:raise RuntimeError('bounded controls stage failed: '+label)
try:
 assert re.fullmatch('[0-9a-f]{40}',SOURCE_SHA)and Path('/kaggle/input').is_dir()and base.is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(25*60)
 assert shutil.disk_usage(base).free>=6*2**30
 memory={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapTotal','SwapFree')};available=memory['MemAvailable']
 if Path('/sys/fs/cgroup/memory.max').exists():
  limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if limit!='max':available=min(available,max(0,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())))
 assert available>=6*2**30
 r['available_ram_bytes']=available
 for executable in ('git','cmake','cc','c++'):assert shutil.which(executable),'missing build executable'
 r['preflight']={'free_disk_bytes':shutil.disk_usage(base).free,'load':os.getloadavg(),'logical_cpus':os.cpu_count(),'memory_swap':{k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapTotal','SwapFree')}}
 for path in(source,scratch):assert not path.exists();path.mkdir();created.append(path)
 for label,args in [('init',['git','init',source]),('remote',['git','-C',source,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']),('fetch',['git','-C',source,'fetch','--depth','1','origin',SOURCE_SHA]),('checkout',['git','-C',source,'checkout','--detach','FETCH_HEAD']),('submodule',['git','-C',source,'submodule','update','--init','--depth','1','ggml'])]:run(label,args)
 assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==SOURCE_SHA
 assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD:ggml'],text=True).strip()==GGML_SHA and subprocess.check_output(['git','-C',source/'ggml','rev-parse','HEAD'],text=True).strip()==GGML_SHA
 venv=scratch/'venv';run('venv',[sys.executable,'-m','venv','--without-pip','--system-site-packages',venv]);python=venv/'bin/python';run('venv-prefix',[python,'-c','import sys,pip; assert sys.prefix!=sys.base_prefix']);run('packages',[python,'-m','pip','install','--disable-pip-version-check','numpy==2.3.5','gguf==0.19.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5'],300)
 run('dependency-preflight',[python,'-c',"import importlib.metadata as m;import numpy,gguf,yaml,tqdm,requests;from gguf.quants import dequantize;from gguf import GGUFReader;assert all(m.version(k)==v for k,v in {'numpy':'2.3.5','gguf':'0.19.0','PyYAML':'6.0.3','tqdm':'4.67.1','requests':'2.32.5'}.items())"])
 for rel in ('src/lighton4b_hf_tiled_out.h','tools/lighton4b_hf_tiled_out_pack.py','tools/ci-heavy/lighton4b_hf_tiled_out_native_controls.py','tests/test_lighton4b_hf_tiled_out_recipe.cpp','tests/test_lighton4b_hf_tiled_out_operator.cpp'):assert(source/rel).is_file()
 env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(CRISPEMBED_BUILD_MODE='portable');build=scratch/'build';run('configure',['cmake','-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_CUDA=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DCMAKE_C_FLAGS=-ffp-contract=off','-DCMAKE_CXX_FLAGS=-ffp-contract=off'],env=env);run('build',['cmake','--build',build,'--target','test-lighton4b-hf-tiled-out-recipe','test-lighton4b-hf-tiled-out-operator','test-lighton4b-imatrix-q4km-recipe','test-lighton4b-physical-native-ops','test-lighton4b-q6-tied-head','test-qwen35-stream-projection','test-lighton4b-release-fence','lighton4b-native-document','lighton4b-native-case-preflight','--parallel','2'],1000,env);shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt')
 run('python-controls',[python,'-m','unittest','discover','-s',source/'tests','-p','test_lighton4b_hf_tiled_out*.py'],env=env)
 run('environment-inventory',[python,source/'tools/list_env_vars.py','--check'],env=env)
 pins=[json.loads((source/('tests/results/qwen35-ocr/quality-quant-'+p+'-inventory-table.json')).read_text())['source_commit']for p in('q8','q8-q6','q4')]
 run('fetch-frozen-receipts',['git','-C',source,'fetch','--depth','1','origin',*sorted(set(pins))]);run('source-pins',[python,source/'tools/qwen35_source_pins.py','--check','--verify-git'],env=env)
 run('native-controls',[python,source/'tools/ci-heavy/lighton4b_hf_tiled_out_native_controls.py'],600,dict(env,HEAVY_OUT=str(out),HEAVY_BUILD=str(build)))
 controls=json.loads((out/'native-controls.json').read_text());assert controls['status']=='passed'and controls['model_payload_bytes']==0
 bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();
 for name in('lighton4b-native-document','lighton4b-native-case-preflight'):
  shutil.copyfile(build/name,bundle/'bin'/name);(bundle/'bin'/name).chmod(0o755)
 for path in build.rglob('libggml*.so*'):
  if path.is_file():shutil.copyfile(path,bundle/'lib'/path.name)
 build.rename(scratch/'build-hidden-after-tests')
 for name in('lighton4b-native-document','lighton4b-native-case-preflight'):
  probe=subprocess.run([str(bundle/'bin'/name)],env=dict(env,LD_LIBRARY_PATH=str(bundle/'lib')),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30);assert probe.returncode==2
 manifest={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in bundle.rglob('*')if p.is_file()};(out/'runtime-manifest.json').write_text(json.dumps({'source_sha':SOURCE_SHA,'ggml_sha':GGML_SHA,'files':manifest},indent=2,sort_keys=True)+'\n');r.update(status='passed',checks=controls['checks'],runtime_manifest_sha256=hashlib.sha256((out/'runtime-manifest.json').read_bytes()).hexdigest())
except Exception as error:r['error_type']=type(error).__name__;r['error']='bounded synthetic controls failed';raise
finally:
 signal.alarm(0)
 r['seconds']=time.monotonic()-started;(out/'bootstrap-receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 for path in reversed(created):shutil.rmtree(path)
