#!/usr/bin/env python3
"""Bounded artifact-only Kaggle consumer; no model forward, conversion or compilation."""
import hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE_SHA='__SOURCE_SHA__'
MODEL_RECEIPT_SHA='__MODEL_RECEIPT_SHA__'
base=Path('/kaggle/working');out=base/'lighton-f16-teacher-readback';source=base/'lighton-f16-teacher-readback-source';scratch=base/'lighton-f16-teacher-readback-scratch';out.mkdir(exist_ok=True);created=[];active=None;started=time.monotonic();r={'schema':'lighton4b-f16-teacher-readback-bootstrap/v1','status':'failed','source_sha':SOURCE_SHA,'model_receipt_sha256':MODEL_RECEIPT_SHA,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_inference_executed':False,'weight_quantization_executed':False,'large_artifacts_copied':False,'gpu_enabled':False}
def stop(*args):
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 raise TimeoutError('bounded read-only artifact consumer deadline')
def run(label,args,timeout=240,env=None):
 global active
 print('PHASE '+label,flush=True)
 with(out/(label+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:code=active.wait(timeout=timeout)
  except subprocess.TimeoutExpired:stop()
  finally:active=None
 if code:raise RuntimeError('bounded reader stage failed: '+label)
try:
 assert re.fullmatch('[0-9a-f]{40}',SOURCE_SHA)and re.fullmatch('[0-9a-f]{64}',MODEL_RECEIPT_SHA)and base.is_dir()and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and shutil.which('git')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(25*60)
 available=int(next(v for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k=='MemAvailable').split()[0])*1024
 if Path('/sys/fs/cgroup/memory.max').exists():
  limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if limit!='max':available=min(available,max(0,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())))
 assert available>=4*2**30 and shutil.disk_usage(base).free>=2**30
 r['preflight']={'available_ram_bytes':available,'free_disk_bytes':shutil.disk_usage(base).free,'logical_cpus':os.cpu_count(),'load':os.getloadavg()}
 for path in(source,scratch):assert not path.exists();path.mkdir();created.append(path)
 for label,args in [('init',['git','init',source]),('remote',['git','-C',source,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']),('fetch',['git','-C',source,'fetch','--depth','1','origin',SOURCE_SHA]),('checkout',['git','-C',source,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
 assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==SOURCE_SHA
 for rel in('tools/ci-heavy/lighton4b_f16_teacher_readback.py','tests/test_lighton4b_f16_teacher_readback.py','tools/gguf_file_type_canonicalize.py','docs/lighton4b-f16-teacher-readback-v1.json'):assert(source/rel).is_file()
 venv=scratch/'venv';run('venv',[sys.executable,'-m','venv','--without-pip','--system-site-packages',venv]);python=venv/'bin/python';run('venv-prefix',[python,'-c','import sys,pip;assert sys.prefix!=sys.base_prefix']);run('numpy',[python,'-m','pip','install','--disable-pip-version-check','numpy==2.3.5'],300);run('dependency-preflight',[python,'-c',"import numpy;assert numpy.__version__=='2.3.5'"])
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2');run('causal-row-controls',[python,source/'tests/test_lighton4b_f16_teacher_readback.py'],60,env)
 run('actual-artifact-readback',[python,source/'tools/ci-heavy/lighton4b_f16_teacher_readback.py','--mount','/kaggle/input','--output',out/'results'],20*60,env)
 p=out/'results/teacher-readback.json';measured=json.loads(p.read_text());assert measured['status']=='readback-verified'and measured['native_receipt_sha256']==MODEL_RECEIPT_SHA and len(measured['rows'])==177 and measured['teacher_contexts_authenticated'];r.update(status='readback-verified',readback_receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),candidate_sha256=measured['candidate_sha256'])
except Exception as error:r['error_type']=type(error).__name__;r['error']='bounded readback failed; negative preserved';raise
finally:
 signal.alarm(0);r['seconds']=time.monotonic()-started;(out/'bootstrap-receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 for path in reversed(created):shutil.rmtree(path)
