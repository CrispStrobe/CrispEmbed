#!/usr/bin/env python3
"""Bounded CPU operator-only comparison of actual quantized buffer routes."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE='26763167f70e2dce6cf187caacc5164e95d01432'
WORK=Path('/tmp/lighton4b-runtime-q4-repack-controls-v1');OUT=Path('/kaggle/working');REPORT=OUT/'runtime-repack-bootstrap.json';START=time.monotonic();active=None
state={'schema':'lighton4b-source-repack-operator-bootstrap/v1','source_sha':SOURCE,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'failed','gpu_requested':False,'model_forward_executed':False,'model_quality_qualified':False}
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(2600)
def run(name,args,env=None,timeout=120,cwd=None):
 global active
 remaining=2600-(time.monotonic()-START)
 if remaining<=0:raise TimeoutError('whole bootstrap bound')
 state['phase']=name;save();print('PHASE '+name,flush=True)
 with(OUT/(name+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,cwd=cwd,start_new_session=True)
  try:code=active.wait(timeout=min(timeout,remaining))
  finally:stop()
 if code:raise RuntimeError(name+' nonzero')
try:
 if not OUT.is_dir()or not Path('/kaggle/input').is_dir()or list(Path('/dev').glob('nvidia[0-9]*')):raise RuntimeError('Kaggle CPU-only worker required')
 for executable in('git','cmake','make','c++','cc'):
  if not shutil.which(executable):raise RuntimeError('required executable missing: '+executable)
 if shutil.disk_usage(OUT).free<8*2**30:raise RuntimeError('8GiB free required before source/build staging')
 WORK.mkdir();repo=WORK/'source'
 for name,args in [('source-init',['git','init',repo]),('source-remote',['git','-C',repo,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']),('source-fetch',['git','-C',repo,'fetch','--depth','1','origin',SOURCE]),('source-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])]:run(name,args)
 if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=SOURCE:raise RuntimeError('exact source differs')
 for path in('tools/repack-probe/runtime_worker.py','tests/lighton4b_physical_q4_repack_probe.cpp','tests/test_lighton4b_physical_q4_repack.cpp','tools/repack-probe/worker.py','tools/repack-probe/probe.cpp','tools/repack-probe/CMakeLists.txt','tools/repack-probe/test_admission.py','docs/lighton4b-source-repack-operator-v1.json','tools/physical-f16-attention-probe/runtime_guard.py','tools/physical-f16-attention-probe/test_runtime_guard.py','tools/kaggle/lighton4b_upstream_physical_decode.py'):
  if not(repo/path).is_file():raise RuntimeError('required source path missing: '+path)
 state['source_file_sha256']={path:hashlib.sha256((repo/path).read_bytes()).hexdigest()for path in('tools/repack-probe/runtime_worker.py','tests/lighton4b_physical_q4_repack_probe.cpp','tests/test_lighton4b_physical_q4_repack.cpp','tools/repack-probe/worker.py','tools/repack-probe/probe.cpp','tools/repack-probe/CMakeLists.txt','tools/repack-probe/test_admission.py','docs/lighton4b-source-repack-operator-v1.json')};save()
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update(CRISPEMBED_BUILD_MODE='portable',HEAVY_OUT=str(OUT),HEAVY_SCRATCH=str(WORK),LIGHTON4B_EXPECTED_SOURCE=SOURCE,PYTHONUNBUFFERED='1',PIP_NO_CACHE_DIR='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5'],env,90);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env);run('dependency-profile',[py,'-c','import sys,numpy;assert sys.prefix!=sys.base_prefix;assert numpy.__version__=="2.3.5"'],env)
 cfg=json.loads((repo/'docs/lighton4b-source-repack-operator-v1.json').read_text());receipts=[p for p in Path('/kaggle/input').rglob('physical-f16-table512.json')if p.is_file()and p.stat().st_size<4*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==cfg['actual_receipt_sha256']]
 if len(receipts)!=1:raise RuntimeError('one exact completed actual512 dependency required')
 receipt=json.loads(receipts[0].read_text())
 if receipt['status']!='measured_extended_table'or receipt['source_sha']!='f789609c2f1ff1a9527c8d02545d096ed10446b0'or not receipt['old_cap256_prefix_anchors_exact']or not receipt['source_observer_full_neutral']:raise RuntimeError('actual dependency source/status')
 closed=[p for p in Path('/kaggle/input').rglob('repack-operator.json')if p.is_file()and p.stat().st_size<4*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()=='4dc28c17717082a36e9f2cad3729a31a83a97c3234b07d3c00002232744a66f5']
 if len(closed)!=1:raise RuntimeError('one exact sealed operator dependency required')
 for name,path in [('runtime-guard','tools/physical-f16-attention-probe/test_runtime_guard.py'),('repack-admission','tools/repack-probe/test_admission.py')]:run('control-'+name,[py,repo/path],env,30,cwd=repo)
 # Reserve the whole worker2400s plus cleanup before the2600s bootstrap bound.
 if 2600-(time.monotonic()-START)<2410:raise TimeoutError('setup exceeded reserved worker budget')
 worker=repo/'tools/repack-probe/runtime_worker.py';run('runtime-repack-controls',[py,'-u',worker],env,2410,cwd=repo)
 receipt=json.loads((OUT/'runtime-repack-controls.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='passed_operator_cache_controls'or receipt['source_sha']!=SOURCE or len(receipt['sourcefed'])!=30 or receipt['model_forward_executed']:raise RuntimeError('complete source-fed cache controls missing')
 state['status']='completed'
except Exception as error:state['error_type']=type(error).__name__;state['error']=str(error)
finally:
 stop();signal.alarm(0);state['elapsed_seconds']=time.monotonic()-START;save();shutil.rmtree(WORK,ignore_errors=True);print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
