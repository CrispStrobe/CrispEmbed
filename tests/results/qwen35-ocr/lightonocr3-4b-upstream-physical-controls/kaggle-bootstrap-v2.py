#!/usr/bin/env python3
"""Authorized CPU bootstrap, exact independent upstream OCR comparison."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
SOURCE='0610af26d9cb8972ecdfe55204e2306e9e740d5e'
WORK=Path('/tmp/lighton4b-upstream-physical-controls-v1')
OUT=Path('/kaggle/working')
REPORT=OUT/'physical-controls-bootstrap.json'
state={'schema':'lighton4b-physical-controls-kaggle-bootstrap/v1','source_sha':SOURCE,'status':'failed','gpu_requested':False,'original_forward_executed':False,'model_quality_qualified':False}
active=None
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
def run(name,args,env=None,timeout=1200,cwd=None):
 global active
 state['phase']=name;save();print('phase',name,flush=True)
 with(OUT/(name+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,cwd=cwd,start_new_session=True)
  try:code=active.wait(timeout=timeout)
  finally:stop()
 if code:raise RuntimeError(name+' nonzero')
try:
 if not OUT.is_dir()or not Path('/kaggle/input').is_dir():raise RuntimeError('authorized Kaggle CPU worker required')
 WORK.mkdir();repo=WORK/'source'
 run('source-init',['git','init',repo]);run('source-remote',['git','-C',repo,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']);run('source-fetch',['git','-C',repo,'fetch','--depth','1','origin',SOURCE]);run('source-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])
 if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=SOURCE:raise RuntimeError('exact source differs')
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update({'CRISPEMBED_BUILD_MODE':'portable','HEAVY_OUT':str(OUT),'HEAVY_SCRATCH':str(WORK),'PYTHONUNBUFFERED':'1','PIP_NO_CACHE_DIR':'1'})
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5'],env);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env)
 # The worker enforces cgroup-aware RAM/disk guards before its upstream build/model downloads.
 worker=repo/'tools/kaggle/lighton4b_upstream_physical_controls.py';state['worker_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest();state['protocol_sha256']=hashlib.sha256((repo/'docs/lighton4b-upstream-physical-controls-v1.json').read_bytes()).hexdigest();save()
 run('weight-free-map-controls',[py,repo/'tests/test_lighton4b_upstream_physical_audit.py'],env,cwd=repo)
 run('physical-model-free-controls',[py,'-u',worker],env,3000,cwd=repo)
 receipt=json.loads((OUT/'physical-controls.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='passed'or receipt['source_sha']!=SOURCE or receipt['model_payload_bytes']!=0 or receipt['learned_model_inference']:raise RuntimeError('complete model-free controls missing')
 state['status']='completed'
except Exception as error:
 state['error_type']=type(error).__name__;state['error']='Bounded independent upstream worker did not complete; retain per-phase logs and receipt'
finally:stop();save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
