#!/usr/bin/env python3
"""Bounded model-free CPU bootstrap for provisional held-out executor controls."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
SOURCE = '874728b71d113bb1a02f7a1fa97717ec2b6281b5'
WORK=Path('/tmp/lighton4b-heldout-controls-v1')
OUT=Path('/kaggle/working')
REPORT=OUT/'heldout-controls-bootstrap.json'
state={'schema':'lighton4b-heldout-controls-bootstrap/v1','source_sha':SOURCE,'status':'failed','gpu_requested':False,'original_forward_executed':False,'model_quality_qualified':False}
active=None
started=time.monotonic();BUDGET=2160
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(BUDGET)
def run(name,args,env=None,timeout=1200,cwd=None):
 global active
 remaining=BUDGET-(time.monotonic()-started)-20
 if remaining<=0:raise TimeoutError('whole-bootstrap budget exhausted')
 timeout=min(timeout,remaining)
 state['phase']=name;save();print('phase',name,flush=True)
 with(OUT/(name+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,cwd=cwd,start_new_session=True)
  done=threading.Event();monitor=None
  try:code=active.wait(timeout=timeout)
  finally:
   stop();done.set()
   if monitor:monitor.join(timeout=2)
 if code:raise RuntimeError(name+' nonzero')
try:
 if not OUT.is_dir()or not Path('/kaggle/input').is_dir():raise RuntimeError('authorized Kaggle CPU worker required')
 WORK.mkdir();repo=WORK/'source'
 run('source-init',['git','init',repo]);run('source-remote',['git','-C',repo,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']);run('source-fetch',['git','-C',repo,'fetch','--depth','1','origin',SOURCE]);run('source-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])
 if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=SOURCE:raise RuntimeError('exact source differs')
 expected_paths={'tools/heldout/lighton4b_prepare.py': '6eef63020be662caa87b6a904df4f0ae7e1fcdb21caa605d8ecc056bbcfc0891', 'tools/heldout/test_source_session.py': 'd9b3601ffa457b2398ae4d6960bd865161a89721868abd1d54c23e2ec6d7bfb0', 'tools/heldout/source_preflight.cpp': '867b4dd7d86dc9a8b9e0ffcc7617098cf62a4abf5dd281412f791b2939546f36', 'tools/heldout/execution.py': 'dfe197929b8b02c00f8ae55a1926f34daadde18c04012f88aadeb0ff66b2f5fd', 'tools/heldout/test_composed_executor.py': 'eff9949bdfe20177b4f8b63710c5cadaab7a511e4f342c2ab74afb616ca861ad', 'tools/heldout/worker.py': '7e1cf3d02329d8ee95c5e8a2d84d0d8fd9cfa0bb4f9f554cdc282a4722758fff', 'tools/heldout/source_session.py': '59f23bfba05ce5cce4f47c4b0ff77646dc5af6b05c39b5f76e0fde8a643bf454', 'tools/heldout/test_lighton4b_prepare.py': 'adf400e13dd720a22cf1fbb1756ccee42948e43c3e167be0286fa35ec0816d9c', 'tools/heldout/test_execution.py': '8ee811d2a817739801fa489adf1b10382ad4143072dc0de4d56373a454834c2e', 'docs/lighton4b-heldout-executor-v1.json': '993980e6f2b8d456a25c178ebe1a5e814011b5b0bf11d712b6eb9b62de319b24', 'docs/lighton4b-heldout-v1.json': '1a5cf84d5d83c22a041e774aaa00e682a28d302a2542ef02606e5dfa0ef3cd4b', 'tools/physical-f16-attention-probe/runtime_guard.py': '30aa63e5a51fdd4d66f8202f0222426fe5e964a6c07d43389501901df40452c4', 'tools/physical-f16-attention-probe/worker.py': 'fa5c24b1d371c2e5ae802635ca56bff80ba873d2056ac1f46ada859711b88bd7', 'tools/kaggle/lighton4b_upstream_physical_decode.py': '2789cd83124265e0a53ec031cc417f05f8bc4a80ceed8acaec0f52f5264e0a21', 'tools/lighton4b_upstream_observer_patch.py': '80c229113c4836c1447661d843dba9cc1fff573840ad128e6aee79b6a0e285a5', 'tools/lighton4b_upstream_physical_pack.py': '0f86cd619b1b25c419bd643a549786cc35d0c546b0bcab382e2efb77e3b24910', 'tests/lighton4b_native_document.cpp': '912b5d0d34f13589e3848b419cfb51c3854acdbbe5f0380526d313c9c77cf879', 'tests/lighton4b_native_case_preflight.cpp': '09b45d7195bd5832ac6e209b62eb4028ae7b4a39f1b558f53607480a2eaff5cf'}
 for relative,digest in expected_paths.items():
  if hashlib.sha256((repo/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError("source path admission "+relative)

 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update({'CRISPEMBED_BUILD_MODE':'portable','HEAVY_OUT':str(OUT),'HEAVY_SCRATCH':str(WORK),'LIGHTON4B_EXPECTED_SOURCE':SOURCE,'PYTHONUNBUFFERED':'1','PIP_NO_CACHE_DIR':'1'})
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','pillow==11.3.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5','charset-normalizer==3.5.2','idna==3.20','urllib3==2.8.0','certifi==2026.10.23'],env);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env)
 # The model-free worker enforces resource guards before builds; no weight acquisition.
 worker=repo/'tools/heldout/worker.py';state['worker_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest();save()
 for test in ['test_execution.py','test_source_session.py','test_lighton4b_prepare.py','test_composed_executor.py']:
  run('control-'+test,[py,repo/'tools/heldout'/test],env,60,cwd=repo)
 if BUDGET-(time.monotonic()-started)<1840:raise TimeoutError('insufficient complete inner-worker budget')
 run('bounded-worker',[py,'-u',worker,'--controls-only'],env,1820,cwd=repo)
 receipt=json.loads((OUT/'heldout-result.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='passed_model_free_heldout_controls' or receipt['model_acquisition'] is not False or receipt['heldout_forward_executed'] is not False:raise RuntimeError('model-free success receipt missing')
 state['status']='completed'
except Exception as error:
 state['error_type']=type(error).__name__;state['error']=str(error)
finally:signal.alarm(0);stop();state['elapsed_seconds']=time.monotonic()-started;save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
