#!/usr/bin/env python3
"""Bounded model-free CPU bootstrap for source-compatible vision controls."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
SOURCE = '0eca8d13ff55169ad47efcb446fa32786bbf9d3e'
WORK=Path('/tmp/lighton4b-vision-profile-controls-v1')
OUT=Path('/kaggle/working')
REPORT=OUT/'vision-controls-bootstrap.json'
state={'schema':'lighton4b-vision-controls-bootstrap/v1','source_sha':SOURCE,'status':'failed','gpu_requested':False,'original_forward_executed':False,'model_quality_qualified':False}
active=None
started=time.monotonic();BUDGET=2700
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
 expected_paths={'tools/vision-profile-probe/controls.py': 'cadf5c775905e9d2749eef4b570bc566513b04f9f4b526d215db9b5d61b1da12', 'tools/vision-profile-probe/test_controls.py': '14060415fc78e90a57b21dc9c87fcdda426df8f909381d2c155d3ec54a527cf7', 'tools/vision-profile-probe/resize_patch.py': 'ff801fb296de7a8c9bc110fc848895d99444f90e42537f06a3b14a3a1a487e2f', 'tools/vision-profile-probe/resize_source.cpp': 'a3008cc908623a8f941cd6752e2a2843d62db622541ee8ba6cf71dc538e042ba', 'tools/vision-profile-probe/native.cpp': 'bdcc5b03dc4cd126ff29706a51098cf2565fa4771a40b641884c89f0d0a59575', 'tools/projector-probe/source_patch.py': '8e53e2bf57a7cbaa2d6fc10fdd9029c58f3577396abe0e4188b8d285868c26ad', 'tools/physical-f16-attention-probe/runtime_guard.py': '30aa63e5a51fdd4d66f8202f0222426fe5e964a6c07d43389501901df40452c4', 'tools/physical-f16-attention-probe/test_runtime_guard.py': '8f870783148e074b5d8ac342f0524f3ee97b0bdc33e146e2f3128f0658dd11a4', 'tools/physical-f16-attention-probe/worker.py': 'fa5c24b1d371c2e5ae802635ca56bff80ba873d2056ac1f46ada859711b88bd7', 'tools/kaggle/lighton4b_upstream_physical_decode.py': '2789cd83124265e0a53ec031cc417f05f8bc4a80ceed8acaec0f52f5264e0a21', 'tests/test_lighton4b_physical_f16_vision.cpp': '243150b612a9de9a9aabfc05b46a33db550e26690097b8b88c574c86aeb47ed3'}
 for relative,digest in expected_paths.items():
  if hashlib.sha256((repo/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError('source path admission '+relative)
 receipts=[p for p in Path('/kaggle/input').rglob('projector-audit.json')if p.is_file()and p.stat().st_size<4<<20 and hashlib.sha256(p.read_bytes()).hexdigest()=='427b38fd03a9d4aec47c9292f15f8771e99e263b2258f415149f0f20dd612f0c']
 if len(receipts)!=1:raise RuntimeError('one complete immutable live projector receipt required')
 receipt=json.loads(receipts[0].read_text())
 if receipt['status']!='measured_projector_only_audit' or not receipt['live_arithmetic_closures_exact']:raise RuntimeError('live source dependency rejected')

 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update({'CRISPEMBED_BUILD_MODE':'portable','HEAVY_OUT':str(OUT),'HEAVY_SCRATCH':str(WORK),'LIGHTON4B_EXPECTED_SOURCE':SOURCE,'PYTHONUNBUFFERED':'1','PIP_NO_CACHE_DIR':'1'})
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','pillow==11.3.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5','charset-normalizer==3.5.2','idna==3.20','urllib3==2.8.0','certifi==2026.10.23'],env);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env)
 # The model-free worker enforces resource guards before builds; no weight acquisition.
 worker=repo/'tools/vision-profile-probe/controls.py';state['worker_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest();save()
 run('control-test_runtime_guard.py',[py,repo/'tools/physical-f16-attention-probe'/'test_runtime_guard.py'],env,60,cwd=repo)
 run('control-test_projector_controls.py',[py,repo/'tools/vision-profile-probe'/'test_controls.py'],env,60,cwd=repo)
 if BUDGET-(time.monotonic()-started)<2440:raise TimeoutError('insufficient complete inner-worker budget')
 run('bounded-worker',[py,'-u',worker],env,2420,cwd=repo)
 receipt=json.loads((OUT/'vision-controls.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='passed_model_free_vision_controls':raise RuntimeError('worker success receipt missing')
 state['status']='completed'
except Exception as error:
 state['error_type']=type(error).__name__;state['error']=str(error)
finally:signal.alarm(0);stop();state['elapsed_seconds']=time.monotonic()-started;save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
