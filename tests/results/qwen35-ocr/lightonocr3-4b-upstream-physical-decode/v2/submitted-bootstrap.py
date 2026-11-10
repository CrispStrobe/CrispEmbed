#!/usr/bin/env python3
"""Authorized CPU bootstrap, exact independent upstream OCR comparison."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
SOURCE='ac34f72ca89823a177c538ad598dfa00fc69f124'
WORK=Path('/tmp/lighton4b-upstream-physical-decode-v1')
OUT=Path('/kaggle/working')
REPORT=OUT/'physical-decode-bootstrap.json'
state={'schema':'lighton4b-physical-decode-kaggle-bootstrap/v1','source_sha':SOURCE,'status':'failed','gpu_requested':False,'original_forward_executed':False,'model_quality_qualified':False}
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
  done=threading.Event();monitor=None
  if name=='physical-comparison':
   def heartbeat():
    previous=None
    while not done.is_set():
     try:
      receipt=OUT/'physical-decode.json'
      if receipt.exists()and receipt.stat().st_size<2*1024**2:
       r=json.loads(receipt.read_text());view={'phase':r.get('phase'),'status':r.get('status'),'preflight':r.get('preflight'),'cases':{}}
       for case,row in r.get('cases',{}).items():
        view['cases'][case]={'source_observer_neutral':row.get('source_observer_neutral'),'arms':{arm:{k:v for k,v in value.items()if k in('status','actual_input_bytes_exact','first_score_vs_source','elapsed_process_seconds','eog_completed','cap_hit')}for arm,value in row.get('arms',{}).items()}}
        progress=OUT/case/'upstream_features/generated-progress.json'
        if progress.exists()and progress.stat().st_size<4096:view['cases'][case]['progress']=json.loads(progress.read_text())
       current=json.dumps(view,sort_keys=True)
       if current!=previous:print('PHYSICAL_HEARTBEAT '+current,flush=True);previous=current
     except(OSError,ValueError):pass
     done.wait(15)
   monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
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
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update({'CRISPEMBED_BUILD_MODE':'portable','HEAVY_OUT':str(OUT),'HEAVY_SCRATCH':str(WORK),'PYTHONUNBUFFERED':'1','PIP_NO_CACHE_DIR':'1'})
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','pillow==11.3.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5','charset-normalizer==3.5.2','idna==3.20','urllib3==2.8.0','certifi==2026.10.23'],env);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env)
 # The worker enforces cgroup-aware RAM/disk guards before its upstream build/model downloads.
 worker=repo/'tools/kaggle/lighton4b_upstream_physical_decode.py';state['worker_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest();state['protocol_sha256']=hashlib.sha256((repo/'docs/lighton4b-upstream-physical-decode-v1.json').read_bytes()).hexdigest();save()
 run('physical-comparison',[py,'-u',worker],env,21800,cwd=repo)
 receipt=json.loads((OUT/'physical-decode.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='measured'or receipt['source_sha']!=SOURCE or len(receipt['cases'])!=3:raise RuntimeError('complete fixed three-case physical result missing')
 state['status']='completed'
except Exception as error:
 state['error_type']=type(error).__name__;state['error']='Bounded independent upstream worker did not complete; retain per-phase logs and receipt'
finally:stop();save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
