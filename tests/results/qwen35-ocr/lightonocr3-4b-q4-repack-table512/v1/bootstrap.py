#!/usr/bin/env python3
"""Bounded CPU operator-only comparison of actual quantized buffer routes."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE='f079dcec64a799a23d2bdcc25c0da0fee6c49d8c'
WORK=Path('/tmp/lighton4b-q4-repack-table512-v1');OUT=Path('/kaggle/working');REPORT=OUT/'q4-repack-table-bootstrap.json';START=time.monotonic();active=None
state={'schema':'lighton4b-source-repack-operator-bootstrap/v1','source_sha':SOURCE,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'failed','gpu_requested':False,'model_forward_executed':None,'model_quality_qualified':False}
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(2940)
def run(name,args,env=None,timeout=120,cwd=None):
 global active
 remaining=2940-(time.monotonic()-START)
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
 for path in ('tools/kaggle/lighton4b_q4_repack_table512.py', 'tools/physical-f16-attention-probe/test_repack_table512_admission.py', 'docs/lighton4b-q4-repack-table512-v1.json', 'docs/lighton4b-upstream-physical-decode-v1.json', 'tools/kaggle/lighton4b_physical_f16_full_form.py', 'tools/physical-f16-attention-probe/runtime_guard.py', 'tools/physical-f16-attention-probe/test_runtime_guard.py', 'tools/kaggle/lighton4b_upstream_physical_decode.py', 'tools/lighton4b_upstream_physical_pack.py', 'tests/lighton4b_native_document.cpp', 'src/lighton4b_native_context.h'):
  if not(repo/path).is_file():raise RuntimeError('required source path missing: '+path)
 state['source_file_sha256']={path:hashlib.sha256((repo/path).read_bytes()).hexdigest()for path in ('tools/kaggle/lighton4b_q4_repack_table512.py', 'tools/physical-f16-attention-probe/test_repack_table512_admission.py', 'docs/lighton4b-q4-repack-table512-v1.json', 'docs/lighton4b-upstream-physical-decode-v1.json', 'tools/kaggle/lighton4b_physical_f16_full_form.py', 'tools/physical-f16-attention-probe/runtime_guard.py', 'tools/physical-f16-attention-probe/test_runtime_guard.py', 'tools/kaggle/lighton4b_upstream_physical_decode.py', 'tools/lighton4b_upstream_physical_pack.py', 'tests/lighton4b_native_document.cpp', 'src/lighton4b_native_context.h')};save()
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update(CRISPEMBED_BUILD_MODE='portable',HEAVY_OUT=str(OUT),HEAVY_SCRATCH=str(WORK),LIGHTON4B_EXPECTED_SOURCE=SOURCE,PYTHONUNBUFFERED='1',PIP_NO_CACHE_DIR='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','Pillow==11.3.0'],env,90);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env);run('dependency-profile',[py,'-c','import sys,numpy,PIL;assert sys.prefix!=sys.base_prefix;assert numpy.__version__=="2.3.5";assert PIL.__version__=="11.3.0"'],env)
 required=[('runtime-repack-controls.json','c7130c6d7b54e6859439bbf87c8e54880c30bd5d2db33c3571ded179e6711cf5','passed_operator_cache_controls'),('physical-f16-table512.json','36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3','measured_extended_table'),('physical-decode.json','712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2','measured')]
 for name,digest,status in required:
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==digest]
  if len(matches)!=1 or json.loads(matches[0].read_text())['status']!=status:raise RuntimeError('one exact complete dependency required: '+name)
 for name,path in [('runtime-guard','tools/physical-f16-attention-probe/test_runtime_guard.py'),('table-admission','tools/physical-f16-attention-probe/test_repack_table512_admission.py')]:run('control-'+name,[py,repo/path],env,30,cwd=repo)
 # Reserve the whole worker2700s plus cleanup before the2940s bootstrap bound.
 if 2940-(time.monotonic()-START)<2710:raise TimeoutError('setup exceeded reserved worker budget')
 worker=repo/'tools/kaggle/lighton4b_q4_repack_table512.py';run('paired-table-model',[py,'-u',worker],env,2710,cwd=repo)
 receipt=json.loads((OUT/'q4-repack-table512.json').read_text());state['worker_status']=receipt['status'];state['model_forward_executed']=bool(receipt.get('arms'))
 if receipt['status']!='measured_same_feature_table_repack'or receipt['source_sha']!=SOURCE or set(receipt['arms'])!={'off','on'}or not receipt['off_282_rows_38_stages_ids_exact']:raise RuntimeError('complete paired table result missing')
 state['status']='completed'
except Exception as error:state['error_type']=type(error).__name__;state['error']=str(error)
finally:
 stop();signal.alarm(0);state['elapsed_seconds']=time.monotonic()-START;save();shutil.rmtree(WORK,ignore_errors=True);print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
