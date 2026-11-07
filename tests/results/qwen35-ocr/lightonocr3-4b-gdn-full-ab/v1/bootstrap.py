#!/usr/bin/env python3
"""Bounded observed table/form GDN full decoder A/B; no qualification."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE='f6a1e60d3cea5a1f6841079697734906f70d8b40'
WORK=Path('/tmp/lighton4b-gdn-full-ab-v1');OUT=Path('/kaggle/working');REPORT=OUT/'gdn-full-ab-bootstrap.json';START=time.monotonic();active=None
state={'schema':'lighton4b-gdn-full-ab-bootstrap/v1','source_sha':SOURCE,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'failed','gpu_requested':False,'model_forward_executed':None,'model_quality_qualified':False}
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(6300)
def run(name,args,env=None,timeout=120,cwd=None):
 global active
 remaining=6300-(time.monotonic()-START)
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
 expected_paths={'tools/gdn-full-ab/worker.py': '75685af1092a92e4b7e454616a5fbefdf397158e8c02f64e6a372d028a24b30d', 'tools/gdn-full-ab/test_controls.py': 'd9e18dac414ffa2269fa0516af1a0ad104a90dedcca2a26002af9c355df34298', 'docs/lighton4b-gdn-full-ab-v1.json': '37bb2ff725ffcfd8c5a1a3acc58e09136856a710d80436456e7ff67b7c9cae63', 'tools/kaggle/lighton4b_physical_f16_full_form.py': 'fd417f272f522e63d103f091589952f43a14c36a8c7068edee33a19cc2202c86', 'tools/kaggle/lighton4b_upstream_physical_decode.py': '2789cd83124265e0a53ec031cc417f05f8bc4a80ceed8acaec0f52f5264e0a21', 'tools/lighton4b_upstream_physical_pack.py': '0f86cd619b1b25c419bd643a549786cc35d0c546b0bcab382e2efb77e3b24910', 'tools/physical-f16-attention-probe/runtime_guard.py': '30aa63e5a51fdd4d66f8202f0222426fe5e964a6c07d43389501901df40452c4', 'tools/physical-f16-attention-probe/test_runtime_guard.py': '8f870783148e074b5d8ac342f0524f3ee97b0bdc33e146e2f3128f0658dd11a4', 'docs/lighton4b-upstream-physical-decode-v1.json': 'c41eb3d8cb7edd7d33f8561461c61f78182df960bf2630c41ae952c1e7dee14d', 'docs/lighton4b-composed-form-vision-v1.json': 'bfc3fbb1d856c36525c7a11bddb68184d92114c5f1a051786fc7bf0988753d90', 'docs/lighton4b-physical-f16-full-form-v1.json': 'da10739fba67295a43ad796c9a3c7b37dc89f9bfbcfcf821d9f2a25d7a753370', 'tests/lighton4b_native_document.cpp': 'e162586bf5cc48684163967612f1cb7aba335421014d7ef8be2e55831bf85e2b', 'CMakeLists.txt': 'e9fe2621980f84d43ea48199c447d030f5dddcf201ddaabcd91f845c1e4a8104'}
 for path,digest in expected_paths.items():
  if not(repo/path).is_file()or hashlib.sha256((repo/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('source path/hash '+path)
 state['source_file_sha256']=expected_paths;save()
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update(CRISPEMBED_BUILD_MODE='portable',HEAVY_OUT=str(OUT),HEAVY_SCRATCH=str(WORK),LIGHTON4B_EXPECTED_SOURCE=SOURCE,PYTHONUNBUFFERED='1',PIP_NO_CACHE_DIR='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','Pillow==11.3.0'],env,90);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env);run('dependency-profile',[py,'-c','import sys,numpy;assert sys.prefix!=sys.base_prefix;assert numpy.__version__=="2.3.5"'],env)
 required=[('gdn-hybrid-probe.json', '6b1f4f9c3150d06df0a195c1f21c42136874e69934b9586205712114354e798d', 'measured_actual_source_hybrid_controls'), ('physical-decode.json', '712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2', 'measured'), ('physical-f16-table512.json', '36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3', 'measured_extended_table'), ('composed-form-vision.json', 'aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b', 'measured_composed_form_vision'), ('composed-vision-calibration.json', 'f69d23ffe6d65d9abf6c7c5c660723613ce84cc901383c5151e17d8f8dc5fe64', 'measured_composed_vision_calibration')]
 for name,digest,status in required:
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==digest]
  if len(matches)!=1 or json.loads(matches[0].read_text())['status']!=status:raise RuntimeError('one exact complete dependency required: '+name)
 for name,path in [('runtime-guard','tools/physical-f16-attention-probe/test_runtime_guard.py'),('gdn-admission','tools/gdn-full-ab/test_controls.py')]:run('control-'+name,[py,repo/path],env,30,cwd=repo)
 # Reserve the whole worker6000s plus cleanup before the6300s bootstrap bound.
 if 6300-(time.monotonic()-START)<6060:raise TimeoutError('setup exceeded reserved worker budget')
 worker=repo/'tools/gdn-full-ab/worker.py';run('gdn-full-ab',[py,'-u',worker],env,6040,cwd=repo)
 receipt=json.loads((OUT/'gdn-full-ab.json').read_text());state['worker_status']=receipt['status'];state['model_forward_executed']=bool(receipt.get('cases'))
 if receipt['status']!='measured_gdn_full_ab'or receipt['source_sha']!=SOURCE or not all(v['off_all_rows_stages_exact']for v in receipt['cases'].values()):raise RuntimeError('complete observed GDN decoder A/B missing')
 state['status']='completed'
except Exception as error:state['error_type']=type(error).__name__;state['error']=str(error)
finally:
 stop();signal.alarm(0);state['elapsed_seconds']=time.monotonic()-START;save();shutil.rmtree(WORK,ignore_errors=True);print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
