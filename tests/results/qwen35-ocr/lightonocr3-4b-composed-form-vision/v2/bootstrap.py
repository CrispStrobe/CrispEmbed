#!/usr/bin/env python3
"""Bounded composed observed-form vision-only gate comparison."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE='b98567cfd46aaf594f33076aaab4ff8e4610edbb'
WORK=Path('/tmp/lighton4b-composed-form-vision-v1');OUT=Path('/kaggle/working');REPORT=OUT/'composed-form-vision-bootstrap.json';START=time.monotonic();active=None
state={'schema':'lighton4b-composed-form-vision-bootstrap/v1','source_sha':SOURCE,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'failed','gpu_requested':False,'model_forward_executed':None,'model_quality_qualified':False}
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(3900)
def run(name,args,env=None,timeout=120,cwd=None):
 global active
 remaining=3900-(time.monotonic()-START)
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
 expected_paths={'tools/kaggle/lighton4b_composed_form_vision.py': '91fdc5c329a49da611a9133c3da5611ec9a0232d298c64d3e68dc5d03eef7cad', 'tools/vision-profile-probe/test_composed_form_vision.py': '1d506a0780c7b3888cd2241ff268eecb3ee4da6b3ef81cd050b97859168856c0', 'docs/lighton4b-composed-form-vision-v1.json': 'bfc3fbb1d856c36525c7a11bddb68184d92114c5f1a051786fc7bf0988753d90', 'tools/vision-profile-probe/component.py': '41f98cd9425a2aaedded36a326539117b403cd0a7c3791445aa3505b70f014bc', 'tools/vision-profile-probe/controls.py': 'cadf5c775905e9d2749eef4b570bc566513b04f9f4b526d215db9b5d61b1da12', 'tools/vision-profile-probe/resize_patch.py': 'ff801fb296de7a8c9bc110fc848895d99444f90e42537f06a3b14a3a1a487e2f', 'tools/projector-probe/source_patch.py': '8e53e2bf57a7cbaa2d6fc10fdd9029c58f3577396abe0e4188b8d285868c26ad', 'tools/projector-probe/canonical.py': 'dcefedfc49a112f0d961199430b47159f9a8f9be517577ca2c97c3c55f8b3fb8', 'tools/projector-probe/worker.py': '56bc0b56a9324ffcc5494196502ec48ece027a430af5d85d9c31f5ae28a5ca2e', 'tools/physical-f16-attention-probe/worker.py': 'fa5c24b1d371c2e5ae802635ca56bff80ba873d2056ac1f46ada859711b88bd7', 'tools/physical-f16-attention-probe/runtime_guard.py': '30aa63e5a51fdd4d66f8202f0222426fe5e964a6c07d43389501901df40452c4', 'tools/physical-f16-attention-probe/test_runtime_guard.py': '8f870783148e074b5d8ac342f0524f3ee97b0bdc33e146e2f3128f0658dd11a4', 'tools/kaggle/lighton4b_physical_f16_full_form.py': 'fd417f272f522e63d103f091589952f43a14c36a8c7068edee33a19cc2202c86', 'tools/kaggle/lighton4b_upstream_physical_decode.py': '2789cd83124265e0a53ec031cc417f05f8bc4a80ceed8acaec0f52f5264e0a21', 'tools/lighton4b_upstream_physical_pack.py': '0f86cd619b1b25c419bd643a549786cc35d0c546b0bcab382e2efb77e3b24910', 'docs/lighton4b-upstream-physical-decode-v1.json': 'c41eb3d8cb7edd7d33f8561461c61f78182df960bf2630c41ae952c1e7dee14d', 'tests/lighton4b_native_document.cpp': 'e162586bf5cc48684163967612f1cb7aba335421014d7ef8be2e55831bf85e2b'}
 for path,digest in expected_paths.items():
  if not(repo/path).is_file()or hashlib.sha256((repo/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('source path/hash '+path)
 state['source_file_sha256']=expected_paths;save()
 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update(CRISPEMBED_BUILD_MODE='portable',HEAVY_OUT=str(OUT),HEAVY_SCRATCH=str(WORK),LIGHTON4B_EXPECTED_SOURCE=SOURCE,PYTHONUNBUFFERED='1',PIP_NO_CACHE_DIR='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','Pillow==11.3.0'],env,90);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env);run('dependency-profile',[py,'-c','import sys,numpy,PIL;assert sys.prefix!=sys.base_prefix;assert numpy.__version__=="2.3.5";assert PIL.__version__=="11.3.0"'],env)
 required=[('vision-component.json', '039b024f41b499cfd8344cb4e89835bbfe559a1f298225866c691000d4e6c8c0', 'measured_same_input_projector_component'), ('projector-audit.json', '427b38fd03a9d4aec47c9292f15f8771e99e263b2258f415149f0f20dd612f0c', 'measured_projector_only_audit'), ('composed-vision.json', '1458d07e6098b4b1b8fa078c18fe0b636f832a0d1219f789b035e1a033ef4ba0', 'measured_observed_projector_pair'), ('physical-decode.json', '712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2', 'measured')]
 for name,digest,status in required:
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==digest]
  if len(matches)!=1 or json.loads(matches[0].read_text())['status']!=status:raise RuntimeError('one exact complete dependency required: '+name)
 for name,path in [('runtime-guard','tools/physical-f16-attention-probe/test_runtime_guard.py'),('form-admission','tools/vision-profile-probe/test_composed_form_vision.py')]:run('control-'+name,[py,repo/path],env,30,cwd=repo)
 # Reserve the whole worker3600s plus cleanup before the3900s bootstrap bound.
 if 3900-(time.monotonic()-START)<3660:raise TimeoutError('setup exceeded reserved worker budget')
 worker=repo/'tools/kaggle/lighton4b_composed_form_vision.py';run('paired-composed-form',[py,'-u',worker],env,3640,cwd=repo)
 receipt=json.loads((OUT/'composed-form-vision.json').read_text());state['worker_status']=receipt['status'];state['model_forward_executed']=bool(receipt.get('arms'))
 if receipt['status']!='measured_composed_form_vision'or receipt['source_sha']!=SOURCE or set(receipt['arms'])!={'old','fixed'}or not receipt['arms']['old']['expected_text_and_rows_exact']:raise RuntimeError('complete composed form result missing')
 state['status']='completed'
except Exception as error:state['error_type']=type(error).__name__;state['error']=str(error)
finally:
 stop();signal.alarm(0);state['elapsed_seconds']=time.monotonic()-START;save();shutil.rmtree(WORK,ignore_errors=True);print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
