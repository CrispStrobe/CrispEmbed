#!/usr/bin/env python3
"""Bounded fixed-case CPU bootstrap for corrected Q6 provisional panel replay."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
SOURCE = 'a6164205620c71427a39c97ab386591406c4ca73'
WORK=Path('/tmp/lighton4b-planner-heldout-historical-receipt-v1')
OUT=Path('/kaggle/working')
REPORT=OUT/'planner-heldout-case-bootstrap.json'
state={'schema':'lighton4b-heldout-controls-bootstrap/v1','source_sha':SOURCE,'status':'failed','gpu_requested':False,'reference_forward_executed':False,'model_quality_qualified':False}
active=None
started=time.monotonic();BUDGET=6300
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
  try:
   if name=='bounded-worker':
    def phases():
     last=None
     while not done.wait(5):
      try:
       phase=json.loads((OUT/'planner-heldout.json').read_text()).get('phase','')
       if phase!=last and phase and len(phase)<=128 and all(ch.isalnum()or ch in '._-'for ch in phase):print('worker-phase',phase,flush=True);last=phase
      except(OSError,ValueError):pass
    monitor=threading.Thread(target=phases,daemon=True);monitor.start()
   code=active.wait(timeout=timeout)
  finally:
   stop();done.set()
   if monitor and monitor.is_alive():monitor.join(timeout=2)
 if code:raise RuntimeError(name+' nonzero')
try:
 if not OUT.is_dir()or not Path('/kaggle/input').is_dir()or list(Path('/dev').glob('nvidia[0-9]*')):raise RuntimeError('authorized Kaggle CPU worker required')
 WORK.mkdir();repo=WORK/'source'
 run('source-init',['git','init',repo]);run('source-remote',['git','-C',repo,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']);run('source-fetch',['git','-C',repo,'fetch','--depth','1','origin',SOURCE]);run('source-checkout',['git','-C',repo,'checkout','--detach','FETCH_HEAD'])
 if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=SOURCE:raise RuntimeError('exact source differs')
 expected_paths={'tools/planner-heldout/worker.py': '598c2b435874d34d515d0f75997524d296cdf857dfe79f6a94d86cd77bfbb48f', 'tools/planner-heldout/contracts.py': 'a036830bb77705468c33491522010061c90e987e7d0da9418038dbd7e465d94f', 'tools/planner-heldout/test_controls.py': '6c4ead81886a062dec3ce5e0a818d4ca206726fbd2996e9231a7dfc9a87904ee', 'docs/lighton4b-planner-heldout-v1.json': '869eea6c711497718b6d467e67f0e80eb3567d18c8ec3b66d5395fe83bca344e', 'docs/lighton4b-heldout-v1.json': '1a5cf84d5d83c22a041e774aaa00e682a28d302a2542ef02606e5dfa0ef3cd4b', 'docs/lighton4b-heldout-executor-v1.json': 'bdd80e7f7c3c7a578478bd9c16d94dc07c32e86220c6a337629e0c95498aaf5f', 'tools/heldout/execution.py': '82f5adc352c79e0779478261ce2a56e90e0df05bdca2d232386527b4a440ee06', 'tools/heldout/lighton4b_prepare.py': '6eef63020be662caa87b6a904df4f0ae7e1fcdb21caa605d8ecc056bbcfc0891', 'tools/heldout/transfer.py': 'c58ab8d4bcc6eeba80f6d52e533bdef4972f9131963a65a1e23ffff9244418bd', 'tools/heldout/test_transfer.py': '725326b0521026aedc8052cbd2f771872826941178df4d6b02de6f253213dca5', 'tools/physical-f16-attention-probe/runtime_guard.py': '30aa63e5a51fdd4d66f8202f0222426fe5e964a6c07d43389501901df40452c4', 'tools/physical-f16-attention-probe/worker.py': 'fa5c24b1d371c2e5ae802635ca56bff80ba873d2056ac1f46ada859711b88bd7', 'tools/kaggle/lighton4b_upstream_physical_decode.py': '2789cd83124265e0a53ec031cc417f05f8bc4a80ceed8acaec0f52f5264e0a21', 'tools/lighton4b_upstream_physical_pack.py': '0f86cd619b1b25c419bd643a549786cc35d0c546b0bcab382e2efb77e3b24910', 'tools/gguf_file_type_canonicalize.py': '838a90879879ce6cbb6262cedac159abda5579e32e66ac13c11a751d2b75d9d7', 'tools/lighton4b_q6_convert.py': 'eb5f8bae51a808cfcd589d534a36da13106369e0c3b8da36bba8a68b04a8d4d1', 'tools/lighton4b_q6_recipe.py': '10cd2eb391ade1a4ccdd8ee5608ec6dabfb49c32a1fb188d82f9aef061daede1', 'tools/ci-heavy/lighton4b_upstream_physical_audit.py': '99e1457b32a47bf5b1e81a3d77e075881eeb7e227331a3ef64f81aaf00ae4712', 'tools/ocr_quality.py': 'de93f01d26759e052e4c99c224b3184a39bdf3bbd11f768318c672ccad7fa0c5', 'tools/ci-heavy/qwen35_quant_calibration.py': 'bf03b5a66fd6d42700eaf5e4dcc9a5705ea208df3e903c2187f576b7b45ebed6', 'tests/lighton4b_native_document.cpp': 'e8c8363d183465600d22ccc2419bd18103a0ca3596fdf9abf5077b2f802dc634', 'tests/lighton4b_native_case_preflight.cpp': 'af06ef417533a1eff2f6658782c8c267126ae7495b1d03ad6c3b839af706b957', 'CMakeLists.txt': 'e226d2764eb16e1904cd59bbda72064f0ff4e8e0e0629dc6913fdfb3db563b89'}
 for relative,digest in expected_paths.items():
  if hashlib.sha256((repo/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError("source path admission "+relative)

 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update({'CRISPEMBED_BUILD_MODE':'portable','HEAVY_OUT':str(OUT),'HEAVY_SCRATCH':str(WORK),'LIGHTON4B_EXPECTED_SOURCE':SOURCE,'PYTHONUNBUFFERED':'1','PIP_NO_CACHE_DIR':'1'})
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','pillow==11.3.0'],env);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env)
 # Worker authenticates historical references and observed controls before new case inference.
 worker=repo/'tools/planner-heldout/worker.py';state['worker_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest();save()
 run('worker-syntax',[py,'-c','import ast,sys;ast.parse(open(sys.argv[1]).read())',worker],env,30,cwd=repo)
 for test in ['tools/planner-heldout/test_controls.py','tools/heldout/test_transfer.py']:
  run('control-'+Path(test).name,[py,repo/test],env,60,cwd=repo)
 required=[('planner-heldout.json', '14eb900165f65c16a7696e90eba7aeb2ce90db76f0d5c16e86059efa187ecdc9', 'passed_model_free_planner_heldout_controls'), ('heldout-result.json', '8bf1ebbdf12b9dd2b20603c11a9b7d4c990e97e7f37a2f5aca3fb3324b3c8dfc', 'measured_provisional_heldout'), ('q6-heldout.json', '243bb2f8c9fbf8ed8f19d6b0df48bb8ecc42f4fa66744c02ee59609a419e96eb', 'measured_provisional_q6_heldout'), ('q6-tiled-full-ab.json', 'f5d0077052b907e2d674caef9e7c4ae09b08456ff48a79f327cbdfd030b2dcc9', 'measured_q6_tiled_full_ab'), ('source-prefill-observed-ab.json', '393bfbd65f0f39c9d15e68110eb8174fb912797200cc68b1c8339e7a0797edcb', 'measured_source_prefill_observed_ab'), ('physical-decode.json', '712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2', 'measured')]
 for name,digest,status in required:
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==digest]
  if len(matches)!=1 or json.loads(matches[0].read_text())['status']!=status:raise RuntimeError('one complete immutable dependency required: '+name)
 if BUDGET-(time.monotonic()-started)<6040:raise TimeoutError('insufficient complete inner-worker budget')
 run('bounded-worker',[py,'-u',worker,'--case','historical-receipt','--controls-receipt-sha','14eb900165f65c16a7696e90eba7aeb2ce90db76f0d5c16e86059efa187ecdc9'],env,6020,cwd=repo)
 receipt=json.loads((OUT/'planner-heldout.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='measured_provisional_planner_heldout'or receipt['case']!='historical-receipt'or not receipt['old_baseline_all_rows_stages_exact']or receipt['qualified']is not False:raise RuntimeError('complete fixed-case diagnostic receipt missing')
 state['status']='completed'
except Exception as error:
 state['error_type']=type(error).__name__;state['error']=str(error)
finally:signal.alarm(0);stop();state['elapsed_seconds']=time.monotonic()-started;save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
