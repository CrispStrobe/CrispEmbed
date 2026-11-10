#!/usr/bin/env python3
"""Bounded bounded CPU bootstrap for one frozen provisional held-out diagnostic."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
SOURCE = 'd378c6cc055e3a31b0d4cc9dad54a5b41f10d792'
WORK=Path('/tmp/lighton4b-heldout-inventory-table-v1')
OUT=Path('/kaggle/working')
REPORT=OUT/'heldout-case-bootstrap.json'
state={'schema':'lighton4b-heldout-case-bootstrap/v1','source_sha':SOURCE,'status':'failed','gpu_requested':False,'original_forward_executed':False,'model_quality_qualified':False}
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
  if name=='bounded-worker':
   def worker_phases():
    last=None
    while not done.wait(5):
     try:
      phase=json.loads((OUT/'heldout-result.json').read_text()).get('phase','')
      if phase!=last and phase and len(phase)<=128 and all(c.isalnum()or c in '._-'for c in phase):
       print('worker-phase',phase,flush=True);last=phase
     except (OSError,ValueError):pass
   monitor=threading.Thread(target=worker_phases,daemon=True);monitor.start()
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
 expected_paths={'tools/heldout/lighton4b_prepare.py': '6eef63020be662caa87b6a904df4f0ae7e1fcdb21caa605d8ecc056bbcfc0891', 'tools/heldout/test_source_session.py': 'a0612505a0515224e40d93af8d0524ef4a8de845821b6f6169316caf9868bd5c', 'tools/heldout/source_preflight.cpp': 'c416a1ba7628191715dc5bee2bd07d9fc170e0c1c58c56e5bf0e03abe02c0860', 'tools/heldout/execution.py': '82f5adc352c79e0779478261ce2a56e90e0df05bdca2d232386527b4a440ee06', 'tools/heldout/test_composed_executor.py': 'eff9949bdfe20177b4f8b63710c5cadaab7a511e4f342c2ab74afb616ca861ad', 'tools/heldout/worker.py': '7c5eacc3af1b45dca066fc00b5707005bb227b3f72526de6cc5e8e4fa79021be', 'tools/heldout/source_session.py': '83e5d8128505fe1eb63612e676824b2d39dd163cd25ce3673d49d5906fcf4baf', 'tools/heldout/test_lighton4b_prepare.py': 'adf400e13dd720a22cf1fbb1756ccee42948e43c3e167be0286fa35ec0816d9c', 'tools/heldout/test_execution.py': 'eef39934a7b8124720787df68a832de12fa6f6c28e7d9a85e7963b0c065a0791', 'docs/lighton4b-heldout-executor-v1.json': 'bdd80e7f7c3c7a578478bd9c16d94dc07c32e86220c6a337629e0c95498aaf5f', 'docs/lighton4b-heldout-v1.json': '1a5cf84d5d83c22a041e774aaa00e682a28d302a2542ef02606e5dfa0ef3cd4b', 'tools/physical-f16-attention-probe/runtime_guard.py': '30aa63e5a51fdd4d66f8202f0222426fe5e964a6c07d43389501901df40452c4', 'tools/physical-f16-attention-probe/worker.py': 'fa5c24b1d371c2e5ae802635ca56bff80ba873d2056ac1f46ada859711b88bd7', 'tools/kaggle/lighton4b_upstream_physical_decode.py': '2789cd83124265e0a53ec031cc417f05f8bc4a80ceed8acaec0f52f5264e0a21', 'tools/lighton4b_upstream_observer_patch.py': '80c229113c4836c1447661d843dba9cc1fff573840ad128e6aee79b6a0e285a5', 'tools/lighton4b_upstream_physical_pack.py': '0f86cd619b1b25c419bd643a549786cc35d0c546b0bcab382e2efb77e3b24910', 'tests/lighton4b_native_document.cpp': '912b5d0d34f13589e3848b419cfb51c3854acdbbe5f0380526d313c9c77cf879', 'tests/lighton4b_native_case_preflight.cpp': '09b45d7195bd5832ac6e209b62eb4028ae7b4a39f1b558f53607480a2eaff5cf', 'tools/heldout/transfer.py': 'c58ab8d4bcc6eeba80f6d52e533bdef4972f9131963a65a1e23ffff9244418bd', 'tools/heldout/test_transfer.py': '725326b0521026aedc8052cbd2f771872826941178df4d6b02de6f253213dca5'}
 for relative,digest in expected_paths.items():
  if hashlib.sha256((repo/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError("source path admission "+relative)

 frozen_receipts={'heldout-result.json':'7b090c04f11cd48e27e49dd290fef1b5405bfbd058d658b6ee13e44e5315bc07','composed-form-vision.json':'aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b','physical-decode.json':'712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'}
 for name,digest in frozen_receipts.items():
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and hashlib.sha256(p.read_bytes()).hexdigest()==digest]
  if len(matches)!=1:raise RuntimeError('one complete immutable dependency '+name)

 env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env.update({'CRISPEMBED_BUILD_MODE':'portable','HEAVY_OUT':str(OUT),'HEAVY_SCRATCH':str(WORK),'LIGHTON4B_EXPECTED_SOURCE':SOURCE,'PYTHONUNBUFFERED':'1','PIP_NO_CACHE_DIR':'1'})
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv]);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5','pillow==11.3.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5','charset-normalizer==3.5.2','idna==3.20','urllib3==2.8.0','certifi==2026.10.23'],env);run('dependency-check',[sys.executable,'-m','pip','--python',py,'check'],env)
 # The frozen case worker admits controls/input seals before held-out forwards.
 worker=repo/'tools/heldout/worker.py';state['worker_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest();save()
 run('prior-case-audit',[py,'-c','import hashlib,json,math,zipfile\nfrom pathlib import Path\nimport numpy as np\nif not __debug__:raise RuntimeError("audit assertions must be enabled")\nassert np.__version__==\'2.3.5\'\nexpected=\'9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722\'\ndef sha(p):\n with p.open(\'rb\')as f:return hashlib.file_digest(f,\'sha256\').hexdigest()\nmatches=[p for p in Path(\'/kaggle/input\').rglob(\'heldout-result.json\')if p.is_file()and p.stat().st_size<4<<20 and sha(p)==expected]\nassert len(matches)==1\np=matches[0];r=json.loads(p.read_text());base=p.parent.resolve()\nassert r[\'source_sha\']==\'d378c6cc055e3a31b0d4cc9dad54a5b41f10d792\'and r[\'status\']==\'measured_provisional_heldout\'and r[\'case\']==\'two-column-page\'and r[\'qualified\']is False and r[\'source_observer_full_neutral\']is True\nassert len(r[\'artifacts\'])<=1000 and sum(x[\'bytes\']for x in r[\'artifacts\'].values())<=8<<30\nverified={}\nfor name,row in r[\'artifacts\'].items():\n f=(base/name).resolve();assert base in f.parents and f.is_file()and f.stat().st_size==row[\'bytes\']and sha(f)==row[\'sha256\'];verified[name]=row\nroles=[(\'observed-control\',r[\'observed_control_rows\'],\'greedy\'),(\'upstream-on\',r[\'source_raw_rows\'],\'source\')]+[(arm,r[\'arms\'][arm][\'raw_rows\'],\'greedy\')for arm in[\'native-composed\',\'native-source-features\']]\nrows_verified=0\nfor folder,meta,stem in roles:\n archive=(base/folder/meta[\'archive\'][\'file\']).resolve();assert base in archive.parents and sha(archive)==meta[\'archive\'][\'sha256\']\n with zipfile.ZipFile(archive)as z:\n  names=[f\'{stem}.{row["position"]}.logits.f32\'if stem==\'greedy\'else f\'source.logits.{row["position"]}.f32\'for row in meta[\'rows\']]\n  assert z.namelist()==names and len(names)<=1536\n  for name,row in zip(names,meta[\'rows\']):\n   info=z.getinfo(name);assert info.file_size==row[\'bytes\']==248320*4 and info.compress_type==zipfile.ZIP_STORED\n   raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==row[\'sha256\'];values=np.frombuffer(raw,dtype=\'<f4\');assert np.isfinite(values).all()and int(values.argmax())==row[\'argmax\'];rows_verified+=1\nout=Path(\'/kaggle/working/prior-case-artifact-audit.json\');out.write_text(json.dumps(dict(schema=\'lighton4b-prior-case-artifact-audit/v1\',receipt_sha256=expected,owned_files_verified=len(verified),owned_bytes_verified=sum(v[\'bytes\']for v in verified.values()),raw_zip_archives_verified=4,raw_rows_finite_argmax_and_hash_verified=rows_verified,artifacts=verified,model_forward_executed=False),indent=2)+\'\\n\')\nprint(\'PRIOR_CASE_ARTIFACT_AUDIT PASS files=\'+str(len(verified))+\' rows=\'+str(rows_verified),flush=True)\n'],env,180,cwd=repo)
 state['prior_case_artifact_audit_sha256']=hashlib.sha256((OUT/'prior-case-artifact-audit.json').read_bytes()).hexdigest();save()
 run('worker-syntax',[py,'-m','py_compile',worker],env,30,cwd=repo)
 for test in ['test_execution.py','test_source_session.py','test_lighton4b_prepare.py','test_composed_executor.py','test_transfer.py']:
  run('control-'+test,[py,repo/'tools/heldout'/test],env,60,cwd=repo)
 if BUDGET-(time.monotonic()-started)<6040:raise TimeoutError('insufficient complete inner-worker budget')
 run('bounded-worker',[py,'-u',worker,'--case','inventory-table'],env,6020,cwd=repo)
 receipt=json.loads((OUT/'heldout-result.json').read_text());state['worker_status']=receipt['status']
 if receipt['status']!='measured_provisional_heldout' or receipt['case']!='inventory-table':raise RuntimeError('frozen case success receipt missing')
 state['status']='completed'
except Exception as error:
 state['error_type']=type(error).__name__;state['error']=str(error)
finally:signal.alarm(0);stop();state['elapsed_seconds']=time.monotonic()-started;save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
