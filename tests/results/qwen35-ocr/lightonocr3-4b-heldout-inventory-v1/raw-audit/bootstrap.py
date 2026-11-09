#!/usr/bin/env python3
"""Read-only final inventory raw-artifact audit; no models, builds or inference."""
import hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
OUT=Path('/kaggle/working');WORK=Path('/tmp/lighton4b-heldout-inventory-audit-v1');REPORT=OUT/'inventory-audit-bootstrap.json';BUDGET=600;started=time.monotonic();active=None
state={'schema':'lighton4b-inventory-audit-bootstrap/v1','status':'failed','model_downloads_allowed':False,'model_forward_executed':False,'quality_qualified':False}
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def memory():
 vals={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in ('MemAvailable','SwapTotal','SwapFree')}
 observed=0;bounds=[]
 for line in Path('/proc/self/cgroup').read_text().splitlines():
  hierarchy,controllers,relative=line.split(':',2)
  if hierarchy=='0'and controllers=='':base=Path('/sys/fs/cgroup');limitname='memory.max';usedname='memory.current'
  elif 'memory'in controllers.split(','):base=Path('/sys/fs/cgroup/memory');limitname='memory.limit_in_bytes';usedname='memory.usage_in_bytes'
  else:continue
  target=base/relative.lstrip('/')
  for node in (target,*target.parents):
   if node!=base and base not in node.parents:break
   try:
    raw=(node/limitname).read_text().strip();used=int((node/usedname).read_text());observed+=1
    if raw!='max'and int(raw)<2**60:bounds.append(max(0,int(raw)-used))
   except(OSError,ValueError):pass
 vals['cgroup_observed']=observed;vals['finite_cgroup_bounds']=len(bounds)
 if bounds:vals['MemAvailable']=min(vals['MemAvailable'],*bounds)
 require(observed>0,'actual memory cgroup inspection')
 return vals
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=5)
 active=None
def interrupt(sig,frame):stop();raise SystemExit(128+sig)
for sig in[signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupt)
signal.alarm(BUDGET)
def run(label,args,env,limit):
 global active
 remain=BUDGET-(time.monotonic()-started)-20;require(remain>0,'global audit deadline');state['phase']=label;save();print('phase',label,flush=True)
 with(OUT/(label+'.log')).open('w')as f:
  active=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:code=active.wait(timeout=min(limit,remain))
  finally:stop()
 require(code==0,'audit command failed '+label)
try:
 require(Path('/kaggle/input').is_dir()and OUT.is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'authorized CPU audit only');state['memory']=memory();require(state['memory']['MemAvailable']>=1<<30 and shutil.disk_usage('/tmp').free>=1<<30 and shutil.disk_usage(OUT).free>=256<<20,'bounded read-only audit resources')
 WORK.mkdir();env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONOPTIMIZE',None);env.update(PYTHONNOUSERSITE='1',PIP_NO_CACHE_DIR='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 venv=WORK/'venv';run('venv',[sys.executable,'-m','venv','--without-pip',venv],env,30);py=venv/'bin/python';run('dependencies',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','--report',OUT/'pip-resolution.json','numpy==2.3.5'],env,120)
 require(BUDGET-(time.monotonic()-started)>=380,'reserve complete 360-second raw audit')
 AUDIT_CODE='import hashlib,json,math,zipfile\nfrom pathlib import Path\nimport numpy as np\nif not __debug__:raise RuntimeError("audit assertions must be enabled")\nassert np.__version__==\'2.3.5\'\nexpected=\'d5b7f992249a3f6b553bd3e41f73cc5e590eeaae2d1380d9d34d81aaf3d316e8\'\ndef sha(p):\n with p.open(\'rb\')as f:return hashlib.file_digest(f,\'sha256\').hexdigest()\nmatches=[p for p in Path(\'/kaggle/input\').rglob(\'heldout-result.json\')if p.is_file()and p.stat().st_size<4<<20 and sha(p)==expected]\nassert len(matches)==1\np=matches[0];r=json.loads(p.read_text());base=p.parent.resolve()\nassert r[\'source_sha\']==\'d378c6cc055e3a31b0d4cc9dad54a5b41f10d792\'and r[\'status\']==\'measured_provisional_heldout\'and r[\'case\']==\'inventory-table\'and r[\'qualified\']is False and r[\'source_observer_full_neutral\']is True\nassert len(r[\'artifacts\'])<=1000 and sum(x[\'bytes\']for x in r[\'artifacts\'].values())<=8<<30\nverified={}\nfor name,row in r[\'artifacts\'].items():\n f=(base/name).resolve();assert base in f.parents and f.is_file()and f.stat().st_size==row[\'bytes\']and sha(f)==row[\'sha256\'];verified[name]=row\nroles=[(\'observed-control\',r[\'observed_control_rows\'],\'greedy\'),(\'upstream-on\',r[\'source_raw_rows\'],\'source\')]+[(arm,r[\'arms\'][arm][\'raw_rows\'],\'greedy\')for arm in[\'native-composed\',\'native-source-features\']]\nrows_verified=0\nfor folder,meta,stem in roles:\n archive=(base/folder/meta[\'archive\'][\'file\']).resolve();assert base in archive.parents and sha(archive)==meta[\'archive\'][\'sha256\']\n with zipfile.ZipFile(archive)as z:\n  names=[f\'{stem}.{row["position"]}.logits.f32\'if stem==\'greedy\'else f\'source.logits.{row["position"]}.f32\'for row in meta[\'rows\']]\n  assert z.namelist()==names and len(names)<=1536\n  for name,row in zip(names,meta[\'rows\']):\n   info=z.getinfo(name);assert info.file_size==row[\'bytes\']==248320*4 and info.compress_type==zipfile.ZIP_STORED\n   raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==row[\'sha256\'];values=np.frombuffer(raw,dtype=\'<f4\');assert np.isfinite(values).all()and int(values.argmax())==row[\'argmax\'];rows_verified+=1\nout=Path(\'/kaggle/working/inventory-raw-audit.json\');out.write_text(json.dumps(dict(schema=\'lighton4b-inventory-raw-audit/v1\',receipt_sha256=expected,owned_files_verified=len(verified),owned_bytes_verified=sum(v[\'bytes\']for v in verified.values()),raw_zip_archives_verified=4,raw_rows_finite_argmax_and_hash_verified=rows_verified,artifacts=verified,model_forward_executed=False),indent=2)+\'\\n\')\nprint(\'INVENTORY_RAW_AUDIT PASS files=\'+str(len(verified))+\' rows=\'+str(rows_verified),flush=True)\n'

 run('inventory-raw-audit',[py,'-c',AUDIT_CODE],env,360)
 result=json.loads((OUT/'inventory-raw-audit.json').read_text());require(result['model_forward_executed']is False and result['owned_files_verified']>0 and result['raw_rows_finite_argmax_and_hash_verified']>0,'complete independent inventory audit');state['audit_report_sha256']=hashlib.sha256((OUT/'inventory-raw-audit.json').read_bytes()).hexdigest();state['status']='completed'
except BaseException as error:state.update(error_type=type(error).__name__,error=str(error))
finally:signal.alarm(0);stop();state['elapsed_seconds']=time.monotonic()-started;save();print(json.dumps(state,indent=2),flush=True)
if state['status']!='completed':sys.exit(1)
