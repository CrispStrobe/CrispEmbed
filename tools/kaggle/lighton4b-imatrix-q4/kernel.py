#!/usr/bin/env python3
"""CPU-only bootstrap template; replace the source token with one reviewed commit."""
import hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE_SHA='__PINNED_SOURCE_SHA__'
GGML_SHA='3a27b6c665b191cb230be367db5c52d81dd3e2ac'
base=Path('/kaggle/working');out=base/'lighton-weighted-q4-results';out.mkdir(exist_ok=True)
report={'schema':'lighton4b-weighted-q4-kaggle-bootstrap/v1','status':'failed','source_sha':SOURCE_SHA,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'quantization_executed':False,'model_forward_executed':False,'gpu_enabled':False,'github_dispatch_executed':False}
source=base/'lighton-weighted-q4-source';scratch=base/'lighton-weighted-q4-scratch';created=[];active=None
try:
 assert re.fullmatch('[0-9a-f]{40}',SOURCE_SHA)and(base.is_dir()and Path('/kaggle/input').is_dir())
 assert not list(Path('/dev').glob('nvidia[0-9]*'))
 report['preflight']={'load':os.getloadavg(),'disk_free_bytes':shutil.disk_usage(base).free,'memory_swap':{k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapTotal','SwapFree')}}
 assert report['preflight']['disk_free_bytes']>18*2**30
 for folder in(source,scratch):assert not folder.exists();folder.mkdir();created.append(folder)
 commands=[['git','init',str(source)],['git','-C',str(source),'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git'],['git','-C',str(source),'fetch','--depth','1','origin',SOURCE_SHA],['git','-C',str(source),'checkout','--detach','FETCH_HEAD'],['git','-C',str(source),'submodule','update','--init','--depth','1','ggml']]
 with(out/'bootstrap.log').open('w')as log:
  for command in commands:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==SOURCE_SHA
 assert subprocess.check_output(['git','rev-parse','HEAD:ggml'],cwd=source,text=True).strip()==GGML_SHA
 assert subprocess.check_output(['git','-C','ggml','rev-parse','HEAD'],cwd=source,text=True).strip()==GGML_SHA
 venv=scratch/'venv'
 subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(venv)],check=True,timeout=120)
 python=venv/'bin/python'
 # Kaggle Debian Python may lack working ensurepip. Its existing pip module is visible
 # through system-site-packages, but this interpreter installs only into this venv.
 subprocess.run([str(python),'-c','import sys; assert sys.prefix != sys.base_prefix; import pip'],check=True,timeout=30)
 with(out/'packages.log').open('w')as log:
  subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','numpy==2.3.5','Pillow==11.3.0','gguf==0.19.0'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 env=dict(os.environ,HEAVY_OUT=str(out),HEAVY_SCRATCH=str(scratch),CRISPEMBED_BUILD_MODE='portable')
 started=time.monotonic()
 with(out/'controller.log').open('w')as log:
  active=subprocess.Popen([str(python),str(source/'tools/ci-heavy/lighton4b_imatrix_q4_candidate.py')],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=active.wait(timeout=110*60)
  except subprocess.TimeoutExpired:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
   report['deadline_reached']=True;raise
 report['seconds']=time.monotonic()-started;report['controller_returncode']=code
 receipt=json.loads((out/'weighted-q4-calibration.json').read_text());assert code==0 and receipt['status']=='measured'and receipt['actual248_applied']and receipt['both_frozen_null_scores_exact']and receipt['execution_host']=='kaggle-cpu';report['status']='measured';report['candidate']=receipt['candidate'];report['quantization_executed']=True;report['model_forward_executed']=True
except Exception as e:
 report['error_type']=type(e).__name__;report['error']='bounded CPU-only weighted-Q4 bootstrap or controller failed'
 raise
finally:
 (out/'bootstrap-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 # Only fresh task-owned transient checkout/builds are removed; all receipts/stats/logs remain.
 for folder in reversed(created):shutil.rmtree(folder)
