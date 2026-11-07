#!/usr/bin/env python3
"""Closed CPU-only original-F16 24-matrix proof; no model graph or donor weights."""
import hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE_SHA='__SOURCE_SHA__'
CONTROLS_SOURCE_SHA='__CONTROLS_SOURCE_SHA__'
CONTROLS_RECEIPT_SHA='__CONTROLS_RECEIPT_SHA__'
RUNTIME_MANIFEST_SHA='__RUNTIME_MANIFEST_SHA__'
base=Path('/kaggle/working');out=base/'lighton-hf-tiled-out-proof';source=base/'lighton-hf-tiled-out-proof-source';scratch=base/'lighton-hf-tiled-out-proof-scratch';out.mkdir(exist_ok=True);created=[];active=None;started=time.monotonic()
r={'schema':'lighton4b-hf-tiled-out-proof-bootstrap/v1','status':'failed','source_sha':SOURCE_SHA,'controls_source_sha':CONTROLS_SOURCE_SHA,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_inference_executed':False,'full_f32_model_instantiated':False,'quantized_donor_used':False,'gpu_enabled':False}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def interrupted(signum,frame):
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 raise TimeoutError('bounded original-F16 proof deadline')
def run(label,args,timeout=240,env=None):
 global active
 print('PHASE '+label,flush=True)
 with(out/(label+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:code=active.wait(timeout=timeout)
  except subprocess.TimeoutExpired:interrupted(signal.SIGALRM,None)
  finally:active=None
 if code:raise RuntimeError('bounded proof stage failed: '+label)
def available():
 memory={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapTotal','SwapFree')};n=memory['MemAvailable']
 if Path('/sys/fs/cgroup/memory.max').exists():
  limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if limit!='max':n=min(n,max(0,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())))
 return n,memory
try:
 assert all(re.fullmatch('[0-9a-f]{40}',v)for v in(SOURCE_SHA,CONTROLS_SOURCE_SHA))and all(re.fullmatch('[0-9a-f]{64}',v)for v in(CONTROLS_RECEIPT_SHA,RUNTIME_MANIFEST_SHA))
 assert base.is_dir()and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))
 for executable in('git',):assert shutil.which(executable)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(3000)
 ram,memory=available();assert ram>=6*2**30 and shutil.disk_usage(base).free>=14*2**30
 cpu={}
 for line in Path('/proc/cpuinfo').read_text().splitlines():
  if ':'in line:
   key,value=line.split(':',1)
   if key.strip()in('vendor_id','model name','flags')and key.strip()not in cpu:cpu[key.strip()]=value.strip()
 r['cpu_profile']=cpu
 r['preflight']={'available_ram_bytes':ram,'free_disk_bytes':shutil.disk_usage(base).free,'load':os.getloadavg(),'memory_swap':memory,'logical_cpus':os.cpu_count()}
 matches=[p for p in Path('/kaggle/input').rglob('bootstrap-receipt.json')if p.stat().st_size<65536 and sha(p)==CONTROLS_RECEIPT_SHA];assert len(matches)==1
 controls=matches[0].parent;c=json.loads(matches[0].read_text());assert c['status']=='passed'and c['source_sha']==CONTROLS_SOURCE_SHA and c['checkpoint_downloaded']is False
 manifest=controls/'runtime-manifest.json';assert sha(manifest)==RUNTIME_MANIFEST_SHA;m=json.loads(manifest.read_text());assert m['source_sha']==CONTROLS_SOURCE_SHA
 assert set(m['files'])and all(not Path(k).is_absolute()and'..'not in Path(k).parts for k in m['files'])
 for name,record in m['files'].items():p=controls/'runtime'/name;assert p.stat().st_size==record['bytes']and sha(p)==record['sha256']
 r['controls_receipt_sha256']=CONTROLS_RECEIPT_SHA;r['runtime_manifest_sha256']=RUNTIME_MANIFEST_SHA
 for path in(source,scratch):assert not path.exists();path.mkdir();created.append(path)
 for label,args in [('init',['git','init',source]),('remote',['git','-C',source,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']),('fetch',['git','-C',source,'fetch','--depth','1','origin',SOURCE_SHA]),('checkout',['git','-C',source,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
 assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==SOURCE_SHA
 protocol=json.loads((source/'docs/lighton4b-hf-tiled-out-reproducer-protocol-v1.json').read_text());r['protocol_sha256']=sha(source/'docs/lighton4b-hf-tiled-out-reproducer-protocol-v1.json')
 for rel in('tools/lighton4b_hf_tiled_out_reproduce.py','tools/lighton4b_imatrix_adapter.py','tools/lighton4b_imatrix_q4km_recipe.py','tools/lighton4b_q6_recipe.py','tools/gguf_file_type_canonicalize.py','tests/test_lighton4b_hf_tiled_out_reproduce.py'):assert(source/rel).is_file()
 venv=scratch/'venv';run('venv',[sys.executable,'-m','venv','--without-pip','--system-site-packages',venv]);python=venv/'bin/python';run('venv-prefix',[python,'-c','import sys,pip;assert sys.prefix!=sys.base_prefix']);run('packages',[python,'-m','pip','install','--disable-pip-version-check','numpy==2.3.5','gguf==0.19.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5'],300)
 run('dependency-preflight',[python,'-c',"import importlib.metadata as m;import numpy,gguf,yaml,tqdm,requests;from gguf.quants import dequantize;from gguf import GGUFReader;assert all(m.version(k)==v for k,v in {'numpy':'2.3.5','gguf':'0.19.0','PyYAML':'6.0.3','tqdm':'4.67.1','requests':'2.32.5'}.items())"])
 bundle=scratch/'runtime';shutil.copytree(controls/'runtime',bundle);helper=bundle/'bin/lighton4b-tiled-out-rows';helper.chmod(0o755);env=dict(os.environ,LD_LIBRARY_PATH=str(bundle/'lib'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');assert subprocess.run([helper],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30).returncode==2
 run('python-controls',[python,'-m','unittest','discover','-s',source/'tests','-p','test_lighton4b_hf_tiled_out_reproduce.py'],env=env)
 # Download/check pins inside the isolated interpreter after all dependency and
 # executable probes, before any quantization. Redirect locations are not logged.
 download=scratch/'download.py';download.write_text('''import hashlib,json,os,sys,requests\nfrom pathlib import Path\np=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);records={}\nfor key in ('statistics','source_original_f16'):\n pin=p[key];path=root/(key+'.gguf');h=hashlib.sha256();n=0;url='https://huggingface.co/'+pin['repository']+'/resolve/'+pin['revision']+'/'+pin['file'];print('DOWNLOAD '+key,flush=True)\n with requests.get(url,stream=True,timeout=(30,120))as response:\n  response.raise_for_status()\n  with path.open('xb')as f:\n   for block in response.iter_content(1048576):\n    n+=len(block);assert n<=pin['bytes'];h.update(block);f.write(block)\n assert n==pin['bytes']and h.hexdigest()==pin['sha256'];records[key]={'bytes':n,'sha256':h.hexdigest()};print('AUTHENTICATED '+key,flush=True)\nPath(sys.argv[3]).write_text(json.dumps(records,indent=2,sort_keys=True)+'\\n')\n''')
 ram,_=available();assert ram>=6*2**30 and shutil.disk_usage(base).free>=14*2**30
 run('download-original-and-stats',[python,download,source/'docs/lighton4b-hf-tiled-out-reproducer-protocol-v1.json',scratch,out/'download-receipt.json'],1200,env)
 run('paired24-proof',[python,source/'tools/lighton4b_hf_tiled_out_reproduce.py','--source',scratch/'source_original_f16.gguf','--statistics',scratch/'statistics.gguf','--helper',helper,'--output',out/'tensors','--receipt',out/'paired24-receipt.json'],1800,env)
 measured=json.loads((out/'paired24-receipt.json').read_text());assert measured['status']=='reproduced'and measured['all24_native_controls_exact']and measured['all24_publisher_tiled_payloads_exact']and measured['candidate_conversion_allowed']and len(measured['tensors'])==24
 r.update(status='reproduced',paired_receipt_sha256=sha(out/'paired24-receipt.json'),all24_native_controls_exact=True,all24_publisher_tiled_payloads_exact=True,candidate_conversion_allowed=True)
except Exception as error:r['error_type']=type(error).__name__;r['error']='bounded original-F16 proof failed; original negative retained';raise
finally:
 signal.alarm(0);r['seconds']=time.monotonic()-started;(out/'bootstrap-receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 for path in reversed(created):shutil.rmtree(path)
