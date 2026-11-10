#!/usr/bin/env python3
"""One bounded CPU-only packing/decoded worker using the exact native control bundle."""
import hashlib,json,os,re,shutil,signal,subprocess,sys,time
from pathlib import Path
SOURCE_SHA='__SOURCE_SHA__'
CONTROLS_SOURCE_SHA='__CONTROLS_SOURCE_SHA__'
CONTROLS_RECEIPT_SHA='__CONTROLS_RECEIPT_SHA__'
RUNTIME_MANIFEST_SHA='__RUNTIME_MANIFEST_SHA__'
PRODUCER_SOURCE_SHA='__PRODUCER_SOURCE_SHA__'
PRODUCER_RECEIPT_SHA='__PRODUCER_RECEIPT_SHA__'
base=Path('/kaggle/working');out=base/'lighton-hf-tiled-out-candidate-decode';source=base/'lighton-hf-tiled-out-decode-source';scratch=base/'lighton-hf-tiled-out-decode-scratch';out.mkdir(exist_ok=True);created=[];active=None;started=time.monotonic()
r={'schema':'lighton4b-hf-tiled-out-candidate-decode-bootstrap/v1','status':'failed','source_sha':SOURCE_SHA,'controls_source_sha':CONTROLS_SOURCE_SHA,'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_downloaded_by_bootstrap':False,'uses_pinned_remote_model_mount':True,'model_inference_executed':False,'gpu_enabled':False}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def stop(*args):
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 raise TimeoutError('bounded candidate worker deadline')
def run(label,args,timeout=240,env=None):
 global active
 print('PHASE '+label,flush=True);start=time.monotonic();cursor=0;carry='';heartbeat=0
 with(out/(label+'.log')).open('w')as log:
  active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   while True:
    remaining=timeout-(time.monotonic()-start)
    if remaining<=0:stop()
    try:code=active.wait(timeout=min(20,remaining));done=True
    except subprocess.TimeoutExpired:done=False
    with(out/(label+'.log')).open()as reader:reader.seek(cursor);carry+=reader.read();cursor=reader.tell()
    lines=carry.split('\n');carry=lines.pop()
    for line in lines:
     if line.startswith(('PHASE ','PROGRESS ','MEASURED ')):print(line,flush=True)
    elapsed=int(time.monotonic()-start)
    if elapsed//60>heartbeat:heartbeat=elapsed//60;print('HEARTBEAT '+label+' elapsed='+str(elapsed),flush=True)
    if done:break
  finally:active=None
 if code:raise RuntimeError('bounded worker stage failed: '+label)
try:
 assert all(re.fullmatch('[0-9a-f]{40}',s)for s in(SOURCE_SHA,CONTROLS_SOURCE_SHA,PRODUCER_SOURCE_SHA))and all(re.fullmatch('[0-9a-f]{64}',s)for s in(CONTROLS_RECEIPT_SHA,RUNTIME_MANIFEST_SHA,PRODUCER_RECEIPT_SHA))
 assert base.is_dir()and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and shutil.which('git')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(260*60)
 memory={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapTotal','SwapFree')};available=memory['MemAvailable']
 if Path('/sys/fs/cgroup/memory.max').exists():
  limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if limit!='max':available=min(available,max(0,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())))
 assert available>=8*2**30 and shutil.disk_usage(base).free>=14*2**30
 r['preflight']={'available_ram_bytes':available,'memory_swap':memory,'free_disk_bytes':shutil.disk_usage(base).free,'logical_cpus':os.cpu_count(),'load':os.getloadavg()}
 matches=[p for p in Path('/kaggle/input').rglob('bootstrap-receipt.json')if p.stat().st_size<65536 and sha(p)==CONTROLS_RECEIPT_SHA];assert len(matches)==1;control_root=matches[0].parent;c=json.loads(matches[0].read_text());assert c['status']=='passed'and c['source_sha']==CONTROLS_SOURCE_SHA and c['checkpoint_downloaded']is False and c['checks']=={'hf-tiled-fullwidth-operator':39,'hf-tiled-recipe':66710,'old-hf-imatrix-recipe':1147,'old-physical-modes':142,'old-q6-tied-head':84,'old-stream-projection':None,'public-release-fence':39}
 manifest=control_root/'runtime-manifest.json';assert sha(manifest)==RUNTIME_MANIFEST_SHA;m=json.loads(manifest.read_text());assert m['source_sha']==CONTROLS_SOURCE_SHA and m['ggml_sha']=='3a27b6c665b191cb230be367db5c52d81dd3e2ac'
 for rel,entry in m['files'].items():
  parts=Path(rel).parts;assert len(parts)==2 and parts[0]in('bin','lib')and all(re.fullmatch('[A-Za-z0-9_.+-]+',p)for p in parts)and'..'not in parts;p=control_root/'runtime'/rel;assert p.stat().st_size==entry['bytes']and sha(p)==entry['sha256']
 r['native_controls_receipt_sha256']=CONTROLS_RECEIPT_SHA;r['runtime_manifest_sha256']=RUNTIME_MANIFEST_SHA
 for path in(source,scratch):assert not path.exists();path.mkdir();created.append(path)
 for label,args in [('init',['git','init',source]),('remote',['git','-C',source,'remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git']),('fetch',['git','-C',source,'fetch','--depth','1','origin',SOURCE_SHA,CONTROLS_SOURCE_SHA]),('checkout',['git','-C',source,'checkout','--detach',SOURCE_SHA])]:run(label,args)
 assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==SOURCE_SHA
 for path in('src','tests/lighton4b_native_document.cpp','tests/lighton4b_native_case_preflight.cpp'):
  current=subprocess.check_output(['git','-C',source,'rev-parse','HEAD:'+path],text=True).strip();expected=subprocess.check_output(['git','-C',source,'rev-parse',CONTROLS_SOURCE_SHA+':'+path],text=True).strip();assert current==expected;r.setdefault('exact_control_source_objects',{})[path]=current
 venv=scratch/'venv';run('venv',[sys.executable,'-m','venv','--without-pip','--system-site-packages',venv]);python=venv/'bin/python';run('venv-prefix',[python,'-c','import sys,pip;assert sys.prefix!=sys.base_prefix']);run('packages',[python,'-m','pip','install','--disable-pip-version-check','numpy==2.3.5','Pillow==11.3.0','gguf==0.19.0','PyYAML==6.0.3','tqdm==4.67.1','requests==2.32.5'],300)
 run('package-preflight',[python,'-c',"import importlib.metadata as m;import numpy,PIL,gguf,yaml,tqdm,requests;from gguf.quants import dequantize;assert all(m.version(k)==v for k,v in {'numpy':'2.3.5','Pillow':'11.3.0','gguf':'0.19.0','PyYAML':'6.0.3','tqdm':'4.67.1','requests':'2.32.5'}.items())"])
 for rel in('tools/ci-heavy/lighton4b_hf_tiled_out_decode.py','tools/lighton4b_hf_tiled_out_pack.py','tools/lighton4b_hf_tiled_out_reproduce.py','docs/lighton4b-hf-tiled-out-reproducer-protocol-v1.json','docs/lighton4b-hf-tiled-out-observed-threecase-v1.json','docs/lighton4b-imatrix-q4km-quality-floor-v1.json','docs/lighton4b-imatrix-q4km-form-floor-v1.json','tests/regression/images/cc0/simple_form.png'):assert(source/rel).is_file()
 runtime=scratch/'runtime';shutil.copytree(control_root/'runtime',runtime)
 for name in('lighton4b-native-document','lighton4b-native-case-preflight'):(runtime/'bin'/name).chmod(0o755)
 env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(HEAVY_OUT=str(out),HEAVY_SCRATCH=str(scratch),HEAVY_BUILD=str(runtime/'bin'),HEAVY_RUNTIME_LIB=str(runtime/'lib'),LD_LIBRARY_PATH=str(runtime/'lib'),LIGHTON4B_PRODUCER_SOURCE_SHA=PRODUCER_SOURCE_SHA,LIGHTON4B_PRODUCER_RECEIPT_SHA256=PRODUCER_RECEIPT_SHA)
 for name in('lighton4b-native-document','lighton4b-native-case-preflight'):assert subprocess.run([runtime/'bin'/name],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30).returncode==2
 # No rebuild, new arithmetic or quantization: copy only verified helper-produced
 # original-F16 payloads, then evaluate the exact control-passed native binaries.
 run('candidate-and-three-observed-decodes',[python,source/'tools/ci-heavy/lighton4b_hf_tiled_out_decode.py'],245*60,env)
 measured=json.loads((out/'hf-tiled-out-complete-decode.json').read_text());assert measured['status']=='measured'and set(measured['cases'])=={'simple_form','simple_table','arabic_printed_line'};r.update(status='measured',candidate=measured['candidate'],complete_decode_receipt_sha256=sha(out/'hf-tiled-out-complete-decode.json'))
except Exception as error:r['error_type']=type(error).__name__;r['error']='bounded worker failed; all partial outputs retained';raise
finally:
 signal.alarm(0);inner=out/'hf-tiled-out-complete-decode.json'
 if inner.exists():
  measured=json.loads(inner.read_text());r['model_inference_attempted']=measured.get('native_model_forward_attempted',False);r['model_inference_executed']=measured.get('native_model_forward_executed',False)
  if'candidate'in measured:r['candidate']=measured['candidate']
 r['seconds']=time.monotonic()-started;(out/'bootstrap-receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 for path in reversed(created):shutil.rmtree(path)
