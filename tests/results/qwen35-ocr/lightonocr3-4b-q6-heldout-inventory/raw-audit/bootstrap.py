#!/usr/bin/env python3
"""Completed artifact integrity only; no Git, compiler, model weights or inference."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
OUT=Path('/kaggle/working');WORK=Path('/tmp/lighton4b-q6-heldout-inventory-raw-audit-v1')
NEW_SHA='4fa06de38118bd5b007fda809ad4f8602e8653961816757de21d563592689db9'
OLD_SHA='d5b7f992249a3f6b553bd3e41f73cc5e590eeaae2d1380d9d34d81aaf3d316e8'
OBS_SHA='f5d0077052b907e2d674caef9e7c4ae09b08456ff48a79f327cbdfd030b2dcc9'
CASE='inventory-table'
CODE='"""Independent completed-artifact audit; no model, build, network or text publication."""\nimport argparse,hashlib,json,signal,time,zipfile\nfrom pathlib import Path\nimport numpy as np\nVOCAB=248320\n\ndef require(ok,msg):\n if not ok:raise ValueError(msg)\ndef sha(path):\n h=hashlib.sha256()\n with Path(path).open(\'rb\')as f:\n  while b:=f.read(8<<20):h.update(b)\n return h.hexdigest()\ndef owned(root,name):\n p=Path(name);require(not p.is_absolute()and \'..\'not in p.parts,\'owned relative path\');f=root/p;require(f.is_file()and not f.is_symlink()and f.resolve().is_relative_to(root.resolve()),\'owned regular artifact\');return f\n\ndef audit_zip(root,record,ids,known=None):\n a=record[\'archive\'];p=owned(root,a[\'file\']);require(p.stat().st_size==a[\'bytes\']and(known.get(p.resolve())==a[\'sha256\']if known is not None else sha(p)==a[\'sha256\']),\'whole ZIP identity\');rows=record[\'rows\'];require(len(rows)==len(ids)and 1<=len(ids)<=1536,\'raw row census\');chain=hashlib.sha256()\n with zipfile.ZipFile(p)as z:\n  expected=[f\'greedy.{i}.logits.f32\'for i in range(len(ids))];require(len(z.infolist())==len(expected)and set(z.namelist())==set(expected),\'closed ZIP names\')\n  for i,(name,row,token)in enumerate(zip(expected,rows,ids)):\n   require(row[\'position\']==i and row[\'bytes\']==VOCAB*4 and row[\'argmax\']==token and type(token)is int and 0<=token<VOCAB,\'record position/vocabulary/argmax\')\n   info=z.getinfo(name);require(info.file_size==VOCAB*4 and not info.is_dir(),\'full vocabulary member\');b=z.read(name);require(hashlib.sha256(b).hexdigest()==row[\'sha256\'],\'each exact raw row SHA\');x=np.frombuffer(b,dtype=\'<f4\');require(x.size==VOCAB and np.isfinite(x).all()and int(x.argmax())==token,\'actual finite row/argmax\');chain.update(bytes.fromhex(row[\'sha256\']))\n return dict(rows=len(ids),archive_bytes=a[\'bytes\'],archive_sha256=a[\'sha256\'],ordered_row_sha256=chain.hexdigest())\n\ndef audit(new_path,old_path,observed_path,new_sha,old_sha,case):\n require(np.__version__==\'2.3.5\',\'pinned NumPy\');require(sha(new_path)==new_sha and sha(old_path)==old_sha,\'immutable result receipts\');require(sha(observed_path)==\'f5d0077052b907e2d674caef9e7c4ae09b08456ff48a79f327cbdfd030b2dcc9\',\'immutable observed producer\')\n root=new_path.parent;oldroot=old_path.parent;r=json.loads(new_path.read_text());old=json.loads(old_path.read_text());observed=json.loads(observed_path.read_text());require(r[\'source_sha\']==\'52e1cfcec5b515b4f4483fbeb86817250868f1f4\'and r[\'status\']==\'measured_provisional_q6_heldout\'and r[\'case\']==old[\'case\']==case and r[\'qualified\']is False and not r[\'reference_forward_executed\'],\'fixed completed diagnostic\');require(old[\'status\']==\'measured_provisional_heldout\'and old[\'source_sha\']==\'d378c6cc055e3a31b0d4cc9dad54a5b41f10d792\'and old[\'qualified\']is False and r[\'old_baseline_all_rows_stages_exact\']and r[\'pre_forward_input_seal_sha256\']==old[\'pre_forward_input_seal_sha256\'],\'retained baseline/input seal\');require(len(r[\'artifacts\'])<=2000 and sum(x[\'bytes\']for x in r[\'artifacts\'].values())<=12<<30,\'bounded artifact inventory\')\n total=new_path.stat().st_size;known={}\n for name,row in r[\'artifacts\'].items():\n  f=owned(root,name);require(f.suffix not in[\'.gguf\',\'.safetensors\'],\'no model payload in auditor\');require(f.stat().st_size==row[\'bytes\']and sha(f)==row[\'sha256\'],\'all owned artifact bytes\');known[f.resolve()]=row[\'sha256\'];total+=row[\'bytes\']\n oldseal=owned(oldroot,\'pre-forward-inputs.json\');require(sha(oldseal)==old[\'pre_forward_input_seal_sha256\']==sha(owned(root,\'pre-forward-inputs.json\')),\'actual original input seal identity\')\n baseline=r[\'arms\'][\'baseline-old\'];prior=old[\'arms\'][\'native-composed\'];require(baseline[\'ids\']==prior[\'ids\']and baseline[\'decoded\']==prior[\'decoded\']and baseline[\'complete\'],\'old complete native output identity\');require([x[\'sha256\']for x in baseline[\'raw_rows\'][\'rows\']]==[x[\'sha256\']for x in prior[\'raw_rows\'][\'rows\']],\'old full rows reference identity\')\n compared=0\n for name,row in old[\'artifacts\'].items():\n  if name.startswith(\'native-composed/prefill.\')or name.startswith(\'native-composed/vision.\')or name.startswith(\'native-composed/processor.\')or name in[\'native-composed/text.embedded.f32\',\'native-composed/prompt.ids\',\'native-composed/modalities.ids\']:\n   actual=owned(root,\'baseline-old/\'+name.split(\'/\',1)[1]);require(sha(actual)==row[\'sha256\'],\'old actual stage/vision byte identity\');compared+=1\n changed={\'CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS\',\'CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK\',\'CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED\'};off=r[\'native_gates\'][\'off\'];on=r[\'native_gates\'][\'on\'];require({k for k in off if off[k]!=on[k]}==changed and all(off[k]==\'0\'and on[k]==\'1\'for k in changed),\'exact three-gate diagnostic\')\n source_feature=owned(oldroot,\'upstream-on/source.vision.0.f32\');require(sha(source_feature)==old[\'artifacts\'][\'upstream-on/source.vision.0.f32\'][\'sha256\'],\'actual original upstream merged feature\');source_feature_sha=sha(source_feature)\n out={}\n for arm in [\'baseline-old\',\'corrected-composed\',\'corrected-source-features\']:\n  data=r[\'arms\'][arm];folder=root/arm;actual_ids=list(map(int,owned(folder,\'generated.ids\').read_text().split()));require(actual_ids==data[\'ids\']and owned(folder,\'generated.txt\').read_bytes().decode()==data[\'decoded\'],\'actual text/IDs identity\');summary=json.loads(owned(folder,\'native-summary.json\').read_text());require(summary==data[\'summary\']and summary[\'generated_tokens\']==len(actual_ids)and summary[\'eos\']==(actual_ids[-1]in[248044,248046])and(summary[\'eos\']or len(actual_ids)==1536),\'actual count/EOS/cap\');require(summary[\'provisional_heldout_cap\']==1536 and summary[\'source_fed_vision\']==(arm==\'corrected-source-features\'),\'actual arm role\');require(summary[\'converted_f32_weight_entries\']==summary[\'converted_f32_weight_bytes\']==0,\'no F32 text cache\');mode=arm!=\'baseline-old\';require(summary[\'physical_q4_repack\']==summary[\'physical_q6_tiled\']==mode and summary[\'physical_repack_entries\']==(216 if mode else 0)and 0<=summary[\'physical_repack_bytes\']<=2<<30 and bool(summary[\'physical_repack_bytes\'])==mode,\'actual corrected cache modes\');require(sha(owned(folder,\'vision.merged.f32\'))==source_feature_sha,\'all actual merged features upstream exact\')\n  for name in [\'prefill.logits.f32\',\'teacher.0.logits.f32\']:require(sha(owned(folder,name))==data[\'raw_rows\'][\'rows\'][0][\'sha256\'],\'actual first vocabulary row closure\')\n  for name in [\'prompt.ids\',\'modalities.ids\',\'text.embedded.f32\',\'prefill.input.f32\',\'prefill.positions.f32\']:\n   require(known[owned(folder,name).resolve()]==known[owned(root/\'baseline-old\',name).resolve()],\'all actual decoder inputs unchanged\')\n  if arm==\'corrected-composed\':\n   for name,row in old[\'artifacts\'].items():\n    if name.startswith(\'native-composed/vision.\')or name.startswith(\'native-composed/processor.\'):require(known[owned(folder,name.split(\'/\',1)[1]).resolve()]==row[\'sha256\'],\'corrected actual full vision/processor unchanged\')\n  out[arm]=audit_zip(folder,data[\'raw_rows\'],actual_ids,known)\n for tag in [\'observed-old\',\'observed-corrected\']:\n  record=r[tag][\'raw_rows\'];ids=[x[\'argmax\']for x in record[\'rows\']];require(len(ids)==167,\'observed full167 census\');out[tag]=audit_zip(root/tag,record,ids,known)\n  expected=[x[\'sha256\']for x in old[\'observed_control_rows\'][\'rows\']]if tag==\'observed-old\'else[observed[\'artifacts\'][f\'simple_form/on/greedy.{i}.logits.f32\'][\'sha256\']for i in range(167)];require([x[\'sha256\']for x in record[\'rows\']]==expected,\'observed rows actual frozen reference\')\n same_corrected_rows=[x[\'sha256\']for x in r[\'arms\'][\'corrected-composed\'][\'raw_rows\'][\'rows\']]==[x[\'sha256\']for x in r[\'arms\'][\'corrected-source-features\'][\'raw_rows\'][\'rows\']]\n return dict(corrected_arms_all_rows_identical=same_corrected_rows,corrected_arms_text_identical=r[\'arms\'][\'corrected-composed\'][\'decoded\']==r[\'arms\'][\'corrected-source-features\'][\'decoded\'],schema=\'lighton4b-q6-heldout-independent-raw-audit/v1\',status=\'complete\',case=case,receipt_sha256=new_sha,old_receipt_sha256=old_sha,owned_files=len(r[\'artifacts\'])+1,owned_bytes=total,raw_rows=sum(x[\'rows\']for x in out.values()),raw_archives=out,old_reference_stages_compared=compared,all_raw_finite_argmax_sha=True,all_owned_artifact_hashes=True,source_merged_all_arms_exact=True,reference_forward_executed=False,model_forward_executed=False,qualified=False,scope=\'Completed diagnostic bytes only. Historical reference rights/provisional-label limits remain binding. No text or IDs published by this auditor.\')\n\ndef main():\n p=argparse.ArgumentParser();p.add_argument(\'--new\',type=Path,required=True);p.add_argument(\'--old\',type=Path,required=True);p.add_argument(\'--observed\',type=Path,required=True);p.add_argument(\'--new-sha\',required=True);p.add_argument(\'--old-sha\',required=True);p.add_argument(\'--case\',required=True);p.add_argument(\'--out\',type=Path,required=True);a=p.parse_args();start=time.monotonic();r=dict(status=\'failed\',qualified=False,model_forward_executed=False)\n def stop(sig,frame):raise TimeoutError(\'bounded artifact audit interrupted\')\n for sig in[signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,stop)\n signal.alarm(600)\n try:r=audit(a.new,a.old,a.observed,a.new_sha,a.old_sha,a.case)\n except Exception as e:r.update(error_type=type(e).__name__,error=str(e));raise\n finally:signal.alarm(0);r[\'elapsed_seconds\']=time.monotonic()-start;a.out.write_text(json.dumps(r,indent=2)+\'\\n\')\nif __name__==\'__main__\':main()\n'
CODE_SHA='cbd38cca098033b7354c464403be818b8f3f53eb1074ad94533e726af03edfd8'
started=time.monotonic();active=None;r=dict(status='failed',case=CASE,model_forward_executed=False,model_acquisition=False,qualified=False)
def require(ok,message):
 if not ok:raise RuntimeError(message)
def save():(OUT/'raw-audit-bootstrap.json').write_text(json.dumps(r,indent=2)+'\n')
def stop():
 global active
 if active is not None and active.poll()is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 active=None
def interrupted(sig,frame):raise TimeoutError('whole bootstrap deadline or cancellation')
for sig in[signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
signal.alarm(900)
def run(label,args,env,seconds):
 global active
 remaining=900-(time.monotonic()-started)-20
 if remaining<=0:raise TimeoutError('bootstrap budget closed')
 r['phase']=label;save();print('phase',label,flush=True)
 with(OUT/(label+'.log')).open('w')as log:
  try:
   active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);code=active.wait(timeout=min(seconds,remaining))
  finally:stop()
 if code:raise RuntimeError(label+' nonzero')
def locate(name,digest):
 rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and hashlib.sha256(p.read_bytes()).hexdigest()==digest]
 if len(rows)!=1:raise RuntimeError('one immutable completed receipt required')
 return rows[0]
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
try:
 if not OUT.is_dir()or not Path('/kaggle/input').is_dir()or list(Path('/dev').glob('nvidia[0-9]*')):raise RuntimeError('private CPU artifact audit only')
 if len(NEW_SHA)!=64 or any(c not in '0123456789abcdef'for c in NEW_SHA):raise RuntimeError('actual completed receipt binding required')
 if hashlib.sha256(CODE.encode()).hexdigest()!=CODE_SHA:raise RuntimeError('exact auditor source SHA required')
 new=locate('q6-heldout.json',NEW_SHA);old=locate('heldout-result.json',OLD_SHA);observed=locate('q6-tiled-full-ab.json',OBS_SHA)
 r['capacity']=memory()
 if r['capacity']['MemAvailable']<1<<30:raise RuntimeError('1GiB cgroup-aware audit RAM required')
 if shutil.disk_usage(OUT).free<1<<30:raise RuntimeError('bounded audit output capacity')
 WORK.mkdir();audit=WORK/'audit.py';audit.write_text(CODE);r['auditor_sha256']=CODE_SHA
 env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONOPTIMIZE',None);env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PIP_NO_CACHE_DIR='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 run('venv',[sys.executable,'-m','venv','--without-pip',WORK/'venv'],env,30);py=WORK/'venv/bin/python';run('numpy',[sys.executable,'-m','pip','--python',py,'install','--disable-pip-version-check','numpy==2.3.5'],env,90);run('dependencies',[sys.executable,'-m','pip','--python',py,'check'],env,30)
 if 900-(time.monotonic()-started)<640:raise TimeoutError('whole auditor reserve unavailable')
 run('audit',[py,'-B',audit,'--new',new,'--old',old,'--observed',observed,'--new-sha',NEW_SHA,'--old-sha',OLD_SHA,'--case',CASE,'--out',OUT/'raw-audit.json'],env,620)
 result=json.loads((OUT/'raw-audit.json').read_text())
 if result['status']!='complete'or result['receipt_sha256']!=NEW_SHA or result['case']!=CASE or not result['all_raw_finite_argmax_sha']:raise RuntimeError('complete integrity result required')
 if result['owned_files']!=466 or result['owned_bytes']!=1767787387 or result['raw_rows']!=1078:raise RuntimeError('exact inventory census required')
 r['status']='complete'
except Exception as e:r.update(error_type=type(e).__name__,error=str(e))
finally:stop();signal.alarm(0);r['elapsed_seconds']=time.monotonic()-started;save();shutil.rmtree(WORK,ignore_errors=True)
if r['status']!='complete':sys.exit(1)
