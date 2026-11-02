#!/usr/bin/env python3
"""Hosted-only same-input replay of the layer3 FFN mul_mat (native vs WASM ggml kernels).

Order: preflight -> build (save full compile inventories, admit kernel profiles
against production) -> weight-free authored selftest on both builds -> only then
fetch the pinned v4 captures and the frozen model -> replay -> report.
Numerical negatives exit 1 after the complete report. No default/gate changes.
"""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/wasm-layer3-ffn-matmul'),str(ROOT/'tools'),str(ROOT/'tools/ci-heavy')]
import analysis as A
from successor_assets import api,artifact
from successor_wasm_trace_contract import MODEL_SHA
PROFILE=ROOT/'docs/wasm-layer3-ffn-matmul-v1.json'
WORKFLOW='.github/workflows/wasm-layer3-ffn-matmul-v1.yml'
SRC=ROOT/'tests/wasm-layer3-ffn-matmul'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
require=A.require

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def git(*a,cwd=ROOT):return subprocess.check_output(['git',*a],cwd=cwd,text=True).strip()

def validate(p):
 require(p['schema']=='crispembed-wasm-layer3-ffn-matmul-v1'and p['workflow']==WORKFLOW and p['ggml']==GGML,'Protocol identity differs')
 require(p['runtime_change']is False and p['qualification']is False and p['candidate_enabled']is False,'Diagnostic-only scope differs')
 require(p['budgets']==dict(max_absolute=A.ABSOLUTE,minimum_row_cosine=A.COSINE),'Budgets differ')
 require(sorted(p['captures']['members'])==sorted(f'{arm}/{n}'for arm in['native','wasm0','wasm1']for n in['attention_output','gelu','fc1_raw','fc2_raw']),'Capture member set differs')
 return p

def source():
 s=os.environ.get('GITHUB_SHA','');require(re.fullmatch('[0-9a-f]{40}',s)and git('rev-parse','HEAD')==s,'Exact source required')
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA only')
 p=validate(read(PROFILE))
 for path,d in p['owned_sha256'].items():require(sha(ROOT/path)==d,'Owned bytes differ: '+path)
 require(not git('status','--porcelain','--',*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty source rejected')
 require(git('rev-parse','HEAD',cwd=ROOT/'ggml')==GGML and not git('status','--porcelain','--untracked-files=no',cwd=ROOT/'ggml'),'Clean actual a06 required')
 meta=api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 require(meta['head_sha']==s and meta['event']=='workflow_dispatch'and meta['status']=='in_progress'and meta['path']==WORKFLOW,'Service/source mismatch')
 return p

def command(out,name,argv,cwd=ROOT,timeout=2400,expected=(0,)):
 log=out/'review'/(name+'.log');t=time.monotonic()
 with log.open('wb')as f:
  try:r=subprocess.run([str(a)for a in argv],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);code,timed=r.returncode,False
  except subprocess.TimeoutExpired:code,timed=None,True
 write(out/'review'/(name+'-command.json'),dict(argv=[str(a)for a in argv],exit=code,timed_out=timed,seconds=time.monotonic()-t,log=log.name,sha256=sha(log)))
 require(not timed and code in expected,'Command rejected: '+name);return code

def probe(out,name,kind,args,timeout=1200):
 """Run the native binary or the WASM module under node; stdout JSON goes to review/<name>.json."""
 argv=[out/'native'/'ffn-matmul-probe',*args]if kind=='native'else['node',out/'wasm'/'ffn-matmul-probe.js',*args]
 stdout=out/'review'/(name+'.json');t=time.monotonic()
 with stdout.open('wb')as f,(out/'review'/(name+'.stderr.log')).open('wb')as e:
  r=subprocess.run([str(a)for a in argv],stdout=f,stderr=e,timeout=timeout)
 write(out/'review'/(name+'-command.json'),dict(argv=[str(a)for a in argv],exit=r.returncode,seconds=time.monotonic()-t,stdout_sha256=sha(stdout)))
 require(r.returncode==0,'Probe rejected: '+name);return read(stdout)

def preflight(out):
 source();out.mkdir(exist_ok=False)
 for d in ['review','raw','selftest/native','selftest/wasm','replay/native','replay/wasm','inputs']:(out/d).mkdir(parents=True)

def build(out):
 p=source()
 command(out,'native-configure',['cmake','-S',SRC,'-B',out/'native','-DCMAKE_BUILD_TYPE=Release'])
 command(out,'wasm-configure',['emcmake','cmake','-S',SRC,'-B',out/'wasm','-DCMAKE_BUILD_TYPE=Release','-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm',
         '-DCMAKE_C_FLAGS=-pthread','-DCMAKE_CXX_FLAGS=-pthread'])
 admitted={}
 for kind in ['native','wasm']:
  data=(out/kind/'compile_commands.json').read_bytes();(out/'review'/(kind+'-compile.json')).write_bytes(data)  # full inventory first
  fixture=SRC/f'frozen-40852835039-full-{kind}-compile.json';require(sha(fixture)==p['production_inventories'][kind],'Production inventory fixture differs')
  try:admitted[kind]=A.admit(kind,json.loads(data),read(fixture),'tests/wasm-layer3-ffn-matmul/probe.cpp')
  except ValueError as e:write(out/'review'/(kind+'-admission.json'),dict(admitted=False,error=str(e)));raise
  write(out/'review'/(kind+'-admission.json'),dict(admitted[kind],admitted=True))
 command(out,'native-build',['cmake','--build',out/'native','--target','ffn-matmul-probe','--parallel','2'])
 command(out,'wasm-build',['cmake','--build',out/'wasm','--target','ffn-matmul-probe','--parallel','2'])
 for kind in ['native','wasm']:require(sha(out/kind/'compile_commands.json')==sha(out/'review'/(kind+'-compile.json')),'Inventory changed during build')

def collect(out,phase,meta):
 files={}
 for side in ['native','wasm']:
  for c in meta[side]['cases']:
   for k in ['production','native_order','generic_order']:files[(side,c['label'],k)]=(out/phase/side/(c['label']+'.'+k+'.f32')).read_bytes()
   files[(side,c['label'],'activation')]=(out/phase/side/(c['label']+'.activation.bin')).read_bytes()
 return files

def selftest(out):
 source();meta={side:probe(out,'selftest-'+side,side,['selftest',out/'selftest'/side])for side in ['native','wasm']}
 result=A.selftest(meta,collect(out,'selftest',meta));write(out/'review/selftest.json',dict(result,meta=meta,model_access=False))
 require(result['passed'],'Weight-free selftest must pass before model access')

def fetch(out,key,pin,limit):
 svc=api('actions/runs/'+str(pin['run']))
 require(svc['head_sha']==pin['source']and svc['path']==pin['workflow']and svc['status']=='completed'and svc['conclusion']==pin['conclusion']
         and svc['event']=='workflow_dispatch'and svc['run_attempt']==1,'Frozen service identity differs: '+key)
 path=out/(key+'.zip');meta=artifact(pin['run'],pin['name'],path,pin['bytes'])
 require(meta['id']==pin['id']and meta['size_in_bytes']==pin['bytes']and meta['digest']=='sha256:'+pin['sha256']and sha(path)==pin['sha256'],'Frozen artifact differs: '+key)
 write(out/'review'/(key+'-artifact.json'),meta);z=zipfile.ZipFile(path)
 require(len(z.infolist())<=limit,'Bounded archive');return z

def prepare(out):
 p=source();require(read(out/'review/selftest.json')['passed'],'Selftest before model access')
 z=fetch(out,'captures',p['captures']['artifact'],8192)
 for name,pin in p['captures']['members'].items():
  data=z.read(pin['member']);require(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],'Capture member differs: '+name)
  (out/'inputs'/(name.replace('/','.')+'.f32')).write_bytes(data)
 z=fetch(out,'model',p['model']['artifact'],1024)
 found=[n for n in z.namelist()if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')];require(len(found)==1,'Exact frozen model member')
 data=z.read(found[0]);require(hashlib.sha256(data).hexdigest()==MODEL_SHA,'Exact frozen model required');(out/'inputs/model.gguf').write_bytes(data)
 write(out/'review/prepared.json',dict(captures=len(p['captures']['members']),model_sha256=MODEL_SHA,original_forward=False))

CASES=[('fc1:fc1_native','native.attention_output'),('fc1:fc1_wasm0','wasm0.attention_output'),('fc1:fc1_wasm1','wasm1.attention_output'),
       ('fc2:fc2_native','native.gelu'),('fc2:fc2_wasm0','wasm0.gelu')]

def replay(out):
 source();require(read(out/'review/prepared.json')['model_sha256']==MODEL_SHA,'Prepared inputs required')
 args=[x for spec,inp in CASES for x in (spec,out/'inputs'/(inp+'.f32'))]
 meta={side:probe(out,'replay-'+side,side,['replay',out/'inputs/model.gguf',out/'replay'/side,*args])for side in ['native','wasm']}
 captured={(arm,op):(out/'inputs'/(arm+'.'+op+'.f32')).read_bytes()for arm in ['native','wasm0','wasm1']for op in ['fc1_raw','fc2_raw']}
 result=A.replay(meta,collect(out,'replay',meta),captured)
 report=dict(schema='crispembed-wasm-layer3-ffn-matmul-result-v1',source=os.environ['GITHUB_SHA'],completed=True,meta=meta,**result,
             passed=result['controls_passed']and result['four_way_within_budget'],candidate_enabled=False,qualification=False,original_forward=False)
 write(out/'review/ffn-matmul.json',report)
 require(result['controls_passed'],'Own-reproduction/emulation controls must pass')
 return 0 if report['passed']else 1

def main():
 ph=dict(preflight=preflight,build=build,selftest=selftest,prepare=prepare,replay=replay)
 a=argparse.ArgumentParser(__doc__);a.add_argument('phase',choices=list(ph));a.add_argument('--output',type=Path,required=True);a=a.parse_args();out=a.output.resolve()
 try:return ph[a.phase](out)
 except Exception as e:
  (out/'review').mkdir(parents=True,exist_ok=True)
  write(out/'review'/('rejection-'+a.phase+'.json'),dict(source=os.environ.get('GITHUB_SHA'),phase=a.phase,completed=False,error=str(e),qualification=False));raise
 finally:
  if (out/'review').exists():
   files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for d in ['review','selftest','replay','inputs']if (out/d).exists()
          for p in (out/d).rglob('*')if p.is_file()and p.name!='file-manifest.json'}
   write(out/'review/file-manifest.json',files)
if __name__=='__main__':raise SystemExit(main())
