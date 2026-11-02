#!/usr/bin/env python3
"""Hosted-only layer0 attention op-by-op same-input replay (native vs WASM ggml).

Order: preflight -> build (save full compile inventories, admit kernel profiles
against production) -> weight-free authored controls (determinism, op isolation)
-> only then the pinned v4 captures and frozen model -> replay -> report.
Any same-input difference is reported; numerical budget failures exit 1.
"""
import argparse,hashlib,importlib.util,json,os,re,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
_s=importlib.util.spec_from_file_location('ffn_caller',ROOT/'tools/ci-heavy/wasm_layer3_ffn_matmul_v1.py')
FC=importlib.util.module_from_spec(_s);_s.loader.exec_module(FC)
_s=importlib.util.spec_from_file_location('layer0_analysis',ROOT/'tools/wasm-layer0-attention/analysis.py')
A=importlib.util.module_from_spec(_s);_s.loader.exec_module(A)
from successor_wasm_trace_contract import MODEL_SHA
PROFILE=ROOT/'docs/wasm-layer0-attention-v1.json'
WORKFLOW='.github/workflows/wasm-layer0-attention-v1.yml'
SRC=ROOT/'tests/wasm-layer0-attention'
FIX=ROOT/'tests/wasm-layer3-ffn-matmul'
require,sha,read,write,git,command=A.require,FC.sha,FC.read,FC.write,FC.git,FC.command

def validate(p):
 require(p['schema']=='crispembed-wasm-layer0-attention-v1'and p['workflow']==WORKFLOW and p['ggml']==FC.GGML,'Protocol identity differs')
 require(p['runtime_change']is False and p['qualification']is False and p['candidate_enabled']is False,'Diagnostic-only scope differs')
 require(p['ops']==list(A.ORDER),'Op order differs')
 require(sorted(p['captures']['members'])==sorted(f'{a}/{n}'for a in['native','wasm0']for n in['norm00_output','norm01_input']),'Capture member set differs')
 return p

def source():
 s=os.environ.get('GITHUB_SHA','');require(re.fullmatch('[0-9a-f]{40}',s)and git('rev-parse','HEAD')==s,'Exact source required')
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA only')
 p=validate(read(PROFILE))
 for path,d in p['owned_sha256'].items():require(sha(ROOT/path)==d,'Owned bytes differ: '+path)
 require(not git('status','--porcelain','--',*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty source rejected')
 require(git('rev-parse','HEAD',cwd=ROOT/'ggml')==FC.GGML and not git('status','--porcelain','--untracked-files=no',cwd=ROOT/'ggml'),'Clean actual a06 required')
 meta=FC.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 require(meta['head_sha']==s and meta['event']=='workflow_dispatch'and meta['status']=='in_progress'and meta['path']==WORKFLOW,'Service/source mismatch')
 return p

def probe(out,name,kind,args,timeout=1200):
 argv=[out/'native'/'layer0-probe',*args]if kind=='native'else['node',out/'wasm'/'layer0-probe.js',*args]
 stdout=out/'review'/(name+'.json');t=time.monotonic()
 with stdout.open('wb')as f,(out/'review'/(name+'.stderr.log')).open('wb')as e:r=subprocess.run([str(a)for a in argv],stdout=f,stderr=e,timeout=timeout)
 write(out/'review'/(name+'-command.json'),dict(argv=[str(a)for a in argv],exit=r.returncode,seconds=time.monotonic()-t,stdout_sha256=sha(stdout)))
 require(r.returncode==0,'Probe rejected: '+name);return read(stdout)

def preflight(out):
 source();out.mkdir(exist_ok=False)
 for d in ['review','inputs','runs']:(out/d).mkdir()

def build(out):
 p=source()
 command(out,'native-configure',['cmake','-S',SRC,'-B',out/'native','-DCMAKE_BUILD_TYPE=Release'])
 command(out,'wasm-configure',['emcmake','cmake','-S',SRC,'-B',out/'wasm','-DCMAKE_BUILD_TYPE=Release','-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm','-DCMAKE_C_FLAGS=-pthread','-DCMAKE_CXX_FLAGS=-pthread'])
 for kind in ['native','wasm']:
  data=(out/kind/'compile_commands.json').read_bytes();(out/'review'/(kind+'-compile.json')).write_bytes(data)
  fixture=FIX/f'frozen-40852835039-full-{kind}-compile.json';require(sha(fixture)==p['production_inventories'][kind],'Production inventory fixture differs')
  try:res=A.admit(kind,json.loads(data),read(fixture),'tests/wasm-layer0-attention/probe.cpp')
  except ValueError as e:write(out/'review'/(kind+'-admission.json'),dict(admitted=False,error=str(e)));raise
  write(out/'review'/(kind+'-admission.json'),dict(res,admitted=True))
 command(out,'native-build',['cmake','--build',out/'native','--target','layer0-probe','--parallel','2'])
 command(out,'wasm-build',['cmake','--build',out/'wasm','--target','layer0-probe','--parallel','2'])
 for kind in ['native','wasm']:require(sha(out/kind/'compile_commands.json')==sha(out/'review'/(kind+'-compile.json')),'Inventory changed during build')

def load_run(folder):return {op:(folder/(op+'.f32')).read_bytes()for op in A.ORDER}

def selftest(out):
 source();x=out/'inputs/authored_x.f32'
 x.write_bytes(struct.pack('<%df'%(384*12),*[((i*7919)%2001-1000)/700.0 for i in range(384*12)]))
 meta,runs={},{}
 for side in ['native','wasm']:
  for r in ['chain1','chain2']:
   d=out/'runs'/('selftest-'+side+'-'+r);d.mkdir()
   meta[(side,r)]=probe(out,f'selftest-{side}-{r}',side,['chain','authored',d,x]);runs[(side,r)]=load_run(d)
  d=out/'runs'/('selftest-'+side+'-ops_own');d.mkdir()
  meta[(side,'ops_own')]=probe(out,f'selftest-{side}-ops_own',side,['ops','authored',d,out/'runs'/('selftest-'+side+'-chain1')]);runs[(side,'ops_own')]=load_run(d)
 result=A.selftest(meta,runs);write(out/'review/selftest.json',dict(result,model_access=False))
 require(result['passed'],'Weight-free controls must pass before model access')

def prepare(out):
 p=source();require(read(out/'review/selftest.json')['passed'],'Selftest before model access')
 z=FC.fetch(out,'captures',p['captures']['artifact'],8192)
 for name,pin in p['captures']['members'].items():
  data=z.read(pin['member']);require(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],'Capture member differs: '+name)
  (out/'inputs'/(name.replace('/','.')+'.f32')).write_bytes(data)
 z=FC.fetch(out,'model',p['model']['artifact'],1024)
 found=[n for n in z.namelist()if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')];require(len(found)==1,'Exact frozen model member')
 data=z.read(found[0]);require(hashlib.sha256(data).hexdigest()==MODEL_SHA,'Exact frozen model required');(out/'inputs/model.gguf').write_bytes(data)
 write(out/'review/prepared.json',dict(captures=len(p['captures']['members']),model_sha256=MODEL_SHA,original_forward=False))

def replay(out):
 source();require(read(out/'review/prepared.json')['model_sha256']==MODEL_SHA,'Prepared inputs required')
 model,x=out/'inputs/model.gguf',out/'inputs/native.norm00_output.f32'
 meta,runs={},{}
 for side in ['native','wasm']:
  d=out/'runs'/(side+'-chain');d.mkdir();meta[(side,'chain')]=probe(out,side+'-chain',side,['chain',model,d,x]);runs[(side,'chain')]=load_run(d)
 for side,other in [('native','wasm'),('wasm','native')]:
  for r,src in [('ops_own',side),('ops_cross',other)]:
   d=out/'runs'/(side+'-'+r);d.mkdir();meta[(side,r)]=probe(out,side+'-'+r,side,['ops',model,d,out/'runs'/(src+'-chain')]);runs[(side,r)]=load_run(d)
 captured={(a,n):(out/'inputs'/(a+'.'+n+'.f32')).read_bytes()for a in ['native','wasm0']for n in ['norm00_output','norm01_input']}
 result=A.replay(meta,runs,captured)
 budget=all(r['same_input_wasm_vs_native']['passed']and r['propagated_chain']['passed']for r in result['ops'])
 report=dict(schema='crispembed-wasm-layer0-attention-result-v1',source=os.environ['GITHUB_SHA'],completed=True,
             meta={f'{k[0]}/{k[1]}':v for k,v in meta.items()},**result,budget_passed=budget,
             passed=result['controls_passed']and budget,candidate_enabled=False,qualification=False,original_forward=False)
 write(out/'review/layer0-attention.json',report)
 require(result['controls_passed'],'Own-reproduction/isolation controls must pass')
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
   files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for d in ['review','runs','inputs']if (out/d).exists()
          for p in (out/d).rglob('*')if p.is_file()and p.name!='file-manifest.json'and p.name!='model.gguf'}
   write(out/'review/file-manifest.json',files)
if __name__=='__main__':raise SystemExit(main())
