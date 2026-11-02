#!/usr/bin/env python3
"""Closed v2 caller for the one-sentence live norm diagnostic.

Only the component compile-inventory admission changes. Every other phase is the
unchanged ff3b8a01 v1 implementation, called with this caller's source/service
guard. The v1 files, the failed run 40867038915 and all old raw negatives stay
immutable; numerical negatives keep v1's strict non-zero exit.
"""
import argparse,hashlib,importlib.util,io,json,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/wasm-live-norm-v2'))
import inventory as I
spec=importlib.util.spec_from_file_location('wasm_live_norm_v1_frozen',ROOT/'tools/ci-heavy/wasm_live_norm_v1.py')
V1=importlib.util.module_from_spec(spec);spec.loader.exec_module(V1)
K=V1.K
PROFILE=ROOT/'docs/wasm-live-norm-operands-v2.json'
WORKFLOW='.github/workflows/wasm-live-norm-operands-v2.yml'

def validate(p):
 K.require(p['schema']=='crispembed-wasm-live-norm-operands-v2'and p['v1_source']=='ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265','v2 schema/base differs')
 K.require(p['v1_profile_sha256']==K.sha(K.PROFILE),'v1 protocol bytes changed')
 K.require(p['workflow']==WORKFLOW and p['scope_unchanged_from_v1']is True and p['runtime_change']is False and p['qualification']is False,'v2 scope differs')
 K.require(p['declared_component_entries']==[list(x)for x in I.DECLARED]and p['forbidden_flags']==list(I.FORBIDDEN),'Declared inventory/flag rules differ')
 K.require(p['retained_negative']['run']==40867038915 and p['retained_negative']['source']=='ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265'
           and p['retained_negative']['phase']=='component-build','Retained v1 component rejection differs')
 f=p['frozen_component_inventory']
 K.require(f['members']==['review/component-native-compile.json','review/component-wasm-compile.json']and f['artifact']=='candidate-review','Frozen component inventory source differs')
 return p

def source():
 cfg=V1.K.source(os.environ.get('GITHUB_SHA',''))
 p=validate(K.read(PROFILE))
 for path,digest in p['owned_sha256'].items():K.require(K.sha(ROOT/path)==digest,'Owned v2 bytes differ: '+path)
 K.require(not K.git('status','--porcelain','--',*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty v2 source rejected')
 K.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA only')
 meta=V1.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 K.require(meta['head_sha']==os.environ['GITHUB_SHA']and meta['event']=='workflow_dispatch'and meta['status']=='in_progress'and meta['path']==WORKFLOW,'Service/source mismatch')
 return cfg

# Every v1 phase resolves source() and __file__ through module globals at call time.
V1.source=source
V1.__file__=str(Path(__file__).resolve())

def frozen_inventories(out):
 """Extract the run-40852835039 component inventories from the already
 SHA-authenticated previous review archive written by v1 prepare."""
 cfg=K.read(K.PROFILE)['previous']['review'];path=out/'candidate-review.zip'
 K.require(K.sha(path)==cfg['sha256'],'Previous review archive differs')
 members=V1.zip_files(path)
 for kind in ['native','wasm']:
  data=V1.one(members,'component-'+kind+'-compile.json')
  K.require(hashlib.sha256(data).hexdigest()==K.read(PROFILE)['frozen_component_inventory']['sha256'][kind],'Frozen component inventory member differs: '+kind)
  json.loads(data);(out/'review'/('frozen-component-'+kind+'-compile.json')).write_bytes(data)

def prepare(out):
 V1.prepare(out);frozen_inventories(out)

def frozen_root(commands):
 roots={str(Path(r['file']).parents[2])for r in commands if r['file'].endswith('/tests/wasm-native-order-norm/controls.cpp')}
 K.require(len(roots)==1,'Frozen source root');return roots.pop()

def admit(out,kind,build):
 """Save the FULL actual inventory first, then check every declared target."""
 data=(build/'compile_commands.json').read_bytes()
 (out/'review'/('component-'+kind+'-compile.json')).write_bytes(data)
 commands=json.loads(data);frozen=K.read(out/'review'/('frozen-component-'+kind+'-compile.json'))
 try:
  report=I.check(kind,commands,frozen,ROOT,build,frozen_root(frozen))
 except ValueError as e:
  K.write(out/'review'/('component-'+kind+'-inventory.json'),dict(kind=kind,admitted=False,error=str(e),total_entries=len(commands)));raise
 K.write(out/'review'/('component-'+kind+'-inventory.json'),dict(report,admitted=True));return report

def component_build(out):
 source();native=out/'component-native';wasm=out/'component-wasm';command=V1.command
 command(out,'component-native-configure',['cmake','-S',ROOT/'tests/wasm-live-norm','-B',native,'-DCMAKE_BUILD_TYPE=Release'])
 command(out,'component-wasm-configure',['emcmake','cmake','-S',ROOT/'tests/wasm-live-norm','-B',wasm,
         '-DCMAKE_BUILD_TYPE=Release','-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm','-DCMAKE_C_FLAGS=-pthread','-DCMAKE_CXX_FLAGS=-pthread'])
 # Admission precedes any build, any component execution and any model access.
 for kind,build in [('native',native),('wasm',wasm)]:admit(out,kind,build)
 command(out,'component-native-build',['cmake','--build',native,'--target','norm-candidate-controls','norm-candidate-ref','live-norm-replay','--parallel','2'])
 command(out,'component-authored',[native/'norm-candidate-controls'])
 command(out,'component-wasm-build',['cmake','--build',wasm,'--target','norm-candidate-controls','live-norm-replay','--parallel','2'])
 for kind,build in [('native',native),('wasm',wasm)]:
  K.require(K.sha(build/'compile_commands.json')==K.sha(out/'review'/('component-'+kind+'-compile.json')),'Inventory changed during build')
 destination=out/'browser/component';destination.mkdir()
 for name in ['norm-candidate-controls','live-norm-replay']:
  for suffix in ['js','wasm']:V1.shutil.copyfile(wasm/(name+'.'+suffix),destination/(name+'.'+suffix))
 command(out,'live-observer-controls',[native/'live-norm-replay'])
 for row in K.read(out/'browser/component-inputs.json'):
  for threads in [1,2]:command(out,'component-'+row['name']+'-'+str(threads),[native/'norm-candidate-controls',
      out/'browser/component-inputs'/(row['name']+'.f32'),out/'raw'/(row['name']+'.native'+str(threads)+'.f32'),str(threads)])

PHASES=dict(preflight=V1.preflight,prepare=prepare,**{'component-build':component_build,'component-check':V1.component_check},model=V1.model,
 **{'build-full':V1.build_full},native=V1.native,**{'prepare-replay':V1.prepare_replay},replay=V1.replay,**{'ref-controls':V1.ref_controls})

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=[*PHASES,'native-worker'])
 p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,2]);a=p.parse_args();out=a.output.resolve()
 phases=dict(PHASES,**{'native-worker':lambda o:V1.native_worker(o,a.threads)})
 try:return phases[a.phase](out)
 except Exception as e:
  out.joinpath('review').mkdir(parents=True,exist_ok=True)
  K.write(out/'review'/('rejection-'+a.phase+'.json'),dict(source=os.environ.get('GITHUB_SHA'),protocol='v2',phase=a.phase,completed=False,error=str(e),qualification=False));raise
 finally:
  if out.exists():
   files={}
   for folder in ['raw','browser','review']:
    for path in (out/folder).rglob('*'):
     if path.is_file()and path.name!='file-manifest.json':files[str(path.relative_to(out))]=K.identity(path)
   K.write(out/'review/file-manifest.json',files)
if __name__=='__main__':raise SystemExit(main())
