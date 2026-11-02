#!/usr/bin/env python3
"""Closed v3 caller for the one-sentence live norm diagnostic.

v2 (component inventory admission) is reused unchanged. v3 swaps only the
disposable observer header (tests/wasm-live-norm-v3/observer.h) used by
build-full. All other phases are the frozen v1 functions under v3's
source/service guard. v1, v2 and every receipt stay immutable; numerical
negatives keep v1's strict non-zero exit.
"""
import importlib.util,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('wasm_live_norm_v2_frozen',ROOT/'tools/ci-heavy/wasm_live_norm_v2.py')
V2=importlib.util.module_from_spec(spec);spec.loader.exec_module(V2)
spec=importlib.util.spec_from_file_location('wasm_live_norm_v3_overlay',ROOT/'tools/wasm-live-norm-v3/overlay.py')
OVERLAY=importlib.util.module_from_spec(spec);spec.loader.exec_module(OVERLAY)
V1=V2.V1;K=V1.K
PROFILE=ROOT/'docs/wasm-live-norm-operands-v3.json'
WORKFLOW='.github/workflows/wasm-live-norm-operands-v3.yml'

def validate(p):
 K.require(p['schema']=='crispembed-wasm-live-norm-operands-v3'and p['v2_profile_sha256']==K.sha(V2.PROFILE)and p['v1_profile_sha256']==K.sha(K.PROFILE),'v3 base profiles differ')
 K.require(p['workflow']==WORKFLOW and p['scope_unchanged_from_v1']is True and p['runtime_change']is False and p['qualification']is False
           and p['component_admission']=='v2'and p['observer']=='tests/wasm-live-norm-v3/observer.h','v3 scope differs')
 K.require([r['run']for r in p['retained_negatives']][:3]==[40867038915,41416683981,41418312256],'Retained negatives differ')
 return p

def source():
 cfg=K.source(os.environ.get('GITHUB_SHA',''))
 V2.validate(K.read(V2.PROFILE))
 for path,digest in K.read(V2.PROFILE)['owned_sha256'].items():K.require(K.sha(ROOT/path)==digest,'Owned v2 bytes differ: '+path)
 p=validate(K.read(PROFILE))
 for path,digest in p['owned_sha256'].items():K.require(K.sha(ROOT/path)==digest,'Owned v3 bytes differ: '+path)
 K.require(not K.git('status','--porcelain','--',*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty v3 source rejected')
 K.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA only')
 meta=V1.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 K.require(meta['head_sha']==os.environ['GITHUB_SHA']and meta['event']=='workflow_dispatch'and meta['status']=='in_progress'and meta['path']==WORKFLOW,'Service/source mismatch')
 return cfg

# Module globals are resolved at call time: route v1 and v2 phases through v3.
V1.source=source;V2.source=source
V1.apply=OVERLAY.apply
V1.__file__=str(Path(__file__).resolve())
def observer_controls(out):
 """Weight-free: build and run the v3 observer metadata controls, before model access."""
 source();K.require(K.read(out/'browser/component-proof.json')['passed'],'Component recoveries required first')
 build=out/'observer-controls'
 V1.command(out,'observer-controls-configure',['cmake','-S',ROOT/'tests/wasm-live-norm-v3','-B',build,'-DCMAKE_BUILD_TYPE=Release'])
 V1.command(out,'observer-controls-build',['cmake','--build',build,'--target','live-norm-observer-controls','--parallel','2'])
 import tempfile
 with tempfile.TemporaryDirectory()as tmp:V1.command(out,'observer-controls',[build/'live-norm-observer-controls',tmp])
 log=(out/'review/observer-controls.log').read_text()
 passed=log.count('PASS ');K.require(passed==11 and 'FAIL 'not in log,'All11 observer controls required')
 K.write(out/'review/observer-controls.json',dict(passed=True,controls=passed,reproduces='41418312256 reservation-graph rejection',qualification=False))

def model(out):
 source();K.require(K.read(out/'review/observer-controls.json')['passed'],'Observer controls BEFORE model access')
 return V1.model(out)

PHASES=dict(V2.PHASES,model=model,**{'observer-controls':observer_controls})

def main():
 import argparse
 p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=[*PHASES,'native-worker'])
 p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,2]);a=p.parse_args();out=a.output.resolve()
 phases=dict(PHASES,**{'native-worker':lambda o:V1.native_worker(o,a.threads)})
 try:return phases[a.phase](out)
 except Exception as e:
  out.joinpath('review').mkdir(parents=True,exist_ok=True)
  K.write(out/'review'/('rejection-'+a.phase+'.json'),dict(source=os.environ.get('GITHUB_SHA'),protocol='v3',phase=a.phase,completed=False,error=str(e),qualification=False));raise
 finally:
  if out.exists():
   files={}
   for folder in ['raw','browser','review']:
    for path in (out/folder).rglob('*'):
     if path.is_file()and path.name!='file-manifest.json':files[str(path.relative_to(out))]=K.identity(path)
   K.write(out/'review/file-manifest.json',files)
if __name__=='__main__':raise SystemExit(main())
