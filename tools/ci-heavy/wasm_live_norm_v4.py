#!/usr/bin/env python3
"""Closed v4 caller for the one-sentence live norm diagnostic.

v2 component admission and the v3 observer are reused unchanged. v4 changes
only the Ref adapter used by actual_ref: tools/wasm-live-norm-v4/diff_adapter_v4.py
(frozen adapter code, explicit exact-shape allowlist incl. the live [12,1536]
layer3 fc1_raw/fc1/gelu rows). The comparison body below is the frozen
wasm_native_order_norm_v1.actual_ref with only the adapter module swapped.
Budgets, full-row metrics and v1's strict numerical exit are unchanged.
"""
import importlib.util,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('wasm_live_norm_v3_frozen',ROOT/'tools/ci-heavy/wasm_live_norm_v3.py')
V3=importlib.util.module_from_spec(spec);spec.loader.exec_module(V3)
spec=importlib.util.spec_from_file_location('diff_adapter_v4',ROOT/'tools/wasm-live-norm-v4/diff_adapter_v4.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
V2=V3.V2;V1=V3.V1;K=V1.K;OLD=V1.old
PROFILE=ROOT/'docs/wasm-live-norm-operands-v4.json'
WORKFLOW='.github/workflows/wasm-live-norm-operands-v4.yml'

def validate(p):
 K.require(p['schema']=='crispembed-wasm-live-norm-operands-v4'and p['v3_profile_sha256']==K.sha(V3.PROFILE)
           and p['v2_profile_sha256']==K.sha(V2.PROFILE)and p['v1_profile_sha256']==K.sha(K.PROFILE),'v4 base profiles differ')
 K.require(p['workflow']==WORKFLOW and p['scope_unchanged_from_v1']is True and p['runtime_change']is False and p['qualification']is False
           and p['component_admission']=='v2'and p['observer']=='v3','v4 scope differs')
 K.require(p['ref_adapter']==dict(path='tools/wasm-live-norm-v4/diff_adapter_v4.py',frozen=str(A.FROZEN.relative_to(ROOT)),frozen_sha256=A.FROZEN_SHA256,
           allowed_shapes=sorted([list(s)for s in A.ALLOWED_SHAPES]),budgets=dict(max_absolute=.002,minimum_row_cosine=.99999)),'v4 Ref adapter contract differs')
 K.require(A.ABSOLUTE==.002 and A.COSINE==.99999,'Fixed budgets differ')
 K.require([r['run']for r in p['retained_negatives']]==[40867038915,41416683981,41418312256,41421952964,41427588368],'Retained negatives differ')
 return p

def source():
 cfg=K.source(os.environ.get('GITHUB_SHA',''))
 for mod in [V2,V3]:
  prof=K.read(mod.PROFILE);mod.validate(prof)
  for path,digest in prof['owned_sha256'].items():K.require(K.sha(ROOT/path)==digest,'Owned earlier bytes differ: '+path)
 p=validate(K.read(PROFILE))
 for path,digest in p['owned_sha256'].items():K.require(K.sha(ROOT/path)==digest,'Owned v4 bytes differ: '+path)
 K.require(not K.git('status','--porcelain','--',*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty v4 source rejected')
 K.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA only')
 meta=V1.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 K.require(meta['head_sha']==os.environ['GITHUB_SHA']and meta['event']=='workflow_dispatch'and meta['status']=='in_progress'and meta['path']==WORKFLOW,'Service/source mismatch')
 return cfg

def frozen_actual_ref(out,label,values):
 """wasm_native_order_norm_v1.actual_ref, with diff_adapter -> diff_adapter_v4 only."""
 folder=out/'raw/adapters';folder.mkdir(exist_ok=True);pairs=[];measurements={}
 for name,(a,b,shape)in values.items():
  measurements[name]=OLD.independent(a,b,shape);pair=dict(name=name)
  for side,v in [('original',a),('native',b)]:
   path=folder/(label+'-'+name+'-'+side+'.f32');path.write_bytes(OLD.f32_bytes(v))
   pair[side]=dict(path=str(path),shape=shape,sha256=K.sha(path))
  pairs.append(pair)
 left,right=folder/(label+'-original.gguf'),folder/(label+'-native.gguf')
 adapters=A.write_pair(left,right,pairs);target=out/'review'/(label+'-ref.json')
 code=OLD.command(out,label+'-ref',[out/'component-native/norm-candidate-ref',left,right,target],timeout=180,expected=(0,1))
 cpp=A.collect(code,target,measurements)
 return dict(cpp=cpp,independent=measurements,adapters=adapters,passed=all(v['passed']for v in measurements.values()))

# Module globals resolve at call time: route every earlier phase through v4.
V1.source=V2.source=V3.source=source
V1.frozen_actual_ref=frozen_actual_ref
V1.__file__=str(Path(__file__).resolve())
PHASES=V3.PHASES

def main():
 import argparse
 p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=[*PHASES,'native-worker'])
 p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,2]);a=p.parse_args();out=a.output.resolve()
 phases=dict(PHASES,**{'native-worker':lambda o:V1.native_worker(o,a.threads)})
 try:return phases[a.phase](out)
 except Exception as e:
  out.joinpath('review').mkdir(parents=True,exist_ok=True)
  K.write(out/'review'/('rejection-'+a.phase+'.json'),dict(source=os.environ.get('GITHUB_SHA'),protocol='v4',phase=a.phase,completed=False,error=str(e),qualification=False));raise
 finally:
  if out.exists():
   files={}
   for folder in ['raw','browser','review']:
    for path in (out/folder).rglob('*'):
     if path.is_file()and path.name!='file-manifest.json':files[str(path.relative_to(out))]=K.identity(path)
   K.write(out/'review/file-manifest.json',files)
if __name__=='__main__':raise SystemExit(main())
