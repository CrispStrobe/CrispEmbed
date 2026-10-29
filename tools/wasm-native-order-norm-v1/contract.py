"""Explicit candidate delta and closed full-model A/B, separate from old guards."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'docs/wasm-native-order-norm-candidate-v1.json'
BASE='c3d6c11715e744547f10d4011e11ef5b5e473753'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
GATE='CRISPEMBED_WASM_NATIVE_ORDER_NORM'
DELTA={'CMakeLists.txt','src/crispembed.cpp','src/core/wasm_native_order_norm.cpp','src/core/wasm_native_order_norm.h'}

def require(v,s):
 if not v:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 with tmp.open('w')as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def tree(rev):
 return {line.split('\t')[1]:line.split('\t')[0]for line in git('ls-tree','-r',rev,'--','src').splitlines()}
def validate(p):
 require(p['schema']=='crispembed-wasm-native-order-norm-candidate-v1'and p['base']==BASE and p['ggml']==GGML
         and p['gate']==GATE and p['default']is False and p['qualification']is False,'Candidate scope differs')
 require(set(p['runtime_delta'])==DELTA and p['norm_gates']==['unset','empty','0','1']
         and p['held_q8_rounding']=='1'and p['additional_legacy_q8_rounding']=='0'
         and p['max_absolute']==.002 and p['minimum_row_cosine']==.99999
         and p['frontend_simd']is False and p['ggml_backend_simd']is True and p['script_no_simd']is True and p['pthreads']is True,'Frozen A/B semantics differ')
 required={'ggml','cmake','VERSION','Cargo.lock','examples','python','flutter','crispembed','crispembed-sys','wasm','tools/quantize.cpp',
 'build-wasm.sh','build-embed-wasm.sh','scripts/check-cpu-baseline.py','scripts/check-bundled-deps.py','tools/successor_asset_manifest.py',
 '.gitattributes','.github/workflows/release.yml','.github/workflows/release-wasm.yml','README.md','LICENSE',
 'tools/ci-heavy/successor_assets.py','tools/successor_wasm_trace_contract.py','tools/ci-heavy/wasm_q8_rounding_ab_v1.py',
 'tools/wasm_q8_rounding_ab_v1.py','tools/ci_process.py','tools/moedern-native-v2/diff_adapter.py','tests/moedern-native-v2/diff.cpp',
 'docs/wasm-q8-rounding-ab-v1.json','docs/wasm-q8-rounding-candidate.json','tests/results/packaging/wasm-rounding-ab-v1.json'}
 require(set(p['protected_git_objects'])==required,'Complete build/binding/reference/helper closure required')
 return p

def source(source):
 require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD')==source,'Exact reviewed source required')
 p=validate(read(PROFILE));before,after=tree(BASE),tree('HEAD')
 for path in set(before)|set(after):
  if path not in DELTA:require(before.get(path)==after.get(path),'Unrelated SRC differs: '+path)
 for path,pin in p['runtime_delta'].items():
  exists=bool(git('ls-tree',BASE,'--',path));old=git('rev-parse',BASE+':'+path)if exists else None
  require(old==pin['base_git_object']and sha(ROOT/path)==pin['sha256'],'Declared runtime delta differs: '+path)
 for path,obj in p['protected_git_objects'].items():
  require(git('rev-parse',BASE+':'+path)==obj and git('rev-parse','HEAD:'+path)==obj,'Protected object differs: '+path)
  if (ROOT/path).is_file():require(git('hash-object','--',path)==obj,'Protected worktree differs: '+path)
 for path,digest in p['owned_sha256'].items():require(sha(ROOT/path)==digest,'Owned source differs: '+path)
 paths=['src',*p['protected_git_objects'],*p['owned_sha256'],*DELTA,str(PROFILE.relative_to(ROOT))]
 require(not git('status','--porcelain','--',*paths),'Dirty protected/candidate source')
 require(git('-C',str(ROOT/'ggml'),'rev-parse','--show-toplevel')==str(ROOT/'ggml')
         and git('-C',str(ROOT/'ggml'),'rev-parse','HEAD')==GGML
         and not git('-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no'),'Actual clean a06 checkout required')
 return p
