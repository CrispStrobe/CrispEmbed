"""Exact current engine-only replay; old publications and originals stay immutable."""
import hashlib,json,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='d7cc37792f345e8dd7eda8c34295cf52ec8bc1dc'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
PROFILE=ROOT/'docs/gemma-decision-first-q4-v1.json'
PROTECTED=['src','ggml','cmake','CMakeLists.txt','LICENSE','README.md','VERSION','Cargo.lock','python','crispembed','crispembed-sys','flutter','examples','wasm','tools/quantize.cpp',
 'tools/gguf_file_type_canonicalize.py','tools/gguf_file_type_values.py','tools/ci_process.py','tools/successor_asset_manifest.py',
 'tools/moedern-native-v2/diff_adapter.py','tools/dump_gemma_decision_reference.py','models/convert-gemma-decision-to-gguf.py',
 'tests/test_gemma_decision_diff.cpp','tests/test_gemma_decision_activation.cpp','tests/test_gemma_decision_schema.cpp','tests/gemma_decision_rendering.py','tests/gemma_decision_parity.py',
 'docs/encoder-precision-evidence-20261029.md','tests/results/encoder-precision/evidence-audit-20261029.json','tests/results/gemma-decision']
def require(v,s):
 if not v:raise ValueError(s)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(4*2**20):h.update(b)
 return h.hexdigest()
def identity(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 with tmp.open('w')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def input_pins():
 audit=read(ROOT/'tests/results/encoder-precision/evidence-audit-20261029.json')
 requests={r['name']:r for r in audit['hub_metadata']['requests']}
 def record(f,dataset=False):
  r=dict(path=f['path'],bytes=f['bytes'],git_blob_oid=f['git_blob_oid'])
  if f['lfs']:r['sha256']=f['lfs'].get('sha256',f['lfs'].get('oid'))
  if dataset:r['dataset']=True
  return r
 m=requests['gemma-models'];f=requests['gemma-fixtures']
 return dict(model=dict(repo=m['repository'],revision=m['revision'],file=record(next(r for r in m['files']if r['path']=='embeddinggemma-2-ft-typed-decision-f32.gguf'))),
  fixtures=dict(repo=f['repository'],revision=f['revision'],files=[record(r,True)for r in f['files']]),
  notices=dict(repo=m['repository'],revision=m['revision'],files=[record(r)for r in m['files']if r['path']in ['LICENSE','README.md','UPSTREAM_README.md']]))

def validate(p):
 require(p['schema']=='crispembed-gemma-decision-first-q4-v1'and p['base']==BASE and p['ggml']==GGML,'Exact source base required')
 require(p['recipes']==['broad-mixed','protected-input']and p['probe_ids']==[18,20,25,19,549]and p['stages_per_probe']==29
         and p['stage_min_cosine']==.99999 and p['stage_max_relative_l2']==.003 and p['absolute_stage_budget']is None
         and p['bank_sizes']==[13,11,6]and p['typed_questions']==8 and p['encoder_forwards_per_model']==88
         and p['head_on_original_calls_per_model']==16 and p['public_capi_qualified']is False and p['publication_qualified']is False
         and p['original_forwards']==0,'Frozen experiment/legacy gates/call scope differ')
 require(set(p['protected_git_objects'])==set(PROTECTED),'Complete engine/converter/fixture closure required')
 require(p['inputs']==input_pins(),'Immutable audited original/publication identities required')
 return p

def source(source):
 require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD')==source,'Exact immutable source required')
 p=validate(read(PROFILE))
 for name,obj in p['protected_git_objects'].items():
  require(git('rev-parse',BASE+':'+name)==obj and git('rev-parse','HEAD:'+name)==obj,'Protected source differs: '+name)
  if (ROOT/name).is_file():require(git('hash-object','--',name)==obj,'Protected worktree differs: '+name)
 for name,digest in p['owned_sha256'].items():require(sha(ROOT/name)==digest,'Owned source differs: '+name)
 require(not git('status','--porcelain','--',*PROTECTED,*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty source rejected')
 require(git('-C',str(ROOT/'ggml'),'rev-parse','--show-toplevel')==str(ROOT/'ggml')and git('-C',str(ROOT/'ggml'),'rev-parse','HEAD')==GGML
         and not git('-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no'),'Actual clean a06 required')
 return p
