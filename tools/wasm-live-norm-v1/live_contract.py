"""One frozen sentence; exact immutable candidate/build/reference closure."""
import hashlib,json,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='74ec9373bd4a44380167b0888ed70f0fb6f6ccc7'
CANDIDATE='64eef68493deb240e81287e249f55def08d5753c'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
PROFILE=ROOT/'docs/wasm-live-norm-operands-v1.json'
PROTECTED=['.gitattributes', '.github/workflows/release-wasm.yml', '.github/workflows/release.yml', '.github/workflows/wasm-native-order-norm-candidate-v1.yml', 'CMakeLists.txt', 'Cargo.lock', 'LICENSE', 'README.md', 'VERSION', 'build-embed-wasm.sh', 'build-wasm.sh', 'cmake', 'crispembed', 'crispembed-sys', 'docs/wasm-native-order-norm-candidate-v1.json', 'docs/wasm-native-order-norm-candidate-v1.md', 'docs/wasm-q8-rounding-ab-v1.json', 'docs/wasm-q8-rounding-candidate.json', 'examples', 'flutter', 'ggml', 'python', 'scripts/check-bundled-deps.py', 'scripts/check-cpu-baseline.py', 'src', 'tests/moedern-native-v2/diff.cpp', 'tests/results/packaging/wasm-rounding-ab-v1.json', 'tests/test_wasm_native_order_norm_v1.py', 'tests/wasm-native-order-norm', 'tests/wasm-native-order-norm/CMakeLists.txt', 'tests/wasm-native-order-norm/component.cjs', 'tests/wasm-native-order-norm/controls.cpp', 'tests/wasm-native-order-norm/panel.cjs', 'tests/wasm-native-order-norm/vector_bytes.cjs', 'tools/ci-heavy/successor_assets.py', 'tools/ci-heavy/wasm_native_order_norm_v1.py', 'tools/ci-heavy/wasm_q8_rounding_ab_v1.py', 'tools/ci_process.py', 'tools/moedern-native-v2/diff_adapter.py', 'tools/quantize.cpp', 'tools/successor_asset_manifest.py', 'tools/successor_wasm_trace_contract.py', 'tools/wasm-native-order-norm-v1', 'tools/wasm-native-order-norm-v1/contract.py', 'tools/wasm_layer3_attribution_v1.py', 'tools/wasm_q8_rounding_ab_v1.py', 'wasm']
def require(v,s):
 if not v:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 with tmp.open('w')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def validate(p):
 require(p['schema']=='crispembed-wasm-live-norm-operands-v1'and p['base']==BASE and p['candidate']==CANDIDATE and p['ggml']==GGML,'Source profile differs')
 require(p['cases']==['original']and p['tokens']==12 and p['norm_nodes']==13 and p['layer3_nodes']==23
         and p['held_q8_rounding']=='1'and p['norm_gates']==['0','1']and p['threads']==[1,2]
         and p['max_absolute']==.002 and p['minimum_row_cosine']==.99999
         and p['runtime_change']is False and p['qualification']is False,'One-sentence diagnostic scope differs')
 names={f'browser/arrays/{gate}-original-observed-{repeat}-stages.gguf'for gate in ['0','1']for repeat in [0,1]}
 require(set(p['retained_wasm_stage_members'])==names and all(pin['bytes']==148000 and re.fullmatch('[0-9a-f]{64}',pin['sha256'])for pin in p['retained_wasm_stage_members'].values()),'Exact four old WASM stage files required')
 require(p['numerical_result_scope']==dict(local_attribution_separate=True,overall_requires_natural_strict=True,all_comparisons_before_numeric_exit=True,numeric_exit=1),'Overall numerical rejection scope differs')
 require(set(p['protected_git_objects'])==set(p['protected_paths'])==set(PROTECTED),'Explicit protected closure differs')
 return p

def source(source):
 require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD')==source,'Exact source required')
 p=validate(read(PROFILE))
 for path,obj in p['protected_git_objects'].items():
  require(git('rev-parse',BASE+':'+path)==obj and git('rev-parse','HEAD:'+path)==obj,'Protected object differs: '+path)
  if (ROOT/path).is_file():require(git('hash-object','--',path)==obj,'Protected worktree differs: '+path)
 for path,digest in p['owned_sha256'].items():require(sha(ROOT/path)==digest,'Owned bytes differ: '+path)
 require(not git('status','--porcelain','--',*p['protected_paths'],*p['owned_sha256'],str(PROFILE.relative_to(ROOT))),'Dirty source rejected')
 require(git('rev-parse',CANDIDATE+':src')==git('rev-parse','HEAD:src')and git('rev-parse',CANDIDATE+':CMakeLists.txt')==git('rev-parse','HEAD:CMakeLists.txt'),'Candidate math/build changed')
 require(git('-C',str(ROOT/'ggml'),'rev-parse','--show-toplevel')==str(ROOT/'ggml')
         and git('-C',str(ROOT/'ggml'),'rev-parse','HEAD')==GGML
         and not git('-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no'),'Clean actual a06 required')
 return p
