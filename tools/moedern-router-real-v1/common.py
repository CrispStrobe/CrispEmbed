"""Stdlib-only closed source/protocol and exact native selector extraction."""
import hashlib,json,math,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/moedern-router-real-v1.json'
BASE='ac068b1b7e2e327c3a17e248382cca3888eb68ae'
SOURCE='f3364a57e29db6ecd58fa1fe2cb60a794d1f14b1'
PUBLIC=ROOT/'tests/results/moedern-colbert/first-row-actual-v4-review.json'
OWNED={str(PATH.relative_to(ROOT)),'docs/moedern-router-real-v1.md','tests/test_moedern_router_real_v1.py','tools/ci-heavy/moedern_router_real_v1.py','tests/moedern-router-real-v1/CMakeLists.txt','tests/moedern-router-real-v1/probe.cpp'}|{'tools/moedern-router-real-v1/'+n for n in ['common.py','prepare.py','worker.py','requirements.txt','wheels.json']}
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(2**20):h.update(b)
 return h.hexdigest()
def identity(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 with tmp.open('w')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def fragment(text):
 start='                std::array<int, EXPERTS> order{};'
 end='\n            }\n            emit(dump, n + "ffn.router_probabilities"'
 require(text.count(start)==1 and text.count(end)==1,'Unique actual native router fragment required')
 first=text.index(start);last=text.index(end,first)
 return text[first:last]+'\n'
def validate(c):
 old=read(ROOT/'docs/moedern-first-row-stages-v1.json');review=read(PUBLIC)
 require(c['schema']=='crispembed-moedern-router-real-input-v1'and c['base']==BASE and c['runtime_source']==old['runtime_source']and c['ggml']==old['ggml'],'Exact source/runtime required')
 require(c['capture']['source']==SOURCE and c['capture']['version_label']=='v4'and c['capture']['status']=='ERROR'and c['capture']['listed_outputs']==400
         and c['capture']['script_sha256']==review['submitted_script_sha256']and c['capture']['independent_review_sha256']==sha(PUBLIC),'Exact retained actual-v4 capture required')
 require(c['model']==old['native_model']and c['threads']==[1,2]and c['repeats']==2 and c['gates']==dict(max_absolute=.002,min_row_cosine=.99999,ordered_indices_exact=True,own_and_repeat_bytes_exact=True),'Immutable F32 model/fixed gates required')
 require(c['counts']==dict(full_model_forwards=0,original_full_model_forwards=0,primary_component_batch_max=dict(linear=32,softmax=64,topk=96,renormalization=96),own_phase_batches=48,cross_phase_batches=48,float_ref_max=144,integer_comparisons_max=72,ref_control_calls=3,invalid_cpp_controls_before_math=2),'Exact component-only counts required')
 require(c['shapes']==dict(original_input=[307,128],native_capture_input=[1,307,128],native_component_flat_view=[307,128],logits=[307,8],indices=[307,2]),'Captured/native flat-byte shapes differ')
 require(c['native_integer_admission']==dict(pre_reviewed_sha256=False,first_observed_hosted_sha256=True,source='frozen v4 actual integer comparison and actual original I64 captures',layer0_all_rows_equal=True,layer1_only_changed_row=58,original_row58=[2,4],native_row58=[2,6],synthesized_gold=False),'Honest semantic native I64 admission required')
 require(c['torch']['publisher_router']==old['publisher_router_source'],'Pinned publisher routing source differs')
 require(c['torch']['version']=='2.8.0+cpu'and c['torch']['numpy']=='2.2.6'and c['torch']['interop_threads']==1 and c['torch']['lock_source']=='c95f45e3cff433f56d2a474bf59cdcd4405e6d60','Exact CPU package profile required')
 expected=set(c['capture']['metadata'])|{c[k]['path']for k in ['ids','mask']}
 for i,layer in enumerate(c['layers']):
  require(layer['index']==i and layer['weight_name']==f'blk.{i}.ffn_gate_inp.weight'and layer['weight_stored_shape']==[128,8]and layer['weight_torch_shape']==[8,128],'Layer/router weight axes differ')
  require(layer['weight_sha256']==['6047c7eb4d02f5e3345c21e9ec3557269fa90006e5c19dedb48c33165b58ecf7','0109c1f52a5ac57c2facff41afc7081a9937ea70953e7fac5b4885271a9f4299'][i],'Loaded original router weight differs')
  require(set(layer['original'])=={'input','logits','probs','values','indices','denominator','weights'}and set(layer['native'])=={'input','logits','probs','indices','weights'},'Complete original/native observed operands required')
  require(layer['original']['input']['shape']==[307,128]and layer['native']['input']['shape']==[1,307,128]and layer['native']['indices']['sha256']is None,'No original flatten/ invented native I64 SHA')
  for side in ['original','native']:
   for kind,pin in layer[side].items():
    shape=([307,128]if side=='original'else[1,307,128])if kind=='input'else[307,8]if kind in ['logits','probs']else[307,1]if kind=='denominator'else[307,2]
    dtype='i64'if kind=='indices'else'f32'
    require(pin['shape']==shape and pin['dtype']==dtype and pin['bytes']==math.prod(shape)*(8 if dtype=='i64'else 4),'Exact captured operand schema required')
  require(layer['native']['indices']['path']==f'retained/output/raw/observed/doc-24726600/{20 if i==0 else 43}.i64','Actual native integer member differs')
  expected|={r['path']for r in layer['original'].values()}|{r['path']for k,r in layer['native'].items()if k!='indices'}
 require(all('moedern-first-row-v4/'+name in review['all_output_names']for name in expected|{r['native']['indices']['path']for r in c['layers']}),'Selected member not in actual400-output inventory')
 require(len(c['layers'])==2 and set(c['capture']['pre_pinned_members'])==expected,'Exact selected-file authentication scope differs')
 require(all(c[k]is False for k in ['production_change','publication_qualified','new_gold','original_full_corpus_recaptured']),'No production/reference/qualification change')
 return c
def declaration():return validate(read(PATH))
def verify(source):
 c=declaration();require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD').decode().strip()==source and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Exact reviewed source required')
 old=read(ROOT/'docs/moedern-first-row-stages-v1.json')
 protected=set(old['protected_sha256'])|set(old['owned_sha256'])|{'docs/moedern-first-row-stages-v1.json',str(PUBLIC.relative_to(ROOT)),'docs/moedern-first-row-actual-v4-result.md','tools/successor_wasm_trace_contract.py','tools/moedern-native-v2/diff_adapter.py','tests/moedern-native-v2/diff.cpp'}
 require(set(c['protected_sha256'])==protected and set(c['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Complete historical/new closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED|{'PLAN.md'},'Undeclared source changes')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in c[group].items():
   raw=(ROOT/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin and git('show','HEAD:'+name)==raw,'Source changed: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw,'Frozen source changed: '+name)
 for name,obj in c['protected_git_objects'].items():require(git('rev-parse','HEAD:'+name).decode().strip()==git('rev-parse',BASE+':'+name).decode().strip()==obj,'Production object changed: '+name)
 require(not git('status','--porcelain','--untracked-files=no'),'Dirty tracked source')
 for name in ['requirements.txt','wheels.json']:require((ROOT/'tools/moedern-router-real-v1'/name).read_bytes()==git('show',c['torch']['lock_source']+':tools/moedern-router-topk-v1/'+name),'Frozen wheel lock changed')
 return c
def extract(directory,c):
 text=(ROOT/c['native_fragment']['path']).read_text();body=fragment(text)
 require(hashlib.sha256(text.encode()).hexdigest()==c['native_fragment']['source_sha256']and hashlib.sha256(body.encode()).hexdigest()==c['native_fragment']['sha256'],'Actual native fragment/source changed')
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);p=directory/'native_router.inc';p.write_text(body)
 write(directory/'provenance.json',dict(source=c['native_fragment'],generated=identity(p),arithmetic_edited=False))
 return p
