"""Separately declared native-F32 development calibration, no old guard retargeting."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/josev2b-final-pair-imatrix-v1.json'
BASE='0d56cebf6394ff45c71e482e44deabcb74e1227b'
RUNTIME='6781befb66b5f65afd47c1b921ac9cd1cb8e568a'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
from josev2b_q4k_contract import require,REFERENCE_SHA,FREEZE_SHA
from quantize_josev2b_final_pair_imatrix import SOURCE,COUNTS,plan
from quantize_josev2b_protected import schema

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(4*2**20):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def recipe_digest():return hashlib.sha256(json.dumps(plan(schema()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate(p):
 require(p['schema']=='crispembed-josev-final-pair-imatrix-v1'and p['base']==BASE and p['runtime_source']==RUNTIME and p['ggml']==GGML,'New exact closed source required')
 require(all(p[k]is False for k in ['runtime_change','default_enabled','publication_qualified','holdout_access','recipe_selected','python_activation_gold']), 'Native-development scope differs')
 require(p['recipes']==['uncalibrated-pair','calibrated-pair']and p['counts']=={str(k):v for k,v in COUNTS.items()}and p['inventory_sha256']==recipe_digest(),'Fixed identical paired-Q4 storage roles required')
 require(p['source_model']==SOURCE and p['original_reference_sha256']==REFERENCE_SHA and p['original_freeze_sha256']==FREEZE_SHA,'Frozen F32/original identity differs')
 require(p['calibration']==dict(layer=23,width=2048,case_ids=['mixed-de','null-and-options','json-unicode','state-only-clip'],rows=[319,197,165,512],total_rows=1193,
  stored_source_type='F32',imatrix_tensors=2,phase='capture-during-F32-control; admit-only-after-strict227-pass',statistics='ordered F64 square/add; one F32 sum cast; U64 row count'), 'Calibration data/format policy differs')
 require(p['gates']==dict(max_absolute=.002,min_row_cosine=.99999,logits_absolute=.002,probabilities_absolute=.0001,ordinal_expectation_absolute=.001)
  and p['model_forward_maximum']==40 and p['original_forwards']==0,'Original gates/call budget differs')
 previous=json.loads((ROOT/'docs/josev2b-last-projection-v1.json').read_text())
 required=set(previous['protected_sha256'])|set(previous['owned_sha256'])|{'docs/josev2b-last-projection-v1.json','src/imatrix.cpp','src/imatrix.h',
  'tests/results/josev2b/protected-recipes-v1-review.json','tests/results/josev2b/last-projection-v1-review.json'}
 require(required<=set(p['protected_sha256'])and not any(n.startswith('src/')for n in p['owned_sha256']),'All previous/runtime helpers protected; only test-owned new code')
 require(p['original_artifact']==previous['original_artifact'],'Original archive differs')
 require(git('rev-parse',BASE+':src')==git('rev-parse',RUNTIME+':src'),'Native math changed from reviewed mixed runtime')
 return p

def verify():
 p=validate(json.loads(PATH.read_text()));relative=str(PATH.relative_to(ROOT))
 require(git('show','HEAD:'+relative)==PATH.read_bytes(),'Dirty protocol')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=set(p['owned_sha256'])|{relative},'Undeclared source delta')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in p[group].items():
   f=ROOT/name;require(f.is_file()and not f.is_symlink()and sha(f)==pin and git('show','HEAD:'+name)==f.read_bytes(),'Source bytes differ: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==f.read_bytes(),'Protected baseline differs: '+name)
 require(git('ls-tree','-r','HEAD','src')==git('ls-tree','-r',BASE,'src')and git('rev-parse','HEAD:ggml').decode().strip()==GGML,'Production source/backend changed')
 require(git('-C',str(ROOT/'ggml'),'rev-parse','--show-toplevel').decode().strip()==str(ROOT/'ggml')
  and git('-C',str(ROOT/'ggml'),'rev-parse','HEAD').decode().strip()==GGML
  and not git('-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no'),'Actual clean a06 checkout required before payload access')
 require(not git('status','--porcelain','--untracked-files=no'),'Dirty tracked source')
 return p
if __name__=='__main__':print(json.dumps(verify(),indent=2))
