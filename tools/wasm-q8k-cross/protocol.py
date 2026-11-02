"""Retained IQ4_XS/Q8_K cross-admission v1. No production/model inference."""
import hashlib,json,math,struct,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='55d6a165388f285c11b33abd1d53d70d4b5a645e'
PATH=ROOT/'docs/wasm-q8k-cross-v1.json'
OWNED={'docs/wasm-q8k-cross-v1.json','docs/wasm-q8k-cross-v1.md','tools/wasm-q8k-cross/protocol.py',
 'tools/ci-heavy/wasm_q8k_cross_v1.py','tests/wasm-q8k-cross/CMakeLists.txt','tests/wasm-q8k-cross/operator.cpp',
 'tests/wasm-q8k-cross/browser.cjs','tests/test_wasm_q8k_cross_v1.py','.github/workflows/wasm-q8k-cross-v1.yml'}
WEIGHT='blk.3.ffn_down.weight'
ARMS=('native_native','native_candidate','candidate_native','candidate_candidate','native_scale_candidate_integers','candidate_scale_native_integers')
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def valid_workspace(raw):
 require(len(raw)==21024,'Exact72 Q8_K blocks required')
 for offset in range(0,len(raw),292):
  require(math.isfinite(struct.unpack_from('<f',raw,offset)[0]),'Nonfinite Q8_K scale')
  qs=struct.unpack_from('<256b',raw,offset+4);sums=struct.unpack_from('<16h',raw,offset+260)
  require(tuple(sum(qs[k:k+16])for k in range(0,256,16))==sums,'Q8_K sum fields inconsistent')
 return raw

def hybrid(scales,integers):
 valid_workspace(scales);valid_workspace(integers)
 return valid_workspace(b''.join(scales[k:k+4]+integers[k+4:k+292]for k in range(0,21024,292)))

def describe(raw):
 valid_workspace(raw)
 return [dict(token=k//(292*6),block=(k//292)%6,scale=struct.unpack_from('<f',raw,k)[0],
  scale_bits=raw[k:k+4].hex(),quant_sha256=sha(raw[k+4:k+260]),sums_sha256=sha(raw[k+260:k+292]),
  block_sha256=sha(raw[k:k+292]))for k in range(0,21024,292)]

def valid_controls(value,threads=False):
 keys={'positive','wrong_extent_rejected','nonfinite_rejected','invalid_workspace_rejected','output_preserved_on_failure','recovery'}
 require(set(value)==keys|({'threads'}if threads else set())and all(value[k]is True for k in keys),'Complete actual backend controls required')
 if threads:require(value['threads']==[1,2],'Actual native thread controls required')
 return value

def source():
 cfg=json.loads(PATH.read_text());git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(cfg['schema']=='crispembed-wasm-q8k-cross-v1'and cfg['base']==BASE and cfg['gates']=={'max_absolute':.002,'minimum_row_cosine':.99999},'Frozen protocol differs')
 require(cfg['arms']==list(ARMS)and cfg['comparisons']==44 and cfg['required_byte_anchors']==18 and cfg['publication_qualified']is False,'Bounded arm/qualification declaration differs')
 required=set(json.loads((ROOT/'docs/wasm-f32-operators-v1.json').read_text())['source_sha256'])|{'docs/wasm-f32-operators-v1.json','tests/wasm-layer3/kernel_capture.inc','tests/results/packaging/wasm-f32-operators-v1.json'}
 require(set(cfg['protected_sha256'])==required,'Exact frozen dependency closure required')
 require(set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact owned closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared source delta')
 require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Dirty protocol')
 for group in ['protected_sha256','owned_sha256']:
  for name,expected in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(sha(raw)==expected and git('show','HEAD:'+name)==raw,'Source SHA differs: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw,'Frozen dependency differs: '+name)
 return cfg
