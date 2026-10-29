"""Closed, test-only weightcast risk protocol; standard library only."""
import hashlib,json,math,struct,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='5df5062c24501e5d8784fda2fef2d44a3c1dada0'
PATH=ROOT/'docs/wasm-layer3-f32-v1.json'
OWNED={'.github/workflows/wasm-layer3-f32-v1.yml','docs/wasm-layer3-f32-v1.json','docs/wasm-layer3-f32-v1.md',
 'tools/wasm-layer3-f32/protocol.py','tools/ci-heavy/wasm_layer3_f32_v1.py','tests/wasm-layer3-f32/CMakeLists.txt',
 'tests/wasm-layer3-f32/operator.cpp','tests/wasm-layer3-f32/browser.cjs','tests/test_wasm_layer3_f32_v1.py'}
VALUES=(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def decode(raw):
 """Independent literal IQ4_XS format interpretation; not a model converter."""
 require(0<len(raw)<=313344 and len(raw)%136==0,'Exact IQ4_XS block extent required')
 result=bytearray();f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
 for at in range(0,len(raw),136):
  d,high=struct.unpack_from('<eH',raw,at);require(math.isfinite(d),'Nonfinite IQ4_XS scale')
  for ib in range(8):
   low=raw[at+4+ib//2];ls=((low>>(4*(ib%2)))&15)|(((high>>(2*ib))&3)<<4)
   dl=f32(d*(ls-32));q=raw[at+8+ib*16:at+24+ib*16]
   for shift in [0,4]:
    for v in q:result.extend(struct.pack('<f',f32(dl*VALUES[(v>>shift)&15])))
 return bytes(result)
def source():
 cfg=json.loads(PATH.read_text());git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(cfg['schema']=='crispembed-wasm-layer3-f32-v1'and cfg['base']==BASE,'Frozen cast declaration differs')
 require(cfg['gates']=={'max_absolute':.002,'minimum_row_cosine':.99999}and cfg['comparisons']==25 and cfg['required_byte_anchors']==6,'Frozen gates/coverage differ')
 require(cfg['production_change']is False and cfg['original_rerun']is False and cfg['qualification']is False,'Test-only scope required')
 old=json.loads((ROOT/'docs/wasm-q8k-cross-v1.json').read_text())
 reviewed=json.loads((ROOT/'tests/results/packaging/wasm-q8k-cross-v1.json').read_text())
 causal=cfg['causal']
 require(causal['run']==reviewed['service']['run']and causal['source']==reviewed['service']['source']and causal['report_sha256']==reviewed['review_members']['review/cross.json']['sha256'],'Exact reviewed causal source/report required')
 for kind in ['review','raw']:
  require(causal[kind]=={k:reviewed['artifacts'][kind][k]for k in ['id','bytes','sha256']},'Exact reviewed artifact descriptor required')
 required=set(old['protected_sha256'])|set(old['owned_sha256'])|{'docs/wasm-q8k-cross-v1.json','docs/wasm-q8k-cross-result-v1.md','tests/results/packaging/wasm-q8k-cross-v1.json'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact protected/owned closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared source delta')
 require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Dirty cast declaration')
 for group in ['protected_sha256','owned_sha256']:
  for name,expected in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(sha(raw)==expected and git('show','HEAD:'+name)==raw,'Dirty or changed source: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw,'Frozen dependency changed: '+name)
 return cfg
