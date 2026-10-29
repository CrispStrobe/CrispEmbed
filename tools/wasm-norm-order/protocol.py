"""Test-only native-order variance risk; no production/model path."""
import hashlib,json,math,re,struct,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='e5929bb094362043d53196db0f03cc302824d0d7'
PATH=ROOT/'docs/wasm-native-order-norm-risk-v1.json'
OWNED={str(PATH.relative_to(ROOT)),'docs/wasm-native-order-norm-risk-v1.md','tools/wasm-norm-order/protocol.py','tools/ci-heavy/wasm_norm_order_v1.py','tests/wasm-norm-order/CMakeLists.txt','tests/wasm-norm-order/operator.cpp','tests/wasm-norm-order/browser.cjs','tests/test_wasm_norm_order_v1.py','tests/results/packaging/wasm-norm-risk-initial-lint.json','.github/workflows/wasm-native-order-norm-risk-v1.yml'}
PACKETS=('attention_native','ffn_native','attention_candidate','ffn_candidate','chain_native','chain_candidate')
WEIGHT='blk.3.ffn_up.weight'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def cpp_tokens(raw):
 text=raw.decode('utf-8')
 tokens=re.findall(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|[^\s]',text,re.S)
 return '\0'.join(t for t in tokens if not t.startswith(('//','/*'))).encode('utf-8')
def group8(squares):
 require(len(squares)==8 and all(math.isfinite(x)for x in squares),'Eight finite authored squares required')
 f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
 a=[f(squares[i+4]+squares[i])for i in range(4)]
 return f(f(a[0]+a[2])+f(a[1]+a[3]))
def workspace(raw):
 require(len(raw)==4896,'Exact144 Q8_0 blocks required');rows=[]
 for i in range(144):
  b=raw[i*34:(i+1)*34];d=struct.unpack_from('<e',b)[0];require(math.isfinite(d),'Finite Q8_0 scale required')
  rows.append(dict(token=i//12,block=i%12,scale=d,scale_bytes=b[:2].hex(),signed_sha256=sha(b[2:]),block_sha256=sha(b)))
 return rows

def source():
 cfg=json.loads(PATH.read_text());git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(cfg['schema']=='crispembed-wasm-native-order-norm-risk-v1'and cfg['base']==BASE and cfg['gates']=={'max_absolute':.002,'minimum_row_cosine':.99999},'Exact native-order risk scope/gates required')
 require(cfg['packets']==list(PACKETS)and cfg['baseline_byte_anchors']==30 and cfg['native_order_norm_bit_controls']==12 and cfg['comparisons']==56,'Frozen phase coverage differs')
 require(all(cfg[k]is False for k in ['production_change','model_forward','original_rerun','qualification']),'Test-only bounded risk required')
 old=json.loads((ROOT/'docs/wasm-layer3-f32-reanalysis-v1.json').read_text());required=set(old['protected_sha256'])|set(old['owned_sha256'])|{'docs/wasm-layer3-f32-reanalysis-v1.json'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact immutable/new source closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Dirty/undeclared norm-risk source')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(sha(raw)==pin and git('show','HEAD:'+name)==raw,'Source differs: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw,'Historical source/guard changed: '+name)
 old_cpp=git('show','f78131c4616ece9f236ddab7695db6cd32a5521f:tests/wasm-norm-order/operator.cpp');new_cpp=(ROOT/'tests/wasm-norm-order/operator.cpp').read_bytes()
 require(cfg['format_predecessor']['source']=='f78131c4616ece9f236ddab7695db6cd32a5521f' and sha(old_cpp)==cfg['format_predecessor']['cpp_sha256'] and cpp_tokens(old_cpp)==cpp_tokens(new_cpp) and sha(cpp_tokens(new_cpp))==cfg['format_predecessor']['cpp_tokens_sha256'],'Formatting-only predecessor token identity required')
 return cfg
