"""Explicit1.5M tokenizer-only closure; no historical guard exceptions."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE='cec6b7f0dd3ea78bfefb8587b3638c8ab23f462f'
REV='9d360b8b88ae11bea5784bf9012ff509fdd5c2da'
REPORT='b50ca1369f454d61b8c95583d8eb77a847c47ede29e52ed66b575784f4e4660f'
TOKENIZER='7357662aa3be57e8fd953365a37f892dab4ee702b817e1efcef9f193a388aff0'
ST_REV='b0bf39968ba461c34b29372144794eff1be06f6d'
ST_SOURCE='e7bd2c7692ceff913fec878b0cb4e7b5d7a8a979b3721593381484a130757f67'
WHEEL='https://files.pythonhosted.org/packages/2e/76/932be4b50ef6ccedf9d3c6639b056a967a86258c6d9200643f01269211ca/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
WHEEL_SHA='369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67'
PAYLOADS={'tokenizer.json': {'bytes': 1956542, 'sha256': '7357662aa3be57e8fd953365a37f892dab4ee702b817e1efcef9f193a388aff0'}, 'tokenizer_config.json': {'bytes': 646, 'sha256': 'fc250460005ed0c1d47fa58829a31ccfd57a5abacf89029c27271869a588dc18'}, 'config_sentence_transformers.json': {'bytes': 758, 'sha256': '007399a787035b4350ed00166bc77380a16fbf5b161153f9452e86b0cb3932a7'}, 'sentence_bert_config.json': {'bytes': 57, 'sha256': '58ffae3751d252be0313f661207de131aa09f3dbaf80df1a95b7dafc5bf45e5f'}, 'config.json': {'bytes': 1226, 'sha256': '54f7e0dc02d5814b3bd9b2f2f7f1f2cbd732aef7c4de7f334eaf2daa43e75e6e'}, 'modeling_modernbert_hash_moe_v10.py': {'bytes': 32641, 'sha256': '528afffbd33210845ed521b12af5569f9b6d0b0eb3f176beebb3efd83cee6668'}, 'LICENSE': {'bytes': 11358, 'sha256': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'}}
PATH=ROOT/'docs/moedern-unigram-frontend.json'
LANE={'src/moedern_text_frontend.h','src/moedern_text_frontend.cpp','tests/moedern-unigram/CMakeLists.txt',
 'tests/moedern-unigram/probe.cpp','tools/moedern-unigram/contract.py','tools/moedern-unigram/original.py',
 'tools/ci-heavy/moedern_unigram.py','tests/test_moedern_unigram.py','tests/fixtures/moedern-unigram-v1.json',
 '.github/workflows/moedern-unigram.yml','.github/workflows/ovisocr2-metadata.yml','docs/moedern-unigram-frontend.md'}
OLD={'ggml','cmake','CMakeLists.txt','VERSION','LICENSE','Cargo.lock','python','flutter','crispembed','crispembed-sys','wasm','examples',
 'docs/moedern-colbert-native-contract-v2.json','docs/moedern-colbert-native-v2.md','tools/moedern-native-v2','tests/moedern-native-v2',
 'tools/ci-heavy/moedern_native_v2.py','tests/test_moedern_native_v2.py','.github/workflows/moedern-native-v2.yml'}


def require(ok,why):
 if not ok:raise ValueError(why)

def sha(data):return hashlib.sha256(data).hexdigest()
def dump(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def old_src():return {x.split('\t')[1] for x in git('ls-tree','-r',BASE,'--','src').splitlines()}
def source():
 cfg=json.loads(PATH.read_text());require(cfg['base']==BASE and cfg['schema']=='crispembed-moedern-unigram-v1' and cfg['default_enabled']is False and cfg['model_weights']is False and cfg['source_payloads']==PAYLOADS,'source policy differs')
 require(set(cfg['source_sha256'])==LANE and set(cfg['protected_objects'])==OLD|old_src(),'source closure omitted')
 current={x.split('\t')[1] for x in git('ls-tree','-r','HEAD','--','src').splitlines()}
 require(current==old_src()|{'src/moedern_text_frontend.cpp','src/moedern_text_frontend.h'},'unexpected native source delta')
 for p,v in cfg['protected_objects'].items():require(git('rev-parse','HEAD:'+p)==v==git('rev-parse',BASE+':'+p),'historical source changed: '+p)
 for p,v in cfg['source_sha256'].items():require(sha((ROOT/p).read_bytes())==v==sha(subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)),'source bytes differ: '+p)
 require(not git('status','--porcelain','--',*sorted(LANE|OLD|old_src()|{'docs/moedern-unigram-frontend.json'})),'dirty measured source')
 return cfg

def gguf(metadata):
 def string(s):b=s.encode();return struct.pack('<Q',len(b))+b
 out=bytearray(b'GGUF'+struct.pack('<IQQ',3,0,len(metadata)))
 for key,value in metadata.items():
  out+=string(key)
  if isinstance(value,str):out+=struct.pack('<I',8)+string(value);continue
  dtype=None
  if isinstance(value,tuple):dtype,value=value
  require(isinstance(value,list) and value,'nonempty typed metadata array required')
  dtype=12 if dtype=='f64' else 8 if isinstance(value[0],str) else 5
  out+=struct.pack('<IIQ',9,dtype,len(value))
  for item in value:out+=string(item) if dtype==8 else struct.pack('<d' if dtype==12 else '<i',item)
 out+=b'\0'*(-len(out)%32);return bytes(out)

def read_gguf(data):
 """Independent bounded metadata reader; zero tensors, F64 kept lossless."""
 offset=0
 def take(n):
  nonlocal offset
  require(0<=n<=16*1024**2 and offset+n<=len(data),'metadata extent differs');value=data[offset:offset+n];offset+=n;return value
 def unpack(fmt):return struct.unpack(fmt,take(struct.calcsize(fmt)))
 def string():return take(unpack('<Q')[0]).decode('utf-8')
 require(take(4)==b'GGUF','metadata magic differs');version,tensors,count=unpack('<IQQ');require(version==3 and tensors==0 and count<=64,'metadata header differs')
 result={}
 for _ in range(count):
  key=string();require(key not in result,'duplicate metadata key');kind=unpack('<I')[0]
  if kind==8:value=string()
  else:
   require(kind==9,'unsupported scalar metadata');dtype,n=unpack('<IQ');require(dtype in (5,8,12) and n<=65536,'metadata array differs')
   values=[string() if dtype==8 else unpack('<d'if dtype==12 else '<i')[0]for _ in range(n)];value=('f64',values)if dtype==12 else values
  result[key]=value
 require(len(data)-offset<32 and not any(data[offset:]),'metadata trailing payload differs')
 return result
