"""Distinct raw-text experiment. Historical producers/guards remain immutable."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[2]
BASE='3da7cf8e9a4fcd6342fc9a6ac00d07de42b7c324'
PATH=ROOT/'docs/moedern-text-composed.json'
MACRO='CRISPEMBED_MOEDERN_COMPOSED_TESTING'
NATIVE_V2={'run':40441341206,'source':'cec6b7f0dd3ea78bfefb8587b3638c8ab23f462f','review':{'id':12706681339,'name':'moedern-native-v2-review-40441341206','bytes':179826,'sha256':'f6e44abcc4aa0fb96ea6ee6d570b688007cce7f55d05d114c11765f04e0337f8'},'report_sha256':'74ce48da604c46a2fba77f00d4187d5e8917d881e1185538fe8ac55df4b6f668'}
GATES={'max_absolute':0.002,'min_row_cosine':0.99999,'maxsim_max_absolute':0.002}


def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def adapt_engine(old):
 """Only interface/namespace and test-only event plumbing; arithmetic untouched."""
 text=old.replace('#include "moedern_colbert.h"','#include "moedern_text_engine_internal.h"').replace('namespace moedern_colbert {','namespace moedern_text_detail {').replace('// namespace moedern_colbert','// namespace moedern_text_detail')
 start=text.index('void emit(');end=text.index('void softmax(',start)
 text=text[:start]+'#ifdef '+MACRO+'\n'+text[start:end]+'''#else
// Production retains the old finiteness checks, with no callback or labels.
#define emit(callback, label, values, ...) require(finite(values), "Nonfinite MoEdern stage")
#define emit_ids(...) ((void)0)
#endif
'''+text[end:]
 text=text.replace('result & output, const observer & dump) {','result & output\n#ifdef '+MACRO+'\n, const observer & dump\n#endif\n) {')
 start=text.index('        std::vector<int64_t> integers(ids, ids + count);');end=text.index('        floats hidden;',start)
 text=text[:start]+'#ifdef '+MACRO+'\n'+text[start:end]+'#endif\n'+text[end:]
 old_names='            const std::string b = "blk." + std::to_string(layer) + ".", n = "layer." + std::to_string(layer) + ".";'
 require(old_names in text,'old stage namespace declaration changed')
 text=text.replace(old_names,'            const std::string b = "blk." + std::to_string(layer) + ".";\n#ifdef '+MACRO+'\n            const std::string n = "layer." + std::to_string(layer) + ".";\n#endif')
 text=text.replace('        ctx->failure = "Unknown observer/encode exception";','#ifdef '+MACRO+'\n        ctx->failure = "Unknown observer/encode exception";\n#else\n        ctx->failure = "Unknown MoEdern encode exception";\n#endif')
 return text+'\n#ifndef '+MACRO+'\n#undef emit\n#undef emit_ids\n#endif\n'

def tokens(text):
 parts=re.findall(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^\s]',text)
 return [p for p in parts if not p.startswith('//') and not p.startswith('/*')]

LANE={'src/moedern_text.h','src/moedern_text.cpp','src/moedern_text_testing.h','src/moedern_text_engine_internal.h','src/moedern_text_engine.cpp',
 'tools/moedern-text/contract.py','tools/ci-heavy/moedern_text_composed.py','tests/moedern-text/CMakeLists.txt','tests/moedern-text/probe.cpp',
 'tests/moedern-text/diff.cpp','tests/moedern-text/controls.cpp','tests/test_moedern_text_composed.py','docs/moedern-text-composed.md',
 '.github/workflows/moedern-text-composed.yml','.github/workflows/ovisocr2-metadata.yml'}
OLD={'ggml','cmake','CMakeLists.txt','VERSION','LICENSE','Cargo.lock','python','flutter','crispembed','crispembed-sys','wasm','examples',
 'tools/moedern-native-v2','tools/ci-heavy/moedern_native_v2.py','tests/moedern-native-v2','tests/test_moedern_native_v2.py',
 'docs/moedern-colbert-native-contract-v2.json','.github/workflows/moedern-native-v2.yml',
 'tools/moedern-unigram','tools/ci-heavy/moedern_unigram.py','tests/moedern-unigram','tests/test_moedern_unigram.py',
 'tests/fixtures/moedern-unigram-v1.json','docs/moedern-unigram-frontend.json','.github/workflows/moedern-unigram.yml'}

def old_src():return {x.split('\t')[1] for x in git('ls-tree','-r',BASE,'--','src').splitlines()}
def pending_guard(cfg):
 pin=cfg['unigram']
 require(pin['reviewed']is True and pin['completed_parity']is True,'reviewed actual Unigram producer pins pending; no API/payload access')
 require(type(pin['run'])is int and pin['run']>0 and re.fullmatch('[0-9a-f]{40}',pin['source']or'')is not None,'exact reviewed Unigram run/source pending')
 for name in ['review','metadata']:
  artifact=pin[name]
  cap=4*1024**2 if name=='review'else 64*1024**2
  require(type(artifact['id'])is int and type(artifact['bytes'])is int and artifact['id']>0 and 0<artifact['bytes']<=cap and
   artifact.get('name')=='moedern-unigram-'+name+'-'+str(pin['run']) and re.fullmatch('[0-9a-f]{64}',artifact['sha256']or'')is not None,'reviewed Unigram artifact identity pending')
 for key in ['report_sha256','component_sha256','component_bytes','original_report_sha256']:
  require(type(pin[key])is int and 0<pin[key]<=16*1024**2 if key=='component_bytes'else re.fullmatch('[0-9a-f]{64}',pin[key]or'')is not None,'reviewed Unigram member/component identity pending')

 for key in ['component_member','gold_member']:
  value=pin.get(key);require(isinstance(value,str)and value and re.fullmatch('[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*',value)is not None and all(part not in ('.','..')for part in value.split('/')),'explicit reviewed Unigram member path pending')

def source(require_review=True):
 cfg=json.loads(PATH.read_text())
 require(cfg['schema']=='crispembed-moedern-text-composed-v1'and cfg['base']==BASE and cfg['default_enabled']is False and cfg['model_recapture']is False,'source policy differs')
 require(cfg['primary']==json.loads((ROOT/'docs/moedern-colbert-native-contract-v2.json').read_text())['primary']and cfg['native_v2']==NATIVE_V2 and cfg['gates']==GATES,'immutable primary/native-v2/gates differ')
 if require_review:pending_guard(cfg) # Deliberately BEFORE Git/dependency API/model access.
 require(set(cfg['source_sha256'])==LANE and set(cfg['protected_objects'])==OLD|old_src(),'source closure omitted')
 current={x.split('\t')[1] for x in git('ls-tree','-r','HEAD','--','src').splitlines()}
 new={p for p in LANE if p.startswith('src/')};require(current==old_src()|new,'unexpected runtime delta')
 for p,v in cfg['protected_objects'].items():require(git('rev-parse','HEAD:'+p)==v==git('rev-parse',BASE+':'+p),'historical source changed: '+p)
 for p,v in cfg['source_sha256'].items():require(sha((ROOT/p).read_bytes())==v==sha(subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)),'source bytes differ: '+p)
 require(not git('status','--porcelain','--',*sorted(LANE|OLD|old_src()|{'docs/moedern-text-composed.json'})),'dirty measured source')
 old=subprocess.check_output(['git','show',BASE+':src/moedern_colbert.cpp'],cwd=ROOT,text=True)
 require(tokens(adapt_engine(old))==tokens((ROOT/'src/moedern_text_engine.cpp').read_text()),'copied kernel arithmetic/declared adaptation differs')
 return cfg
