"""Unfinished public MoE source draft; no execution/qualification guard waiver."""
from pathlib import Path
import hashlib
import json
import subprocess
ROOT=Path(__file__).resolve().parents[2]
BASE='1a053116908224cf1c9358d7d24c85ba36e82eff'
PATH=ROOT/'docs/moedern-public-text.json'
CHANGED={'src/moedern_text.cpp','src/moedern_text.h','src/moedern_text_frontend.cpp','src/moedern_text_frontend.h'}
NEW={'src/moedern_packed_profile.h','src/moedern_text_api.cpp','src/moedern_text_api.h',
 'python/crispembed/_moedern_text.py','crispembed/src/moedern_text.rs','flutter/crispembed/lib/src/moedern_text.dart',
 'tools/pack_moedern_text_f32.py','tools/moedern-public/public_contract.py','tests/test_moedern_public_text.py','docs/moedern-public-text.md'}
ROOTS={'ggml','CMakeLists.txt','cmake','src/crispembed.h','python/crispembed/_binding.py','python/crispembed/__init__.py',
 'crispembed/src/lib.rs','crispembed-sys/src/lib.rs','flutter/crispembed/lib/crispembed.dart','examples','VERSION','Cargo.lock',
 'tools/moedern-text','tools/ci-heavy/moedern_text_composed.py','docs/moedern-text-composed.json','.github/workflows/ovisocr2-metadata.yml'}
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def old_src():return set(git('ls-tree','-r','--name-only',BASE,'src').decode().splitlines())
def source():
 cfg=json.loads(PATH.read_text());require(cfg['schema']=='crispembed-moedern-public-source-draft-v1'and cfg['base']==BASE and cfg['default_enabled']is False and cfg['public_model_proof']is False,'draft source policy differs')
 require(set(cfg['owned_sha256'])==NEW|CHANGED and set(cfg['protected_objects'])==(old_src()-CHANGED)|ROOTS,'source closure omitted')
 require(set(git('ls-tree','-r','--name-only','HEAD','src').decode().splitlines())==old_src()|{p for p in NEW if p.startswith('src/')},'unexpected runtime source delta')
 for name,value in cfg['protected_objects'].items():require(git('rev-parse','HEAD:'+name).decode().strip()==value==git('rev-parse',BASE+':'+name).decode().strip(),'old source differs: '+name)
 for name,value in cfg['owned_sha256'].items():require(sha((ROOT/name).read_bytes())==value and git('show','HEAD:'+name)==(ROOT/name).read_bytes(),'draft source differs: '+name)
 require(not git('status','--porcelain','--',*sorted(NEW|CHANGED|ROOTS|old_src()|{'docs/moedern-public-text.json'})),'dirty draft source')
 before=git('show',BASE+':src/moedern_text_frontend.cpp').decode();after=(ROOT/'src/moedern_text_frontend.cpp').read_text();marker='std::string text_frontend::normalize('
 require(before[before.index(marker):]==after[after.index(marker):],'frontend inference body changed')
 return cfg
if __name__=='__main__':
 cfg=source();print(json.dumps({'source_closed':True,'prepared_only':True,'public_model_proof':False,'compiled':False,'owned':len(cfg['owned_sha256']),'protected':len(cfg['protected_objects'])}))
