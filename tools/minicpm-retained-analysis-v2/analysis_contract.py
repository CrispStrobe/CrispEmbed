"""Separate analysis-only correction; every19c source/runtime guard unchanged."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/minicpm-retained-v3'),str(ROOT/'tools/minicpm-ocr'),str(ROOT/'tools')]
import common as inherited
BASE='4570505fec70efaea8066273c2fb21e4adc62590'
SOURCE='19c3f7bdf6d452614721946adefb95f6d3cf0180'
PATH=ROOT/'docs/minicpm-graph-retained-analysis-v2.json'
INITIAL=ROOT/'tests/results/minicpm-ocr/retained-analysis-initial.json'
OWNED={'docs/minicpm-graph-retained-analysis-v2.json','docs/minicpm-graph-retained-analysis-v2.md','docs/minicpm-retained-analysis-initial.md','tests/results/minicpm-ocr/retained-analysis-initial.json','tools/minicpm-retained-analysis-v2/analysis_contract.py','tools/minicpm-retained-analysis-v2/prior.py','tools/minicpm-retained-analysis-v2/analyze.py','tests/test_minicpm_retained_analysis_v2.py','.github/workflows/minicpm-graph-retained-analysis-v2.yml'}
require,atomic,sha=inherited.require,inherited.atomic,inherited.sha

def declaration():
 cfg=json.loads(PATH.read_text());old=json.loads(inherited.ANALYSIS.read_text())
 require(cfg['schema']=='crispembed-minicpm-retained-analysis-v2'and cfg['base']==BASE and cfg['inherited_source']==SOURCE and cfg['runtime_source']==inherited.OLD_SOURCE,'Exact analysis-only source scope required')
 require(cfg['budgets']==old['budgets']and cfg['carried_arms']==6 and cfg['new_arms']==17 and cfg['carried_comparisons']==36 and cfg['new_comparisons']==102 and cfg['total_comparisons']==138,'Frozen coverage/budgets differ')
 require(cfg['native_forwards']==0 and type(cfg['native_forwards'])is int and all(cfg[k]is False for k in ['runtime_math_changes','model_access','original_rerun','qualified']),'Zero-forward analysis only')
 require(cfg['merger_policy']=='first-list-element-matches-only-native-overview'and cfg['unmapped_policy']=='every-later-native-part-span-magnitudes-no-reference','Exact source-proven merger scope required')
 initial=json.loads(INITIAL.read_text());require(cfg['prior']==initial['artifacts']and cfg['prior_service']==initial['service'],'Exact immutable partial analysis required')
 return cfg

def source_guard(reviewed):
 cfg=declaration();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(git('rev-parse','HEAD').decode().strip()==reviewed,'Explicit v2 caller source differs')
 require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes()and set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Dirty/undeclared v2 source')
 old=json.loads(inherited.ANALYSIS.read_text());required=set(old['protected_sha256'])|set(old['owned_sha256'])|{str(inherited.ANALYSIS.relative_to(ROOT))}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact inherited/new closure required')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin and git('show','HEAD:'+name)==raw,'Source changed: '+name)
   if group=='protected_sha256':require(git('show',SOURCE+':'+name)==raw,'Immutable19c source changed: '+name)
 require(git('show','HEAD:CMakeLists.txt')==git('show',inherited.OLD_SOURCE+':CMakeLists.txt'),'Root CMake differs from83')
 original=inherited.source_guard('analysis',reviewed)
 return dict(source=reviewed,contract_sha256=sha(PATH),inherited_guard=original,runtime_math_changes=False)
