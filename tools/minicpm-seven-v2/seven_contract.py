"""Separately closed seven-only caller; historical19c/cec guards unchanged."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/minicpm-retained-v3'),str(ROOT/'tools/minicpm-ocr'),str(ROOT/'tools')]
import common as old
BASE='102e80c9376e6eb21b90573a66ffafe1287e43b7'
PATH=ROOT/'docs/minicpm-graph-seven-arm-v2.json'
REVIEW=ROOT/'tests/results/minicpm-ocr/retained-analysis-v2-review.json'
OWNED={'docs/minicpm-graph-seven-arm-v2.json','docs/minicpm-graph-seven-arm-v2.md','tools/minicpm-seven-v2/seven_contract.py','tools/minicpm-seven-v2/run.py','tools/minicpm-seven-v2/vision.py','tests/test_minicpm_seven_v2.py','.github/workflows/minicpm-graph-seven-arm-v2.yml'}
require,atomic,sha=old.require,old.atomic,old.sha

def declaration():
 cfg=json.loads(PATH.read_text());v1=json.loads(old.CONTINUE.read_text());review=json.loads(REVIEW.read_text())
 require(cfg['schema']=='crispembed-minicpm-seven-arm-v2'and cfg['base']==BASE and cfg['runtime_source']==old.OLD_SOURCE,'Exact seven-v2 source/runtime required')
 require(cfg['budgets']==v1['budgets']and cfg['remaining_arms']==v1['remaining_arms']and cfg['carried_arms']==23 and cfg['new_ordinary_forwards']==7,'Frozen remaining seven/gates required')
 require(cfg['working_output_cap_bytes']==old.OUT_CAP and cfg['working_reserve_bytes']==old.RESERVE and cfg['native_threads']==4 and cfg['child_seconds']==14400 and cfg['workflow_minutes']==360,'Existing budgets/profile only')
 require(all(cfg[k]is False for k in ['original_rerun','runtime_math_changes','default_enabled','publication_qualified']),'No original/runtime/default/publication change')
 require(sha(REVIEW)==cfg['completed_analysis_review_sha256']and cfg['completed_analysis_service']==review['service']and cfg['completed_analysis_artifact']==review['artifact']and cfg['completed_analysis_report']==review['report'],'Exact reviewed cec dependency required')
 return cfg

def source_guard(reviewed,hosted=False):
 cfg=declaration();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(git('rev-parse','HEAD').decode().strip()==reviewed and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Exact clean new caller source required')
 oldcfg=json.loads(old.CONTINUE.read_text());cec=json.loads((ROOT/'docs/minicpm-graph-retained-analysis-v2.json').read_text())
 required=set(oldcfg['protected_sha256'])|set(oldcfg['owned_sha256'])|{str(old.CONTINUE.relative_to(ROOT))}|set(cec['protected_sha256'])|set(cec['owned_sha256'])|{'docs/minicpm-graph-retained-analysis-v2.json',str(REVIEW.relative_to(ROOT)),'docs/minicpm-retained-analysis-v2-result.md'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact inherited/new closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared seven-v2 change')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin and git('show','HEAD:'+name)==raw,'Changed source: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw,'Historical guard/runtime/reference changed: '+name)
 require(git('show','HEAD:CMakeLists.txt')==git('show',old.OLD_SOURCE+':CMakeLists.txt'),'Root CMake changed')
 inherited=old.source_guard('continue',reviewed)
 if hosted:require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and os.environ.get('GITHUB_SHA')==reviewed,'Genuine source-bound GHA required')
 return dict(source=reviewed,contract_sha256=sha(PATH),inherited_guard=inherited,protected=len(required),owned=len(cfg['owned_sha256']),runtime_math_changes=False)

def verify_analysis(report):
 require(report['completed']is True and report['qualified']is False and report['native_forwards']==0 and report['model_accessed']is False and report['original_rerun']is False,'Actual completed zero-forward analysis required')
 require((report['comparisons'],report['carried_comparisons'],report['new_comparisons'],report['carried_arms'],report['new_arms'])==(138,36,102,6,17),'Completed138 coverage required')
 catalog=json.loads(old.CATALOG.read_text());keys=lambda rows:[tuple(r[k]for k in ['id','dtype','graph'])for r in rows]
 require(keys(report['cases'])==keys(catalog['cases'])and report['source']['source']=='cec1faebd39582a6103774dd9598b3e26d6533f1','Exact retained23/source ordering required')
 for row in report['cases']:
  require(row['completed']is True and row['pixels_bit_exact']is True and set(row['vision']['stages'])=={'vision_'+str(i)for i in range(6)}and row['vision']['actual_ref']['stages']==6 and row['vision']['actual_ref']['all_metrics_agree']is True,'Completed actual Ref/pixel coverage required')
  for name,v in row['vision']['stages'].items():
   require(v['cpp_ref']['name']==name and v['passed']==bool(v['independent_passed']and v['cpp_ref']['passed'])and all(x['passed']is True for x in v['metric_agreement'].values()),'Prior numeric flags may not change')
 require(report['all_pixels_bit_exact']is True and report['historical_numeric_failures_preserved']is True,'Prior pixel/failure proof required')
 return report
