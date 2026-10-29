"""Closed honest CPU provider; no old guard mutation or GitHub impersonation."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/gemma-decision-q4-v1'))
import gemma_contract as old
BASE='e0957f74b08ded853ea981891647e85c9edc324b'
PREVIOUS='6e327e902ebdd0feb531c3beae4b52902b7e781b'
PROFILE=ROOT/'docs/gemma-decision-q4-cpu-v2.json'
GGML=old.GGML
PROTECTED=old.PROTECTED+['tools/gemma-decision-q4-v1','docs/gemma-decision-first-q4-v1.json','docs/gemma-decision-first-q4-v1.md',
 'tools/ci-heavy/gemma_decision_first_q4.py','tests/gemma-decision-q4','tests/test_gemma_decision_first_q4.py','.github/workflows/gemma-decision-first-q4-v1.yml']
OWNED={'tools/gemma-decision-q4-cpu-v2/cpu_contract.py','tools/gemma-decision-q4-cpu-v2/kernel.py',
 'tools/ci-heavy/gemma_decision_q4_cpu_v2.py','tests/test_gemma_q4_cpu_v2.py','docs/gemma-decision-q4-cpu-v2.md'}
require,sha,identity,read,write,git=old.require,old.sha,old.identity,old.read,old.write,old.git

def declaration():
 p=read(PROFILE)
 require(p['schema']=='crispembed-gemma-decision-q4-cpu-v2'and p['base']==BASE and p['previous_source']==PREVIOUS and p['ggml']==GGML,'Exact successor source required')
 require(set(p['protected_git_objects'])==set(PROTECTED)and set(p['owned_sha256'])==OWNED,'Complete exact old/new closure required')
 require(p['provider']=='Kaggle CPU'and p['external_exact_version_Get_Status_every_List_page_required']is True
  and p['resources']==dict(minimum_ram_bytes=8*2**30,minimum_disk_bytes=16*2**30,minimum_cpus=2,output_cap_bytes=4*2**30,minimum_live_reserve_bytes=2*2**30,single_file_cap_bytes=2*2**30)
  and p['scope']==dict(encoder_forward_maximum=264,encoder_per_completed_model=88,original_forwards=0,quantizer_runs=2,public_capi_qualified=False,publication_qualified=False,recipe_selected=False),'Same bounded experiment/provider required')
 require(p['old_protocol']==identity(old.PROFILE),'Old protocol identity changed')
 return p

def audit_tree(source):
 p=declaration();require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD')==source,'Exact source argument required')
 rel=str(PROFILE.relative_to(ROOT));require(git('show','HEAD:'+rel)==PROFILE.read_text().strip(),'Dirty successor protocol')
 require(set(git('diff','--name-only',BASE,'HEAD').splitlines())<=OWNED|{rel},'Undeclared successor changes')
 for name,obj in p['protected_git_objects'].items():
  require(git('rev-parse',PREVIOUS+':'+name)==obj and git('rev-parse',BASE+':'+name)==obj and git('rev-parse','HEAD:'+name)==obj,'Frozen predecessor/runtime object differs: '+name)
 for name,digest in p['owned_sha256'].items():require(sha(ROOT/name)==digest and git('show','HEAD:'+name)==(ROOT/name).read_text().strip(),'Owned source differs: '+name)
 require(not git('status','--porcelain','--untracked-files=no'),'Dirty tracked checkout')
 return p

def provider(source):
 require(os.environ.get('GITHUB_ACTIONS')is None and os.environ.get('GITHUB_SHA')is None,'No GitHub provider impersonation')
 require(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')
  and os.environ.get('CRISPEMBED_CAPTURE_SOURCE')==source,'Genuine Kaggle CPU/source required')

def source(reviewed):
 audit_tree(reviewed);provider(reviewed)
 require(git('-C',str(ROOT/'ggml'),'rev-parse','--show-toplevel')==str(ROOT/'ggml')and git('-C',str(ROOT/'ggml'),'rev-parse','HEAD')==GGML
  and not git('-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no'),'Actual clean a06 checkout required')
 return old.validate(read(old.PROFILE))
