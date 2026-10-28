"""NEW native owner prerequisites; original110 contract/source remain immutable."""
import hashlib,json,os,re,subprocess
from pathlib import Path
from josev2b_task_quality_v1 import (declaration,panel,protocol,expected_stages,require,sha256,compare_public)
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/josev2b-task-quality-native-v1.json'
BASE='6af6865598c592b1d86ffc930c9054f643b15064'
ORIGINAL='110620ea1ada0748007e6b6698e0c5babb4940ed'
RUNTIME='9c475d0dac70264adc123a63b6c1827b4b7280d5'
MIXED_SOURCE='54c2b5767ae82c853d8649870abd234a2fbe7ee5'
MIXED_SHA='f6f7b694f5080e1cf59fefc84400abe93b76b92664187f7b251ed0b354075e12'
SELECTION_RULE='strictF32control+structural/criticalbytes+alloldargmax+eachpublicprobability.01+eacholdordinal.001; smallestbytes'
OWNED={'PLAN.md','docs/josev2b-task-quality-native-v1.json','docs/josev2b-task-quality-native-v1.md',
 'tools/josev2b_task_quality_native_v1.py','tools/ci-heavy/josev2b_task_quality_native_v1.py',
 'tools/josev2b_task_quality_native_panel_v1.py','tools/josev-new20-owner/provider.py',
 'tools/ci-heavy/josev2b_task_quality_owner_v1.py','tests/test_josev2b_task_quality_native_v1.py',
 '.github/workflows/josev-new20-native-v1.yml','tools/ci_process.py','tools/josev-new20-owner/requirements.lock'}

def config():
 d=json.loads(PATH.read_text())
 require(d['schema']=='crispembed-josev-new20-native-owner-v1'and d['base']==BASE and d['baseline_runtime_source']==RUNTIME and
  d['original_source']==ORIGINAL and d['mixed_development_protocol']=={'source':MIXED_SOURCE,'path':'docs/josev2b-mixed-recipes-v1.json','sha256':MIXED_SHA}and
  d['native_child_seconds']==14400 and d['workflow_minutes']==360 and d['initial_free_disk_bytes']==64*2**30 and
  d['minimum_available_ram_bytes']==14*2**30 and d['threads']==2 and d['publication_authorized']is False and d['heldout_reselection_allowed']is False and d['selection_rule']==SELECTION_RULE and
  set(d['baseline_runtime_git_objects'])=={'src','ggml','CMakeLists.txt','python/crispembed','tests','cmake','examples','tools','models','.gitmodules','VERSION'},'New native scope/resource declaration differs')
 return d

def source_guard():
 d=config();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Dirty NEW20 native declaration')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared NEW20 owner delta')
 old=declaration();required=set(old['source_sha256'])|{'docs/josev2b-task-quality-v1.json'}
 require(set(d['protected_sha256'])==required and set(d['owned_sha256'])==OWNED-{'PLAN.md',str(PATH.relative_to(ROOT))},'Complete frozen/new native closure required')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in d[group].items():
   path=ROOT/name;require(path.is_file()and not path.is_symlink()and sha256(path)==pin and git('show','HEAD:'+name)==path.read_bytes(),'Native owner source differs: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==path.read_bytes(),'Original/native dependency changed: '+name)
 mixed=git('show',MIXED_SOURCE+':docs/josev2b-mixed-recipes-v1.json')
 require(hashlib.sha256(mixed).hexdigest()==MIXED_SHA,'Pre-output mixed recipe freeze differs')
 for source,target in [('tools/ci_process.py','tools/ci_process.py'),('tools/kaggle/josev-q4k/requirements.lock','tools/josev-new20-owner/requirements.lock')]:
  require(git('show',RUNTIME+':'+source)==(ROOT/target).read_bytes(),'Exact audited process/NumPylock source differs')
 panel()
 return dict(source=git('rev-parse','HEAD').decode().strip(),protocol_sha256=sha256(PATH),original_source=ORIGINAL,mixed_source=MIXED_SOURCE,mixed_protocol_sha256=MIXED_SHA)

def selected_candidate(d=None):
 d=config()if d is None else d;s=d['selected_candidate']
 require(s['reviewed']is True and s['recipe']in('A','B','C')and s['selection_rule']==d['selection_rule']and
  s['old_four_development_criteria_passed']is True,'Reviewed development-selected mixed candidate pending; no API/payload/model access')
 fields=('service_source','checkout_source','report_sha256','model_sha256','protocol_sha256')
 require(all(isinstance(s.get(k),str)and re.fullmatch('[0-9a-f]{40}'if k.endswith('source')else'[0-9a-f]{64}',s[k])for k in fields),'Exact candidate source/report/model pins pending')
 sizes={'A':3613774304,'B':4651864544,'C':5375381984}
 require(s['model_bytes']==sizes[s['recipe']]and
  s['model_sha256']!='717f0283d9839d4b277e0591f50c7c3b75bbecad3c1cb7d3ecaeda99213640ef','Old all150 candidate forbidden for NEW20')
 for key,cap in [('review_artifact',16*2**20),('raw_artifact',20*2**30)]:
  a=s[key];require(type(a.get('run'))is int and type(a.get('id'))is int and 0<a.get('bytes',0)<=cap and
   isinstance(a.get('sha256'),str)and re.fullmatch('[0-9a-f]{64}',a['sha256']) and isinstance(a.get('member'),str)and a['member'],'Exact bounded candidate artifact/member pins pending')
 require(s['review_artifact']['run']==s['raw_artifact']['run']and isinstance(s['workflow'],str)and s['workflow'].startswith('.github/workflows/'),'Candidate artifact service identity differs')
 actual=subprocess.check_output(['git','show',s['checkout_source']+':docs/josev2b-mixed-recipes-v1.json'],cwd=ROOT)
 require(hashlib.sha256(actual).hexdigest()==s['protocol_sha256'],'Reviewed actual development producer protocol differs')
 initial=json.loads(subprocess.check_output(['git','show',MIXED_SOURCE+':docs/josev2b-mixed-recipes-v1.json'],cwd=ROOT));actual=json.loads(actual)
 for key in ['runtime_source','ggml','source_model','recipes','gates','original_reference_sha256','original_freeze_sha256']:
  require(actual[key]==initial[key],'Pre-output recipe definition changed: '+key)
 return s

def runtime_prerequisites(d=None):
 d=config()if d is None else d;r=d['runtime_activation']
 require(r['reviewed']is True and isinstance(r['source'],str)and re.fullmatch('[0-9a-f]{40}',r['source'])and
  set(r['git_objects'])==set(d['baseline_runtime_git_objects'])and isinstance(r['delta_review_sha256'],str)and re.fullmatch('[0-9a-f]{64}',r['delta_review_sha256']),'Reviewed exact metadata-aware runtime successor pending')
 names=subprocess.check_output(['git','diff','--name-only',RUNTIME,r['source']],cwd=ROOT,text=True).splitlines()
 require(set(names)==set(r['allowed_changed_files']),'Exact reviewed metadata admission source delta differs')
 for name,pin in r['allowed_changed_files'].items():
  for source,key in [(RUNTIME,'before_sha256'),(r['source'],'after_sha256')]:
   blob=subprocess.run(['git','show',source+':'+name],cwd=ROOT,capture_output=True)
   actual=hashlib.sha256(blob.stdout).hexdigest()if blob.returncode==0 else None
   require(actual==pin[key],'Exact metadata admission file hash differs: '+name)
 for name,pin in r['git_objects'].items():
  require(subprocess.check_output(['git','rev-parse',r['source']+':'+name],cwd=ROOT,text=True).strip()==pin,'Reviewed runtime object differs: '+name)
 return r

def canonical_prerequisites(d=None):
 d=config()if d is None else d;m=d['canonicalization']
 # The canonicalizer and same-payload report schemas are being frozen in a
 # separately owned lane. This checkpoint cannot infer/activate those proofs.
 require(m['requires_followup_source_version']is False and m['proof_reviewed']is True and
  m['same_payload_control_reviewed']is True and isinstance(m['canonical_model_sha256'],str)and
  re.fullmatch('[0-9a-f]{64}',m['canonical_model_sha256']),'Canonicalization/same-payload proof schema and source adapter pending')
 return m

def native_prerequisites():
 d=config();s=selected_candidate(d);r=runtime_prerequisites(d);m=canonical_prerequisites(d);p=d['original']
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Native NEW20 execution is hosted only')
 return dict(reviewed_original20=True,reviewed_current_q4_four_case=True,runtime_source=r['source'],
  runtime_protected_git_objects=r['git_objects'],original_source=ORIGINAL,
  original_reference_sha256=p['reports']['reference.json']['sha256'],original_freeze_sha256=p['reports']['input-freeze.json']['sha256'],
  original_raw_manifest_sha256=p['reports']['raw-manifest.json']['sha256'],original_protocol_sha256=p['protocol_sha256'],
  candidate_report_sha256=s['report_sha256'],f32_model_sha256=d['f32_model']['sha256'],q4_model_sha256=m['canonical_model_sha256'],
  f32_profile='official-pytorch-text-batch1-512-no-extra-temperature',q4_profile='official-pytorch-text-batch1-512-no-extra-temperature')
