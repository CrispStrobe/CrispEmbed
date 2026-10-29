"""NEW native owner prerequisites; original110 contract/source remain immutable."""
import hashlib,json,os,re,subprocess
from pathlib import Path
from josev2b_task_quality_v1 import (declaration,panel,protocol,expected_stages,require,sha256,compare_public)
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/josev2b-task-quality-native-v1.json'
BASE='6af6865598c592b1d86ffc930c9054f643b15064'
ORIGINAL='110620ea1ada0748007e6b6698e0c5babb4940ed'
RUNTIME='9c475d0dac70264adc123a63b6c1827b4b7280d5'
METADATA_RUNTIME='56ccd91673a7989e13ef409c4874fac47c185b51'
METADATA_PROFILE_SHA='5170e09b51bf54d708690020740bb00c294891ad0f53566c1d26e441db489d14'
BASELINE_633='633f435bdee220a7f424a9cfcf54847f5bf20878'
MOE_DELTA={'src/moedern_family_engine.cpp','src/moedern_family_engine.h','src/moedern_q4k_profile.h','src/moedern_text.cpp','src/moedern_text_frontend.cpp','src/moedern_text_frontend.h'}
METADATA_DELTA={'src/josev2b_packed_profile.h','src/moedern_q4k_profile.h','src/ultrabert_profile.h','tools/quantize.cpp','tools/quantize_josev2b_q4k.py','tools/quantize_josev2b_mixed.py','tools/moedern-colbert/quantize_q4k_padded.py'}
MIXED_SOURCE='54c2b5767ae82c853d8649870abd234a2fbe7ee5'
MIXED_SHA='f6f7b694f5080e1cf59fefc84400abe93b76b92664187f7b251ed0b354075e12'
SELECTION_RULE='strictF32control+structural/criticalbytes+alloldargmax+eachpublicprobability.01+eacholdordinal.001; smallestbytes'
OWNED={'PLAN.md','docs/josev2b-task-quality-native-v1.json','docs/josev2b-task-quality-native-v1.md',
 'tools/josev2b_task_quality_native_v1.py','tools/ci-heavy/josev2b_task_quality_native_v1.py',
 'tools/josev2b_task_quality_native_panel_v1.py','tools/josev-new20-owner/provider.py','tools/josev-new20-owner/canonical.py',
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
 runtime_source_proof(d)
 panel()
 return dict(source=git('rev-parse','HEAD').decode().strip(),protocol_sha256=sha256(PATH),original_source=ORIGINAL,mixed_source=MIXED_SOURCE,mixed_protocol_sha256=MIXED_SHA)

def selected_candidate(d=None):
 d=config()if d is None else d;s=d['selected_candidate']
 require(s['reviewed']is True and s['recipe']in('A','B','C')and s['selection_rule']==d['selection_rule']and
  s['old_four_development_criteria_passed']is True and s['model_identity']=='legacy-development-only','Reviewed development-selected mixed candidate pending; no API/payload/model access')
 fields=('service_source','checkout_source','report_sha256','legacy_model_sha256','protocol_sha256')
 require(all(isinstance(s.get(k),str)and re.fullmatch('[0-9a-f]{40}'if k.endswith('source')else'[0-9a-f]{64}',s[k])for k in fields),'Exact candidate source/report/model pins pending')
 sizes={'A':3613774304,'B':4651864544,'C':5375381984}
 require(s['legacy_model_bytes']==sizes[s['recipe']]and
  s['legacy_model_sha256']!='717f0283d9839d4b277e0591f50c7c3b75bbecad3c1cb7d3ecaeda99213640ef','Old all150 candidate forbidden for NEW20')
 for key,cap in [('review_artifact',16*2**20),('raw_artifact',20*2**30)]:
  a=s[key];require(type(a.get('run'))is int and type(a.get('id'))is int and 0<a.get('bytes',0)<=cap and
   isinstance(a.get('sha256'),str)and re.fullmatch('[0-9a-f]{64}',a['sha256']) and isinstance(a.get('member'),str)and a['member'],'Exact bounded candidate artifact/member pins pending')
 require(s['review_artifact']['run']==s['raw_artifact']['run']and isinstance(s['workflow'],str)and s['workflow'].startswith('.github/workflows/')and type(s['run_attempt'])is int and s['run_attempt']>=1,'Candidate artifact service identity differs')
 actual=subprocess.check_output(['git','show',s['checkout_source']+':docs/josev2b-mixed-recipes-v1.json'],cwd=ROOT)
 require(hashlib.sha256(actual).hexdigest()==s['protocol_sha256'],'Reviewed actual development producer protocol differs')
 initial=json.loads(subprocess.check_output(['git','show',MIXED_SOURCE+':docs/josev2b-mixed-recipes-v1.json'],cwd=ROOT));actual=json.loads(actual)
 for key in ['runtime_source','ggml','source_model','recipes','gates','original_reference_sha256','original_freeze_sha256']:
  require(actual[key]==initial[key],'Pre-output recipe definition changed: '+key)
 return s

def runtime_source_proof(d=None):
 d=config()if d is None else d;r=d['runtime_activation'];git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(r['source']==METADATA_RUNTIME and r['source_profile']=={'path':'docs/gguf-standard-file-type-v1.json','sha256':METADATA_PROFILE_SHA},'Exact metadata-aware source/profile required')
 profile=git('show',METADATA_RUNTIME+':'+r['source_profile']['path'])
 require(hashlib.sha256(profile).hexdigest()==METADATA_PROFILE_SHA,'Frozen metadata source profile differs');profile=json.loads(profile)
 baseline=r['baseline_633'];require(baseline['source']==BASELINE_633 and baseline['src_tree']=='0c4341808c7136b84f0085da85e8c22b951c27a1','Exact633 source baseline required')
 require(git('rev-parse',BASELINE_633+':src').decode().strip()==baseline['src_tree'],'633 source object differs')
 require(set(git('diff','--name-only',RUNTIME,BASELINE_633,'--','src').decode().splitlines())==MOE_DELTA and set(baseline['preserved_moedern_delta'])==MOE_DELTA,'Exact six preserved MoE baseline changes required')
 for name,pin in baseline['preserved_moedern_delta'].items():
  for source,key in [(RUNTIME,'before_sha256'),(BASELINE_633,'baseline_sha256')]:
   blob=subprocess.run(['git','show',source+':'+name],cwd=ROOT,capture_output=True)
   require((hashlib.sha256(blob.stdout).hexdigest()if blob.returncode==0 else None)==pin[key],'Preserved MoE baseline file differs: '+name)
 require(set(r['metadata_admission_delta'])==METADATA_DELTA,'Exact seven metadata/tool changes required')
 for name,pin in r['metadata_admission_delta'].items():
  require(pin=={'before_sha256':profile['existing_delta'][name]['base_sha256'],'after_sha256':profile['existing_delta'][name]['sha256']},'Declared metadata delta differs: '+name)
  for source,key in [(BASELINE_633,'before_sha256'),(METADATA_RUNTIME,'after_sha256')]:
   require(hashlib.sha256(git('show',source+':'+name)).hexdigest()==pin[key],'Metadata admission file differs: '+name)
 header='src/core/gguf_file_type.h'
 require(r['new_metadata_runtime_header']=={header:profile['owned_sha256'][header]}and hashlib.sha256(git('show',METADATA_RUNTIME+':'+header)).hexdigest()==r['new_metadata_runtime_header'][header],'New standard metadata header differs')
 allowed=MOE_DELTA|{n for n in METADATA_DELTA if n.startswith('src/')}|{header}
 require(set(git('diff','--name-only',RUNTIME,METADATA_RUNTIME,'--','src').decode().splitlines())==allowed,'Josev/Qwen/core source math drift outside exact baseline/metadata changes')
 require(not git('diff','--name-only',RUNTIME,METADATA_RUNTIME,'--','CMakeLists.txt','cmake','python','crispembed','crispembed-sys','flutter','Cargo.toml','Cargo.lock'),'Linked CMake/bindings changed from9c')
 require(set(r['git_objects'])==set(d['baseline_runtime_git_objects']),'Complete actual successor source/build objects required')
 for name,pin in r['git_objects'].items():require(git('rev-parse',METADATA_RUNTIME+':'+name).decode().strip()==pin,'Actual56 object differs: '+name)
 paths=git('ls-tree','-r','--name-only',RUNTIME,'src').decode().splitlines()
 linked={n for n in paths if n.startswith(('src/josev2b_','src/qwen35_','src/core/'))and n not in allowed}
 require(r['linked_josev_math_unchanged_from']==RUNTIME and set(r['linked_josev_math_paths'])==linked,'Exact unchanged Josev/Qwen/core math closure required')
 for name,pin in r['linked_josev_math_paths'].items():
  require(name.startswith('src/')and name not in allowed and git('rev-parse',RUNTIME+':'+name).decode().strip()==pin==git('rev-parse',METADATA_RUNTIME+':'+name).decode().strip(),'Linked Josev math object changed: '+name)
 helper=r['canonicalizer_helper'];require(helper['path']=='tools/gguf_file_type_canonicalize.py'and helper['sha256']==profile['owned_sha256'][helper['path']]and hashlib.sha256(git('show',METADATA_RUNTIME+':'+helper['path'])).hexdigest()==helper['sha256'],'Frozen canonical parser helper differs')
 require(r['git_objects']['src']=='4847b2e679dcd0e0e7a82bcec9d26c9ae3108358'and r['git_objects']['ggml']==d['baseline_runtime_git_objects']['ggml'],'Actual56 SRC/GGML object required')
 return r

def runtime_prerequisites(d=None):
 d=config()if d is None else d;r=d['runtime_activation']
 require(r['reviewed']is True and isinstance(r['delta_review_sha256'],str)and re.fullmatch('[0-9a-f]{64}',r['delta_review_sha256']),'Reviewed exact metadata-aware runtime successor pending')
 return runtime_source_proof(d)

def canonical_prerequisites(d=None):
 d=config()if d is None else d;m=d['canonicalization']
 require(m['requires_followup_source_version']is False and m['proof_reviewed']is True and m['same_payload_control_reviewed']is True,'Reviewed canonicalization/same-payload evidence pending')
 require(m['runtime_source']==METADATA_RUNTIME and m['source_profile_sha256']==METADATA_PROFILE_SHA and
  m['proof']['schema']=='crispembed-gguf-file-type-canonicalization-v1'and m['same_payload_control']['schema']=='crispembed-gguf-file-type-same-payload-v1','Exact canonical source/profile/schema required')
 s=selected_candidate(d)
 require(m['legacy_model']=={'bytes':s['legacy_model_bytes'],'sha256':s['legacy_model_sha256']},'Legacy development model identity differs')
 require(type(m['canonical_model']['bytes'])is int and m['canonical_model']['bytes']==s['legacy_model_bytes']and
  isinstance(m['canonical_model']['sha256'],str)and re.fullmatch('[0-9a-f]{64}',m['canonical_model']['sha256'])and m['canonical_model']['sha256']!=s['legacy_model_sha256'],'Distinct same-size canonical model pins pending')
 service=m['bridge_service'];require(service['checkout_source']==METADATA_RUNTIME and re.fullmatch('[0-9a-f]{40}',service.get('service_source')or'')and isinstance(service.get('workflow'),str)and service['workflow'].startswith('.github/workflows/')and type(service.get('run_attempt'))is int and service['run_attempt']>=1,'Exact reviewed bridge service pins pending')
 for key in ['descriptor','proof','same_payload_control']:
  row=m[key];require(type(row.get('bytes'))is int and 0<row['bytes']<=8*2**20 and re.fullmatch('[0-9a-f]{64}',row.get('sha256')or'')and isinstance(row.get('member'),str)and row['member'],'Exact small canonical evidence member pins pending')
 require(re.fullmatch('[0-9a-f]{64}',m['descriptor'].get('canonical_json_sha256')or''),'Canonical descriptor semantic digest pending')
 for key,cap in [('review_artifact',16*2**20),('raw_artifact',20*2**30)]:
  a=m[key];require(type(a.get('run'))is int and type(a.get('id'))is int and type(a.get('bytes'))is int and 0<a['bytes']<=cap and re.fullmatch('[0-9a-f]{64}',a.get('sha256')or''),'Exact canonical artifact pins pending')
 require(m['review_artifact']['run']==m['raw_artifact']['run']and isinstance(m['raw_artifact'].get('member'),str)and m['raw_artifact']['member'],'Canonical service/member differs')
 return m

def native_prerequisites():
 d=config();s=selected_candidate(d);r=runtime_prerequisites(d);m=canonical_prerequisites(d);p=d['original']
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Native NEW20 execution is hosted only')
 return dict(reviewed_original20=True,reviewed_current_q4_four_case=True,runtime_source=r['source'],
  runtime_protected_git_objects=r['git_objects'],original_source=ORIGINAL,
  original_reference_sha256=p['reports']['reference.json']['sha256'],original_freeze_sha256=p['reports']['input-freeze.json']['sha256'],
  original_raw_manifest_sha256=p['reports']['raw-manifest.json']['sha256'],original_protocol_sha256=p['protocol_sha256'],
  candidate_report_sha256=s['report_sha256'],f32_model_sha256=d['f32_model']['sha256'],q4_model_sha256=m['canonical_model']['sha256'],
  f32_profile='official-pytorch-text-batch1-512-no-extra-temperature',q4_profile='official-pytorch-text-batch1-512-no-extra-temperature')
