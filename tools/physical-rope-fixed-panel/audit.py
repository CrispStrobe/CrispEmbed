#!/usr/bin/env python3
"""Independent artifact-only inventory, raw-row and decoded-score audit; no inference."""
import argparse,hashlib,importlib.util,json,math,signal,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];VOCAB=248320
PROTOCOL_SHA='78afbb0c0480ed1c9dbf8b9ba0600026238948f02cf359321bca49e802a447fa'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def owned(root,name):
 n=Path(name);require(not n.is_absolute()and '..'not in n.parts,'closed relative artifact path');p=Path(root)/n;require(p.is_file()and not p.is_symlink()and p.resolve().is_relative_to(Path(root).resolve()),'owned regular artifact');return p
def raw_archive(root,record,ids,known):
 a=record['archive'];p=owned(root,a['file']);require(p.stat().st_size==a['bytes']and known[p.resolve()]==a['sha256'],'actual whole archived bytes');rows=record['rows'];require(len(rows)==len(ids)and 1<=len(ids)<=1536,'raw row count');chain=hashlib.sha256()
 with zipfile.ZipFile(p)as z:
  names=[f'greedy.{i}.logits.f32'for i in range(len(ids))];require(len(z.infolist())==len(names)and set(z.namelist())==set(names),'closed raw ZIP members')
  for i,(name,row,token)in enumerate(zip(names,rows,ids)):
   require(row['position']==i and row['bytes']==VOCAB*4 and row['argmax']==token and type(token)is int and 0<=token<VOCAB,'frozen actual row identity');info=z.getinfo(name);require(info.file_size==VOCAB*4 and info.compress_type==zipfile.ZIP_STORED,'bounded lossless actual vocabulary member');b=z.read(name);require(len(b)==VOCAB*4 and hashlib.sha256(b).hexdigest()==row['sha256'],'all archived row SHA');x=np.frombuffer(b,dtype='<f4');require(np.isfinite(x).all()and int(x.argmax())==token,'all finite actual greedy argmaxes');chain.update(bytes.fromhex(row['sha256']))
 return dict(rows=len(ids),archive_sha256=a['sha256'],ordered_row_sha256=chain.hexdigest())
def edit_count(a,b):
 if not a:return len(b)
 mask=(1<<len(a))-1;high=1<<(len(a)-1);peq={}
 for i,item in enumerate(a):peq[item]=peq.get(item,0)|(1<<i)
 pv=mask;mv=0;score=len(a)
 for item in b:
  eq=peq.get(item,0);xv=eq|mv;xh=(((eq&pv)+pv)^pv)|eq;ph=mv|~(xh|pv);mh=pv&xh;score+=bool(ph&high)-bool(mh&high);ph=((ph<<1)|1)&mask;mh=(mh<<1)&mask;pv=(mh|~(xv|ph))&mask;mv=ph&xv
 return score
def independent_edits(gold,text,score):
 result={}
 for view,reference,actual in [('raw',gold['text'],text),('content',' '.join(gold['text'].split()),score.get('parsed_text'))]:
  if actual is None:result[view]=None;continue
  ce=edit_count(reference,actual);we=edit_count(reference.split(),actual.split());require(ce/max(1,len(reference))==score[view]['cer']and we/max(1,len(reference.split()))==score[view]['wer'],'independent bit-parallel character/word distances');result[view]=dict(character_edits=ce,reference_codepoints=len(reference),word_edits=we,reference_words=len(reference.split()))
 return result
def compare_derived(actual,expected):
 require(set(actual)==set(expected),'same derived metric fields');maximum=0.
 for key,value in actual.items():
  reference=expected[key]
  if key=='rms'or key.endswith('_rms_magnitude'):
   require(math.isfinite(value)and math.isfinite(reference),'finite derived RMS');unit=max(math.ulp(value),math.ulp(reference));distance=abs(value-reference)/unit;require(distance<=8,'predeclared eight-ULP RMS recomputation bound');maximum=max(maximum,distance)
  else:require(value==reference,'exact maxima/identity fields')
 return maximum
def safe_score(score):
 return {k:v for k,v in score.items()if k in ['policy','complete','raw','characters','lines','parse_ok','content','table_gold_available','content_exact','table','exact_quality','expected_table_shape','table_shape_valid']}
def actual_summary(s,ids,cap,expected_n=None,expected_eos=None):
 require(s['generated_tokens']==len(ids)and 1<=len(ids)<=cap and s['cap']==cap and s['eos']==(ids[-1]in[248044,248046])and(s['eos']or len(ids)==cap),'actual count/EOS/cap');require(not any(token in[248044,248046]for token in ids[:-1]),'no ignored earlier EOS row')
 if expected_n is not None:require(len(ids)==expected_n and s['eos']is expected_eos,'historical exact count/stop')
 require(s['upstream_physical']and not s['source_fed_vision']and s['teacher_positions']==1 and s['physical_source_prefill_chunks']and s['physical_q6_tiled']and s['physical_q4_repack'],'actual pixel/planner/quantized mode');require(s['physical_repack_entries']==216 and 0<s['physical_repack_bytes']<=2<<30 and s['converted_f32_weight_entries']==s['converted_f32_weight_bytes']==0,'bounded quantized-only cache');require(1<=s['prompt_tokens']<=512 and 0<s['image_tokens']<=256 and s['prompt_tokens']+cap<=2048,'actual context/image bounds')

OBSERVED_SHA='aaf5298b944cea3d53d7e6e4ce88da78c58c3cec6a651733c5aff993fa4c6ffc'
PREFLIGHT_PINS={'tests/lighton4b_native_case_preflight.cpp':'2cf633f776a5627512ae557bd1cb7a9e7bc684d80096ae017e15f43e5c2a9ddd','src/lighton4b_tokenizer.cpp':'38b0e336aa058cf984a078fe901dee1214cd57d43f28bb96b8d07869b370cc93','CMakeLists.txt':'8eaa99c62356e8b6d9844d4f02fe1cacee63ae921ed5a1d48eea3885f414dbde'}
FLAGS={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
def manifest(root,rows):
 known={}
 for name,row in rows.items():
  p=owned(root,name);require(type(row['bytes'])is int and row['bytes']>=0 and p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'all owned artifact sizes/SHA');known[p.resolve()]=row['sha256']
 return known

def control_manifest(root,rows,candidate):
 name='physical-rope-bootstrap.json'
 pending={'bytes':312,'sha256':'db1fa9c9a743752a988b2f9b5f8aaa7f79c8f6caac15232698570bc6bb8d0829'}
 final={'bytes':355,'sha256':'5c17c09753a65764d2779d8b83ed4a18040569339d9b242ba6ba71131e889c22'}
 require(rows[name]==pending,'one exact recorded intermediate control-bootstrap snapshot')
 path=owned(root,name);require(path.stat().st_size==final['bytes']and sha(path)==final['sha256'],'one exact retained finalized control bootstrap');b=json.loads(path.read_text())
 require(b['status']=='complete'and b['candidate_source']==candidate and not any(b[k]for k in ['model_forward_executed','source_capture_repeated','full_f32_model_instantiated','qualified']),'exact completed model-free control wrapper')
 known=manifest(root,{n:row for n,row in rows.items()if n!=name})
 return known,dict(file=name,recorded_intermediate=pending,retained_final=final,retained_final_independently_rehashed=True,recorded_intermediate_bytes_available=False,recorded_intermediate_independently_rehashed=False,scope='The controls worker inventoried its outer bootstrap before finalization. The original intermediate snapshot bytes were not retained. Only this exact journal collision is admitted; the finalized wrapper, all other control artifacts and actual tested runtime bundle are independently rehashed.')

def admit_complete(r,cfg):
 require(r['schema']==cfg['schema']and r['status']=='measured_physical_rope_fixed_panel'and r['candidate_source']==cfg['candidate_source']and r['controller_source']==cfg['controller_source'],'complete exact candidate/controller producer')
 require(r['qualified']is False and r['references_expert_adjudicated']is False and r['reference_forward_executed']is False and r['source_feature_forward_executed']is False and r['heldout_forward_executed']is True,'complete native-only provisional diagnostic')
 require(list(r['cases'])==cfg['case_order']and [name+'/'+arm for name,c in r['cases'].items()for arm in c['arms']]==cfg['arm_order'],'all six arms in frozen book-first order')
 for name,c in r['cases'].items():
  require(set(c['arms'])=={'off','on'}and c['off_all_historical_rows_stages_stop_exact'],'complete paired inventory without waived OFF failure')
  for arm in ['off','on']:
   label=name+'-'+arm;command=r['commands'][label];require(command['status']=='completed'and command['returncode']==0 and r['admissions'][label]['accepted'],'actual completed admitted arm')
 require(r['observed_receipt_sha256']==OBSERVED_SHA and r['observed_controls_reused_without_forward']and r['all_three_historical_inputs_baselines_audits_authenticated_before_models'],'authenticated original sources and observed reuse')

def audit(receipt,digest,bootstrap,bootstrap_digest):
 require(np.__version__=='2.3.5','pinned arrays');require(sha(receipt)==digest and len(digest)==64 and sha(bootstrap)==bootstrap_digest and len(bootstrap_digest)==64,'immutable completed receipt/bootstrap SHA')
 root=receipt.parent;r=json.loads(receipt.read_text());cfgp=Path(__file__).with_name('protocol.json');require(sha(cfgp)==PROTOCOL_SHA,'exact frozen fixed protocol');cfg=json.loads(cfgp.read_text());admit_complete(r,cfg)
 b=json.loads(bootstrap.read_text());require(b['status']=='complete'and b['actual_model_arms_completed']==6,'complete six-arm outer session');probe=b['metadata_probe'];require(probe['source']==cfg['candidate_source']==r['preflight_candidate_source']and probe['source_files']==PREFLIGHT_PINS and probe['binary_sha256']==r['preflight_binary_sha256']and probe['only_requested_target']=='lighton4b-native-case-preflight','pinned independent metadata preflight provenance')
 cache=owned(root,'preflight-CMakeCache.txt');require(sha(cache)==probe['cmake_cache_sha256'],'actual retained metadata CMake cache');lines=cache.read_text().splitlines()
 for key,value in FLAGS.items():require(probe['flags'][key]==value and key+':BOOL='+value in lines,'actual metadata profile')
 for name,h in {'tools/heldout/execution.py':'82f5adc352c79e0779478261ce2a56e90e0df05bdca2d232386527b4a440ee06','tools/heldout/lighton4b_prepare.py':'6eef63020be662caa87b6a904df4f0ae7e1fcdb21caa605d8ecc056bbcfc0891'}.items():require(sha(ROOT/name)==h,'immutable audit/source scoring helpers')
 sys.path.insert(0,str(ROOT/'tools/heldout'));import execution as ex
 selectionp=ROOT/'docs/lighton4b-heldout-v1.json';require(sha(selectionp)==cfg['selection_protocol_sha256']and sha(ROOT/'docs/lighton4b-heldout-executor-v1.json')==cfg['frozen_executor_sha256'],'frozen source selection/executor');selection=json.loads(selectionp.read_text());ex.prepare.admit(selection)
 for name,h in selection['reference']['scorer'].items():require(sha(ROOT/name)==h,'frozen scorer')
 require(sha(ROOT/'tests/regression/images/cc0/ground_truth.json')==selection['reference']['gold_registry_sha256'],'frozen registry')
 def locate(name,h):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==h];require(len(found)==1,'one exact immutable prerequisite');return found[0],json.loads(found[0].read_text())
 known=manifest(root,r['artifacts']);cp,control=locate('physical-rope-controls.json',cfg['candidate_controls_sha256']);require(r['controls_sha256']==sha(cp)and control['passed']and control['all_source_stages_exact']and control['actual_cache_toggle_recovery_pass']and control['candidate_source']==cfg['candidate_source']and not control['checkpoint_loaded']and not control['model_forward_executed'],'exact actual supported component controls');require(len(control['unit_controls'])==8 and all('pass'in v.lower()and'skip'not in v.lower()for v in control['unit_controls'].values()),'eight actual supported/guard controls')
 require(r['runtime_bundle']==control['runtime_bundle'],'exact tested runtime identity');control_known,control_journal=control_manifest(cp.parent,control['artifacts'],cfg['candidate_source'])
 for name,row in control['runtime_bundle'].items():require(control_known[owned(cp.parent/'runtime',name).resolve()]==row['sha256'],'rehash actual tested executable/libs')
 op,observed=locate('physical-rope-observed-ab.json',OBSERVED_SHA);require(observed['status']=='measured_physical_rope_observed_ab'and observed['runtime_producer_source']==cfg['candidate_source']and observed['candidate_controls_sha256']==sha(cp)and observed['runtime_bundle']==control['runtime_bundle'],'same-runtime complete observed six-arm screen');observed_known=manifest(op.parent,observed['artifacts']);observed_rows=0
 for name,c in observed['cases'].items():
  require(c['off_all_rows_stages_exact']and set(c['arms'])=={'off','on'},'all observed pairs')
  for arm,v in c['arms'].items():
   dest=op.parent/name/arm;ids=list(map(int,owned(dest,'generated.ids').read_text().split()));s=json.loads(owned(dest,'native-summary.json').read_text());text=owned(dest,'generated.txt').read_bytes().decode();actual_summary(s,ids,512);require(ids==v['generated_ids']and text==v['decoded']and s==v['summary']and s['physical_rope']==(arm=='on')and s['eos'],'actual observed bytes/stops/gates')
   if arm=='on':require(v['meets_observed_upstream_floor'],'original observed source floor unchanged')
   for i,token in enumerate(ids):
    p=owned(dest,f'greedy.{i}.logits.f32');require(p.stat().st_size==VOCAB*4,'observed full vocabulary row');x=np.fromfile(p,dtype='<f4');require(np.isfinite(x).all()and int(x.argmax())==token,'all observed actual finite greedy argmaxes');observed_rows+=1
 require(observed_rows==849,'all observed rows reused without inference')
 off,on=r['gates']['off'],r['gates']['on'];require(set(off)==set(on)and {k for k in off if off[k]!=on[k]}==set(cfg['changed_gates'])and off[cfg['changed_gates'][0]]=='0'and on[cfg['changed_gates'][0]]=='1'and off['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU']==on['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU']=='1','one frozen intervention with fused SwiGLU retained')
 for key,value in observed['cases']['simple_form']['arms']['off']['gates'].items():
  expected='1'if key in ['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT','CRISPEMBED_LIGHTON4B_FROZEN_CASES']else value
  require(off[key]==expected,'same observed gate tuple with explicit fixed-cap admissions')
 report=dict(schema='lighton4b-physical-rope-fixed-independent-raw-audit/v1',status='complete',receipt_sha256=digest,bootstrap_sha256=bootstrap_digest,candidate_source=cfg['candidate_source'],controller_source=cfg['controller_source'],owned_artifacts=len(known),owned_artifact_bytes=sum(row['bytes']for row in r['artifacts'].values()),all_owned_artifact_hashes=True,all_raw_finite_argmax_sha=True,observed_receipt_sha256=OBSERVED_SHA,observed_raw_rows_rehashed=observed_rows,observed_owned_artifacts_rehashed=len(observed_known),controls_sha256=sha(cp),controls_owned_artifacts_rehashed=len(control_known),controls_outer_bootstrap_retention=control_journal,tested_runtime_bundle_rehashed=True,metadata_preflight_binary_bytes_retained=False,metadata_preflight_binary_independently_rehashed=False,metadata_preflight_scope='Original metadata probe binary bytes were not retained. Its recorded hash/source-file pins, retained CMake profile and actual input seals are checked. Tested inference executable/libs are independently rehashed.',model_forward_executed=False,reference_forward_executed=False,source_feature_forward_executed=False,qualified=False,rights='Book text and token-ID sequences remain private; counts, metrics and hashes only.',derived_statistics_scope='Recomputed RMS/magnitudes use the existing at-most-eight double ULP portability bound. Raw SHA, argmax, stops, maximum errors and quality/floor fields remain exact.',cases={});total=0;baseline_rows=0
 for c in cfg['cases']:
  name=c['id'];v=r['cases'][name];folder=root/name;bp,baseline=locate(c['baseline_receipt'],c['baseline_sha256']);ap,prior_audit=locate(c['audit_receipt'],c['audit_sha256']);sp,original=locate('heldout-result.json',c['reference_sha256']);require(prior_audit['status']==c['audit_status']and prior_audit['receipt_sha256']==sha(bp)and prior_audit['all_owned_artifact_hashes']and prior_audit['all_raw_finite_argmax_sha']and name in prior_audit['cases'],'exact independently audited original baseline');oldcase=baseline['cases'][name];old=oldcase['arms']['on'];require(baseline['candidate_source']==cfg['prior_fused_candidate_source'],'frozen prior runtime');a=old['raw_rows']['archive'];oldzip=owned(bp.parent/c['baseline_prefix'],a['file']);require(sha(oldzip)==a['sha256'],'actual retained old raw archive');oldraw=raw_archive(bp.parent/c['baseline_prefix'],old['raw_rows'],old['ids'],{oldzip.resolve():a['sha256']});baseline_rows+=oldraw['rows']
  require(sha(owned(folder,'pre-forward-inputs.json'))==c['seal_sha256']==v['pre_forward_input_seal_sha256']==original['pre_forward_input_seal_sha256'],'actual same original input seal');selected=next(x for x in selection['cases']if x['id']==name)
  for rel,key in [('inputs/input.png','png_sha256'),('inputs/input.rgb','rgb_sha256')]:require(sha(owned(sp.parent,rel))==selected['prepared'][key]==original['artifacts'][rel]['sha256'],'actual original prepared pixels')
  require(sha(ROOT/selected['gold'])==selected['gold_sha256']and sha(ROOT/selected['source'])==selected['source_sha256'],'frozen selected source/gold');gold=json.loads((ROOT/selected['gold']).read_text());responses=[]
  for role in ['upstream-off','upstream-on']:
   rel=role+'/response.json';p=owned(sp.parent,rel);require(sha(p)==original['artifacts'][rel]['sha256'],'immutable actual source response');responses.append(json.loads(p.read_text()))
  ex.neutral(*responses);require(ex.source_response(responses[0],responses[0]['tokens_evaluated']),'complete neutral source');source_score=ex.score(gold,responses[0]['content'],True);require(source_score==v['reference_score'],'unchanged source score');sourcep=owned(sp.parent,'upstream-on/first-score.f32');require(sha(sourcep)==original['artifacts']['upstream-on/first-score.f32']['sha256'],'source full vocabulary hash');source=np.fromfile(sourcep,dtype='<f4').astype('f8');require(source.size==VOCAB and np.isfinite(source).all()and int(source.argmax())==responses[0]['tokens'][0],'actual source first-score finite argmax');feature=owned(sp.parent,'upstream-on/source.vision.0.f32');require(sha(feature)==original['artifacts']['upstream-on/source.vision.0.f32']['sha256'],'actual original merged feature');seal=json.loads(owned(folder,'pre-forward-inputs.json').read_text());pre=folder/'native-preflight'
  for file,key in [('prompt.ids','prompt_ids'),('modalities.ids','modalities'),('grid.ids','grid')]:require(list(map(int,owned(pre,file).read_text().split()))==seal[key],'actual preflight metadata seal')
  require(sha(owned(pre,'prefill.positions.f32'))==seal['positions_sha256'],'preflight position seal');out=dict(reference_score=safe_score(source_score),reference_independent_edits=independent_edits(gold,responses[0]['content'],source_score),baseline_raw_rows_rehashed=oldraw,historical_control_log_retention=prior_audit['model_free_command_log_retention'],arms={})
  for arm in ['off','on']:
   av=v['arms'][arm];dest=folder/arm;s=json.loads(owned(dest,'native-summary.json').read_text());ids=list(map(int,owned(dest,'generated.ids').read_text().split()));text=owned(dest,'generated.txt').read_bytes().decode();actual_summary(s,ids,1536,c['baseline_tokens']if arm=='off'else None,c['baseline_eos']if arm=='off'else None);require(s==av['summary']and ids==av['ids']and text==av['decoded']and s['physical_rope']==(arm=='on')and s['provisional_heldout_cap']==1536,'actual complete arm bytes/gate/stop');require(s['prompt_tokens']==seal['prompt_tokens']and s['image_tokens']==seal['image_tokens'],'actual sealed counts');require(list(map(int,owned(dest,'prompt.ids').read_text().split()))==seal['prompt_ids']and list(map(int,owned(dest,'modalities.ids').read_text().split()))==seal['modalities']and sha(owned(dest,'prefill.positions.f32'))==seal['positions_sha256'],'actual arm input seal');require(known[owned(dest,'vision.merged.f32').resolve()]==sha(feature),'actual source merged vision equality');raw=raw_archive(dest,av['raw_rows'],ids,known);total+=len(ids);first=owned(dest,'prefill.logits.f32');require(known[first.resolve()]==known[owned(dest,'teacher.0.logits.f32').resolve()]==av['raw_rows']['rows'][0]['sha256'],'actual first-row triple identity');scores=ex.score(gold,text,s['eos']);require(scores==av['score'],'independent unchanged score');floor=ex.floor(scores,source_score,name=='inventory-table');require(floor==av['meets_provisional_source_floor'],'unchanged floor including negative/capped arms');x=np.fromfile(first,dtype='<f4').astype('f8');d=x-source;num=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),source_rms_magnitude=float(np.sqrt(np.mean(source*source))));ulps=compare_derived(num,av['firstscore']);out['arms'][arm]=dict(tokens=len(ids),eos=s['eos'],score=safe_score(scores),independent_edits=independent_edits(gold,text,scores),source_floor=floor,raw=raw,firstscore=num,derived_rms_max_ulps_from_producer=ulps)
   if arm=='off':
    require(ids==old['ids']and text==old['decoded']and [x['sha256']for x in av['raw_rows']['rows']]==[x['sha256']for x in old['raw_rows']['rows']],'every historical OFF raw row/decoded output');require({k:s[k]for k in old['summary']}==old['summary']and set(s)==set(old['summary'])|{'physical_rope'},'historical OFF summary plus false new gate');prefix=c['baseline_prefix']+'/'
    for file,row in baseline['artifacts'].items():
     if file.startswith((prefix+'prefill.',prefix+'vision.',prefix+'processor.'))or file in[prefix+'prompt.ids',prefix+'modalities.ids',prefix+'text.embedded.f32']:
      require(sha(owned(bp.parent,file))==row['sha256']==known[owned(dest,file[len(prefix):]).resolve()],'all actual historical OFF stages')
   else:
    for p in(folder/'off').iterdir():
     if p.is_file()and(p.name.startswith(('vision.','processor.'))or p.name in['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32']or p.name in[f'prefill.layer.{i}.f32'for i in range(3)]):require(known[owned(dest,p.name).resolve()]==known[p.resolve()],'unchanged native inputs/vision/layers0to2')
  layers=[]
  for i in range(32):
   a=np.fromfile(owned(folder/'off',f'prefill.layer.{i}.f32'),dtype='<f4').astype('f8');b=np.fromfile(owned(folder/'on',f'prefill.layer.{i}.f32'),dtype='<f4').astype('f8');require(a.size and a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'paired finite full native layer');d=b-a;layers.append(dict(layer=i,bit_exact=bool(np.array_equal(a,b)),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),off_rms_magnitude=float(np.sqrt(np.mean(a*a))),on_rms_magnitude=float(np.sqrt(np.mean(b*b)))))
  require(len(v['paired_native_layers']['layers'])==32,'full paired-layer inventory');layer_ulps=max(compare_derived(a,b)for a,b in zip(layers,v['paired_native_layers']['layers']));first=next((x['layer']for x in layers if not x['bit_exact']),None);require(first==v['paired_native_layers']['first_different_layer'],'earliest actual paired boundary');out['paired_native_layers']=dict(first_different_layer=first,layers=layers,derived_rms_max_ulps_from_producer=layer_ulps,scope='Same-source native OFF/ON, not source hidden states.');report['cases'][name]=out
 report.update(raw_rows=total,baseline_raw_rows_rehashed=baseline_rows,all_on_complete=all(c['arms']['on']['eos']for c in report['cases'].values()),on_source_floors_passed=sum(c['arms']['on']['source_floor']for c in report['cases'].values()));require(baseline_rows==2249 and report['all_on_complete']==r['all_on_complete']and report['on_source_floors_passed']==r['on_source_floors_passed'],'actual totals without waived negative outcomes');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha',required=True);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--bootstrap-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.monotonic();r=dict(status='failed',model_forward_executed=False,qualified=False)
 def interrupted(sig,frame):raise TimeoutError('fixed artifact-only audit bound')
 for sig in[signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
 signal.alarm(2400)
 try:r=audit(a.receipt,a.receipt_sha,a.bootstrap,a.bootstrap_sha)
 except BaseException as error:r['error_type']=type(error).__name__;raise
 finally:signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;a.out.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
