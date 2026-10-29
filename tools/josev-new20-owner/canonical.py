"""Strict small canonical bridge evidence and hosted streamed model verification."""
import hashlib,json,math,os,re,subprocess,types
from pathlib import Path
from josev2b_task_quality_native_v1 import ROOT,METADATA_RUNTIME,METADATA_PROFILE_SHA,require,sha256

HEX=r'[0-9a-f]{64}'
def digest(value):return hashlib.sha256(value).hexdigest()
def flag(value,keys):
 for key in keys:require(value.get(key)is True,'Required canonical control rejected: '+key)
def false(value,keys):
 for key in keys:require(value.get(key)is False,'Canonicalization must not claim numerical/publication/original execution: '+key)
def sha(value):require(isinstance(value,str)and re.fullmatch(HEX,value),'Exact canonical digest required');return value

def reports(cfg,descriptor,proof,same,development):
 s=cfg['selected_candidate'];m=cfg['canonicalization'];legacy=m['legacy_model'];canonical=m['canonical_model']
 producer={'bytes':s['report_bytes'],'sha256':s['report_sha256']}
 require(descriptor['schema']=='crispembed-gguf-file-type-input-v1'and descriptor['reviewed']is True and descriptor['family']=='josev'and descriptor['recipe']==s['recipe']and descriptor['producer_source']==s['checkout_source'],'Canonical descriptor producer/recipe differs')
 require(descriptor['source_model']==legacy and descriptor['producer_report']==producer and descriptor['architecture']=='josev2b_systemone','Canonical descriptor model/report/architecture differs')
 require(descriptor['profile_markers'].get('josev2b.packed.format')=='crispembed-josev-packed-q4k-v1'and descriptor['profile_markers'].get('josev2b.packed.profile')=='official-pytorch-text-batch1-512-no-extra-temperature','Canonical packed profile markers differ')
 for key,path in [('report_model_sha256_path',['recipes',s['recipe'],'conversion','output_sha256']),('report_model_bytes_path',['recipes',s['recipe'],'conversion','output_bytes']),('report_recipe_path',['recipes',s['recipe'],'conversion','recipe'])]:
  require(descriptor[key]==path,'Exact mixed-recipe producer JSON path required')
 canonical_json=digest(json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode())
 require(canonical_json==m['descriptor']['canonical_json_sha256']==proof['input_descriptor_sha256'],'Canonical descriptor semantic digest differs')
 require(proof['schema']=='crispembed-gguf-file-type-canonicalization-v1'and proof['family']=='josev'and proof['recipe']==s['recipe']and proof['producer_source']==s['checkout_source']and proof['producer_report']==producer,'Canonical proof producer differs')
 require(proof['runtime_source']==METADATA_RUNTIME and proof['source_profile_sha256']==METADATA_PROFILE_SHA and proof['input_descriptor_file_sha256']==m['descriptor']['sha256'],'Canonical proof tool/profile/descriptor identity differs')
 require(proof['source_model']==legacy and proof['canonical_model']==canonical and legacy['bytes']==canonical['bytes']and legacy['sha256']!=canonical['sha256'],'Distinct byte-preserving model identities required')
 flag(proof,['completed','payloads_byte_exact','semantic_shapes_unchanged','source_preserved','metadata_only_bridge']);false(proof,['numerical_qualified','publication_qualified','original_rerun'])
 changes=proof['allowed_changes'];require(isinstance(changes,list)and len(changes)==1,'Exactly one metadata change required');change=changes[0]
 require(set(change)=={'key','kind','offset','old','new'}and change['key']=='general.file_type'and change['kind']=='UINT32'and change['old']==12 and change['new']==14 and type(change['offset'])is int and 24<=change['offset']<64*2**20,'Only exact four-byte12-to14 header repair permitted')
 for key in ['unchanged_prefix_sha256','unchanged_suffix_sha256','unchanged_metadata_sha256']:sha(proof[key])
 inventory=descriptor['inventory'];tensors=proof['tensors'];conversion=development['recipes'][s['recipe']]['conversion'];old=conversion['tensors']
 require(len(inventory)==len(tensors)==len(old)==442,'Complete442 tensor bridge inventory required')
 by_name=lambda rows:{r['name']:r for r in rows}
 source,declared,migrated=by_name(old),by_name(inventory),by_name(tensors)
 require(len(source)==len(declared)==len(migrated)==442 and set(source)==set(declared)==set(migrated),'Duplicate/unknown bridge tensor')
 types={'F32':0,'F16':1,'Q4_K':12};counts={k:0 for k in types}
 for name,a in source.items():
  e,t=declared[name],migrated[name];kind=types[a['type']];counts[a['type']]+=1
  require(e['name']==t['name']==name and e['type']==t['type']==kind and e['semantic_shape']==t['semantic_shape']==a['semantic_shape']and e['bytes']==t['bytes']==a['bytes'],'Tensor recipe/semantic/type/extent changed: '+name)
  require(e['stored_shape']==t['stored_shape']and type(t['storage_rank'])is int and t['storage_rank']==len(t['stored_shape'])==a['output_storage_rank'],'Physical storage rank changed: '+name)
  shape=t['stored_shape'];logical=t['semantic_shape'];require(1<=len(shape)<=4 and 1<=len(logical)<=4 and all(type(n)is int and 0<n<=2**31 for n in shape+logical),'Bounded original tensor axes required')
  require(shape+[1]*(4-len(shape))==logical+[1]*(4-len(logical)),'Josev semantic/storage shape changed: '+name)
  elements=math.prod(shape);size=elements*4 if kind==0 else elements*2 if kind==1 else elements//256*144
  require((kind!=12 or shape[0]%256==0)and t['bytes']==size and type(t['offset'])is int and 0<=t['offset']<canonical['bytes']and t['offset']%32==0,'Invalid aligned tensor storage: '+name)
  require(sha(e['sha256'])==sha(t['source_sha256'])==sha(t['canonical_sha256'])==sha(a['stored_sha256']),'Tensor payload differs from authenticated development model: '+name)
  if a['critical_f32']:require(kind==0 and a['stored_sha256']==a['source_f32_sha256'],'Critical F32 source byte identity changed: '+name)
 require(counts==conversion['counts']and sum(a['critical_f32']for a in old)==292,'Frozen recipe/critical tensor counts differ')
 require(same['schema']=='crispembed-gguf-file-type-same-payload-v1'and same['source']==METADATA_RUNTIME and same['runtime_source']==METADATA_RUNTIME and same['source_profile_sha256']==METADATA_PROFILE_SHA and same['canonicalization_report_sha256']==m['proof']['sha256']and same['source_model']==legacy and same['canonical_model']==canonical,'Same-payload exact source/proof/models differ')
 flag(same,['completed','all_mapped_stage_bytes_exact','all_typed_response_bytes_exact','all_public_capi_response_bytes_exact']);false(same,['numerical_qualified','publication_qualified','old_numeric_results_replaced'])
 require(same['f32']['compared_stages']==227 and same['f32']['strict_original_f32_numeric_and_typed_passed']is True,'Bridge strict original-four F32 control missing')
 for label in ['legacy','canonical']:
  flag(same[label],['structural_passed']);require(same[label]['compared_stages']==227,'Complete strict bridge panels required')
 require(type(same['legacy']['strict_original_numeric_passed'])is bool and same['legacy']['strict_original_numeric_passed']==same['canonical']['strict_original_numeric_passed'],'Old strict numerical flags changed by metadata')
 old_cases=development['recipes'][s['recipe']]['panel']['quant_task_decisions']['cases'];ids=[r['id']for r in old_cases]
 require(len(ids)==4 and len(set(ids))==4 and [r['id']for r in same['four_cases']]==ids,'Old-four case order/coverage differs')
 totals={mode:0 for mode in ['plain','observed','repeat']}
 for case in same['four_cases']:
  require([r['mode']for r in case['modes']]==list(totals),'All three same-payload modes required in order')
  for mode in case['modes']:
   flag(mode,['all_payload_sha_exact','response_bytes_exact']);require(type(mode['stages'])is int and mode['stages']>0,'Actual same-payload stage count required');totals[mode['mode']]+=mode['stages']
 require(set(totals.values())=={227},'Full227 stage bytes per old-four mode required')
 public=same['public_capi'];flag(public,['completed','all_public_capi_response_bytes_exact','production_observers_excluded']);false(public,['publication_qualified'])
 require(public['source']==METADATA_RUNTIME and public['cases']==4,'Actual public C ABI source/case coverage differs');sha(public['library_sha256'])
 arms=public['arms'];require(set(arms)=={'legacy','canonical'},'Both actual public models required')
 for arm in arms.values():
  flag(arm,['null_rejected','recovery_exact']);require(len(arm['response_texts'])==len(arm['response_sha256'])==4,'Four actual public responses required')
  require(all(digest(text.encode())==h for text,h in zip(arm['response_texts'],arm['response_sha256'])),'Public response UTF8/SHA differs')
  keys={(r['gate'],r['value'])for r in arm['gate_controls']};require(len(arm['gate_controls'])==6 and keys=={(g,v)for g in ['CRISPEMBED_JOSEV_QUANT','CRISPEMBED_QWEN35_QUANT']for v in [None,'','0']},'All quant gate negatives required')
  for row in arm['gate_controls']:flag(row,['init_rejected','run_rejected'])
 require(arms['legacy']['response_texts']==arms['canonical']['response_texts'],'Actual public response bytes changed')
 return dict(completed=True,legacy_model=legacy,canonical_model=canonical,canonicalization_report_sha256=m['proof']['sha256'],same_payload_report_sha256=m['same_payload_control']['sha256'],tensors=442,all_tensor_hashes_match_development=True,old_four_stage_bytes=totals,actual_public_capi=True,qualification=False)


def verify_model(path,cfg,proof):
 """Run only hosted, after SHA-pinned small evidence; no legacy model transfer."""
 from josev2b_task_quality_native_v1 import canonical_prerequisites
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Canonical model payload verification is hosted only')
 canonical_prerequisites(cfg)
 pin=cfg['canonicalization']['canonical_model'];require(path.is_file()and not path.is_symlink()and path.stat().st_size==pin['bytes']and sha256(path)==pin['sha256'],'Exact canonical model differs')
 # The immutable reviewed parser is pure stdlib. Its exact Git blob is used,
 # never whatever helper happens to exist in an unrelated working tree.
 helper=cfg['runtime_activation']['canonicalizer_helper'];raw=subprocess.check_output(['git','show',METADATA_RUNTIME+':'+helper['path']],cwd=ROOT)
 require(digest(raw)==helper['sha256'],'Frozen canonical header/parser helper differs')
 module=types.ModuleType('_frozen_56_canonicalizer');module.__file__=helper['path'];exec(compile(raw,helper['path'],'exec'),module.__dict__)
 parsed=module.parse(path);meta=parsed['metadata'];change=proof['allowed_changes'][0]
 require(meta['general.file_type']['kind']==4 and meta['general.file_type']['value']==14 and meta['general.file_type']['offset']==change['offset']and meta['general.quantization_version']['kind']==4 and meta['general.quantization_version']['value']==2,'Canonical standard14/quantization2 header differs')
 rows={r['name']:r for r in proof['tensors']};require(set(rows)==set(parsed['tensors']),'Canonical model tensor names differ')
 with path.open('rb')as source:
  for name,row in parsed['tensors'].items():
   expected=rows[name]
   for key in ['stored_shape','storage_rank','type','offset','bytes']:require(row[key]==expected[key],'Canonical parsed tensor differs: '+name)
   require(module.span(source,row['data_offset'],row['bytes'])==expected['canonical_sha256'],'Canonical tensor payload SHA differs: '+name)
  require(module.span(source,0,change['offset'])==proof['unchanged_prefix_sha256']and module.span(source,change['offset']+4,pin['bytes']-change['offset']-4)==proof['unchanged_suffix_sha256'],'Canonical off-span bytes differ')
 other={k:v['raw_sha256']for k,v in meta.items()if k!='general.file_type'}
 require(digest(json.dumps(other,sort_keys=True,separators=(',',':')).encode())==proof['unchanged_metadata_sha256'],'Other metadata bytes changed')
 return dict(completed=True,bytes=pin['bytes'],sha256=pin['sha256'],standard_file_type=14,quantization_version=2,tensors=442,all_payload_sha_verified=True,source_model_downloaded=False)
