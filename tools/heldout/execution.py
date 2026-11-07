"""Frozen held-out input/response contracts, scoring and raw-row preservation."""
import hashlib,importlib.util,json,math,os,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('heldout_prepare',Path(__file__).with_name('lighton4b_prepare.py'));prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
require=prepare.require
ARMS=['upstream-off','upstream-on','native-composed','native-source-features']
CAP=1536
OBSERVED_TEXT_ANCHORS=['prefill.input.f32','prefill.positions.f32','prefill.logits.f32','prefill.norm.f32','teacher.0.logits.f32','greedy.0.logits.f32']+[f'prefill.layer.{i}.f32'for i in range(32)]

def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def admit(cfg):
 original=(ROOT/'docs/lighton4b-heldout-v1.json').read_bytes();prepare.admit(json.loads(original))
 require(cfg['schema']=='lighton4b-provisional-heldout-executor/v1'and cfg['selection_protocol_sha256']==hashlib.sha256(original).hexdigest(),'unchanged frozen selection')
 require(cfg['cases']==prepare.CASE_IDS and cfg['arms']==ARMS and cfg['cap']==1536 and cfg['context']==2048 and cfg['maximum_prompt_tokens']==512 and cfg['maximum_image_tokens']==256,'frozen sequence and capacity')
 require(cfg['threads']==4 and cfg['build_threads']==2 and cfg['case_worker_seconds']==6000 and cfg['arm_seconds']==1200 and cfg['workers_max']==1,'frozen resource quotas')
 require(cfg['candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43'and cfg['upstream_source']=='6c01da6278ab6c69ed67d9124806b50d2b299ba7'and cfg['production_source']=='696845d0562e0e96f403fce8d31c5b653767ac0c','fixed arithmetic/weights')
 require(cfg['observed_form_receipt_sha256']=='aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b','fixed observed control')
 require(cfg['qualified']is False and cfg['references_expert_adjudicated']is False and cfg['normalization_policy']=='crispembed-ocr-quality-v1','provisional references and fixed scorer')
 require(sha(ROOT/'tests/lighton4b_native_document.cpp')==cfg['driver_sha256']and sha(ROOT/'tests/lighton4b_native_case_preflight.cpp')==cfg['native_preflight_sha256']and sha(ROOT/'tools/heldout/source_preflight.cpp')==cfg['source_preflight_sha256'],'frozen reference executable changes')
 old=json.loads(original)
 require(sha(ROOT/'tests/regression/images/cc0/ground_truth.json')==old['reference']['gold_registry_sha256'],'frozen reference registry')
 baseline=json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text());upstream=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text())
 for key in ['model_repo','model_revision','models','base_model','candidate_sha256']:require(cfg[key]==baseline[key],'frozen model '+key)
 require(cfg['projector']==upstream['projector']and cfg['expected_media_prompt']==upstream['expected_media_prompt'],'frozen projector/template')
 for path,digest in old['reference']['scorer'].items():require(sha(ROOT/path)==digest,'frozen scorer '+path)

def native_args(binary,candidate,rgb,h,w,teacher,dest,arm,feature=None):
 require(arm in ['native-composed','native-source-features'],'frozen native role')
 args=[binary,candidate,rgb,h,w,teacher,dest,1536,4,'greedy-full']
 if arm=='native-source-features':require(feature is not None,'explicit secondary feature input');args+=['--upstream-vision',feature]
 return args+['--provisional-heldout-cap1536']

def seal_inputs(path,case,ids,mods,grid,native_positions,source_ids,source_mods,source_positions,template,cfg):
 require(ids==source_ids and mods==source_mods and len(ids)==len(mods) and all(type(x)is int and 0<=x<248320 for x in ids) and all(type(x)is int and x in[0,1]for x in mods),'native/source exact prompt and modalities')
 require(len(grid)==3 and grid[0]==1 and grid[1]%2==grid[2]%2==0 and grid[1]*grid[2]//4==mods.count(1),'actual single-image geometry')
 require(grid==[1,case['prepared']['size'][1]//16,case['prepared']['size'][0]//16],'frozen prepared geometry unchanged')
 prepare.budget(len(ids),mods.count(1));require(len(native_positions)==len(source_positions)==3*len(ids)*4 and native_positions==source_positions,'actual source/native positional bytes')
 require(template==cfg['expected_media_prompt'],'frozen empty OCR chat template')
 manifest=dict(schema='lighton4b-heldout-pre-forward-inputs/v1',case=case['id'],selection_protocol_sha256=cfg['selection_protocol_sha256'],prepared=case['prepared'],prompt_ids=ids,modalities=mods,grid=grid,prompt_tokens=len(ids),image_tokens=mods.count(1),cap=1536,context=2048,positions_sha256=hashlib.sha256(native_positions).hexdigest(),abstract_media_template=template,heldout_candidate_outputs_read=False,heldout_forward_calls_before_seal=0)
 with Path(path).open('x')as f:json.dump(manifest,f,indent=2,ensure_ascii=False);f.write('\n')
 return sha(path)

def source_response(response,prompt_tokens):
 ids=response['tokens'];require(isinstance(ids,list)and 1<=len(ids)<=1536 and all(type(i)is int and 0<=i<248320 for i in ids),'source complete bounded token IDs')
 require(response['tokens_evaluated']==prompt_tokens and not response['truncated'],'source prompt untrimmed')
 require(response['stop_type']in['eos','limit']and(response['stop_type']=='eos')==(ids[-1]in[248044,248046])and(response['stop_type']!='limit'or len(ids)==1536),'source actual EOS/cap closure')
 require(len(response['completion_probabilities'])==len(ids)and isinstance(response['content'],str),'source actual full probabilities/text')
 return response['stop_type']=='eos'

def neutral(off,on):
 for k in ['tokens','content','completion_probabilities','stop_type','tokens_evaluated','truncated']:require(off[k]==on[k],'complete source observer neutrality '+k)

def audit_rows(folder,ids,stem='greedy',archive=True):
 import numpy as np
 root=Path(folder);require(stem in['greedy','source']and all(type(i)is int and 0<=i<248320 for i in ids),'actual row role/IDs');require(1<=len(ids)<=1536,'bounded actual row count');rows=[];paths=[]
 for j,token in enumerate(ids):
  p=root/(f'source.logits.{j}.f32'if stem=='source'else f'greedy.{j}.logits.f32');require(p.stat().st_size==248320*4,'complete vocabulary row')
  values=np.fromfile(p,dtype='<f4');require(np.isfinite(values).all()and int(values.argmax())==token,'actual finite greedy row/argmax');rows.append(dict(position=j,sha256=sha(p),bytes=p.stat().st_size,argmax=token));paths.append(p)
 if archive:
  destination=root/(stem+'-raw-rows.zip')
  with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_STORED,allowZip64=True)as z:
   for p in paths:z.write(p,p.name)
  with zipfile.ZipFile(destination)as z:
   for p,row in zip(paths,rows):
    with z.open(p.name)as f:require(hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],'closed archived raw row')
  # Only these task-owned, complete score files are removed after archive closure.
  for p in paths:p.unlink()
  return dict(rows=rows,archive=dict(file=destination.name,bytes=destination.stat().st_size,sha256=sha(destination)),scope='Raw F32 rows retained remotely in lossless ZIP_STORED; per-row hashes/argmax preserved.')
 return dict(rows=rows)

def score(gold,text,complete):
 spec=importlib.util.spec_from_file_location('frozen_ocr_quality',ROOT/'tools/ocr_quality.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);result=module.score(gold,text,complete=complete)
 if gold['tables']is not None:
  result['expected_table_shape']=[[len(row)for row in table]for table in gold['tables']]
  result['table_shape_valid']=bool(result['parse_ok']and [[len(row)for row in table]for table in result.get('parsed_tables',[])]==result['expected_table_shape'])
 return result

def floor(native,source,inventory=False):
 if not(native['complete']and source['complete']and native['parse_ok']and source['parse_ok']):return False
 ok=all(native['content'][key]<=source['content'][key]for key in ['cer','wer'])
 if inventory:ok=ok and native['table']is not None and source['table']is not None and native['table']['cell_errors']<=source['table']['cell_errors']and native['table_shape_valid']
 return bool(ok)
