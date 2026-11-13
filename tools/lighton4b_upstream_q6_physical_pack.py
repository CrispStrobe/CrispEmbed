#!/usr/bin/env python3
"""Lossless physical full-Q6→native-format6 layout import; no requantization/F32 model."""
import argparse,hashlib,importlib.util,json,math,os,struct
from pathlib import Path
from gguf_file_type_canonicalize import parse,span
from lighton4b_q6_convert import skip_value,text
ROOT=Path(__file__).resolve().parents[1]
RECIPE='lighton4b-upstream-q6-physical-v1'
Q4_SHA='b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26'
BASE_SHA='f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'
UP_SHA='290e2b46627c9d3f932c25ef6697fc71a37a9b226337d761a0190185eaa738b7'
TYPES={0:(1,4),8:(32,34),12:(256,144),14:(256,210)}
spec=importlib.util.spec_from_file_location('physical_mapping',ROOT/'tools/ci-heavy/lighton4b_upstream_physical_audit.py');mapping=importlib.util.module_from_spec(spec);spec.loader.exec_module(mapping)
def require(x,why):
 if not x:raise ValueError(why)
def digest(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metadata_strings():
 return {'qwen35_ocr.mixed_quant_recipe':RECIPE,'qwen35_ocr.upstream_physical.repository':'richardyoung/LightOnOCR-3-4B-GGUF','qwen35_ocr.upstream_physical.revision':'8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01','qwen35_ocr.upstream_physical.sha256':UP_SHA,'qwen35_ocr.upstream_physical.protected_source_sha256':BASE_SHA,'qwen35_ocr.upstream_physical.mapping_sha256':'f12db3045f003dd2bcde85ba4159beda86cbaec5617ebe7d9f40388336796390','qwen35_ocr.upstream_physical.source_code':'6c01da6278ab6c69ed67d9124806b50d2b299ba7','qwen35_ocr.upstream_physical.semantics':'gamma-direct,a-direct,hf-state,tiled-out-input,upstream-control-types'}
def metadata_uint32():
 return {'qwen35_ocr.upstream_physical.version':1,'qwen35_ocr.upstream_physical.row_permuted_tensors':96,'qwen35_ocr.upstream_physical.retained_out_tensors':24,'qwen35_ocr.upstream_physical.protected_vision_tensors':297}
def array(raw,row):
 require(row['kind']==9,'typed array required');sub,n=struct.unpack_from('<IQ',raw,row['offset']);at=row['offset']+12;require(n<=248320,'bounded tokenizer array')
 if sub==8:
  result=[]
  for _ in range(n):size=struct.unpack_from('<Q',raw,at)[0];at+=8;require(size<=1048576,'bounded tokenizer string');result.append(raw[at:at+size].decode());at+=size
  return result
 require(sub==5,'signed int32 array required');return list(struct.unpack_from('<'+'i'*n,raw,at))
def tokenizer_equivalence(base,up,base_path,up_path):
 with base_path.open('rb')as f:a=f.read(base['data_offset'])
 with up_path.open('rb')as f:b=f.read(up['data_offset'])
 for key in ['tokenizer.ggml.model','tokenizer.ggml.pre','tokenizer.ggml.merges','tokenizer.ggml.eos_token_id','tokenizer.ggml.padding_token_id','tokenizer.chat_template']:
  require(key in base['metadata']and key in up['metadata']and base['metadata'][key]['raw_sha256']==up['metadata'][key]['raw_sha256'],'complete tokenizer field differs: '+key)
 av=array(a,base['metadata']['tokenizer.ggml.tokens']);bv=array(b,up['metadata']['tokenizer.ggml.tokens']);types=array(b,up['metadata']['tokenizer.ggml.token_type']);added=set(array(a,base['metadata']['qwen35_ocr.added_token_ids']));special=set(array(a,base['metadata']['qwen35_ocr.special_token_ids']));require(len(av)==len(bv)==len(types)==248320,'full vocabulary shape');unused=0
 for i,(x,y,t)in enumerate(zip(av,bv,types)):
  if x=='':require(y=='[PAD'+str(i)+']'and t==5 and i not in added,'only declared unused PAD rows may differ');unused+=1
  else:require(x==y and t in(1,3,4)and((i in added)==(t!=1)),'real token/type semantics differ')
 require(unused==243,'complete unused vocabulary domain');up_special={i for i,t in enumerate(types)if t==3};require(special<=up_special and len(up_special-special)==6 and up_special<=added,'source control type domain');return {'all248320_decode_semantics_checked':True,'same_real_tokens':248077,'upstream_unused_pad_rows_native_empty':unused,'upstream_control_ids':sorted(up_special),'additional_upstream_control_ids':sorted(up_special-special),'all_merges_and_prompt_template_raw_bytes_exact':True,'note':'Upstream BPE token_to_piece suppresses UNUSED type5; native empty rows decode identically.'}
def row_order(name,row):
 if name.endswith('.linear_attn.conv1d.weight'):return list(range(4096))+[4096+x for x in mapping.head_perm()]
 if name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')):return mapping.head_perm(1)
 if len(row['stored_shape'])==2:return mapping.rows_for(name,row['stored_shape'][1])
 return None

def prepare_header(base,up,base_path,tokenizer):
 with base_path.open('rb')as f:raw=f.read(base['data_offset'])
 records=[];changed={'general.file_type','qwen35_ocr.format_version','qwen35_ocr.zero_centered_text_norm','qwen35_ocr.special_token_ids'}
 require(not any(k in base['metadata']for k in metadata_strings()),'base must be immutable plain source, without a recipe')
 for name,row in base['metadata'].items():
  start=row['offset']-12-len(name.encode());end=skip_value(raw,row['offset'],row['kind']);record=raw[start:end];require(hashlib.sha256(record).hexdigest()==row['raw_sha256'],'raw base metadata identity')
  if name=='general.file_type':record=text(name)+struct.pack('<II',4,18)
  elif name=='qwen35_ocr.format_version':record=text(name)+struct.pack('<II',4,6)
  elif name=='qwen35_ocr.zero_centered_text_norm':record=text(name)+struct.pack('<IB',7,0)
  elif name=='qwen35_ocr.special_token_ids':
   ids=tokenizer['upstream_control_ids'];record=text(name)+struct.pack('<IIQ',9,5,len(ids))+struct.pack('<'+'i'*len(ids),*ids)
  records.append(record)
 for name,value in metadata_strings().items():records.append(text(name)+struct.pack('<I',8)+text(value))
 for name,value in metadata_uint32().items():records.append(text(name)+struct.pack('<II',4,value))
 records.append(text('qwen35_ocr.upstream_physical.eog_ids')+struct.pack('<IIQii',9,5,2,248044,248046))
 descriptors=[];layout=[];offset=0;mapped={mapping.physical_name(n):n for n in base['tensors']if n.startswith('l.')};require(len(mapped)==426 and set(mapped)==set(up['tensors']),'complete426 source map');identity={p:{'native_name':n,'upstream_shape':up['tensors'][p]['stored_shape'],'type':up['tensors'][p]['type']}for p,n in mapped.items()};require(hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()==metadata_strings()['qwen35_ocr.upstream_physical.mapping_sha256'],'fixed name/shape/type descriptor')
 for name,row in base['tensors'].items():
  physical=mapping.physical_name(name)if name.startswith('l.')else None;source=up['tensors'][physical]if physical else row;kind=source['type'];order=row_order(name,source)if physical else None;require(math.prod(source['stored_shape'])==math.prod(row['stored_shape']),'source geometry product');count=source['bytes'];offset=(offset+31)//32*32;layout.append({'name':name,'physical_name':physical,'source_offset':source['data_offset'],'offset':offset,'bytes':count,'type':kind,'stored_shape':row['stored_shape'],'source_shape':source['stored_shape'],'row_order':order});descriptors.append(text(name)+struct.pack('<I',row['storage_rank'])+struct.pack('<'+'Q'*row['storage_rank'],*row['stored_shape'])+struct.pack('<IQ',kind,offset));offset+=count
 payload=b'GGUF'+struct.pack('<IQQ',3,len(layout),len(records))+b''.join(records)+b''.join(descriptors);payload+=b'\0'*((-len(payload))%32)
 for row in layout:row['data_offset']=len(payload)+row['offset']
 return payload,layout,len(payload)+offset,changed

def transport_tensor(source,target,source_offset,target_offset,count,order,kind):
 target.seek(target_offset);h=hashlib.sha256();permuted=order is not None and order!=list(range(len(order)))
 if permuted:
  require(sorted(order)==list(range(len(order)))and count%len(order)==0,'whole-row permutation');width=count//len(order);require(0<width<=16384,'bounded row transport')
  if kind in(12,14):require(width%TYPES[kind][1]==0,'whole quantized blocks only')
  for i in order:
   source.seek(source_offset+i*width);b=source.read(width);require(len(b)==width,'source row complete');target.write(b);h.update(b)
 else:
  source.seek(source_offset);remaining=count
  while remaining:
   b=source.read(min(remaining,4*1024**2));require(bool(b),'source span complete');target.write(b);h.update(b);remaining-=len(b)
 return h.hexdigest(),permuted

def validate_small_identity(up_path, up, q4_path, q4):
 require(set(up['tensors'])==set(q4['tensors']),'complete same-source text tensor domain')
 result=[]
 with Path(up_path).open('rb')as a,Path(q4_path).open('rb')as b:
  for name,row in up['tensors'].items():
   previous=q4['tensors'][name]
   require(row['stored_shape']==previous['stored_shape'],'same-source tensor geometry')
   if row['type']==0:
    require(previous['type']==0 and row['bytes']==previous['bytes'],'same-source small tensor type/size')
    ah=span(a,row['data_offset'],row['bytes']);bh=span(b,previous['data_offset'],previous['bytes'])
    require(ah==bh,'actual small source parameter differs: '+name)
    result.append({'name':name,'bytes':row['bytes'],'sha256':ah})
 require(len(result)==177 and sum(r['type']==14 for r in up['tensors'].values())==249,'exact177small/249Q6 profile')
 return result

def validate_paths(sources, output, partial, report):
 # Validate before entering the exception handler: even failure reports write.
 # resolve() covers lexical/symlink aliases; samefile() also covers hardlinks.
 def aliases(a,b):
  return a.resolve()==b.resolve() or (a.exists() and b.exists() and a.samefile(b))
 destinations=[output,partial,report]
 for i,path in enumerate(destinations):
  require(not any(aliases(path,other) for other in [*sources,*destinations[:i]]),
          'import destinations must not alias sources or each other')

def convert(base_path,up_path,output,report_path,q4_control):
 base_path=Path(base_path);up_path=Path(up_path);q4_control=Path(q4_control);output=Path(output);report_path=Path(report_path);partial=output.with_name(output.name+'.partial');report={'schema':'lighton4b-upstream-full-q6-physical-import/v1','status':'failed','recipe':RECIPE,'original_forward_executed':False,'model_inference_executed':False,'requantization_executed':False,'full_f32_model_created':False,'quality_qualified':False,'tensors':[]}
 validate_paths([base_path,up_path,q4_control],output,partial,report_path)
 def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
 try:
  require(not output.exists()and not partial.exists(),'task target exists');require(base_path.stat().st_size==4030957952 and digest(base_path)==BASE_SHA,'whole protected source identity');require(up_path.stat().st_size==3464055776 and digest(up_path)==UP_SHA,'whole physical upstream identity');base=parse(base_path,TYPES);up=parse(up_path,TYPES);require(q4_control.stat().st_size==2708804576 and digest(q4_control)==Q4_SHA,'whole historical Q4M source identity');q4=parse(q4_control,TYPES);report['small_source_tensors']=validate_small_identity(up_path,up,q4_control,q4);require(len(base['tensors'])==723 and len(up['tensors'])==426,'exact source inventories');report['tokenizer_equivalence']=tokenizer_equivalence(base,up,base_path,up_path);header,layout,size,changed=prepare_header(base,up,base_path,report['tokenizer_equivalence']);report.update(candidate_bytes=size,header_bytes=len(header));save()
  with partial.open('xb')as f:f.write(header);f.truncate(size)
  with base_path.open('rb')as original,up_path.open('rb')as physical,partial.open('r+b')as target:
   for row in layout:
    src=physical if row['physical_name']else original;source_sha=span(src,row['source_offset'],row['bytes']);order=row['row_order'];new_sha,permuted=transport_tensor(src,target,row['source_offset'],row['data_offset'],row['bytes'],order,row['type'])
    target.flush()
    with partial.open('rb')as verify:require(span(verify,row['data_offset'],row['bytes'])==new_sha,'persisted tensor readback')
    record={k:v for k,v in row.items()if k!='row_order'};record.update(source_packed_sha256=source_sha,sha256=new_sha,whole_rows_permuted=len(order)if permuted else 0,protected_vision=row['physical_name']is None,out_columns_retained=row['name'].endswith('.linear_attn.out_proj.weight'),quant_blocks_unchanged=True);report['tensors'].append(record);save()
  final=parse(partial,TYPES);require(len(final['tensors'])==723 and all(final['tensors'][r['name']]['type']==r['type']and final['tensors'][r['name']]['data_offset']==r['data_offset']for r in layout),'persisted complete header');require(all(final['metadata'][k]['raw_sha256']==v['raw_sha256']for k,v in base['metadata'].items()if k not in changed),'all unchanged native source metadata');require(all(final['metadata'][k]['value']==v for k,v in metadata_strings().items())and all(final['metadata'][k]['kind']==4 and final['metadata'][k]['value']==v for k,v in metadata_uint32().items()),'typed physical provenance')
  require(final['metadata']['qwen35_ocr.format_version']['kind']==4 and final['metadata']['qwen35_ocr.format_version']['value']==6 and final['metadata']['general.file_type']['kind']==4 and final['metadata']['general.file_type']['value']==18 and final['metadata']['qwen35_ocr.zero_centered_text_norm']['kind']==7 and final['metadata']['qwen35_ocr.zero_centered_text_norm']['value']==0,'distinct physical format semantics');
  with partial.open('rb')as f:raw=f.read(final['data_offset'])
  require(array(raw,final['metadata']['qwen35_ocr.special_token_ids'])==report['tokenizer_equivalence']['upstream_control_ids']and array(raw,final['metadata']['qwen35_ocr.upstream_physical.eog_ids'])==[248044,248046],'complete persisted source token/stop semantics');
  census={str(t):sum(r['type']==t for r in layout)for t in {r['type']for r in layout}};require(census=={'0':474,'14':249},'complete target census');require(sum(r['protected_vision']for r in report['tensors'])==297 and sum(r['whole_rows_permuted']>0 and r['type']in(12,14)for r in report['tensors'])==96 and sum(r['out_columns_retained']for r in report['tensors'])==24,'exact transport domains');require(digest(base_path)==BASE_SHA and digest(up_path)==UP_SHA and digest(q4_control)==Q4_SHA,'source files immutable after copy');report.update(status='converted',candidate_sha256=digest(partial),census=census,protected_vision_tensors=297,physical_text_tensors=426);partial.rename(output);save();return report
 except Exception:report['error']='physical transport failed; partial candidate not admitted';save();raise

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('base','upstream','output','report','q4-control'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();require(Path('/kaggle/working').is_dir(),'Kaggle-only physical import');convert(a.base,a.upstream,a.output,a.report,a.q4_control)
if __name__=='__main__':main()
