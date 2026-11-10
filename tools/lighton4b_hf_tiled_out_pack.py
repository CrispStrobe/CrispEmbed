#!/usr/bin/env python3
"""Replace only24 original-F16-proven tiled Q4 spans; preserve HF scalars/tokenizer."""
import argparse,hashlib,json,os,struct,time
from pathlib import Path
from gguf_file_type_canonicalize import parse,sha,span
from lighton4b_q6_convert import skip_value,text
import lighton4b_hf_tiled_out_reproduce as proof
BASE_SHA='5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29'
BASE_BYTES=4041905312
PAIRED_SHA='e87e6b28936e1022b6207cb942ba5e1d468b309bf731c24bf44f136dbd20b215'
PROTOCOL_SHA='c21003e25e95509ecf71e386017b813d410490a696737eb31d45b3c08caa310f'
RECIPE='lighton4b-weighted-q4km-hf-tiled-out-v1'
TYPES={0:(1,4),12:(256,144),14:(256,210)}
def require(ok,why):
 if not ok:raise ValueError(why)
def strings():
 return {'qwen35_ocr.mixed_quant_recipe':RECIPE,'qwen35_ocr.quant_coordinate_profile':'hf-scalars-tiled-gdn-out-v1','crispembed.quantization.imatrix_profile':'lighton4b-text-inputs-hf-with-tiled-gdn-out-v1','qwen35_ocr.hf_tiled_out.base_candidate_sha256':BASE_SHA,'qwen35_ocr.hf_tiled_out.proof_protocol_sha256':PROTOCOL_SHA,'qwen35_ocr.hf_tiled_out.coordinate_map_sha256':proof.MAP_SHA,'qwen35_ocr.hf_tiled_out.paired_receipt_sha256':PAIRED_SHA}
def numbers():
 return {'qwen35_ocr.hf_tiled_out.tensor_count':24,'qwen35_ocr.hf_tiled_out.protected_tensor_count':699,'crispembed.quantization.imatrix_hf_grouped_entries':224,'crispembed.quantization.imatrix_upstream_tiled_entries':24}
def prepare_header(base,path):
 replacement=strings();newnumbers=numbers();require(not any(k.startswith('qwen35_ocr.hf_tiled_out.')for k in base['metadata']),'immutable HF baseline required')
 with Path(path).open('rb')as f:raw=f.read(base['data_offset'])
 records=[];changed=set()
 for name,row in base['metadata'].items():
  begin=row['offset']-12-len(name.encode());end=skip_value(raw,row['offset'],row['kind']);record=raw[begin:end];require(hashlib.sha256(record).hexdigest()==row['raw_sha256'],'complete original raw metadata record differs')
  if name in replacement:record=text(name)+struct.pack('<I',8)+text(replacement[name]);changed.add(name)
  records.append(record)
 require(changed=={'qwen35_ocr.mixed_quant_recipe','qwen35_ocr.quant_coordinate_profile','crispembed.quantization.imatrix_profile'},'closed baseline metadata profile required')
 for name,value in replacement.items():
  if name not in changed:records.append(text(name)+struct.pack('<I',8)+text(value))
 for name,value in newnumbers.items():require(name not in base['metadata'],'no duplicate metadata');records.append(text(name)+struct.pack('<II',4,value))
 descriptors=[]
 for name,row in base['tensors'].items():
  descriptors.append(text(name)+struct.pack('<I',row['storage_rank'])+struct.pack('<'+'Q'*row['storage_rank'],*row['stored_shape'])+struct.pack('<IQ',row['type'],row['data_offset']-base['data_offset']))
 header=b'GGUF'+struct.pack('<IQQ',3,len(descriptors),len(records))+b''.join(records)+b''.join(descriptors);header+=b'\0'*((-len(header))%32)
 return header,changed

def pack(base_path,paired_path,tensors,output,receipt):
 base_path=Path(base_path);paired_path=Path(paired_path);tensors=Path(tensors);output=Path(output);receipt=Path(receipt);partial=output.with_name(output.name+'.partial')
 require(all(not p.exists()and not p.is_symlink()for p in(output,receipt,partial))and len({p.absolute()for p in(output,receipt,partial)})==3 and output.parent.is_dir()and receipt.parent.is_dir(),'fresh distinct outputs required')
 started=time.monotonic();r={'schema':'lighton4b-hf-tiled-out-pack/v1','status':'failed','recipe':RECIPE,'original_forward_executed':False,'requantization_executed':False,'model_inference_executed':False,'physical_scalar_semantics_enabled':False,'quality_qualified':False,'tensors':[]}
 def save():receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 try:
  p=proof.protocol();require(sha(proof.PROTOCOL)==PROTOCOL_SHA,'immutable proof protocol differs');require(paired_path.stat().st_size<2**20 and sha(paired_path)==PAIRED_SHA,'exact measured original-F16 receipt required');paired=json.loads(paired_path.read_text());require(paired['status']=='reproduced'and paired['candidate_conversion_allowed']and len(paired['tensors'])==24,'complete actual proof required')
  require(base_path.stat().st_size==BASE_BYTES and sha(base_path)==BASE_SHA,'whole immutable HF baseline differs');base=parse(base_path,TYPES);require(len(base['tensors'])==723,'complete723 tensor baseline required')
  expected={'qwen35_ocr.mixed_quant_recipe':'lighton4b-weighted-q4km-style-q6head-v1','qwen35_ocr.quant_coordinate_profile':'hf-grouped-v1','qwen35_ocr.quant_source_f16_sha256':p['source_original_f16']['sha256'],'crispembed.quantization.imatrix_source_sha256':p['statistics']['sha256'],'crispembed.quantization.imatrix_profile':'lighton4b-text-inputs-hf-grouped-v1'}
  require(all(base['metadata'][k]['kind']==8 and base['metadata'][k]['value']==v for k,v in expected.items()),'closed HF baseline provenance differs');require(base['metadata']['qwen35_ocr.format_version']['value']==5 and base['metadata']['qwen35_ocr.zero_centered_text_norm']['value']==1,'HF scalar semantics required')
  selected={row['name']:row for row in p['selected']};require(set(paired['tensors'])==set(selected),'exact measured24 names required');payloads={}
  for name,row in selected.items():
   tensor=base['tensors'][name];path=tensors/name/'tiled.q4';actual=paired['tensors'][name];require(tensor['type']==12 and tensor['stored_shape']==[4096,2560]and tensor['bytes']==row['expected_q4_bytes'],'24 Q4 output geometry differs');require(actual['hf_control_exact']and actual['publisher_tiled_exact']and actual['tiled_q4_sha256']==row['expected_publisher_tiled_q4_sha256']and actual['hf_control_q4_sha256']==row['expected_hf_control_q4_sha256'],'actual proof payload identity differs');require(path.is_file()and not path.is_symlink()and path.stat().st_size==tensor['bytes']and sha(path)==actual['tiled_q4_sha256'],'authenticated original-F16 tiled bytes required');payloads[name]=path
  header,changed=prepare_header(base,base_path);r.update(base_sha256=BASE_SHA,paired_receipt_sha256=PAIRED_SHA,proof_protocol_sha256=PROTOCOL_SHA,header_bytes=len(header));save()
  with base_path.open('rb')as src,partial.open('xb')as dest:
   dest.write(header);src.seek(base['data_offset'])
   while block:=src.read(4*1024**2):dest.write(block)
  final=parse(partial,TYPES);require(set(final['tensors'])==set(base['tensors']),'persisted tensor domain differs')
  with partial.open('r+b')as dest:
   for name,path in payloads.items():
    dest.seek(final['tensors'][name]['data_offset'])
    with path.open('rb')as src:
     while block:=src.read(65536):dest.write(block)
  with base_path.open('rb')as before,partial.open('rb')as after:
   for name,a in base['tensors'].items():
    b=final['tensors'][name];require(a['stored_shape']==b['stored_shape']and a['type']==b['type']and a['bytes']==b['bytes'],'geometry/type changed');old=span(before,a['data_offset'],a['bytes']);new=span(after,b['data_offset'],b['bytes']);chosen=name in selected;require(new==(selected[name]['expected_publisher_tiled_q4_sha256']if chosen else old),'exact699 protected/24 replaced payload audit failed');require(not chosen or old==selected[name]['expected_hf_control_q4_sha256'],'old24 HF payload audit failed');r['tensors'].append({'name':name,'type':a['type'],'bytes':a['bytes'],'source_sha256':old,'sha256':new,'replaced_out':chosen});save()
  require(all(final['metadata'][k]['raw_sha256']==v['raw_sha256']for k,v in base['metadata'].items()if k not in changed),'unchanged HF scalar/tokenizer metadata bytes differ');require(all(final['metadata'][k]['kind']==8 and final['metadata'][k]['value']==v for k,v in strings().items())and all(final['metadata'][k]['kind']==4 and final['metadata'][k]['value']==v for k,v in numbers().items()),'typed new metadata differs');require(sha(base_path)==BASE_SHA and sha(paired_path)==PAIRED_SHA,'immutable sources changed')
  r.update(status='packed',candidate_sha256=sha(partial),candidate_bytes=partial.stat().st_size,protected_tensor_count=699,replaced_out_tensor_count=24,all699_protected_payloads_exact=True,all24_publisher_tiled_payloads_exact=True,all_other_metadata_raw_exact=True);partial.rename(output);return r
 except Exception as e:r['error_type']=type(e).__name__;r['error']='closed HF-only output-layout packing failed';raise
 finally:r['seconds']=time.monotonic()-started;save()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in('base','paired-receipt','tensors','output','receipt'):p.add_argument('--'+name,required=True,type=Path)
 a=p.parse_args();pack(a.base,a.paired_receipt,a.tensors,a.output,a.receipt)
