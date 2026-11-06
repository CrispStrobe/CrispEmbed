#!/usr/bin/env python3
"""Copy691 pinned tensors, quantize32 directly from original BF16; no F32 model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from gguf_file_type_canonicalize import parse,span
from lighton4b_q6_recipe import plan,RECIPE
BASE_SHA='f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'
SOURCE_SHA='6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67'
ORACLE_SHA='03119ea4d0fbbf5cc86b91f80b47bbda22d90b417d89dd0c16b8fafa1a85a89b'
TYPES={0:(1,4),1:(1,2),8:(32,34),12:(256,144),14:(256,210)}
def require(value,text):
 if not value:raise ValueError(text)
def digest(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def text(value):
 value=value.encode();return struct.pack('<Q',len(value))+value
def skip_value(raw,pos,kind):
 formats={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
 if kind==8:return pos+8+struct.unpack_from('<Q',raw,pos)[0]
 if kind==9:
  sub,count=struct.unpack_from('<IQ',raw,pos);pos+=12
  for _ in range(count):pos=skip_value(raw,pos,sub)
  return pos
 require(kind in formats,'unsupported original metadata kind');return pos+formats[kind]
def header(base,path,p):
 """Preserve all raw metadata except typed M15; append explicit provenance."""
 with Path(path).open('rb')as f:raw=f.read(base['data_offset'])
 records=[]
 for name,row in base['metadata'].items():
  start=row['offset']-12-len(name.encode());end=skip_value(raw,row['offset'],row['kind']);record=raw[start:end]
  require(hashlib.sha256(record).hexdigest()==row['raw_sha256'],'raw metadata clone differs')
  records.append(text(name)+struct.pack('<II',4,15)if name=='general.file_type'else record)
 require('qwen35_ocr.mixed_quant_recipe'not in base['metadata'],'base must remain the plain Q4 file')
 for name,value in [('qwen35_ocr.mixed_quant_recipe',p.get('recipe',RECIPE)),('qwen35_ocr.q6_source_checkpoint_sha256',SOURCE_SHA),('qwen35_ocr.q6_names_sha256',p['q6_names_sha256'])]:records.append(text(name)+struct.pack('<I',8)+text(value))
 for name,value in p.get('metadata_strings',{}).items():records.append(text(name)+struct.pack('<I',8)+text(value))
 for name,value in p.get('metadata_uint32',{}).items():records.append(text(name)+struct.pack('<II',4,value))
 layout=[];offset=0;expected={r['name']:r for r in p['all_tensors']}
 require(set(expected)==set(base['tensors']),'all723 base tensor names differ')
 descriptors=[]
 for name,row in base['tensors'].items():
  wanted=expected[name];shape=wanted['source_shape'];stored=[shape[0],math.prod(shape[1:])]if name=='v.patch_embed.proj.weight'else shape
  old_type=12 if wanted['type']==14 else wanted['type']
  require(row['type']==old_type and row['stored_shape']==list(reversed(stored)),'base exact shape/type differs: '+name)
  new=wanted['type'];block,width=TYPES[new];size=math.prod(shape)//block*width;offset=(offset+31)//32*32
  layout.append({**wanted,'old_offset':row['data_offset'],'old_bytes':row['bytes'],'offset':offset,'bytes':size,'copied':new!=14})
  descriptors.append(text(name)+struct.pack('<I',row['storage_rank'])+struct.pack('<'+'Q'*row['storage_rank'],*row['stored_shape'])+struct.pack('<IQ',new,offset));offset+=size
 payload=b'GGUF'+struct.pack('<IQQ',3,len(layout),len(records))+b''.join(records)+b''.join(descriptors);payload+=b'\0'*((-len(payload))%32)
 for row in layout:row['data_offset']=len(payload)+row['offset']
 return payload,layout,len(payload)+offset

def independent_bf16(path,offset,size):
 import numpy as np
 from gguf.quants import dequantize
 from gguf import GGMLQuantizationType
 require(np.__version__=='2.3.5','pinned independent array decoder required')
 h=hashlib.sha256()
 with path.open('rb')as f:
  f.seek(offset)
  while size:
   count=min(size,256*210);data=f.read(count);require(len(data)==count,'Q6 independent read truncated')
   values=dequantize(np.frombuffer(data,dtype=np.uint8),GGMLQuantizationType.Q6_K).astype('<f4',copy=False)
   require(values.size<=65536 and np.isfinite(values).all(),'bounded finite independent Q6 decode required')
   bits=values.view('<u4');rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2');h.update(rounded.tobytes());size-=count
 return h.hexdigest()

def convert(base_path,checkpoint,oracle_path,binary,output,report_path,recipe_plan=None):
 p=recipe_plan or plan();fresh=len(p['q6_names']);copied=723-fresh;report={'schema':'lighton4b-original-bf16-selective-q6/v1','status':'blocked','recipe':p['recipe'],'source_checkpoint_sha256':SOURCE_SHA,'base_sha256':BASE_SHA,'quality_qualified':False,'held_out_selection':False,'model_inference_executed':False,'f32_model_instantiated':False,'requantization_executed':False,'tensors':[]}
 def save():report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 partial=output.with_name(output.name+'.partial')
 try:
  require(not output.exists()and not partial.exists(),'candidate target already exists');require(base_path.stat().st_size==4030957952 and digest(base_path)==BASE_SHA,'immutable base Q4 file differs');require(checkpoint.stat().st_size==9078620536 and digest(checkpoint)==SOURCE_SHA,'immutable original BF16 checkpoint differs');require(digest(oracle_path)==ORACLE_SHA,'retained original723 hash oracle differs')
  oracle=json.loads(oracle_path.read_text());source_hash=oracle['original_bf16_tensor_sha256'];base_rows={r['gguf']:r for r in oracle['q4_k-weights']['tensors']};base=parse(base_path);payload,layout,size=header(base,base_path,p)
  with checkpoint.open('rb')as f:
   length=struct.unpack('<Q',f.read(8))[0];raw=f.read(length)
  require(length==89456 and hashlib.sha256(raw).hexdigest()==p['source_header_sha256'],'original BF16 header differs');source_header=json.loads(raw)
  with partial.open('xb')as f:f.write(payload);f.truncate(size)
  report.update(stage='copy'+str(copied)+'-and-quantize'+str(fresh),q6_names_sha256=p['q6_names_sha256'],candidate_bytes=size,header_bytes=len(payload));save()
  with base_path.open('rb')as src,partial.open('r+b')as dst:
   for row in layout:
    expected=base_rows[row['name']];original=source_header[row['source']];elements=math.prod(row['source_shape']);old_sha=span(src,row['old_offset'],row['old_bytes']);require(old_sha==expected['raw_sha256'],'base tensor hash differs: '+row['name'])
    if row['copied']:
     src.seek(row['old_offset']);dst.seek(row['data_offset']);remaining=row['bytes']
     while remaining:
      block=src.read(min(remaining,4*2**20));require(block,'base copy truncated');dst.write(block);remaining-=len(block)
     dst.flush();new_sha=old_sha;record={'name':row['name'],'type':row['type'],'copied':True,'source_base_sha256':old_sha,'sha256':new_sha,'bytes':row['bytes'],'bf16_sha256':expected['bf16_sha256']}
    else:
     receipt=output.parent/(row['name']+'.quant.json');offset=8+length+original['data_offsets'][0]
     subprocess.run([str(binary),str(checkpoint),str(offset),str(elements),str(row['source_shape'][-1]),str(partial),str(row['data_offset']),source_hash[row['source']],str(receipt),'65536'],check=True,timeout=300)
     q=json.loads(receipt.read_text());require(q['source_bf16_sha256']==source_hash[row['source']]and q['bytes']==row['bytes']and not q['requantized'],'original32 Q6 receipt differs');new_sha=q['q6_sha256'];record={'name':row['name'],'type':14,'copied':False,'source_original_bf16_sha256':q['source_bf16_sha256'],'sha256':new_sha,'bytes':row['bytes'],'max_chunk_elements':q['max_chunk_elements'],'bf16_sha256':q['dequant_bf16_sha256']}
     require(independent_bf16(partial,row['data_offset'],row['bytes'])==q['dequant_bf16_sha256'],'independent Python/native Q6 BF16 decode differs')
    with partial.open('rb')as verify:require(span(verify,row['data_offset'],row['bytes'])==new_sha,'candidate tensor readback differs')
    report['tensors'].append(record);save()
  candidate=parse(partial,TYPES);require(len(candidate['tensors'])==723 and candidate['metadata']['general.file_type']['value']==15 and candidate['metadata']['qwen35_ocr.mixed_quant_recipe']['value']==p['recipe'],'candidate final header differs');require(all(candidate['tensors'][r['name']]['type']==r['type']and candidate['tensors'][r['name']]['data_offset']==r['data_offset']and candidate['tensors'][r['name']]['bytes']==r['bytes']for r in layout),'final parsed candidate layout differs');require(all(candidate['metadata'][k]['raw_sha256']==v['raw_sha256']for k,v in base['metadata'].items()if k!='general.file_type'),'retained metadata bytes differ');require(sum(not r['copied']for r in report['tensors'])==fresh and sum(r['copied']for r in report['tensors'])==copied,'exact selected/copied coverage differs')
  report.update(status='converted',candidate_sha256=digest(partial),copied_tensors=copied,original_bf16_to_q6_tensors=fresh,tied_q8_head_copied_exact=True,protected474_copied_exact=True,independent_q6_dequant_bf16_verified=True,effective_bf16_tensor_sha256={r['name']:r['bf16_sha256']for r in report['tensors']if r['type']!=0});partial.rename(output);save();return report
 except Exception:
  report['error']='bounded original-BF16 conversion failed; partial candidate never admitted';save();raise

def main():
 a=argparse.ArgumentParser(description=__doc__)
 for name in ('base','checkpoint','oracle','binary','output','report'):a.add_argument('--'+name,type=Path,required=True)
 p=a.parse_args();require(os.environ.get('GITHUB_ACTIONS')=='true'or Path('/kaggle/working').is_dir(),'hosted conversion only');convert(p.base,p.checkpoint,p.oracle,p.binary,p.output,p.report)
if __name__=='__main__':main()
