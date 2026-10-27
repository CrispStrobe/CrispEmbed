#!/usr/bin/env python3
"""Bounded byte-preserving original F32 + reviewed tokenizer metadata package.

Production entrypoint is hosted only. Internal primitives accept tiny authored
fixtures for stdlib controls; they never create replacement model/reference gold.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

MODEL={'bytes':5941568,'sha256':'9f9792f0ae76430496ac93951cc09a0804468ba3f22b4314b303aea85dfbe8fa'}
TOKENIZER={'bytes':1095840,'sha256':'3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1'}
COMPOSED='98af43c6780bbd11b1ec9ec4f26c7b2e3a2348507072408408bf726b2e89482e'
CHUNK=1024*1024
SIZES={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
FORMATS={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def require(ok,why):
 if not ok:raise ValueError(why)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(CHUNK):h.update(b)
 return h.hexdigest()
def string(value):
 b=value.encode();return struct.pack('<Q',len(b))+b
def scalar(key,value):return string(key)+struct.pack('<I',8)+string(value)
def parse(path):
 path=Path(path);size=path.stat().st_size;require(0<size<=16*CHUNK,'file bound differs')
 with path.open('rb')as f:
  def read(n):
   require(0<=n<=4*CHUNK and f.tell()+n<=min(size,4*CHUNK),'header bound/truncation differs');b=f.read(n);require(len(b)==n,'truncated header');return b
  def integer(fmt):return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
  def text():
   n=integer('Q');require(n<=CHUNK,'string budget');return read(n).decode('utf-8')
  def value(kind):
   if kind==8:return text()
   if kind==9:
    element=integer('I');n=integer('Q');require(element in SIZES or element==8,'nested/unknown array');require(n<=65536,'array count')
    return (element,[value(element)for _ in range(n)])
   require(kind in SIZES,'unknown metadata kind');v=integer(FORMATS[kind]);require(not isinstance(v,float)or math.isfinite(v),'nonfinite metadata');return v
  require(read(4)==b'GGUF'and integer('I')==3,'GGUFv3 required');nt,nk=integer('Q'),integer('Q');require(nt<=32 and nk<=128,'metadata/tensor count')
  metadata={}
  for _ in range(nk):
   start=f.tell();key=text();require(key and len(key)<=256 and '\0'not in key and key not in metadata,'metadata key differs');kind=integer('I');v=value(kind);end=f.tell();f.seek(start);raw=read(end-start);metadata[key]={'kind':kind,'value':v,'raw':raw}
  alignment=metadata.get('general.alignment');require(alignment is None or alignment['kind']==4 and alignment['value']==32,'alignment must be32')
  tensors=[];names=set()
  for _ in range(nt):
   name=text();require(name and len(name)<=256 and '\0'not in name and name not in names,'tensor name differs');names.add(name)
   rank=integer('I');require(1<=rank<=4,'tensor rank');shape=[integer('Q')for _ in range(rank)];require(all(0<n<=30002 for n in shape)and math.prod(shape)<=2*CHUNK,'tensor dimensions');require(integer('I')==0,'lossless F32 only');offset=integer('Q');tensors.append({'name':name,'shape':shape,'offset':offset,'bytes':4*math.prod(shape)})
  end=f.tell();data_start=(end+31)//32*32;require(not any(read(data_start-end)),'nonzero header padding');position=0
  for t in sorted(tensors,key=lambda t:t['offset']):
   aligned=(position+31)//32*32;require(t['offset']==aligned and data_start+aligned+t['bytes']<=size,'tensor offset/extent differs')
   f.seek(data_start+position);require(not any(f.read(aligned-position)),'nonzero tensor padding');position=aligned+t['bytes']
  require(0<=size-data_start-position<=31,'trailing extent differs');f.seek(data_start+position);require(not any(f.read()),'nonzero trailing padding')
 return {'path':path,'bytes':size,'data_start':data_start,'metadata':metadata,'tensors':tensors}

def _pack(model,tokenizer,output,extra):
 """No inference/conversion; preserve raw metadata records and tensor bytes."""
 metadata=dict(model['metadata'])
 for name,row in tokenizer['metadata'].items():
  if name=='general.architecture':continue
  if name in metadata:
   require(name in ('general.license','general.alignment')and metadata[name]['raw']==row['raw'],'undeclared metadata collision: '+name)
  else:metadata[name]=row
 for name,value in extra.items():
  require(name not in metadata,'packed provenance collision');metadata[name]={'kind':8,'value':value,'raw':scalar(name,value)}
 require(len(metadata)<=128,'packed metadata budget');tensors=model['tensors'];offset=0;table=[]
 for row in tensors:
  offset=(offset+31)//32*32;table.append(dict(row,packed_offset=offset));offset+=row['bytes']
 output=Path(output);require(not output.exists(),'fresh packed output required');partial=output.with_name(output.name+'.partial');identities={}
 with partial.open('xb')as target,model['path'].open('rb')as source:
  target.write(b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(metadata)))
  for row in metadata.values():target.write(row['raw'])
  for row in table:
   target.write(string(row['name'])+struct.pack('<I',len(row['shape']))+b''.join(struct.pack('<Q',n)for n in row['shape'])+struct.pack('<IQ',0,row['packed_offset']))
  target.write(b'\0'*(-target.tell()%32));start=target.tell();require(start<=4*CHUNK,'packed header budget')
  for row in table:
   target.write(b'\0'*(start+row['packed_offset']-target.tell()));source.seek(model['data_start']+row['offset']);remaining=row['bytes'];h=hashlib.sha256()
   while remaining:
    b=source.read(min(CHUNK,remaining));require(b,'tensor source truncated');target.write(b);h.update(b);remaining-=len(b)
   identities[row['name']]={'shape':row['shape'],'bytes':row['bytes'],'sha256':h.hexdigest()}
 readback=parse(partial);require({k:v['raw']for k,v in readback['metadata'].items()}=={k:v['raw']for k,v in metadata.items()},'metadata byte readback differs')
 with partial.open('rb')as f:
  for row in readback['tensors']:
   f.seek(readback['data_start']+row['offset']);h=hashlib.sha256();remaining=row['bytes']
   while remaining:
    b=f.read(min(CHUNK,remaining));require(b,'packed tensor truncated');h.update(b);remaining-=len(b)
   require(identities[row['name']]=={'shape':row['shape'],'bytes':row['bytes'],'sha256':h.hexdigest()},'packed tensor readback differs')
 partial.rename(output)
 return {'bytes':output.stat().st_size,'sha256':digest(output),'tensors':identities,'maximum_copy_bytes':CHUNK,'tensor_math_performed':False}

def pack(model,tokenizer,report,output):
 require((os.environ.get('GITHUB_ACTIONS')=='true' or bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE')))and sys.platform.startswith('linux'),'production repack hosted Linux only')
 from importlib.util import spec_from_file_location,module_from_spec
 path=Path(__file__).resolve().parent/'moedern-public/root_contract.py';spec=spec_from_file_location('moedern_public_packer_guard',path);guard=module_from_spec(spec);spec.loader.exec_module(guard);guard.source() # versioned root closure; draft guard unchanged
 require(digest(report)==COMPOSED,'exact reviewed composition report required');j=json.loads(Path(report).read_text());require(j['completed']is True and j['text_model_parity']is True and not j['failures']and j['source']=='1a053116908224cf1c9358d7d24c85ba36e82eff'and j['coverage']=={'old_final_vectors':20,'old_maxsim':3,'cases':10},'reviewed composition outcome differs')
 for path,pin in [(model,MODEL),(tokenizer,TOKENIZER)]:require(Path(path).is_file()and not Path(path).is_symlink()and Path(path).stat().st_size==pin['bytes']and digest(path)==pin['sha256'],'authenticated component differs')
 m,t=parse(model),parse(tokenizer);require(len(m['tensors'])==32 and not t['tensors']and sum(row['bytes']for row in m['tensors'])==1484448*4,'32F32 parameter inventory differs')
 require(m['metadata']['general.architecture']['value']=='modernbert_hash_moe_v10'and t['metadata']['general.architecture']['value']=='moedern_unigram_tokenizer','component architectures differ')
 extra={'moedern.packed.format':'crispembed-moedern-text-f32-v1','moedern.packed.tokenizer_architecture':'moedern_unigram_tokenizer','moedern.packed.model_sha256':MODEL['sha256'],'moedern.packed.tokenizer_sha256':TOKENIZER['sha256'],'moedern.packed.original_sha256':'b50ca1369f454d61b8c95583d8eb77a847c47ede29e52ed66b575784f4e4660f','moedern.packed.composed_source':j['source'],'moedern.packed.composed_report_sha256':COMPOSED}
 return dict(schema='crispembed-moedern-packed-f32-v1',completed=True,output=_pack(m,t,output,extra),original_recomputed=False,model_qualified=False,public_api_qualified=False)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--composed-report',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
 result=pack(a.model,a.tokenizer,a.composed_report,a.output);Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n')
