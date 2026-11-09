"""Byte-preserving GGUF composition; never dequantizes, calibrates or runs inference."""
import hashlib
import json
from pathlib import Path
import re
import struct
from gguf_file_type_canonicalize import parse,require

RECIPE='lighton4b-q4k-gdn-out-q8-v1'
BASE_SHA='f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'
DONOR_SHA='5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1'
MANIFEST_SHA='895d1f509006c3f8ed599555e85033b7cec1064ca33699ef4ca1ac151f60c663'
# Byte-preserving composer adapted from reviewed e2beb641 tools/lighton4b_mixed_pack.py.
from gguf_file_type_canonicalize import sha
KEY='qwen35_ocr.mixed_quant_recipe'
def upgraded(name,row):
    match=re.fullmatch(r'l\.layers\.([0-9]|[12][0-9]|3[01])\.linear_attn\.out_proj\.weight',name)
    return len(row['stored_shape'])==2 and bool(match) and int(match.group(1))%4!=3
def metadata_fields():
    values={KEY:RECIPE,'qwen35_ocr.mixed_quant_base_sha256':BASE_SHA,'qwen35_ocr.mixed_quant_donor_sha256':DONOR_SHA,
      'qwen35_ocr.mixed_quant_manifest_sha256':MANIFEST_SHA,'qwen35_ocr.mixed_quant_selected_count':24,'qwen35_ocr.mixed_quant_payload_delta':125829120}
    return values
def string(value):
    value=value.encode();return struct.pack('<Q',len(value))+value

def metadata_end(path):
    with Path(path).open('rb')as stream:
        header=stream.read(24);magic,version,count,keys=struct.unpack('<4sIQQ',header);require(magic==b'GGUF'and version==3,'GGUFv3 required')
        def number(fmt):return struct.unpack('<'+fmt,stream.read(struct.calcsize('<'+fmt)))[0]
        def text():stream.seek(number('Q'),1)
        widths={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        for _ in range(keys):
            text();kind=number('I')
            if kind==8:text()
            elif kind==9:
                sub=number('I');n=number('Q')
                if sub==8:
                    for _ in range(n):text()
                else:stream.seek(widths[sub]*n,1)
            else:stream.seek(widths[kind],1)
            require(stream.tell()<=64*2**20,'metadata bound')
        end=stream.tell();stream.seek(24);metadata=stream.read(end-24)
        return magic,version,count,keys,metadata

def compose(q4_path,q8_path,output,expected):
    require(sha(q4_path)==BASE_SHA and sha(q8_path)==DONOR_SHA,'authenticated paired source files required')
    q4=parse(q4_path);q8=parse(q8_path);output=Path(output);require(not output.exists(),'fresh candidate required')
    require(KEY not in q4['metadata'] and KEY not in q8['metadata'],'input already mixed-tagged')
    require(set(q4['tensors'])==set(q8['tensors'])==set(expected['q4_k'])==set(expected['q8_0']),'paired tensor census differs')
    magic,version,count,keys,metadata=metadata_end(q4_path);records=[];offset=0;infos=[]
    for name,base in q4['tensors'].items():
        other=q8['tensors'][name];require(base['stored_shape']==other['stored_shape'],'tensor source geometry differs')
        recipe='q8_0'if upgraded(name,base)else'q4_k';row=(q8 if recipe=='q8_0'else q4)['tensors'][name];pin=expected[recipe][name]
        require(row['bytes']==pin['bytes']and row['stored_shape']==pin['shape'],'authenticated source tensor geometry differs')
        offset=(offset+31)//32*32;infos.append(string(name)+struct.pack('<I',row['storage_rank'])+struct.pack('<%dQ'%row['storage_rank'],*row['stored_shape'])+struct.pack('<IQ',row['type'],offset))
        records.append({'name':name,'source_recipe':recipe,'type':row['type'],'shape':row['stored_shape'],'bytes':row['bytes'],'sha256':pin['sha256'],'offset':offset,'source_offset':row['data_offset']});offset+=row['bytes']
    require(count==723 and sum(row['source_recipe']=='q8_0'for row in records)==24,'exact24 whole donor tensors required')
    require(all(key not in q4['metadata'] for key in metadata_fields()),'recipe metadata collision')
    added=b''.join(string(k)+(struct.pack('<I',8)+string(v)if isinstance(v,str)else struct.pack('<II',4,v))for k,v in metadata_fields().items());prefix=struct.pack('<4sIQQ',magic,version,count,keys+len(metadata_fields()))+metadata+added+b''.join(infos);padding=(-len(prefix))%32;data_start=len(prefix)+padding
    temporary=output.with_suffix(output.suffix+'.partial');require(not temporary.exists(),'fresh partial candidate required')
    with Path(q4_path).open('rb')as a,Path(q8_path).open('rb')as b,temporary.open('xb')as dest:
        dest.write(prefix);dest.write(b'\0'*padding)
        for row in records:
            require(dest.tell()<=data_start+row['offset'],'output tensor overlap');dest.write(b'\0'*(data_start+row['offset']-dest.tell()));source=b if row['source_recipe']=='q8_0'else a;source.seek(row['source_offset']);remaining=row['bytes'];h=hashlib.sha256()
            while remaining:
                block=source.read(min(remaining,8*2**20));require(bool(block),'truncated source payload');dest.write(block);h.update(block);remaining-=len(block)
            require(h.hexdigest()==row['sha256'],'source payload digest differs')
        dest.flush();import os;os.fsync(dest.fileno())
    temporary.replace(output);verified=parse(output);require(verified['metadata'][KEY]['value']==RECIPE,'mixed recipe tag differs')
    with output.open('rb')as stream:
        for row in records:
            tensor=verified['tensors'][row['name']];require(tensor['bytes']==row['bytes']and tensor['type']==row['type'],'persisted output type/size differs');stream.seek(tensor['data_offset']);remaining=row['bytes'];h=hashlib.sha256()
            while remaining:
                block=stream.read(min(remaining,8*2**20));require(bool(block),'persisted candidate truncated');h.update(block);remaining-=len(block)
            require(h.hexdigest()==row['sha256'],'persisted candidate payload differs')
    require(all(verified['metadata'][k]['value']==v for k,v in metadata_fields().items()),'persisted recipe metadata differs')
    require(all(verified['metadata'][k]['raw_sha256']==v['raw_sha256'] for k,v in q4['metadata'].items()),'base source/tokenizer/geometry metadata changed')
    require({t:sum(row['type']==t for row in records)for t in (0,8,12)}=={0:474,8:25,12:224},'persisted profile census')
    return {'recipe':RECIPE,'tensor_count':len(records),'payload_bytes':offset,'data_offset':data_start,'bytes':output.stat().st_size,'metadata_file_type':verified['metadata']['general.file_type']['value'],'records':records}
