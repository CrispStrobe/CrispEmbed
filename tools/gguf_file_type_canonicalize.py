#!/usr/bin/env python3
"""Authenticated, four-byte legacy-Q4 file-header repair; tensor bytes untouched."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

SCHEMA='crispembed-gguf-file-type-canonicalization-v1'
FAMILIES={'josev','ultrabert','moedern','unlimited-ocr2','minicpm-ocr'}


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while b:=f.read(4*2**20):h.update(b)
    return h.hexdigest()


def span(f,offset,count):
    f.seek(offset);h=hashlib.sha256()
    while count:
        data=f.read(min(count,4*2**20));require(data,'truncated authenticated span');h.update(data);count-=len(data)
    return h.hexdigest()


def parse(path,storage_types=None):
    path=Path(path);require(path.is_file()and not path.is_symlink()and 24<path.stat().st_size<=8*2**30,'regular bounded GGUF required')
    size=path.stat().st_size;meta={};tensors={}
    supported={0:(1,4),1:(1,2),8:(32,34),12:(256,144)}if storage_types is None else storage_types
    with path.open('rb')as f:
        def take(n):
            require(type(n)is int and n>=0 and f.tell()+n<=min(size,64*2**20),'bounded header/truncation')
            b=f.read(n);require(len(b)==n,'header truncated');return b
        def number(fmt):return struct.unpack('<'+fmt,take(struct.calcsize('<'+fmt)))[0]
        def text(keep=True):
            n=number('Q');require(n<=2**20,'string exceeds1MiB')
            b=take(n)
            return b.decode('utf-8')if keep else None
        formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
        def value(kind):
            if kind==8:return text()
            if kind==9:
                sub,n=number('I'),number('Q');require(sub in formats or sub==8,'unknown/nested array')
                require(n<=1048576,'array bound')
                if sub==8:
                    for _ in range(n):text(False)
                else:take(n*struct.calcsize('<'+formats[sub]))
                return dict(element=sub,count=n)
            require(kind in formats,'unknown metadata kind');v=number(formats[kind])
            require(not isinstance(v,float)or math.isfinite(v),'nonfinite scalar')
            require(kind!=7 or v in(0,1),'invalid bool');return v
        require(take(4)==b'GGUF'and number('I')==3,'GGUFv3 little-endian required')
        count,keys=number('Q'),number('Q');require(1<=count<=8192 and keys<=4096,'tensor/metadata bounds')
        for _ in range(keys):
            start=f.tell();name=text();require(name and len(name)<=256 and '\0'not in name and name not in meta,'invalid/duplicate metadata key')
            kind=number('I');offset=f.tell();v=value(kind);end=f.tell()
            raw=span(f,start,end-start);f.seek(end)
            meta[name]=dict(kind=kind,value=v,offset=offset,raw_sha256=raw)
        if 'general.alignment'in meta:require(meta['general.alignment']['kind']==4 and meta['general.alignment']['value']==32,'alignment32 required')
        ranges=[]
        for _ in range(count):
            name=text();rank=number('I');require(name and len(name)<=256 and '\0'not in name and name not in tensors and 1<=rank<=4,'invalid/duplicate tensor/rank')
            ne=[number('Q')for _ in range(rank)];require(all(0<n<=2**31 for n in ne),'positive bounded axes')
            kind,offset=number('I'),number('Q');require(kind in supported,'only declared F32/F16/Q8_0/Q4_K storage accepted')
            block,width=supported[kind]
            require(ne[0]%block==0,'unaligned storage width')
            n=math.prod(ne)//block*width;require(n<=size and offset%32==0 and offset+n<=size,'bounded storage extent')
            tensors[name]=dict(stored_shape=ne,storage_rank=rank,type=kind,offset=offset,bytes=n);ranges.append((offset,offset+n,name))
        at=f.tell();data=(at+31)//32*32;require(not any(take(data-at)),'nonzero header padding');end=0
        for begin,finish,name in sorted(ranges):
            require(begin==(end+31)//32*32 and data+finish<=size,'gap/overlap/truncated storage')
            f.seek(data+end);require(not any(f.read(begin-end)),'nonzero tensor padding')
            tensors[name]['data_offset']=data+begin;end=finish
        require(0<=size-data-end<=31,'unexpected tail');f.seek(data+end);require(not any(f.read()),'nonzero tail')
    return dict(bytes=size,data_offset=data,metadata=meta,tensors=tensors)


def get(value,path):
    require(isinstance(path,list)and 1<=len(path)<=8 and all((isinstance(p,str)and p)or(type(p)is int and p>=0)for p in path),'explicit report JSON path required')
    for key in path:
        require((isinstance(value,dict)and isinstance(key,str)and key in value)or
                (isinstance(value,list)and type(key)is int and key<len(value)),'producer report field missing');value=value[key]
    return value


def descriptor(cfg,report_path):
    require(cfg['schema']=='crispembed-gguf-file-type-input-v1'and cfg['reviewed']is True and cfg['family']in FAMILIES,'reviewed explicit model descriptor required')
    require(isinstance(cfg['recipe'],str)and 0<len(cfg['recipe'])<=128,'explicit frozen recipe required')
    require(re.fullmatch('[0-9a-f]{40}',cfg['producer_source'])is not None,'exact producer source required')
    for pin in (cfg['source_model'],cfg['producer_report']):
        require(type(pin['bytes'])is int and pin['bytes']>0 and re.fullmatch('[0-9a-f]{64}',pin['sha256'])is not None,'explicit artifact SHA/bytes required')
    p=Path(report_path);pin=cfg['producer_report']
    require(p.is_file()and not p.is_symlink()and p.stat().st_size==pin['bytes']and pin['bytes']<=8*2**20 and sha(p)==pin['sha256'],'producer report identity differs')
    report=json.loads(p.read_text())
    require(get(report,cfg['report_model_sha256_path'])==cfg['source_model']['sha256']and
            get(report,cfg['report_model_bytes_path'])==cfg['source_model']['bytes'],'producer/model receipt identity differs')
    if cfg.get('report_recipe_path')is not None:require(get(report,cfg['report_recipe_path'])==cfg['recipe'],'producer recipe differs')
    inventory=cfg['inventory'];require(isinstance(inventory,list)and 1<=len(inventory)<=8192,'exact model inventory required')
    for row in inventory:
        require(isinstance(row['name'],str)and 0<len(row['name'])<=256 and '\0'not in row['name']and
                type(row['type'])is int and row['type']in(0,1,8,12)and type(row['bytes'])is int and row['bytes']>0 and
                isinstance(row['stored_shape'],list)and 1<=len(row['stored_shape'])<=4 and
                all(type(v)is int and 0<v<=2**31 for v in row['stored_shape'])and
                re.fullmatch('[0-9a-f]{64}',row['sha256'])is not None,'typed descriptor inventory required')
    require(len({r['name']for r in inventory})==len(inventory),'duplicate descriptor tensor')
    require(isinstance(cfg['architecture'],str)and cfg['architecture']and isinstance(cfg['profile_markers'],dict)and cfg['profile_markers'],'explicit architecture/profile markers required')
    require(not({'general.architecture','general.file_type','general.quantization_version'}&set(cfg['profile_markers'])),'reserved profile marker override')
    return cfg


def canonicalize(source,output,cfg,report_path):
    """Caller authenticates this tool's closed source before invoking on payloads."""
    cfg=descriptor(cfg,report_path);source,output=Path(source),Path(output);partial=output.with_name(output.name+'.partial')
    require(not output.exists()and not output.is_symlink()and not partial.exists()and not partial.is_symlink(),'fresh destination required; source never mutated')
    pin=cfg['source_model'];require(source.is_file()and not source.is_symlink()and source.stat().st_size==pin['bytes']and sha(source)==pin['sha256'],'exact reviewed source model differs')
    a=parse(source);m=a['metadata'];field=m.get('general.file_type',{})
    require(field.get('kind')==4 and field.get('value')==12,'only known legacy Q4 UINT32 header12 may migrate')
    require(m.get('general.quantization_version',{}).get('kind')==4 and m['general.quantization_version']['value']==2,'quantization version differs')
    markers={'general.architecture':cfg['architecture'],**cfg['profile_markers']}
    for key,val in markers.items():require(key in m and m[key]['value']==val,'reviewed source profile differs: '+key)
    expected={r['name']:r for r in cfg['inventory']};require(set(expected)==set(a['tensors']),'exact tensor inventory differs')
    require(any(r['type']==12 for r in a['tensors'].values()),'real Q4_K storage required; no blind Q3/Q6 header mapping')
    tensor_rows=[]
    with source.open('rb')as f:
        for name,row in a['tensors'].items():
            e=expected[name]
            require(row['stored_shape']==e['stored_shape']and row['type']==e['type']and row['bytes']==e['bytes'],'tensor descriptor differs: '+name)
            logical=e['semantic_shape'];require(isinstance(logical,list)and 1<=len(logical)<=4 and all(type(v)is int and v>0 for v in logical),'semantic axes required')
            stored=row['stored_shape']+[1]*(4-len(row['stored_shape']));semantic=logical+[1]*(4-len(logical))
            if cfg['family']=='moedern'and row['type']==12:
                require(semantic[1:]==stored[1:]and stored[0]==((semantic[0]+255)//256)*256,'declared MoE semantic/padding shape differs')
            else:require(semantic==stored,'semantic/storage axes differ')
            actual=span(f,row['data_offset'],row['bytes']);require(actual==e['sha256'],'tensor payload differs: '+name)
            tensor_rows.append(dict(name=name,semantic_shape=logical,stored_shape=row['stored_shape'],storage_rank=row['storage_rank'],
                type=row['type'],offset=row['offset'],bytes=row['bytes'],source_sha256=actual,canonical_sha256=actual))
        offset=field['offset'];prefix=span(f,0,offset);suffix=span(f,offset+4,a['bytes']-offset-4)
    output.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb')as left,partial.open('xb')as right:
        ls,rs=os.fstat(left.fileno()),os.fstat(right.fileno())
        require((ls.st_dev,ls.st_ino)!=(rs.st_dev,rs.st_ino),'source/destination alias')
        while block:=left.read(4*2**20):right.write(block)
        right.seek(offset);right.write(struct.pack('<I',14));right.flush();os.fsync(right.fileno())
    b=parse(partial)
    require(a['bytes']==b['bytes']and a['data_offset']==b['data_offset']and a['tensors']==b['tensors'],'canonical tensor/header layout changed')
    require(set(m)==set(b['metadata']),'metadata keys changed')
    for key in m:
        if key!='general.file_type':require(m[key]==b['metadata'][key],'other logical/raw metadata changed: '+key)
    require(b['metadata']['general.file_type']['kind']==4 and b['metadata']['general.file_type']['value']==14 and b['metadata']['general.file_type']['offset']==offset,'canonical header readback differs')
    with partial.open('rb')as f:
        require(span(f,0,offset)==prefix and span(f,offset+4,b['bytes']-offset-4)==suffix,'off-span byte mutation')
        for row in tensor_rows:require(span(f,b['tensors'][row['name']]['data_offset'],row['bytes'])==row['source_sha256'],'canonical tensor bytes changed')
    require(sha(source)==pin['sha256'],'source changed during migration')
    canonical=dict(bytes=b['bytes'],sha256=sha(partial));partial.chmod(0o444);partial.replace(output)
    other={k:v['raw_sha256']for k,v in m.items()if k!='general.file_type'}
    return dict(schema=SCHEMA,completed=True,family=cfg['family'],recipe=cfg['recipe'],producer_source=cfg['producer_source'],
        producer_report=cfg['producer_report'],source_model=pin,canonical_model=canonical,
        allowed_changes=[dict(key='general.file_type',kind='UINT32',offset=offset,old=12,new=14)],
        unchanged_prefix_sha256=prefix,unchanged_suffix_sha256=suffix,
        unchanged_metadata_sha256=hashlib.sha256(json.dumps(other,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        tensors=tensor_rows,payloads_byte_exact=True,semantic_shapes_unchanged=True,source_preserved=True,
        metadata_only_bridge=True,numerical_qualified=False,publication_qualified=False,original_rerun=False,
        input_descriptor_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest())


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from gguf_file_type_contract import verify
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for name in ('source','output','descriptor','producer-report','report'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();profile=verify(a.reviewed_source)
    require(os.environ.get('GITHUB_ACTIONS')=='true','hosted canonicalization required')
    result=dict(schema=SCHEMA,completed=False,runtime_source=a.reviewed_source)
    try:
        result=canonicalize(a.source,a.output,json.loads(a.descriptor.read_text()),a.producer_report)
        result.update(runtime_source=a.reviewed_source,input_descriptor_file_sha256=sha(a.descriptor),
                      source_profile_sha256=sha(Path(__file__).resolve().parents[1]/'docs/gguf-standard-file-type-v1.json'))
    except Exception as error:result['error']=type(error).__name__+': '+str(error);raise
    finally:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        temporary=a.report.with_name(a.report.name+'.partial')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(a.report)
