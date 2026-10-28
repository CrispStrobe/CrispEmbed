#!/usr/bin/env python3
"""Hosted-only protected Q4_K packing; no change to any F32 model or loader.

The existing C++ quantizer performs the numeric conversion. This module checks
all442 logical shapes/types and every critical tensor's bytes before changing
the output's same-length packed-format string in place. It needs only stdlib.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
SOURCE=dict(repo='cstr/Josev-2B-SystemOne-GGUF',revision='bffe62522f826b4c33f2e731d5b870ced5150a17',
            filename='josev2b-systemone-f32.gguf',bytes=7659180448,
            sha256='72684e39d0eed526b40c5150c78b2d881d324c39e16af48060b3d741f526ccf2')
OLD_FORMAT='crispembed-josev-packed-f32-v1'
NEW_FORMAT='crispembed-josev-packed-q4k-v1'
PROJECTIONS=('.mlp.gate_proj.weight','.mlp.up_proj.weight','.mlp.down_proj.weight',
             '.self_attn.q_proj.weight','.self_attn.k_proj.weight','.self_attn.v_proj.weight','.self_attn.o_proj.weight',
             '.linear_attn.in_proj_qkv.weight','.linear_attn.in_proj_z.weight','.linear_attn.out_proj.weight')
PARAMETERS=1912297796


def require(ok,why):
    if not ok:raise ValueError(why)


def hosted():
    require(os.environ.get('GITHUB_ACTIONS')=='true' or
            (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and
             os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')),'hosted execution required')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(4*2**20):h.update(block)
    return h.hexdigest()


def projection(name):return name.startswith('l.layers.')and any(name.endswith(s)for s in PROJECTIONS)


def schema():
    """Authoritative existing original-conversion shapes, not runtime guesses."""
    from josev2b_head_schema import tensor_schema
    backbone=json.loads((ROOT/'tests/results/josev2b/backbone-f32.json').read_text())['conversion']['tensors']
    rows={r['target']:list(reversed(r['shape']))for r in backbone}
    require(len(rows)==len(backbone)==320,'exact frozen320 backbone tensors required')
    rows.update({'head.'+name:list(reversed(shape))or[1]for name,shape in tensor_schema().items()})
    require(len(rows)==442 and sum(math.prod(s)for s in rows.values())==PARAMETERS,'exact442 parameter inventory required')
    return rows


def recipe(rows):
    selected={name for name,shape in rows.items()if projection(name)}
    require(len(selected)==150 and all(len(rows[n])==2 and rows[n][0]%256==0 for n in selected),
            'pinned150 Q4_K matrices/block widths required; no fallback')
    require(all(not n.startswith('head.')and n!='l.embed_tokens.weight'for n in selected),'critical tensor selected')
    return {name:12 if name in selected else 0 for name in rows}


def hash_range(stream,start,count,*,q4k=False):
    """Hash bounded chunks; Q4_K scales must decode to finite FP16 values."""
    stream.seek(start);h=hashlib.sha256();left=count
    chunk=144*4096 if q4k else 4*2**20
    while left:
        data=stream.read(min(chunk,left));require(data and len(data)<=left,'truncated tensor payload')
        if q4k:
            require(len(data)%144==0 and all(math.isfinite(a)and math.isfinite(b)for a,b in struct.iter_unpack('<ee140x',data)),
                    'nonfinite Q4_K scale/storage block')
        h.update(data);left-=len(data)
    return h.hexdigest()


def parse(path,expected):
    """Bounded GGUF header, exact logical ne axes and contiguous storage."""
    path=Path(path);require(path.is_file()and not path.is_symlink()and 24<path.stat().st_size<=8*2**30,'bounded regular packed GGUF required')
    size=path.stat().st_size;metadata={};tensors={}
    with path.open('rb')as f:
        def take(n):
            require(type(n)is int and n>=0 and f.tell()+n<=min(size,64*2**20),'header bound/truncation')
            data=f.read(n);require(len(data)==n,'truncated header');return data
        def number(fmt):return struct.unpack('<'+fmt,take(struct.calcsize('<'+fmt)))[0]
        def string(keep=True):
            n=number('Q');require(n<=2**20,'metadata string exceeds1MiB')
            if keep:return take(n).decode('utf-8')
            require(f.tell()+n<=min(size,64*2**20),'header string exceeds bound');f.seek(n,1)
        formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
        def value(kind):
            if kind==8:return string()
            if kind==9:
                element,count=number('I'),number('Q');require(element in formats or element==8,'unknown/nested metadata array')
                require(count<=1048576,'metadata array count exceeds bound')
                if element==8:
                    for _ in range(count):string(False)
                else:
                    n=count*struct.calcsize('<'+formats[element]);require(f.tell()+n<=min(size,64*2**20),'array exceeds header bound');f.seek(n,1)
                return dict(element=element,count=count)
            require(kind in formats,'unknown metadata scalar type');out=number(formats[kind])
            require(not isinstance(out,float)or math.isfinite(out),'nonfinite metadata scalar');return out
        require(take(4)==b'GGUF'and number('I')==3,'GGUFv3 required')
        nt,nk=number('Q'),number('Q');require(nt==len(expected)and 1<=nt<=442 and nk<=4096,'exact bounded tensor/metadata count required')
        for _ in range(nk):
            start=f.tell();key=string();require(key and len(key)<=256 and '\0'not in key and key not in metadata,'invalid/duplicate metadata key')
            kind=number('I');position=f.tell();v=value(kind);end=f.tell()
            raw=hash_range(f,start,end-start);f.seek(end)
            metadata[key]=dict(kind=kind,value=v,raw_sha256=raw,value_offset=position,
                               string_offset=position+8 if kind==8 else None)
        if 'general.alignment'in metadata:require(metadata['general.alignment']['kind']==4 and metadata['general.alignment']['value']==32,'alignment32 required')
        ranges=[]
        for _ in range(nt):
            name=string();rank=number('I');require(name in expected and name not in tensors and 1<=rank<=4,'unknown/duplicate tensor or rank')
            ne=[number('Q')for _ in range(rank)];require(ne+[1]*(4-rank)==expected[name]+[1]*(4-len(expected[name])),'semantic shape differs: '+name)
            elements=math.prod(ne);kind,offset=number('I'),number('Q')
            require(kind in (0,12)and (kind!=12 or ne[0]%256==0),'only F32/real aligned Q4_K supported')
            count=elements*4 if kind==0 else elements//256*144
            require(offset%32==0 and offset<=size and count<=size-offset,'tensor storage extent differs')
            tensors[name]=dict(shape=ne,type=kind,bytes=count,offset=offset)
            ranges.append((offset,offset+count,name))
        at=f.tell();start=(at+31)//32*32;require(not any(take(start-at)),'nonzero header padding');end=0
        for left,right,name in sorted(ranges):
            require(left==(end+31)//32*32 and start+right<=size,'overlapping/gapped/truncated tensors')
            f.seek(start+end);require(not any(f.read(left-end)),'nonzero tensor padding')
            tensors[name]['data_offset']=start+left;end=right
        require(0<=size-start-end<=31,'unexpected trailing payload');f.seek(start+end);require(not any(f.read()),'nonzero tail padding')
    return dict(bytes=size,data_start=start,metadata=metadata,tensors=tensors)


def scalar(parsed,name,kind,expected):
    row=parsed['metadata'].get(name,{})
    require(row.get('kind')==kind and row.get('value')==expected,'packed metadata differs: '+name)


def verify_conversion(source,output,expected,*,seal=False):
    """No mutation until all442 shapes/types and critical byte hashes pass."""
    require(len(expected)==442 and sum(math.prod(s)for s in expected.values())==PARAMETERS,'exact semantic parameter inventory required')
    a,b=parse(source,expected),parse(output,expected);wanted=recipe(expected)
    for p in (a,b):
        scalar(p,'general.architecture',8,'josev2b_systemone');scalar(p,'general.license',8,'apache-2.0')
        scalar(p,'josev2b.packed.format',8,OLD_FORMAT)
    scalar(b,'general.file_type',4,12);scalar(b,'general.quantization_version',4,2)
    changed={'general.file_type','general.quantization_version'}
    require(set(b['metadata'])==set(a['metadata'])|changed,'metadata names differ')
    for name in set(a['metadata'])-changed:
        require(a['metadata'][name]['raw_sha256']==b['metadata'][name]['raw_sha256'],'tokenizer/profile metadata changed: '+name)
    rows=[]
    with Path(source).open('rb')as left,Path(output).open('rb')as right:
        for name in expected:
            x,y=a['tensors'][name],b['tensors'][name]
            require(x['type']==0 and y['type']==wanted[name],'critical F32 / actual Q4_K recipe differs: '+name)
            original=hash_range(left,x['data_offset'],x['bytes'])
            actual=hash_range(right,y['data_offset'],y['bytes'],q4k=y['type']==12)
            require(y['type']==12 or actual==original,'critical F32 tensor bytes changed: '+name)
            rows.append(dict(name=name,semantic_shape=expected[name],source_storage_rank=len(x['shape']),
                             output_storage_rank=len(y['shape']),type='Q4_K'if y['type']==12 else'F32',
                             original_bytes=x['bytes'],bytes=y['bytes'],source_f32_sha256=original,stored_sha256=actual))
    require(len(rows)==442 and sum(r['type']=='Q4_K'for r in rows)==150 and b['bytes']<a['bytes'], 'actual compression/inventory differs')
    require(len(OLD_FORMAT.encode())==len(NEW_FORMAT.encode()),'format patch must be exactly same length')
    if seal:
        position=b['metadata']['josev2b.packed.format']['string_offset']
        with Path(output).open('r+b')as f:
            f.seek(position);require(f.read(len(OLD_FORMAT))==OLD_FORMAT.encode(),'format changed before sealing')
            f.seek(position);require(f.write(NEW_FORMAT.encode())==len(NEW_FORMAT),'short format write');f.flush();os.fsync(f.fileno())
        final=parse(output,expected);scalar(final,'josev2b.packed.format',8,NEW_FORMAT)
        for name in set(b['metadata'])-{'josev2b.packed.format'}:
            require(final['metadata'][name]['raw_sha256']==b['metadata'][name]['raw_sha256'],'sealing changed unrelated metadata')
        require(final['tensors']==b['tensors']and final['bytes']==b['bytes'],'sealing changed tensor/header extents')
    return dict(tensors=rows,q4k_tensors=150,f32_tensors=292,parameters=PARAMETERS,
                output_bytes=b['bytes'],sealed=seal,semantic_shapes_unchanged=True,critical_bytes_exact=True)


def quantize(source,output,executable):
    hosted();source,output,executable=map(Path,(source,output,executable))
    require(source.is_file()and not source.is_symlink()and source.stat().st_size==SOURCE['bytes']and digest(source)==SOURCE['sha256'],'immutable published F32 source differs')
    require(executable.is_file()and not output.exists(),'quantizer/fresh candidate required')
    expected=schema();source_header=parse(source,expected);wanted=recipe(expected)
    require(all(r['type']==0 for r in source_header['tensors'].values()),'all442 source tensors must be F32')
    output.parent.mkdir(parents=True,exist_ok=True);log=output.with_suffix('.quantize.log')
    args=[str(executable),str(source),str(output),'q4_k']
    args+=['--tensor-type','head.*=f32']
    for name,kind in wanted.items():
        if kind==0 and not name.startswith('head.'):args+=['--tensor-type',name+'=f32']
    env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')}
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    from ci_process import run
    process=run(args,log,env=env,timeout=1800)
    log.with_name(log.name+'.process.json').write_text(json.dumps(process,indent=2)+'\n')
    require(not process['timed_out']and process['exit']==0,'quantizer child rejected; process receipt retained')
    require('(fallback 'not in log.read_text(),'quantizer fallback rejected; not a real Q4_K recipe')
    result=verify_conversion(source,output,expected,seal=True)
    return dict(schema='crispembed-josev-q4k-conversion-v1',completed=True,source=SOURCE,
                output_sha256=digest(output),quantizer_sha256=digest(executable),quantizer_log_sha256=digest(log),quantizer_process=process,
                **result,publication_qualified=False,task_quality=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('quantizer');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    a.report.write_text(json.dumps(quantize(a.source,a.output,a.quantizer),indent=2)+'\n')
