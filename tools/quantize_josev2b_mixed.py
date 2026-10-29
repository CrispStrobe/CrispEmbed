#!/usr/bin/env python3
"""Versioned A/B/C Josev storage experiment; actual existing C quantizer only.

The bounded GGUF parser below is adapted from quantize_josev2b_q4k.py at9c475d0d;
only F16 storage acceptance/extent is added. Historical converter is unchanged.
No framework imports or inference. Production entry authenticates the full F32.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from gguf_file_type_values import q4_k_file_type
import quantize_josev2b_q4k as old
SOURCE=old.SOURCE
PARAMETERS=old.PARAMETERS
OLD_FORMAT=old.OLD_FORMAT
NEW_FORMAT=old.NEW_FORMAT
require=old.require
hosted=old.hosted
digest=old.digest
hash_range=old.hash_range
scalar=old.scalar
schema=old.schema
COUNTS={'A':{0:292,1:78,12:72},'B':{0:316,1:78,12:48},'C':{0:316,1:118,12:8}}
EXPECTED_BYTES={'A':3613774304,'B':4651864544,'C':5375381984}
TYPES={0:'F32',1:'F16',12:'Q4_K'}


def recipe(rows,identity):
    require(identity in COUNTS,'unknown immutable recipe')
    selected=old.recipe(rows) # Authenticate all150 original projection names/widths.
    result={}
    for name in rows:
        if selected[name]==0:kind=0
        elif '.mlp.'not in name:kind=1
        elif identity=='A':kind=12
        elif '.down_proj.'in name:kind=0
        elif identity=='B':kind=12
        else:kind=12 if int(name.split('.')[2])>=20 else 1
        result[name]=kind
    require({t:sum(k==t for k in result.values())for t in TYPES}==COUNTS[identity],'role counts differ')
    return result


def f16_bits(bits):
    """Independent integer IEEE754 F32→binary16, nearest/ties-even, finite only.

    Preserve signed zero/subnormals. Reject NaN/Inf and finite overflow, rather
    than accepting a nonfinite quantizer result. No ggml conversion is reused.
    """
    sign=(bits>>16)&0x8000;exponent=(bits>>23)&255;fraction=bits&0x7fffff
    require(exponent!=255,'nonfinite source F32')
    unbiased=exponent-127
    if exponent==0:return sign # Every F32 subnormal is below half a F16 subnormal.
    significand=(1<<23)|fraction
    shift=13 if unbiased>=-14 else -unbiased-1
    if shift>25:return sign
    quotient,remainder=divmod(significand,1<<shift)
    halfway=1<<(shift-1)
    quotient+=remainder>halfway or (remainder==halfway and quotient&1)
    if unbiased<-14:return sign|quotient # Includes carry into smallest normal.
    target=unbiased+15
    if quotient==2048:target+=1;quotient=1024
    require(target<31,'finite F32 overflows F16')
    return sign|(target<<10)|(quotient&1023)


def vector_f32_bits(data):
    """At most65,536 values; integer finiteness test, no float conversion."""
    import numpy as np
    require(0<len(data)<=4*65536 and len(data)%4==0,'bounded F32 chunk required')
    bits=np.frombuffer(data,dtype='<u4')
    require(bool(np.all(((bits>>23)&255)!=255)),'nonfinite source F32')
    return bits


def vector_f16(data):
    """Same scalar integer IEEE oracle, vectorized with uint32/uint64 only.

    No astype(float16), ggml or native half converter. Per-chunk memory is bounded;
    returned bytes are just this ≤128KiB slice, never a materialized matrix.
    """
    import numpy as np
    bits=vector_f32_bits(data)
    sign=(bits>>16)&0x8000;exponent=(bits>>23)&255
    unbiased=exponent.astype(np.int32)-127
    shift=np.where(unbiased>=-14,13,-unbiased-1)
    active=(exponent!=0)&(shift<=25)
    safe=np.where(active,shift,13).astype(np.uint64)
    significand=((bits&0x7fffff)|(1<<23)).astype(np.uint64)
    divisor=np.left_shift(np.uint64(1),safe)
    quotient=np.right_shift(significand,safe)
    remainder=significand&(divisor-np.uint64(1));halfway=divisor>>np.uint64(1)
    quotient+=((remainder>halfway)|((remainder==halfway)&((quotient&np.uint64(1))!=0))).astype(np.uint64)
    target=unbiased+15+(quotient==2048).astype(np.int32)
    normal=active&(unbiased>=-14);subnormal=active&(unbiased<-14)
    require(bool(np.all(target[normal]<31)),'finite F32 overflows F16')
    result=sign.copy()
    result[subnormal]|=quotient[subnormal].astype(np.uint32)
    result[normal]|=(target[normal].astype(np.uint32)<<10)|(quotient[normal].astype(np.uint32)&1023)
    return result.astype('<u2').tobytes()


def source_hash_and_half(stream,offset,count,half_stream=None,half_offset=0,*,vector=False):
    stream.seek(offset)
    if half_stream is not None:half_stream.seek(half_offset)
    h=hashlib.sha256();rounded=hashlib.sha256();left=count
    while left:
        data=stream.read(min(left,4*65536));require(data and len(data)%4==0,'truncated source F32')
        h.update(data);left-=len(data)
        if vector:
            if half_stream is None:vector_f32_bits(data)
            else:expected=vector_f16(data)
        else:
            bits=list(struct.iter_unpack('<I',data))
            require(all((x[0]>>23)&255!=255 for x in bits),'nonfinite source F32')
            if half_stream is not None:
                expected=bytearray(2*len(bits))
                for i,(value,)in enumerate(bits):struct.pack_into('<H',expected,2*i,f16_bits(value))
        if half_stream is not None:
            actual=half_stream.read(len(expected))
            require(actual==expected,'independent IEEE F16 rounding differs')
            rounded.update(expected)
    return h.hexdigest(),rounded.hexdigest() if half_stream is not None else None


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
            require(kind in (0,1,12)and (kind!=12 or ne[0]%256==0),'only F32/F16/real aligned Q4_K supported')
            count=elements*4 if kind==0 else elements*2 if kind==1 else elements//256*144
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



def verify_conversion(source,output,expected,identity,*,seal=False,vector=False):
    require(len(expected)==442 and sum(math.prod(s)for s in expected.values())==PARAMETERS,'exact442 semantic inventory required')
    a,b=parse(source,expected),parse(output,expected);wanted=recipe(expected,identity)
    for p in (a,b):
        scalar(p,'general.architecture',8,'josev2b_systemone');scalar(p,'general.license',8,'apache-2.0')
        scalar(p,'josev2b.packed.format',8,OLD_FORMAT)
    require(b['metadata'].get('general.file_type',{}).get('kind')==4 and
            q4_k_file_type(b['metadata']['general.file_type']['value']),'unsupported Q4_K file-type metadata')
    scalar(b,'general.quantization_version',4,2)
    changed={'general.file_type','general.quantization_version'}
    require(set(b['metadata'])==set(a['metadata'])|changed,'metadata names differ')
    for name in set(a['metadata'])-changed:
        require(a['metadata'][name]['raw_sha256']==b['metadata'][name]['raw_sha256'],'component/tokenizer metadata changed: '+name)
    records=[]
    with Path(source).open('rb')as left,Path(output).open('rb')as right:
        for name in expected:
            x,y=a['tensors'][name],b['tensors'][name]
            require(x['type']==0 and y['type']==wanted[name],'actual recipe type differs: '+name)
            original,half=source_hash_and_half(left,x['data_offset'],x['bytes'],right if y['type']==1 else None,y['data_offset'],vector=vector)
            actual=hash_range(right,y['data_offset'],y['bytes'],q4k=y['type']==12)
            require(y['type']!=0 or actual==original,'protected F32 bytes changed: '+name)
            require(y['type']!=1 or actual==half,'F16 hash differs')
            records.append(dict(name=name,semantic_shape=expected[name],source_storage_rank=len(x['shape']),
                                output_storage_rank=len(y['shape']),type=TYPES[y['type']],bytes=y['bytes'],
                                source_f32_sha256=original,stored_sha256=actual,
                                independent_ieee_f16_exact=y['type']==1,critical_f32=not old.projection(name)))
    counts={t:sum(r['type']==TYPES[t]for r in records)for t in TYPES}
    require(counts==COUNTS[identity]and b['bytes']<a['bytes'],'actual recipe compression/counts differ')
    require(sum(r['critical_f32']for r in records)==292,'critical292 coverage differs')
    if seal:
        position=b['metadata']['josev2b.packed.format']['string_offset']
        require(len(OLD_FORMAT)==len(NEW_FORMAT),'same-length format seal required')
        with Path(output).open('r+b')as f:
            f.seek(position);require(f.read(len(OLD_FORMAT))==OLD_FORMAT.encode(),'format changed before sealing')
            f.seek(position);require(f.write(NEW_FORMAT.encode())==len(NEW_FORMAT),'short seal');f.flush();__import__('os').fsync(f.fileno())
        final=parse(output,expected);scalar(final,'josev2b.packed.format',8,NEW_FORMAT)
        require(final['tensors']==b['tensors']and final['bytes']==b['bytes'],'seal changed payload extents')
        for name in set(b['metadata'])-{'josev2b.packed.format'}:
            require(final['metadata'][name]['raw_sha256']==b['metadata'][name]['raw_sha256'],'seal changed other metadata')
    return dict(recipe=identity,tensors=records,counts={TYPES[t]:n for t,n in counts.items()},parameters=PARAMETERS,
                output_bytes=b['bytes'],general_file_type=b['metadata']['general.file_type']['value'],sealed=seal,critical292_f32_byte_exact=True,independent_f16_rounding_exact=True,
                semantic_shapes_unchanged=True,publication_qualified=False)


def quantize(source,output,executable,identity):
    hosted();source,output,executable=map(Path,(source,output,executable))
    require(source.is_file()and not source.is_symlink()and source.stat().st_size==SOURCE['bytes']and digest(source)==SOURCE['sha256'],
            'immutable full published F32 differs')
    require(identity in COUNTS and executable.is_file()and not output.exists(),'fresh reviewed recipe/quantizer required')
    expected=schema();wanted=recipe(expected,identity)
    args=[str(executable),str(source),str(output),'q4_k']
    for name,kind in wanted.items():
        if kind!=12:args+=['--tensor-type',name+'='+TYPES[kind].lower()]
    output.parent.mkdir(parents=True,exist_ok=True);log=output.with_suffix('.quantize.log')
    env={k:v for k,v in __import__('os').environ.items()if not k.startswith('CRISPEMBED_')}
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    from ci_process import run
    process=run(args,log,env=env,timeout=2400)
    log.with_name(log.name+'.process.json').write_text(json.dumps(process,indent=2)+'\n')
    require(not process['timed_out']and process['exit']==0,'actual C quantizer rejected; logs retained')
    require('(fallback 'not in log.read_text(),'quantizer fallback forbidden')
    result=verify_conversion(source,output,expected,identity,seal=True,vector=True)
    require(result['output_bytes']==EXPECTED_BYTES[identity],'full candidate exact byte extent differs')
    return dict(schema='crispembed-josev-mixed-conversion-v1',completed=True,source=SOURCE,
                output_sha256=digest(output),quantizer_sha256=digest(executable),quantizer_log_sha256=digest(log),
                quantizer_process=process,**result)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('recipe',choices=COUNTS);p.add_argument('source');p.add_argument('output');p.add_argument('quantizer')
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    a.report.write_text(json.dumps(quantize(a.source,a.output,a.quantizer,a.recipe),indent=2)+'\n')
