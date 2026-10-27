#!/usr/bin/env python3
"""Hosted streaming repack of three reviewed Josev components; no tensor arithmetic.

The public CLI requires the exact442-tensor profile and reviewed composed proof.
Internal parser/repack primitives also serve small, model-free stdlib controls.
No package, framework, checkpoint, model registry or network access is performed.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROFILE_PATH='docs/josev2b-packed-f32-profile.json'
PROFILE_SHA='1523832a70fdcc6b510f35a38d5405362fd1e8a0b60f8127e3d54c995734d622'
FILE_CAP=8*2**30
HEADER_CAP=64*2**20
CHUNK=4*2**20
ALIGNMENT=32
SCALARS={0:('<B',1),1:('<b',1),2:('<H',2),3:('<h',2),4:('<I',4),5:('<i',4),6:('<f',4),7:('<B',1),10:('<Q',8),11:('<q',8),12:('<d',8)}
COMMON=frozenset(('general.license','general.alignment'))
REPLACED=frozenset(('general.architecture','general.name'))
OWN=(PROFILE_PATH,'tools/pack_josev2b_f32.py','tests/test_josev2b_packed_f32.py','docs/josev2b-packed-f32.md')
PROTECTED=('README.md','.gitmodules','Cargo.toml','docs/josev2b-source-contract.json','tests/results/josev2b/original-reference.json','tests/results/josev2b/original-input-freeze.json','src','ggml','CMakeLists.txt','cmake','VERSION','LICENSE','python','flutter','crispembed','crispembed-sys','wasm','examples','Cargo.lock','tools/josev2b_composed_contract.py','tools/josev2b_composed_panel.py','tools/josev2b_composed_diff.py','tools/ci-heavy/josev2b_composed.py','docs/josev2b-composed-source.json','docs/josev2b-composed-dependencies.json')


def require(condition,why):
    if not condition:raise ValueError(why)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(CHUNK):h.update(block)
    return h.hexdigest()


def profile():
    path=ROOT/PROFILE_PATH
    require(sha_file(path)==PROFILE_SHA,'Packed profile bytes differ')
    return json.loads(path.read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def source_guard():
    cfg=profile()
    for path in PROTECTED:
        require(git('rev-parse','HEAD:'+path)==git('rev-parse',cfg['base']+':'+path),'Frozen composed/runtime source changed: '+path)
    require(not git('status','--porcelain','--',*PROTECTED,*OWN),'Dirty packed/protected source')
    files={}
    for path in OWN:
        require(git('hash-object',path)==git('rev-parse','HEAD:'+path),'Uncommitted packer source: '+path)
        files[path]=sha_file(ROOT/path)
    for component in cfg['components'].values():
        if 'schema_receipt' in component:
            require(sha_file(ROOT/component['schema_receipt'])==component['schema_receipt_sha256'],'Reviewed tensor schema receipt changed')
    return dict(head=git('rev-parse','HEAD'),profile_sha256=PROFILE_SHA,files=files,
                protected_git_objects={p:git('rev-parse','HEAD:'+p)for p in PROTECTED})


@dataclass(frozen=True)
class KV:
    kind:int
    value:object
    raw:bytes


@dataclass(frozen=True)
class Tensor:
    name:str
    ne:tuple[int,...]
    offset:int
    nbytes:int
    kind:int=0


@dataclass
class GGUF:
    path:Path
    size:int
    data_start:int
    metadata:dict[str,KV]
    tensors:dict[str,Tensor]


class Reader:
    def __init__(self,stream,size):self.stream,self.size=stream,size
    def read(self,n):
        require(type(n)is int and 0<=n<=HEADER_CAP and self.stream.tell()+n<=min(HEADER_CAP,self.size),'GGUF header/string extent exceeds bound')
        data=self.stream.read(n);require(len(data)==n,'Truncated GGUF header');return data
    def unpack(self,fmt):return struct.unpack(fmt,self.read(struct.calcsize(fmt)))[0]
    def string(self,cap=16*2**20):
        length=self.unpack('<Q');require(length<=cap,'GGUF string length exceeds bound')
        return self.read(length).decode('utf-8')
    def value(self,kind):
        if kind==8:return self.string()
        if kind==9:
            element=self.unpack('<I');n=self.unpack('<Q')
            require(element in SCALARS or element==8,'Nested/unsupported GGUF array')
            require(n<=1000000,'GGUF array count exceeds bound')
            require(n*(8 if element==8 else SCALARS[element][1])<=min(HEADER_CAP,self.size)-self.stream.tell(),'GGUF array extent exceeds header bound')
            return (element,tuple(self.value(element)for _ in range(n)))
        require(kind in SCALARS,'Unsupported GGUF metadata type')
        value=self.unpack(SCALARS[kind][0])
        if kind==7:require(value in (0,1),'Noncanonical GGUF boolean')
        if kind in (6,12):require(math.isfinite(value),'Nonfinite GGUF metadata')
        return value


def zeros(stream,start,count):
    require(0<=count<ALIGNMENT,'Unexpected GGUF padding extent')
    stream.seek(start);require(stream.read(count)==b'\0'*count,'Nonzero GGUF padding')


def parse(path):
    path=Path(path)
    require(not path.is_symlink() and path.is_file() and 24<=path.stat().st_size<=FILE_CAP,'Bounded regular GGUF required')
    size=path.stat().st_size
    with path.open('rb')as stream:
        reader=Reader(stream,size)
        require(reader.read(4)==b'GGUF' and reader.unpack('<I')==3,'Only little-endian GGUFv3 accepted')
        count=reader.unpack('<Q');metadata_count=reader.unpack('<Q')
        require(count<=442 and metadata_count<=2048,'GGUF tensor/metadata count exceeds bound')
        metadata={}
        for _ in range(metadata_count):
            start=stream.tell();name=reader.string(512)
            require(name and '\0'not in name and name not in metadata,'Duplicate/invalid metadata key')
            kind=reader.unpack('<I');value=reader.value(kind);end=stream.tell()
            stream.seek(start);raw=stream.read(end-start);stream.seek(end)
            metadata[name]=KV(kind,value,raw)
        if 'general.alignment'in metadata:
            item=metadata['general.alignment'];require(item.kind==4 and item.value==ALIGNMENT,'Only explicit32-byte alignment accepted')
        tensors={}
        for _ in range(count):
            name=reader.string(256);rank=reader.unpack('<I')
            require(name and '\0'not in name and name not in tensors and 1<=rank<=4,'Duplicate/invalid tensor name or rank')
            ne=tuple(reader.unpack('<Q')for _ in range(rank))
            require(all(1<=n<=1048576 for n in ne),'Tensor dimensions exceed bound')
            nbytes=4*math.prod(ne);require(0<nbytes<=FILE_CAP,'Tensor element extent exceeds bound')
            kind=reader.unpack('<I');offset=reader.unpack('<Q')
            require(kind==0 and offset%ALIGNMENT==0,'Only aligned F32 tensors accepted')
            tensors[name]=Tensor(name,ne,offset,nbytes,kind)
        end=stream.tell();data_start=(end+31)//32*32
        require(data_start<=size,'Header alignment exceeds file extent');zeros(stream,end,data_start-end)
        previous=0
        for tensor in sorted(tensors.values(),key=lambda t:t.offset):
            expected=(previous+31)//32*32
            require(tensor.offset==expected and data_start+tensor.offset+tensor.nbytes<=size,'Tensor overlap/gap/truncated payload')
            zeros(stream,data_start+previous,tensor.offset-previous)
            previous=tensor.offset+tensor.nbytes
        end=data_start+previous
        require(size in (end,(end+31)//32*32),'Unexpected GGUF trailing bytes')
        zeros(stream,end,size-end)
    return GGUF(path,size,data_start,metadata,tensors)


def string(value):
    data=value.encode('utf-8');return struct.pack('<Q',len(data))+data


def kv(name,kind,value):
    if kind==8:encoded=string(value)
    elif kind in SCALARS:encoded=struct.pack(SCALARS[kind][0],value)
    else:raise ValueError('New packed metadata must be scalar')
    return KV(kind,value,string(name)+struct.pack('<I',kind)+encoded)


def merged_metadata(components,extra):
    """Preserve distinct component records verbatim; explicit general collisions only."""
    result={}
    for label,component in components.items():
        for name,item in component.metadata.items():
            if name in REPLACED:
                origin_name='josev2b.packed.'+label+'.'+name.replace('.','_')
                require(origin_name not in result,'Packed provenance key collision')
                result[origin_name]=kv(origin_name,item.kind,item.value)
                continue
            if name in result:
                require(name in COMMON and result[name].raw==item.raw,'Different or undeclared duplicate metadata: '+name)
            else:
                result[name]=item
    for name,item in extra.items():
        require(name not in result,'New packed metadata collides with component: '+name)
        result[name]=item
    require(result.get('general.license')is not None and result['general.license'].kind==8 and result['general.license'].value=='apache-2.0','Exact common Apache2 license required')
    return result


def hash_range(stream,start,count):
    stream.seek(start);h=hashlib.sha256()
    while count:
        block=stream.read(min(CHUNK,count));require(block,'Truncated F32 payload during readback')
        h.update(block);count-=len(block)
    return h.hexdigest()


def _repack(components,output,extra,expected,source_checks=None):
    """Internal streaming primitive; production pack supplies authenticated442 schema."""
    output=Path(output);partial=output.with_name(output.name+'.partial')
    require(not output.exists() and not partial.exists(),'Fresh packed output/partial required')
    metadata=merged_metadata(components,extra)
    tensors={};offset=0
    for label,component in components.items():
        require(set(component.tensors)==set(expected[label]),'Exact component tensor names required')
        for name,tensor in component.tensors.items():
            row=expected[label][name]
            require(name not in tensors and list(tensor.ne)==row['ne'] and tensor.nbytes==row['bytes'] and tensor.kind==0,'Duplicate tensor or exact F32 dimensions differ: '+name)
            offset=(offset+31)//32*32
            tensors[name]=(label,tensor,offset);offset+=tensor.nbytes
    output.parent.mkdir(parents=True,exist_ok=True)
    records={}
    with partial.open('xb')as target:
        target.write(b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(metadata)))
        for item in metadata.values():target.write(item.raw)
        for name,(_,tensor,off)in tensors.items():
            target.write(string(name)+struct.pack('<I',len(tensor.ne))+struct.pack('<'+'Q'*len(tensor.ne),*tensor.ne)+struct.pack('<IQ',0,off))
        require(target.tell()<=HEADER_CAP,'Packed header exceeds64MiB')
        data_start=(target.tell()+31)//32*32;target.write(b'\0'*(data_start-target.tell()))
        require(data_start+(offset+31)//32*32<=FILE_CAP,'Packed GGUF exceeds8GiB')
        for name,(label,tensor,off)in tensors.items():
            target.write(b'\0'*(data_start+off-target.tell()));remaining=tensor.nbytes;h=hashlib.sha256()
            with components[label].path.open('rb')as source:
                source.seek(components[label].data_start+tensor.offset)
                while remaining:
                    block=source.read(min(CHUNK,remaining));require(block,'Source tensor truncated during copy')
                    target.write(block);h.update(block);remaining-=len(block)
            digest=h.hexdigest();require(digest==expected[label][name]['f32_sha256'],'F32 source tensor bits differ: '+name)
            records[name]=dict(component=label,ne=list(tensor.ne),bytes=tensor.nbytes,f32_sha256=digest,output_offset=off)
        target.write(b'\0'*(-target.tell()%32))
    readback=parse(partial)
    require({k:v.raw for k,v in readback.metadata.items()}=={k:v.raw for k,v in metadata.items()},'Packed metadata byte readback differs')
    require(set(readback.tensors)==set(tensors),'Packed tensor readback coverage differs')
    with partial.open('rb')as stream:
        for name,tensor in readback.tensors.items():
            row=records[name]
            require(list(tensor.ne)==row['ne'] and tensor.nbytes==row['bytes'] and tensor.offset==row['output_offset'] and hash_range(stream,readback.data_start+tensor.offset,tensor.nbytes)==row['f32_sha256'],'Packed F32 tensor readback differs: '+name)
    if source_checks is not None:
        for label,component in components.items():
            require(sha_file(component.path)==source_checks[label],'Input changed during repack: '+label)
    # Link publishes exclusively after readback; never overwrite a racing output.
    os.link(partial,output);partial.unlink()
    return dict(bytes=output.stat().st_size,sha256=sha_file(output),tensors=records,
                metadata={k:dict(kind=v.kind,raw_record_sha256=hashlib.sha256(v.raw).hexdigest())for k,v in metadata.items()})


def reviewed_composition(cfg,path):
    pin=cfg['composed_review']
    require(pin['reviewed']is True and isinstance(pin['receipt_sha256'],str)and len(pin['receipt_sha256'])==64,'Explicit completed composed review pending; no component access')
    path=Path(path);require(not path.is_symlink()and path.is_file()and path.stat().st_size<=8*2**20,'Bounded composed review receipt required')
    require(sha_file(path)==pin['receipt_sha256'],'Reviewed composed receipt bytes differ')
    report=json.loads(path.read_text(encoding='utf-8'))
    require(report['completed']is True and report['source']==pin['source'] and report['numeric_and_complete_typed_parity']is True and report['compared_stages']==227 and report['failures']==[],'Reviewed composed outcome differs')
    return dict(run=pin['run'],source=pin['source'],receipt_sha256=pin['receipt_sha256'])


def pack(backbone,head,tokenizer,composed_receipt,output):
    require(os.environ.get('GITHUB_ACTIONS')=='true'and sys.platform.startswith('linux'),'Production pack is hosted Linux only; local fixtures use internal primitives')
    closure=source_guard();cfg=profile();review=reviewed_composition(cfg,composed_receipt)
    paths=dict(backbone=Path(backbone),head=Path(head),tokenizer=Path(tokenizer));components={}
    require(len(set(p.resolve()for p in paths.values()))==3,'Distinct component files required')
    for label,path in paths.items():
        pin=cfg['components'][label]
        require(not path.is_symlink()and path.is_file()and path.stat().st_size==pin['bytes']and sha_file(path)==pin['sha256'],'Exact reviewed component bytes differ: '+label)
        parsed=parse(path);architecture=parsed.metadata.get('general.architecture')
        require(architecture is not None and architecture.kind==8 and architecture.value==pin['architecture'],'Component architecture differs: '+label)
        components[label]=parsed
    require(sum(len(c.tensors)for c in components.values())==442 and not components['tokenizer'].tensors,'Exact320+122+0 tensor profile required')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(Path(output).parent).free>=sum(t.nbytes for c in components.values()for t in c.tensors.values())+HEADER_CAP,'Insufficient hosted output scratch')
    extra={'general.architecture':kv('general.architecture',8,cfg['architecture']),
           'general.name':kv('general.name',8,'Josev-2B native SystemOne text F32'),
           'josev2b.packed.format':kv('josev2b.packed.format',8,cfg['format']),
           'josev2b.packed.profile':kv('josev2b.packed.profile',8,cfg['profile'])}
    for label,pin in cfg['components'].items():
        key='josev2b.packed.'+label+'_gguf_sha256';extra[key]=kv(key,8,pin['sha256'])
    result=_repack(components,output,extra,{k:v['tensors']for k,v in cfg['components'].items()},
                   {k:v['sha256']for k,v in cfg['components'].items()})
    return dict(schema='crispembed-josev-packed-f32-v1',completed=True,source=closure,composed_review=review,
                architecture=cfg['architecture'],format=cfg['format'],profile=cfg['profile'],components={k:{f:v[f]for f in ('bytes','sha256','architecture')}for k,v in cfg['components'].items()},
                output=result,tensor_count=442,parameter_count=sum(t.nbytes//4 for c in components.values()for t in c.tensors.values()),
                tensor_math_performed=False,maximum_copy_bytes=CHUNK,header_bound_bytes=HEADER_CAP,
                full_model_qualified=False,release_qualified=False,limits='Byte-preserving F32 packaging only; packed borrowed loader/API and task quality remain unqualified.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('backbone','head','tokenizer','composed-receipt','output','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.report.exists(),'Fresh receipt required')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    try:result=pack(args.backbone,args.head,args.tokenizer,args.composed_receipt,args.output)
    except Exception as error:
        result=dict(schema='crispembed-josev-packed-f32-v1',completed=False,error_type=type(error).__name__,error=str(error),full_model_qualified=False,release_qualified=False)
        args.report.write_text(json.dumps(result,indent=2)+'\n');raise
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
