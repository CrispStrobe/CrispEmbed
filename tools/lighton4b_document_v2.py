#!/usr/bin/env python3
"""Document-v2 packaging and bounded diagnostic consumption; never a forward.

This version is independent of document-v1. Actual original/native outputs are
not admitted. Synthetic packaging/consumption is executable now; readiness
reports distinguish that proof from actual4B outputs and large retention.
"""
import hashlib
import argparse
import json
import math
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_document_parity as v1
import lighton4b_sharded_transport as transport
from gguf_file_type_canonicalize import parse as parse_gguf

ADMITTED_ORIGINAL_PRODUCERS=()
ADMITTED_NATIVE_PRODUCERS=()
MAX_STAGE_BYTES=64*2**20*4


def require(value,message):
    if not value:raise ValueError(message)


def ordered_stages(positions):
    result=[row for row in v1.stages(positions)if not row[0].startswith('generated.')]
    for step in range(positions):
        prefix=f'generated.{step}.'
        result.append((prefix+'decoder.position_ids','exact'))
        for i in range(32):
            result += [(prefix+f'cache.{i}.'+field,'float')for field in
                       (('keys','values')if i%4==3 else('conv_states','recurrent_states'))]
        result += [(prefix+'decoder.norm','float'),(f'generated.scores.{step}','float')]
    return result


class LocalReader:
    def __init__(self,root):self.root=Path(root)
    def read(self,file,offset,count):
        transport.safe_name(file);path=self.root/file
        require(path.resolve().is_relative_to(self.root.resolve())and not path.is_symlink(),'contained regular stage source required')
        require(type(offset)is int and offset>=0 and type(count)is int and 0<count<=MAX_STAGE_BYTES,'bounded stage range')
        with path.open('rb')as stream:stream.seek(offset);data=stream.read(count)
        require(len(data)==count,'stage range truncated');return data


class RemoteReader:
    """One verified part cache per side; never restore a complete capture."""
    def __init__(self,backend,prefix,revision,manifest_sha256,cache):
        self.backend=backend;self.prefix=prefix;self.revision=revision;self.cache=Path(cache)
        require(not self.cache.exists(),'fresh remote part cache required');self.cache.mkdir(parents=True)
        path=self.cache/'transport.json';backend.get(prefix+'/manifest.json',revision,path)
        require(transport.digest(path)==manifest_sha256,'source transport manifest hash differs')
        self.manifest=transport.read_json(path)
        require(self.manifest['schema']=='lighton4b-original-sharded-transport/v1'and
                self.manifest['case_protocol_sha256']==transport.PROTOCOL_SHA,'source transport identity differs')
        rows=self.manifest['files'];require(len(rows)<=67*256+4 and len({r['file']for r in rows})==len(rows),'source file inventory')
        require(sum(r['bytes']for r in rows)<=transport.MAX_TOTAL,'source total bound')
        parts,segments=transport.layout(rows,self.manifest['part_bytes'])
        require(segments==self.manifest['segments']and len(parts)==len(self.manifest['parts']),'source ranges differ')
        require(all(a['file']==b['file']and a['bytes']==b['bytes']for a,b in zip(parts,self.manifest['parts'])),'source part inventory differs')
        self.parts={p['file']:p for p in self.manifest['parts']};self.files={r['file']:r for r in rows};self.current=None
    def read(self,file,offset,count):
        require(file in self.files and type(offset)is int and offset>=0 and type(count)is int and
                0<count<=MAX_STAGE_BYTES and offset+count<=self.files[file]['bytes'],'remote stage range outside source')
        data=bytearray();end=offset+count
        for segment in self.manifest['segments']:
            if segment['file']!=file:continue
            begin=max(offset,segment['file_offset']);finish=min(end,segment['file_offset']+segment['bytes'])
            if begin>=finish:continue
            name=segment['part'];part=self.parts[name]
            if self.current!=name:
                pending=self.cache/'part.partial';self.backend.get(self.prefix+'/'+name,self.revision,pending)
                require(pending.stat().st_size==part['bytes']and transport.digest(pending)==part['sha256'],'corrupt/missing remote part')
                pending.replace(self.cache/'part.bin');self.current=name
            with(self.cache/'part.bin').open('rb')as stream:
                stream.seek(segment['part_offset']+begin-segment['file_offset']);left=finish-begin
                while left:
                    chunk=stream.read(min(left,2**20));require(bool(chunk),'remote slice truncated');data.extend(chunk);left-=len(chunk)
        require(len(data)==count,'incomplete remote stage range');return bytes(data)


def package(root,source_manifest,side,output,role='synthetic-control',precision='f32'):
    require(side in('original','native'),'explicit package side required')
    require(role=='synthetic-control','real4B output packaging is not admitted by document-v2')
    require(precision in('f32','f16'),'quantized packaging needs separate qualification')
    output=Path(output);require(not output.exists(),'fresh package manifest required')
    root=Path(root);rows,observer=transport.source_files(root,role)
    source=transport.read_json(source_manifest)
    require(source['schema']=='lighton4b-original-sharded-transport/v1'and source['role']==role and
            source['files']==rows,'source transport/source file hashes differ')
    legacy=parse_gguf(root/'official-delta.gguf',storage_types={0:(1,4)})
    records=[];reader=LocalReader(root)
    for name,kind in ordered_stages(observer['positions']):
        if name.startswith('generated.'):
            raw=(f'generated.{name.split(".")[-1]}.raw_logits'if name.startswith('generated.scores.')else name)
            row=observer['records'][raw];file='generated/'+row['file'];offset=0;shape=row['shape'];size=row['bytes']
        else:
            require(name in legacy['tensors'],'missing exact legacy stage: '+name)
            row=legacy['tensors'][name];file='official-delta.gguf';offset=row['data_offset'];shape=list(reversed(row['stored_shape']));size=row['bytes']
        require(size==math.prod(shape)*4 and 0<size<=MAX_STAGE_BYTES,'stage shape/byte bound')
        data=reader.read(file,offset,size)
        records.append(dict(name=name,kind=kind,file=file,offset=offset,bytes=size,shape=shape,
                            sha256=hashlib.sha256(data).hexdigest()))
    result={'schema':'lighton4b-document-package/v2','role':role,'side':side,'precision':precision,
        'case_protocol_sha256':v1.sha(v1.PROTOCOL),'source_transport_sha256':transport.digest(source_manifest),
        'source_file_sha256':{r['file']:r['sha256']for r in rows},'positions':observer['positions'],
        'generated_ids':observer['generated_ids'],'teacher_forced_ids':observer.get('teacher_forced_ids',observer['generated_ids']),
        'decoded':observer.get('decoded','synthetic fixture'),'eos_completed':observer['eos_completed'],'cap_hit':observer['cap_hit'],
        'stages':records,'original_execution_admitted':False,'native_execution_admitted':False,
        'large_retention_qualified':False,'actual_original_output_ready':False,'quality_qualified':False}
    validate(result);payload=transport.encoded(result)
    require(len(payload)<=16*2**20,'v2 manifest byte bound')
    output.write_bytes(payload);return result


def validate(package):
    require(package['schema']=='lighton4b-document-package/v2'and package['role']=='synthetic-control',
            'real output role/admission is closed')
    require(package['side']in('original','native')and package['precision']in('f32','f16'),'package side/precision differs')
    require(package['case_protocol_sha256']==v1.sha(v1.PROTOCOL),'frozen case contract differs')
    for flag in('original_execution_admitted','native_execution_admitted','large_retention_qualified','actual_original_output_ready','quality_qualified'):
        require(package[flag]is False,'synthetic package cannot assert real readiness/admission')
    expected=ordered_stages(package['positions']);rows=package['stages']
    require([(r['name'],r['kind'])for r in rows]==expected,'exact stage coverage/order differs')
    ids=package['generated_ids'];require(isinstance(ids,list)and 1<=len(ids)<=256 and
        all(type(x)is int and 0<=x<248320 for x in ids)and ids[-1]==248046 and 248046 not in ids[:-1]and
        package['eos_completed']is True and package['cap_hit']is False,'complete EOS output required')
    require(isinstance(package['decoded'],str)and len(package['decoded'].encode())<=2**20,'bounded decoded text required')
    forced=package['teacher_forced_ids']
    require(isinstance(forced,list)and len(forced)==package['positions']and all(type(x)is int and 0<=x<248320 for x in forced),'all original-position scores required')
    if package['side']=='original':require(forced==ids,'original generated/score positions differ')
    require(isinstance(package['source_file_sha256'],dict)and len(package['source_file_sha256'])<=67*256+4,'source file table bound')
    for file,digest in package['source_file_sha256'].items():
        transport.safe_name(file);require(isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest)is not None,'source file SHA differs')
    for row in rows:
        transport.safe_name(row['file']);shape=row['shape']
        require(isinstance(shape,list)and 1<=len(shape)<=4 and all(type(x)is int and x>0 for x in shape),'stage shape differs')
        require(type(row['offset'])is int and row['offset']>=0 and row['bytes']==math.prod(shape)*4 and 0<row['bytes']<=MAX_STAGE_BYTES,'stage range bound')
        require(row['file']in package['source_file_sha256'],'stage file outside frozen source')
        require(isinstance(row['sha256'],str)and re.fullmatch('[0-9a-f]{64}',row['sha256'])is not None,'stage SHA differs')
    return package


def readiness(package):
    validate(package)
    scores=[r for r in package['stages']if r['name'].startswith('generated.scores.')]
    return {'schema':'lighton4b-document-readiness/v2','synthetic_packaging_valid':True,
        'all_generated_positions_present':len(scores)==package['positions'],
        'actual_full_vocabulary_geometry':all(r['shape']==[1,248320]for r in scores),
        'actual_original_output_ready':False,'original_execution_admitted':False,'native_execution_admitted':False,
        'large_retention_qualified':False,'blocking_reasons':['No admitted actual4B producer/output exists.',
            'Original capture hard block and40GiB/48GiB guards remain unchanged.',
            'Tiny provider proof does not qualify large capture retention.']}


def tiny_archive(path,name,shape,data):
    raw=name.encode();ne=list(reversed(shape));header=struct.pack('<4sIQQQ',b'GGUF',3,1,0,len(raw))+raw
    header+=struct.pack('<I',len(ne))+struct.pack('<'+'Q'*len(ne),*ne)+struct.pack('<IQ',0,0)
    with Path(path).open('wb')as stream:stream.write(header);stream.write(b'\0'*((-len(header))%32));stream.write(data);stream.write(b'\0'*((-len(data))%32))


def consume(original,native,original_reader,native_reader,binary,output):
    validate(original);validate(native)
    require(original['side']=='original'and native['side']=='native','paired sides required')
    require(native['teacher_forced_ids']==original['generated_ids'],'native scores must cover original-generated prefixes')
    require([(r['name'],r['shape'])for r in original['stages']]==[(r['name'],r['shape'])for r in native['stages']],'paired stage geometry differs')
    for package,reader in((original,original_reader),(native,native_reader)):
        if isinstance(reader,RemoteReader):
            require(package['source_transport_sha256']==transport.digest(reader.cache/'transport.json'),'package/remote source binding differs')
            require(package['source_file_sha256']=={r['file']:r['sha256']for r in reader.manifest['files']},'remote source file hashes differ')
    output=Path(output);require(not output.exists(),'fresh consumer report directory required');output.mkdir(parents=True)
    report={'schema':'lighton4b-document-consumption/v2','status':'failed','compared_stages':0,'earliest_differing_stage':None,
        'original_model_forward_executed':False,'native_model_forward_executed':False,'actual_original_output_ready':False,
        'large_retention_qualified':False,'quality_qualified':False}
    try:
        with tempfile.TemporaryDirectory(prefix='lighton4b-bounded-stage-')as temporary,(output/'crispembed-diff.stdout').open('w')as log:
            temporary=Path(temporary)
            for a,b in zip(original['stages'],native['stages']):
                for row,reader,label in((a,original_reader,'original'),(b,native_reader,'native')):
                    data=reader.read(row['file'],row['offset'],row['bytes'])
                    require(hashlib.sha256(data).hexdigest()==row['sha256'],'stage slice hash differs: '+row['name'])
                    tiny_archive(temporary/(label+'.gguf'),row['name'],row['shape'],data)
                axes=list(reversed(a['shape']))+[1]*(4-len(a['shape']))
                order=temporary/'order.tsv';order.write_text(a['name']+' '+a['kind']+' '+' '.join(str(n)for n in axes)+'\n')
                result=subprocess.run([str(binary),temporary/'original.gguf',temporary/'native.gguf',order],capture_output=True,text=True)
                log.write(result.stdout);report['compared_stages']+=1
                if result.returncode and report['earliest_differing_stage']is None:report['earliest_differing_stage']=a['name']
        exact=original['generated_ids']==native['generated_ids']and original['decoded']==native['decoded']
        report.update(status='passed'if report['earliest_differing_stage']is None and exact else'failed',
            decoded_and_ids_exact=exact,original_decoded=original['decoded'],native_decoded=native['decoded'])
        return report
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:(output/'consumption.json').write_bytes(transport.encoded(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('package-synthetic')
    for key in('source','transport','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--side',choices=('original','native'),required=True);p.add_argument('--precision',choices=('f32','f16'),default='f32')
    p=sub.add_parser('readiness');p.add_argument('--manifest',type=Path,required=True)
    p=sub.add_parser('consume-local')
    for key in('original','native','original-root','native-root','binary','output'):p.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.command=='package-synthetic':package(a.source,a.transport,a.side,a.output,precision=a.precision);return 0
    if a.command=='readiness':print(json.dumps(readiness(transport.read_json(a.manifest)),indent=2));return 0
    result=consume(transport.read_json(a.original),transport.read_json(a.native),LocalReader(a.original_root),LocalReader(a.native_root),a.binary,a.output)
    return 0 if result['status']=='passed'else 1


if __name__=='__main__':raise SystemExit(main())
