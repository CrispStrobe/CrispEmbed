#!/usr/bin/env python3
"""Versioned immutable diagnostic transport. Never executes a model.

Original-capture publication remains disabled until independent source/provider
review. Synthetic fixtures can exercise the complete remote/readback mechanism.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
MAX_PART=512*2**20
MAX_TOTAL=48*2**30
ORIGINAL_TRANSPORT_ADMITTED=False
PROTOCOL_SHA='afee0e1fbebc73b9ded744343ee9e2ae467f3b214ab3682e9fb0e8fd733ad674'


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while chunk:=stream.read(2**20):h.update(chunk)
    return h.hexdigest()


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def unique_object(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON key');result[key]=value
    return result


def read_json(path):
    require(Path(path).stat().st_size<=16*2**20,'manifest byte bound')
    return json.loads(Path(path).read_text(),object_pairs_hook=unique_object)


def safe_name(name):
    require(isinstance(name,str)and 0<len(name)<=240,'invalid file name')
    p=PurePosixPath(name)
    require(not p.is_absolute()and '..'not in p.parts and str(p)==name and '\\'not in name,'unsafe file path')
    return name


def fields(positions):
    require(type(positions)is int and 1<=positions<=256,'position bound')
    names=set()
    for step in range(positions):
        prefix=f'generated.{step}.'
        names.update(prefix+x for x in('decoder.norm','decoder.position_ids','raw_logits'))
        for i in range(32):
            names.update(prefix+f'cache.{i}.'+x for x in(('keys','values')if i%4==3 else('conv_states','recurrent_states')))
    return names


def source_files(root,role):
    root=Path(root)
    require(role=='synthetic-control'or(role=='original-capture'and ORIGINAL_TRANSPORT_ADMITTED),
            'original capture transport is not admitted')
    observer=read_json(root/'generated/observer.json')
    if role=='synthetic-control':
        require(observer.get('synthetic_control')is True and observer.get('original_forward_observed')is False,
                'synthetic transport cannot relabel an original model capture')
    require(observer['status']=='passed'and observer['eos_completed']is True and observer['cap_hit']is False,
            'incomplete/capped observer capture')
    ids=observer['generated_ids'];require(isinstance(ids,list)and 1<=len(ids)<=256 and ids[-1]==248046 and
        all(type(x)is int and 0<=x<248320 for x in ids)and 248046 not in ids[:-1],'invalid original IDs/EOS')
    require(observer['positions']==len(ids)and set(observer['records'])==fields(len(ids)),'complete67 fields per position required')
    require(observer['observer_off_ids_and_all_logits_exact']is True and observer['raw_logits_equal_generate_scores']is True,
            'observer-neutral original proof missing')
    rows=[]
    for name,record in sorted(observer['records'].items()):
        filename='generated/'+safe_name(record['file'])
        require(record['file']==name+'.f32','field/file identity differs')
        path=root/filename
        require(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(root.resolve()),'regular contained source field required')
        shape=record['shape'];require(isinstance(shape,list)and 1<=len(shape)<=4 and all(type(x)is int and x>0 for x in shape),'invalid field shape')
        count=1
        for axis in shape:count*=axis
        require(0<count<=64*2**20 and path.stat().st_size==record['bytes']==count*4,'source field shape/bytes differ')
        require(digest(path)==record['sha256'],'source field hash differs')
        rows.append(dict(file=filename,bytes=record['bytes'],sha256=record['sha256'],shape=shape,dtype='F32-LE'))
    require(sum(r['bytes']for r in rows)<=24*2**30,'generated shard budget')
    for name in('generated/observer.json','adapter.json','original-reference.npz','official-delta.gguf'):
        path=root/name
        if path.is_file():
            require(not path.is_symlink()and path.stat().st_size<=9*2**30,'regular bounded metadata/legacy file required')
            rows.append(dict(file=name,bytes=path.stat().st_size,sha256=digest(path)))
    require(sum(r['bytes']for r in rows)<=MAX_TOTAL,'total transport source budget')
    return sorted(rows,key=lambda r:r['file']),observer


def layout(rows,part_bytes):
    require(type(part_bytes)is int and 4<=part_bytes<=MAX_PART,'part size outside bound')
    segments=[];parts=[];index=0;used=0
    for row in rows:
        safe_name(row['file']);left=row['bytes'];offset=0
        require(type(left)is int and left>0,'nonempty file required')
        while left:
            take=min(left,part_bytes-used)
            segments.append(dict(file=row['file'],file_offset=offset,part=f'part-{index:05d}.bin',part_offset=used,bytes=take))
            used+=take;left-=take;offset+=take
            if used==part_bytes:
                parts.append(dict(file=f'part-{index:05d}.bin',bytes=used));index+=1;used=0
    if used:parts.append(dict(file=f'part-{index:05d}.bin',bytes=used))
    return parts,segments


def publish(root,backend,prefix,receipt_path,role='synthetic-control',part_bytes=MAX_PART):
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{7,79}',prefix)is not None,'unique bounded session prefix required')
    receipt_path=Path(receipt_path);require(not receipt_path.exists(),'fresh upload receipt required')
    receipt={'schema':'lighton4b-transport-publication/v1','status':'incomplete','prefix':prefix,'parts':[],
        'role':role,'original_capture_admitted':False,'original_forward_executed':False,'quality_qualified':False}
    try:
        rows,observer=source_files(root,role);parts,segments=layout(rows,part_bytes)
        backend.prepare(prefix)
        for part in parts:
            with tempfile.TemporaryDirectory(prefix='lighton4b-part-')as folder:
                local=Path(folder)/'part.bin'
                with local.open('wb')as target:
                    for segment in segments:
                        if segment['part']!=part['file']:continue
                        with (Path(root)/segment['file']).open('rb')as source:
                            source.seek(segment['file_offset']);left=segment['bytes']
                            while left:
                                chunk=source.read(min(left,2**20));require(bool(chunk),'source truncated while publishing')
                                target.write(chunk);left-=len(chunk)
                require(local.stat().st_size==part['bytes'],'part length differs')
                part['sha256']=digest(local)
                revision=backend.put(prefix+'/'+part['file'],local)
                receipt['parts'].append(dict(part,revision=revision))
                readback=Path(folder)/'empty-cache-readback.bin'
                backend.get(prefix+'/'+part['file'],revision,readback)
                require(readback.stat().st_size==part['bytes']and digest(readback)==part['sha256'],'independent part readback differs')
                receipt_path.write_bytes(encoded(receipt))
        # Detect source mutation during transport; no final manifest is committed
        # for partial uploads, wrong parts or changed input files.
        require(all(digest(Path(root)/r['file'])==r['sha256']for r in rows),'source changed during publication')
        manifest={'schema':'lighton4b-original-sharded-transport/v1','role':role,'case_protocol_sha256':PROTOCOL_SHA,
            'positions':observer['positions'],'generated_ids':observer['generated_ids'],'files':rows,'parts':parts,
            'segments':segments,'part_bytes':part_bytes,'capture_transport_admitted':False,'quality_qualified':False}
        with tempfile.TemporaryDirectory(prefix='lighton4b-manifest-')as folder:
            local=Path(folder)/'manifest.json';local.write_bytes(encoded(manifest))
            revision=backend.put(prefix+'/manifest.json',local)
            restored=Path(folder)/'readback.json';backend.get(prefix+'/manifest.json',revision,restored)
            require(restored.read_bytes()==local.read_bytes(),'manifest readback differs')
            receipt.update(status='passed',revision=revision,manifest_sha256=digest(local))
        return receipt
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:receipt_path.write_bytes(encoded(receipt))


def restore(backend,prefix,revision,manifest_sha256,output):
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{7,79}',prefix)is not None,'invalid restore session prefix')
    output=Path(output);require(not output.exists(),'empty independent restore directory required');output.mkdir(parents=True)
    receipt={'schema':'lighton4b-transport-readback/v1','status':'incomplete','original_capture_admitted':False,'quality_qualified':False}
    try:
        manifest_path=output/'manifest.json';backend.get(prefix+'/manifest.json',revision,manifest_path)
        require(digest(manifest_path)==manifest_sha256,'immutable manifest hash differs')
        manifest=read_json(manifest_path)
        require(manifest['schema']=='lighton4b-original-sharded-transport/v1'and manifest['case_protocol_sha256']==PROTOCOL_SHA,
                'transport protocol differs')
        rows=manifest['files'];require(len(rows)<=67*256+4 and len({r['file']for r in rows})==len(rows),'file inventory differs')
        require(sum(r['bytes']for r in rows)<=MAX_TOTAL,'restore total byte bound')
        planned,segments=layout(rows,manifest['part_bytes'])
        require(segments==manifest['segments']and len(planned)==len(manifest['parts']),'noncanonical/overlapping/missing segments')
        require(all(all(a[k]==b[k]for k in('file','bytes'))for a,b in zip(planned,manifest['parts'])),'part inventory differs')
        require(shutil.disk_usage(output).free>=sum(r['bytes']for r in rows)+manifest['part_bytes']+2**30,'restore disk headroom')
        for row in rows:
            path=output/'files'/safe_name(row['file']);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('wb')as stream:stream.truncate(row['bytes'])
        for part in manifest['parts']:
            with tempfile.TemporaryDirectory(prefix='lighton4b-restore-part-')as folder:
                local=Path(folder)/'part.bin';backend.get(prefix+'/'+part['file'],revision,local)
                require(local.stat().st_size==part['bytes']and digest(local)==part['sha256'],'missing/corrupt/truncated remote part')
                with local.open('rb')as source:
                    for segment in segments:
                        if segment['part']!=part['file']:continue
                        source.seek(segment['part_offset']);left=segment['bytes']
                        with (output/'files'/segment['file']).open('r+b')as target:
                            target.seek(segment['file_offset'])
                            while left:
                                chunk=source.read(min(left,2**20));require(bool(chunk),'remote segment truncated')
                                target.write(chunk);left-=len(chunk)
        require(all(digest(output/'files'/r['file'])==r['sha256']for r in rows),'independent whole-file hash differs')
        # Re-run source-level completeness checks on independently reconstructed
        # files, including67 fields/position and original stop/neutrality markers.
        restored_rows,_=source_files(output/'files',manifest['role'])
        require(restored_rows==rows,'restored semantic source inventory differs')
        receipt.update(status='passed',manifest_sha256=manifest_sha256,files=len(rows),parts=len(manifest['parts']),
            total_bytes=sum(r['bytes']for r in rows),independent_whole_file_hashes_verified=True)
        return receipt
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:(output/'readback.json').write_bytes(encoded(receipt))


class HuggingFaceBackend:
    """Real optional provider backend; caller supplies an authorized existing repo.

    Every write has a checked parent commit. Readback uses immutable commits and
    a fresh temporary cache. Tokens are never part of source or manifests.
    """
    def __init__(self,repo_id,repo_type='dataset'):
        from huggingface_hub import HfApi
        self.repo=repo_id;self.kind=repo_type;self.api=HfApi(token=os.environ.get('HF_TOKEN'))
        self.head=self.api.repo_info(repo_id=repo_id,repo_type=repo_type).sha
    def prepare(self,prefix):
        require(not any(n.startswith(prefix+'/')for n in self.api.list_repo_files(self.repo,repo_type=self.kind,revision=self.head)),
                'remote session prefix already exists')
    def put(self,name,path):
        commit=self.api.upload_file(path_or_fileobj=str(path),path_in_repo=name,repo_id=self.repo,repo_type=self.kind,
            parent_commit=self.head,commit_message='Retain experimental diagnostic transport object')
        self.head=commit.oid;return self.head
    def get(self,name,revision,destination):
        require(re.fullmatch(r'[0-9a-f]{40}',revision)is not None,'immutable Hub revision required')
        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory(prefix='lighton4b-empty-hub-cache-')as cache:
            downloaded=hf_hub_download(repo_id=self.repo,repo_type=self.kind,revision=revision,filename=name,
                cache_dir=cache,token=os.environ.get('HF_TOKEN'),force_download=True)
            shutil.copyfile(downloaded,destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo-id',required=True)
    parser.add_argument('--repo-type',choices=('dataset','model'),default='dataset');sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('publish-synthetic');p.add_argument('--source',type=Path,required=True);p.add_argument('--prefix',required=True);p.add_argument('--receipt',type=Path,required=True)
    p=sub.add_parser('readback');p.add_argument('--prefix',required=True);p.add_argument('--revision',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();backend=HuggingFaceBackend(args.repo_id,args.repo_type)
    if args.command=='publish-synthetic':publish(args.source,backend,args.prefix,args.receipt)
    else:restore(backend,args.prefix,args.revision,args.manifest_sha256,args.output)
