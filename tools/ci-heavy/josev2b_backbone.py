#!/usr/bin/env python3
"""Hosted original-first F32 text backbone audit. No full Josev/head inference."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_backbone_probe as c
from josev2b_backbone_diff import write_pair,collect
from josev2b_head_schema import reference,sha256
from convert_josev2b_backbone_f32 import convert,schema
spec=importlib.util.spec_from_file_location('frozen_head_artifact',ROOT/'tools/ci-heavy/josev2b_head_artifact.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))


def resources(path):
    free=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(path).free
    if free<10*2**30 or disk<25*2**30:raise ValueError('Hosted preflight requires10GiBavailableRAM/25GiBfree disk')
    return dict(available_ram_bytes=free,free_disk_bytes=disk,required_ram_bytes=10*2**30,required_disk_bytes=25*2**30)


def archive(kind,scratch):
    pin=c.ARTIFACTS[kind];row=api('actions/artifacts/'+str(pin['id']))
    if (row['workflow_run']['id']!=c.RUN or row['workflow_run']['head_sha']!=c.PRODUCER or row['expired'] or
            row['name']!=pin['name'] or row['size_in_bytes']!=pin['bytes'] or row['digest']!='sha256:'+pin['sha256']):
        raise ValueError('Reviewed original artifact service identity differs')
    target=scratch/(kind+'.zip')
    with target.open('xb') as stream:
        subprocess.run(['gh','api',f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{pin["id"]}/zip'],stdout=stream,check=True,timeout=600)
    if target.stat().st_size!=pin['bytes'] or sha256(target)!=pin['sha256']:raise ValueError('Original ZIP digest/extent differs')
    with zipfile.ZipFile(target) as z:
        entries=z.infolist();names=[e.filename for e in entries]
        if (len(entries)>2048 or len(set(names))!=len(names) or sum(e.file_size for e in entries)>2**30 or
                any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names) or
                any((e.external_attr>>16)&0o170000==0o120000 for e in entries)):
            raise ValueError('Original ZIP safety/extent scope differs')
        z.extractall(scratch/kind)
    return {k:row[k] for k in ('id','name','size_in_bytes','digest','expires_at')}


def retained(scratch):
    root=scratch/'raw';receipt=json.loads((scratch/'review/artifact.json').read_text())
    if ((root/'review/artifact.json').read_bytes()!=(scratch/'review/artifact.json').read_bytes() or
            receipt['completed']is not True or receipt['phase']!='completed' or receipt['source_commit']!=c.PRODUCER or
            receipt['original_source']!=c.SOURCE or receipt['reference_sha256']!=c.RECEIPT or receipt['arrays']!=251 or
            receipt['gguf_sha256']!='5b0a4bfe769526569102e3cde115887593875124c94bb723fb9087bc40c29507' or
            receipt['gguf_bytes']!=121900352):raise ValueError('Reviewed original-preservation receipt differs')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(receipt['files'])|{'review/artifact.json'}:raise ValueError('Complete original member coverage differs')
    for name,pin in receipt['files'].items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe producer manifest path')
        path=root/name
        if path.is_symlink() or path.stat().st_size!=pin['bytes'] or sha256(path)!=pin['sha256']:
            raise ValueError('Retained original member digest differs: '+name)
    source=root/'original'
    report,cfg,_=reference(source/'reference.json',c.RECEIPT,c.SOURCE)
    if ((source/'reference.json').read_bytes()!=(ROOT/'tests/results/josev2b/original-reference.json').read_bytes() or
            sha256(source/'input-freeze.json')!=c.FREEZE):raise ValueError('Frozen original/public receipt bytes differ')
    freeze=json.loads((source/'input-freeze.json').read_text());packet=c.packet(freeze)
    c.original_events(report)
    # Authenticate every declaration/header before optional NumPy/GGUF imports.
    if len(report['arrays'])!=251 or len({r['file'] for r in report['arrays']})!=251:raise ValueError('Complete251array inventory required')
    for row in report['arrays']:
        original.safe_member(row['file']);path=source/row['file']
        original.array_header(path,row)
        if sha256(path)!=row['sha256']:raise ValueError('Original NPY declaration digest differs')
    return source,report,cfg,freeze,packet


def prepare(out,scratch):
    measured=c.source_guard()
    if measured['head']!=os.environ.get('GITHUB_SHA'):raise ValueError('Actual service/source differs')
    state=api('actions/runs/'+str(c.RUN))
    if (state['head_sha']!=c.PRODUCER or state['run_attempt']!=1 or state['status']!='completed' or state['conclusion']!='success'):
        raise ValueError('Reviewed original preservation run differs')
    if scratch.exists():raise ValueError('Fresh scratch required')
    scratch.mkdir();preflight=resources(scratch)
    proof={kind:archive(kind,scratch) for kind in ('review','raw')}
    _,report,_,_,packet=retained(scratch)
    (out/'inputs.packet').write_bytes(packet)
    record=dict(schema='crispembed-josev-backbone-prepared-v1',source=measured,service=proof,resources=preflight,
        original_receipt_sha256=c.RECEIPT,input_freeze_sha256=c.FREEZE,input_packet_sha256=c.sha(packet),
        stages=104,arrays_authenticated=251,packages_imported=False,new_original_outputs=False,native_outputs=False)
    write(out/'review/prepared.json',record)


def download(pin,name,cfg,target):
    h=hashlib.sha256();count=0
    url=f'https://huggingface.co/{cfg["repository"]}/resolve/{cfg["revision"]}/{name}'
    with urllib.request.urlopen(url,timeout=120) as response,target.open('xb') as output:
        while chunk:=response.read(4*1024*1024):
            count+=len(chunk)
            if count>pin['bytes']:raise ValueError('Frozen source download exceeds exact extent')
            h.update(chunk);output.write(chunk)
    if count!=pin['bytes'] or h.hexdigest()!=pin['sha256']:raise ValueError('Frozen source download digest differs')


def command(args,log,env,timeout):
    begin=time.monotonic()
    with log.open('w') as stream:
        r=subprocess.run([str(x) for x in args],stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout,check=False)
    return dict(exit=r.returncode,elapsed_seconds=time.monotonic()-begin,log_bytes=log.stat().st_size,log_sha256=sha256(log))


def metrics(a,b,np):
    if a.shape!=b.shape or a.dtype!=np.dtype('<f4') or b.dtype!=np.dtype('<f4') or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Exact finite F32 metric shape/dtype required')
    x=a.astype(np.float64);y=b.astype(np.float64);delta=y-x
    aa=(x*x).sum(axis=-1);bb=(y*y).sum(axis=-1);dot=(x*y).sum(axis=-1)
    cosine=np.zeros_like(dot);valid=(aa>1e-18)&(bb>1e-18);cosine[(aa<=1e-18)&(bb<=1e-18)]=1
    cosine[valid]=dot[valid]/(np.sqrt(aa[valid])*np.sqrt(bb[valid]))
    result=dict(shape=list(a.shape),elements=a.size,max_absolute=float(np.abs(delta).max()),mean_absolute=float(np.abs(delta).mean()),
        mean_error=float(delta.mean()),rms=float(np.sqrt((delta*delta).mean())),min_row_cosine=float(cosine.min()),
        original_l2=float(np.sqrt((x*x).sum())),native_l2=float(np.sqrt((y*y).sum())),
        bit_exact=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))))
    for label,value in (('original',x),('native',y)):
        result.update({label+'_min':float(value.min()),label+'_max':float(value.max()),label+'_abs_min':float(np.abs(value).min()),label+'_abs_max':float(np.abs(value).max())})
    result['passed']=result['max_absolute']<=.002 and result['min_row_cosine']>=.99999
    return result


def compare(binary,directory,pairs,env,np):
    directory.mkdir(parents=True);independent={};adapters=[]
    for i,(frozen,native,shape) in enumerate(pairs):
        a=np.load(frozen,mmap_mode='r',allow_pickle=False) if frozen.suffix=='.npy' else np.memmap(frozen,mode='r',dtype='<f4',shape=shape)
        b=np.memmap(native,mode='r',dtype='<f4',shape=shape)
        if list(a.shape)!=shape:raise ValueError('Original shape must stay unchanged')
        name=f'stage.{i:02d}';independent[name]=metrics(a,b,np)
        original_raw=directory/(name+'.original.f32')
        # Exact storage export, no shape/precision transformation.
        with original_raw.open('xb') as stream:a.tofile(stream)
        adapters.append(dict(name=name,original=dict(path=original_raw,shape=shape,sha256=sha256(original_raw)),
            native=dict(path=native,shape=shape,sha256=sha256(native))))
    lossless=write_pair(directory/'original.gguf',directory/'native.gguf',adapters)
    result=command([binary,directory/'original.gguf',directory/'native.gguf',directory/'diff.json'],directory/'diff.log',env,120)
    actual=collect(result['exit'],directory/'diff.json',independent)
    if not actual['all_metrics_agree']:raise ValueError('ActualRef/independentNumPy reporting differs')
    return dict(actual_ref=actual,command=result,lossless=lossless,metrics=independent,
        passed=all(v['passed'] for v in independent.values()))


def run(out,scratch):
    measured=c.source_guard();prepared=json.loads((out/'review/prepared.json').read_text())
    if prepared['source']!=measured or measured['head']!=os.environ.get('GITHUB_SHA'):raise ValueError('Prepared source changed before model access')
    source,report,cfg,freeze,packet=retained(scratch)
    if (out/'inputs.packet').read_bytes()!=packet:raise ValueError('Frozen input packet changed')
    if subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()!=c.GGML:raise ValueError('Actual ggml checkout differs')
    resources(scratch)
    for name,version in (('numpy','2.2.6'),('gguf','0.17.1')):
        if importlib.metadata.version(name)!=version:raise ValueError('Pinned adapter packages required')
    if any(m in sys.modules for m in ('torch','transformers','tokenizers')):raise ValueError('No new original model/tokenizer execution allowed')
    import numpy as np
    # Reuse unchanged verifier for every251NPY finite/mask policy and orderedevent.
    original.verify_original(source)
    raw=out/'raw';raw.mkdir();notices=raw/'notices';notices.mkdir()
    result=dict(schema='crispembed-josev-backbone-audit-v1',source=measured,completed=False,backbone_numeric_pass=False,
        full_model=False,composed_typed_outputs=False,model_quality=False,release_qualified=False,cases=[],commands={})
    def save():write(out/'review/backbone.json',result)
    save();env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update(CRISPEMBED_QWEN35_OCR='1',CRISPEMBED_JOSEV_BACKBONE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    build=scratch/'build'
    for label,args in [('configure',['cmake','-S',ROOT/'tests/josev-backbone','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),
                       ('build',['cmake','--build',build,'--parallel','2']),('metadata',[build/'test-josev-backbone'])]:
        result['commands'][label]=command(args,out/'review'/(label+'.log'),env,1200);save()
        if result['commands'][label]['exit']:raise ValueError('Hosted standalone build/metadata rejected: '+label)
    shutil.copyfile(build/'compile_commands.json',out/'review/compile_commands.json')
    for name in ('josev-backbone-probe','josev-backbone-diff'):shutil.copyfile(build/name,raw/name)
    for gate in ('CRISPEMBED_JOSEV_BACKBONE','CRISPEMBED_QWEN35_OCR'):
        for value in (None,'','0'):
            disabled=dict(env);disabled.pop(gate,None)
            if value is not None:disabled[gate]=value
            control=subprocess.run([str(build/'josev-backbone-probe'),'absent','absent','absent'],env=disabled,capture_output=True,timeout=10)
            if control.returncode!=2 or control.stdout or control.stderr:raise ValueError('Disabled gate accessed model/input or emitted output')
    result['commands']['packet']=command([build/'josev-backbone-probe','--contracts',out/'inputs.packet'],out/'review/packet.log',env,30)
    if result['commands']['packet']['exit']:raise ValueError('Frozen actual packet parser rejected')
    # Exercise the compiled parser without loading even a metadata-only model.
    bad_packets={'truncated':packet[:-1],'extra':packet+b'x',
        'case-count':packet[:176]+struct.pack('<I',3)+packet[180:]}
    first_id=180+8+len(c.CASES[0][0].encode())
    bad_packets['token-id']=packet[:first_id]+struct.pack('<I',248320)+packet[first_id+4:]
    mask_at=first_id+4*c.CASES[0][1]
    bad_packets['mask']=packet[:mask_at]+b'\0'+packet[mask_at+1:]
    for name,data in bad_packets.items():
        path=scratch/(name+'.packet');path.write_bytes(data)
        rejected=subprocess.run([str(build/'josev-backbone-probe'),'--contracts',str(path)],env=env,capture_output=True,timeout=10)
        if rejected.returncode!=2 or b'Backbone probe structural rejection' not in rejected.stderr:
            raise ValueError('Compiled packet negative control failed: '+name)
    save()
    # Compiled Ref positive and non-equivalent row controls before the native model.
    control=raw/'comparator-control';control.mkdir()
    a=np.asarray([[1,0],[0,1],[1,0]],dtype='<f4');b=a.copy();b[-1]=[0,1]
    np.save(control/'original.npy',a,allow_pickle=False);a.tofile(control/'same.f32');b.tofile(control/'changed.f32')
    for name,code in (('same',0),('changed',1)):
        r=compare(build/'josev-backbone-diff',control/name,[(control/'original.npy',control/(name+'.f32'),[3,2])],env,np)
        if r['actual_ref']['exit_code']!=code:raise ValueError('ActualRef numeric return-code control failed')
    for name in ('LICENSE','README.md','joint_schema_model.py','joint_head_config.json','config.json',
                 'unsloth_decision_config.json','tokenizer_config.json','processor_config.json'):
        download(cfg['files'][name],name,cfg,notices/name)
    model=scratch/'model.safetensors';index=scratch/'model.safetensors.index.json'
    filename='model.safetensors-00001-of-00001.safetensors'
    download(report['snapshot'][filename],filename,cfg,model)
    download(report['snapshot']['model.safetensors.index.json'],'model.safetensors.index.json',cfg,index)
    gguf=raw/'backbone-f32.gguf'
    conversion=convert(model,index,source/'reference.json',c.RECEIPT,c.SOURCE,gguf)
    if conversion['tensor_count']!=320 or conversion['parameter_count']!=1881825088:raise ValueError('Exact320tensor conversion required')
    shutil.copyfile(raw/'backbone-f32.conversion.json',out/'review/backbone-conversion.json')
    result['conversion']=conversion;save()
    # Authenticated immutable HF source remains available remotely; free only task-owned scratch copies.
    model.unlink();index.unlink()
    if gguf.stat().st_size!=conversion['gguf_bytes'] or sha256(gguf)!=conversion['gguf_sha256']:
        raise ValueError('Converted backbone changed before native model access')
    result['commands']['probe']=command(['/usr/bin/time','-v',build/'josev-backbone-probe',gguf,out/'inputs.packet',raw/'native'],out/'review/native.log',env,10800)
    save();native=json.loads((raw/'native/native.json').read_text())
    if native.get('schema')!='crispembed-josev-backbone-native-v1' or native.get('backend')!='CPU' or len(native.get('cases',[]))!=4:
        raise ValueError('Native complete four-case panel required')
    events=c.original_events(report)
    for i,(case,token_count) in enumerate(c.CASES):
        row=dict(id=case,tokens=token_count,passed=False)
        try:
            rows=c.validate_native(i,native['cases'][i],raw/'native')
            pairs=[(source/a['file'],raw/'native'/b['file'],b['shape']) for a,b in zip(events[case],rows)]
            row['comparison']=compare(build/'josev-backbone-diff',raw/'comparisons'/f'{i:02d}',pairs,env,np)
            row['stages']=c.comparison_records(events[case],rows,row['comparison']['metrics'])
            row['first_differing_stage']=next((s for s in row['stages'] if not s['bit_exact']),None)
            row['first_failing_stage']=next((s for s in row['stages'] if not s['passed']),None)
            row['passed']=row['comparison']['passed']
        except Exception as error:
            row['error']=type(error).__name__+': '+str(error)
        result['cases'].append(row);save()
    result.update(completed=True,backbone_numeric_pass=result['commands']['probe']['exit']==0 and all(r['passed'] for r in result['cases']),
        compared_stages=sum(len(r.get('stages',[])) for r in result['cases']),
        limits='Four fixed batch1 all-attended text cases, original GPUF32 versus native CPUF32; no head/composed responses, tokenizer, representative quality, broad context, runtime integration/default/release qualification.')
    save()
    if not result['backbone_numeric_pass'] or result['compared_stages']!=104:raise ValueError('Retained full104stage backbone numerical gate rejected')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('prepare','run'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
    if os.environ.get('GITHUB_ACTIONS')!='true':p.error('All downloads/frameworks/builds/inference belong on hosted CPUCI only')
    if a.phase=='prepare' and a.output.exists():p.error('Fresh task-owned output required')
    if a.phase=='run' and (a.output/'review/backbone.json').exists():p.error('Preserve earlier execution; fresh run required')
    (a.output/'review').mkdir(parents=True,exist_ok=True)
    try:
        if a.phase=='prepare':prepare(a.output,a.scratch)
        else:run(a.output,a.scratch)
    except Exception as error:
        write(a.output/'review'/('rejection-'+a.phase+'.json'),dict(phase=a.phase,error=type(error).__name__+': '+str(error),full_model=False,release_qualified=False));raise
    finally:
        write(a.output/'review'/('files-'+a.phase+'.json'),{str(f.relative_to(a.output)):dict(bytes=f.stat().st_size,sha256=sha256(f))
            for f in a.output.rglob('*') if f.is_file() and f.name!=('files-'+a.phase+'.json')})


if __name__=='__main__':main()
