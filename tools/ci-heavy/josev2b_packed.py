#!/usr/bin/env python3
"""Hosted packing and exact-source packed-runtime parity; no public-API changes."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import pack_josev2b_f32 as packer
PROTOCOL='docs/josev2b-packed-validation-v1.json'
PROTOCOL_SHA='fe889becbeabee70fe16495f2a96ef971a2b5793752a30f01b16b0da8970ca0c'
OWN=(PROTOCOL,'tools/ci-heavy/josev2b_packed.py','tests/test_josev2b_packed_worker.py',
     '.github/workflows/josev2b-packed.yml','docs/josev2b-packed-f32.md','.github/workflows/ovisocr2-metadata.yml')


def require(ok,why):
    if not ok:raise ValueError(why)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path):return json.loads(path.read_text())
def sha(path):return packer.sha_file(path)
def git(*args):return packer.git(*args)
def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))


def policy():
    require(os.environ.get('GITHUB_ACTIONS')=='true','Hosted CPU packing/qualification only')
    require(git('rev-parse','HEAD')==os.environ.get('GITHUB_SHA'),'Caller/source service identity differs')
    require(sha(ROOT/PROTOCOL)==PROTOCOL_SHA,'Packed validation protocol differs')
    cfg=load(ROOT/PROTOCOL);closure=packer.source_guard()
    require(cfg['schema']=='crispembed-josev-packed-validation-v1'and cfg['pack_profile_sha256']==packer.PROFILE_SHA,'Explicit packer/runtime protocol differs')
    for path in OWN:
        require(git('hash-object',path)==git('rev-parse','HEAD:'+path),'Uncommitted qualification source: '+path)
    require(not git('status','--porcelain','--',*OWN),'Dirty qualification source')
    for path,obj in cfg['runtime_objects'].items():
        require(git('rev-parse',cfg['runtime_source']+':'+path)==obj,'Declared runtime object differs: '+path)
    return cfg,dict(caller=closure,worker_files={p:sha(ROOT/p)for p in OWN},runtime_source=cfg['runtime_source'],runtime_objects=cfg['runtime_objects'])


def fetch(pin,scratch,*,include=None):
    """Exact retained service archive; selected members have explicit pinned names."""
    run=api('actions/runs/'+str(pin['run']))
    require(run['head_sha']==pin['source']and run['path']=='.github/workflows/ovisocr2-metadata.yml'and
            run['event']=='workflow_dispatch'and run['run_attempt']==1 and run['status']=='completed'and run['conclusion']=='success','Immutable producer service/source/attempt differs')
    a=api('actions/artifacts/'+str(pin['id']))
    require(a['workflow_run']['id']==pin['run']and a['workflow_run']['head_sha']==pin['source']and
            a['name']==pin['name']and a['size_in_bytes']==pin['bytes']and a['digest']=='sha256:'+pin['sha256']and not a['expired'],'Retained artifact service identity differs')
    require(0<pin['bytes']<=5*2**30,'Declared artifact download budget exceeded')
    target=scratch/str(pin['id']);target.mkdir();archive=scratch/(str(pin['id'])+'.zip')
    with archive.open('xb')as stream:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip'],stdout=stream,check=True,timeout=1800)
    require(archive.stat().st_size==pin['bytes']and sha(archive)==pin['sha256'],'Downloaded immutable ZIP bytes differ')
    with zipfile.ZipFile(archive)as z:
        items=z.infolist();names=[r.filename for r in items]
        require(len(items)<=8192 and len(set(names))==len(names)and sum(r.file_size for r in items)<=20*2**30,'Archive member/extent budget exceeded')
        for row in items:
            name=row.filename;p=PurePosixPath(name)
            require(name and not p.is_absolute()and '..'not in p.parts and '\\'not in name and p.as_posix()==name.rstrip('/')and
                    (row.external_attr>>16)&0o170000!=0o120000,'Unsafe retained member')
        if include is not None:require(set(include)<=set(names),'Exact retained component member missing')
        for row in items:
            if row.is_dir()or include is not None and row.filename not in include:continue
            path=target/row.filename;path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(row)as src,path.open('xb')as dst:shutil.copyfileobj(src,dst,4*2**20)
    archive.unlink() # Only this task-owned download; immutable producer retained remotely.
    return target,{k:a[k]for k in ('id','name','size_in_bytes','digest','expires_at')}


def command(args,log,*,cwd=ROOT,env=None,timeout=1200):
    start=time.monotonic()
    with log.open('w')as stream:r=subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    return dict(exit=r.returncode,elapsed_seconds=time.monotonic()-start,log_sha256=sha(log))


def runtime_checkout(cfg,scratch):
    source=cfg['runtime_source'];target=scratch/'runtime'
    subprocess.run(['git','cat-file','-e',source+'^{commit}'],cwd=ROOT,check=True,timeout=30)
    subprocess.run(['git','worktree','add','--detach',str(target),source],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','submodule','update','--init','ggml'],cwd=target,check=True,timeout=300)
    for path,obj in cfg['runtime_objects'].items():
        actual=subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=target,text=True).strip()
        require(actual==obj,'Hosted runtime checkout object differs: '+path)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=target/'ggml',text=True).strip()==cfg['runtime_objects']['ggml'],'Actual packed backend checkout differs')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=target/'ggml',text=True),'Dirty packed backend checkout')
    return target


def comparison_helpers(cfg,scratch):
    directory=scratch/'comparison-tools';directory.mkdir()
    for path,digest in cfg['reference_helpers'].items():
        data=subprocess.check_output(['git','show',cfg['reference_source']+':'+path],cwd=ROOT)
        require(hashlib.sha256(data).hexdigest()==digest,'Exact original comparison helper source differs')
        (directory/Path(path).name).write_bytes(data)
    sys.path.insert(0,str(directory))
    require('josev2b_composed_diff'not in sys.modules,'Comparison module already imported from another source')
    spec=importlib.util.spec_from_file_location('packed_original_comparison',directory/'josev2b_composed_panel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_head_archive(directory):
    receipt=load(directory/'review/artifact.json')
    require(receipt['completed']is True and receipt['arrays']==251 and receipt['reference_sha256']=='51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090','Frozen original archive scope differs')
    files={p.relative_to(directory).as_posix()for p in directory.rglob('*')if p.is_file()}
    require(files==set(receipt['files'])|{'review/artifact.json'},'Frozen original archive complete coverage differs')
    for name,pin in receipt['files'].items():
        path=directory/name;require(path.stat().st_size==pin['bytes']and sha(path)==pin['sha256'],'Frozen original member bytes differ: '+name)
    reference=directory/'original'
    require(sha(reference/'reference.json')=='51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090'and
            sha(reference/'input-freeze.json')=='9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7','Frozen requests/reference bytes differ')
    return reference


def run(out,scratch):
    cfg,closure=policy();require(importlib.metadata.version('numpy')=='2.2.6','Pinned independent comparison package differs')
    require(not out.exists()and not scratch.exists(),'Fresh task-owned output/scratch required')
    out.mkdir(parents=True);scratch.mkdir(parents=True)
    available=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    require(available>=cfg['resources']['minimum_ram_bytes']and shutil.disk_usage(scratch).free>=cfg['resources']['minimum_disk_bytes'],'Hosted10GiBRAM/25GiBdisk required')
    report=dict(schema='crispembed-josev-packed-qualification-v1',completed=False,source=closure,commands={},full_model_qualified=False,release_qualified=False,public_api_qualified=False)
    review=out/'review';review.mkdir();raw=out/'raw';raw.mkdir();write(review/'hosted.json',report)
    env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')};env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        pp=packer.profile();composed=pp['composed_review']
        directory,service=fetch(dict(composed['review_artifact'],run=composed['run'],source=composed['source']),scratch)
        composed_path=directory/composed['member'];packer.reviewed_composition(pp,composed_path)
        require(sha(directory/'comparison/panel.json')==cfg['composed_panel_sha256'],'Reviewed composed panel bytes differ')
        shutil.copyfile(composed_path,review/'composed-prerequisite.json');report['composed_prerequisite']=service
        runtime=runtime_checkout(cfg,scratch);build=scratch/'build';panel=comparison_helpers(cfg,scratch)
        # Build and actual weight-free contract/Ref controls BEFORE bulk archives.
        jobs=[('runtime-source',[sys.executable,runtime/'tools/josev2b_packed_runtime_contract.py']),
              ('configure',['cmake','-S',runtime/'tests/josev-packed','-B',build,'-DCMAKE_BUILD_TYPE=Release']),
              ('build',['cmake','--build',build,'--parallel','2']),
              ('metadata-controls',[build/'test-josev-packed']),
              ('gates-plain',[build/'josev-packed-plain-probe','--gates']),
              ('gates-observed',[build/'josev-packed-observed-probe','--gates'])]
        for name,args in jobs:
            report['commands'][name]=command(args,review/(name+'.log'),cwd=runtime,env=env)
            write(review/'hosted.json',report);require(report['commands'][name]['exit']==0,'Packed build/contract control rejected: '+name)
        panel.ref_controls(build/'josev-packed-diff',review/'ref-controls')
        commands=load(build/'compile_commands.json');names={'josev2b_composed.cpp','josev2b_backbone.cpp','josev2b_head.cpp','josev2b_text_frontend.cpp','josev2b_answer_serializer.cpp','qwen35_decoder.cpp','shortest_double.c'}
        selected=[r for r in commands if Path(r['file']).name in names]
        require(len(selected)==14 and all('-ffp-contract=off'in r['command']and not any(f in r['command']for f in ('-ffast-math','-Ofast','-ffp-contract=fast'))for r in selected),'Actual packed compiler arithmetic flags differ')
        write(review/'compile-commands.json',selected)
        archives={}
        for kind in ('plain','observed'):
            archive=build/('libjosev-packed-'+kind+'.a');symbols=subprocess.check_output(['nm','-C',archive],text=True);strings=subprocess.check_output(['strings',archive],text=True)
            if kind=='plain':require(not any(t in symbols for t in ('composed_test::','josev2b_head::test::','run_observed'))and 'head/option_context_projection'not in strings,'Observer leaked into ordinary packed archive')
            else:require('josev2b::composed_test::observe'in symbols and 'josev2b_head::test::run_observed'in symbols,'Observed packed archive positive control missing')
            archives[kind]=dict(sha256=sha(archive),bytes=archive.stat().st_size)
        report['archives']=archives;report['ordinary_excludes_observers']=True
        components={};services={}
        # Explicit selected-member pin proves the tokenizer layout before GB fetch.
        for kind in ('tokenizer','backbone','head'):
            pin=pp['components'][kind];include=None if kind=='head'else {cfg['component_members'][kind]}
            target,proof=fetch(pin['source_artifact'],scratch,include=include);components[kind]=target/cfg['component_members'][kind];services[kind]=proof
        head_root=components['head'].parent.parent;reference=validate_head_archive(head_root)
        report['components']=services;write(review/'hosted.json',report)
        packed=raw/'josev2b-systemone-f32.gguf'
        repack=packer.pack(components['backbone'],components['head'],components['tokenizer'],composed_path,packed)
        write(review/'repack.json',repack);report['packed_asset']=dict(bytes=packed.stat().st_size,sha256=sha(packed),tensor_count=442)
        for name in cfg['notices']:
            src=head_root/'head'/name;require(src.is_file(),'Required original notice missing: '+name)
            dest=raw/'notices'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest)
        original=load(reference/'reference.json');freeze=load(reference/'input-freeze.json');requests=raw/'requests.json'
        write(requests,[dict(id=r['id'],request_json=json.dumps(r['request'],ensure_ascii=False,allow_nan=False,separators=(',',':')))for r in freeze['records']])
        env.update(cfg['gates']);outputs={}
        for label,variant in [('plain','plain'),('observed','observed'),('repeat','observed')]:
            target=raw/label;outputs[label]=target
            report['commands'][label]=command([build/('josev-packed-'+variant+'-probe'),packed,requests,target],review/(label+'.log'),env=env,timeout=1800)
            write(review/'hosted.json',report)
        require(all(report['commands'][k]['exit']==0 for k in outputs),'Packed native structural rejection; all3arms retained')
        result=panel.compare(reference,freeze,original,outputs,build/'josev-packed-diff',raw/'comparison')
        shutil.copyfile(raw/'comparison/panel.json',review/'panel.json')
        for i in range(4):
            src=raw/'comparison'/str(i)/'diff.json'
            if src.exists():shutil.copyfile(src,review/('diff-'+str(i)+'.json'))
        passed=result['numeric_and_complete_typed_parity']and result['compared_stages']==227
        report.update(completed=True,packed_numeric_and_complete_typed_parity=passed,compared_stages=result['compared_stages'],failures=result['failures'])
        write(review/'hosted.json',report);return int(not passed)
    except Exception as error:
        message=str(error)
        for path,label in ((str(ROOT),'<caller>'),(str(scratch),'<scratch>'),(str(out),'<out>')):message=message.replace(path,label)
        report.update(error_type=type(error).__name__,error=message);write(review/'hosted.json',report);return 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args()
    try:return run(a.output.resolve(),a.scratch.resolve())
    except Exception as error:
        # Preserve early source/resource rejections without changing an older run.
        if not a.output.exists()or not any(a.output.iterdir()):
            message=str(error)
            for path,label in ((str(ROOT),'<caller>'),(str(a.scratch.resolve()),'<scratch>'),(str(a.output.resolve()),'<out>')):message=message.replace(path,label)
            write(a.output/'review/initial-rejection.json',dict(schema='crispembed-josev-packed-qualification-v1',completed=False,error_type=type(error).__name__,error=message,full_model_qualified=False,release_qualified=False))
        return 1


if __name__=='__main__':sys.exit(main())
