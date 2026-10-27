#!/usr/bin/env python3
"""Authenticate and install the immutable lock inside the original GPU task only."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_environment import isolated_prefix,locked_files,owner_metadata_file,verify_installed
SPEC=importlib.util.spec_from_file_location('josev_archive_audit',ROOT/'tools/ci-heavy/josev2b_wheel_metadata.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)
CONTRACT_SHA='6fbe3efe8958b2d87f5fe42c52fa71857c2d644efe3ad363bb60d8dfa52885da'
MAX_SECONDS=1380


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def hardware(rows):
    if not rows or any(len(r)!=5 for r in rows):raise ValueError('GPU hardware declaration unavailable')
    for _,cap,total,free,driver in rows:
        if tuple(map(int,cap.strip().split('.')))<(7,5):raise ValueError('unsupported GPU; no retry')
        if tuple(map(int,driver.strip().split('.')))<(580,65,6):raise ValueError('unsupported CUDA13 driver; no retry')
        if int(total)<14*1024 or int(free)<11*1024:raise ValueError('GPU memory bound not met')
    return [[s.strip() for s in row] for row in rows]


def guard(args):
    # No wheel request or framework import may precede genuine task/GPU proof.
    if (os.environ.get('CRISPEMBED_JOSEV_GPU_REFERENCE_SETUP')!='1' or
            not Path('/kaggle/working').is_dir()):raise ValueError('Hosted original GPU task only')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if current!=args.reviewed_source:raise ValueError('Reviewed worker source differs')
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=ROOT,check=True)
    cfg_path=ROOT/'docs/josev2b-source-contract.json'
    if audit.sha(cfg_path.read_bytes())!=CONTRACT_SHA:raise ValueError('Frozen model/profile contract differs')
    proof=json.loads(args.gpu_preflight.read_text())
    if (proof['worker_source']!=current or proof['contract_sha256']!=CONTRACT_SHA or
            proof['runtime_profile']!='cuda-f32-python313-v3-locked' or
            proof['model_payload_accessed'] is not False or proof['retry_permitted'] is not False):
        raise ValueError('Original GPU bootstrap preflight identity differs')
    previous=hardware(proof['hardware'])
    actual=hardware([r.split(',') for r in subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,compute_cap,memory.total,memory.free,driver_version',
        '--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()])
    # Free memory may change, while every physical GPU/driver identity must agree.
    if [r[:3]+r[4:] for r in previous]!=[r[:3]+r[4:] for r in actual]:raise ValueError('Preflight/current GPU identities differ')
    isolation=isolated_prefix(args.expected_prefix)
    cfg=json.loads(cfg_path.read_text());lock=locked_files(cfg)
    limits=json.loads(audit.PROTOCOL.read_text())
    for name,sha in limits['old_source_files'].items():
        if name!='tools/josev2b_environment.py' and audit.sha((ROOT/name).read_bytes())!=sha:
            raise ValueError('Immutable environment/model declaration differs: '+name)
    if len(lock['packages'])!=57 or sum(r['bytes'] for r in lock['packages'])!=2790174915:
        raise ValueError('Complete57-wheel scope differs')
    if any(m in sys.modules for m in ('torch','torchvision','numpy','transformers','tokenizers','accelerate','safetensors')):
        raise ValueError('Framework imported before wheel audit barrier')
    return cfg,lock,limits,dict(head=current,hardware=actual,isolation=isolation,
        environment_lock_sha256=cfg['environment_lock']['sha256'],contract_sha256=CONTRACT_SHA,
        files={p:audit.sha((ROOT/p).read_bytes()) for p in ('tools/ci-heavy/josev2b_gpu_install.py',
            'tools/ci-heavy/josev2b_wheel_metadata.py','tools/josev2b_environment.py',
            'docs/josev2b-wheel-metadata-protocol-v1.json')})


def installed_owner(row,prefix):
    matches=[d for d in metadata.distributions() if audit.canonical(d.metadata['Name'])==row['name']]
    if len(matches)!=1:raise ValueError('Installed distribution missing/ambiguous: '+row['name'])
    dist=matches[0];prefix=prefix.resolve()
    if dist.version!=row['version'] or not Path(dist.locate_file('')).resolve().is_relative_to(prefix):
        raise ValueError('Installed version/origin differs: '+row['name'])
    path=Path(dist.locate_file(owner_metadata_file(dist,row['name'],row['version']))).resolve()
    if not path.is_relative_to(prefix):raise ValueError('Installed owner escapes isolated prefix')
    data=path.read_bytes()
    if len(data)!=row['core_metadata_bytes'] or audit.sha(data)!=row['core_metadata_sha256']:
        raise ValueError('Installed owner metadata bytes differ: '+row['name'])
    return dict(version=dist.version,origin_inside_venv=True,owner_metadata_bytes=len(data),owner_metadata_sha256=audit.sha(data))


def install_local(wheel,row,scratch,out,deadline):
    # Rehash immediately before pip; it receives a local file, never a live URL.
    if (wheel.is_symlink() or wheel.resolve().parent!=scratch.resolve() or wheel.name!=row['filename'] or
            wheel.stat().st_size!=row['bytes']):raise ValueError('Authenticated local wheel path/extent differs')
    h=__import__('hashlib').sha256()
    with wheel.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=row['sha256']:raise ValueError('Authenticated local wheel changed before installation')
    requirements=scratch/'authenticated.lock'
    requirements.write_text(wheel.resolve().as_uri()+' --hash=sha256:'+row['sha256']+'\n')
    log=out/(row['name']+'.install.log')
    remaining=deadline-time.monotonic()
    if remaining<=0:raise ValueError('Total audited install deadline exceeded')
    with log.open('w') as stream:
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--require-hashes',
            '--only-binary=:all:','--disable-pip-version-check','--no-cache-dir','-r',str(requirements)],
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=min(240,remaining))
    return {'exit_code':0,'log':log.name,'log_sha256':audit.sha(log.read_bytes()),
            'source':'authenticated task-local wheel; no index/dependency resolution'}


def one(row,out,scratch_root,prefix,deadline,limits,result,save):
    result.update(name=row['name'],version=row['version'],filename=row['filename'],audited=False,installed=False)
    save()
    # Only this task's single temporary wheel is removed, including on rejection.
    with tempfile.TemporaryDirectory(prefix='wheel-',dir=scratch_root) as directory:
        scratch=Path(directory)
        if shutil.disk_usage(scratch).free<row['bytes']+512*1024**2:raise ValueError('Insufficient single-wheel scratch')
        sidecar=out/(row['name']+'.sidecar.metadata')
        result['sidecar_transfer']=audit.transfer(row,sidecar,sidecar=True,deadline=deadline,limits=limits);save()
        wheel=scratch/row['filename']
        result['wheel_transfer']=audit.transfer(row,wheel,sidecar=False,deadline=deadline,limits=limits);save()
        inspected,owner=audit.archive_metadata(wheel,row,sidecar.read_bytes(),limits)
        result['archive']=inspected
        (out/(row['name']+'.owner.metadata')).write_bytes(owner);save()
        if not inspected['passed']:raise ValueError('Exact wheel owner/sidecar/dependency audit rejected before install')
        result['audited']=True;save()
        result['install']=install_local(wheel,row,scratch,out,deadline);save()
        result['installed_owner']=installed_owner(row,prefix)
        result['installed']=True;save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed-source',required=True);p.add_argument('--gpu-preflight',required=True,type=Path)
    p.add_argument('--expected-prefix',required=True,type=Path);p.add_argument('--scratch',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists() or args.scratch.exists():p.error('Fresh task-owned output/scratch required')
    args.output.mkdir(parents=True)
    report=dict(schema='crispembed-josev-gpu-audited-install-v1',completed=False,passed=False,
        framework_imported=False,model_tokenizer_accessed=False,retry_permitted=False,packages=[],errors=[])
    def save():write(args.output/'audit.json',report)
    save()
    try:
        cfg,lock,limits,report['source']=guard(args);save()
        args.scratch.mkdir(parents=True);deadline=time.monotonic()+MAX_SECONDS
        for row in lock['packages']:
            result={};report['packages'].append(result)
            try:one(row,args.output,args.scratch,args.expected_prefix,deadline,limits,result,save)
            except Exception as exc:
                result['error']=type(exc).__name__+': '+str(exc);save();raise
            print(json.dumps(dict(name=row['name'],audited=result['audited'],installed=result['installed'])),flush=True)
        if not all(r['audited'] and r['installed'] for r in report['packages']) or len(report['packages'])!=57:
            raise ValueError('Incomplete audited installation')
        with (args.output/'pip-check.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','pip','check'],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=60)
        report['pip_check']=True
        report['installed_environment']=verify_installed(cfg,args.expected_prefix,import_runtime=False)
        report.update(completed=True,passed=True);save()
    except Exception as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc));save();raise
    finally:
        report['files']={f.name:dict(bytes=f.stat().st_size,sha256=audit.sha(f.read_bytes()))
            for f in args.output.iterdir() if f.is_file() and f.name!='audit.json'};save()

if __name__=='__main__':main()
