#!/usr/bin/env python3
"""Genuine GHA admission and exact-version original fetch; no inference changes.

Signed URLs and private account/token values stay in memory, never receipts.
This entry point never reruns originals and has no model/framework downloader.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path,PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/moedern-scifact')]
from contract_gha_v1 import declaration,source,require,sha,git
from providers_v1 import api


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def environment(env):
    expected={'GITHUB_ACTIONS':'true','RUNNER_ENVIRONMENT':'github-hosted','RUNNER_OS':'Linux',
              'RUNNER_ARCH':'X64','GITHUB_REPOSITORY':'CrispStrobe/CrispEmbed'}
    require(all(env.get(k)==v for k,v in expected.items()),'Genuine GitHub-hosted Linux x64 service required; no Kaggle spoof')
    require(env.get('GITHUB_WORKFLOW_REF','').startswith('CrispStrobe/CrispEmbed/.github/workflows/moedern-scifact-gha-adapter-v1.yml@')and
            env.get('GITHUB_EVENT_NAME')=='workflow_dispatch','Explicit standalone manual workflow required')
    require(all(re.fullmatch('[1-9][0-9]*',env.get(k,''))for k in ['GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT'])and
            re.fullmatch('[0-9a-f]{40}',env.get('GITHUB_SHA','')),'Exact GHA service identity required')
    require(env.get('GH_TOKEN')and env.get('KAGGLE_API_TOKEN')and env.get('KAGGLE_ACCOUNT'),
            'Existing workflow artifact-read/Kaggle secrets absent; stop before fetch')
    require(platform.system()=='Linux'and platform.machine()=='x86_64'and hasattr(os,'wait4'),
            'Linux x64 wait4 required')


def limits(ram,disk,cpus):
    cfg=declaration()
    require(ram>=cfg['minimum_available_ram_bytes']and disk>=cfg['minimum_free_disk_bytes']and cpus>=2,
            'Unchanged14GiB available RAM/24GiB free disk/two CPU admission rejected before fetch')


def admission(scratch,reviewed):
    environment(os.environ)
    require(re.fullmatch('[0-9a-f]{40}',reviewed)and git('rev-parse','HEAD').decode().strip()==reviewed,
            'Explicit checked-out source differs')
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    ram=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(scratch).free;cpus=os.cpu_count()or 0;limits(ram,disk,cpus)
    run=api('actions/runs/'+os.environ['GITHUB_RUN_ID'],os.environ['GH_TOKEN'])
    require(run['id']==int(os.environ['GITHUB_RUN_ID'])and run['head_sha']==os.environ['GITHUB_SHA']and
            run['run_attempt']==int(os.environ['GITHUB_RUN_ATTEMPT'])and run['status']=='in_progress'and
            run['event']=='workflow_dispatch'and run['repository']['full_name']=='CrispStrobe/CrispEmbed'and
            run['path'].split('@')[0]=='.github/workflows/moedern-scifact-gha-adapter-v1.yml',
            'Live GHA run service identity differs')
    return dict(provider='GitHub Actions hosted CPU',run=run['id'],attempt=run['run_attempt'],
                workflow_source=run['head_sha'],checkout_source=reviewed,available_ram_bytes=ram,
                free_disk_bytes=disk,cpu_count=cpus,threads=2,gpu_execution=False,
                secrets_present=True,resource_budget_lowered=False)


def stream(url,target,identity):
    require(url.startswith('https://')and type(identity['bytes'])is int and 0<identity['bytes']<=1024**3 and
            re.fullmatch('[0-9a-f]{64}',identity['sha256']),'Exact bounded HTTPS member identity required')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();count=0
    try:
        with urllib.request.urlopen(url,timeout=180)as response,target.open('xb')as output:
            length=response.headers.get('Content-Length')
            require(length is None or int(length)==identity['bytes'],'Original member Content-Length differs')
            while block:=response.read(2**20):
                count+=len(block);require(count<=identity['bytes'],'Original member exceeds extent')
                h.update(block);output.write(block)
    except Exception:raise ValueError('Immutable original transfer rejected; task-owned partial retained: '+target.name)from None
    require(count==identity['bytes']and h.hexdigest()==identity['sha256'],'Original member final size/SHA differs: '+target.name)
    return dict(bytes=count,sha256=h.hexdigest())


def indexed_outputs(client,Get,List,Status,owner,pin):
    def request(kind):
        obj=kind();obj.user_name=owner;obj.kernel_slug=pin['kernel_slug'];obj.version_label=pin['version_label']
        require(json.loads(str(obj)).get('versionLabel')==pin['version_label'],
                'Actual SDK request serialization lost explicit version; no CLI fallback')
        return obj
    response=client.get_kernel(request(Get));metadata=json.loads(str(response))
    submitted=metadata['blob']['source']
    require(hashlib.sha256(submitted.encode()).hexdigest()==pin['submitted_script_sha256']and pin['source']in submitted,
            'Exact submitted original source differs')
    status=client.get_kernel_session_status(request(Status))
    require(str(status.status).split('.')[-1]=='COMPLETE','Exact-version original is not complete')
    files={};token=None;pages=0
    while True:
        q=request(List);q.page_size=100
        if token:q.page_token=token
        page=client.list_kernel_session_output(q);pages+=1
        require(pages<=100,'Bounded original output pagination exceeded')
        for item in page.files or []:
            name=item.file_name
            require(name not in files,'Duplicate original output member')
            files[name]=item.url
        token=page.next_page_token
        if not token:break
    return files,dict(kernel_slug=pin['kernel_slug'],version_label=pin['version_label'],source=pin['source'],
                      submitted_script_sha256=pin['submitted_script_sha256'],status='COMPLETE',pages=pages,
                      actual_sdk_serialized_version_on_get_status_and_all_pages=True)


def fetch(variant,mount,out):
    cfg=source(variant);pin=declaration()['original_service'][variant]
    # Tooling lives outside the isolated five-package numerical worker env.
    import importlib.metadata
    require(importlib.metadata.version('kaggle')=='2.2.2','Exact Kaggle SDK tooling version required')
    mod=importlib.import_module('kaggle.api.kaggle_api_extended');api_client=mod.KaggleApi();api_client.authenticate()
    Get=mod.ApiGetKernelRequest;List=mod.ApiListKernelSessionOutputRequest;Status=mod.ApiGetKernelSessionStatusRequest
    with api_client.build_kaggle_client()as client:
        files,proof=indexed_outputs(client.kernels.kernels_api_client,Get,List,Status,os.environ['KAGGLE_ACCOUNT'],pin)
    root=Path(mount)/'full-original';require(not root.exists(),'Fresh original mount required')
    report_member=pin['root_member']+'/review/original.json'
    require(report_member in files,'Exact original report member missing')
    original=cfg['originals'][variant];target=root/'review/original.json'
    identity=dict(bytes=original['report_bytes'],sha256=original['report_sha256'])
    stream(files[report_member],target,identity)
    report=json.loads(target.read_text())
    require(report['completed']is True and report['native_executed']is False and report['variant']==variant and
            report['source']==pin['source']and report['arrays']==original['arrays'],
            'Reviewed complete original report/raw identity differs')
    expected={'data.json','rows.json','all.f32','selected.f32','scores.npy','rankings.json'}
    require(set(report['arrays'])==expected,'Exact six original raw members required')
    proof['tooling_packages']={d.metadata['Name']:{'version':d.version,'metadata_sha256':sha(Path(d._path)/'METADATA')}
        for d in importlib.metadata.distributions() if d.metadata['Name']}
    proof['kaggle_sdk_wheel']=declaration()['kaggle_sdk_wheel']
    proof['members']={report_member:identity}
    for name in sorted(expected):
        member=pin['root_member']+'/raw/'+name
        require(member in files,'Original raw member missing: '+name)
        proof['members'][member]=stream(files[member],root/'raw'/name,report['arrays'][name])
        write(Path(out)/'original-service.json',proof)
    proof.update(completed=True,all_six_raw_sha_verified=True,original_rerun=False,model_accessed=False)
    write(Path(out)/'original-service.json',proof)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['admit','fetch','run'])
    parser.add_argument('--variant',choices=['1.5M'],required=True)
    parser.add_argument('--reviewed-source',required=True)
    for name in ['scratch','mount','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker-python',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    report=dict(schema='crispembed-moedern-scifact-gha-adapter-receipt-v1',completed=False,phase=args.phase,
                inference_changes=False,publication_qualified=False,original_rerun=False)
    try:
        baseline=source(args.variant);report['source_adapter']=baseline['provider_adapter']
        report['service']=admission(args.scratch,args.reviewed_source);write(args.out/'admission.json',report)
        if args.phase=='fetch':report['original']=fetch(args.variant,args.mount,args.out)
        if args.phase=='run':
            require(args.worker_python and args.worker_python.is_file(),'Isolated worker Python required')
            proof=json.loads((args.out/'original-service.json').read_text())
            require(proof.get('completed')is True and proof.get('all_six_raw_sha_verified')is True,
                    'Complete authenticated original fetch receipt required before worker')
            from ci_process import run as run_child
            result=run_child([args.worker_python,'-I',ROOT/'tools/moedern-scifact/worker_gha_v1.py',
                '--variant',args.variant,'--reviewed-source',args.reviewed_source,'--mount',args.mount,
                '--scratch',args.scratch/'model','--out',args.out/'result'],args.out/'worker.log',
                cwd=ROOT,env=os.environ.copy(),timeout=270*60)
            report['process']=dict(result,log='worker.log',sha256=sha(args.out/'worker.log'))
            write(args.out/'worker-process.json',report['process'])
            require(not result['timed_out']and result['exit']==0,'Hosted worker rejected; all partial results retained')
        report['completed']=True
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error);raise
    finally:write(args.out/(args.phase+'-adapter.json'),report)


if __name__=='__main__':main()
