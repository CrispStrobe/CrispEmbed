#!/usr/bin/env python3
"""Hosted exact-version canonical fetch; analysis never fetches checkpoint files."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,CATALOG,atomic,require,sha,safe,canonical,duplicate,source_guard,resources,numerical_environment,ContractError


def transfer(url,path,pin,cap):
    require(url.startswith('https://')and 0<cap<=8*2**30,'Bounded HTTPS transfer required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();count=0
    try:
        with urllib.request.urlopen(url,timeout=180)as response,path.open('xb')as output:
            length=int(response.headers['Content-Length']);require(0<=length<=cap,'Member exceeds predeclared header budget')
            if 'bytes'in pin:require(length==pin['bytes'],'Pinned member extent differs')
            while block:=response.read(2**20):
                count+=len(block);require(count<=length,'Member exceeds advertised extent');digest.update(block);output.write(block)
    except Exception:raise ValueError('Authenticated retained transfer rejected; task-owned partial retained: '+path.name)from None
    require(count==length and digest.hexdigest()==pin['sha256'],'Retained whole-file SHA differs: '+path.name)
    return dict(bytes=count,sha256=digest.hexdigest())


def client_index(api,pin):
    module=importlib.import_module('kaggle.api.kaggle_api_extended')
    Get=module.ApiGetKernelRequest;List=module.ApiListKernelSessionOutputRequest;Status=module.ApiGetKernelSessionStatusRequest
    def request(kind):
        q=kind();q.user_name=os.environ['KAGGLE_ACCOUNT'];q.kernel_slug=pin['kernel_slug'];q.version_label=pin['version_label']
        require(json.loads(str(q)).get('versionLabel')==pin['version_label'],'SDK lost exact serialized version')
        return q
    files={};token=None;pages=0
    try:
        with api.build_kaggle_client()as client:
            provider=client.kernels.kernels_api_client
            source=json.loads(str(provider.get_kernel(request(Get))))['blob']['source']
            require(hashlib.sha256(source.encode()).hexdigest()==pin['submitted_script_sha256']and pin['source']in source,'Frozen submitted source differs')
            status=provider.get_kernel_session_status(request(Status))
            require(str(status.status).split('.')[-1]==pin['status'],'Exact retained success/failure service status differs')
            while True:
                q=request(List);q.page_size=200
                if token:q.page_token=token
                response=provider.list_kernel_session_output(q);pages+=1;require(pages<=100,'Output index exceeds bound')
                for item in response.files or []:
                    require(item.file_name not in files,'Duplicate retained output member');files[item.file_name]=item.url
                token=response.next_page_token
                if not token:break
    except ContractError:raise
    except Exception:raise ValueError('Exact-version retained service request rejected')from None
    return files,dict(kernel_slug=pin['kernel_slug'],version_label=pin['version_label'],source=pin['source'],
        submitted_script_sha256=pin['submitted_script_sha256'],status=pin['status'],pages=pages)



def fetch(mode,stage,out):
    import importlib.metadata
    require(os.environ.get('KAGGLE_API_TOKEN')and os.environ.get('KAGGLE_ACCOUNT'),'Existing Kaggle secrets absent; no fetch')
    require(importlib.metadata.version('kaggle')=='2.2.2','Exact SDK tooling version required')
    module=importlib.import_module('kaggle.api.kaggle_api_extended');api=module.KaggleApi()
    try:api.authenticate()
    except Exception:raise ValueError('Existing hosted Kaggle credentials rejected')from None
    cfg=json.loads(CATALOG.read_text());urls,service=client_index(api,cfg['service'])
    root=Path(stage)/'retained';root.mkdir(parents=True,exist_ok=False)
    transfer(urls['minicpm-graph/panel/panel.json'],root/'panel.json',cfg['panel'],2**20)
    for pin in cfg['cases']:
        folder=Path(pin['case_report']).parent;prefix='minicpm-graph/panel/'+str(folder)
        transfer(urls[prefix+'/case-report.json'],root/pin['case_report'],pin['report'],2*2**20)
        transfer(urls[prefix+'/files.json'],root/folder/'files.json',pin['files_manifest'],2*2**20)
        inventory=json.loads((root/folder/'files.json').read_text())
        for name,identity in inventory.items():
            if duplicate(name):continue
            require(canonical(name)or Path(name).suffix in ['.json','.jsonl','.log','.stdout','.stderr'],'Unexpected retained bulk member')
            member=prefix+'/'+name;require(member in urls,'Retained canonical member absent')
            transfer(urls[member],safe(root/folder,name),identity,1024**3 if canonical(name)else 2*2**20)
    for name,identity in cfg['initial_files'].items():
        require(name in urls,'Initial retained producer/control member absent')
        transfer(urls[name],safe(root/'initial',name),identity,2*2**20)
    original_urls,original_service=client_index(api,cfg['original_service']);original=Path(stage)/'original';original.mkdir(exist_ok=False)
    transfer(original_urls['minicpm-ocr/reference.json'],original/'reference.json',cfg['original_report'],2**20)
    ref=json.loads((original/'reference.json').read_text());require(ref['completed']is True and len(ref['cases'])==5,'Exact frozen original panel required')
    for case in ref['cases']:
        name=case['id']+'.png';transfer(original_urls['minicpm-ocr/'+name],original/name,dict(sha256=case['image_sha256']),2*2**20)
        for label,row in case['arrays'].items():
            member='minicpm-ocr/'+row['path'];require(member in original_urls,'Frozen original array absent')
            cap=8*__import__('math').prod(row['shape'])+4096
            transfer(original_urls[member],safe(original,row['path']),row,cap)
    if mode=='continue':
        source=json.loads((ROOT/'docs/minicpm-ocr-source.json').read_text());checkpoint=Path(stage)/'checkpoint';checkpoint.mkdir(exist_ok=False)
        for name,pin in source['files'].items():
            member='checkpoint/'+name;require(member in original_urls,'Exact original checkpoint member missing')
            transfer(original_urls[member],safe(checkpoint,name),pin,8*2**30)
    proof=dict(completed=True,mode=mode,retained_service=service,original_service=original_service,
               catalog_sha256=sha(CATALOG),model_files_accessed=mode=='continue',original_rerun=False)
    atomic(Path(out)/'fetch.json',proof)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['analysis','continue']);parser.add_argument('--reviewed-source',required=True)
    for name in ['stage','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    result=dict(completed=False,mode=args.mode,original_rerun=False)
    try:
        result['source']=source_guard(args.mode,args.reviewed_source);result['resources']=resources(args.out,args.stage)
        result['tooling']=numerical_environment()
        result.update(fetch(args.mode,args.stage,args.out))
    except Exception as error:result['error']=type(error).__name__+': '+str(error);raise
    finally:atomic(args.out/'fetch.json',result)


if __name__=='__main__':main()
