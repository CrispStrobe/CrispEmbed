"""Hosted retained-byte authentication and exact three-row packet before weights."""
import argparse
import hashlib
import importlib
import json
import math
import struct
import os
import sys
from pathlib import Path
import re
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,PATH,KEYS,ContractError,atomic,declaration,require,sha,span,verify


def index(client,api,owner,pin):
    def request(kind):
        obj=kind();obj.user_name=owner;obj.kernel_slug=pin['kernel_slug'];obj.version_label=pin['version_label']
        require(json.loads(str(obj)).get('versionLabel')==pin['version_label'],'explicit SDK version serialization lost')
        return obj
    try:
        returned=client.get_kernel(request(api.ApiGetKernelRequest));body=json.loads(str(returned))['blob']['source']
        require(hashlib.sha256(body.encode()).hexdigest()==pin['submitted_script_sha256']and pin['source']in body,'exact retained submitted source differs')
        status=client.get_kernel_session_status(request(api.ApiGetKernelSessionStatusRequest))
        require(str(status.status).split('.')[-1]==pin['status'],'exact retained success/failure state differs')
        files={};token=None;pages=0
        while True:
            obj=request(api.ApiListKernelSessionOutputRequest);obj.page_size=200
            if token:obj.page_token=token
            page=client.list_kernel_session_output(obj);pages+=1;require(pages<=100,'output index exceeds bound')
            for item in page.files or []:
                require(item.file_name not in files,'duplicate retained output member');files[item.file_name]=item.url
            token=page.next_page_token
            if not token:break
    except ContractError:raise
    except Exception:raise ValueError('Exact-version retained service request rejected')from None
    return files,dict(source=pin['source'],kernel_slug=pin['kernel_slug'],version_label=pin['version_label'],
        status=pin['status'],submitted_script_sha256=pin['submitted_script_sha256'],pages=pages,
        explicit_Get_Status_List_version=True)


def transfer(url,path,pin):
    require(url.startswith('https://')and type(pin['bytes'])is int and 0<pin['bytes']<=1024**3 and
            re.fullmatch('[0-9a-f]{64}',pin['sha256']),'bounded authenticated retained transfer required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
    try:
        with urllib.request.urlopen(url,timeout=180)as response,path.open('xb')as output:
            length=response.headers.get('Content-Length');require(length is None or int(length)==pin['bytes'],'declared extent differs')
            while data:=response.read(2**20):
                n+=len(data);require(n<=pin['bytes'],'retained extent exceeded');h.update(data);output.write(data)
    except Exception:raise ValueError('Retained transfer failed; task-owned partial preserved: '+path.name)from None
    require(n==pin['bytes']and h.hexdigest()==pin['sha256'],'retained whole SHA/extent differs: '+path.name)


def tokens(row):
    return dict(ids=row['ids'],supplied_mask=list(map(int,row['supplied_mask'])),effective_mask=list(map(int,row['effective_mask'])),
                scoring_mask=list(map(int,row['scoring_mask'])),expansion_positions=[])


def packet(original_root,native_root,out,cfg):
    """Pure stdlib: authenticate complete frozen streams, then extract only3 spans."""
    for side,root in (('original',original_root),('native',native_root)):
        for name,pin in cfg[side]['members'].items():
            p=root/name;require(p.is_file()and not p.is_symlink()and p.stat().st_size==pin['bytes']and sha(p)==pin['sha256'],'frozen whole input differs: '+side+'/'+name)
    original=json.loads((original_root/'rows.json').read_text());data=json.loads((original_root/'data.json').read_text())
    require(len(original)==5483 and len(data['documents'])==5183 and len(data['queries'])==300,'full frozen row coverage differs')
    require(len({r['key']for r in original})==5483 and len({r['id']for r in data['documents']})==5183,'duplicate frozen IDs')
    old={r['key']:r for r in original if r['key']in KEYS};texts={'d:'+r['id']:r['text']for r in data['documents']if'd:'+r['id']in KEYS}
    current={}
    with (native_root/'rows.jsonl').open()as f:
        count=0
        for line in f:
            row=json.loads(line);count+=1
            if row['key']in KEYS:
                require(row['key']not in current,'duplicate native selected row');current[row['key']]=row
    require(count==5483 and set(old)==set(texts)==set(current)==set(KEYS),'exact three selected keys missing')
    cases=[]
    for key in KEYS:
        a,b=old[key],current[key];text=texts[key];ids=a['ids'];n=len(ids);selected=sum(a['scoring_mask'])
        require(a['role']=='document'and a['supplied_ids']==ids and 3<=n<=512 and ids[:2]==[2,30001]and
                b['completed']is True and b['typed_exact']is True and b['role']=='document','frozen document contract differs')
        require(all(b[k]==v for k,v in tokens(a).items()),'original/native IDs or masks differ')
        require(isinstance(text,str)and '\0'not in text and len(text.encode())<=65536,'raw document bounds differ')
        require([n,128]==cfg['failing_shapes'][key]and selected>0,'frozen failed-row shape differs')
        case=dict(key=key,id='doc-'+key[2:],role='document',text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  **tokens(a),anchors={})
        for side,row,root in (('original',a,original_root),('native',b,native_root)):
            case['anchors'][side]={}
            for kind,length in (('all',n),('selected',selected)):
                count=row[kind+'_bytes'];offset=row[kind+'_offset'];require(count==length*128*4,'frozen row extent differs')
                block=span(root/(kind+'.f32'),offset,count);digest=hashlib.sha256(block).hexdigest()
                require(digest==row[kind+'_sha256']and all(math.isfinite(v[0])for v in struct.iter_unpack('<f',block)),'frozen row span SHA/finite policy differs')
                path=out/'anchors'/side/(case['id']+'-'+kind+'.f32');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(block)
                case['anchors'][side][kind]=dict(path=str(path.relative_to(out)),shape=[length,128],bytes=count,sha256=digest,
                    retained_file=kind+'.f32',retained_offset=offset,retained_whole_sha256=cfg[side]['members'][kind+'.f32']['sha256'])
        cases.append(case)
    value=dict(schema='crispembed-moedern-three-doc-input-v1',completed=True,keys=KEYS,cases=cases,source_protocol_sha256=sha(PATH),
        original_service=cfg['original']['service'],native_service=cfg['native']['service'],
        all_raw_inputs_authenticated_before_framework=True,full_original_recapture=False,publication_qualified=False)
    atomic(out/'input-packet.json',value);return value


def run(args):
    cfg=verify(args.reviewed_source)
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('GITHUB_SHA')==args.reviewed_source,'genuine GHA CPU source required')
    require(os.environ.get('KAGGLE_API_TOKEN')and os.environ.get('KAGGLE_ACCOUNT'),'existing Kaggle workflow secrets absent; no fetch')
    args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,source=args.reviewed_source,framework_imported=False,models_accessed=False,service={})
    try:
        ram=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
        import shutil
        require(ram>=14*2**30 and shutil.disk_usage(args.scratch).free>=24*2**30 and os.cpu_count()>=2,'14GiB available RAM/24GiB free disk/two CPUs required before fetch')
        require(os.environ.get('GH_TOKEN')and os.environ.get('GITHUB_RUN_ID'),'existing GHA service token/run identity required')
        req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/runs/'+os.environ['GITHUB_RUN_ID'],
            headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
        try:
            with urllib.request.urlopen(req,timeout=60)as response:
                body=response.read(2*2**20+1);require(len(body)<=2*2**20,'GHA service metadata exceeds bound');service=json.loads(body)
        except Exception:raise ValueError('Genuine GHA service admission rejected')from None
        require(service['head_sha']==args.reviewed_source and service['status']=='in_progress'and service['event']=='workflow_dispatch'and
                service['path']=='.github/workflows/moedern-three-doc-v1.yml'and service['id']==int(os.environ['GITHUB_RUN_ID']),
                'GHA source/workflow service identity differs')
        report['gha_service']=dict(run=service['id'],source=service['head_sha'],attempt=service['run_attempt'],workflow=service['path'])
        api=importlib.import_module('kaggle.api.kaggle_api_extended');client=api.KaggleApi()
        try:client.authenticate()
        except Exception:raise ValueError('Existing Kaggle workflow credentials rejected')from None
        with client.build_kaggle_client()as service:
            provider=service.kernels.kernels_api_client
            for side in ('original','native'):
                pin=cfg[side];urls,proof=index(provider,api,os.environ['KAGGLE_ACCOUNT'],pin['service']);report['service'][side]=proof
                folder=args.scratch/side;folder.mkdir()
                transfer(urls[pin['report_member']],folder/'report.json',pin['report_identity'])
                for name,identity in pin['members'].items():
                    member=pin['service']['root_member']+'/'+name;require(member in urls,'frozen retained member absent: '+member)
                    transfer(urls[member],folder/name,identity)
        packet(args.scratch/'original',args.scratch/'native',args.out,cfg)
        provenance=json.loads((ROOT/cfg['original']['public_report']).read_text())['data']
        for name,pin in provenance['sources'].items():
            if name.endswith('.md'):transfer(pin['url'],args.out/'notices'/name,pin)
        atomic(args.out/'dataset-provenance.json',provenance)
        report.update(completed=True,input_packet_sha256=sha(args.out/'input-packet.json'));return report
    except Exception as error:report['error']=type(error).__name__+': '+str(error);raise
    finally:atomic(args.out/'prepare.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for n in ('out','scratch'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
