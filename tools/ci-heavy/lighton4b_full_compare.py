#!/usr/bin/env python3
"""Hosted frozen BF16 versus admitted actual177-position Q8/Q4 comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
from lighton4b_smoke_compare import ROOT,api,digest,require
sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_bf16_quant as compare

def main():
    parser=argparse.ArgumentParser()
    for name in ('native-run','native-source','artifact-id','artifact-sha256','receipt-sha256','native-receipt-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--recipe',choices=('q8_0','q4_k'),required=True);args=parser.parse_args()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable comparison required')
    evidence={'schema':'lighton4b-bf16-native-full-replay/v1','status':'blocked','stage':'authenticate-admission','native_forward_executed_by_comparer':False,'original_forward_executed':False,'quality_qualified':False}
    def run(label,command):
        with(out/(label+'.log')).open('w')as log:subprocess.run([str(x)for x in command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        protocol=compare.protocol()
        weights={'q8_0':'5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1','q4_k':'f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'}
        identity={'capture_sha256':args.native_receipt_sha256,'source_sha':args.native_source,'run_url':'https://github.com/CrispStrobe/CrispEmbed/actions/runs/'+args.native_run,'weights_sha256':weights[args.recipe],'precision':args.recipe}
        require(identity in protocol['native_producers'],'actual native receipt tuple is not independently admitted')
        run('dependencies',[sys.executable,'-m','pip','install','kaggle==2.2.5','six==1.17.0'])
        from kaggle.api.kaggle_api_extended import KaggleApi
        evidence['stage']='authenticate-native-artifact'
        service=api('actions/runs/'+args.native_run);require((service['head_sha'],service['conclusion'],service['run_attempt'])==(args.native_source,'success',1),'native service identity differs')
        artifact=api('actions/artifacts/'+args.artifact_id)
        require(artifact['workflow_run']['id']==int(args.native_run)and not artifact['expired']and artifact['digest']=='sha256:'+args.artifact_sha256 and 0<artifact['size_in_bytes']<=2*1024**3,'native artifact identity/bound differs')
        archive=scratch/'native-full.zip'
        with archive.open('xb')as stream:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+args.artifact_id+'/zip'],stdout=stream,check=True,timeout=600)
        with archive.open('rb')as stream:archive_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        require(archive_sha==args.artifact_sha256,'native archive digest differs')
        original_root=scratch/'original';native_root=scratch/'native';original_root.mkdir();native_root.mkdir()
        with zipfile.ZipFile(archive)as z:
            names=z.namelist();require(len(names)==len(set(names)),'duplicate native ZIP names')
            def member(suffix,limit):
                matches=[name for name in names if name==suffix or name.endswith('/'+suffix)]
                require(len(matches)==1 and z.getinfo(matches[0]).file_size<=limit,'native member missing/ambiguous/oversized')
                return z.read(matches[0])
            report_bytes=member('native-document.json',2**20);require(digest(report_bytes)==args.receipt_sha256,'reviewed native report differs');report=json.loads(report_bytes)
            require(report['status']=='completed'and report['mode']=='full'and report['recipe']==args.recipe and report['source_sha']==args.native_source and report['model']['sha256']==weights[args.recipe]and report['native']['teacher_positions']==177,'native full capture closure differs')
            native_bytes=member(args.recipe+'/native.json',2**20);require(digest(native_bytes)==args.native_receipt_sha256 and report['native_receipt_sha256']==args.native_receipt_sha256,'admitted native capture differs')
            native=json.loads(native_bytes);(native_root/'native.json').write_bytes(native_bytes)
            require(len(native['scores'])==177,'native score coverage differs')
            for row in native['scores']:
                name=row['file'];require(name==f"teacher.{row['position']}.logits.f32"and row['bytes']==993280,'native score file/geometry differs')
                data=member(args.recipe+'/'+name,993280);require(len(data)==993280 and digest(data)==row['sha256'],'native score payload differs');(native_root/name).write_bytes(data)
        evidence['stage']='authenticate-original-version-v1'
        kaggle=KaggleApi();kaggle.authenticate();Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
        owner=os.environ['KAGGLE_ACCOUNT'];slug='lightonocr-4b-bf16-cpu-capture-v1';urls={};page_token=None
        with kaggle.build_kaggle_client()as client:
            request=Get();request.user_name=owner;request.kernel_slug=slug;request.version_label='v1'
            metadata=json.loads(str(client.kernels.kernels_api_client.get_kernel(request)));require(digest(metadata['blob']['source'].encode())=='c370b097a45c2d8158343fa1a3a3c510f9bea3b84912565e896b5ded37fd93f6','original submitted bootstrap differs')
            while True:
                request=List();request.user_name=owner;request.kernel_slug=slug;request.version_label='v1';request.page_size=100
                if page_token:request.page_token=page_token
                page=client.kernels.kernels_api_client.list_kernel_session_output(request)
                for item in page.files or []:
                    require(item.file_name not in urls,'duplicate original provider member');urls[item.file_name]=item.url
                page_token=page.next_page_token
                if not page_token:break
        evidence['stage']='retrieve-exact-original-capture-and177-scores'
        capture_member='lighton4b-bf16-freegen/capture.json';require(capture_member in urls,'original capture receipt missing')
        with urllib.request.urlopen(urls[capture_member],timeout=90)as response:data=response.read(2**20+1)
        require(len(data)<=2**20 and digest(data)==protocol['original_capture_sha256'],'original receipt bytes differ');(original_root/'capture.json').write_bytes(data)
        for row in protocol['score_records']:
            name=row['file'];require(name==f"scores-{row['position']:03d}.f32"and row['bytes']==993280,'frozen original score geometry differs');key='lighton4b-bf16-freegen/'+name;require(key in urls,'original score member missing')
            with urllib.request.urlopen(urls[key],timeout=90)as response:data=response.read(993281)
            require(len(data)==993280 and digest(data)==row['sha256'],'original full-vocabulary score hash differs');(original_root/name).write_bytes(data)
        evidence['stage']='build-actual-ref';build=scratch/'full-diff-build'
        run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF'])
        run('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
        evidence['stage']='compare177-original-prefix-rows-and-independent-generation'
        code=compare.compare(original_root,native_root,build/'crispembed-diff-lighton4b-document',out/'comparison')
        evidence.update(status='compared'if code in(0,1)else'blocked',comparison_returncode=code,native_admission=identity,native_artifact={key:artifact[key]for key in('id','digest','size_in_bytes','expires_at')},original_capture_sha256=protocol['original_capture_sha256'],original_provider={'service':'Kaggle','kernel_slug':slug,'version_label':'v1'},protocol_sha256=compare.sha(compare.PROTOCOL))
        return code
    except Exception as error:
        message=str(error)if isinstance(error,ValueError)else'hosted capture retrieval or comparison failed'
        for value,label in ((ROOT,'[repository]'),(scratch,'[scratch]'),(out,'[report]'),(os.environ.get('KAGGLE_ACCOUNT',''),'[account]')):
            if str(value):message=message.replace(str(value),label)
        evidence.update(error_type=type(error).__name__,error=message)
        code=getattr(error,'code',None)or getattr(getattr(error,'response',None),'status_code',None)
        if isinstance(code,int):evidence['http_status']=code
        return 2
    finally:(out/'full-replay.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
