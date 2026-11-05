#!/usr/bin/env python3
"""Guarded hosted first-score comparison; never admits a full native producer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from lighton4b_document_v2 import tiny_archive

def require(ok,why):
    if not ok:raise ValueError(why)
def digest(data):return hashlib.sha256(data).hexdigest()
def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))
def main():
    p=argparse.ArgumentParser()
    for name in ('native-run','native-source','artifact-id','artifact-sha256','receipt-sha256'):p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable comparison required')
    report={'schema':'lighton4b-bf16-native-first-score/v1','status':'blocked','native_execution_admitted':False,'full_document_parity_qualified':False,'quality_qualified':False,'original_forward_executed':False,'native_forward_executed_by_comparer':False,'compared_positions':0,'internal_stage_attribution_available':False}
    def run(label,command):
        with(out/(label+'.log')).open('w')as log:subprocess.run([str(x)for x in command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run('dependencies',[sys.executable,'-m','pip','install','kaggle==2.2.5'])
        protocol=json.loads((ROOT/'docs/lighton4b-bf16-quant-v1.json').read_text());pin=protocol['score_records'][0]
        # Exact hosted run and artifact are supplied after independent receipt review.
        report['stage']='authenticate-native-artifact'
        service=api('actions/runs/'+args.native_run)
        require((service['head_sha'],service['conclusion'],service['run_attempt'])==(args.native_source,'success',1),'native producer service differs')
        artifact=api('actions/artifacts/'+args.artifact_id)
        require(artifact['workflow_run']['id']==int(args.native_run)and not artifact['expired']and artifact['digest']=='sha256:'+args.artifact_sha256 and 0<artifact['size_in_bytes']<=512*2**20,'native artifact identity/bound differs')
        archive=scratch/'smoke.zip'
        with archive.open('xb')as stream:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+args.artifact_id+'/zip'],stdout=stream,check=True,timeout=300)
        require(digest(archive.read_bytes())==args.artifact_sha256,'native archive digest differs')
        with zipfile.ZipFile(archive)as z:
            names=z.namelist();require(len(names)==len(set(names)),'duplicate native ZIP members')
            hits=[n for n in names if n.endswith('native-document.json')];require(len(hits)==1,'native receipt missing/ambiguous')
            info=z.getinfo(hits[0]);require(info.file_size<=2**20,'native receipt bound')
            data=z.read(hits[0]);require(digest(data)==args.receipt_sha256,'reviewed native receipt differs');native=json.loads(data)
            require(native['status']=='completed'and native['mode']=='smoke'and native['recipe']=='q8_0'and native['source_sha']==args.native_source,'actual native smoke scope differs')
            require(native['model']['sha256']=='5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1'and native['native']['teacher_positions']==1,'native weights/position scope differs')
            row=native['artifacts']['teacher.0.logits.f32'];require(row['bytes']==993280,'native vocabulary differs')
            hits=[n for n in names if n.endswith('/q8_0/teacher.0.logits.f32')or n=='q8_0/teacher.0.logits.f32'];require(len(hits)==1 and z.getinfo(hits[0]).file_size==993280,'native score member differs')
            native_bytes=z.read(hits[0]);require(digest(native_bytes)==row['sha256'],'native score bytes differ')
        # Read only the1MiB frozen original row from explicitly pinned version1.
        report['stage']='retrieve-authenticated-original-row'
        from kaggle.api.kaggle_api_extended import KaggleApi
        kaggle=KaggleApi();kaggle.authenticate()
        Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
        owner=os.environ['KAGGLE_ACCOUNT'];slug='lightonocr-4b-bf16-cpu-capture-v1';urls={};page_token=None
        with kaggle.build_kaggle_client()as client:
            request=Get();request.user_name=owner;request.kernel_slug=slug;request.version_label='1'
            metadata=json.loads(str(client.kernels.kernels_api_client.get_kernel(request)))
            require(digest(metadata['blob']['source'].encode())=='c370b097a45c2d8158343fa1a3a3c510f9bea3b84912565e896b5ded37fd93f6','original submitted bootstrap source differs')
            while True:
                request=List();request.user_name=owner;request.kernel_slug=slug;request.version_label='1';request.page_size=100
                if page_token:request.page_token=page_token
                page=client.kernels.kernels_api_client.list_kernel_session_output(request)
                for item in page.files or []:
                    require(item.file_name not in urls,'duplicate original output member');urls[item.file_name]=item.url
                page_token=page.next_page_token
                if not page_token:break
        member='lighton4b-bf16-freegen/scores-000.f32';require(member in urls,'frozen original score missing')
        try:
            with urllib.request.urlopen(urls[member],timeout=90)as response:original_bytes=response.read(pin['bytes']+1)
        except Exception:raise ValueError('authenticated bounded original transfer failed')from None
        require(len(original_bytes)==pin['bytes']and digest(original_bytes)==pin['sha256'],'original row hash differs')
        for name,data in [('original',original_bytes),('native',native_bytes)]:tiny_archive(scratch/(name+'.gguf'),'generated.scores.0',[1,248320],data)
        order=scratch/'order.tsv';order.write_text('generated.scores.0 float 248320 1 1 1\n')
        report['stage']='build-actual-ref'
        build=scratch/'smoke-diff-build'
        run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF'])
        run('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
        binary=build/'crispembed-diff-lighton4b-document'
        report['stage']='compare-first-score'
        result=subprocess.run([str(binary),str(scratch/'original.gguf'),str(scratch/'native.gguf'),str(order)],capture_output=True,text=True)
        (out/'crispembed-diff.log').write_text(result.stdout+result.stderr)
        stages=[line for line in result.stdout.splitlines()if line.startswith('STAGE ')];require(len(stages)==1 and result.returncode in(0,1),'complete actualRef first-score result required')
        metrics=dict(item.split('=',1)for item in stages[0].split()[2:]if '='in item)
        report.update(status='compared',numerical_passed=result.returncode==0,compared_positions=1,metrics=metrics,earliest_observed_score_difference=0 if float(metrics['max_abs'])>0 else None,earliest_score_budget_failure=0 if result.returncode else None,original_row_sha256=pin['sha256'],native_row_sha256=row['sha256'],native_run=args.native_run,native_source=args.native_source,native_receipt_sha256=args.receipt_sha256,native_artifact={k:artifact[k]for k in('id','digest','size_in_bytes','expires_at')},original_provider={'service':'Kaggle','kernel_slug':slug,'version_label':'1'},original_first_id=protocol['original_generated_ids'][0],original_decoded=protocol['original_decoded'],native_smoke_decoded=native['decoded'],binary_sha256=digest(binary.read_bytes()))
    except Exception as error:
        # Do not serialize credential, signed service URLs or private account names.
        message=str(error) if isinstance(error,ValueError) else 'hosted source retrieval or comparison failed'
        for value,label in ((ROOT,'[repository]'),(scratch,'[scratch]'),(out,'[report]'),(os.environ.get('KAGGLE_ACCOUNT',''),'[account]')):
            if str(value):message=message.replace(str(value),label)
        report.update(error_type=type(error).__name__,error=message[:1000]);return 1
    finally:(out/'first-score-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
