#!/usr/bin/env python3
"""Closed corrected-mixed full177 comparison; no inference or model qualification."""
import argparse
import hashlib
import json
import os
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
from lighton4b_smoke_compare import ROOT,api,digest,require
sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_bf16_quant as compare
import lighton4b_native_document as native_common
SOURCE='189fc91bb1cc3a49ec918c804c30d072135546af'
RUN='42109519607'
RECIPE='lighton4b-q4k-mixers-q8-v1'
WEIGHTS='d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474'
MODEL={'run':42006450732,'source':'e2beb641773c649394bd4e36526ff50c665f84cb','artifact':13405370923,'archive_bytes':3830524370,'archive_sha256':'6d7e72a423ae0e0b1ab5fd331bad240174c2b29421fe7b585130868cb8a0ee10','receipt_sha256':'87cb309fd4bb3f9d988042bf66b9dc4fe10e6980f1e2248741404043a9096977','file':'lightonocr3-4b-q4k-mixers-q8-v1.gguf','bytes':4683041248,'sha256':WEIGHTS}
FROZEN_PROTOCOL='4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc'
VISION_GATES=tuple('CRISPEMBED_LIGHTON4B_'+x for x in ('VISION_BF16','VISION_BF16_NORM','VISION_BF16_FMA_NORM','VISION_BF16_LINEAR','VISION_BF16_CONV','VISION_BF16_ATTENTION','VISION_BF16_SOFTMAX_AVX2','VISION_BF16_GELU_AVX2','VISION_BF16_MERGER_ERF_AVX2','VISION_BF16_ROTARY_TABLE'))
DISABLED=tuple('CRISPEMBED_'+x for x in ('LIGHTON4B_BF16_LINEAR','LIGHTON4B_VISION_BF16_ROTARY_AVX2','LIGHTON4B_DIRECT_QUANT','QWEN35_WEIGHT_CACHE','QWEN35_PRECISE_RMS','QWEN35_IMATRIX'))

def normalize(report,p):
    require(report['schema']=='lighton4b-corrected-mixed-native-smoke/v1'and report['status']=='measured'and report['full_completion_executed']is True,'actual full mixed producer closure required')
    require(report['source_sha']==SOURCE and report['recipe']==RECIPE and report['model']==MODEL,'fixed source/recipe/weights differ')
    require(report['requested_teacher_positions']==177 and report['requested_generation_cap']==256 and report['native_summary']['teacher_positions']==177 and report['native_summary']['cap']==256,'fixed177/cap256 differs')
    require(report['full_teacher_score_comparison_executed']is False and report['quality_qualified']is False and report['original_forward_executed']is False,'producer/comparer scope differs')
    require(set(report['vision_gates'])==set(VISION_GATES)and report['disabled_gates']==dict.fromkeys(DISABLED,'0'),'corrected gate contract differs')
    require(report['enabled_gates']==dict.fromkeys(tuple(native_common.GATES)+VISION_GATES+('CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4',),'1'),'enabled native gates differ')
    source=report['original_bf16'];require(source['capture_sha256']==p['original_capture_sha256']and source['first_score_sha256']==p['score_records'][0]['sha256']and source['first_id']==p['original_generated_ids'][0]and source['stage_manifest_sha256']=='036a936cdbd360208b5cf54c2604d671b9a919f1c8777bc7cf97b2585b010a78','original reference binding differs')
    require(report['rotary_table_sha256']=='9f52f11eff527011693a3dcae1cafe9cfbff85c7c146453f8c41b54d90ed6d0e'and report['complete_rotary_profile_sha256']=='2700720dee65b9ed85c6f7d0c270347fe72f0f21d0d692cd01449641be5ac293','corrected rotary provenance differs')
    wanted=['processor.patches','vision.patch','vision.position','vision.embedded']+[f'vision.layer.{i}'for i in range(24)]+['vision.merged']
    require(set(report['raw_vision_controls'])==set(wanted),'all29 source vision controls required')
    for name,row in report['raw_vision_controls'].items():
        artifact=report['artifacts'][name+'.f32'];require(row['bit_exact']is True and row['source_sha256']==row['native_sha256']==artifact['sha256']and row['bytes']==artifact['bytes'],'source vision closure differs')
    require(set(report['prepared_controls'])=={'input_ids','image_grid_thw','mm_token_type_ids','pixel_values','attention_mask'},'all five prepared controls required')
    for row in report['prepared_controls'].values():require(row['exact']is True and row['canonical_npy_sha256']==row['source_npy_sha256'],'prepared control differs')
    scores=[]
    for i in range(177):
        name=f'teacher.{i}.logits.f32';row=report['artifacts'][name];require(row['bytes']==993280,'complete full vocabulary geometry differs');scores.append({'position':i,'shape':[1,248320],'bytes':993280,'file':name,'sha256':row['sha256']})
    ids=report['decoded']['ids'];eos=bool(ids and ids[-1]==248046)
    require(report['decoded']['native_eos']==report['native_summary']['eos']==eos,'independent EOS summaries differ')
    return {'schema':'lighton4b-bf16-quant-native/v1','precision':RECIPE,'original_capture_sha256':p['original_capture_sha256'],'protocol_sha256':compare.sha(compare.PROTOCOL),'teacher_forced_ids':p['original_generated_ids'],'free_generation_independent':True,'generated_ids':ids,'decoded':report['decoded']['text'],'stop':'eos'if eos else'max_tokens','cap_hit':not eos and len(ids)==256,'scores':scores,'native_forward_executed':True,'producer_source_sha':SOURCE,'producer_run_url':'https://github.com/CrispStrobe/CrispEmbed/actions/runs/'+RUN,'weights_sha256':WEIGHTS,'normalization':'comparer-derived from authenticated corrected-mixed-full.json and raw artifact; not producer-authored native.json'}


def main():
    parser=argparse.ArgumentParser()
    for name in ('artifact-id','artifact-sha256','receipt-sha256','normalized-receipt-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();args.native_run=RUN;args.native_source=SOURCE
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable comparison required')
    evidence={'schema':'lighton4b-corrected-mixed-bf16-full-replay/v1','status':'blocked','stage':'authenticate-admission','native_forward_executed_by_comparer':False,'original_forward_executed':False,'quality_qualified':False,'held_out_calibration':False,'exploratory_observed_form_only':True,'worker_timeout_seconds':1200,'comparer_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    evidence['requested_native']={'run':RUN,'source':SOURCE,'artifact_id':args.artifact_id,'artifact_sha256':args.artifact_sha256,'producer_report_sha256':args.receipt_sha256,'normalized_receipt_sha256':args.normalized_receipt_sha256,'weights_sha256':WEIGHTS}
    def run(label,command,env=None):
        with(out/(label+'.log')).open('w')as log:subprocess.run([str(x)for x in command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        def timeout(signum,frame):raise TimeoutError('bounded comparison deadline')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
        mem={k:int(v.strip().split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())}
        evidence['preflight']={'load_average':os.getloadavg(),'available_memory_bytes':mem['MemAvailable'],'swap_total_bytes':mem['SwapTotal'],'swap_free_bytes':mem['SwapFree'],'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
        require(mem['MemAvailable']>=2*2**30 and shutil.disk_usage(scratch).free>=4*2**30 and shutil.disk_usage(out).free>=256*2**20,'bounded comparison capacity unavailable')
        run('admission-controls',[sys.executable,ROOT/'tests/test_lighton4b_corrected_mixed_full_compare.py'])
        protocol=compare.protocol()
        require(compare.sha(compare.PROTOCOL)==FROZEN_PROTOCOL,'frozen177-row protocol differs')
        identity={'capture_sha256':args.normalized_receipt_sha256,'source_sha':SOURCE,'run_url':'https://github.com/CrispStrobe/CrispEmbed/actions/runs/'+RUN,'weights_sha256':WEIGHTS,'precision':RECIPE}
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
            report_bytes=member('corrected-mixed-full.json',2**20);require(digest(report_bytes)==args.receipt_sha256,'independently reviewed producer report differs');report=json.loads(report_bytes)
            (out/'producer-report.json').write_bytes(report_bytes)
            native=normalize(report,protocol);native_bytes=(json.dumps(native,indent=2,sort_keys=True)+'\n').encode()
            require(digest(native_bytes)==args.normalized_receipt_sha256,'independently reviewed normalized receipt differs')
            (native_root/'native.json').write_bytes(native_bytes);(out/'normalized-native.json').write_bytes(native_bytes)
            for name,row in report['raw_vision_controls'].items():
                require(0<row['bytes']<=8*2**20,'source vision member bound')
                data=member('mixed/'+name+'.f32',row['bytes']);require(len(data)==row['bytes']and digest(data)==row['source_sha256'],'actual raw vision source byte control differs')
            for row in native['scores']:
                data=member('mixed/'+row['file'],993280);require(len(data)==993280 and digest(data)==row['sha256'],'native score payload differs');(native_root/row['file']).write_bytes(data)
            for name,expected in [('generated.ids',(''.join(str(i)+'\n'for i in native['generated_ids'])).encode()),('generated.txt',native['decoded'].encode())]:
                data=member('mixed/'+name,2**20);record=report['artifacts'][name];require(len(data)==record['bytes']and digest(data)==record['sha256'],'independent decoded raw artifact differs')
                if name=='generated.ids':require([int(x)for x in data.decode().split()]==native['generated_ids'],'independent raw IDs differ')
                else:require(data==expected,'independent raw decoded text differs')
        evidence['stage']='authenticate-original-version-v1'
        kaggle=KaggleApi();kaggle.authenticate();Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
        owner=os.environ['KAGGLE_ACCOUNT'];slug='lightonocr-4b-bf16-cpu-capture-v1';urls={};page_token=None;pages=0
        with kaggle.build_kaggle_client()as client:
            request=Get();request.user_name=owner;request.kernel_slug=slug;request.version_label='v1'
            metadata=json.loads(str(client.kernels.kernels_api_client.get_kernel(request)));require(digest(metadata['blob']['source'].encode())=='c370b097a45c2d8158343fa1a3a3c510f9bea3b84912565e896b5ded37fd93f6','original submitted bootstrap differs')
            while True:
                pages+=1;require(pages<=16,'original listing page bound')
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
        actual_original=compare.original(original_root,protocol)
        require(all(report['prepared_controls'][k]['source_npy_sha256']==actual_original['prepared_inputs'][k]['sha256']for k in report['prepared_controls']),'original prepared-input receipt differs')
        evidence['stage']='build-actual-ref';build=scratch/'full-diff-build'
        run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF'])
        run('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
        control_env=os.environ.copy();control_env['LIGHTON4B_DOCUMENT_DIFF']=str(build/'crispembed-diff-lighton4b-document')
        run('actual-ref-controls',[sys.executable,ROOT/'tests/test_lighton4b_bf16_quant.py'],control_env)
        evidence['actual_ref_equal_changed_tampered_controls_passed']=True
        evidence['stage']='compare177-original-prefix-rows-and-independent-generation'
        admitted=dict(protocol);admitted['precisions']=protocol['precisions']+[RECIPE];admitted['native_producers']=protocol['native_producers']+[identity]
        compare.candidate(native,actual_original,admitted,False,args.normalized_receipt_sha256)
        code=compare.compare(original_root,native_root,build/'crispembed-diff-lighton4b-document',out/'comparison',admitted_protocol=admitted)
        evidence.update(status='compared'if code in(0,1)else'blocked',comparison_returncode=code,native_admission=identity,producer_report_sha256=args.receipt_sha256,normalized_receipt_sha256=args.normalized_receipt_sha256,normalized_receipt_origin='comparer-derived, not producer-authored',native_artifact={key:artifact[key]for key in('id','digest','size_in_bytes','expires_at')},original_capture_sha256=protocol['original_capture_sha256'],original_provider={'service':'Kaggle','kernel_slug':slug,'version_label':'v1'},protocol_sha256=compare.sha(compare.PROTOCOL))
        return code
    except Exception as error:
        message=str(error)if isinstance(error,ValueError)else'hosted capture retrieval or comparison failed'
        for value,label in ((ROOT,'[repository]'),(scratch,'[scratch]'),(out,'[report]'),(os.environ.get('KAGGLE_ACCOUNT',''),'[account]')):
            if str(value):message=message.replace(str(value),label)
        evidence.update(error_type=type(error).__name__,error=message)
        code=getattr(error,'code',None)or getattr(getattr(error,'response',None),'status_code',None)
        if isinstance(code,int):evidence['http_status']=code
        return 2
    finally:
        signal.alarm(0)
        (out/'corrected-mixed-full-replay.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
