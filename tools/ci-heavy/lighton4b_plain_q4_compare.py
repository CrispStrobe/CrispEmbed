#!/usr/bin/env python3
"""Authenticate/archive full Q4 captures and compare shared causal rows; no inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile
from lighton4b_mixed_step33_compare import ROOT,api,read_url,fetch,require,sha,tiny_archive
SOURCE='bd87593040c9a31cf5ca6ec2fd2369de6f34b995'
SOURCE_SHA='1096b5f0f6ce26f0fa26ff9cc86dd606fbb3da24a3feca06768f2f8cd66d6d1b'
BOOTSTRAP='b94074a5eb158a4ea6a861b0571f7bc2948f9dab2c7a61edcc4cf16bbcf68602'
NATIVE_RUN=42183620520
NATIVE_SOURCE='4443861693fc7bc4afeaa85f97e4cb2820250610'
NATIVE_SHA='c726b09da906562a7f8cdd20a696d7e7d634745f7b329400f056fc4275e45d4e'
ARTIFACT=13488399938
ARCHIVE_SHA='e896e23af456ad3bc1a1a482aa2420f5dfbfc93fafbf7def27a76c80a0f7a61e'

def shared_positions(a,b):
    require(a and b and all(type(i)is int and 0<=i<248320 for i in a+b),'complete finite-vocabulary independent IDs required')
    first=next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),min(len(a),len(b)))
    return list(range(min(first+1,len(a),len(b))))

def provider():
    from kaggle.api.kaggle_api_extended import KaggleApi
    service=KaggleApi();service.authenticate();Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest'];urls={};token=None
    with service.build_kaggle_client()as client:
        req=Get();req.user_name=os.environ['KAGGLE_ACCOUNT'];req.kernel_slug='lighton4b-bf16-plain-q4-free-generation-v1';req.version_label='v4'
        metadata=json.loads(str(client.kernels.kernels_api_client.get_kernel(req)));require(sha(metadata['blob']['source'].encode())==BOOTSTRAP,'version4 source bootstrap differs')
        for _ in range(16):
            req=List();req.user_name=os.environ['KAGGLE_ACCOUNT'];req.kernel_slug='lighton4b-bf16-plain-q4-free-generation-v1';req.version_label='v4';req.page_size=100
            if token:req.page_token=token
            page=client.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in page.files or []:require(item.file_name not in urls,'duplicate source member');urls[item.file_name]=item.url
            token=page.next_page_token
            if not token:break
        require(not token,'source listing bound exceeded')
    return urls

def main():
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable read-only scope required')
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    report={'schema':'lighton4b-plain-q4-causal-comparison/v1','status':'blocked','source_sha':SOURCE,'source_receipt_sha256':SOURCE_SHA,'native_run':NATIVE_RUN,'native_source':NATIVE_SOURCE,'native_receipt_sha256':NATIVE_SHA,'inference_executed':False,'quality_qualified':False,'held_out_calibration':False,'source_stages_captured':False,'rows':[],'consumer_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def save():(out/'plain-q4-comparison.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    def run(label,cmd):
        with(out/(label+'.log')).open('w')as log:subprocess.run([str(x)for x in cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        def deadline(*args):raise TimeoutError('bounded read-only comparison deadline')
        signal.signal(signal.SIGALRM,deadline);signal.alarm(1200)
        report['stage']='preflight';save();mem={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())}
        report['preflight']={'load_average':os.getloadavg(),'available_memory_bytes':mem['MemAvailable'],'swap_total_bytes':mem['SwapTotal'],'swap_free_bytes':mem['SwapFree'],'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
        require(mem['MemAvailable']>=2*2**30 and shutil.disk_usage(scratch).free>=4*2**30 and shutil.disk_usage(out).free>=2**30,'read-only capacity unavailable')
        import numpy as np
        require(np.__version__=='2.3.5','frozen serializer differs')
        run('causal-admission-controls',[sys.executable,ROOT/'tests/test_lighton4b_plain_q4_compare.py'])
        report['stage']='authenticate-source';save();urls=provider();prefix='bf16-plain-q4-freegen/quant-oracle/'
        raw=read_url(urls[prefix+'oracle.json'],2**20);require(sha(raw)==SOURCE_SHA,'reviewed complete source report differs');source=json.loads(raw);(out/'source-report.json').write_bytes(raw)
        require(source['source_sha']==SOURCE and source['comparison_completed']and source['full_generation_completed']and not source.get('error')and not source['f32_model_instantiated'],'authentic genuine BF16 completion required')
        require(source['reference_profile']=='frozen-Intel-AVX2'and source['cpu_profile_key']=='GenuineIntel/6/79'and source['original_first_score_bit_exact']and source['plain_q4_first_score_bit_exact']and not source['frozen_reference_substituted'],'frozen Intel source profile/anchors differ')
        arm=source['arms']['q4_k'];require(len(arm['ids'])==len(arm['scores'])==256 and not arm['eos_completed']and arm['cap_hit']and arm['forward_counts']=={'root':256,'text':256,'vision':1}and source['total_model_forward_calls']==257,'source independent cap/forward counts differ')
        require(len(source['plain_q4_final_bf16_parameter_sha256'])==249 and len(source['original_bf16_tensor_sha256'])==723 and source['q4_k-weights']['protected_tensors']==474 and source['q4_k-weights']['tied_head_preserved'],'source full parameter/protected/tie census differs')
        audit=read_url(urls['bf16-plain-q4-freegen/original-adapter-audit/capture.json'],2**20);require(sha(audit)==source['original_adapter_audit_sha256'],'source final actual audit differs');ad=json.loads(audit);require(ad['status']=='captured'and ad['model_forward_calls']==257 and ad['checkpoint']['payload_verified']and ad['checkpoint']['sha256']=='6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67','actual audit/checkpoint differs');(out/'source-audit.json').write_bytes(audit)
        for name in ('lighton4b-bf16-plain-q4-freegen-bootstrap.json','contract-controls.log','mixed-contract-controls.log','full-contract-controls.log','plain-q4-contract-controls.log','amd-diagnostic-controls.log','dual-profile-controls.log'):
            require(name in urls,'source bootstrap/control member missing');(out/name).write_bytes(read_url(urls[name],2**20))
        artifact=api('actions/artifacts/'+str(ARTIFACT));require(not artifact['expired']and artifact['workflow_run']['id']==NATIVE_RUN and artifact['digest']=='sha256:'+ARCHIVE_SHA and artifact['size_in_bytes']==181791946,'native archive service identity differs')
        archive=scratch/'native.zip'
        with archive.open('xb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(ARTIFACT)+'/zip'],stdout=f,check=True,timeout=600)
        with archive.open('rb')as f:require(hashlib.file_digest(f,'sha256').hexdigest()==ARCHIVE_SHA,'complete native archive digest differs')
        build=scratch/'ref-build';binary=build/'crispembed-diff-lighton4b-document'
        run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF']);run('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
        def compare(a,b,label):
            require(len(a)==len(b)and len(a)%4==0,'paired score geometry differs')
            for name,data in [('source',a),('native',b)]:tiny_archive(scratch/(name+'.gguf'),'stage',[1,len(data)//4],data)
            (scratch/'order.tsv').write_text(f'stage float {len(a)//4} 1 1 1\n')
            result=subprocess.run([str(binary),str(scratch/'source.gguf'),str(scratch/'native.gguf'),str(scratch/'order.tsv')],capture_output=True,text=True,timeout=60);stages=[line for line in result.stdout.splitlines()if line.startswith('STAGE ')];require(result.returncode in(0,1)and len(stages)==1,'actual Ref comparison incomplete');(out/(label+'.log')).write_text(result.stdout+result.stderr)
            aa=np.frombuffer(a,dtype='<f4').astype(np.float64);bb=np.frombuffer(b,dtype='<f4').astype(np.float64);require(np.isfinite(aa).all()and np.isfinite(bb).all(),'finite retained scores required');e=bb-aa
            return {'returncode':result.returncode,'stage':stages[0],'max_abs':float(np.abs(e).max()),'rms':float(np.sqrt(np.mean(e*e))),'cosine':float(aa@bb/(np.linalg.norm(aa)*np.linalg.norm(bb)))}
        control=np.ones(2560,dtype='<f4');require(compare(control.tobytes(),control.tobytes(),'control-equal')['returncode']==0,'Ref equal control failed');changed=control.copy();changed[0]+=1;require(compare(control.tobytes(),changed.tobytes(),'control-changed')['returncode']==1,'Ref changed control failed');report['actual_ref_controls_passed']=True
        with zipfile.ZipFile(archive)as z:
            names=z.namelist();require(len(names)==len(set(names)),'duplicate native ZIP entry')
            def member(suffix,size):
                hits=[n for n in names if n==suffix or n.endswith('/'+suffix)];require(len(hits)==1 and z.getinfo(hits[0]).file_size<=size,'ambiguous/missing native member');return z.read(hits[0])
            raw=member('corrected-plain-q4-head-full.json',2**20);require(sha(raw)==NATIVE_SHA,'reviewed native receipt differs');native=json.loads(raw);(out/'native-report.json').write_bytes(raw)
            require(native['status']=='measured'and native['source_sha']==NATIVE_SOURCE and native['full_completion_executed']and native['model']['sha256']=='f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91','native same immutable Q4 completion differs')
            require(len(native['decoded']['ids'])==256 and native['decoded']['cap_hit']and not native['decoded']['native_eos'],'independent native actual closure differs')
            require(len(native['prepared_controls'])==5 and all(v['exact']and v['canonical_npy_sha256']==v['source_npy_sha256']==source['prepared_inputs'][k]['sha256']for k,v in native['prepared_controls'].items()),'same prepared source/native inputs differ')
            require(len(native['raw_vision_controls'])==29 and all(v['bit_exact']and v['native_sha256']==v['source_sha256']for v in native['raw_vision_controls'].values()),'frozen native vision prerequisite differs')
            positions=shared_positions(arm['ids'],native['decoded']['ids']);require(positions==list(range(172)),'reviewed shared causal window differs');report['same_causal_score_positions']=positions
            source_out=out/'source256';source_out.mkdir();native_out=out/'native-causal172';native_out.mkdir();total=0
            report['stage']='preserve-source256-and-compare-causal172';save()
            for i,row in enumerate(arm['scores']):
                require(row['position']==i and row['bytes']==993280 and row['shape']==[1,248320]and row['file']==f'q4_k.scores-{i:03d}.f32','full source score geometry/order differs');data=fetch(urls,prefix+row['file'],row);(source_out/row['file']).write_bytes(data);total+=len(data)
                if i in positions:
                    require(arm['ids'][:i]==native['decoded']['ids'][:i],'actual causal context differs');name=f'greedy.{i}.logits.f32';record=native['artifacts'][name];other=member('q4_k/'+name,993280);require(len(other)==record['bytes']==993280 and sha(other)==record['sha256'],'native actual causal score differs');(native_out/name).write_bytes(other)
                    aa=np.frombuffer(data,dtype='<f4');bb=np.frombuffer(other,dtype='<f4');require(int(aa.argmax())==arm['ids'][i]and int(bb.argmax())==native['decoded']['ids'][i],'actual independent greedy selections differ')
                    ids=sorted(set([arm['ids'][i],native['decoded']['ids'][i]]));report['rows'].append({'position':i,'same_previous_ids':True,'source_sha256':row['sha256'],'native_sha256':record['sha256'],'source_id':arm['ids'][i],'native_id':native['decoded']['ids'][i],'selected_logits':{str(k):{'source':float(aa[k]),'native':float(bb[k])}for k in ids},'raw':compare(data,other,f'causal-{i:03d}')})
                if i%16==0:report['source_rows_archived']=i+1;save()
            for key in ('ids_file','decoded_file'):
                row=arm[key];(source_out/row['file']).write_bytes(fetch(urls,prefix+row['file'],row))
            require(total==254279680 and len(report['rows'])==172,'complete preservation/comparison census differs')
        report.update(status='compared',source_rows_archived=256,source_score_bytes=total,actual_ref_binary_sha256=sha(binary.read_bytes()),raw_numerical_passed=all(r['raw']['returncode']==0 for r in report['rows']),native_artifact={k:artifact[k]for k in ('id','digest','size_in_bytes','expires_at')},no_hidden_stage_comparison='No source/native row171 hidden/operator capture exists; logits only. Source BF16 versus native F32-dequant weights remain distinguished.')
        return 0 if report['raw_numerical_passed']else 1
    except Exception as e:
        report.update(error_type=type(e).__name__,error=str(e)[:240]if isinstance(e,ValueError)else'bounded authenticated read-only Q4 comparison failed');return 2
    finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
