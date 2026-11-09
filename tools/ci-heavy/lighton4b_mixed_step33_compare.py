#!/usr/bin/env python3
"""Read-only paired actual step33 stages; no inference or replacement arrays/gold."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from lighton4b_bf16_io import tiny_archive
from lighton4b_bf16_mixed_step33 import stage_names
NATIVE_RUN=42168142185
NATIVE_SOURCE='b98883abeccd0cae39e0026571c150e444b71786'
SOURCE='8eeb67bdeb1bf0e55b0d9ffd833a600967615765'
BOOTSTRAP='1dcd5aa80d6418a41f9cc8e70eca118c81a79b743d379532f9dad51f02f95c94'
FULL_SHA='ed4ef1b60c41d4d3b2a131d107c2563ee971c931873202f5f470683ebb9a1d58'
FULL_URL='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/8e19dd865/tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-freegen/oracle.json'
def require(value,message):
    if not value:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))
def read_url(url,limit):
    with urllib.request.urlopen(url,timeout=90)as response:raw=response.read(limit+1)
    require(len(raw)<=limit,'bounded source transfer exceeded');return raw
def provider(slug,bootstrap):
    from kaggle.api.kaggle_api_extended import KaggleApi
    service=KaggleApi();service.authenticate();Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest'];urls={};token=None
    with service.build_kaggle_client()as client:
        req=Get();req.user_name=os.environ['KAGGLE_ACCOUNT'];req.kernel_slug=slug;req.version_label='v1'
        metadata=json.loads(str(client.kernels.kernels_api_client.get_kernel(req)));require(sha(metadata['blob']['source'].encode())==bootstrap,'exact source service version/bootstrap differs')
        for page_index in range(16):
            req=List();req.user_name=os.environ['KAGGLE_ACCOUNT'];req.kernel_slug=slug;req.version_label='v1';req.page_size=100
            if token:req.page_token=token
            page=client.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in page.files or []:require(item.file_name not in urls,'duplicate source member');urls[item.file_name]=item.url
            token=page.next_page_token
            if not token:break
        require(not token,'source listing bound exceeded')
    return urls
def fetch(urls,name,record):
    require(name in urls,'source member missing');raw=read_url(urls[name],record['bytes'])
    require(len(raw)==record['bytes']and sha(raw)==record['sha256'],'source payload identity differs');return raw
def validate_source(s,full):
    require(s['source_sha']==SOURCE and s['activation_trace_completed']and s['comparison_completed']and not s.get('error'),'completed authentic source trace required')
    require(s['trace_observer_all34_scores_bit_exact']and s['trace_observer_all34_ids_exact']and set(s['trace_stages'])==stage_names(),'source observer neutrality/all50 stages required')
    require(s['mixed_final_bf16_parameter_sha256']==full['mixed_final_bf16_parameter_sha256']and s['original_bf16_tensor_sha256']==full['original_bf16_tensor_sha256'],'same effective BF16 parameter identities differ')
    require(len(s['mixed_final_bf16_parameter_sha256'])==249 and len(s['original_bf16_tensor_sha256'])==723,'full source249/723 tensor census differs')
    arm=s['arms']['mixed'];gold=full['arms']['mixed']
    require(arm['ids']==gold['ids'][:34]and len(arm['scores'])==34 and all(a['sha256']==b['sha256']and a['position']==i and a['shape']==[1,248320]and a['bytes']==993280 for i,(a,b)in enumerate(zip(arm['scores'],gold['scores'][:34]))),'all34 retained source scores/IDs differ')
    require(s['total_model_forward_calls']==35 and arm['forward_counts']=={'root':34,'text':34,'vision':1},'source actual cache/forward counts differ')
    require(arm['generation_cap']==34 and arm['cap_hit']and not arm['eos_completed'],'source partial cap/EOS scope differs')
    require(s['trace_input_ids']==[gold['ids'][32]]and s['head_output_matches_generated_score'],'actual source input/head-score boundary differs')
def rne_bf16(values,np):
    bits=values.view(np.uint32);finite=(bits&0x7f800000)!=0x7f800000
    rounded=bits.copy();rounded[finite]=(bits[finite]+np.uint32(0x7fff)+((bits[finite]>>16)&1))&np.uint32(0xffff0000)
    return rounded.view(np.float32)

def main():
    if sys.argv[1:]==['--plain-q4-full']:
        from lighton4b_plain_q4_compare import main as plain_q4_main
        return plain_q4_main()
    p=argparse.ArgumentParser()
    for name in ('artifact-id','artifact-sha256','receipt-sha256','source-receipt-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--preserve-full-source',action='store_true');args=p.parse_args()
    require(args.artifact_id.isdigit()and all(len(v)==64 and set(v)<=set('0123456789abcdef')for v in (args.artifact_sha256,args.receipt_sha256,args.source_receipt_sha256)),'reviewed artifact/digest arguments required')
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','closed hosted read-only comparison required')
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    report={'schema':'lighton4b-mixed-step33-comparison/v1','status':'blocked','source':SOURCE,'native_source':NATIVE_SOURCE,'native_run':NATIVE_RUN,'requested':vars(args),'comparer_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'inference_executed':False,'quality_qualified':False,'held_out_calibration':False,'rows':[]}
    report['ref_source']={'target':'crispembed-diff-lighton4b-document','unit_sha256':sha((ROOT/'tests/lighton4b_document_diff.cpp').read_bytes()),'header_sha256':sha((ROOT/'src/crispembed_diff.h').read_bytes()),'max_abs':0.002,'min_cosine':0.99999}
    def run(label,command):
        with(out/(label+'.log')).open('w')as log:subprocess.run([str(x)for x in command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    def save():(out/'step33-comparison.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    try:
        def deadline(signum,frame):raise TimeoutError('bounded comparison deadline')
        signal.signal(signal.SIGALRM,deadline);signal.alarm(1200)
        memory={k:int(v.strip().split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())}
        report['preflight']={'load_average':os.getloadavg(),'available_memory_bytes':memory['MemAvailable'],'swap_total_bytes':memory['SwapTotal'],'swap_free_bytes':memory['SwapFree'],'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
        require(memory['MemAvailable']>=2*2**30 and shutil.disk_usage(scratch).free>=4*2**30 and shutil.disk_usage(out).free>=512*2**20,'comparison capacity unavailable')
        import numpy as np
        require(np.__version__=='2.3.5','frozen array serializer required')
        run('admission-controls',[sys.executable,ROOT/'tests/test_lighton4b_mixed_step33_compare.py'])
        tie=np.array([18.55112075805664,18.55838394165039,1.00390625,1.01171875],dtype='<f4')
        require(rne_bf16(tie,np).tobytes()==np.array([18.5,18.5,1.,1.015625],dtype='<f4').tobytes(),'actual row33/nearest-even rounding controls failed')
        report['rne_control_passed']=True
        report['stage']='authenticate-source';save();full_raw=read_url(FULL_URL,2*2**20);require(sha(full_raw)==FULL_SHA,'retained178-row source differs');full=json.loads(full_raw)
        urls=provider('lighton4b-bf16-mixed-step33-v1',BOOTSTRAP);prefix='bf16-mixed-step33/quant-oracle/'
        require(prefix+'oracle.json'in urls,'source trace receipt missing');raw=read_url(urls[prefix+'oracle.json'],2*2**20);require(sha(raw)==args.source_receipt_sha256,'independently reviewed source trace receipt differs');source=json.loads(raw);validate_source(source,full);(out/'source-trace.json').write_bytes(raw)
        audit_raw=read_url(urls['bf16-mixed-step33/original-adapter-audit/capture.json'],2**20);require(sha(audit_raw)==source['original_adapter_audit_sha256'],'source final audit bytes differ');audit=json.loads(audit_raw)
        require(audit['status']=='captured'and audit['model_forward_calls']==35 and audit['checkpoint']['payload_verified'],'source final driver/checkpoint audit failed')
        (out/'source-audit.json').write_bytes(audit_raw)
        report['stage']='authenticate-native';save();service=api('actions/runs/'+str(NATIVE_RUN));require((service['head_sha'],service['conclusion'],service['run_attempt'])==(NATIVE_SOURCE,'success',1),'native actual producer differs')
        artifact=api('actions/artifacts/'+args.artifact_id);require(not artifact['expired']and artifact['workflow_run']['id']==NATIVE_RUN and artifact['digest']=='sha256:'+args.artifact_sha256 and 0<artifact['size_in_bytes']<=2*2**30,'native artifact identity/bound differs')
        archive=scratch/'native.zip'
        with archive.open('xb')as stream:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+args.artifact_id+'/zip'],stdout=stream,check=True,timeout=600)
        with archive.open('rb')as stream:require(hashlib.file_digest(stream,'sha256').hexdigest()==args.artifact_sha256,'complete native archive hash differs')
        build=scratch/'ref-build';binary=build/'crispembed-diff-lighton4b-document'
        def compare_arrays(original,candidate,label):
            require(len(original)==len(candidate)and len(original)%4==0,'paired actual stage geometry differs')
            for name,data in [('original',original),('native',candidate)]:tiny_archive(scratch/(name+'.gguf'),'stage',[1,len(data)//4],data)
            order=scratch/'order.tsv';order.write_text(f'stage float {len(original)//4} 1 1 1\n')
            result=subprocess.run([str(binary),str(scratch/'original.gguf'),str(scratch/'native.gguf'),str(order)],capture_output=True,text=True,timeout=60)
            stages=[line for line in result.stdout.splitlines()if line.startswith('STAGE ')];require(result.returncode in(0,1)and len(stages)==1,'actual Ref comparison incomplete')
            (out/(label+'.log')).write_text(result.stdout+result.stderr);return {'returncode':result.returncode,'stage':stages[0]}
        with zipfile.ZipFile(archive)as z:
            names=z.namelist();require(len(names)==len(set(names)),'duplicate native ZIP member')
            def member(suffix,limit):
                hits=[n for n in names if n==suffix or n.endswith('/'+suffix)];require(len(hits)==1 and z.getinfo(hits[0]).file_size<=limit,'native member missing/ambiguous/oversized');return z.read(hits[0])
            raw=member('head-bf16-native-ab.json',2**20);require(sha(raw)==args.receipt_sha256,'reviewed native report differs');native=json.loads(raw);(out/'native-report.json').write_bytes(raw)
            require(native['status']=='measured'and native['source_sha']==NATIVE_SOURCE and native['full_completion_executed']and native['model']['sha256']=='d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474','native completion/model differs')
            require(set(native['prepared_controls'])==set(full['prepared_inputs'])and len(native['prepared_controls'])==5,'all actual source/native inputs required')
            require(all(v['exact']and v['canonical_npy_sha256']==v['source_npy_sha256']==full['prepared_inputs'][k]['sha256']for k,v in native['prepared_controls'].items()),'actual source/native prepared inputs differ')
            require(len(native['raw_vision_controls'])==29 and all(v['bit_exact']and v['native_sha256']==v['source_sha256']for v in native['raw_vision_controls'].values()),'native exact vision prerequisites differ')
            ab=native['head_bf16_ab'];require(ab['trace_position']==33 and ab['trace_same_source_prefix_off']and native['decoded']['ids'][:33]==full['arms']['mixed']['ids'][:33],'actual shared prefix differs')
            require('greedy.33.logits.f32'in ab['baseline_shared_artifacts_exact']and 'HEAD_BF16 passed=1 source_row33_tie=1'in ab['controls_stdout'],'native frozen baseline/actual head controls differ')
            run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF']);run('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
            control=np.ones(2560,dtype='<f4');require(compare_arrays(control.tobytes(),control.tobytes(),'control-equal')['returncode']==0,'actual Ref equal control failed');changed=control.copy();changed[0]+=1;require(compare_arrays(control.tobytes(),changed.tobytes(),'control-changed')['returncode']==1,'actual Ref rejection control failed')
            report['actual_ref_controls_passed']=True
            report['native_on_same_source_prefix']=ab['trace_same_source_prefix_on'];report['native_ab_decoded']={k:ab[k]for k in ('on_complete_ids_equal_source_mixed','on_text_equal_source_mixed','off_first_differing_id_vs_source_mixed','on_first_differing_id_vs_source_mixed')}
            source_arrays={};source_out=out/'source-activations';source_out.mkdir()
            for name,row in source['trace_stages'].items():
                source_arrays[name]=fetch(urls,prefix+row['file'],row);(source_out/row['file']).write_bytes(source_arrays[name])
            order=['input','positions']+['layer0.'+n for n in ('input_norm','qkv','z','a','b','attention','attention_residual','post_norm','gate','up','activation','down')]+['layer.'+str(i)for i in range(32)]+['norm','head.output']
            report['stage']='compare-actual-shared-prefix-stages';save()
            for arm,folder,records in [('off','mixed',native['artifacts'])]+([('on','head-bf16',ab['on_artifacts'])]if ab['trace_same_source_prefix_on']else[]):
                native_out=out/('native-'+arm);native_out.mkdir()
                ids=native['decoded']['ids']if arm=='off'else ab['on_ids'];text=native['decoded']['text']if arm=='off'else ab['on_text']
                for filename in ('generated.ids','generated.txt'):
                    record=records[filename];raw=member(folder+'/'+filename,2**20);require(len(raw)==record['bytes']and sha(raw)==record['sha256'],'raw independent native decoded artifact differs')
                    require(([int(v)for v in raw.decode().split()]==ids)if filename=='generated.ids'else(raw==text.encode()),'raw native IDs/text disagree with receipt')
                    (native_out/filename).write_bytes(raw)
                for suffix in order:
                    name='greedy.33.'+suffix;source_bytes=source_arrays[name];filename='greedy.33.logits.f32'if suffix=='head.output'else name+'.f32';row=records[filename];candidate=member(folder+'/'+filename,len(source_bytes));require(len(candidate)==row['bytes']==len(source_bytes)and sha(candidate)==row['sha256'],'native actual stage bytes differ')
                    (native_out/filename).write_bytes(candidate)
                    if suffix=='positions':require(source_bytes==candidate,'source/native actual rotary positions differ')
                    source_view=np.frombuffer(source_bytes,dtype='<f4');native_view=np.frombuffer(candidate,dtype='<f4');require(np.isfinite(source_view).all(),'actual source activation nonfinite')
                    nonfinite=int((~np.isfinite(native_view)).sum())
                    item={'arm':arm,'name':name,'source_sha256':sha(source_bytes),'native_sha256':sha(candidate),'elements':len(candidate)//4,'source_original_dtype':source['trace_stages'][name]['original_dtype'],'raw':compare_arrays(source_bytes,candidate,arm+'-'+suffix),'source_abs_max':float(np.abs(source_view).max()),'native_abs_max':None if nonfinite else float(np.abs(native_view).max()),'native_nonfinite_elements':nonfinite}
                    if item['source_original_dtype']=='torch.bfloat16':
                        rounded=rne_bf16(np.frombuffer(candidate,dtype='<f4'),np).tobytes();item['native_rne_bf16_control']=compare_arrays(source_bytes,rounded,arm+'-'+suffix+'-rne');item['native_rne_bit_exact_source']=rounded==source_bytes
                    if suffix=='head.output'and not nonfinite:
                        report.setdefault('actual_head_choice',{})[arm]={'source_winner':int(np.argmax(source_view)),'native_winner':int(np.argmax(native_view)),'native_rne_winner':int(np.argmax(rne_bf16(native_view,np))),'source401':float(source_view[401]),'source1778':float(source_view[1778]),'native401':float(native_view[401]),'native1778':float(native_view[1778])}
                    report['rows'].append(item);save()
        report['stage']='authenticate-shared-causal-window';save();all_urls=provider('lighton4b-bf16-mixed-free-generation-v1','ee97b217a97d303391228c3e2c34b5fa1c74adfe5eeda4a93cd7d312429c03fe');prefix_full='bf16-mixed-freegen/quant-oracle/';raw=read_url(all_urls[prefix_full+'oracle.json'],2*2**20);require(sha(raw)==FULL_SHA,'actual178 service receipt differs');destination=out/('retained-source178'if args.preserve_full_source else'retained-source26');destination.mkdir();(destination/'oracle.json').write_bytes(raw)
        source_rows={}
        for row in full['arms']['mixed']['scores']if args.preserve_full_source else full['arms']['mixed']['scores'][:26]:
            data=fetch(all_urls,prefix_full+row['file'],row);(destination/row['file']).write_bytes(data)
            if row['position']<26:source_rows[row['position']]=data
        if args.preserve_full_source:
            for key in ('ids_file','decoded_file'):
                row=full['arms']['mixed'][key];(destination/row['file']).write_bytes(fetch(all_urls,prefix_full+row['file'],row))
            report['retained_full_source_rows']=178
        original_protocol_raw=(ROOT/'docs/lighton4b-bf16-quant-v1.json').read_bytes();require(sha(original_protocol_raw)=='4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc','frozen official BF16 full protocol differs');official=json.loads(original_protocol_raw)
        require(full['arms']['mixed']['ids'][:25]==official['original_generated_ids'][:25]and full['arms']['mixed']['ids'][25]!=official['original_generated_ids'][25],'closed25 shared ID prefix differs')
        original_urls=provider('lightonocr-4b-bf16-cpu-capture-v1','c370b097a45c2d8158343fa1a3a3c510f9bea3b84912565e896b5ded37fd93f6');original_prefix='lighton4b-bf16-freegen/';raw=read_url(original_urls[original_prefix+'capture.json'],2**20);require(sha(raw)==official['original_capture_sha256'],'official original BF16 receipt differs');original_capture=json.loads(raw)
        require(original_capture['status']=='captured'and original_capture['requested_dtype']=='bfloat16'and original_capture['source_sha']==official['original_source_sha']and original_capture['generated_ids']==official['original_generated_ids']and original_capture['scores']==official['score_records']and original_capture['eos_completed']and not original_capture['cap_hit'],'official full original closure differs')
        require({k:v['sha256']for k,v in original_capture['prepared_inputs'].items()}=={k:v['sha256']for k,v in full['prepared_inputs'].items()},'official/mixed original prepared inputs differ')
        for key in ('torch','transformers','tokenizers','numpy'):require(original_capture['packages'][key]==full['runtime_packages'][key],'official/mixed original core framework versions differ')
        require(original_capture['upstream']['loaded_source_sha256']==full['original_module_source']['loaded_source_sha256'],'official/mixed original actual model/config source differs')
        window=out/'shared-causal-window26';window.mkdir();(window/'official-capture.json').write_bytes(raw);report['causal_window26']={'shared_previous_ids':25,'score_positions':26,'rows':[],'limit':'OfficialBF16 versus mixedBF16 includes quantization+dequant-to-BF16 rounding; mixedBF16 versus native includes runtime+effective-weight-precision. No later different contexts compared.'}
        with zipfile.ZipFile(archive)as z:
            for i in range(26):
                original_row=official['score_records'][i];original_data=fetch(original_urls,original_prefix+original_row['file'],original_row);(window/('official-'+original_row['file'])).write_bytes(original_data)
                filename=f'greedy.{i}.logits.f32';record=native['artifacts'][filename];hits=[n for n in z.namelist()if n=='mixed/'+filename or n.endswith('/mixed/'+filename)];require(len(hits)==1 and z.getinfo(hits[0]).file_size==993280,'native causal score missing/ambiguous');native_data=z.read(hits[0]);require(len(native_data)==record['bytes']==993280 and sha(native_data)==record['sha256'],'native causal score identity differs');(window/filename).write_bytes(native_data)
                report['causal_window26']['rows'].append({'position':i,'official_sha256':original_row['sha256'],'mixed_bf16_sha256':full['arms']['mixed']['scores'][i]['sha256'],'native_sha256':record['sha256'],'official_vs_mixed':compare_arrays(original_data,source_rows[i],f'causal-{i:02d}-official-mixed'),'mixed_vs_native':compare_arrays(source_rows[i],native_data,f'causal-{i:02d}-mixed-native'),'official_vs_native':compare_arrays(original_data,native_data,f'causal-{i:02d}-official-native')});save()
        passed={arm:all(row['raw']['returncode']==0 for row in report['rows']if row['arm']==arm)for arm in {row['arm']for row in report['rows']}}
        if 'on'in passed:
            off={r['name']:r['native_sha256']for r in report['rows']if r['arm']=='off'and r['name']!='greedy.33.head.output'}
            on={r['name']:r['native_sha256']for r in report['rows']if r['arm']=='on'and r['name']!='greedy.33.head.output'}
            report['native_step33_nonhead_off_on_byte_exact']=off==on
            require(off==on,'head-only candidate changed same-prefix nonhead stages')
        report['earliest_raw_difference_by_arm']={arm:next((row['name']for row in report['rows']if row['arm']==arm and row['source_sha256']!=row['native_sha256']),None)for arm in passed}
        report['earliest_budget_failure_by_arm']={arm:next((row['name']for row in report['rows']if row['arm']==arm and row['raw']['returncode']),None)for arm in passed}
        report.update(status='compared',raw_numerical_passed_by_arm=passed,native_artifact={k:artifact[k]for k in ('id','digest','size_in_bytes','expires_at')},source_receipt_sha256=args.source_receipt_sha256,native_receipt_sha256=args.receipt_sha256,binary_sha256=sha(binary.read_bytes()),rounding_control_limit='Diagnostic output rounding only; raw arrays/gold unchanged. Different effective BF16/F32 weights and prior state still require operator attribution.',scope='observed form sharedprefix33 only; no full-model/heldout qualification')
        return 0 if all(passed.values())else 1
    except Exception as error:
        message=str(error)if isinstance(error,ValueError)else'bounded authenticated stage comparison failed'
        for value,label in ((ROOT,'[repository]'),(scratch,'[scratch]'),(out,'[report]'),(os.environ.get('KAGGLE_ACCOUNT',''),'[account]')):
            if str(value):message=message.replace(str(value),label)
        report.update(error_type=type(error).__name__,error=message);return 2
    finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
