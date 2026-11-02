"""Seven missing arms only; original run_case math/gates copied from83.

Old module/source guard unchanged. Only reconstructible adapter storage and
retention receipts differ; canonical dumps/logs stay in working output.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'tools/minicpm-ocr'),str(ROOT/'tools')]
from common import (atomic,sha,require,source_guard,resources,numerical_environment,authenticate_old,CATALOG,
                    remaining,used,room,canonical,duplicate)
import retention
from graph_panel import (captured,parse_timing,load_npy,raw_spec,spec,write_gguf,
    first_difference,edits,MAXIMUM,verify_reference,tensor_inventory,protected_q4,protected_q8_patch,
    recover_case,reference_controls)
write=atomic

STOP=False

def stop(_signal,_frame):
    global STOP
    STOP=True
    raise RuntimeError('Hosted continuation termination requested; retain partial canonical output')


def captured(command,folder,name,env,timeout):
    import os,signal,types
    stdout=folder/(name+'.stdout');stderr=folder/(name+'.stderr');timed=False
    with stdout.open('wb')as a,stderr.open('wb')as b:
        child=subprocess.Popen(list(map(str,command)),stdout=a,stderr=b,env=env,start_new_session=True)
        try:child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed=True;os.killpg(child.pid,signal.SIGKILL);child.wait()
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait();raise
    return types.SimpleNamespace(returncode=child.returncode,stdout=stdout.read_bytes(),stderr=stderr.read_bytes(),timed_out=timed)


def digest(path):return sha(path)

def run_case(args, base, case, dtype, graph, model, output, scratch):
    import numpy as np
    folder=output/f'{dtype}-graph{graph}'/case['id']; dump=folder/'dump';dump.mkdir(parents=True)
    temporary=scratch/f'{dtype}-graph{graph}'/case['id'];temporary.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CRISPEMBED_MINICPM_OCR='1',CRISPEMBED_QWEN35_OCR='1',CRISPEMBED_QWEN35_QUANT='1' if dtype in ('q8_0','q4_k') else '0',CRISPEMBED_MINICPM_OCR_GRAPH=str(graph),CRISPEMBED_QWEN35_WEIGHT_CACHE='0',CRISPEMBED_MINICPM_OCR_TIMING='1',CRISPEMBED_MINICPM_OCR_DUMP=str(dump),OMP_NUM_THREADS=str(args.threads))
    for name in ['CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_QWEN35_PARALLEL_F16','CRISPEMBED_MINICPM_OCR_VISION_FLASH']:
        env.pop(name,None)
    start=time.monotonic()
    process=captured([str(args.cli),'-m',str(model),'--ocr',str(base/(case['id']+'.png')),'--ocr-prompt',case['task'],'--ocr-max-tokens',str(MAXIMUM),'-t',str(args.threads)],folder,'native',env,5400)
    row=dict(id=case['id'],dtype=dtype,graph=graph,exit=process.returncode,wall_seconds=time.monotonic()-start,stdout_sha256=digest(folder/'native.stdout'),stderr_sha256=digest(folder/'native.stderr'),timed_out=process.timed_out,timings=parse_timing(process.stderr.decode('utf-8','replace')))
    if process.returncode:
        return dict(row,passed=False,error='native command failed before complete output')
    ids=np.load(dump/'ids.npy',allow_pickle=False).tolist()
    native_input=np.load(dump/'input_ids.npy',allow_pickle=False).tolist()
    text=process.stdout.decode('utf-8')
    if text.endswith('\n'):text=text[:-1]
    row.update(ids=ids,ids_exact=ids==case['ids'],input_ids_exact=native_input==case['input_ids'],text=text,text_exact=text==case['text'],eos=bool(ids) and ids[-1] in (248044,248046),cap_hit=len(ids)>=MAXIMUM)
    row.update(first_input_id_difference=first_difference(native_input,case['input_ids']),first_generated_id_difference=first_difference(ids,case['ids']),characters=len(text),reference_characters=len(case['text']),cer=edits(text,case['text'])/max(1,len(case['text'])),wer=edits(text.split(),case['text'].split())/max(1,len(case['text'].split())))
    write(folder/'partial-case.json',row)
    confidence=load_npy(dump/'confidences.npy').reshape(-1)
    require(confidence.size==len(ids) and ((confidence>=0)&(confidence<=1)).all(), 'every native confidence finite/bounded')
    row['confidences']=confidence.tolist()
    row['mean_confidence']=float(confidence.astype('float64').mean())
    write(folder/'partial-case.json',row)
    pairs=[]
    for i,label in enumerate([f'model.language_model.layers.{i}' for i in range(24)]+['model.language_model.norm','lm_head']):
        original=load_npy(base/case['arrays'][label]['path'])
        native=load_npy(dump/(label.replace('.','_')+'.npy'))
        # Publisher stores batch1 features [1,T,C]; native stores explicit
        # token rows [T,C]. Removing only known singleton batch is declared.
        if original.ndim==3:
            require(original.shape[0]==1,'only known batch1 alignment');original=original[0]
        if label=='lm_head':
            require(original.size==248100 and native.shape==(1,248100),'first vocabulary row shape')
            original=original.reshape(1,248100)
        require(original.shape==native.shape,'prefill stage shape differs: '+label)
        pairs.append(dict(name=f'stage_{i}',original=raw_spec(temporary/f'original_{i}.f32',original),native=raw_spec(temporary/f'native_{i}.f32',native)))
    for i,pair in enumerate(pairs):
        label=retention.LABELS[i]
        pair['reconstruction']=dict(
            original=retention.anchor(base/case['arrays'][label]['path'],case['arrays'][label]['path'],'Known singleton batch1 removed; F32 C-order'),
            native=retention.anchor(dump/(label.replace('.','_')+'.npy'),'dump/'+label.replace('.','_')+'.npy','F32 C-order'))
    row['prefill']=retention.compare(pairs,args.diff,temporary/'prefill',folder/'prefill')
    row['first_divergence']=next((dict(stage=i,**v) for i,v in row['prefill']['stages'].items() if not v['passed']),None)
    row['passed']=row['input_ids_exact'] and row['ids_exact'] and row['text_exact'] and row['eos']==case['eos'] and row['cap_hit']==case['cap_hit'] and row['prefill']['passed']
    write(folder/'partial-case.json',row)
    if graph and dtype in ('f32','f16'):
        audit=folder/'projected';audit.mkdir()
        idfile=raw_spec(temporary/'ids-as-exact-f32.f32',np.asarray(case['input_ids'],dtype='<f4'))
        merged=raw_spec(temporary/'merged.f32',load_npy(dump/'model_merger.npy'))
        write_gguf(temporary/'probe-input.gguf',[dict(name='input_ids',**idfile),dict(name='merged',**merged)])
        probe=captured([str(args.probe),str(model),str(temporary/'probe-input.gguf'),str(audit),str(args.threads)],audit,'probe',env,3600)
        (audit/'probe.log').write_bytes(probe.stdout+probe.stderr)
        require(probe.returncode==0,'projected probe failed')
        journal=json.loads((audit/'projections.json').read_text());require(journal['completed'] and journal['calls']==187 and journal['ordinary_observer_repeat_exact'] and journal['cache_exact'],'projection coverage/neutrality')
        pp=[]
        for entry in journal['stages']:
            key=entry['key'];shape=[len(entry['selected_rows']),entry['output_width']]
            pp.append(dict(name=key,original=spec(audit/(key+'.control.f32'),shape),native=spec(audit/(key+'.native.f32'),shape)))
        # Actual root CLI and the isolated observed build must compute the
        # identical full prefill, not merely similar sampled projections.
        for i in range(24):
            require(digest(audit/f'layer_{i}.f32')==digest(temporary/f'native_{i}.f32'),'root/isolated full layer byte mismatch')
        require(digest(audit/'norm.f32')==digest(temporary/'native_24.f32') and digest(audit/'logits.f32')==digest(temporary/'native_25.f32'),'root/isolated final byte mismatch')
        for pair in pp:
            pair['reconstruction']={side:retention.anchor(pair[side]['path'],'projected/'+Path(pair[side]['path']).name,'Raw F32 sampled rows; unchanged')for side in ['original','native']}
        row['projected']=retention.compare(pp,args.diff,temporary/'projected-comparison',audit/'comparison')
        row['projected']['scope']=journal['sampling']
        row['projected']['stored_types']=journal['stages']
        row['first_projected_divergence']=next((dict(weight=item['weight'],stored_type=item['stored_type'],**row['projected']['stages'][item['key']]) for item in journal['stages'] if not row['projected']['stages'][item['key']]['passed']),None)
        row['passed'] &= row['projected']['passed']
    return row



def seal(folder,out,scratch):
    """Canonical copies persist. Drop only proven full-prefill probe duplicates."""
    folder=Path(folder);journal={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p))
                              for p in folder.rglob('*')if p.is_file()and not p.is_symlink()}
    plan=[]
    for name,pin in journal.items():
        if not duplicate(name):continue
        require(name.startswith('projected/'),'Only probe-prefill duplicates should reside in canonical output')
        stem=Path(name).stem;label=('model_language_model_layers_'+stem[6:]if stem.startswith('layer_')else
                                  'model_language_model_norm'if stem=='norm'else'lm_head')
        original=folder/'dump'/(label+'.npy');values=retention.npy(original)
        spec=dict(path=original,kind='npy',shape=list(values.shape),raw_sha256=pin['sha256'])
        for _ in retention.payload(spec):pass
        plan.append(dict(member=name,identity=pin,source_npy='dump/'+original.name,source_npy_sha256=sha(original),shape=list(values.shape),raw_f32_sha256=pin['sha256']))
    atomic(folder/'probe-reconstruction.json',dict(duplicates=plan,canonical_rechecked=True,canonical_removed=False))
    for row in plan:
        path=folder/row['member'];require(sha(path)==row['identity']['sha256'],'Probe duplicate changed before cleanup');path.unlink()
    atomic(folder/'files.json',{str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in folder.rglob('*')if p.is_file()and p.name!='files.json'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['retained','reference','scratch','out','cli','probe','diff']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--build-receipt',type=Path,required=True)
    parser.add_argument('--reviewed-source',required=True);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--model',action='append',required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=True)
    result=dict(completed=False,qualified=False,old23_mutated=False,original_rerun=False,new_cases=[])
    try:
        require(args.threads==4,'Frozen v3 four-thread execution required')
        result['source']=source_guard('continue',args.reviewed_source);result['resources']=resources(args.out,args.scratch)
        built=json.loads(args.build_receipt.read_text());require(built['completed']and built['mode']=='continue'and built['source']['source']==args.reviewed_source,'Exact same-source model-free build receipt required')
        require(all(sha(getattr(args,key))==built['binary_sha256'][key]for key in ['cli', 'probe', 'diff']),'Built binary identity differs')
        result['build_receipt_sha256']=sha(args.build_receipt)
        result['tooling']=numerical_environment()
        catalog,old_panel,old=authenticate_old(args.retained);ref,identity=verify_reference(args.reference);result['original']=identity
        models=dict(v.split('=',1)for v in args.model);require(set(models)=={'f32','f16','q8_0','q4_k'},'Exact four frozen models required')
        for dtype,path in models.items():
            pin=catalog['models'][dtype];require(Path(path).stat().st_size==pin['bytes']and sha(path)==pin['sha256'],'Reconstructed model whole SHA differs')
        inventory={dtype:tensor_inventory(path)for dtype,path in models.items()}
        result['q4_protection']=protected_q4(inventory['f16'],inventory['q4_k']);result['q8_patch_protection']=protected_q8_patch(inventory['f16'],inventory['q8_0'])
        result['ref_controls']=reference_controls(args.diff,args.out/'ref-controls')
        result['carried']=retention.compact_old(old,ref,args.reference,args.out/'retained-v3')
        shutil.copyfile(args.retained/'panel.json',args.out/'retained-v3/panel.json')
        shutil.copytree(args.retained/'initial',args.out/'retained-v3/initial')
        result['carried_rows']=[item['summary']for item in old]
        references={c['id']:c for c in ref['cases']}
        for cid,dtype,gate in remaining(catalog):
            # The conservative per-arm2GiB bound exceeds frozen tall/tiny
            # canonical bytes. Scratch adapters never count against working.
            result['resources_before_next']=resources(args.out,args.scratch,next_bound=2*2**30)
            folder=args.out/'new'/f'{dtype}-graph{gate}'/cid
            try:row=run_case(args,args.reference,references[cid],dtype,gate,Path(models[dtype]),args.out/'new',args.scratch/'new')
            except Exception as error:row=recover_case(folder,cid,dtype,gate,error)
            if folder.is_dir():seal(folder,args.out,args.scratch)
            row['files_manifest_sha256']=sha(folder/'files.json')
            row['case_report']=str((folder/'case-report.json').relative_to(args.out));atomic(folder/'case-report.json',row)
            row['case_report_sha256']=sha(folder/'case-report.json')
            result['new_cases'].append(row);atomic(args.out/'continuation.json',result)
            require(not STOP,'Hosted termination retained partial canonical traces')
        all_rows=result['carried_rows']+result['new_cases']
        require(len(all_rows)==30 and len({(r['id'],r['dtype'],r['graph'])for r in all_rows})==30,'Full30 unique arms required')
        pairs=[]
        for dtype in ['f32','f16']:
            for cid in ['line','para-en','table','tall','tiny']:
                both=[next(r for r in all_rows if(r['id'],r['dtype'],r['graph'])==(cid,dtype,g))for g in [0,1]]
                valid=both[0].get('ids')==both[1].get('ids')and len(both[0].get('confidences',[]))==len(both[1].get('confidences',[]))>0
                error=max((abs(a-b)for a,b in zip(both[0].get('confidences',[]),both[1].get('confidences',[]))),default=None)if valid else None
                pairs.append(dict(id=cid,dtype=dtype,max_absolute=error,passed=valid and error<=.002))
        structural=all(not r.get('error')and r.get('exit')==0 and len(r.get('prefill',{}).get('stages',{}))==26 and
            r['prefill']['actual_ref']['all_metrics_agree']and
            (not(r['graph']and r['dtype']in ['f32','f16'])or(len(r.get('projected',{}).get('stages',{}))==187 and r['projected']['actual_ref']['all_metrics_agree']))
            for r in result['new_cases'])
        result.update(completed=structural,attempted_new_arms=7,combined_rows=30,paired_confidences=pairs,
            all_required_capture_and_ref_structural=structural,diagnostic_completion_is_not_numerical_qualification=True,
            all_decoded_ids_text_eos_exact=all(r.get('ids_exact')and r.get('text_exact')and r.get('eos')and not r.get('cap_hit')for r in all_rows),
            qualified=all(r.get('passed')for r in all_rows)and all(r['passed']for r in pairs),
            historical_numeric_failures_preserved=True,old_run_completed=False,default_enabled=False,publication_qualified=False,
            limits='23 immutablev3 rows plus7 new source-identical-runtime rows; exact per-case source/binary provenance remains separate. No original rerun, gate waiver or backend-cause claim.')
        require(structural,'Seven attempts retained but structural capture/Ref rejected')
    except Exception as error:result['error']=type(error).__name__+': '+str(error);raise
    finally:atomic(args.out/'continuation.json',result)


if __name__=='__main__':
    import signal
    signal.signal(signal.SIGTERM,stop)
    main()
