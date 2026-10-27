#!/usr/bin/env python3
"""Hosted1.5M CPU/a06 experiment: old primary gold plus reviewed supplemental leaves."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
import contract_v2 as c
import replay as r
from diff_adapter import write_pair


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))


def fetch(pin,kind,folder):
    cfg=pin[kind+'_artifact'];row=api('actions/artifacts/'+str(cfg['id']))
    if (row['workflow_run']['id']!=pin['run'] or row['workflow_run']['head_sha']!=pin['source'] or row['expired'] or
            row['name']!=cfg['name'] or row['size_in_bytes']!=cfg['bytes'] or row['digest']!='sha256:'+cfg['sha256']):
        raise ValueError('Immutable service/artifact lineage differs')
    archive=folder/(kind+'.zip')
    with archive.open('xb') as stream:
        subprocess.run(['gh','api',f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{cfg["id"]}/zip'],stdout=stream,check=True,timeout=300)
    if archive.stat().st_size!=cfg['bytes'] or c.digest(archive)!=cfg['sha256']:raise ValueError('Frozen ZIP SHA/extent differs')
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();names=z.namelist()
        if (len(entries)>2048 or len(set(names))!=len(names) or sum(i.file_size for i in entries)>512*1024**2 or
                any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names) or
                any((i.external_attr>>16)&0o170000==0o120000 for i in entries)):raise ValueError('Unsafe retained ZIP')
        z.extractall(folder/kind)
    return {k:row[k] for k in ('id','name','size_in_bytes','digest','expires_at')}


def check_run(pin,conclusion):
    row=api('actions/runs/'+str(pin['run']))
    if row['head_sha']!=pin['source'] or row['run_attempt']!=1 or row['status']!='completed' or row['conclusion']!=conclusion:
        raise ValueError('Exact original/diagnostic service outcome differs')
    return {k:row[k] for k in ('id','head_sha','status','conclusion','run_attempt','html_url')}


def load_retained(scratch):
    cfg=c.protocol();reports={};roots={}
    for kind in ('primary','supplemental'):
        root=scratch/kind/'raw/moedern-original';data=(root/'review/original.json').read_bytes()
        if c.digest(root/'review/original.json')!=cfg[kind]['receipt_sha256'] or data!=(scratch/kind/'review/moedern-original/review/original.json').read_bytes():
            raise ValueError('Exact primary/supplemental receipt differs')
        reports[kind]=json.loads(data);roots[kind]=root;c.verify_arrays(root,reports[kind])
    expected={'completed':False,'error':cfg['supplemental']['error']}
    if json.loads((roots['supplemental']/'review/rejection.json').read_text())!=expected:
        raise ValueError('Initial observer strict rejection must remain unchanged')
    lineage=c.check_reports(reports['primary'],reports['supplemental'])
    if c.coverage(reports['primary'])!=cfg['coverage']:raise ValueError('Declared mapped/unmapped primary coverage changed')
    diagnosis=scratch/'diagnosis/review/diagnostic.json'
    if c.digest(diagnosis)!=cfg['diagnosis']['receipt_sha256']:raise ValueError('Reviewed SDPA receipt differs')
    proof=json.loads(diagnosis.read_text())
    if (proof['completed']is not True or any(proof[k]is not False for k in ('qualification','new_gold','native_outputs','old_continuity_accepted')) or
            proof['source']['head']!=cfg['diagnosis']['source'] or proof['historical']['observer_continuity_remains_rejected']is not True or
            proof['historical']['old_array_count']!=250 or proof['historical']['observer_array_count']!=271):
        raise ValueError('Reviewed operator diagnosis is not a new completed original')
    for case in proof['cases']:
        if case['id'] not in c.panel()['full_capture'] or not case['shared_inputs_bit_exact'] or not case['module_observer_repeat_bit_exact']:
            raise ValueError('Reviewed original-source operator inputs/neutrality differ')
        for backend in ('default','math','cpu-flash'):
            arm=case['arms'][backend]
            if (arm['status']!='completed-operator-control' or not arm['ordinary_repeat_observer_profiled_bit_exact'] or
                    not arm['backend_state_restored'] or not arm['actual_input_objects_layout_and_bits_unchanged']):
                raise ValueError('Reviewed SDPA backend/neutrality control differs')
    return roots,reports,lineage


def prepare(out,scratch):
    measured=c.source()
    if measured['head']!=os.environ.get('GITHUB_SHA'):raise ValueError('Service/source differs')
    if scratch.exists():raise ValueError('Fresh scratch required')
    scratch.mkdir()
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    if available<2*2**30 or shutil.disk_usage(scratch).free<4*2**30:raise ValueError('Hosted2GiBavailableRAM/4GiBfree disk required')
    cfg=c.protocol();service={}
    for kind,conclusion in (('primary','success'),('supplemental','failure'),('diagnosis','success')):
        pin=cfg[kind];folder=scratch/kind;folder.mkdir()
        service[kind]=dict(run=check_run(pin,conclusion),review=fetch(pin,'review',folder))
        if kind!='diagnosis':service[kind]['raw']=fetch(pin,'raw',folder)
    _,_,lineage=load_retained(scratch)
    write(out/'review/prepared.json',dict(source=measured,protocol_sha256=c.digest(c.PATH),service=service,lineage=lineage,
        primary_replaced=False,old_original_continuity_accepted=False,frameworks_imported=False,native_executed=False))


def array(root,row,np):
    dtype={'f32':'<f4','i64':'<i8','u8':'u1'}[row['dtype']]
    return np.fromfile(root/row['path'],dtype=dtype).reshape(row['shape'])


def spec(root,row):return dict(path=root/row['path'],shape=row['shape'],sha256=row['sha256'],label=row['path'])


def compare_group(diff,folder,pairs,checks):
    if not pairs:return None
    folder.mkdir(parents=True,exist_ok=True)
    adapters=write_pair(folder/'primary.gguf',folder/'comparison.gguf',pairs)
    write(folder/'adapters.json',adapters)
    result=r.actual_ref(diff,folder/'primary.gguf',folder/'comparison.gguf',folder/'ref.json',folder/'ref.log',checks)
    if not result['all_metrics_agree']:raise ValueError('Independent metrics/actualRef calculation differs')
    return dict(actual_ref=result,adapters=adapters)


def original_drift(diff,roots,reports,out,np,small):
    """All250 shared rows, with no acceptance of byte continuity or new gold."""
    old,new=c.indexed(reports['primary']),c.indexed(reports['supplemental']);results=[]
    for case in reports['primary']['cases']:
        rows=[x for x in reports['primary']['stage_arrays'] if x['case']==case['id']];checks={};pairs=[];records=[]
        for i,left in enumerate(rows):
            right=new[case['id'],left['stage']];a=array(roots['primary'],left,np);b=array(roots['supplemental'],right,np)
            name=f'drift_{i:03d}';m=r.compare(a,b,integer=left['dtype']!='f32');m['byte_exact']=left['sha256']==right['sha256'];checks[name]=m
            records.append(dict(stage=left['stage'],dtype=left['dtype'],shape=left['shape'],primary_sha256=left['sha256'],supplemental_sha256=right['sha256'],metrics=m))
            if left['dtype']=='f32':pairs.append(dict(name=name,original=spec(roots['primary'],left),native=spec(roots['supplemental'],right)))
        cpp=compare_group(diff,out/case['id'],pairs,{p['name']:checks[p['name']] for p in pairs})
        results.append(dict(case=case['id'],records=records,actual_ref=cpp,passed=all(x['passed'] for x in checks.values())))
    value=dict(schema='crispembed-moedern-original-reproducibility-v2',shared_rows=sum(len(x['records']) for x in results),
        cases=results,primary_replaced=False,original_byte_continuity_accepted=False,
        score_drift=[dict(query=a['query'],document=a['document'],primary=a['original_maxsim'],supplemental=b['original_maxsim'],
                          absolute_difference=abs(a['original_maxsim']-b['original_maxsim'])) for a,b in zip(reports['primary']['scores'],reports['supplemental']['scores'])])
    write(out/'original-drift.json',value)
    write(small,value)
    if value['shared_rows']!=250 or not all(x['passed'] for x in results):raise ValueError('Original drift exceeds unchanged numerical envelope; no native forward')
    return value


def skiplist_ids(tokenizer,config,tokenizer_config):
    vocab=tokenizer['model']['vocab']
    vocab={item[0]:i for i,item in enumerate(vocab)} if isinstance(vocab,list) else dict(vocab)
    for item in tokenizer.get('added_tokens',[]):vocab[item['content']]=item['id']
    unknown=tokenizer_config['unk_token'];unknown=unknown['content'] if isinstance(unknown,dict) else unknown
    return sorted({vocab.get(word,vocab[unknown]) for word in config['skiplist_words'] if vocab.get(word,vocab[unknown])!=vocab[unknown] or word==unknown})


def execute(out,scratch):
    measured=c.source();prepared=json.loads((out/'review/prepared.json').read_text())
    if prepared['source']!=measured or prepared['protocol_sha256']!=c.digest(c.PATH) or measured['head']!=os.environ.get('GITHUB_SHA'):
        raise ValueError('Prepared source/protocol differs before optional packages/build')
    roots,reports,lineage=load_retained(scratch);cfg=c.protocol();schema=c.schema_module(scratch/'schema-source')
    identities,inventory=schema.validate_checkpoint(roots['primary']/'checkpoint','1.5M')
    for key in ('primary','supplemental'):
        ids,inv=schema.validate_checkpoint(roots[key]/'checkpoint','1.5M')
        if ids!=identities or ids!=reports[key]['checkpoint_files'] or inv!=inventory or inv!=reports[key]['inventory']:
            raise ValueError('Exact source checkpoint inventories differ across originals')
    primary=roots['primary'];model=primary/'model-F32.gguf';conversion=primary/'model-F32.conversion.json'
    if c.digest(model)!=cfg['primary']['gguf_sha256'] or c.digest(conversion)!=cfg['primary']['conversion_receipt_sha256']:
        raise ValueError('Original-first primary F32 model/conversion differs')
    conversion_report=json.loads(conversion.read_text())
    if (conversion_report['completed']is not True or conversion_report['original_report_sha256']!=cfg['primary']['receipt_sha256'] or
            conversion_report['gguf_sha256']!=cfg['primary']['gguf_sha256'] or conversion_report['tensor_count']!=32):
        raise ValueError('Primary conversion lineage differs')
    if subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()!=cfg['ggml']:raise ValueError('Actual ggmla06 checkout differs')
    for name,version in (('numpy','2.2.6'),('gguf','0.17.1')):
        if importlib.metadata.version(name)!=version:raise ValueError('Pinned hosted array tools required')
    if any(n in sys.modules for n in ('torch','transformers','sentence_transformers')):raise ValueError('No original framework/recapture allowed')
    import numpy as np
    output=out/'raw';output.mkdir();result=dict(schema='crispembed-moedern-native-dual-reference-v2',completed=False,native_parity=False,
        source=measured,primary_replaced=False,old_original_continuity_accepted=False,full_model_recapture=False,
        retrieval_quality=False,release_qualified=False,reference_lineage=lineage,cases=[],scores=[],failures=[])
    def save():write(out/'review/native-v2.json',result)
    save();build=scratch/'build'
    env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')};env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    # Utilities execute child commands with the same sanitized environment.
    os.environ.clear();os.environ.update(env)
    r.run(['cmake','-S',ROOT/'tests/moedern-native-v2','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],out/'review/configure.log')
    r.run(['cmake','--build',build,'--parallel','2'],out/'review/build.log')
    controls=build/'moedern-native-v2-controls';probe=build/'moedern-native-v2-probe';diff=build/'moedern-native-v2-diff'
    shutil.copyfile(build/'compile_commands.json',out/'review/compile_commands.json')
    result['products']={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in (controls,probe,diff)}
    for p in (controls,probe,diff):shutil.copyfile(p,output/p.name)
    r.run([controls],out/'review/controls.log');fixtures=scratch/'fixtures';fixtures.mkdir()
    r.typed_fixtures(controls,fixtures,out/'review',schema.tensor_schema)
    result['ref_controls']=r.comparator_controls(diff,output/'ref-controls');save()
    drift=original_drift(diff,roots,reports,output/'original-drift',np,out/'review/original-drift.json')
    result['original_drift']=dict(shared_rows=250,old_original_continuity_accepted=False,primary_replaced=False);save()
    checkpoint=primary/'checkpoint';skip=skiplist_ids(json.loads((checkpoint/'tokenizer.json').read_text()),json.loads((checkpoint/'config_sentence_transformers.json').read_text()),json.loads((checkpoint/'tokenizer_config.json').read_text()))
    primary_rows=c.indexed(reports['primary']);supplemental_rows=c.indexed(reports['supplemental']);selected={}
    def frozen(kind,cid,name):
        rows=primary_rows if kind=='primary' else supplemental_rows;row=rows[cid,name]
        return array(roots[kind],row,np),row
    for case in reports['primary']['cases']:
        cid=case['id'];tokens=case['token_ids'];mask=case['supplied_attention_mask'];folder=output/cid
        record=dict(id=cid,role=case['role'],tokens=len(tokens),status='started',checks={},first_differing_stage=None,first_failing_stage=None)
        result['cases'].append(record);save()
        try:
            inputs=scratch/(cid+'.tokens');inputs.write_text('MOEDERN_TOKENS_V1 '+case['role']+' '+str(len(tokens))+'\nids '+' '.join(map(str,tokens))+'\nmask '+' '.join(map(str,map(int,mask)))+'\nskip '+str(len(skip))+' '+' '.join(map(str,skip))+'\n')
            r.run([probe,model,inputs,folder],out/'review'/(cid+'.log'))
            observed,files=r.stages(folder)
            if set(observed)!=c.native_names():raise ValueError('Strict complete55native-stage set differs')
            if (folder/'controls.txt').read_text()!='ordinary_dump_bit_exact=true\nrepeat_bit_exact=true\nobserver_rejection_recovery=true\ngate0_recovery=true\ncontext_ownership=true\n':raise ValueError('Actual native neutrality/ownership controls differ')
            checks=record['checks'];pairs=[];supp_pairs=[]
            def check(label,a,b,integer=False,source_row=None,native_spec=None,kind='primary'):
                m=r.compare(a,b,integer=integer);m.update(reference_kind=kind,bit_exact=bool(m['exact']) if integer else a.dtype==b.dtype and a.tobytes()==b.tobytes());checks[label]=m
                if not integer:
                    if source_row is None or native_spec is None:raise ValueError('Every mappedF32 needs exact authenticated Ref inputs')
                    target=pairs if kind=='primary' else supp_pairs
                    target.append(dict(name=label,original=spec(roots[kind],source_row),native=native_spec))
                return m
            check('contract.ids',np.asarray(tokens,dtype='<i8')[None,:],observed['input_ids'],True,kind='input-contract')
            check('contract.supplied',np.asarray(mask,dtype='<i8')[None,:],observed['supplied_mask'],True,kind='input-contract')
            check('contract.effective',np.asarray(case['effective_attention_mask'],dtype='<i8')[None,:],observed['effective_mask'],True,kind='input-contract')
            scoring=np.fromfile(folder/'scoring.u8',dtype='u1');check('contract.scoring',np.asarray(case['scoring_mask']),scoring,True,kind='input-contract')
            if cid in c.panel()['full_capture']:
                for native_name,name in c.primary_map().items():
                    a,source_row=frozen('primary',cid,name);integer=source_row['dtype']!='f32'
                    check('primary.'+native_name,a,observed[native_name],integer,source_row,files.get(native_name))
                derived=np.broadcast_to(observed['effective_mask'].astype(bool)[:,None,None,:],(1,1,len(tokens),len(tokens)))
                for layer in range(2):
                    a,_=frozen('primary',cid,f'actual.sdpa.{layer}.input.kwargs.attn_mask');check(f'contract.derived_mask.{layer}',a,derived,True,kind='input-contract')
                for stage in cfg['supplemental_extra_descriptors']:
                    if stage['case']!=cid:continue
                    a,source_row=frozen('supplemental',cid,stage['stage']);name='supplemental.'+stage['stage'].replace('#','_').replace('.','_')
                    if stage['dtype']=='f32':
                        native_name='normalized' if stage['stage'].startswith('pipeline.3.') else 'dense'
                        check(name,a,observed[native_name],False,source_row,files[native_name],kind='supplemental')
                    else:
                        if stage['stage'].endswith('.input_ids'):b=observed['input_ids']
                        elif stage['stage'].endswith('.query_expansion_positions'):b=np.logical_not(observed['supplied_mask']).astype('u1')
                        elif stage['stage'].startswith('pipeline.1.'):b=observed['supplied_mask']
                        else:b=scoring[None,:]
                        check(name,a,b,True,kind='supplemental')
            all_vectors=np.fromfile(folder/'all.f32',dtype='<f4').reshape(len(tokens),128)
            if c.digest(folder/'all.f32')!=files['normalized']['sha256']:raise ValueError('Ordinary all-vector storage differs from normalized observer')
            for label,path in (('all_vectors',folder/'all.f32'),('selected',folder/'selected.f32')):
                a,source_row=frozen('primary',cid,'ordinary.features.token_embeddings' if label=='all_vectors' else 'ordinary.selected')
                b=all_vectors if label=='all_vectors' else np.fromfile(path,dtype='<f4').reshape(-1,128)
                check('primary.'+label,a,b,False,source_row,dict(path=path,shape=list(b.shape),sha256=c.digest(path)))
                if label=='selected':selected[cid]=b
            record['primary_ref']=compare_group(diff,folder/'primary-ref',pairs,{p['name']:checks[p['name']] for p in pairs})
            record['supplemental_ref']=compare_group(diff,folder/'supplemental-ref',supp_pairs,{p['name']:checks[p['name']] for p in supp_pairs})
            record['first_differing_stage']=next((dict(stage=n,**m) for n,m in checks.items() if not m['bit_exact']),None)
            record['first_failing_stage']=next((dict(stage=n,**m) for n,m in checks.items() if not m['passed']),None)
            record['status']='passed' if all(x['passed'] for x in checks.values()) else 'numeric_rejection'
            result['failures'] += [dict(case=cid,stage=n,**m) for n,m in checks.items() if not m['passed']]
        except Exception as error:
            record.update(status='structural_rejection',error=type(error).__name__+': '+str(error));result['failures'].append(dict(case=cid,error=record['error']))
        save()
    for target in reports['primary']['scores']:
        q,d=target['query'],target['document'];row=dict(query=q,document=d,primary_original=target['original_maxsim'],passed=False)
        if q in selected and d in selected:
            path=output/(q+'--'+d+'.f32')
            try:
                r.run([probe,'--maxsim',output/q/'selected.f32',output/d/'selected.f32',path],out/'review'/(q+'--'+d+'.log'))
                value=np.fromfile(path,dtype='<f4')
                if value.shape!=(1,) or not np.isfinite(value).all():raise ValueError('Invalid native MaxSim')
                actual=float(value[0]);independent=float((selected[q]@selected[d].T).max(axis=1).sum())
                row.update(native=actual,independent=independent,passed=max(abs(actual-target['original_maxsim']),abs(actual-independent))<=cfg['gates']['maxsim_max_absolute'])
            except Exception as error:row['error']=type(error).__name__+': '+str(error)
        else:row['error']='Required primary selected vectors unavailable after structural case rejection'
        result['scores'].append(row)
        if not row['passed']:result['failures'].append(dict(stage='primary.maxsim',**row))
        save()
    primary_count=sum(sum(k.startswith('primary.') for k in row['checks']) for row in result['cases'])
    supplemental_count=sum(sum(k.startswith('supplemental.') for k in row['checks']) for row in result['cases'])
    result['comparison_coverage']=dict(mapped_primary_arrays=primary_count,supplemental_only_arrays=supplemental_count,
        retained_primary_arrays=250,unmapped_primary_arrays=138,native_required_stages_per_case=55)
    if primary_count!=112 or supplemental_count!=21 or len(result['cases'])!=10 or len(result['scores'])!=3:
        result['failures'].append(dict(stage='coverage',error='Mandatory112primary/21supplemental/10cases/3scores coverage incomplete'))
    result.update(completed=True,native_parity=not result['failures'],status='all10cases/3scores retained; primary250unchanged,21supplementalonly')
    save()
    if result['failures']:raise ValueError('Unchanged nativev2 gates rejected; full panel and primary targets retained')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('prepare','run'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
    if os.environ.get('GITHUB_ACTIONS')!='true':p.error('All raw/weights/build/inference access hostedCPU only')
    if a.phase=='prepare' and a.output.exists():p.error('Fresh task-owned output required')
    if a.phase=='run' and (a.output/'review/native-v2.json').exists():p.error('Preserve earlier panel; fresh run required')
    (a.output/'review').mkdir(parents=True,exist_ok=True)
    try:
        if a.phase=='prepare':prepare(a.output,a.scratch)
        else:execute(a.output,a.scratch)
    except Exception as error:
        write(a.output/'review'/('rejection-'+a.phase+'.json'),dict(error=type(error).__name__+': '+str(error),primary_replaced=False,old_original_continuity_accepted=False));raise
    finally:
        write(a.output/'review'/('files-'+a.phase+'.json'),{str(f.relative_to(a.output)):dict(bytes=f.stat().st_size,sha256=c.digest(f)) for f in a.output.rglob('*') if f.is_file() and f.name!=('files-'+a.phase+'.json')})


if __name__=='__main__':main()
