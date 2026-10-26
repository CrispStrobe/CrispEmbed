#!/usr/bin/env python3
"""Hosted-only native 1.5M replay; pending original pins block all model access."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import shutil
import sys
import zipfile

from contract import read_contract, tensor_schema, validate_checkpoint
from native_contract import ROOT, PATH, check_original, dependencies, digest, protocol, skiplist_ids, source
from native_diff_adapter import collect as collect_ref, write_pair


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], timeout=120))


def artifact(run, pin, target):
    actual = api('actions/artifacts/' + str(pin['id']))
    if (actual['workflow_run']['id'] != run or actual['workflow_run']['head_sha'] != dependencies()['source'] or
            actual['expired'] or actual['size_in_bytes'] != pin['bytes'] or actual['digest'] != 'sha256:' + pin['sha256']):
        raise ValueError('Reviewed artifact service lineage differs')
    archive = target.with_suffix('.zip')
    with archive.open('xb') as stream:
        subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/artifacts/' + str(pin['id']) + '/zip'],
                       stdout=stream, check=True, timeout=600)
    if archive.stat().st_size != pin['bytes'] or digest(archive) != pin['sha256']:
        raise ValueError('Actual retained ZIP identity differs')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if (len(names) != len(set(names)) or sum(x.file_size for x in z.infolist()) > 512 * 1024**2 or
                any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
            raise ValueError('Unsafe original artifact')
        z.extractall(target)
    return actual


def run(args, log, cwd=None):
    with log.open('w') as stream:
        subprocess.run([str(a) for a in args], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)


def typed_fixtures(binary, directory, logs):
    """Independent writer/schema creates positive and malformed loader inputs."""
    import numpy as np
    import gguf
    cfg = read_contract()['variants']['1.5M']
    arch = 'modernbert_hash_moe_v10'
    from convert_f32 import ARCH
    if ARCH != arch:
        raise ValueError('Original conversion architecture changed')
    rows = tensor_schema(cfg['config'])
    for mutation, expected in [('valid', 'OK'), ('missing', 'Exact 32'), ('nonfinite', 'Nonfinite weight'),
                               ('wrong_type', 'Only lossless F32'), ('wrong_hidden', 'Unsupported 1.5M'),
                               ('unknown', 'Unknown/duplicate tensor'), ('wrong_shape', 'Tensor shape differs')]:
        path = directory / (mutation + '.gguf')
        writer = gguf.GGUFWriter(str(path), arch=arch)
        for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                    'hash_projections', 'max_position_embeddings', 'num_routed_experts', 'num_shared_experts', 'expert_dim', 'top_k']:
            writer.add_uint32(arch + '.' + key, 129 if key == 'hidden_size' and mutation == 'wrong_hidden' else cfg['config'][key])
        for key, value in [('head_dim',64), ('out_dim',128), ('query_marker_token_id',30000), ('document_marker_token_id',30001), ('mask_token_id',4)]:
            writer.add_uint32(arch + '.' + key, value)
        for key, value in [('rope_theta',160000.), ('norm_eps',1e-5), ('hash_norm_eps',1e-5), ('router_renorm_eps',1e-9)]:
            writer.add_float32(arch + '.' + key, value)
        for key, value in [('hidden_activation','gelu_erf'), ('quant_recipe','f32'), ('source_revision',cfg['original']['revision']),
                           ('tokenizer_sidecar_git_blob_sha1',cfg['original']['files']['tokenizer.json']['git_blob_sha1']),
                           ('tokenizer_sidecar_sha256','0'*64)]:
            writer.add_string(arch + '.' + key,value)
        for index, row in enumerate(rows):
            if mutation == 'missing' and index == 0:
                continue
            shape = list(row['shape'])
            if row['transform'] == 'expert_transpose':
                shape[-2:] = reversed(shape[-2:])
            if mutation == 'wrong_shape' and index == 0:
                shape[-1] -= 1
            values = np.zeros(shape, dtype=np.float32)
            if mutation == 'nonfinite' and index == 0:
                values.flat[0] = np.nan
            if mutation == 'wrong_type' and index == 0:
                values = values.astype(np.float16)
            writer.add_tensor('not.a.tensor' if mutation == 'unknown' and index == 0 else row['target'], values)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        run([binary, '--load', path, expected], logs / ('loader-' + mutation + '.log'))


def stages(path):
    import numpy as np
    result, records = {}, {}
    with (path / 'stages.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name = row['file']
            if PurePosixPath(name).name != name or row['stage'] in result:
                raise ValueError('Unsafe/duplicate native stage')
            shape = tuple(map(int,row['shape'].split(',')))
            dtype = '<f4' if row['type'] == 'f32' else '<i8'
            data = np.fromfile(path / name, dtype=dtype)
            if data.nbytes != int(row['bytes']) or data.size != __import__('math').prod(shape):
                raise ValueError('Native shape/bytes differ')
            result[row['stage']] = data.reshape(shape)
            if row['type'] == 'f32':
                records[row['stage']] = {'path':path/name,'shape':shape,'sha256':digest(path/name),
                                          'label':path.name+'/'+name}
    return result, records


def compare(a, b, *, integer=False):
    import numpy as np
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Actual stage shape/finiteness differs')
    if integer:
        exact=bool(np.array_equal(a,b))
        return {'exact':exact,'passed':exact}
    difference=b.astype(np.float64)-a.astype(np.float64)
    delta = float(np.max(np.abs(difference),initial=0))
    left, right = a.astype(np.float64).reshape(-1,a.shape[-1]), b.astype(np.float64).reshape(-1,b.shape[-1])
    n1,n2=np.linalg.norm(left,axis=1),np.linalg.norm(right,axis=1)
    cos=np.divide((left*right).sum(axis=1),n1*n2,out=np.zeros_like(n1),where=(n1*n2)!=0)
    cos[(n1==0)&(n2==0)]=1
    worst=float(cos.min(initial=1))
    result={'shape':list(a.shape),'elements':int(a.size),'max_absolute':delta,'min_row_cosine':worst,'mean_absolute':float(np.abs(difference).mean()),
            'mean_error':float(difference.mean()),'rms':float(np.sqrt((difference*difference).mean())),
            'original_l2':float(np.linalg.norm(a.astype(np.float64).ravel())),
            'native_l2':float(np.linalg.norm(b.astype(np.float64).ravel())),
            'passed':delta<=protocol()['gates']['max_absolute'] and worst>=protocol()['gates']['min_row_cosine']}
    for name,value in [('original',a),('native',b)]:
        result.update({name+'_min':float(value.min()),name+'_max':float(value.max()),
                       name+'_abs_min':float(np.abs(value).min()),name+'_abs_max':float(np.abs(value).max())})
    return result


def actual_ref(binary, original, native, report, log, independent):
    """Numeric rc=1 is collected; only structural/process errors interrupt."""
    env={k:v for k,v in os.environ.items() if k not in ('CRISPEMBED_DIFF_COS_THRESHOLD','CRISPEMBED_DIFF_DUMP_DIR')}
    with log.open('w') as stream:
        completed=subprocess.run([str(binary),str(original),str(native),str(report)],stdout=stream,
            stderr=subprocess.STDOUT,env=env,timeout=1200,check=False)
    return collect_ref(completed.returncode,report,independent)


def comparator_controls(binary, directory):
    """Hosted tiny actual-Ref positive/numeric-negative controls, not model gold."""
    import numpy as np
    records=[]
    original=np.asarray([[1.,0.],[0.,1.],[1.,0.]],dtype='<f4')
    for name,expected_exit in [('same',0),('changed_row',1)]:
        output=directory/name;output.mkdir(parents=True)
        native=original.copy()
        if name=='changed_row':native[-1]=[0.,1.]
        specs=[]
        for label,value in [('original',original),('native',native)]:
            path=output/(label+'.f32');path.write_bytes(value.tobytes())
            specs.append({'path':path,'shape':list(value.shape),'sha256':digest(path),'label':label+'.f32'})
        adapters=write_pair(output/'original.gguf',output/'native.gguf',
                            [{'name':'control','original':specs[0],'native':specs[1]}])
        measured={'control':compare(original,native)}
        result=actual_ref(binary,output/'original.gguf',output/'native.gguf',output/'diff.json',output/'diff.log',measured)
        if (result['exit_code']!=expected_exit or not result['all_metrics_agree'] or
                measured['control']['passed']!=(expected_exit==0)):
            raise ValueError('Actual Ref positive/numeric-negative control failed')
        records.append({'control':name,'expected_exit':expected_exit,'actual':result,
                        'metrics':measured,'adapters':adapters})
    write(directory/'controls.json',records)
    return records


def stage_map():
    mapping={'embedding.lookup':'backbone.embeddings.hash_embeddings.output#0',
             'embedding.projection':'backbone.embeddings.projection.output#0',
             'embedding.norm':'backbone.embeddings.norm.output#0'}
    for i in range(2):
        n=f'layer.{i}.'; b=f'backbone.layers.{i}.'
        for native,original in [('attn.norm','attention.norm'),('attn.q','attention.Q_proj'),
                                ('attn.k','attention.K_proj'),('attn.v','attention.V_proj')]:
            mapping[n+native]=b+original+'.output#0'
        for native,original in [('attn.q_rope',f'actual.sdpa.{i}.input.args.0'),
                                ('attn.k_rope_repeat',f'actual.sdpa.{i}.input.args.1'),
                                ('attn.v_repeat',f'actual.sdpa.{i}.input.args.2'),
                                ('attn.sdpa',f'actual.sdpa.{i}.output')]:
            mapping[n+native]=original
        for native,original in [('attn.output_projection','attention.O_proj'),
                                ('attn.residual','attention'),('ffn.norm','ffn_norm'),('ffn.router','ffn.router')]:
            mapping[n+native]=b+original+'.output#0'
        for native,source_suffix in [('router_probabilities','probabilities'),('topk_indices','topk_indices'),
                                     ('topk_weights','renormalized_weights'),('selection_margin','selection_margin')]:
            mapping[n+'ffn.'+native]=f'derived.layers.{i}.ffn.router.{source_suffix}#0'
        for native,original in [('ffn.shared_gate','ffn.shared_gate'),('ffn.shared_up','ffn.shared_up'),
                                ('ffn.shared_down','ffn.shared_down'),('ffn.output','ffn')]:
            mapping[n+native]=b+original+'.output#0'
        mapping[n+'output']=f'backbone.layers.{i}.output#0'
    mapping.update(final_norm='backbone.norm.output#0',dense='pipeline.1.output#0.token_embeddings',
                   normalized='pipeline.3.output#0.token_embeddings')
    return mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['preflight','run'],default='preflight')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required; preserve earlier failures')
    args.output.mkdir(parents=True)
    report={'schema':'crispembed-moedern-native-1p5m-v1','completed':False,'cases':[],
            'qualification':'Experimental token-ID numerical panel only; no public tokenizer/API/retrieval/release qualification'}
    try:
        pin=dependencies(); report['source']=source();report['protocol_sha256']=digest(PATH)
        if args.phase=='preflight':
            report['status']='reviewed dependencies and source passed; no model access';return
        if os.environ.get('GITHUB_ACTIONS')!='true':
            raise ValueError('All builds/model access hosted only')
        from importlib.metadata import version
        if version('numpy')!='2.2.6' or version('gguf')!='0.17.1':
            raise ValueError('Pinned array/fixture environment differs')
        service=api('actions/runs/'+str(pin['run']))
        if (service['head_sha']!=pin['source'] or service['status']!='completed' or service['conclusion']!='success' or
                service['event']!='workflow_dispatch' or service['path'] not in (
                    '.github/workflows/ovisocr2-metadata.yml','.github/workflows/moedern-colbert-original.yml')):
            raise ValueError('Successful immutable reviewed original run required')
        jobs=api('actions/runs/'+str(pin['run'])+'/jobs?per_page=100')['jobs']
        original_jobs=[j for j in jobs if j['name'] in ('original','moedern-original-1p5m / original')]
        if len(original_jobs)!=1 or original_jobs[0]['conclusion']!='success':
            raise ValueError('Actual original producer job did not succeed')
        scratch=Path(os.environ['RUNNER_TEMP'])/'moedern-native-scratch';scratch.mkdir(exist_ok=False)
        report['original_service']=service['html_url'];report['original_artifacts']={}
        for key in ['review_artifact','raw_artifact']:
            report['original_artifacts'][key]=artifact(pin['run'],pin[key],scratch/key)
        reviews=list((scratch/'review_artifact').rglob('original.json'))
        if len(reviews)!=1 or digest(reviews[0])!=pin['receipt_sha256']:
            raise ValueError('Reviewed original receipt bytes differ')
        original=json.loads(reviews[0].read_text());panel=check_original(original)
        raw_reports=list((scratch/'raw_artifact').rglob('original.json'))
        if len(raw_reports)!=1 or raw_reports[0].read_bytes()!=reviews[0].read_bytes():
            raise ValueError('Raw/report original receipt mismatch')
        raw=raw_reports[0].parent.parent; checkpoint=raw/'checkpoint'
        identities,_=validate_checkpoint(checkpoint,'1.5M')
        if identities!=original['checkpoint_files']:
            raise ValueError('Actual original checkpoint differs')
        model=raw/'model-F32.gguf';conversion=raw/'model-F32.conversion.json'
        if digest(model)!=pin['gguf_sha256'] or digest(conversion)!=pin['conversion_receipt_sha256']:
            raise ValueError('Reviewed lossless conversion differs')
        conversion_report=json.loads(conversion.read_text())
        if conversion_report.get('completed') is not True or conversion_report['original_report_sha256']!=pin['receipt_sha256'] or conversion_report['gguf_sha256']!=pin['gguf_sha256'] or conversion_report['tensor_count']!=32:
            raise ValueError('Original-first conversion lineage differs')
        for row in original['stage_arrays']:
            p=PurePosixPath(row['path'])
            if p.is_absolute() or '..' in p.parts or '\\' in row['path']:
                raise ValueError('Unsafe original raw-array path')
            path=raw/row['path']
            if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
                raise ValueError('Actual original array bytes changed')
        # All original outputs are now authenticated, before native build/forward.
        build=scratch/'build'
        run(['cmake','-S',ROOT/'tests/moedern-native','-B',build,'-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],args.output/'configure.log')
        run(['cmake','--build',build,'--parallel','2'],args.output/'build.log')
        controls=build/'moedern-native-controls';probe=build/'moedern-native-probe';diff=build/'moedern-native-diff'
        report['binaries']={p.name:digest(p) for p in (controls,probe,diff)}
        shutil.copyfile(build/'compile_commands.json',args.output/'compile_commands.json')
        report['comparator_controls']=comparator_controls(diff,args.output/'comparator-controls')
        run([controls],args.output/'controls.log')
        fixture=scratch/'typed-fixtures';fixture.mkdir();typed_fixtures(controls,fixture,args.output)
        import numpy as np
        skip=skiplist_ids(json.loads((checkpoint/'tokenizer.json').read_text()),
            json.loads((checkpoint/'config_sentence_transformers.json').read_text()),
            json.loads((checkpoint/'tokenizer_config.json').read_text()))
        selected={}; failures=[]; records={(r['case'],r['stage']):r for r in original['stage_arrays']}
        if len(records)!=len(original['stage_arrays']):
            raise ValueError('Duplicate original stage')
        def old(cid,name):
            row=records[cid,name];dtype={'f32':'<f4','i64':'<i8','u8':'u1'}[row['dtype']]
            return np.fromfile(raw/row['path'],dtype=dtype).reshape(row['shape'])
        def original_file(cid,name):
            row=records[cid,name]
            if row['dtype']!='f32':
                raise ValueError('F32 adapters cannot cast original integer/boolean data')
            return {'path':raw/row['path'],'shape':row['shape'],'sha256':row['sha256'],'label':row['path']}
        for case in original['cases']:
            cid=case['id'];inputs=scratch/(cid+'.tokens');tokens=case['token_ids'];mask=case['supplied_attention_mask']
            inputs.write_text('MOEDERN_TOKENS_V1 '+case['role']+' '+str(len(tokens))+'\nids '+ ' '.join(map(str,tokens))+
                '\nmask '+' '.join(map(str,map(int,mask)))+'\nskip '+str(len(skip))+' '+' '.join(map(str,skip))+'\n')
            output=args.output/cid;run([probe,model,inputs,output],args.output/(cid+'.log'))
            observed,native_files=stages(output);checks={};pairs=[]
            row={'id':cid,'role':case['role'],'tokens':len(tokens),'checks':checks,'first_captured_divergence':None,
                 'first_divergence_record':None}
            report['cases'].append(row)
            def check(label,a,b,integer=False,original_source=None,native_source=None):
                try:
                    measured=compare(a,b,integer=integer)
                except Exception as exc:
                    raise ValueError(cid+' stage '+label+': '+str(exc)) from exc
                checks[label]=measured
                if not integer:
                    if original_source is None or native_source is None:
                        raise ValueError('Every mapped F32 stage requires actual lossless Ref inputs: '+label)
                    pairs.append({'name':label,'original':original_source,'native':native_source})
                return measured
            check('input_ids',np.asarray(tokens,dtype=np.int64)[None,:],observed['input_ids'],True)
            check('supplied_mask',np.asarray(mask,dtype=np.int64)[None,:],observed['supplied_mask'],True)
            check('effective_mask',np.asarray(case['effective_attention_mask'],dtype=np.int64)[None,:],observed['effective_mask'],True)
            computed_mask=np.fromfile(output/'scoring.u8',dtype='u1')
            check('scoring_mask',np.asarray(case['scoring_mask']),computed_mask,True)
            if cid in panel['full_capture']:
                for native,name in stage_map().items():
                    integer=native.endswith('topk_indices')
                    check(native,old(cid,name),observed[native],integer,
                          original_file(cid,name) if not integer else None,native_files.get(native))
                derived_mask=np.broadcast_to(observed['effective_mask'].astype(bool)[:,None,None,:],(1,1,len(tokens),len(tokens)))
                for layer in range(2):
                    check(f'layer.{layer}.derived_attention_mask',old(cid,f'actual.sdpa.{layer}.input.kwargs.attn_mask'),derived_mask,True)
            all_vectors=np.fromfile(output/'all.f32',dtype='<f4').reshape(len(tokens),128)
            check('all_vectors',old(cid,'ordinary.features.token_embeddings'),all_vectors,
                  original_source=original_file(cid,'ordinary.features.token_embeddings'),
                  native_source={'path':output/'all.f32','shape':list(all_vectors.shape),
                                 'sha256':digest(output/'all.f32'),'label':cid+'/all.f32'})
            selected[cid]=np.fromfile(output/'selected.f32',dtype='<f4').reshape(-1,128)
            check('selected',old(cid,'ordinary.selected'),selected[cid],
                  original_source=original_file(cid,'ordinary.selected'),
                  native_source={'path':output/'selected.f32','shape':list(selected[cid].shape),
                                 'sha256':digest(output/'selected.f32'),'label':cid+'/selected.f32'})
            original_adapter,native_adapter=output/'original-reference.gguf',output/'native-reference.gguf'
            row['lossless_adapters']=write_pair(original_adapter,native_adapter,pairs)
            write(output/'diff-adapters.json',row['lossless_adapters'])
            row['actual_crispembed_diff']=actual_ref(diff,original_adapter,native_adapter,output/'diff.json',
                output/'diff.log',{p['name']:checks[p['name']] for p in pairs})
            for name,measured in checks.items():
                if not measured['passed']:
                    failure={'case':cid,'stage':name,**measured};failures.append(failure)
                    if row['first_captured_divergence'] is None:
                        row['first_captured_divergence']=name;row['first_divergence_record']=failure
            report['failures']=failures
            write(args.output/'native.json',report)
        scores=[]
        for score in original['scores']:
            q,d=selected[score['query']],selected[score['document']]
            independent=float((q@d.T).max(axis=1).sum())
            scorefile=args.output/(score['query']+'--'+score['document']+'.f32')
            run([probe,'--maxsim',args.output/score['query']/'selected.f32',args.output/score['document']/'selected.f32',scorefile],scorefile.with_suffix('.log'))
            actual_array=np.fromfile(scorefile,dtype='<f4')
            if actual_array.shape!=(1,) or not np.isfinite(actual_array).all():
                raise ValueError('Invalid actual native MaxSim result')
            actual=float(actual_array[0])
            passed=max(abs(actual-score['original_maxsim']),abs(actual-independent))<=protocol()['gates']['maxsim_max_absolute']
            measured={'query':score['query'],'document':score['document'],'original':score['original_maxsim'],
                      'native':actual,'independent':independent,'passed':passed}
            scores.append(measured)
            if not passed:
                failures.append({'stage':'maxsim',**measured})
        report.update(completed=not failures,status='bounded token-ID F32 native panel passed' if not failures else 'numerical failure; full panel retained',
            scores=scores,failures=failures,checkpoint=identities,skiplist_ids=skip,gguf_sha256=digest(model),original_receipt_sha256=pin['receipt_sha256'])
        if failures:
            raise ValueError('Unchanged native numerical budgets failed; all completed case/stage/score results retained')
    except Exception as e:
        if report.get('status')!='numerical failure; full panel retained':
            report['status']='rejected structural/source contract'
        report['error']=type(e).__name__+': '+str(e);raise
    finally:
        report['files']={str(p.relative_to(args.output)):{'bytes':p.stat().st_size,'sha256':digest(p)}
            for p in args.output.rglob('*') if p.is_file() and p.name!='native.json'}
        write(args.output/'native.json',report)


if __name__=='__main__':
    main()
