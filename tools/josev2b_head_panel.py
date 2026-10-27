#!/usr/bin/env python3
"""Hosted head-only replay; no network, install, build or original regeneration."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_head_schema import HEAD_CONFIG,tensor_schema,reference,sha256
from josev2b_head_native import validate_manifest,validate_events
import josev2b_head_diff_adapter as adapter
SPEC=importlib.util.spec_from_file_location('head_artifact',ROOT/'tools/ci-heavy/josev2b_head_artifact.py')
artifact=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(artifact)
RECEIPT=artifact.RECEIPT
SOURCE=artifact.SOURCE
FROZEN_HEAD='d8b99a9c8e3a5cf0f6699fc026d3396d26e03763'
GGML='890278a8342c620197c90e702e1188bcab94f510'
AUTHOR_SHA='0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'
FROZEN_FILES=('src/josev2b_head.cpp','src/josev2b_head.h','tests/josev2b_head_observer.h',
              'tests/test_josev2b_head_probe.cpp','docs/josev2b-head-native-contract.json',
              'tools/josev2b_head_native.py','src/core/gguf_loader.cpp','src/core/gguf_loader.h',
              'src/core/env_gate.h','src/core/ggml_cpu_backend.h','src/crispembed_diff.h')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def member(path,declaration):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size!=declaration['bytes'] or
            sha256(path)!=declaration['sha256']):raise ValueError('Artifact file identity differs: '+path.name)


def hosted(backends):
    if list(backends)==['cpu'] and os.environ.get('GITHUB_ACTIONS')=='true':return {'service':'GitHub Actions','backends':['cpu']}
    if list(backends)!=['cpu','CUDA0'] or not Path('/kaggle/working').is_dir() or not Path('/kaggle/input').is_dir():
        raise ValueError('CPU-only GHA or genuine CPU+CUDA0 GPU task required')
    rows=[r.split(',') for r in subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,compute_cap,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()]
    if not rows or any(len(r)!=5 for r in rows):raise ValueError('Actual CUDA hardware proof unavailable')
    for _,cap,total,free,driver in rows:
        if tuple(map(int,cap.strip().split('.')))<(7,5) or tuple(map(int,driver.strip().split('.')))<(580,65,6) or int(total)<14336 or int(free)<11264:
            raise ValueError('Approved CUDA13 hardware/resource bounds differ; no retry')
    return {'service':'Kaggle GPU','hardware':[[s.strip() for s in r] for r in rows],'backends':list(backends)}


def source_and_build(build,out):
    identities={}
    measured=(*FROZEN_FILES,'tools/josev2b_head_panel.py','tools/josev2b_head_diff_adapter.py',
        'tools/ci-heavy/josev2b_head_artifact.py','tools/josev2b_head_schema.py',
        'tests/josev-head/CMakeLists.txt','tests/josev-head/diff.cpp')
    if subprocess.check_output(['git','status','--porcelain','--',*measured],cwd=ROOT):
        raise ValueError('Dirty/untracked measured panel source')
    for name in FROZEN_FILES:
        old=subprocess.check_output(['git','show',FROZEN_HEAD+':'+name],cwd=ROOT)
        if (ROOT/name).read_bytes()!=old:raise ValueError('Frozen native head source differs: '+name)
        identities[name]=hashlib.sha256(old).hexdigest()
    if not (ROOT/'ggml/.git').exists():raise ValueError('Explicit initialized ggml890 checkout required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'ggml',text=True).strip()!=GGML:
        raise ValueError('Actual ggml checkout differs')
    if subprocess.check_output(['git','diff','--exit-code','HEAD'],cwd=ROOT/'ggml'):
        raise ValueError('Dirty ggml source')
    commands=json.loads((build/'compile_commands.json').read_text())
    head=[r for r in commands if Path(r['file']).resolve()==ROOT/'src/josev2b_head.cpp']
    if len(head)!=2:raise ValueError('Both ordinary and observer head translations required')
    states=[]
    for row in head:
        args=row.get('arguments') or shlex.split(row['command'])
        states.append(any(a in ('-DCRISPEMBED_JOSEV_HEAD_TESTING','-DCRISPEMBED_JOSEV_HEAD_TESTING=1') for a in args))
        if any(a in ('-ffast-math','-Ofast','--use_fast_math') for a in args):raise ValueError('Unexpected head fast-math compile mode')
    if sorted(states)!=[False,True]:raise ValueError('Ordinary/observer compilation scopes differ')
    plain=build/'libjosev-head-plain.a';observed=build/'libjosev-head-observed.a';probe=build/'test-josev2b-head';diff=build/'josev-head-diff'
    for path in (plain,observed,probe,diff):
        if not path.is_file() or path.is_symlink():raise ValueError('Exact standalone build products required')
    symbols=subprocess.check_output(['nm','-C',str(plain)],text=True,timeout=30)
    strings=subprocess.check_output(['strings',str(plain)],text=True,timeout=30)
    observed_symbols=subprocess.check_output(['nm','-C',str(observed)],text=True,timeout=30)
    if any(x in symbols for x in ('run_observed','metadata_controls','test::stage')) or any(x in strings for x in ('head/hidden_norm','head/evidence/','head/residual_scorer')):
        raise ValueError('Reference probes leaked into ordinary archive')
    if any(x not in symbols for x in ('josev2b_head::init(', 'josev2b_head::run(', 'josev2b_head::free(')) or any(x not in observed_symbols for x in ('run_observed','metadata_controls')):
        raise ValueError('Expected ordinary/observer component symbols missing')
    (out/'plain-symbols.txt').write_text(symbols);(out/'plain-strings.txt').write_text(strings)
    (out/'observer-symbols.txt').write_text(observed_symbols)
    (out/'compile_commands.json').write_bytes((build/'compile_commands.json').read_bytes())
    return dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_head=FROZEN_HEAD,ggml=GGML,frozen_files=identities,observer_excluded=True,
        harness_files={n:sha256(ROOT/n) for n in measured},
        products={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in (plain,observed,probe,diff)})


def verify_head(head,report,cfg):
    import numpy as np
    import gguf
    conversion_path=head.with_suffix('.conversion.json');conversion=json.loads(conversion_path.read_text())
    if (conversion.get('schema')!='crispembed-josev-head-f32-conversion-v1' or conversion.get('completed') is not True or
            conversion.get('full_model') is not False or conversion.get('original_receipt_sha256')!=RECEIPT or
            conversion.get('original_worker_source')!=SOURCE or conversion.get('checkpoint_sha256')!=cfg['files']['joint_head.safetensors']['sha256'] or
            conversion.get('tensor_count')!=122 or conversion.get('parameter_count')!=30472708 or
            conversion.get('native_parity') is not False or head.stat().st_size!=conversion['gguf_bytes'] or sha256(head)!=conversion['gguf_sha256']):
        raise ValueError('Completed exact original-first head conversion required')
    rows=conversion['tensors'];expected=tensor_schema();proof=report['weights']['head_tensors']
    if [r['source'] for r in rows]!=list(expected) or len(rows)!=122:raise ValueError('Complete ordered conversion tensor proof required')
    reader=gguf.GGUFReader(str(head))
    if len(reader.tensors)!=122:raise ValueError('Actual head requires exactly122 tensors')
    actual={t.name:t for t in reader.tensors}
    if set(actual)!={'head.'+n for n in expected}:raise ValueError('Actual F32 head tensor names differ')
    for row in rows:
        name=row['source'];shape=expected[name] or [1];tensor=actual['head.'+name]
        source_dtype='BF16' if proof[name]['checkpoint_dtype'] in ('BF16','bfloat16','torch.bfloat16') else 'F32'
        payload=tensor.data.astype('<f4',copy=False).tobytes(order='C')
        if (row['target']!='head.'+name or row['source_shape']!=expected[name] or row['gguf_shape_numpy']!=shape or
                row['f32_sha256']!=proof[name]['loaded_f32_sha256'] or row['source_dtype']!=source_dtype or
                row['transformation']!=('exact BF16 promotion' if source_dtype=='BF16' else 'identity') or
                row['scalar_adapter']!=('rank-zero to one-element vector' if not expected[name] else None) or
                tensor.tensor_type!=gguf.GGMLQuantizationType.F32 or list(tensor.shape)!=list(reversed(shape)) or
                len(payload)!=4*math.prod(shape) or hashlib.sha256(payload).hexdigest()!=proof[name]['loaded_f32_sha256'] or
                not np.isfinite(tensor.data).all()):raise ValueError('Actual head bytes/shape differ from loaded original: '+name)
    return conversion


def verify_evidence(reference_dir,head,input_dir):
    # Exact original before optional payload readers. No reference regeneration.
    reference(reference_dir/'reference.json',RECEIPT,SOURCE)
    report,cfg,freeze,events=artifact.verify_original(reference_dir)
    conversion=verify_head(head,report,cfg)
    root=head.parent.parent;review=root/'review';prepared=json.loads((review/'artifact.json').read_text())
    if (prepared.get('schema')!='crispembed-josev-head-artifact-v1' or prepared.get('completed') is not True or
            prepared.get('reference_sha256')!=RECEIPT or prepared.get('original_source')!=SOURCE or
            prepared.get('arrays')!=251 or prepared.get('array_payload_bytes')!=279087860 or
            prepared.get('gguf_sha256')!=conversion['gguf_sha256'] or prepared.get('gguf_bytes')!=conversion['gguf_bytes'] or
            prepared.get('tensors')!=122 or prepared.get('full_model') is not False or prepared.get('native_parity') is not False):
        raise ValueError('Exact original/head/input artifact completion proof required')
    files=prepared['files']
    for name,path in [('head/'+head.name,head),('head/'+head.with_suffix('.conversion.json').name,head.with_suffix('.conversion.json')),
                      ('review/head-conversion.json',review/'head-conversion.json'),('review/ordered-events.json',review/'ordered-events.json'),
                      ('review/native-input-adapters.json',review/'native-input-adapters.json')]:member(path,files[name])
    if (review/'head-conversion.json').read_bytes()!=head.with_suffix('.conversion.json').read_bytes():raise ValueError('Conversion receipt copies differ')
    if json.loads((review/'ordered-events.json').read_text())!=events:raise ValueError('All251 ordered original events differ')
    proofs=json.loads((review/'native-input-adapters.json').read_text())
    if [p['case'] for p in proofs]!=[r['id'] for r in freeze['records']]:raise ValueError('Complete ordered input adapter cases required')
    import numpy as np
    manifests={}
    for record,proof in zip(freeze['records'],proofs):
        case=record['id'];target=input_dir/case;path=target/'input.json';member(path,files['inputs/'+case+'/input.json'])
        manifest=validate_manifest(json.loads(path.read_text()));fields=record['encoded']['questions']
        if (proof['reference_sha256']!=RECEIPT or manifest['tokens']!=len(record['encoded']['input_ids']) or
                len(manifest['fields'])!=len(fields)):raise ValueError('Original input span/token/field scope differs')
        expected=[(manifest['hidden_f32'],'head/hidden_norm/input/0',0,[1,manifest['tokens'],2048])]
        for index,(field,q) in enumerate(zip(manifest['fields'],fields)):
            if ({k:v for k,v in field.items() if k!='lexical_f32'}!=dict(id=q['question_id'],type=q['question_type'],question_start=q['question_span'][0],question_end=q['question_span'][1],options=[dict(start=a,end=b) for a,b in q['option_spans']])):
                raise ValueError('Exact original field type/order/spans required')
            expected.append((field['lexical_f32'],'head/option_lexical_projection/input/0',index,[len(q['option_spans']),2048]))
        if set(proof['files'])!={row[0] for row in expected}:raise ValueError('Exact hidden/lexical input file coverage required')
        for filename,stage,occurrence,shape in expected:
            rows=[e for e in events if (e['case'],e['stage'],e['occurrence'])==(case,stage,occurrence)]
            if len(rows)!=1:raise ValueError('Original input source event absent/ambiguous')
            row=rows[0];p=proof['files'][filename];raw=target/filename
            member(raw,files['inputs/'+case+'/'+filename]);member(raw,p)
            original=np.load(reference_dir/artifact.safe_member(row['file']),allow_pickle=False,mmap_mode='r')
            if (row['shape']!=shape or p['original_shape']!=shape or p['original_array']!=row['file'] or
                    p['original_array_sha256']!=row['sha256'] or p['stage']!=stage or p['occurrence']!=occurrence or
                    p['storage_adapter']!='C-order bytes unchanged' or original.dtype!=np.dtype('<f4') or
                    original.shape!=tuple(shape) or raw.read_bytes()!=original.tobytes(order='C')):
                raise ValueError('Lossless actual original input adapter required')
        manifests[case]=manifest
    author=head.parent/'joint_schema_model.py';member(author,files['head/joint_schema_model.py'])
    if sha256(author)!=AUTHOR_SHA:raise ValueError('Pinned publisher answer source required')
    return report,freeze,events,manifests,conversion,author.read_bytes()


def _compile_answer(function):
    # Used only after the complete pinned module hash is checked below. Preserve
    # the pure publisher AST, including option order/ties/rounding semantics.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
    ast.fix_missing_locations(module);scope={}
    exec(compile(module,'pinned-systemone-answer','exec'),scope)
    return scope['systemone_answer']


def official_answer(source):
    if hashlib.sha256(source).hexdigest()!=AUTHOR_SHA:raise ValueError('Only exact pinned publisher answer function allowed')
    tree=ast.parse(source);functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='systemone_answer']
    if len(functions)!=1 or functions[0].decorator_list:raise ValueError('Unique undecorated original answer function required')
    return _compile_answer(functions[0])


def metrics(original,native):
    import numpy as np
    if original.dtype!=np.dtype('<f4') or native.dtype!=np.dtype('<f4') or original.shape!=native.shape or not original.size or not np.isfinite(original).all() or not np.isfinite(native).all():
        raise ValueError('Exact finite F32 comparison shape required; no original reshape')
    a=original.astype(np.float64);b=native.astype(np.float64);error=b-a
    # Derived contiguous rows only; original shape/bytes remain unchanged.
    ar=a.reshape(-1,a.shape[-1]);br=b.reshape(-1,b.shape[-1]);aa=np.sum(ar*ar,axis=1);bb=np.sum(br*br,axis=1);dot=np.sum(ar*br,axis=1)
    cos=np.where((aa>1e-18)&(bb>1e-18),dot/np.maximum(np.sqrt(aa*bb),1e-300),np.where((aa<=1e-18)&(bb<=1e-18),1.,0.))
    row=dict(shape=list(original.shape),elements=original.size,max_absolute=float(np.max(abs(error))),mean_absolute=float(np.mean(abs(error))),
        mean_error=float(np.mean(error)),rms=float(np.sqrt(np.mean(error*error))),min_row_cosine=float(np.min(cos)),
        original_l2=float(np.linalg.norm(a.ravel())),native_l2=float(np.linalg.norm(b.ravel())),original_min=float(a.min()),original_max=float(a.max()),
        original_abs_min=float(abs(a).min()),original_abs_max=float(abs(a).max()),native_min=float(b.min()),native_max=float(b.max()),
        native_abs_min=float(abs(b).min()),native_abs_max=float(abs(b).max()))
    row['passed']=row['max_absolute']<=0.002 and row['min_row_cosine']>=0.99999
    return row


def execute(args,log,env,timeout):
    with log.open('w') as stream:
        return subprocess.run([str(a) for a in args],stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout).returncode


def ref_controls(diff,out,env):
    import struct
    root=out/'ref-controls';root.mkdir();a=root/'original.f32';b=root/'bad.f32'
    a.write_bytes(struct.pack('<4f',0.0001,0.,1.,0.));b.write_bytes(struct.pack('<4f',-0.0001,0.,1.,0.))
    def spec(path):return dict(path=str(path),shape=[2,2],sha256=sha256(path))
    results=[]
    for kind,path,expected in [('positive',a,0),('bad-row',b,1)]:
        directory=root/kind;directory.mkdir();adapter.write_pair(directory/'original.gguf',directory/'native.gguf',[dict(name='stage_00000',original=spec(a),native=spec(path))])
        rc=execute([diff,directory/'original.gguf',directory/'native.gguf',directory/'comparison.json'],directory/'comparison.log',dict(env,CRISPEMBED_DIFF_COS_THRESHOLD='0.0001'),30)
        data=json.loads((directory/'comparison.json').read_text())
        if rc!=expected or data['status']!=('passed' if expected==0 else 'numeric_rejection') or len(data['stages'])!=1:
            raise ValueError('Actual Ref positive/bad-row control failed')
        if data['stages'][0]['found'] is not True or data['stages'][0]['elements']!=4 or data['stages'][0]['ggml_shape']!=[2,2]:raise ValueError('Ref control did not compare exact shape/elements')
        if kind=='positive' and (data['stages'][0]['max_absolute']!=0 or data['stages'][0]['min_row_cosine']!=1 or data['stages'][0]['passed'] is not True):raise ValueError('Actual identical Ref control differs')
        if kind=='bad-row' and (data['stages'][0]['max_absolute']>0.002 or data['stages'][0]['min_row_cosine']>=0.99999):
            raise ValueError('Ref bad-row control did not exercise fixed contiguous cosine gate')
        results.append(dict(case=kind,exit_code=rc,report=data))
    wrong=root/'wrong-shape';wrong.mkdir()
    adapter.write_gguf(wrong/'original.gguf',[dict(name='stage_00000',**spec(a))])
    adapter.write_gguf(wrong/'native.gguf',[dict(name='stage_00000',**{**spec(a),'shape':[1,4]})])
    rc=execute([diff,wrong/'original.gguf',wrong/'native.gguf',wrong/'comparison.json'],wrong/'comparison.log',env,30)
    proof=json.loads((wrong/'comparison.json').read_text())
    if rc!=2 or proof.get('status')!='structural_rejection':raise ValueError('Actual Ref wrong-shape control did not reject structurally')
    results.append(dict(case='wrong-shape',exit_code=rc,report=proof))
    return results


def run_head_panel(reference_dir,headGGUF,input_dir,build,out,backends):
    reference_dir,head,input_dir,build,out=(Path(v).resolve() for v in (reference_dir,headGGUF,input_dir,build,out))
    if out.exists():raise ValueError('Fresh panel output required')
    out.mkdir(parents=True);review=out/'review';review.mkdir()
    result=dict(schema='crispembed-josev-head-panel-v1',completed=False,component_parity=False,full_model=False,qualification=False,
        original_receipt_sha256=RECEIPT,original_worker=SOURCE,backends=list(backends),cases=[],structural_failures=[],numeric_failures=[],decoded_failures=[])
    def save():write(review/'panel.json',result)
    save();phase='service/source';started=time.monotonic();deadline=started+1800
    def budget(maximum):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Head panel30min execution bound reached; earlier results retained')
        return min(maximum,remaining)
    try:
        result['service']=hosted(backends);result['source']=source_and_build(build,review);save()
        for name,version in [('numpy','2.2.6'),('gguf','0.17.1')]:
            if importlib.metadata.version(name)!=version:raise ValueError('Exact hosted array/GGUF tooling required')
        phase='original/artifact/input-guards'
        report,freeze,events,manifests,conversion,author=verify_evidence(reference_dir,head,input_dir)
        result.update(original_arrays_verified=251,expected_head_stages_per_backend=123,conversion=conversion,inputs_verified=True)
        answer=official_answer(author);env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        env.update(CRISPEMBED_JOSEV_HEAD='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        phase='build-controls'
        controls=[]
        for flag in ('--contracts','--gates'):
            log=review/(flag[2:]+'.log');rc=execute([build/'test-josev2b-head',flag],log,env,60)
            if rc:raise ValueError('Actual built control failed: '+flag)
            controls.append(dict(flag=flag,exit_code=rc,log=log.name,sha256=sha256(log)))
        result['build_controls']=controls;result['ref_controls']=ref_controls(build/'josev-head-diff',out,env);save()
        import numpy as np
        phase='native-panel'
        for backend in backends:
            for case,record in zip(report['cases'],freeze['records']):
                case_id=case['id'];directory=out/backend/case_id;directory.mkdir(parents=True)
                row=dict(case=case_id,requested_backend=backend,status='pending',stages=[]);result['cases'].append(row);save()
                case_started=time.monotonic();print(json.dumps(dict(case=case_id,backend=backend,phase='begin')),flush=True)
                try:
                    probe=directory/'probe'
                    rc=execute([build/'test-josev2b-head',head,input_dir/case_id/'input.json',probe,backend,RECEIPT,SOURCE],directory/'probe.log',env,budget(600))
                    row['probe_exit_code']=rc
                    if rc:raise ValueError('Native probe structural/process rejection: '+str(rc))
                    proof=json.loads((probe/'probe.json').read_text());actual=[json.loads(line) for line in (probe/'stages.jsonl').read_text().splitlines()]
                    validate_events(manifests[case_id],actual)
                    originals=[e for e in events if e['case']==case_id and e['stage'].startswith(('head/','logits/','probabilities/'))]
                    if (proof.get('completed') is not True or proof.get('full_model') is not False or proof.get('qualification') is not False or
                            proof.get('observer_bit_exact') is not True or proof.get('repeat_bit_exact') is not True or
                            proof.get('input_failure_output_empty') is not True or proof.get('observer_failure_output_empty') is not True or
                            proof.get('post_failure_recovery_bit_exact') is not True or proof.get('stages')!=len(actual) or
                            proof.get('backend')!=('CPU' if backend=='cpu' else backend) or len(actual)!=len(originals) or
                            [f['id'] for f in proof.get('fields',[])]!=[q['question_id'] for q in record['encoded']['questions']]):
                        raise ValueError('Actual backend/observer/repeat/recovery/coverage proof differs')
                    pairs=[];independent={};native_probabilities={}
                    for index,(original,native) in enumerate(zip(originals,actual)):
                        if any(native[k]!=original[k] for k in ('stage','occurrence','shape')):raise ValueError('Original/native ordered occurrence/logical shape differs')
                        native_path=probe/native['file'];shape=original['shape']
                        if native_path.stat().st_size!=4*math.prod(shape):raise ValueError('Actual native stage extent differs')
                        a=np.load(reference_dir/artifact.safe_member(original['file']),allow_pickle=False,mmap_mode='r')
                        b=np.fromfile(native_path,dtype='<f4').reshape(tuple(native['shape']))
                        measured=metrics(a,b);name=f'stage_{index:05d}'
                        reference_raw=directory/(name+'-original.f32');reference_raw.write_bytes(a.tobytes(order='C'))
                        pairs.append(dict(name=name,original=dict(path=str(reference_raw),shape=shape,sha256=sha256(reference_raw)),
                                          native=dict(path=str(native_path),shape=native['shape'],sha256=sha256(native_path))))
                        measured.update(stage=original['stage'],occurrence=original['occurrence'],original_npy_sha256=original['sha256'],native_f32_sha256=sha256(native_path))
                        independent[name]=measured;row['stages'].append(measured)
                        if original['stage'].startswith(('logits/','probabilities/')):
                            kind,field_id=original['stage'].split('/',1);field_index=next(i for i,f in enumerate(proof['fields']) if f['id']==field_id)
                            prefix=proof['fields'][field_index]['raw_prefix']
                            if prefix!='field-'+str(field_index) or np.asarray(proof['fields'][field_index][kind],dtype='<f4').tobytes()!=native_path.read_bytes():raise ValueError('Probe final JSON/raw F32 identity differs')
                            for run in ('ordinary','repeat'):
                                path=probe/(prefix+'-'+run+'-'+kind+'.f32')
                                if path.read_bytes()!=native_path.read_bytes():raise ValueError('Independent ordinary/observer/repeat final bytes differ')
                            if kind=='probabilities':native_probabilities[field_id]=b.tolist()
                    row['probe']=proof
                    row['adapters']=adapter.write_pair(directory/'original.gguf',directory/'native.gguf',pairs)
                    rc=execute([build/'josev-head-diff',directory/'original.gguf',directory/'native.gguf',directory/'comparison.json'],directory/'comparison.log',env,budget(120))
                    row['cpp_ref']=adapter.collect(rc,directory/'comparison.json',independent)
                    if not row['cpp_ref']['all_metrics_agree']:raise ValueError('Actual Ref/independent NumPy metric agreement rejected')
                    original_answers=case['official_response']['answers'];native_answers={}
                    for question in record['encoded']['questions']:
                        field_id=question['question_id'];probabilities=dict(zip(question['option_ids'],native_probabilities[field_id]))
                        native_answers[field_id]=answer(record['request']['questions'][field_id],probabilities)
                    native_response=dict(model=record['request']['model'],answers=native_answers,usage=dict(input_tokens=manifests[case_id]['tokens'],output_tokens=0))
                    row['decoded']=dict(original_response=case['official_response'],native_response=native_response,original_answers=original_answers,native_answers=native_answers,exact=native_response==case['official_response'],
                        function_source_sha256=AUTHOR_SHA,method='Unmodified pinned publisher systemone_answer AST; actual native F32 probabilities')
                    row['numeric_passed']=all(s['passed'] for s in row['stages'])
                    row['status']='passed' if row['numeric_passed'] and row['decoded']['exact'] else 'numeric_or_decoded_rejection'
                    if not row['numeric_passed']:result['numeric_failures'].append(dict(case=case_id,backend=backend,stages=[dict(stage=s['stage'],occurrence=s['occurrence'],max_absolute=s['max_absolute'],min_row_cosine=s['min_row_cosine']) for s in row['stages'] if not s['passed']]))
                    if not row['decoded']['exact']:result['decoded_failures'].append(dict(case=case_id,backend=backend))
                except Exception as error:
                    row.update(status='structural_rejection',error_type=type(error).__name__,error_detail=str(error))
                    result['structural_failures'].append(dict(case=case_id,backend=backend,error_detail=str(error)))
                row['wall_seconds']=time.monotonic()-case_started;save()
                print(json.dumps(dict(case=case_id,backend=backend,status=row['status'],seconds=row['wall_seconds'])),flush=True)
        result['completed']=len(result['cases'])==4*len(backends)
        result['component_parity']=result['completed'] and not any(result[k] for k in ('structural_failures','numeric_failures','decoded_failures'))
        result['limits']='Head-only frozen four-case smoke; no backbone, representative quality, media or release qualification.'
    except Exception as error:
        result['structural_failures'].append(dict(phase=phase,error_type=type(error).__name__,error_detail=str(error)))
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['maximum_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        result['files']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in out.rglob('*') if p.is_file() and p!=review/'panel.json'};save()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir','head-gguf','input-dir','build','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backends',nargs='+',required=True);a=p.parse_args()
    result=run_head_panel(a.reference_dir,a.head_gguf,a.input_dir,a.build,a.output,a.backends)
    raise SystemExit(0 if result['component_parity'] else 2 if result['structural_failures'] else 1)
if __name__=='__main__':main()
