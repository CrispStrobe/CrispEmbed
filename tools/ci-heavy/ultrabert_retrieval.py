#!/usr/bin/env python3
"""Original-first owned retrieval smoke; require a qualified API source first.

CPU/model/build work belongs on a hosted runner or separately authorized machine.
No new model/default/release or replacement full-model references are created.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ultrabert_contract import MODEL, REVISION, QUERY, profile, sha256, validate_sources, validate_weight_file
from ultrabert_retrieval import FIXTURE_SHA256, checked_vectors, evaluate, fixture, rankings

REPO = 'CrispStrobe/CrispEmbed'
ST_SOURCE = 'b0acd1a4d22ba2bc10072868ddea7612561c9424'
ST_WRAPPER_SHA = 'b0a945cf57425b1755d13ee5d7d97b7a33af38546a62578d530267ffb7e26e6f'
ST_POOL_SHA = '371f6fb2069387f920851c2f82e125044f769093ebea1d0f4036c8631d89a9e3'


def checked_run(run, source):
    if (run.get('status') != 'completed' or run.get('conclusion') != 'success'
            or run.get('head_sha') != source or run.get('path') != '.github/workflows/ultrabert-api.yml'):
        raise ValueError('API run is incomplete, failed or from an unqualified source/workflow')


def checked_receipt(receipt, source):
    if (receipt.get('source_commit') != source or
            receipt.get('status') != 'completed bounded public F32 pooled-output checks; no independent task-quality qualification' or
            receipt.get('f32_model_sha256') != '89478b9cbbad2fd1121be4031d760f1529c989fedf23b6b19251ba2c87422566'):
        raise ValueError('Wrong/incomplete CPU/F32 API receipt')
    if (receipt.get('context_limit')!=512 or receipt.get('original_source')!='0573f1847db68914052a98341a22fceeb1358518'
            or receipt.get('frozen_report_sha256')!='a3c16863b1a78ebaebcc2d50e24555b696e691d3309ee17b75075c0051032a36'
            or not re.fullmatch('[0-9a-f]{64}',receipt.get('library_sha256',''))
            or not re.fullmatch('[0-9a-f]{64}',receipt.get('driver_sha256',''))):
        raise ValueError('Missing/changed original/API payload fingerprints')
    cases = receipt.get('cases', [])
    expected = {'empty','document','query','unicode','window-boundary','padded-batch'}
    if len(cases) != 6 or {c.get('case') for c in cases} != expected or sum(len(c.get('ids',[])) for c in cases) != 7:
        raise ValueError('API receipt missing original cases/rows')
    for case in cases:
        for ids in case['ids']:
            if not ids or len(ids)>512 or ids[0]!=0 or any(type(x) is not int or not 0<=x<49152 for x in ids):
                raise ValueError('Invalid qualified API token rows')
        if case.get('failures') != 0 or set(case.get('stages',{})) != {'cls_raw','cls_normalized'}:
            raise ValueError('Failed/incomplete original pooled comparison')
        for stage in case['stages'].values():
            values=[stage.get(k) for k in ('cos_min','max_abs','mean_abs','rms','native_norm','reference_norm')]
            if (not all(isinstance(x,(int,float)) and math.isfinite(x) for x in values) or
                    values[0] < .99999 or values[1] > .002 or min(values[1:]) < 0 or stage.get('pass') is not True):
                raise ValueError('API numerical verdict mismatch')
    if not all(receipt.get('surfaces',{}).get(k) is True for k in ('c_api','python','cli','http')):
        raise ValueError('Incomplete API consumer checks')
    hashes=receipt.get('native_source_sha256',{})
    required={'src/ultrabert_embed.cpp','src/ultrabert_profile.h','src/crispembed.cpp','tests/test_ultrabert_api.cpp'}
    if set(hashes)!=required or not all(re.fullmatch('[0-9a-f]{64}',h or '') for h in hashes.values()):
        raise ValueError('Incomplete native source fingerprints')


def extract_receipt(archive, destination):
    """Only the small exact receipt; never extract checkpoint-controlled paths."""
    with zipfile.ZipFile(archive) as handle:
        candidates=[]
        for item in handle.infolist():
            path=PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
                raise ValueError('Unsafe receipt artifact path')
            if path.name=='ultrabert-api.json':candidates.append(item)
        if len(candidates)!=1 or not 0<candidates[0].file_size<2_000_000:
            raise ValueError('Missing/oversize API receipt')
        payload=handle.read(candidates[0]);destination.write_bytes(payload)
        return json.loads(payload)


def api_json(path):
    return json.loads(subprocess.check_output(['gh','api','repos/'+REPO+'/'+path],text=True,timeout=60))


def qualification(run_id, source, scratch, out):
    if not run_id.isdigit() or not re.fullmatch('[0-9a-f]{40}',source):
        raise ValueError('Immutable numeric API run/full source required')
    run=api_json('actions/runs/'+run_id);checked_run(run,source)
    listing=api_json('actions/runs/'+run_id+'/artifacts?per_page=100')
    candidates=[a for a in listing['artifacts'] if a['name']=='ultrabert-f32-api-'+run_id and not a['expired']]
    if len(candidates)!=1 or listing['total_count']>100 or candidates[0]['size_in_bytes']>100_000_000:
        raise ValueError('Missing or oversize successful API artifact')
    artifact=candidates[0]
    archive=scratch/'api-receipt.zip'
    with archive.open('wb') as stream:
        subprocess.run(['gh','api','repos/'+REPO+'/actions/artifacts/'+str(artifact['id'])+'/zip'],stdout=stream,check=True,timeout=300)
    digest=sha256(archive)
    if artifact.get('digest')!='sha256:'+digest:raise ValueError('API artifact digest mismatch')
    receipt=extract_receipt(archive,out/'qualified-api.json');checked_receipt(receipt,source)
    return receipt,{'run':run_id,'source':source,'workflow':'https://github.com/'+REPO+'/actions/runs/'+run_id,
                    'artifact_id':artifact['id'],'artifact_zip_sha256':digest,
                    'receipt_sha256':sha256(out/'qualified-api.json')}


def checked_original(data, record):
    checked_vectors(data,record)
    provenance=record.get('provenance',{})
    expected=profile()['inspected_file_sha256']
    if (provenance.get('model')!=MODEL or provenance.get('revision')!=REVISION
            or provenance.get('source_files')!=expected or provenance.get('hook_neutral') is not True
            or provenance.get('loaded_custom_source_sha256')!=expected['modeling_ultrabert.py']
            or provenance.get('weight_sha256')!='da8be29564b69e4f90fc8f2683e1f369c3d3575e75f7fe798643e28de02e2b9c'
            or provenance.get('wrapper_source')!=ST_SOURCE or provenance.get('wrapper_sha256')!=ST_WRAPPER_SHA
            or provenance.get('pooling_sha256')!=ST_POOL_SHA or provenance.get('wrapper')!='6.0.1'
            or provenance.get('transformers')!='5.16.1' or provenance.get('torch','').split('+')[0]!='2.14.0'):
        raise ValueError('Original record source/module provenance changed')
    for row, captured in zip(data['queries']+data['documents'], record['vectors']):
        text=(QUERY if row['id'].startswith('q') else '')+row['text']
        ids=captured.get('ids',[]);mask=captured.get('attention_mask',[])
        if (captured.get('input_utf8_sha256')!=hashlib.sha256(text.encode()).hexdigest()
                or not 0<len(ids)<=512 or ids[0]!=0 or any(type(x) is not int or not 0<=x<49152 for x in ids)
                or mask!=[1]*len(ids)):
            raise ValueError('Original input identity/IDs/mask changed')


def original_records(directory, data, out):
    # Original inspected sources and payload are checked before custom code runs.
    sources=validate_sources(directory);weights=validate_weight_file(directory/'model.safetensors')
    import numpy as np
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    if (torch.__version__.split('+')[0]!='2.14.0' or transformers.__version__!='5.16.1'
            or sentence_transformers.__version__!='6.0.1'):
        raise ValueError('Pinned original wrapper environment required')
    torch.set_num_threads(2)
    if sha256(inspect.getfile(SentenceTransformer))!=ST_WRAPPER_SHA:
        raise ValueError('Loaded SentenceTransformer wrapper source differs')
    model=SentenceTransformer(str(directory),device='cpu',trust_remote_code=True,local_files_only=True,
                             model_kwargs={'attn_implementation':'eager','dtype':torch.float32})
    if len(model)!=2 or sha256(inspect.getfile(type(model[1])))!=ST_POOL_SHA:
        raise ValueError('Loaded original Pooling source differs')
    if (model[1].pooling_mode!='cls' or model.default_prompt_name is not None
            or model.prompts.get('query')!=QUERY or not model[1].include_prompt):
        raise ValueError('Original module/prompt/CLS contract changed')
    if any(v.dtype!=torch.float32 or v.device.type!='cpu' for v in model.parameters()):
        raise ValueError('Original must remain CPU/F32')
    auto=model[0].auto_model
    if auto.config._attn_implementation!='eager':raise ValueError('Original eager attention required')
    loaded_source=sha256(inspect.getfile(type(auto)))
    if loaded_source!=sources['modeling_ultrabert.py']:raise ValueError('Loaded custom source differs')
    rows=data['queries']+data['documents'];encoded={};lookup={}
    for row in rows:
        text=(QUERY if row['id'].startswith('q') else '')+row['text']
        tokens=model.tokenize([text]);ids=tokens['input_ids'][0].tolist();mask=tokens['attention_mask'][0].tolist()
        if not 0<len(ids)<=512 or ids[0]!=0 or not all(mask):raise ValueError('Unqualified original IDs/length/mask')
        key=tuple(ids)
        if key in lookup:raise ValueError('Ambiguous captured input IDs')
        lookup[key]=row['id'];encoded[row['id']]={'ids':ids,'attention_mask':mask,'input_utf8_sha256':hashlib.sha256(text.encode()).hexdigest()}
    observed={}
    def capture_pool(module, args):
        features=args[0]
        ids=features['input_ids'];mask=features['attention_mask'];hidden=features['token_embeddings']
        if ids.ndim!=2 or mask.shape!=ids.shape or hidden.ndim!=3 or not torch.all(mask[:,0]==1):
            raise ValueError('Original wrapper did not use attended BOS/right-padded batches')
        for i in range(len(ids)):
            key=tuple(ids[i][mask[i].bool()].tolist());name=lookup[key]
            if name in observed:raise ValueError('Duplicate original CLS observation')
            observed[name]=hidden[i,0].detach().cpu().numpy().copy()
    handle=model[1].register_forward_pre_hook(capture_pool)
    def encode_all(normalized):
        query=model.encode([q['text'] for q in data['queries']],prompt_name='query',batch_size=2,
                           normalize_embeddings=normalized,convert_to_numpy=True,show_progress_bar=False)
        documents=model.encode([d['text'] for d in data['documents']],prompt='',batch_size=2,
                               normalize_embeddings=normalized,convert_to_numpy=True,show_progress_bar=False)
        return np.concatenate((query,documents))
    try:raw=encode_all(False)
    finally:handle.remove()
    plain=encode_all(False)
    if not np.array_equal(raw,plain):raise ValueError('Original pooling hook changed wrapper output')
    normal=encode_all(True)
    if raw.shape!=(32,768) or normal.shape!=raw.shape or set(observed)!=set(encoded):
        raise ValueError('Missing original wrapper/CLS coverage')
    vectors=[]
    for i,row in enumerate(rows):
        if not np.array_equal(raw[i],observed[row['id']]):raise ValueError('Wrapper output is not observed raw CLS')
        vectors.append(dict(id=row['id'],raw_cls=raw[i].tolist(),normalized_cls=normal[i].tolist(),**encoded[row['id']]))
    record=dict(pooling='cls',query_prompt=QUERY,document_prompt='',fixture_sha256=FIXTURE_SHA256,vectors=vectors,
                provenance=dict(model=MODEL,revision=REVISION,weight_sha256=weights['sha256'],source_files=sources,
                                loaded_custom_source_sha256=loaded_source,hook_neutral=True,wrapper='6.0.1',wrapper_source=ST_SOURCE,
                                wrapper_sha256=ST_WRAPPER_SHA,pooling_sha256=ST_POOL_SHA,
                                torch=torch.__version__,transformers=transformers.__version__,threads=2,
                                torch_config=torch.__config__.show(),cpu_capability=torch.backends.cpu.get_cpu_capability()))
    checked_original(data,record)
    (out/'original.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    del model,auto;gc.collect()
    return record


def native_records(directory, data, original, source, receipt, scratch, out):
    # Build a portable CPU library from the already qualified native source.
    # This is a distinct measured binary, not a claim of bit-identical API assets.
    engine=scratch/'engine';engine.mkdir()
    def command(args,name):
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str(a) for a in args],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    for args,name in [(['git','-C',engine,'init'],'git-init'),
                      (['git','-C',engine,'remote','add','origin','https://github.com/'+REPO+'.git'],'git-remote'),
                      (['git','-C',engine,'fetch','--depth=1','origin',source],'git-fetch'),
                      (['git','-C',engine,'checkout','--detach',source],'git-checkout'),
                      (['git','-C',engine,'submodule','update','--init','--recursive'],'git-submodules')]:command(args,name)
    for name,digest in receipt['native_source_sha256'].items():
        if sha256(engine/name)!=digest:raise ValueError('Qualified native source differs')
    build=scratch/'native-build'
    command(['cmake','-S',engine,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCRISPEMBED_BUILD_SHARED=ON',
             '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF'],'configure')
    command([sys.executable,engine/'scripts/check-cpu-baseline.py',build],'cpu-baseline')
    command(['cmake','--build',build,'--target','crispembed-shared','--parallel','2'],'build')
    converter_spec=importlib.util.spec_from_file_location('original_converter',ROOT/'models/convert-ultrabert-to-gguf.py')
    converter=importlib.util.module_from_spec(converter_spec);converter_spec.loader.exec_module(converter)
    model_path=scratch/'native-f32.gguf';converted=converter.convert(directory,model_path,'f32')
    if converted['sha256']!=receipt['f32_model_sha256']:raise ValueError('Original conversion bytes differ')
    suffix='dylib' if sys.platform=='darwin' else 'so'
    library=next(build.rglob('libcrispembed.'+suffix))
    sys.path.insert(0,str(engine/'python'))
    from crispembed import CrispEmbed
    os.environ['CRISPEMBED_ULTRABERT']='1';os.environ['CRISPEMBED_FORCE_CPU']='1'
    model=CrispEmbed(str(model_path),n_threads=2,lib_path=str(library),offline=True)
    record=dict(pooling='cls',query_prompt=QUERY,document_prompt='',fixture_sha256=FIXTURE_SHA256,vectors=[],
                provenance=dict(source_commit=source,library_sha256=sha256(library),model_sha256=sha256(model_path),
                                qualified_api_library_sha256=receipt['library_sha256'],
                                scope='Fresh portable CPU binary measured on this task; API receipt qualifies source, not byte-identical artifact',
                                ggml_source=subprocess.check_output(['git','-C',str(engine/'ggml'),'rev-parse','HEAD'],text=True).strip()))
    for row,reference in zip(data['queries']+data['documents'],original['vectors']):
        model.set_prefix(QUERY if row['id'].startswith('q') else '')
        ids,hidden=model.encode_tokens(row['text'],normalize=False)
        if ids.tolist()!=reference['ids']:
            record['first_id_mismatch']={'id':row['id'],'native_ids':ids.tolist(),'original_ids':reference['ids'],
                                         'actual_raw_cls':hidden[0].tolist()}
            (out/'native.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            raise ValueError('Native task text IDs differ')
        normal=model.encode(row['text'])
        record['vectors'].append(dict(id=row['id'],ids=ids.tolist(),raw_cls=hidden[0].tolist(),normalized_cls=normal.tolist()))
        (out/'native.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    checked_vectors(data,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-run',required=True)
    parser.add_argument('--native-source',required=True)
    parser.add_argument('--reuse-original',type=Path)
    parser.add_argument('--original-sha256')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'ultrabert-retrieval'
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    report=dict(task_source=os.environ['GITHUB_SHA'],driver_sha256=sha256(__file__),fixture_sha256=FIXTURE_SHA256,
                scope='Owned retrieval smoke only; full-stage debt/broad benchmark quality/release gates remain separate',status='started')
    def save():(out/'retrieval-producer.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        data=fixture(ROOT/'tests/regression/ultrabert/retrieval-smoke-v1.json')
        if bool(args.reuse_original)!=bool(args.original_sha256):
            raise ValueError('Original reuse requires both verified report and its recorded SHA256')
        if args.reuse_original:
            original_payload=args.reuse_original.read_bytes()
            if hashlib.sha256(original_payload).hexdigest()!=args.original_sha256:
                raise ValueError('Original report bytes differ')
            original=json.loads(original_payload);checked_original(data,original)
        receipt,proof=qualification(args.api_run,args.native_source,scratch,out);report['qualification']=proof;save()
        if args.preflight:
            report['status']='preflight passed; no model/build executed';save();return 0
        from huggingface_hub import snapshot_download
        directory=Path(snapshot_download(MODEL,revision=REVISION,
                       allow_patterns=list(profile()['inspected_file_sha256'])+['model.safetensors']))
        if args.reuse_original:
            validate_sources(directory);validate_weight_file(directory/'model.safetensors')
            (out/'original.json').write_bytes(original_payload)
            report['original_reused']=True
        else:
            original=original_records(directory,data,out)
            report['original_reused']=False
        report['original_sha256']=sha256(out/'original.json')
        (out/'original-scores.json').write_text(json.dumps(rankings(data,checked_vectors(data,original)),indent=2,allow_nan=False)+'\n')
        report['status']='original saved before native';save()
        native=native_records(directory,data,original,args.native_source,receipt,scratch,out)
        report['native_sha256']=sha256(out/'native.json')
        score=evaluate(data,original,native)
        (out/'scores.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
        report['status']='completed smoke' if score['passed'] else 'failed numerical/ranking/quality smoke'
        report['passed']=score['passed'];report['scores_sha256']=sha256(out/'scores.json');save()
        return 0 if score['passed'] else 1
    except BaseException as error:
        report['status']='failed producer';report['error_type']=type(error).__name__;save();raise


if __name__=='__main__':raise SystemExit(main())
