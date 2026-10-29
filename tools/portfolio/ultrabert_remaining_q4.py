#!/usr/bin/env python3
"""One immutable Q4 candidate: import shard0, run remaining seven, aggregate300."""
import argparse,ctypes,hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ultrabert_q4k_sharded as legacy
import ultrabert_remaining_q4_contract as contract
PROTOCOL=contract.PATH
RUNTIME,FIRST_RECIPE,ROWS,WIDTH=legacy.RUNTIME,legacy.FIRST_RECIPE,legacy.ROWS,legacy.WIDTH
require,read,sha,write,identity,checked_file=legacy.require,legacy.read,legacy.sha,legacy.write,legacy.identity,legacy.checked_file
indices,row_identity,resume_prefix,commit_row,compare=legacy.indices,legacy.row_identity,legacy.resume_prefix,legacy.commit_row,legacy.compare
approval=contract.approval
CACHE={}

def artifact(old,ident,target,pin,maximum):
    # This NEW caller owns its cache; no historical module/global is overridden.
    cfg=contract.declaration();kind=next((k for k in ['freeze','model','shard0']if ident==(cfg['shard0']['artifact']if k=='shard0'else cfg['reviewed_dependencies'][k])['artifact_id']),None)
    require(kind is not None,'Undeclared artifact requested')
    expected=cfg['shard0']['artifact']if kind=='shard0'else cfg['reviewed_dependencies'][kind]
    require(pin==expected,'Artifact dependency identity differs')
    key=str(target.resolve())
    if key not in CACHE:
        service=old.api('actions/runs/'+str(cfg['service_dependencies'][kind]['id']))
        require({k:service[k]for k in cfg['service_dependencies'][kind]}==cfg['service_dependencies'][kind],'Exact immutable workflow/source/status required')
        meta=old.api('actions/artifacts/'+str(ident))
        require(meta['workflow_run']['id']==service['id'],'Artifact belongs to different producer')
        root=legacy.artifact(old,ident,target,pin,maximum)
        CACHE[key]=(ident,pin,maximum,root)
    ident0,pin0,max0,root=CACHE[key]
    require((ident,pin,maximum)==(ident0,pin0,max0)and root==target,'Cached artifact declaration changed')
    legacy.validate_manifest(root)
    return root

def frozen_inputs(args,old):
    pin=approval('freeze');root=artifact(old,args.freeze_artifact,args.scratch/'freeze',pin,256*2**20)
    checked_file(root/'freeze.json',dict(bytes=pin['receipt_bytes'],sha256=pin['receipt_sha256']))
    receipt=read(root/'freeze.json');review=read(contract.REVIEW)
    require(sha(root/'manifest.json')==contract.declaration()['freeze_manifest_sha256']and legacy.validate_manifest(root)==review['payload_manifest']['members'],'Exact full77-member freeze payload differs')
    require(receipt['completed']is True and receipt['f32_qualified']is True and receipt['original_forwards']==receipt['f32_forwards']==receipt['model_or_original_forwards']==0 and receipt['source']['audit_source']==pin['source']and receipt['source']['runtime_source']==RUNTIME,'Reviewed zero-forward F32 freeze required')
    return root,receipt

def shard(args, old, report):
    import numpy as np
    frozen, freeze = frozen_inputs(args, old)
    pin = approval('model')
    prepared = artifact(old, args.model_artifact, args.scratch / 'model', pin, 1024 * 2**20)
    checked_file(prepared / 'model.json', dict(bytes=pin['receipt_bytes'], sha256=pin['receipt_sha256']))
    model_record = read(prepared / 'model.json'); model = prepared / 'model.gguf'
    require(model_record['completed'] and model_record['recipe'] == FIRST_RECIPE and model_record['actual_stored_types'] == {'F32': 206, 'Q4_K': 12}, 'reviewed recipe preparation differs')
    checked_file(model, model_record['model'])
    require(model_record['model']==pin['model'],'Explicit reviewed converted model SHA differs')
    activation = read(PROTOCOL)['activation']
    require(activation['reviewed'] is True and args.shard in activation['shards'] and activation['recipe'] == FIRST_RECIPE,
            'shard not explicitly activated; no native forwards')
    data = read(frozen / 'dataset.json'); expected = {r['id']: r for r in read(frozen / 'lengths.json')}
    allrows = data['queries'] + data['documents']; require(len(allrows) == ROWS, 'all5483 frozen inputs required')
    take = indices(args.shard); rows = [dict(r, input_sha256=row_identity(r, expected)) for r in (allrows[j] for j in take)]
    binding = dict(source={k: report['source'][k] for k in ['runtime_source', 'runtime_git_objects', 'ggml', 'controller_sha256', 'new_controller_sha256']}, library=identity(args.library), comparator=identity(args.diff),
                   frozen_receipt_sha256=sha(frozen / 'freeze.json'), model=model_record['model'],
                   model_receipt_sha256=sha(prepared / 'model.json'), shard=args.shard, recipe=FIRST_RECIPE,
                   inputs=[dict(id=r['id'], input_sha256=r['input_sha256']) for r in rows])
    if args.resume_artifact:
        resume_pin = approval('resume')
        prior = artifact(old, args.resume_artifact, args.scratch / 'resume', resume_pin, 128 * 2**20)
        checked_file(prior / 'shard.json', dict(bytes=resume_pin['receipt_bytes'], sha256=resume_pin['receipt_sha256']))
        require(not read(prior / 'shard.json')['completed'], 'resume requires declared incomplete shard evidence')
        for name in ['binding.json', 'raw.f32', 'normalized.f32', 'rows.jsonl']:
            require(not (args.out / name).exists(), 'resume import would overwrite existing task output')
            shutil.copyfile(prior / name, args.out / name)
        report['resume_receipt_sha256'] = sha(prior / 'shard.json')
    start = resume_prefix(args.out, binding, rows)
    require(start == 0, "This first continuation never resumes or repeats committed forwards")
    report.update(raw_forward_attempts=0, normalized_forward_attempts=0)
    # Workflow launches a fresh Python with the existing exact build/lib path.
    allowed = {'CRISPEMBED_ULTRABERT': '1', 'CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN': '1',
               'CRISPEMBED_ULTRABERT_TRUNCATE': '1', 'CRISPEMBED_ULTRABERT_QUANT': '1'}
    require({k: v for k, v in os.environ.items() if k.startswith('CRISPEMBED_')} == allowed, 'exact four native gates required')
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]; lib.crispembed_init.restype = ctypes.c_void_p
    lib.crispembed_free.argtypes = [ctypes.c_void_p]; lib.crispembed_free.restype = None
    lib.crispembed_encode_tokens_raw.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]; lib.crispembed_encode_tokens_raw.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]; lib.crispembed_encode.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_last_token_ids.argtypes = [ctypes.c_void_p]; lib.crispembed_last_token_ids.restype = ctypes.POINTER(ctypes.c_int32)
    # Retain default/unset, empty and literal0 rejection before gate1 model load.
    report['quant_gate_rejections'] = {}
    for value in [None, '', '0']:
        if value is None: os.environ.pop('CRISPEMBED_ULTRABERT_QUANT', None)
        else: os.environ['CRISPEMBED_ULTRABERT_QUANT'] = value
        bad = lib.crispembed_init(str(model).encode(), 2)
        if bad: lib.crispembed_free(bad)
        require(not bad, 'quant model accepted with quant gate off')
        report['quant_gate_rejections'][str(value)] = True
    os.environ['CRISPEMBED_ULTRABERT_QUANT'] = '1'
    ctx = lib.crispembed_init(str(model).encode(), 2); require(ctx, 'reviewed genuine Q4 model rejected')
    began = time.monotonic()
    try:
        for i in range(start, len(rows)):
            require(time.monotonic() - began < 18000, 'bounded five-hour shard deadline; committed rows retained')
            require(shutil.disk_usage(args.out).free >= 2 * 2**30, 'output disk reserve exhausted; committed rows retained')
            row = rows[i]; text = ((old.QUERY if row['id'].startswith('q:') else '') + row['text']).encode()
            count = ctypes.c_int(); dim = ctypes.c_int()
            report['raw_forward_attempts'] += 1
            require(report['raw_forward_attempts'] <= len(rows), 'Raw per-shard forward cap')
            write(args.out / 'shard.json', report)
            ptr = lib.crispembed_encode_tokens_raw(ctx, text, ctypes.byref(count), ctypes.byref(dim))
            require(bool(ptr) and dim.value == WIDTH, 'raw public CLS rejected: ' + row['id'])
            ids = lib.crispembed_last_token_ids(ctx)
            require(bool(ids) and count.value == len(expected[row['id']]['ids']) and list(ids[:count.value]) == expected[row['id']]['ids'], 'public raw token IDs differ')
            raw = np.ctypeslib.as_array(ptr, shape=(count.value * WIDTH,))[:WIDTH].copy()
            report['normalized_forward_attempts'] += 1
            require(report['normalized_forward_attempts'] <= len(rows), 'Normalized per-shard forward cap')
            write(args.out / 'shard.json', report)
            ptr = lib.crispembed_encode(ctx, text, ctypes.byref(dim)); require(bool(ptr) and dim.value == WIDTH, 'normalized public output rejected')
            norm = np.ctypeslib.as_array(ptr, shape=(WIDTH,)).copy()
            ids = lib.crispembed_last_token_ids(ctx)
            require(bool(ids) and list(ids[:count.value]) == expected[row['id']]['ids'] and np.isfinite(raw).all() and np.isfinite(norm).all(), 'normalized IDs/nonfinite output rejected')
            commit_row(args.out, dict(index=i, global_index=take[i], id=row['id'], input_sha256=row['input_sha256'], actual_ids_exact=True),
                       raw.astype('<f4').tobytes(), norm.astype('<f4').tobytes())
            report.update(completed_rows=i+1, total_rows=len(rows), elapsed_this_process=time.monotonic()-began, resumed_committed_rows=start)
            write(args.out / 'shard.json', report)
    finally: lib.crispembed_free(ctx)
    original_raw = np.memmap(frozen / f'f32/{args.shard}/original-raw.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    original_norm = np.memmap(frozen / f'f32/{args.shard}/original-normalized.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    native_raw = np.memmap(args.out / 'raw.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    native_norm = np.memmap(args.out / 'normalized.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    report['actual_ref'] = compare(old, args.diff, args.out / 'comparison', original_raw, original_norm, native_raw, native_norm)
    report.update(completed=True, completed_rows=len(rows), total_rows=len(rows), recipe=FIRST_RECIPE, shard=args.shard, all_ids_exact=True,
                  streams={k: identity(args.out / k) for k in ['raw.f32', 'normalized.f32', 'rows.jsonl']},
                  strict_f32_vector_flags=report['actual_ref']['exit'] == 0, quantized_qualified=False,
                  original_forwards=0, f32_forwards=0)

def authenticate_shard(root,index,cfg,frozen,source=None):
    saved=legacy.validate_manifest(root);report=read(root/'shard.json');binding=read(root/'binding.json')
    if index==0:
        pin=cfg['shard0'];checked_file(root/'shard.json',dict(bytes=pin['artifact']['receipt_bytes'],sha256=pin['artifact']['receipt_sha256']))
        require(saved==pin['manifest']and sha(root/'manifest.json')==pin['manifest_sha256'],'Exact imported shard0 manifest required')
        require(report['source']==pin['source_binding']and report['conversion_review_sha256']==pin['model_review_sha256'],'Imported producer/controller/model review changed')
    else:
        require(report['source']['audit_source']==source and report['source']['new_controller_sha256']==sha(Path(__file__))and report['source']['new_protocol_sha256']==sha(PROTOCOL),'New shard caller binding differs')
    require(report['completed']is True and report['recipe']==FIRST_RECIPE and report['shard']==index and report['completed_rows']==report['total_rows']==len(indices(index))and report['all_ids_exact']is True and report['original_forwards']==report['f32_forwards']==0,'Incomplete/mismatched shard')
    require(report['quantized_qualified']is False and report['strict_f32_vector_flags']==(report['actual_ref']['exit']==0),'Strict vector outcome changed')
    require(binding['source']=={k:report['source'][k]for k in ['runtime_source','runtime_git_objects','ggml','controller_sha256','new_controller_sha256']}
            and binding['source']['runtime_source']==RUNTIME and binding['source']['ggml']==cfg['ggml']
            and binding['source']['runtime_git_objects']==read(legacy.PROTOCOL)['runtime_git_objects']
            and binding['library']==report['build']['library']and binding['comparator']==report['build']['comparator'],'Runtime/library/comparator binding differs')
    require(binding['model']==cfg['model']and binding['model_receipt_sha256']==cfg['reviewed_dependencies']['model']['receipt_sha256']
            and binding['frozen_receipt_sha256']==sha(frozen/'freeze.json')and binding['shard']==index and binding['recipe']==FIRST_RECIPE,'Frozen input/model binding differs')
    data=read(frozen/'dataset.json');allrows=data['queries']+data['documents'];expected={r['id']:r for r in read(frozen/'lengths.json')}
    wanted=[dict(id=allrows[j]['id'],input_sha256=row_identity(allrows[j],expected))for j in indices(index)]
    require(binding['inputs']==wanted,'Frozen row/text/ID identities differ')
    records=(root/'rows.jsonl').read_bytes();require(records.endswith(b'\n'),'Torn journal rejected')
    records=[json.loads(line)for line in records.splitlines()];require(len(records)==len(wanted),'Exact shard journal coverage required')
    with (root/'raw.f32').open('rb')as raw,(root/'normalized.f32').open('rb')as norm:
        for j,(record,want,global_index)in enumerate(zip(records,wanted,indices(index))):
            require(record['index']==j and record['global_index']==global_index and record['id']==want['id']and record['input_sha256']==want['input_sha256']and record['actual_ids_exact']is True,'Journal row/input coverage differs')
            for kind,stream in [('raw',raw),('normalized',norm)]:
                payload=stream.read(WIDTH*4)
                require(len(payload)==WIDTH*4 and hashlib.sha256(payload).hexdigest()==record[kind+'_sha256'],'Committed row vector hash differs')
        require(raw.read(1)==norm.read(1)==b'','Uncommitted vector tails rejected')
    for name in ['raw.f32','normalized.f32','rows.jsonl']:checked_file(root/name,report['streams'][name])
    result=report['actual_ref']['result']
    require(result==read(root/'comparison/diff.json')and result['max_absolute_budget']==.002 and result['minimum_row_cosine']==.99999
            and result['status']==('passed'if report['actual_ref']['exit']==0 else'numeric_rejection')and len(result['stages'])==2
            and {r['name']for r in result['stages']}=={'raw','normalized'}
            and all(r['ggml_shape']==[WIDTH,len(wanted)]and r['row_axis']==0 and r['elements']==WIDTH*len(wanted)and r['found']is True for r in result['stages']),'Actual Ref shape/schema/flags differ')
    return report

def aggregate(args,old,cfg,frozen,roots,reports):
    import numpy as np
    require(sorted(roots)==list(range(8))and sorted(reports)==list(range(8)),'All eight complete shards required')
    raw=np.empty((ROWS,WIDTH),dtype='<f4');norm=np.empty_like(raw);original_raw=np.empty_like(raw);original_norm=np.empty_like(raw)
    for index in range(8):
        root=roots[index];take=indices(index);r=authenticate_shard(root,index,cfg,frozen,args.reviewed_source)
        for kind,target,original in [('raw',raw,original_raw),('normalized',norm,original_norm)]:
            target[take]=np.memmap(root/(kind+'.f32'),mode='r',dtype='<f4',shape=(len(take),WIDTH))
            original[take]=np.memmap(frozen/f'f32/{index}/original-{kind}.f32',mode='r',dtype='<f4',shape=(len(take),WIDTH))
        independent=dict(raw=old.metrics(original_raw[take],raw[take]),normalized=old.metrics(original_norm[take],norm[take]))
        legacy.metric_agreement(r['actual_ref']['independent'],independent)
        legacy.metric_agreement(independent,{s['name']:s for s in r['actual_ref']['result']['stages']})
        require(all(r['actual_ref']['independent'][k]['f32_gate_passed']==independent[k]['f32_gate_passed']for k in independent),'Strict vector flags changed')
    data=read(frozen/'dataset.json');require(len(data['queries'])==300 and len(data['queries'])+len(data['documents'])==ROWS,'Full5483/300 frozen panel required')
    before=old.rankings(data,original_norm);after=old.rankings(data,norm);quality=old.quality(before,after)
    require(quality['original']==read(frozen/'freeze.json')['quality']['original'],'Frozen original ranking metrics differ')
    write(args.out/'rankings.json',dict(original=before,candidate=after))
    query_summary=[dict(query_id=a['query_id'],original_top1=a['top1'],candidate_top1=b['top1'],top1_agrees=a['top1']==b['top1'],
                         original={k:a[k]for k in ['ndcg_at_10','mrr_at_10','recall_at_100']},candidate={k:b[k]for k in ['ndcg_at_10','mrr_at_10','recall_at_100']})for a,b in zip(before,after)]
    write(args.out/'query-summary.json',query_summary)
    measurements={};allrows=data['queries']+data['documents']
    for kind,reference,actual in [('raw',original_raw,raw),('normalized',original_norm,norm)]:
        values=old.metrics(reference,actual);delta=np.abs(actual.astype('float64')-reference.astype('float64'));row,col=np.unravel_index(np.argmax(delta),delta.shape)
        values.update(original_min=float(reference.min()),original_max=float(reference.max()),native_min=float(actual.min()),native_max=float(actual.max()),worst_absolute_row=dict(global_index=int(row),id=allrows[row]['id'],component=int(col),original=float(reference[row,col]),native=float(actual[row,col])))
        measurements[kind]=values
    calls=sum(r['raw_forward_attempts']+r['normalized_forward_attempts']for i,r in reports.items()if i!=0)
    require(calls==9594 and all(r['raw_forward_attempts']==r['normalized_forward_attempts']==len(indices(i))for i,r in reports.items()if i),'Exact9594 new paired public calls required')
    return dict(completed=True,rows=ROWS,queries=300,recipe=FIRST_RECIPE,model=cfg['model'],raw=measurements['raw'],normalized=measurements['normalized'],
                quality=quality,strict_f32_vector_flags=all(v['f32_gate_passed']for v in measurements.values()),quantized_task_qualified=quality['passed'],passed=quality['passed'],
                imported_public_calls=1372,new_public_calls=calls,aggregate_public_calls=calls+1372,original_forwards=0,f32_forwards=0,quantizer_invocations=0,
                publication_qualified=False,release_qualified=False,standard_header_publication_qualified=False,legacy_header_from_frozen018_quantizer=True,
                rankings=identity(args.out/'rankings.json'),query_summary=identity(args.out/'query-summary.json'),actual_ref_stages=16,
                shards=[dict(shard=i,receipt=identity(roots[i]/'shard.json'),manifest=identity(roots[i]/'manifest.json'),streams=reports[i]['streams'],strict_f32_vector_flags=reports[i]['strict_f32_vector_flags'],actual_ref=reports[i]['actual_ref'])for i in range(8)])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed-source',required=True)
    for name in ['runtime','out','scratch','library','quantizer','diff']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();report=dict(schema='crispembed-ultrabert-remaining-q4-result-v1',completed=False,phase='source',new_public_calls=0,imported_public_calls=0,original_forwards=0,f32_forwards=0,quantizer_invocations=0,publication_qualified=False,release_qualified=False)
    args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=True)
    receipt=args.out/'aggregate.json';reports={};roots={}
    try:
        cfg=contract.verify_source(args.reviewed_source)
        report['source']=dict(legacy.source(args.runtime,args.reviewed_source),new_controller_sha256=sha(Path(__file__)),new_protocol_sha256=sha(PROTOCOL))
        report['resources']=legacy.admission(args.out);report['build']=legacy.build_proof(args)
        old=legacy.old_module(args.runtime)
        service=old.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
        require(service['head_sha']==args.reviewed_source and service['path']=='.github/workflows/ultrabert-remaining-q4-v1.yml'and service['event']=='workflow_dispatch'and service['status']=='in_progress'and service['run_attempt']==1,'Current standalone producer/service differs')
        report['service']={k:service[k]for k in ['id','head_sha','path','event','status','run_attempt']}
        args.freeze_artifact=approval('freeze',cfg)['artifact_id'];args.model_artifact=approval('model',cfg)['artifact_id'];args.resume_artifact=0
        frozen,freeze=frozen_inputs(args,old)
        model_pin=approval('model',cfg);artifact(old,args.model_artifact,args.scratch/'model',model_pin,1024*2**20)
        (args.out/'shards').mkdir()
        first=cfg['shard0']['artifact'];roots[0]=artifact(old,first['artifact_id'],args.out/'shards/0',first,128*2**20)
        reports[0]=authenticate_shard(roots[0],0,cfg,frozen)
        report.update(phase='all-inputs-and-prior-shard-authenticated-before-new-model-calls',imported_public_calls=1372,imported_shard0_receipt_sha256=sha(roots[0]/'shard.json'));write(receipt,report)
        # No conversion, original/F32 rerun, shard0 replay or resumed forward.
        for index in range(1,8):
            args.shard=index;root=args.out/'shards'/str(index);root.mkdir(parents=True);roots[index]=root
            current=dict(schema='crispembed-ultrabert-remaining-q4-shard-v1',source=report['source'],build=report['build'],completed=False,raw_forward_attempts=0,normalized_forward_attempts=0,original_forwards=0,f32_forwards=0,publication_qualified=False)
            reports[index]=current;report['phase']='native-shard-'+str(index);write(receipt,report)
            try:shard(shard_args(args,root),old,current)
            except Exception as error:current['error']=type(error).__name__+': '+str(error);raise
            finally:write(root/'shard.json',current);write(root/'manifest.json',legacy.manifest(root))
            authenticate_shard(root,index,cfg,frozen,args.reviewed_source)
            report['new_public_calls']=sum(r.get('raw_forward_attempts',0)+r.get('normalized_forward_attempts',0)for i,r in reports.items()if i);write(receipt,report)
        report.update(aggregate(args,old,cfg,frozen,roots,reports),phase='complete')
        write(receipt,report);return 0 if report['passed']else 1
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error);return 1
    finally:
        report['new_public_calls']=sum(r.get('raw_forward_attempts',0)+r.get('normalized_forward_attempts',0)for i,r in reports.items()if i)
        write(receipt,report);write(args.out/'manifest.json',legacy.manifest(args.out))

def shard_args(args,out):
    values=vars(args).copy();values['out']=out;return argparse.Namespace(**values)

if __name__=='__main__':raise SystemExit(main())
