#!/usr/bin/env python3
"""Model-first then separately reviewed shard0; no original/F32 replay."""
import argparse,ctypes,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ultrabert_q4k_sharded as legacy
import ultrabert_first_q4_contract as contract
PROTOCOL=contract.PATH
RUNTIME,F32_SHA,FIRST_RECIPE,ROWS,WIDTH=legacy.RUNTIME,legacy.F32_SHA,legacy.FIRST_RECIPE,legacy.ROWS,legacy.WIDTH
require,read,sha,write,identity,checked_file=legacy.require,legacy.read,legacy.sha,legacy.write,legacy.identity,legacy.checked_file
artifact,indices,row_identity,resume_prefix,commit_row,compare=legacy.artifact,legacy.indices,legacy.row_identity,legacy.resume_prefix,legacy.commit_row,legacy.compare
approval=contract.approval


def frozen_inputs(args,old):
    pin=approval('freeze');root=artifact(old,args.freeze_artifact,args.scratch/'freeze',pin,256*2**20)
    checked_file(root/'freeze.json',dict(bytes=pin['receipt_bytes'],sha256=pin['receipt_sha256']))
    receipt=read(root/'freeze.json');review=read(contract.REVIEW)
    require(sha(root/'manifest.json')==contract.declaration()['freeze_manifest_sha256']and legacy.validate_manifest(root)==review['payload_manifest']['members'],'Exact full77-member freeze payload differs')
    require(receipt['completed']is True and receipt['f32_qualified']is True and receipt['original_forwards']==receipt['f32_forwards']==receipt['model_or_original_forwards']==0 and receipt['source']['audit_source']==pin['source']and receipt['source']['runtime_source']==RUNTIME,'Reviewed zero-forward F32 freeze required')
    return root,receipt


def prepare_model(args, old, report):
    from huggingface_hub import snapshot_download
    from gguf import GGUFReader
    frozen, freeze = frozen_inputs(args, old)
    # No module/model execution: reproduce the authenticated conversion only.
    snapshot = snapshot_download(old.MODEL, revision=old.REVISION,
                                 allow_patterns=list(old.profile()['inspected_file_sha256']) + ['model.safetensors'])
    path = args.runtime / 'models/convert-ultrabert-to-gguf.py'
    spec = importlib.util.spec_from_file_location('frozen_ultrabert_converter', path)
    converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
    f32 = args.scratch / 'ultrabert-f32.gguf'; converter.convert(snapshot, f32, 'f32')
    checked_file(f32, dict(bytes=609437696, sha256=F32_SHA))
    model = args.out / 'model.gguf'
    rules = old.recipes()[FIRST_RECIPE]
    with (args.out / 'quantizer.log').open('wb') as log:
        call = subprocess.run([str(args.quantizer.resolve()), str(f32), str(model), 'q4_k', *rules],
                              stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    require(call.returncode == 0, 'actual frozen C quantizer failed')
    a = GGUFReader(f32); b = GGUFReader(model)
    original = {t.name: t for t in a.tensors}; inventory = {}; types = {}
    require(len(original) == len(b.tensors) == 218, 'exact218 tensor inventory required')
    for tensor in b.tensors:
        require(tensor.name in original and list(map(int, tensor.shape)) == list(map(int, original[tensor.name].shape)), 'semantic tensor shape changed')
        typ = tensor.tensor_type.name; types[typ] = types.get(typ, 0) + 1
        wanted = 'Q4_K' if re.fullmatch(r'blk\.(20|21|22|23)\.ffn_(gate|up|down)\.weight', tensor.name) else 'F32'
        require(typ == wanted, 'actual role precision differs: ' + tensor.name)
        checksum = hashlib.sha256(tensor.data).hexdigest()
        if typ == 'F32': require(checksum == hashlib.sha256(original[tensor.name].data).hexdigest(), 'critical F32 payload changed')
        inventory[tensor.name] = dict(type=typ, shape=list(map(int, tensor.shape)), payload_sha256=checksum)
    require(types == {'F32': 206, 'Q4_K': 12}, 'declared twelve genuine Q4_K/206F32 tensors required')
    report.update(completed=True, recipe=FIRST_RECIPE, model=identity(model), original_f32=dict(bytes=609437696, sha256=F32_SHA),
                  inventory=inventory, actual_stored_types=types, quantizer=identity(args.quantizer), quantizer_log=identity(args.out / 'quantizer.log'),
                  freeze_receipt_sha256=sha(frozen / 'freeze.json'), model_or_original_forwards=0,
                  legacy_header_from_frozen018_quantizer=True, standard_header_publication_qualified=False)


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
            ptr = lib.crispembed_encode_tokens_raw(ctx, text, ctypes.byref(count), ctypes.byref(dim))
            require(bool(ptr) and dim.value == WIDTH, 'raw public CLS rejected: ' + row['id'])
            ids = lib.crispembed_last_token_ids(ctx)
            require(bool(ids) and count.value == len(expected[row['id']]['ids']) and list(ids[:count.value]) == expected[row['id']]['ids'], 'public raw token IDs differ')
            raw = np.ctypeslib.as_array(ptr, shape=(count.value * WIDTH,))[:WIDTH].copy()
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['model','shard0']);p.add_argument('--reviewed-source',required=True)
    for name in ['runtime','out','scratch','library','quantizer','diff']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume-artifact',type=int,default=0);a=p.parse_args();a.shard=0;a.out.mkdir(parents=True,exist_ok=True);a.scratch.mkdir(parents=True,exist_ok=True)
    report=dict(schema='crispembed-ultrabert-first-q4-shard-v1',phase=a.phase,completed=False,original_forwards=0,f32_forwards=0,release_qualified=False,publication_qualified=False,q8_activation_authorized=False)
    receipt=a.out/('model.json'if a.phase=='model'else'shard.json')
    try:
        cfg=contract.verify_source(a.reviewed_source);contract.phase_admission(a.phase,cfg)
        require(os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine hosted GitHub CPU runner required')
        report['source']=dict(legacy.source(a.runtime,a.reviewed_source),new_controller_sha256=sha(Path(__file__)),new_protocol_sha256=sha(PROTOCOL))
        report['resources']=legacy.admission(a.out);report['build']=legacy.build_proof(a)
        a.freeze_artifact=approval('freeze',cfg)['artifact_id'];a.model_artifact=approval('model',cfg)['artifact_id']if a.phase=='shard0'else 0
        if a.resume_artifact:require(a.phase=='shard0'and approval('resume',cfg)['artifact_id']==a.resume_artifact,'Explicit reviewed partial-shard resume only')
        original=legacy.old_module(a.runtime)
        (prepare_model if a.phase=='model'else shard)(a,original,report)
        require(report['completed']is True,'Phase structurally incomplete');write(receipt,report);write(a.out/'manifest.json',legacy.manifest(a.out));return 0
    except Exception as e:
        report['error']=type(e).__name__+': '+str(e);write(receipt,report);write(a.out/'manifest.json',legacy.manifest(a.out));return 1

if __name__=='__main__':raise SystemExit(main())
