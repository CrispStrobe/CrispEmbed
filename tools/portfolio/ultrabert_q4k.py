#!/usr/bin/env python3
"""Hosted-only UltraBERT original-first Q4_K retrieval qualification.

No original-model forward is executed. Every reference array, input and ID is
reused from authenticated frozen SciFact artifacts. F32 gates are unchanged;
quantized vector drift is reported separately from the unchanged ranking gates.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from ultrabert_contract import MODEL, REVISION, QUERY, profile, sha256
from ultrabert_scifact import quality, rank_scores
from josev2b_composed_diff import write_gguf

def require(ok,why):
    if not ok: raise ValueError(why)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def api(path): return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))
def artifact(pin,out,selected):
    run=api('actions/runs/'+str(pin['run']))
    source=pin.get('head',pin.get('source'))
    require(run['head_sha']==source and run['status']=='completed' and run['conclusion']=='success' and run['run_attempt']==1,'original producer differs')
    ident=pin.get('artifact_id',pin.get('id')); row=api('actions/artifacts/'+str(ident))
    digest=pin.get('digest','sha256:'+str(pin.get('sha256')))
    require(row['workflow_run']['id']==pin['run'] and row['workflow_run']['head_sha']==source and not row['expired'] and row['name']==pin['name'] and row['size_in_bytes']==pin['bytes'] and row['digest']==digest and 0<pin['bytes']<=32*1024**2,'immutable artifact identity differs')
    archive=out.with_suffix('.zip');out.mkdir()
    with archive.open('xb')as f:subprocess.run(['gh','api',f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{ident}/zip'],stdout=f,check=True,timeout=300)
    require(archive.stat().st_size==pin['bytes'] and 'sha256:'+sha256(archive)==digest,'artifact ZIP SHA differs')
    with zipfile.ZipFile(archive)as z:
        names=[r.filename for r in z.infolist()]
        require(len(names)==len(set(names)) and len(names)<=1024 and sum(r.file_size for r in z.infolist())<=128*1024**2,'artifact duplicate/extent exceeds bounds')
        chosen={}
        for r in z.infolist():
            p=PurePosixPath(r.filename)
            require(not p.is_absolute() and '..'not in p.parts and '\\'not in r.filename and ((r.external_attr>>16)&0o170000)!=0o120000,'unsafe artifact member')
            for name in selected:
                if r.filename==name or r.filename.endswith('/'+name):
                    require(name not in chosen,'ambiguous original member');chosen[name]=r
        require(set(chosen)==set(selected),'original member absent')
        for name,r in chosen.items():
            data=z.read(r); expected=selected[name]
            require(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],'frozen original member changed')
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    archive.unlink()
    return {k:row[k]for k in ['id','name','size_in_bytes','digest','expires_at']}

def recipes():
    # Selection is declared before measuring any quantized result. Every recipe
    # protects vocabulary and norms; stricter variants protect all attention.
    base=['--tensor-type','token_embd.weight=f32']
    attention=[]
    for kind in ['q','k','v','output']:
        attention+=['--tensor-type',f'blk.*.attn_{kind}.weight=f32']
    late=[]
    for i in range(20):
        for kind in ['gate','up','down']:
            late+=['--tensor-type',f'blk.{i}.ffn_{kind}.weight=f32']
    return {'q4k-all-matrices':base,'q4k-ffn':base+attention,'q4k-last-four-ffn':base+attention+late}

def native(lib,model,rows,expected):
    import numpy as np
    lib.crispembed_init.argtypes=[ctypes.c_char_p,ctypes.c_int];lib.crispembed_init.restype=ctypes.c_void_p
    lib.crispembed_free.argtypes=[ctypes.c_void_p];lib.crispembed_free.restype=None
    lib.crispembed_encode.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)];lib.crispembed_encode.restype=ctypes.POINTER(ctypes.c_float)
    lib.crispembed_encode_tokens_raw.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)];lib.crispembed_encode_tokens_raw.restype=ctypes.POINTER(ctypes.c_float)
    lib.crispembed_last_token_ids.argtypes=[ctypes.c_void_p];lib.crispembed_last_token_ids.restype=ctypes.POINTER(ctypes.c_int32)
    ctx=lib.crispembed_init(str(model).encode(),2);require(ctx,'native model rejected')
    raw=np.empty((len(rows),768),dtype='<f4');norm=np.empty_like(raw);start=time.monotonic()
    try:
        for i,row in enumerate(rows):
            text=((QUERY if row['id'].startswith('q:')else'')+row['text']).encode()
            count=ctypes.c_int();dim=ctypes.c_int();ptr=lib.crispembed_encode_tokens_raw(ctx,text,ctypes.byref(count),ctypes.byref(dim));require(bool(ptr) and dim.value==768,'raw CLS rejected '+row['id'])
            ids=lib.crispembed_last_token_ids(ctx)
            require(bool(ids) and count.value==len(expected[row['id']]['ids']) and list(ids[:count.value])==expected[row['id']]['ids'],'actual native IDs differ '+row['id'])
            raw[i]=np.ctypeslib.as_array(ptr,shape=(count.value*768,))[:768]
            ptr=lib.crispembed_encode(ctx,text,ctypes.byref(dim));require(bool(ptr) and dim.value==768,'normalized output rejected '+row['id'])
            norm[i]=np.ctypeslib.as_array(ptr,shape=(768,))
            require(np.isfinite(raw[i]).all() and np.isfinite(norm[i]).all(),'nonfinite native output')
    finally:lib.crispembed_free(ctx)
    return raw,norm,time.monotonic()-start

def metrics(a,b):
    import numpy as np
    a,b=a.astype('float64'),b.astype('float64');d=np.abs(a-b)
    cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    require(np.isfinite(cosine).all(),'invalid cosine')
    return dict(max_absolute=float(d.max()),mean_absolute=float(d.mean()),rms=float(np.sqrt(np.mean(d*d))),min_row_cosine=float(cosine.min()),original_l2=float(np.linalg.norm(a)),native_l2=float(np.linalg.norm(b)),f32_gate_passed=bool(d.max()<=.002 and cosine.min()>=.99999))

def rankings(data,values):
    import numpy as np
    m=values.astype('float64');m/=np.linalg.norm(m,axis=1,keepdims=True);scores=m[:300]@m[300:].T
    ids=[r['id']for r in data['documents']]
    return [dict(query_id=r['id'],**rank_scores(ids,scores[i],r['relevant_document_ids']))for i,r in enumerate(data['queries'])]

def run(args):
    require(os.environ.get('GITHUB_ACTIONS')=='true','hosted qualification only')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==os.environ['GITHUB_SHA'],'source/service mismatch')
    require(not subprocess.check_output(['git','status','--porcelain','--','src','CMakeLists.txt','tools/portfolio','models/convert-ultrabert-to-gguf.py'],cwd=ROOT,text=True),'dirty runtime/worker')
    import numpy as np
    from huggingface_hub import snapshot_download
    from gguf import GGUFReader
    out=args.output;scratch=args.scratch;out.mkdir();scratch.mkdir()
    report=dict(schema='crispembed-ultrabert-q4k-v1',source=os.environ['GITHUB_SHA'],completed=False,original_rerun=False,arms={},qualified=[],release_qualified=False)
    save=lambda:write(out/'review.json',report);save()
    try:
        frozen=json.loads((ROOT/'tests/results/ultrabert/scifact-original-freeze.json').read_text())
        prep=json.loads((ROOT/'tests/results/ultrabert/scifact-canonical-preparation.json').read_text())
        needed={k:prep['files'][k]for k in ['dataset.json','lengths.json']}
        report['preparation']=artifact(frozen['preparation'],scratch/'prepared',needed)
        data=json.loads((scratch/'prepared/dataset.json').read_text());lengths=json.loads((scratch/'prepared/lengths.json').read_text());expected={r['id']:r for r in lengths}
        rows=data['queries']+data['documents'];require(len(rows)==len(expected)==5483 and len(data['queries'])==300,'frozen input coverage differs')
        original_raw=np.empty((5483,768),dtype='<f4');original_norm=np.empty_like(original_raw)
        report['originals']=[]
        for shard in frozen['shards']:
            i=shard['shard'];dest=scratch/('original-'+str(i));members={k:shard['raw_member_identities'][k]for k in ['raw.npy','normalized.npy','ids.json']}
            report['originals'].append(artifact(shard['raw_artifact_lineage'],dest,members))
            r=np.load(dest/'raw.npy',allow_pickle=False);n=np.load(dest/'normalized.npy',allow_pickle=False);ids=json.loads((dest/'ids.json').read_text());indices=list(range(i,5483,8))
            require(r.shape==n.shape==(len(indices),768) and r.dtype==n.dtype==np.dtype('<f4'),'original array layout differs')
            require([v['id']for v in ids]==[rows[j]['id']for j in indices] and all(v['ids']==expected[v['id']]['ids']for v in ids),'original ID lineage differs')
            original_raw[indices]=r;original_norm[indices]=n
        before=rankings(data,original_norm);save()
        snapshot=snapshot_download(MODEL,revision=REVISION,allow_patterns=list(profile()['inspected_file_sha256'])+['model.safetensors'])
        spec=importlib.util.spec_from_file_location('converter',ROOT/'models/convert-ultrabert-to-gguf.py');converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        f32=scratch/'ultrabert-f32.gguf';converter.convert(snapshot,f32,'f32')
        env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')};env.update(CRISPEMBED_ULTRABERT='1',CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN='1',CRISPEMBED_ULTRABERT_TRUNCATE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        os.environ.clear();os.environ.update(env)
        lib=ctypes.CDLL(str(args.library.resolve()))
        for name,rules in [('f32',None),*recipes().items()]:
            model=f32
            if rules is not None:
                model=scratch/(name+'.gguf')
                with (out/(name+'-quantizer.log')).open('w')as log:subprocess.run([str(args.quantizer.resolve()),str(f32),str(model),'q4_k',*rules],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                os.environ['CRISPEMBED_ULTRABERT_QUANT']='0';require(not lib.crispembed_init(str(model).encode(),2),'quantized model accepted without quant gate')
                os.environ['CRISPEMBED_ULTRABERT_QUANT']='1'
            else:os.environ.pop('CRISPEMBED_ULTRABERT_QUANT',None)
            reader=GGUFReader(model);types={}
            for t in reader.tensors:types[t.tensor_type.name]=types.get(t.tensor_type.name,0)+1
            require(rules is None or types.get('Q4_K',0)>0,'mislabeled quantized model');del reader
            raw,norm,seconds=native(lib,model,rows,expected);rr=rankings(data,norm);q=quality(before,rr)
            record=dict(model_bytes=model.stat().st_size,sha256=sha256(model),stored_types=types,raw=metrics(original_raw,raw),normalized=metrics(original_norm,norm),quality=q,elapsed_seconds=seconds,all_ids_exact=True,actual_ref=[])
            for i in range(8):
                folder=out/name/str(i);folder.mkdir(parents=True)
                indices=list(range(i,5483,8));left=[];right=[]
                for kind,a,b in [('raw',original_raw,raw),('normalized',original_norm,norm)]:
                    for label,values,specs in [('original',a,left),('native',b,right)]:
                        p=folder/(label+'-'+kind+'.f32');values[indices].tofile(p)
                        specs.append(dict(name=kind,path=str(p),shape=[len(indices),768],sha256=sha256(p)))
                a,b=folder/'original.gguf',folder/'native.gguf';write_gguf(a,left);write_gguf(b,right)
                call=subprocess.run([str(args.diff.resolve()),str(a),str(b),str(folder/'diff.json')],capture_output=True,text=True,timeout=90)
                require(call.returncode in (0,1),'actual Ref structural failure')
                result=json.loads((folder/'diff.json').read_text());require(len(result['stages'])==2,'actual Ref stage coverage differs')
                independent={'raw':metrics(original_raw[indices],raw[indices]),'normalized':metrics(original_norm[indices],norm[indices])}
                for stage in result['stages']:
                    require(stage['name']in independent,'unexpected Ref stage')
                    for field in ['max_absolute','mean_absolute','rms','min_row_cosine','original_l2','native_l2']:
                        a,b=stage[field],independent[stage['name']][field]
                        require(abs(a-b)<=2e-7*max(1.,abs(a),abs(b)),'Ref/independent metric disagreement '+field)
                record['actual_ref'].append(dict(shard=i,exit=call.returncode,result=result,independent=independent))
            record['qualified']=bool(q['passed'] and (rules is not None or (record['raw']['f32_gate_passed'] and record['normalized']['f32_gate_passed'] and all(r['exit']==0 for r in record['actual_ref']))))
            report['arms'][name]=record;save()
            if rules is None:require(record['qualified'],'F32 control fails frozen gates before quantized qualification')
            if rules is not None and record['qualified']:
                report['qualified'].append(name);(out/'assets').mkdir(exist_ok=True);shutil.copyfile(model,out/'assets'/model.name);save()
        report['completed']=True;save();return int(not report['qualified'])
    except Exception as e:report['error']=str(e);save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['output','scratch','library','quantizer','diff']:p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
