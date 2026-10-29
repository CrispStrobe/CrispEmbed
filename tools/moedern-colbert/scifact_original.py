#!/usr/bin/env python3
"""New original-first full SciFact coverage; never runs a native model.

This does not replace or regenerate the retained ten-case original captures.
All canonical rows, exact original role pipeline and selected token vectors are
retained remotely before any corresponding native/quantized retrieval replay.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools')]
from contract import CONTRACT,digest,read_contract,validate_checkpoint,verify_file
from reference import download,observe_forward
import moedern_scifact as scoring

def require(ok,why):
    if not ok:raise ValueError(why)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')

def hosted():
    require(os.environ.get('GITHUB_ACTIONS')=='true' or
            (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and
             os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch','Interactive')),'hosted original-only capture required')

def source_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def load_original(variant,checkpoint):
    # Same checkpoint/runtime/loaded-state authentication as fresh_reference.py;
    # no handwritten replacement of Transformer/Dense/Mask/Normalize modules.
    import torch
    import sentence_transformers
    from sentence_transformers import MultiVectorEncoder
    from safetensors import safe_open
    cfg=read_contract();declaration=cfg['variants'][variant]['original'];runtime=cfg['reference_runtime']
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','sentence-transformers','numpy')}
    for name,key in [('torch','torch'),('transformers','transformers'),('numpy','numpy')]:
        require(versions[name].split('+')[0]==runtime[key],'reference package version differs: '+name)
    require(versions['sentence-transformers']==runtime['sentence_transformers']['version'],'sentence-transformers version differs')
    st_root=Path(inspect.getfile(sentence_transformers)).parent.parent
    for name,pin in runtime['sentence_transformers']['files'].items():verify_file(st_root/name,pin)
    download(checkpoint,declaration);identities,inventory=validate_checkpoint(checkpoint,variant)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    model=MultiVectorEncoder(str(checkpoint),device='cpu',trust_remote_code=True,local_files_only=True,model_kwargs={'dtype':torch.float32});model.eval()
    verify_file(inspect.getfile(type(model[0].model)),declaration['files']['modeling_modernbert_hash_moe_v10.py'])
    require([type(m).__name__ for m in model]==['Transformer','Dense','MultiVectorMask','Normalize'],'original pipeline differs')
    expansion={k:v for k,v in model[0].query_expansion.items()if v is not None}
    require(expansion=={'strategy':'fixed','attend':False,'length':32},'query expansion differs')
    require(all(p.dtype==torch.float32 and p.device.type=='cpu'for p in model.parameters()),'CPU F32 original required')
    loaded={}
    for file,module in [('model.safetensors',model[0].model),('1_Dense/model.safetensors',model[1])]:
        state=module.state_dict();loaded[file]={}
        with safe_open(str(checkpoint/file),framework='pt',device='cpu')as weights:
            require(set(state)==set(weights.keys()),'loaded original weight inventory differs')
            for name,value in state.items():
                require(torch.equal(value.detach(),weights.get_tensor(name)),'loaded original weight differs: '+name)
                loaded[file][name]=hashlib.sha256(value.detach().contiguous().numpy().astype('<f4',copy=False).tobytes()).hexdigest()
    return model,dict(revision=declaration['revision'],versions=versions,checkpoint_files=identities,inventory=inventory,loaded_tensor_sha256=loaded)

def compare_rankings(original,candidate,gates):
    require(len(original)==len(candidate)==300,'all300 query rankings required')
    require([r['query_id']for r in original]==[r['query_id']for r in candidate],'query order differs')
    a,b=scoring.mean_metrics(original),scoring.mean_metrics(candidate)
    drops={k:a[k]-b[k]for k in scoring.KEYS};top1=sum(x['top1']==y['top1']for x,y in zip(original,candidate))/300
    return dict(original=a,candidate=b,drops=drops,top1_agreement=top1,
                passed=all(v<=gates['max_metric_drop']for v in drops.values())and top1>=gates['min_top1_agreement'])

def run(args):
    hosted()
    import numpy as np
    import torch
    protocol=ROOT/'docs/moedern-scifact-parity-v1.json';cfg=json.loads(protocol.read_text())
    require(cfg['original_first'] is True and cfg['variants']==['1.5M','27M','115M'],'original-first protocol differs')
    for name,pin in cfg['source_sha256'].items():require(digest(ROOT/name)==pin,'declared original source changed: '+name)
    out=args.output;out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed-moedern-scifact-original-v1',variant=args.variant,completed=False,native_executed=False,
                protocol_sha256=digest(protocol),source=os.environ.get('GITHUB_SHA',os.environ.get('CRISPEMBED_CAPTURE_SOURCE')),
                driver_sha256={str(p.relative_to(ROOT)):digest(p)for p in [Path(__file__),CONTRACT,Path(__file__).with_name('contract.py'),Path(__file__).with_name('reference.py'),ROOT/'tools/moedern_scifact.py',ROOT/'tools/ci-heavy/moedern_scifact.py']},
                environment=dict(system=platform.system(),machine=platform.machine(),cpu_count=os.cpu_count(),threads=2),rows=0)
    save=lambda:write(out/'review/original.json',report);save()
    try:
        data_driver=source_module(ROOT/'tools/ci-heavy/moedern_scifact.py','scifact_data')
        data,identity=data_driver.dataset(out,args.scratch);write(out/'raw/data.json',data);report['data']=identity;save()
        rows=[dict(**r,role='query',key='q:'+r['id'])for r in data['queries']]+[dict(**r,role='document',key='d:'+r['id'])for r in data['documents']]
        require(len(rows)==5483 and len({r['key']for r in rows})==5483,'canonical5483 input rows required')
        model,report['model']=load_original(args.variant,args.scratch/'checkpoint');save()
        backbone=model[0].model
        def equal(a,b):
            return set(a)==set(b)and all(torch.equal(a[k],b[k])if isinstance(a[k],torch.Tensor)else a[k]==b[k] for k in a)
        def encode(row,observed):
            method=model.encode_query if row['role']=='query'else model.encode_document
            captured={}
            if not observed:return method(row['text'],output_value=None,show_progress_bar=False),captured
            def before(_args,kwargs):
                captured['supplied_ids']=kwargs['input_ids'].detach().cpu().tolist()[0]
                captured['supplied_mask']=kwargs['attention_mask'].detach().cpu().tolist()[0]
            def first(_module,_args,kwargs):captured['effective_mask']=kwargs['attention_mask'].detach().cpu().tolist()[0]
            handle=backbone.layers[0].register_forward_pre_hook(first,with_kwargs=True)
            try:
                with observe_forward(backbone,before,lambda value:None):result=method(row['text'],output_value=None,show_progress_bar=False)
            finally:handle.remove()
            return result,captured
        # Deterministic input-only selection, declared before model outcomes.
        preflight=rows[:3]+sorted(rows[300:],key=lambda r:(-len(r['text'].encode()),r['key']))[:10]
        controls=[];start=time.monotonic()
        for row in preflight:
            plain,_=encode(row,False);observed,_=encode(row,True);require(equal(plain,observed),'observer changes original features: '+row['key'])
            method=model.encode_query if row['role']=='query'else model.encode_document
            public=method(row['text'],show_progress_bar=False)
            require(torch.equal(public,plain['token_embeddings'][plain['attention_mask'].bool()]),'public selected-slice differs: '+row['key']);controls.append(row['key'])
        projected=(time.monotonic()-start)/len(preflight)/3*len(rows)
        report['preflight']=dict(keys=controls,all_features_bit_exact=True,public_slice_bit_exact=True,projected_seconds=projected,budget_seconds=cfg['encode_budget_seconds']);save()
        require(projected<=cfg['encode_budget_seconds'],'original projected encode exceeds declared budget; shard with a new input-only protocol')
        vectors=[];journal=[];all_at=selected_at=0;start=time.monotonic()
        with (out/'raw/all.f32').open('xb')as all_file,(out/'raw/selected.f32').open('xb')as selected_file:
            for i,row in enumerate(rows):
                stamp=time.monotonic();features,observed=encode(row,True)
                ids=features['input_ids'].detach().cpu().tolist();mask=features['attention_mask'].detach().cpu().tolist()
                a=features['token_embeddings'].detach().contiguous().cpu().numpy().astype('<f4',copy=False)
                require(ids==observed['supplied_ids']and a.shape==(len(ids),128)and np.isfinite(a).all(),'original IDs/vector layout differs')
                require(3<=len(ids)<=512 and ids[:2]==[2,30000 if row['role']=='query'else 30001],'original prefix/length differs')
                require(observed['effective_mask']==([1]*len(ids)if row['role']=='query'else observed['supplied_mask']),'effective mask differs')
                require(row['role']!='query'or(len(ids)==32 and all(mask)),'query expansion/scoring mask differs')
                selected=a[np.asarray(mask,dtype=bool)];require(len(selected)>0,'empty selected vectors')
                av,sv=a.tobytes(),selected.tobytes();all_file.write(av);selected_file.write(sv);vectors.append(selected.copy())
                journal.append(dict(key=row['key'],id=row['id'],role=row['role'],ids=ids,scoring_mask=mask,**observed,
                                    all_offset=all_at,selected_offset=selected_at,all_bytes=len(av),selected_bytes=len(sv),
                                    all_sha256=hashlib.sha256(av).hexdigest(),selected_sha256=hashlib.sha256(sv).hexdigest(),seconds=time.monotonic()-stamp))
                all_at+=len(av);selected_at+=len(sv)
                if (i+1)%100==0:report['rows']=i+1;report['encode_elapsed_seconds']=time.monotonic()-start;save()
        report['rows']=len(rows);report['encode_elapsed_seconds']=time.monotonic()-start;write(out/'raw/rows.json',journal);save()
        start=time.monotonic();scores=data_driver.maxsim(vectors[:300],vectors[300:],np);np.save(out/'raw/scores.npy',scores,allow_pickle=False)
        ranks=[dict(query_id=q['id'],**scoring.rank_scores([d['id']for d in data['documents']],scores[i].tolist(),q['relevant']))for i,q in enumerate(data['queries'])]
        write(out/'raw/rankings.json',ranks)
        # Verify the original author scorer independently on predeclared pairs;
        # F32 torch summation can differ from the full protocol's F64 sum.
        pairs=[]
        for qi,di in cfg['author_score_pairs']:
            q,d=torch.from_numpy(vectors[qi]),torch.from_numpy(vectors[300+di])
            author=float(model.similarity([q],[d]).item());explicit=float((q@d.T).max(dim=1).values.sum().item())
            require(abs(author-explicit)<=.002 and abs(author-float(scores[qi,di]))<=.002,'original MaxSim rule differs')
            pairs.append(dict(query=qi,document=di,author=author,explicit_f32=explicit,protocol=float(scores[qi,di])))
        report.update(completed=True,metrics=scoring.mean_metrics(ranks),maxsim_seconds=time.monotonic()-start,author_score_checks=pairs,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,arrays={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))for p in (out/'raw').iterdir()if p.is_file()})
        save();return 0
    except Exception as e:report['error']=str(e);save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['1.5M','27M','115M'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);raise SystemExit(run(p.parse_args()))
