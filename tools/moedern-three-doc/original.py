"""Only three fresh original CPU documents; frozen full corpus is never recaptured."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,PATH,KEYS,atomic,module,require,sha,verify


def run(args):
    cfg=verify(args.reviewed_source);packet=json.loads((args.input/'input-packet.json').read_text())
    require(sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,12),'isolated original CPython3.12 required')
    require(os.environ.get('GITHUB_ACTIONS')=='true'and packet['completed']is True and packet['keys']==KEYS and
            packet['source_protocol_sha256']==sha(PATH),'hosted exact authenticated input packet required')
    require(len(packet['cases'])==3 and [c['key']for c in packet['cases']]==KEYS,'exact three input cases required')
    prepared=json.loads((args.input/'prepare.json').read_text())
    require(prepared['completed']is True and prepared['input_packet_sha256']==sha(args.input/'input-packet.json'),'prepared packet identity differs')
    for case in packet['cases']:
        for side in ('original','native'):
            for row in case['anchors'][side].values():
                p=args.input/row['path'];require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'frozen span changed before framework import')
    # Source and every old input/span is authenticated BEFORE optional frameworks.
    sys.path[:0]=[str(ROOT/'tools/moedern-colbert'),str(ROOT/'tools/moedern-native-v2')]
    source=module(ROOT/'tools/moedern-colbert/scifact_original.py','frozen_scifact_original_loader')
    reference=module(ROOT/'tools/moedern-native-v2/replay.py','frozen_native_comparison')
    observer=module(ROOT/'tools/moedern-colbert/reference.py','frozen_actual_observers')
    import torch
    import numpy as np
    import torch.nn.functional as functional
    args.out.mkdir(parents=True,exist_ok=False);report=dict(completed=False,source=args.reviewed_source,phase='load-original',cases=[],
        full_original_recapture=False,original_documents=3,publication_qualified=False,arrays=[])
    def save():atomic(args.out/'original.json',report)
    save()
    try:
        model,report['model']=source.load_original('1.5M',args.checkpoint);save()
        old=json.loads((ROOT/cfg['original']['public_report']).read_text())
        require(report['model']==old['model'],'loaded original source/runtime/weights differ from frozen full original')
        report['hardware']=dict(cpu_model=next((l.split(':',1)[1].strip()for l in Path('/proc/cpuinfo').read_text().splitlines()if l.startswith('model name')),'unknown'),
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),mkldnn_enabled=torch.backends.mkldnn.enabled)
        backbone=model[0].model;method=model.encode_document;ordinary={};anchor_pairs=[];anchor_metrics={}
        def copied(features):return {k:v.detach().clone()if isinstance(v,torch.Tensor)else v for k,v in features.items()}
        def equal(a,b):return set(a)==set(b)and all(torch.equal(a[k],b[k])if isinstance(a[k],torch.Tensor)else a[k]==b[k]for k in a)
        with torch.inference_mode():
            for case in packet['cases']:
                cid=case['id'];features=copied(method(case['text'],output_value=None,show_progress_bar=False));ordinary[cid]=features
                ids=features['input_ids'].tolist();mask=features['attention_mask'].tolist()
                require(ids==case['ids']and list(map(int,mask))==case['scoring_mask'],'fresh original IDs/scoring mask differ')
                all_values=features['token_embeddings'];selected=all_values[features['attention_mask'].bool()]
                record=dict(key=case['key'],id=cid,anchor={},observer_neutral=False);report['cases'].append(record)
                for kind,value in (('all',all_values),('selected',selected)):
                    frozen=case['anchors']['original'][kind];target=args.input/frozen['path']
                    a=np.fromfile(target,dtype='<f4').reshape(frozen['shape']);b=value.detach().cpu().numpy()
                    require(b.dtype==np.float32 and list(b.shape)==frozen['shape']and np.isfinite(b).all(),'fresh original finite/shape differs')
                    path=args.out/cid/(kind+'.f32');path.parent.mkdir(parents=True,exist_ok=True);b.tofile(path)
                    measured=reference.compare(a,b);measured.update(bit_exact=sha(path)==frozen['sha256'],frozen_sha256=frozen['sha256'],
                        current_sha256=sha(path),current_file=str(path.relative_to(args.out)));record['anchor'][kind]=measured
                    name=cid+'.'+kind;anchor_metrics[name]=measured
                    anchor_pairs.append(dict(name=name,original=dict(path=target,shape=frozen['shape'],sha256=frozen['sha256']),
                        native=dict(path=path,shape=frozen['shape'],sha256=sha(path))))
                save()
        adapters=reference.write_pair(args.out/'anchor-original.gguf',args.out/'anchor-current.gguf',anchor_pairs)
        measured=reference.actual_ref(args.diff,args.out/'anchor-original.gguf',args.out/'anchor-current.gguf',args.out/'anchor-ref.json',args.out/'anchor-ref.log',anchor_metrics)
        report['anchor_actual_ref']=measured;report['anchor_adapters']=adapters;save()
        require(measured['all_metrics_agree']and all(r['passed']for r in anchor_metrics.values()),'fresh original strict final anchor rejected; stage attribution forbidden')
        report['phase']='neutral-three-document-observers';save()
        total=0
        def capture(cid,stage,tree):
            nonlocal total
            for label,value in observer.capture_leaves(stage,tree,lambda x:isinstance(x,torch.Tensor)):
                value=value.detach().clone().contiguous().cpu()
                require(value.numel()>0,'empty captured stage')
                if value.is_floating_point():
                    require(value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'captured original nonfinite/nonF32')
                    data=value.numpy().astype('<f4',copy=False);dtype='f32'
                elif value.dtype==torch.bool:data=value.numpy().astype('u1');dtype='u8'
                else:data=value.numpy().astype('<i8');dtype='i64'
                raw=data.tobytes();total+=len(raw);require(total<=384*2**20,'bounded three-document original trace exceeded')
                relative=cid+'/'+hashlib.sha256(label.encode()).hexdigest()[:24]+'.'+dtype
                path=args.out/relative;require(not path.exists(),'duplicate original stage label');path.write_bytes(raw)
                report['arrays'].append(dict(case=cid,stage=label,path=relative,dtype=dtype,shape=list(value.shape),bytes=len(raw),sha256=sha(path)))
        for case,record in zip(packet['cases'],report['cases']):
            cid=case['id'];counts={};handles=[];captured={};sdpa_count=0
            def event(name,value):
                i=counts.get(name,0);counts[name]=i+1;capture(cid,name+'#'+str(i),value)
            def input_capture(args_,kwargs):
                captured['supplied_ids']=kwargs['input_ids'].detach().cpu().tolist()[0]
                captured['supplied_mask']=kwargs['attention_mask'].detach().cpu().tolist()[0]
                event('backbone..input',dict(args=args_,kwargs=kwargs))
            def first(_module,_args,kwargs):captured['effective_mask']=kwargs['attention_mask'].detach().cpu().tolist()[0]
            handles.append(backbone.layers[0].register_forward_pre_hook(first,with_kwargs=True))
            for name,layer in backbone.named_modules():
                if not name:continue
                def before(_m,args_,kwargs,name=name):event('backbone.'+name+'.input',dict(args=args_,kwargs=kwargs))
                def after(_m,_args,value,name=name):
                    event('backbone.'+name+'.output',value)
                    if name.endswith('.ffn.router'):
                        probabilities=value.detach().clone().softmax(dim=-1);weights,indices=probabilities.topk(backbone.config.top_k,dim=-1)
                        event('derived.'+name+'.probabilities',probabilities)
                        event('derived.'+name+'.topk_indices',indices)
                        event('derived.'+name+'.renormalized_weights',weights/(weights.sum(dim=-1,keepdim=True)+1e-9))
                        ordered=probabilities.sort(dim=-1,descending=True).values
                        event('derived.'+name+'.selection_margin',ordered[...,backbone.config.top_k-1]-ordered[...,backbone.config.top_k])
                handles.append(layer.register_forward_pre_hook(before,with_kwargs=True));handles.append(layer.register_forward_hook(after))
            for i,layer in enumerate(model):
                if i:handles.append(layer.register_forward_hook(lambda _m,_a,value,i=i:event('pipeline.'+str(i)+'.output',value)))
            original_sdpa=functional.scaled_dot_product_attention
            def sdpa(*args_,**kwargs):
                nonlocal sdpa_count
                name='actual.sdpa.'+str(sdpa_count);sdpa_count+=1
                capture(cid,name+'.input',dict(args=args_,kwargs=kwargs));value=original_sdpa(*args_,**kwargs);capture(cid,name+'.output',value);return value
            try:
                with torch.inference_mode():
                    functional.scaled_dot_product_attention=sdpa
                    with observer.observe_forward(backbone,input_capture,lambda value:event('backbone..output',value)):
                        observed=method(case['text'],output_value=None,show_progress_bar=False)
                    require(equal(ordinary[cid],observed),'original observer changed features')
            finally:
                functional.scaled_dot_product_attention=original_sdpa
                for handle in handles:handle.remove()
            require(all(counts.get('pipeline.'+str(i)+'.output')==1 for i in (1,2,3)),'pipeline observer occurrence count differs')
            require(sdpa_count==2 and captured['supplied_ids']==case['ids']and
                    list(map(int,captured['supplied_mask']))==case['supplied_mask']and
                    list(map(int,captured['effective_mask']))==case['effective_mask'],'actual original supplied/effective blueprint differs')
            with torch.inference_mode():
                repeated=method(case['text'],output_value=None,show_progress_bar=False)
                public=method(case['text'],show_progress_bar=False)
            require(equal(ordinary[cid],repeated)and torch.equal(public,ordinary[cid]['token_embeddings'][ordinary[cid]['attention_mask'].bool()]),'ordinary/repeat/public slice differs')
            names={r['stage']for r in report['arrays']if r['case']==cid}
            require(set(reference.stage_map().values())<=names,'mandatory mapped original stages missing')
            record.update(observer_neutral=True,repeat_exact=True,public_slice_exact=True,actual_sdpa_calls=2,
                          supplied_ids_exact=True,supplied_mask_exact=True,effective_mask_exact=True);save()
        report.update(completed=True,phase='three-document-original-complete',raw_bytes=total,
            limits='New intermediate observations anchored within unchanged frozen final budgets; exact old byte identity reported separately. No full-corpus recapture or new gold replacement.');save()
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise


if __name__=='__main__':
    def terminate(*_):raise RuntimeError('original diagnostic termination requested')
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for n in ('input','out','checkpoint','diff'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
