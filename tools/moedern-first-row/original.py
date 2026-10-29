"""One targeted original document, five explicitly counted forwards maximum."""
import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,PATH,CID,atomic,module,require,sha,verify,hosted,authenticated_input,actual_mapping


def fingerprint(value,torch):
    """Hash raw tensor bits (including signed zero); never torch.equal/repr fallback."""
    if isinstance(value,torch.Tensor):
        from torch.utils._python_dispatch import _disable_current_modes
        with _disable_current_modes():
            v=value.detach().clone().contiguous().cpu()
            data=v.numpy().tobytes()
        return dict(kind='tensor',dtype=str(v.dtype),shape=list(v.shape),stride=list(value.stride()),device=str(value.device),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    if isinstance(value,Mapping):return dict(kind='mapping',items=[[k,fingerprint(v,torch)]for k,v in value.items()if isinstance(k,str)]) if all(isinstance(k,str)for k in value)else invalid()
    if isinstance(value,(list,tuple)):return dict(kind=type(value).__name__,items=[fingerprint(v,torch)for v in value])
    if value is None or type(value)in(bool,int,str):return dict(kind=type(value).__name__,value=value)
    if type(value)is float:
        require(math.isfinite(value),'nonfinite scalar feature');return dict(kind='float64_bits',bits=struct.pack('<d',value).hex())
    raise ValueError('Unsupported feature/operation argument type: '+type(value).__name__)


def invalid():raise ValueError('Non-string feature mapping key')


def barriers(root,case):
    b=json.loads((root/'barriers.json').read_text())
    require(b['completed']is True and b['original_strict_anchor_passed']is True and b['native93_plain_anchor_exact']is True and
            b['native_plain_full_computations']==6,'both ordinary barriers must close BEFORE any observer')
    baseline=json.loads((root/'ordinary/original.json').read_text())
    require(b['original_report_sha256']==sha(root/'ordinary/original.json')and b['plain_report_sha256']==sha(root/'plain'/CID/'controls.json')and
            baseline['completed']is True and baseline['original_forward_completed']==1,'ordinary barrier identity differs')
    for kind in ('all','selected'):
        require(sha(root/'plain'/CID/(kind+'.f32'))==case['anchors']['native'][kind]['sha256'],'native barrier changed')
    return baseline


def run(args):
    c=verify(args.reviewed_source);hosted(args.reviewed_source);case=authenticated_input(args.input,c)
    require(sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,12),'isolated original CPython3.12 required')
    # Authenticate all retained bytes and both barriers before importing frameworks.
    baseline=barriers(args.barrier_root,case)if args.phase=='observe'else None
    sys.path[:0]=[str(ROOT/'tools/moedern-colbert'),str(ROOT/'tools/moedern-native-v2')]
    source=module(ROOT/'tools/moedern-colbert/scifact_original.py','one_row_original_loader')
    ref=module(ROOT/'tools/moedern-native-v2/replay.py','one_row_original_ref')
    oldobserver=module(ROOT/'tools/moedern-colbert/reference.py','one_row_original_hooks')
    import torch
    import numpy as np
    import torch.nn.functional as functional
    from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed-moedern-first-row-original-v1',source=args.reviewed_source,phase=args.phase,completed=False,
        original_forward_attempts=0,original_forward_completed=0,original_rerun=False,original_full_corpus_recaptured=False,
        publication_qualified=False,new_gold=False,arrays=[],calls=[],operators=[],sdpa_dispatches=[],publisher_router_source=c['publisher_router_source'],sdpa_dispatch_metadata_scope=c['sdpa_dispatch_metadata'])
    save=lambda:atomic(args.out/'original.json',report);save()
    def call(label,**kwargs):
        require(report['original_forward_attempts']<(1 if args.phase=='ordinary'else 4),'targeted original forward budget exceeded')
        report['original_forward_attempts']+=1;report['original_rerun']=True;report['calls'].append(dict(label=label,completed=False));save()
        with torch.inference_mode():result=model.encode_document(case['text'],show_progress_bar=False,**kwargs)
        report['original_forward_completed']+=1;report['calls'][-1]['completed']=True;save();return result
    try:
        model,report['model']=source.load_original('1.5M',args.checkpoint)
        require(report['model']==json.loads((ROOT/c['original_public_report']).read_text())['model'],'loaded original source/runtime/weights differ')
        report['hardware']=dict(threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),mkldnn_enabled=torch.backends.mkldnn.enabled)
        if args.phase=='ordinary':
            features=call('ordinary-barrier',output_value=None);report['feature_fingerprint']=fingerprint(features,torch)
            require(features['input_ids'].tolist()==case['ids']and list(map(int,features['attention_mask'].tolist()))==case['scoring_mask'],'ordinary original token/mask contract differs')
            pairs=[];metrics={};report['anchor']={}
            for kind,value in [('all',features['token_embeddings']),('selected',features['token_embeddings'][features['attention_mask'].bool()])]:
                frozen=case['anchors']['original'][kind];path=args.out/(kind+'.f32');data=value.detach().cpu().numpy()
                require(data.dtype==np.float32 and list(data.shape)==frozen['shape']and np.isfinite(data).all(),'ordinary original shape/dtype/finiteness differs')
                data.tofile(path);a=np.fromfile(args.input/frozen['path'],dtype='<f4').reshape(frozen['shape'])
                measured=ref.compare(a,data);report['anchor'][kind]=measured;metrics[kind]=measured
                pairs.append(dict(name=kind,original=dict(path=args.input/frozen['path'],shape=frozen['shape'],sha256=frozen['sha256']),native=dict(path=path,shape=frozen['shape'],sha256=sha(path))))
            report['anchor_adapters']=ref.write_pair(args.out/'anchor-original.gguf',args.out/'anchor-current.gguf',pairs)
            report['anchor_actual_ref']=ref.actual_ref(args.diff,args.out/'anchor-original.gguf',args.out/'anchor-current.gguf',args.out/'anchor-ref.json',args.out/'anchor-ref.log',metrics)
            require(report['anchor_actual_ref']['all_metrics_agree']and all(r['passed']for r in metrics.values()),'ordinary original strict frozen final anchor rejected; all observation forbidden')
            report.update(completed=True,strict_anchor_passed=True);save();return report
        backbone=model[0].model;expected=baseline['feature_fingerprint'];total=0
        def traced(label):
            nonlocal total
            counts={};operators=[];sdpa_dispatches=[];sdpa_active=[];rows=[];handles=[];stack=[];sdpa_count=0;seen={};raw= args.out/label;raw.mkdir()
            # Mutable trace inventories survive an early structural/operation failure.
            prefix=''if label=='observed'else'observed_repeat_'
            report[prefix+'arrays']=rows;report[prefix+'operators']=operators;report[prefix+'sdpa_dispatches']=sdpa_dispatches
            def capture(stage,tree):
                nonlocal total
                with _disable_current_modes():
                    for name,value in oldobserver.capture_leaves(stage,tree,lambda x:isinstance(x,torch.Tensor)):
                        v=value.detach().clone().contiguous().cpu();require(v.numel()>0,'empty captured stage')
                        if v.is_floating_point():
                            require(v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'observed nonfinite/nonF32 original');dtype='f32';data=v.numpy().astype('<f4',copy=False).tobytes()
                        elif v.dtype==torch.bool:dtype='u8';data=v.numpy().astype('u1').tobytes()
                        else:
                            require(v.dtype in(torch.int64,torch.int32),'unsupported observed integer');dtype='i64';data=v.numpy().astype('<i8').tobytes()
                        total+=len(data);require(total<=512*2**20,'two traces exceeded raw capture budget')
                        filename=hashlib.sha256(name.encode()).hexdigest()+'.'+dtype;p=raw/filename;require(not p.exists(),'duplicate original stage')
                        p.write_bytes(data);rows.append(dict(case=CID,stage=name,path=label+'/'+filename,dtype=dtype,shape=list(v.shape),bytes=len(data),sha256=sha(p)))
            def event(name,value):
                i=counts.get(name,0);counts[name]=i+1;capture(name+'#'+str(i),value)
            def input_capture(args_,kwargs):
                seen['supplied_ids']=kwargs['input_ids'].detach().cpu().tolist()[0];seen['supplied_mask']=kwargs['attention_mask'].detach().cpu().tolist()[0]
                event('backbone..input',dict(args=args_,kwargs=kwargs))
            def first(_m,_a,kwargs):seen['effective_mask']=kwargs['attention_mask'].detach().cpu().tolist()[0]
            handles.append(backbone.layers[0].register_forward_pre_hook(first,with_kwargs=True))
            for name,layer in backbone.named_modules():
                if name.endswith('.ffn'):
                    handles.append(layer.register_forward_pre_hook(lambda _m,_a,name=name:stack.append(name)))
            for name,layer in backbone.named_modules():
                if not name:continue
                handles.append(layer.register_forward_pre_hook(lambda _m,a,k,name=name:event('backbone.'+name+'.input',dict(args=a,kwargs=k)),with_kwargs=True))
                handles.append(layer.register_forward_hook(lambda _m,_a,v,name=name:event('backbone.'+name+'.output',v)))
            for name,layer in backbone.named_modules():
                if name.endswith('.ffn'):
                    def pop(_m,_a,_v,name=name):require(stack and stack.pop()==name,'FFN context nesting differs')
                    handles.append(layer.register_forward_hook(pop,always_call=True))
            for i,layer in enumerate(model):
                if i:handles.append(layer.register_forward_hook(lambda _m,_a,v,i=i:event('pipeline.'+str(i)+'.output',v)))
            class ActualOperations(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    kwargs={}if kwargs is None else kwargs
                    if sdpa_active:
                        require(len(sdpa_dispatches)<256,'bounded SDPA dispatch metadata exceeded')
                        sdpa_dispatches.append(dict(sequence=len(sdpa_dispatches),sdpa=sdpa_active[-1],operator=str(func),schema=str(func._schema)))
                    selected=str(func)in('aten._softmax.default','aten.topk.default','aten.div.Tensor')and bool(stack)
                    if selected:
                        context=stack[-1];kind={'aten._softmax.default':'softmax','aten.topk.default':'topk','aten.div.Tensor':'div'}[str(func)]
                        name='actual.'+context+'.'+kind;require(name not in counts,'unexpected repeated actual router operation')
                        before=fingerprint(dict(args=args,kwargs=kwargs),torch);capture(name+'.input',dict(args=args,kwargs=kwargs))
                    value=func(*args,**kwargs)
                    if selected:
                        counts[name]=1;capture(name+'.values'if kind=='topk'else name,value[0]if kind=='topk'else value)
                        if kind=='topk':capture(name+'.indices',value[1])
                        operators.append(dict(stage=name,operator=str(func),schema=str(func._schema),context=context,input=before,output=fingerprint(value,torch)))
                    return value
            original_sdpa=functional.scaled_dot_product_attention
            def sdpa(*a,**k):
                nonlocal sdpa_count
                name='actual.sdpa.'+str(sdpa_count);sdpa_count+=1
                before=fingerprint(dict(args=a,kwargs=k),torch);capture(name+'.input',dict(args=a,kwargs=k))
                # Record only dispatches made by the real SDPA call, excluding snapshot bookkeeping.
                sdpa_active.append(name)
                try:value=original_sdpa(*a,**k)
                finally:require(sdpa_active.pop()==name,'SDPA context nesting differs')
                capture(name+'.output',value)
                operators.append(dict(stage=name,operator='torch.nn.functional.scaled_dot_product_attention',input=before,output=fingerprint(value,torch)));return value
            try:
                functional.scaled_dot_product_attention=sdpa
                with ActualOperations(),oldobserver.observe_forward(backbone,input_capture,lambda v:event('backbone..output',v)):
                    observed=call(label,output_value=None)
                require(fingerprint(observed,torch)==expected,'actual-operation/module observation changed complete feature raw bits')
            finally:
                functional.scaled_dot_product_attention=original_sdpa
                for h in handles:h.remove()
            require(not stack and not sdpa_active and sdpa_count==2 and len(operators)==8,'actual two-layer softmax/topk/div/SDPA coverage differs')
            require({r['sdpa']for r in sdpa_dispatches}=={'actual.sdpa.0','actual.sdpa.1'},'actual ATen SDPA dispatch metadata missing')
            for field in ('supplied_ids','supplied_mask','effective_mask'):
                require(list(map(int,seen[field]))==case['ids'if field=='supplied_ids'else field],'actual original IDs/masks differ')
            names={r['stage']for r in rows};require(set(actual_mapping(ref).values())<=names,'actual mandatory stages missing')
            return rows,operators,sdpa_dispatches
        first,ops,dispatches=traced('observed');report['arrays']=first;report['operators']=ops;report['sdpa_dispatches']=dispatches;save()
        repeated,repeated_ops,repeated_dispatches=traced('observed-repeat')
        def identity(rows):return [{k:v for k,v in r.items()if k!='path'}for r in rows]
        require(identity(first)==identity(repeated)and ops==repeated_ops and dispatches==repeated_dispatches,'captured module/actual ATen/SDPA args/returns or dispatch metadata are not repeat neutral')
        repeated_features=call('unhooked-repeat',output_value=None)
        require(fingerprint(repeated_features,torch)==expected,'unhooked repeat feature bits differ')
        public=call('ordinary-public-selected')
        require(fingerprint(public,torch)==fingerprint(repeated_features['token_embeddings'][repeated_features['attention_mask'].bool()],torch),'public selected slice differs')
        report.update(completed=True,raw_bytes=total,observer_feature_raw_bits_neutral=True,repeat_feature_raw_bits_exact=True,
            actual_operator_inputs_outputs_repeat_raw_bits_exact=True,sdpa_dispatch_metadata_repeat_exact=True,observed_repeat_sdpa_dispatches=repeated_dispatches,public_slice_raw_bits_exact=True,observed_repeat_arrays=repeated,
            original_forward_total_including_barrier=5,derived_router_gold_used=False);save();return report
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True);p.add_argument('--phase',choices=('ordinary','observe'),required=True)
    for n in ('input','out','checkpoint','diff','barrier-root'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
