"""Three-row native old93 anchors first, then mapped-stage actual Ref diagnostics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,PATH,KEYS,atomic,module,require,sha,verify


def run(args):
    cfg=verify(args.reviewed_source);packet=json.loads((args.input/'input-packet.json').read_text());original=json.loads((args.original/'original.json').read_text())
    require(original['completed']is True and len(original['cases'])==3 and packet['keys']==KEYS,'anchored original three-document capture required')
    sys.path[:0]=[str(ROOT/'tools/moedern-native-v2')]
    ref=module(ROOT/'tools/moedern-native-v2/replay.py','three_doc_existing_comparator')
    import numpy as np
    args.out.mkdir(parents=True,exist_ok=False);result=dict(schema='crispembed-moedern-three-doc-diagnostic-v1',source=args.reviewed_source,
        completed=False,publication_qualified=False,full_original_recapture=False,cases=[],original_sha256=sha(args.original/'original.json'),
        original_final_anchor=original['anchor_actual_ref'],native93_final_anchor_exact=False,
        retained_historical_six_negative_matrices=cfg['historical_failures'])
    def save():atomic(args.out/'diagnostic.json',result)
    save()
    try:
        original_index={(r['case'],r['stage']):r for r in original['arrays']};native={}
        # Exact old93 native final bytes exclude a changed probe/build/route.
        for case in packet['cases']:
            cid=case['id'];meta=None
            for mode,root in (('plain',args.plain),('observed',args.observed)):
                folder=root/cid;current=json.loads((folder/'tokens.json').read_text());controls=json.loads((folder/'controls.json').read_text())
                require(all(current[k]==case[k]for k in ('ids','supplied_mask','effective_mask','scoring_mask','expansion_positions')),'native exact input/role contract differs')
                require(all(controls.get(k)is True for k in ('repeat','gate_recovery','role_bounds_recovery','context_ownership')),'native ordinary/repeat/ownership controls absent')
                if mode=='observed':require(controls.get('observer_neutrality')is True and controls.get('observer_recovery')is True and controls.get('stages')==55,'native observer controls absent')
                for kind in ('all','selected'):
                    row=case['anchors']['native'][kind];path=folder/(kind+'.f32')
                    require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'native93 exact '+kind+' final anchor failed: '+cid)
                all_values=np.fromfile(folder/'all.f32',dtype='<f4').reshape(-1,128)
                require(np.isfinite(all_values).all()and all_values[np.asarray(case['scoring_mask'],dtype=bool)].tobytes()==(folder/'selected.f32').read_bytes(),'native public selection is not exact mask slice')
                meta=current
            stages,files=ref.stages(args.observed/cid)
            native[cid]=(stages,files)
            inputs={name:np.asarray([case[key]],dtype='<i8')for name,key in [('input_ids','ids'),('supplied_mask','supplied_mask'),('effective_mask','effective_mask')]}
            require(all(np.array_equal(stages[k],v)for k,v in inputs.items()),'actual scalar engine inputs differ')
            require(set(stages)==set(cfg['native_stages'])and sha(args.observed/cid/'all.f32')==files['normalized']['sha256'],'actual55-stage/normalized final coverage differs')
        result['native93_final_anchor_exact']=True;save()
        mapping=ref.stage_map();require(len(mapping)==48,'frozen original/native map differs')
        for case in packet['cases']:
            cid=case['id'];stages,files=native[cid];record=dict(key=case['key'],id=cid,metrics=[],integer=[],unmapped=[],router=[],status='pending');result['cases'].append(record)
            pairs=[];independent={};ordered=[]
            for name,value in stages.items():
                if name in ('input_ids','supplied_mask','effective_mask'):continue
                if name not in mapping:
                    require(name in cfg['unmapped_native'],'unexpected unreferenced observation')
                    record['unmapped'].append(dict(stage=name,reason=cfg['unmapped_native'][name],shape=list(value.shape),
                        l2=float(np.linalg.norm(value.astype(np.float64))),min=float(value.min()),max=float(value.max()),
                        abs_max=float(np.abs(value).max()),finite=bool(np.isfinite(value).all()),reference_parity_claimed=False));continue
                target=original_index[cid,mapping[name]];path=args.original/target['path']
                require(sha(path)==target['sha256']and path.stat().st_size==target['bytes'],'new anchored original stage changed')
                a=np.fromfile(path,dtype='<i8'if target['dtype']=='i64'else'<f4').reshape(target['shape'])
                require(a.shape==value.shape,'matched stage logical shape differs: '+name)
                if name.endswith('.topk_indices'):
                    require(target['dtype']=='i64'and value.dtype==np.dtype('<i8'),'routing indices must remain integer')
                    rows=np.flatnonzero(np.any(a!=value,axis=-1));check=dict(stage=name,exact=bool(np.array_equal(a,value)),changed_rows=len(rows),first_changed_rows=rows[:16].tolist())
                    record['integer'].append(check);ordered.append(dict(stage=name,passed=check['exact'],bit_exact=check['exact']));continue
                require(target['dtype']=='f32'and value.dtype==np.dtype('<f4'),'actual F32 comparison only')
                measured=ref.compare(a,value);measured['bit_exact']=bool(np.array_equal(a.view(np.uint32),value.view(np.uint32)))
                measured['max_absolute_index']=[int(v)for v in np.unravel_index(np.abs(value.astype(np.float64)-a.astype(np.float64)).argmax(),a.shape)]
                record['metrics'].append(dict(stage=name,original_stage=mapping[name],**measured));ordered.append(dict(stage=name,**measured));independent[name]=measured
                pairs.append(dict(name=name,original=dict(path=path,shape=target['shape'],sha256=target['sha256']),native=files[name]))
            require(len(record['metrics'])==46 and len(record['integer'])==2 and len(record['unmapped'])==4,'explicit mapped/unmapped counts differ')
            for layer in range(2):
                p=f'layer.{layer}.ffn.';target=original_index[cid,mapping[p+'selection_margin']]
                margins=np.fromfile(args.original/target['path'],dtype='<f4').reshape(target['shape'])
                indices=next(v for v in record['integer']if v['stage']==p+'topk_indices')
                index_target=original_index[cid,mapping[p+'topk_indices']]
                original_indices=np.fromfile(args.original/index_target['path'],dtype='<i8').reshape(index_target['shape'])
                record['router'].append(dict(layer=layer,original_min_margin=float(margins.min()),native_min_margin=float(stages[p+'selection_margin'].min()),
                    changed_rows=indices['changed_rows'],rows=[dict(position=i,original_margin=float(margins[i]),native_margin=float(stages[p+'selection_margin'][i]),
                    original_indices=original_indices[i].tolist(),native_indices=stages[p+'topk_indices'][i].tolist())for i in indices['first_changed_rows']]))
            folder=args.out/cid;folder.mkdir();adapters=ref.write_pair(folder/'original.gguf',folder/'native.gguf',pairs)
            actual=ref.actual_ref(args.diff,folder/'original.gguf',folder/'native.gguf',folder/'ref.json',folder/'ref.log',independent)
            require(actual['all_metrics_agree'],'independent/actual Ref metrics disagree')
            for measured in record['metrics']:measured.update(independent[measured['stage']])
            for measured in ordered:
                if measured['stage']in independent:measured.update(independent[measured['stage']])
            record.update(actual_ref=actual,adapters=adapters,first_bit_difference=next((r for r in ordered if not r['bit_exact']),None),
                first_strict_failure=next((r for r in ordered if not r['passed']),None),
                status='numeric_rejection'if any(not r['passed']for r in ordered)else'within_frozen_budgets')
            save()
        result.update(completed=True,compared_f32_stages=138,integer_topk_stages=6,native_observations=165,
            limits='Three existing failed development documents only. Fresh original stage observations are anchored within unchanged old final budgets, not new frozen gold. Four native-only stages per document have no independent reference. No source fix, whole-corpus rerun, Q4 or publication qualification.');save()
    except Exception as error:result['error']=type(error).__name__+': '+str(error);save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for n in ('input','original','plain','observed','diff','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
