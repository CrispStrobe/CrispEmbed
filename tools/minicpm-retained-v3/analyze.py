#!/usr/bin/env python3
"""Retained arrays only: actual Ref vision attribution, no model/forward access."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'tools'),str(ROOT/'tools/minicpm-ocr')]
from common import atomic,require,sha,source_guard,resources,numerical_environment,authenticate_old
import retention
from graph_panel import verify_reference,reference_controls

VISION=['model.vision_tower.embeddings']+[f'model.vision_tower.encoder.layers.{i}'for i in [0,6,26]]+['model.vision_tower.post_layernorm','model.merger']


def aligned_shape(shape,lengths,width,merger=False):
    expected=[sum(lengths),width]
    require(shape==expected or(not merger and shape==[1,*expected]),'Only frozen concatenated rows/known batch1 may align; no heuristic padding')
    return expected


def pixel_plan(parts,target):
    expected=[[p['h']//14,p['w']//14]for p in parts]
    require(target==expected and all(p['w']%28==0 and p['h']%28==0 for p in parts),'Exact original target_sizes/part geometry required')
    require(parts[0]['overview']is True and all(not p['overview']for p in parts[1:]),'Overview-first part order differs')
    require(all(p['tokens']==h*w//4 for p,(h,w)in zip(parts,target)),'Merger token geometry differs')
    return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['retained','reference','scratch','out','diff']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--build-receipt',type=Path,required=True)
    parser.add_argument('--reviewed-source',required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=True)
    result=dict(completed=False,native_forwards=0,model_accessed=False,original_rerun=False,qualified=False,cases=[])
    try:
        result['source']=source_guard('analysis',args.reviewed_source);result['resources']=resources(args.out,args.scratch)
        built=json.loads(args.build_receipt.read_text());require(built['completed']and built['mode']=='analysis'and built['source']['source']==args.reviewed_source,'Exact same-source model-free build receipt required')
        require(all(sha(getattr(args,key))==built['binary_sha256'][key]for key in ['diff']),'Built binary identity differs')
        result['build_receipt_sha256']=sha(args.build_receipt)
        result['tooling']=numerical_environment()
        catalog,panel,retained=authenticate_old(args.retained)
        reference,identity=verify_reference(args.reference);result['original']=identity
        result['old_panel_sha256']=sha(args.retained/'panel.json')
        result['ref_controls']=reference_controls(args.diff,args.out/'ref-controls')
        result['compaction']=retention.compact_old(retained,reference,args.reference,args.out/'retained-v3')
        __import__('shutil').copyfile(args.retained/'panel.json',args.out/'retained-v3/panel.json')
        __import__('shutil').copytree(args.retained/'initial',args.out/'retained-v3/initial')
        # Every original and canonical byte is authenticated above before the
        # optional numerical package is imported. No checkpoints are opened.
        import numpy as np
        refs={c['id']:c for c in reference['cases']}
        for item in retained:
            row=dict(id=item['pin']['id'],dtype=item['pin']['dtype'],graph=item['pin']['graph'],completed=False)
            result['cases'].append(row)
            try:
                case=refs[row['id']];folder=item['folder'];meta=json.loads((folder/'dump/native_meta.json').read_text());parts=meta['parts']
                sizes=np.load(args.reference/case['arrays']['inputs.target_sizes']['path'],allow_pickle=False)
                require(sizes.dtype==np.dtype('<i4'),'Integer original target_sizes required');pixel_plan(parts,sizes.tolist())
                patches=[p['w']//14*(p['h']//14)for p in parts];tokens=[p['tokens']for p in parts]
                packed=[]
                for n,p in enumerate(parts):
                    a=retention.npy(folder/f'dump/native_pixels_{n}.npy');require(list(a.shape)==[3,p['h'],p['w']],'Native CHW pixel axes differ')
                    # Exact publisher unfold packing: concatenate per-image
                    # [C,14,L*14] along final sequence axis, not flat-part order.
                    packed.append(a.reshape(3,p['h']//14,14,p['w']//14,14).transpose(0,2,1,3,4).reshape(3,14,-1))
                pixels=np.concatenate(packed,axis=2)[None]
                target=retention.npy(args.reference/case['arrays']['inputs.pixel_values']['path'],case['arrays']['inputs.pixel_values'])
                require(target.shape==pixels.shape,'Original packed pixel axes differ')
                row['pixels_bit_exact']=bool(np.array_equal(target.view(np.uint32),pixels.view(np.uint32)))
                pairs=[];tmp=args.scratch/str(len(result['cases'])-1);tmp.mkdir(exist_ok=False)
                for index,label in enumerate(VISION):
                    original=args.reference/case['arrays'][label]['path'];native=folder/'dump'/ (label.replace('.','_')+'.npy')
                    a=retention.npy(original,case['arrays'][label]);b=retention.npy(native)
                    lengths=tokens if label=='model.merger'else patches;width=1024 if label=='model.merger'else 1152
                    shape=aligned_shape(list(a.shape),lengths,width,label=='model.merger');require(list(b.shape)==shape,'Native concatenated vision axes differ')
                    a=a.reshape(shape)
                    pairs.append(dict(name='vision_'+str(index),original=retention.raw(tmp/f'original_{index}.f32',a),
                        native=retention.raw(tmp/f'native_{index}.f32',b),reconstruction=dict(
                            original=retention.anchor(original,case['arrays'][label]['path'],'known singleton batch1 removed; roworder unchanged'),
                            native=retention.anchor(native,'dump/'+native.name,'C-order F32 payload'))))
                measured=retention.compare(pairs,args.diff,tmp/'adapters',args.out/str(len(result['cases'])-1))
                row.update(completed=True,vision=measured,first_bit_difference=next((dict(stage=VISION[i],**v)for i,v in enumerate(measured['stages'].values())if not v['bit_exact']),None),first_strict_failure=next((dict(stage=VISION[i],**v)for i,v in enumerate(measured['stages'].values())if not v['passed']),None))
                for i,v in enumerate(measured['stages'].values()):v['max_absolute_over_original_abs_max']=v['max_absolute']/max(v['original_abs_max'],1e-30)
            except Exception as error:row['error']=type(error).__name__+': '+str(error)
            atomic(args.out/'analysis.json',result)
        result.update(completed=all(c['completed']for c in result['cases'])and len(result['cases'])==23,
            comparisons=sum(len(c.get('vision',{}).get('stages',{}))for c in result['cases']),historical_numeric_failures_preserved=True,
            all_pixels_bit_exact=all(c.get('pixels_bit_exact')for c in result['cases']),
            limits='Retained arrays only. Relative errors are context, never replacement gates. No attribution to a backend or numerical/default promotion.')
    finally:atomic(args.out/'analysis.json',result)


if __name__=='__main__':
    import signal
    def stop(_signal,_frame):raise SystemExit('Hosted analysis termination requested')
    signal.signal(signal.SIGTERM,stop)
    main()
