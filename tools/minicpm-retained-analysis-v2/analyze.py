#!/usr/bin/env python3
"""Zero-forward17-arm correction; first merger LIST element maps only overview."""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis_contract as c
import prior
from common import resources,numerical_environment,authenticate_old
import retention
from graph_panel import verify_reference,reference_controls
spec=importlib.util.spec_from_file_location('_frozen_19c_analysis',c.ROOT/'tools/minicpm-retained-v3/analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
VISION=old.VISION


def view_plan(original_shape,native_shape,patches,tokens,width,merger=False):
 c.require(patches and len(patches)==len(tokens)and all(type(x)is int and x>0 for x in patches+tokens),'Positive exact part geometry required')
 if not merger:
  expected=[sum(patches),width];c.require(original_shape in [expected,[1,*expected]]and native_shape==expected,'Full concatenated tower axes required')
  return dict(shape=expected,native_rows=[0,sum(patches)],unmapped=[])
 # Pinned MiniCPMOcrMerger returns a list in target_sizes order. Frozen keep()
 # saves only output[0]; it never captured the other list tensors.
 c.require(original_shape==[tokens[0],width]and native_shape==[sum(tokens),width],'Original first merger list element/native full row extents differ')
 spans=[];start=tokens[0]
 for index,n in enumerate(tokens[1:],1):
  spans.append(dict(part=index,native_rows=[start,start+n],shape=[n,width],reference_parity_claimed=False));start+=n
 c.require(start==sum(tokens),'Exact merger prefix/suffix partition required')
 return dict(shape=[tokens[0],width],native_rows=[0,tokens[0]],unmapped=spans)


def finish(result,path):
 # Derive completion from actual stable keys/coverage, never a caller boolean.
 catalog=json.loads(c.inherited.CATALOG.read_text());expected={tuple(p[k]for k in ['id','dtype','graph'])for p in catalog['cases']}
 rows=result.get('cases',[]);keys=[tuple(p.get(k)for k in ['id','dtype','graph'])for p in rows]
 carried=[p for p in rows if isinstance(p.get('carried'),dict)];fresh=[p for p in rows if p.get('carried')is False and p.get('completed')is True]
 comparisons=sum(len(p.get('vision',{}).get('stages',{}))for p in rows)
 complete=(len(rows)==23 and len(set(keys))==23 and set(keys)==expected and len(carried)==6 and len(fresh)==17 and
           all(p.get('completed')is True and p.get('pixels_bit_exact')is True and set(p.get('vision',{}).get('stages',{}))=={'vision_'+str(i)for i in range(6)}for p in rows)and
           all(p['id']=='line'for p in carried)and comparisons==138)
 result.update(completed=complete,carried_arms=len(carried),new_arms=len(fresh),comparisons=comparisons,carried_comparisons=6*len(carried),new_comparisons=6*len(fresh))
 # Always publish the final atomic structural classification BEFORE rejecting.
 c.atomic(path,result);c.require(complete,'Structural retained analysis incomplete; atomic partial receipt preserved')


def run(args):
 args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=True)
 result=dict(schema='crispembed-minicpm-retained-analysis-v2',completed=False,native_forwards=0,model_accessed=False,original_rerun=False,qualified=False,cases=[],carried_arms=0,new_arms=0)
 try:
  result['source']=c.source_guard(args.reviewed_source);cfg=c.declaration();result['resources']=resources(args.out,args.scratch)
  built=json.loads(args.build_receipt.read_text());c.require(built['completed']and built['mode']=='analysis'and built['source']['source']==args.reviewed_source and c.sha(args.diff)==built['binary_sha256']['diff'],'Same-source model-free Ref build identity differs')
  result['build_receipt_sha256']=c.sha(args.build_receipt);result['tooling']=numerical_environment()
  catalog,panel,retained=authenticate_old(args.retained);reference,identity=verify_reference(args.reference);result['original']=identity
  initial,members=prior.authenticate(cfg,args.out/'prior');result['prior_analysis_sha256']=hashlib.sha256(members['analysis/analysis.json']).hexdigest()
  result['old_panel_sha256']=c.sha(args.retained/'panel.json');c.require(result['old_panel_sha256']==initial['old_panel_sha256'],'Original23 frozen panel changed')
  result['ref_controls']=reference_controls(args.diff,args.out/'ref-controls')
  # Reuse the actual old compaction proof and keep6.8GB canonical payloads at
  # their immutable prior artifact. No second full compaction/copy is needed.
  result['canonical_parent']=dict(artifact=cfg['prior']['canonical'],reconstruction_manifest_sha256=initial['compaction']['reconstruction_sha256'],new_canonical_copy=False,immutable_remote_preserved=True)
  import numpy as np
  refs={r['id']:r for r in reference['cases']};previous={(r['id'],r['dtype'],r['graph']):(i,r)for i,r in enumerate(initial['cases'])}
  for ordinal,item in enumerate(retained):
   key=tuple(item['pin'][k]for k in ['id','dtype','graph']);old_index,old_row=previous[key]
   if old_row['completed']:
    c.require(key[0]=='line','Only six completed line arms may be carried');row=copy.deepcopy(old_row)
    row['carried']=dict(producer_run=cfg['prior_service']['run'],producer_checkout=c.SOURCE,original_row_canonical_json_sha256=hashlib.sha256(json.dumps(old_row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),new_comparisons=0)
    result['cases'].append(row);result['carried_arms']+=1
    for name in ['comparison.json','diff.json','diff.log','reconstruction.json','cleanup.json']:
     target=args.out/str(ordinal)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(members[f'analysis/{old_index}/'+name])
    c.atomic(args.out/'analysis.json',result);continue
   row=dict(id=key[0],dtype=key[1],graph=key[2],completed=False,carried=False);result['cases'].append(row)
   try:
    case=refs[row['id']];folder=item['folder'];meta=json.loads((folder/'dump/native_meta.json').read_text());parts=meta['parts']
    sizes=np.load(args.reference/case['arrays']['inputs.target_sizes']['path'],allow_pickle=False);c.require(sizes.dtype==np.dtype('<i4'),'Original integer target_sizes required');old.pixel_plan(parts,sizes.tolist())
    patches=[p['w']//14*(p['h']//14)for p in parts];tokens=[p['tokens']for p in parts];packed=[]
    for n,p in enumerate(parts):
     pixels=retention.npy(folder/f'dump/native_pixels_{n}.npy');c.require(list(pixels.shape)==[3,p['h'],p['w']],'Native CHW axes differ')
     packed.append(pixels.reshape(3,p['h']//14,14,p['w']//14,14).transpose(0,2,1,3,4).reshape(3,14,-1))
    pixels=np.concatenate(packed,axis=2)[None];target=retention.npy(args.reference/case['arrays']['inputs.pixel_values']['path'],case['arrays']['inputs.pixel_values'])
    c.require(target.shape==pixels.shape,'Original packed pixel axes differ');row['pixels_bit_exact']=bool(np.array_equal(target.view(np.uint32),pixels.view(np.uint32)));c.require(row['pixels_bit_exact'],'Frozen native/original packed pixels differ')
    pairs=[];tmp=args.scratch/str(ordinal);tmp.mkdir(exist_ok=False);row['unmapped_native_merger_parts']=[]
    for index,label in enumerate(VISION):
     original=args.reference/case['arrays'][label]['path'];native=folder/'dump'/(label.replace('.','_')+'.npy')
     a=retention.npy(original,case['arrays'][label]);b=retention.npy(native);merger=label=='model.merger';width=1024 if merger else 1152
     plan=view_plan(list(a.shape),list(b.shape),patches,tokens,width,merger)
     a=a if a.ndim==2 else a[0];start,end=plan['native_rows'];mapped=b[start:end]
     c.require(list(a.shape)==list(mapped.shape)==plan['shape'],'Exact paired logical shapes required')
     for span in plan['unmapped']:
      first,last=span['native_rows'];value=b[first:last];values=value.astype(np.float64)
      row['unmapped_native_merger_parts'].append(dict(**span,overview=False,grid_row=parts[span['part']]['row'],grid_col=parts[span['part']]['col'],source_npy_sha256=c.sha(native),source_shape=list(b.shape),raw_span_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest(),bytes=int(value.nbytes),finite=bool(np.isfinite(value).all()),minimum=float(value.min()),maximum=float(value.max()),absolute_maximum=float(np.abs(value).max()),l2=float(np.linalg.norm(values)),rms=float(np.sqrt(np.mean(values*values))),reason='Original hook selected only first merger output list element; no independent original tensor for this part'))
     transform='Exact overview part0 prefix rows[0,'+str(tokens[0])+'); later parts unmapped'if merger else'C-order F32 full concatenated patch rows'
     if merger:row['merger_mapping']=dict(original_shape=list(a.shape),native_source_shape=list(b.shape),native_mapped_rows=plan['native_rows'],mapped_part=0,overview=True,list_element=0,source_policy='First list tensor from first module invocation; no original replication/padding')
     pairs.append(dict(name='vision_'+str(index),original=retention.raw(tmp/f'original_{index}.f32',a),native=retention.raw(tmp/f'native_{index}.f32',mapped),reconstruction=dict(original=retention.anchor(original,case['arrays'][label]['path'],'Known singleton batch removed for full tower; merger LIST[0] unchanged'),native=retention.anchor(native,'dump/'+native.name,transform))))
    measured=retention.compare(pairs,args.diff,tmp/'adapters',args.out/str(ordinal))
    row.update(completed=True,vision=measured,first_bit_difference=next((dict(stage=VISION[i],**v)for i,v in enumerate(measured['stages'].values())if not v['bit_exact']),None),first_strict_failure=next((dict(stage=VISION[i],**v)for i,v in enumerate(measured['stages'].values())if not v['passed']),None))
    for value in measured['stages'].values():value['max_absolute_over_original_abs_max']=value['max_absolute']/max(value['original_abs_max'],1e-30)
    result['new_arms']+=1
   except Exception as error:row['error']=type(error).__name__+': '+str(error)
   c.atomic(args.out/'analysis.json',result)
  comparisons=sum(len(row.get('vision',{}).get('stages',{}))for row in result['cases'])
  result.update(completed=len(result['cases'])==23 and all(row['completed']for row in result['cases'])and result['carried_arms']==6 and result['new_arms']==17 and comparisons==138,comparisons=comparisons,carried_comparisons=36,new_comparisons=6*result['new_arms'],all_pixels_bit_exact=all(row.get('pixels_bit_exact')for row in result['cases']),historical_numeric_failures_preserved=True,limits='Zero forwards/models. Six existing line arms/36Ref carried;17 missing arms/102Ref only. Five tower stages retain full concatenated rows; merger original LIST[0] maps only native overview, later spans explicitly unreferenced. Numerical failures remain; structural completion is not qualification.')
 finally:c.atomic(args.out/'analysis.json',result)
 finish(result,args.out/'analysis.json')

if __name__=='__main__':
 import signal
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(SystemExit('Analysis terminated; preserve atomic receipt')))
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed-source',required=True)
 for name in ['retained','reference','scratch','out','diff','build-receipt']:p.add_argument('--'+name,type=Path,required=True)
 run(p.parse_args())
