"""Zero-forward seven-case vision comparison, using exact frozen cec LIST policy."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import seven_contract as c
sys.path.insert(0,str(c.ROOT/'tools/minicpm-retained-analysis-v2'))
s=importlib.util.spec_from_file_location('_frozen_cec_vision',c.ROOT/'tools/minicpm-retained-analysis-v2/analyze.py');frozen=importlib.util.module_from_spec(s);s.loader.exec_module(frozen)
import retention

def compare(folder,case,reference,diff,scratch,out):
 import numpy as np
 meta=json.loads((folder/'dump/native_meta.json').read_text());parts=meta['parts']
 sizes=np.load(reference/case['arrays']['inputs.target_sizes']['path'],allow_pickle=False);c.require(sizes.dtype==np.dtype('<i4'),'Original target_sizes dtype required');frozen.old.pixel_plan(parts,sizes.tolist())
 patches=[p['w']//14*(p['h']//14)for p in parts];tokens=[p['tokens']for p in parts];packed=[]
 for i,p in enumerate(parts):
  pixels=retention.npy(folder/f'dump/native_pixels_{i}.npy');c.require(list(pixels.shape)==[3,p['h'],p['w']],'Native CHW axes differ')
  packed.append(pixels.reshape(3,p['h']//14,14,p['w']//14,14).transpose(0,2,1,3,4).reshape(3,14,-1))
 actual=np.concatenate(packed,axis=2)[None];target=retention.npy(reference/case['arrays']['inputs.pixel_values']['path'],case['arrays']['inputs.pixel_values'])
 c.require(actual.shape==target.shape and np.array_equal(actual.view(np.uint32),target.view(np.uint32)),'Original/native packed pixels must be bit exact')
 pairs=[];unmapped=[];mapping=None;scratch.mkdir(parents=True,exist_ok=False)
 for index,label in enumerate(frozen.VISION):
  original=reference/case['arrays'][label]['path'];native=folder/'dump'/(label.replace('.','_')+'.npy');a=retention.npy(original,case['arrays'][label]);b=retention.npy(native);merger=label=='model.merger';width=1024 if merger else 1152
  plan=frozen.view_plan(list(a.shape),list(b.shape),patches,tokens,width,merger);a=a if a.ndim==2 else a[0];start,end=plan['native_rows'];mapped=b[start:end]
  c.require(list(a.shape)==list(mapped.shape)==plan['shape'],'Exact original/native view shape required')
  for span in plan['unmapped']:
   first,last=span['native_rows'];value=b[first:last];v=value.astype(np.float64)
   unmapped.append(dict(**span,overview=False,grid_row=parts[span['part']]['row'],grid_col=parts[span['part']]['col'],source_npy_sha256=c.sha(native),source_shape=list(b.shape),raw_span_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest(),bytes=int(value.nbytes),finite=bool(np.isfinite(value).all()),minimum=float(value.min()),maximum=float(value.max()),absolute_maximum=float(np.abs(value).max()),l2=float(np.linalg.norm(v)),rms=float(np.sqrt(np.mean(v*v))),reason='Original hook captured only first merger LIST element; no original tensor for later native part'))
  transform='Exact part0 overview prefix; later parts unmapped'if merger else'Full concatenated patch rows; known batch1 removed'
  if merger:mapping=dict(original_shape=list(a.shape),native_source_shape=list(b.shape),native_mapped_rows=plan['native_rows'],mapped_part=0,list_element=0,overview=True)
  pairs.append(dict(name='vision_'+str(index),original=retention.raw(scratch/f'original_{index}.f32',a),native=retention.raw(scratch/f'native_{index}.f32',mapped),reconstruction=dict(original=retention.anchor(original,case['arrays'][label]['path'],'Frozen original bytes; only known tower batch1 removed'),native=retention.anchor(native,'dump/'+native.name,transform))))
 result=retention.compare(pairs,diff,scratch/'adapters',out)
 # Actual adapters were reconstituted and checked before retention.compare removed
 # them; remove only these new raw duplicate views after recording their hashes.
 for pair in pairs:
  for side in ['original','native']:
   p=Path(pair[side]['path']);c.require(p.parent==scratch and c.sha(p)==pair[side]['sha256'],'New scratch raw changed');p.unlink()
 return dict(completed=True,pixels_bit_exact=True,merger_mapping=mapping,unmapped_native_merger_parts=unmapped,vision=result,first_strict_failure=next((dict(stage=frozen.VISION[i],**v)for i,v in enumerate(result['stages'].values())if not v['passed']),None))
