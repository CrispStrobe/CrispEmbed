"""Bounded canonical projector comparisons, no model inference or quantizer."""
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


ROWS_SHA = 'd3dc056b29d3b2b57b50297c40928d69a3b4a215cbfde434d5db2a87cc2cca3c'

PINS = {'upstream_commit': '6c01da6278ab6c69ed67d9124806b50d2b299ba7', 'publisher_revision': '8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01', 'native_base_url': 'https://huggingface.co/cstr/LightOnOCR-3-4B-GGUF-experimental-6b9dfe91eaa2/resolve/a7ec2e54103165a96b8bb6dc3478a2887edd7459/lightonocr3-4b-q4_k-vision-f32.gguf', 'projector_url': 'https://huggingface.co/richardyoung/LightOnOCR-3-4B-GGUF/resolve/8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01/mmproj-LightOnOCR-3-4B-F16.gguf', 'native_base_bytes': 4030957952, 'form_receipt_sha256': '1458d07e6098b4b1b8fa078c18fe0b636f832a0d1219f789b035e1a033ef4ba0', 'native_pack_receipt_sha256': '920b4d5db276414bdd743fcc80a7ed1033ed1c0ca3b76859ef922aa667de9dee', 'projector_file': 'mmproj-LightOnOCR-3-4B-F16.gguf'}

def admit(cfg):
    require({key:cfg.get(key) for key in PINS}==PINS, 'closed source/provenance/receipt pins')
    require(hashlib.sha256(json.dumps(cfg['tensors'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==ROWS_SHA, 'frozen tensor mapping/offset/hash manifest')
    require(cfg['schema']=='lighton4b-projector-canonical/v1' and len(cfg['tensors'])==297, 'fixed projector domain')
    require(cfg['native_vision_range']==[2696900992,4030957951] and
            cfg['maximum_download_bytes']==2100000000 and cfg['worker_seconds']==3600 and
            cfg['threads']==4 and cfg['build_threads']==2, 'closed resource/domain bounds')
    require(cfg['projector_bytes']==672423200 and cfg['projector_sha256']=='21eae623f60580f19ae4f04716292cd76ad36b420ee01e761254c8b1ec966321' and
            cfg['native_base_sha256']=='f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91', 'immutable model inputs')
    rows=cfg['tensors'];names=[r['native_name']for r in rows];up=[s['name']for r in rows for s in r['upstream']]
    require(len(set(names))==297 and len(up)==len(set(up))==298 and
            sum(len(r['upstream'])==2 for r in rows)==1, 'complete bijective split domain')
    require(sum(r['native_bytes']for r in rows)==1334056960 and
            min(r['native_offset']for r in rows)==2696900992 and
            max(r['native_offset']+r['native_bytes']for r in rows)==4030957952, 'complete native suffix')
    split=next(r for r in rows if len(r['upstream'])==2)
    require(split['native_name']=='v.patch_embed.proj.weight' and split['native_shape']==[1536,1024], 'temporal split only')


def temporal_join(slices):
    import numpy as np
    require(len(slices)==2 and slices[0].shape==slices[1].shape and slices[0].ndim==4, 'two [out,channel,h,w] temporal slices')
    return np.stack(slices,axis=2).reshape(-1).astype('<f4')


def compare(native, upstream, chunk=65536):
    import numpy as np
    require(native.size==upstream.size and native.size>0, 'equal canonical domain')
    mismatch=half_mismatch=bf16_exact=zero_sign=half_numeric=0;max_abs=half_max=0.;sum_sq=0.;samples=[]
    for start in range(0,native.size,chunk):
        a=np.asarray(native[start:start+chunk],dtype='<f4');b=np.asarray(upstream[start:start+chunk],dtype='<f4')
        require(np.isfinite(a).all()and np.isfinite(b).all(),'finite canonical parameters')
        mismatch+=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')))
        rounded=a.astype('<f2').astype('<f4');different=rounded.view('<u4')!=b.view('<u4')
        half_mismatch+=int(np.count_nonzero(different));half_numeric+=int(np.count_nonzero(rounded!=b))
        zero_sign+=int(np.count_nonzero(different&(rounded==0)&(b==0)));half_max=max(half_max,float(abs(rounded.astype('f8')-b.astype('f8')).max()))
        for index in np.flatnonzero(different)[:max(0,8-len(samples))]:
            i=int(index);samples.append(dict(index=start+i,native=float(a[i]),rounded_f16=float(rounded[i]),source=float(b[i]),native_bits=int(a.view('<u4')[i]),rounded_bits=int(rounded.view('<u4')[i]),source_bits=int(b.view('<u4')[i])))
        bf16_exact+=int(np.count_nonzero((a.view('<u4')&0xffff)==0))
        d=a.astype('f8')-b.astype('f8');sum_sq+=float(np.sum(d*d));max_abs=max(max_abs,float(abs(d).max()))
    return dict(elements=int(native.size),different_f32_bits=mismatch,different_after_native_f16_round=half_mismatch,
                different_after_f16_round_numeric=half_numeric,zero_sign_only_after_f16_round=zero_sign,max_abs_after_f16_round=half_max,f16_round_mismatch_samples=samples,
                native_exact_bf16_elements=bf16_exact,max_abs=max_abs,rms=float((sum_sq/native.size)**.5),bit_exact=mismatch==0)


def audit(cfg,native_suffix,projector):
    import numpy as np
    admit(cfg);require(Path(projector).stat().st_size==cfg['projector_bytes']and sha(projector)==cfg['projector_sha256'],'whole projector SHA')
    start,end=cfg['native_vision_range'];require(Path(native_suffix).stat().st_size==end-start+1,'complete native suffix bytes')
    results={}
    for row in cfg['tensors']:
        offset=row['native_offset']-start
        with Path(native_suffix).open('rb')as f:
            f.seek(offset);remaining=row['native_bytes'];h=hashlib.sha256()
            while remaining:
                b=f.read(min(1<<20,remaining));require(bool(b),'native span complete');h.update(b);remaining-=len(b)
        require(h.hexdigest()==row['native_sha256'],'authenticated native tensor '+row['native_name'])
        a=np.memmap(native_suffix,mode='r',dtype='<f4',offset=offset,shape=(row['native_bytes']//4,))
        parts=[]
        for s in row['upstream']:
            require(s['type']in(0,1)and 0<=s['offset']<s['offset']+s['bytes']<=cfg['projector_bytes'],'source tensor range/type')
            dtype='<f2'if s['type']==1 else'<f4';parts.append(np.memmap(projector,mode='r',dtype=dtype,offset=s['offset'],shape=tuple(reversed(s['shape']))))
        b=temporal_join(parts)if len(parts)==2 else parts[0].reshape(-1)
        result=compare(a,b);types=[s['type']for s in row['upstream']];require(types in([0],[1],[1,1]),'declared precision domain')
        result.update(native_sha256=row['native_sha256'],source_names=[s['name']for s in row['upstream']],source_types=types,transform=row['transform'],source_precision_transform_bit_exact=(result['bit_exact']if types==[0]else result['different_after_native_f16_round']==0));results[row['native_name']]=result
        del a,b,parts
    return dict(native_tensors=297,upstream_tensors=298,tensors=results,all_canonical_f32_bit_exact=all(r['bit_exact']for r in results.values()),
                all_source_precision_transforms_bit_exact=all(r['source_precision_transform_bit_exact']for r in results.values()),
                all_native_f16_round_equal=all(r['different_after_native_f16_round']==0 for r in results.values()),all_native_f16_round_numeric_equal=all(r['different_after_f16_round_numeric']==0 for r in results.values()),
                original_checkpoint_identity_proven=False,scope='These two admitted projector payloads only; no assertion about unpublished prequantization checkpoints.')
