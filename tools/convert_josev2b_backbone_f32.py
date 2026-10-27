#!/usr/bin/env python3
"""Exact text-only F32 checkpoint component; hosted conversion, no inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import struct
import tempfile
from josev2b_head_schema import reference, sha256, unique_pairs

PREFIX='model.language_model.'
ARCH='josev2b_backbone_reference'
# Exact auxiliary namespace in the authenticated original index. The official
# Clef path calls model.language_model, which consumes none of these tensors.
MTP_KEYS=frozenset(('mtp.fc.weight', 'mtp.norm.weight', 'mtp.pre_fc_norm_embedding.weight', 'mtp.pre_fc_norm_hidden.weight', 'mtp.layers.0.input_layernorm.weight', 'mtp.layers.0.post_attention_layernorm.weight', 'mtp.layers.0.mlp.down_proj.weight', 'mtp.layers.0.mlp.gate_proj.weight', 'mtp.layers.0.mlp.up_proj.weight', 'mtp.layers.0.self_attn.q_proj.weight', 'mtp.layers.0.self_attn.k_proj.weight', 'mtp.layers.0.self_attn.v_proj.weight', 'mtp.layers.0.self_attn.o_proj.weight', 'mtp.layers.0.self_attn.q_norm.weight', 'mtp.layers.0.self_attn.k_norm.weight'))


def schema():
    result={'embed_tokens.weight':[248320,2048],'norm.weight':[2048]}
    for i in range(24):
        p=f'layers.{i}.'
        result.update({p+'input_layernorm.weight':[2048],p+'post_attention_layernorm.weight':[2048],
            p+'mlp.gate_proj.weight':[6144,2048],p+'mlp.up_proj.weight':[6144,2048],p+'mlp.down_proj.weight':[2048,6144]})
        if i%4==3:
            result.update({p+'self_attn.q_proj.weight':[4096,2048],p+'self_attn.k_proj.weight':[512,2048],
                p+'self_attn.v_proj.weight':[512,2048],p+'self_attn.o_proj.weight':[2048,2048],
                p+'self_attn.q_norm.weight':[256],p+'self_attn.k_norm.weight':[256]})
        else:
            result.update({p+'linear_attn.in_proj_qkv.weight':[6144,2048],p+'linear_attn.in_proj_z.weight':[2048,2048],
                p+'linear_attn.in_proj_a.weight':[16,2048],p+'linear_attn.in_proj_b.weight':[16,2048],
                p+'linear_attn.out_proj.weight':[2048,2048],p+'linear_attn.conv1d.weight':[6144,1,4],
                p+'linear_attn.A_log':[16],p+'linear_attn.dt_bias':[16],p+'linear_attn.norm.weight':[128]})
    return {PREFIX+k:v for k,v in result.items()}


def backbone_inventory(path,expected_bytes):
    """Full-checkpoint extent, bounded header; head-only128MiB policy is separate."""
    path=Path(path)
    if (type(expected_bytes)is not int or not 8<expected_bytes<=8*2**30 or path.is_symlink() or
            not path.is_file() or path.stat().st_size!=expected_bytes):
        raise ValueError('Exact bounded regular original backbone checkpoint required')
    with path.open('rb') as stream:
        raw=stream.read(8)
        if len(raw)!=8:raise ValueError('Missing original backbone header')
        length=struct.unpack('<Q',raw)[0]
        if not 2<=length<=1024*1024 or 8+length>=expected_bytes:
            raise ValueError('Original backbone header bound differs')
        header=json.loads(stream.read(length),object_pairs_hook=unique_pairs)
    if not isinstance(header,dict):raise ValueError('Original backbone header object required')
    entries=[]
    for name,row in header.items():
        if name=='__metadata__':continue
        if (not isinstance(name,str) or not name or not isinstance(row,dict) or
                set(row)!={'shape','dtype','data_offsets'} or not isinstance(row['shape'],list) or
                len(row['shape'])>8 or any(type(n)is not int or not 1<=n<=1048576 for n in row['shape']) or
                row['dtype'] not in ('BF16','F32') or not isinstance(row['data_offsets'],list) or
                len(row['data_offsets'])!=2 or any(type(n)is not int or n<0 for n in row['data_offsets'])):
            raise ValueError('Invalid bounded original backbone tensor descriptor')
        entries.append((name,row))
    result={};end=0
    for name,row in sorted(entries,key=lambda item:item[1]['data_offsets'][0]):
        a,b=row['data_offsets'];width=2 if row['dtype']=='BF16' else 4
        if a!=end or b-a!=math.prod(row['shape'])*width or b>expected_bytes-8-length:
            raise ValueError('Original backbone tensor gap/overlap/extent differs')
        result[name]=dict(shape=row['shape'],dtype=row['dtype'],offsets=[a,b]);end=b
    if not result or 8+length+end!=expected_bytes:
        raise ValueError('Original backbone trailing/missing payload extent')
    return 8+length,result


def checked_text(rows,index,filename):
    expected=schema()
    selected={k:v for k,v in rows.items() if k.startswith(PREFIX)}
    if {k:v['shape'] for k,v in selected.items()}!=expected:
        raise ValueError('Exact320tensor original text backbone required')
    weight_map=index.get('weight_map')
    if not isinstance(weight_map,dict) or set(weight_map)!=set(rows) or any(v!=filename for v in weight_map.values()):
        raise ValueError('Original single-shard index and tensor inventory disagree')
    # The original model ties output embeddings. Never silently select an
    # additional vocabulary matrix or a different source tensor namespace.
    if ('lm_head.weight' in rows or {k for k in rows if k.startswith('mtp.')}!=MTP_KEYS or
            any(not k.startswith((PREFIX,'model.visual.')) and k not in MTP_KEYS for k in rows)):
        raise ValueError('Unexpected untied head, auxiliary MTP set or checkpoint namespace')
    return selected


def chunk_hash(array):
    view=memoryview(array).cast('B');h=hashlib.sha256()
    for start in range(0,len(view),4*1024*1024):h.update(view[start:start+4*1024*1024])
    return h.hexdigest()


def promote(stream,offset,row,target,np):
    width=2 if row['dtype']=='BF16' else 4
    remaining=math.prod(row['shape']);source_hash=hashlib.sha256();result_hash=hashlib.sha256()
    stream.seek(offset+row['offsets'][0])
    with target.open('xb') as out:
        while remaining:
            n=min(remaining,1024*1024)
            raw=stream.read(n*width)
            if len(raw)!=n*width:raise ValueError('Truncated original text tensor')
            source_hash.update(raw)
            value=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4') if width==2 else np.frombuffer(raw,dtype='<f4')
            if not np.isfinite(value).all():raise ValueError('Nonfinite original text tensor')
            payload=value.tobytes(order='C');out.write(payload);result_hash.update(payload);remaining-=n
    return source_hash.hexdigest(),result_hash.hexdigest()


def convert(checkpoint,index_path,original_report,original_sha,original_source,output):
    report,cfg,_=reference(original_report,original_sha,original_source)
    checkpoint,index_path,output=Path(checkpoint),Path(index_path),Path(output)
    receipt=output.with_suffix('.conversion.json')
    if output.exists() or receipt.exists():raise ValueError('Fresh output required; preserve earlier artifacts')
    filename='model.safetensors-00001-of-00001.safetensors'
    for name,path in ((filename,checkpoint),('model.safetensors.index.json',index_path)):
        pin=report['snapshot'][name]
        if path.is_symlink() or path.stat().st_size!=pin['bytes'] or sha256(path)!=pin['sha256']:
            raise ValueError('Original checkpoint/index byte identity differs')
    offset,rows=backbone_inventory(checkpoint,report['snapshot'][filename]['bytes'])
    selected=checked_text(rows,json.loads(index_path.read_text()),filename)
    if report['weights']['embedding_tied'] is not True or sys.byteorder!='little':
        raise ValueError('Original tied text-only/little-endian profile required')
    # Guards above precede optional frameworks and payload transformations.
    import numpy as np
    import gguf
    output.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='josev-backbone-f32-',dir=output.parent) as scratch:
        mapped=[]
        writer=gguf.GGUFWriter(str(output),arch=ARCH)
        writer.add_name('Josev-2B original F32 text-only backbone reference')
        writer.add_string('general.license','apache-2.0')
        writer.add_string(ARCH+'.source_revision',cfg['revision'])
        writer.add_string(ARCH+'.checkpoint_sha256',report['snapshot'][filename]['sha256'])
        writer.add_string(ARCH+'.original_receipt_sha256',original_sha)
        writer.add_string(ARCH+'.original_worker_source',original_source)
        writer.add_uint32(ARCH+'.hidden_size',2048);writer.add_uint32(ARCH+'.intermediate_size',6144)
        writer.add_uint32(ARCH+'.layers',24);writer.add_uint32(ARCH+'.max_validated_tokens',512)
        with checkpoint.open('rb') as stream:
            for i,name in enumerate(schema()):
                row=selected[name];target=Path(scratch)/f'{i:04d}.f32'
                source_sha,f32_sha=promote(stream,offset,row,target,np)
                value=np.memmap(target,mode='r',dtype='<f4',shape=tuple(row['shape']))
                mapped.append(value)
                tensor='l.'+name.removeprefix(PREFIX)
                writer.add_tensor(tensor,value,raw_dtype=gguf.GGMLQuantizationType.F32)
                records.append(dict(source=name,target=tensor,shape=row['shape'],source_dtype=row['dtype'],
                    source_bytes_sha256=source_sha,f32_sha256=f32_sha,
                    transformation='exact BF16 bit promotion' if row['dtype']=='BF16' else 'identity'))
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        actual={t.name:t for t in gguf.GGUFReader(str(output)).tensors}
        if set(actual)!={r['target'] for r in records}:raise ValueError('Written text tensor inventory differs')
        for row in records:
            tensor=actual[row['target']]
            if (tensor.tensor_type!=gguf.GGMLQuantizationType.F32 or list(tensor.shape)!=list(reversed(row['shape'])) or
                    tensor.data.nbytes!=4*math.prod(row['shape']) or chunk_hash(tensor.data)!=row['f32_sha256']):
                raise ValueError('Written text shape/type/F32 byte identity differs')
    result=dict(schema='crispembed-josev-backbone-f32-conversion-v1',completed=True,full_model=False,
        native_parity=False,representative_quality=False,original_receipt_sha256=original_sha,
        original_worker_source=original_source,checkpoint_sha256=report['snapshot'][filename]['sha256'],
        gguf_bytes=output.stat().st_size,gguf_sha256=sha256(output),tensor_count=len(records),
        parameter_count=sum(math.prod(v) for v in schema().values()),tensors=records,
        excluded_original_tensors=[dict(name=name,shape=row['shape'],dtype=row['dtype'],
            reason='Auxiliary MTP outside original language_model call' if name in MTP_KEYS else 'Unused conditional vision component')
            for name,row in rows.items() if name not in selected],
        limits='Source checkpoint byte identity and exact F32 promotion/readback; no original loaded-backbone hash probe or native inference claim.')
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','index','original-report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--original-sha256',required=True);p.add_argument('--original-source',required=True)
    args=p.parse_args()
    convert(args.checkpoint,args.index,args.original_report,args.original_sha256,args.original_source,args.output)


if __name__=='__main__':main()
