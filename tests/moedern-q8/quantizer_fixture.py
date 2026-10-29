#!/usr/bin/env python3
"""Hosted actual C quantizer; tiny authored storage, not a model-quality result."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('q8recipe', ROOT / 'tools/moedern-colbert/quantize_q8_critical.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


def fixture(quantizer, out):
    out.mkdir(parents=True, exist_ok=True)
    shapes = {'blk.0.attn_q.weight': [128, 2], 'blk.0.ffn_up_exps.weight': [128, 2, 2],
              'token_embd.hash.weight': [16, 2], 'token_embd.proj.weight': [16, 128],
              'blk.0.ffn_gate_inp.weight': [128, 8], 'dense_head.weight': [128, 2],
              'token_embd.norm.weight': [128], 'token_embd.norm.bias': [128],
              'blk.0.conv_dw.weight': [128, 3]}
    rows = [q.plan(n, s) for n, s in shapes.items()]
    def text(s):
        b = s.encode(); return struct.pack('<Q', len(b)) + b
    header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(rows), 3))
    for key, value in [('general.architecture', 'modernbert_hash_moe_v10'),
                       ('moedern.packed.format', 'crispembed-moedern-text-f32-v1'),
                       ('modernbert_hash_moe_v10.quant_recipe', 'f32')]:
        header += text(key) + struct.pack('<I', 8) + text(value)
    payload = bytearray(); values = {}
    for row in rows:
        name, shape = row['name'], row['shape']
        v = [((i % 31) - 15) / 32 for i in range(math.prod(shape))]
        raw = struct.pack('<' + 'f' * len(v), *v); values[name] = (v, raw)
        offset = (len(payload) + 31) // 32 * 32; payload += bytes(offset-len(payload))
        header += text(name) + struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', 0, offset)
        payload += raw
    header += bytes((-len(header)) % 32)
    source = out/'authored-f32.gguf'; source.write_bytes(header + payload)
    result = out/'authored-q8.gguf'; log = out/'quantizer.log'
    with log.open('wb') as f:subprocess.run(q.command(quantizer,source,result,rows),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
    before=q.parse(source,storage_types={0:(1,4)});after=q.parse(result,storage_types={0:(1,4),8:(32,34)})
    q.metadata_readback(before,after);records=[]
    with result.open('rb') as f:
        for row in rows:
            t=after['tensors'][row['name']];q.require(t['type']==row['type_id'] and t['stored_shape']==row['shape'] and t['bytes']==row['bytes'],'actual C type/shape/bytes differs')
            f.seek(t['data_offset']);raw=f.read(t['bytes']);original,source_raw=values[row['name']]
            if row['type']=='F32':q.require(raw==source_raw,'critical F32 raw bytes changed')
            else:
                decoded=[]
                for at in range(0,len(raw),34):
                    d=struct.unpack_from('<e',raw,at)[0];signed=struct.unpack_from('<32b',raw,at+2)
                    q.require(math.isfinite(d),'nonfinite actual Q8 scale')
                    decoded.extend(d*x for x in signed)
                for at in range(0,len(original),32):
                    block=original[at:at+32];d=max(map(abs,block))/127
                    stored=struct.unpack_from('<e',raw,(at//32)*34)[0]
                    bound=.5*d+abs(stored-d)*127+2**-22
                    q.require(max(abs(a-b)for a,b in zip(block,decoded[at:at+32]))<=bound,'independent Q8 half-scale/integer readback bound failed')
            records.append(dict(name=row['name'],type=row['type'],shape=row['shape'],payload_sha256=hashlib.sha256(raw).hexdigest(),critical_byte_exact=row['type']=='F32'))
    return dict(schema='crispembed-moedern-authored-q8-storage-v1',completed=True,model_qualified=False,actual_c_quantizer=True,
                quantizer_sha256=q.sha(quantizer),source_sha256=q.sha(source),output_sha256=q.sha(result),general_file_type=7,
                actual_q8_tensors=2,critical_f32_tensors=7,records=records,log_sha256=q.sha(log))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quantizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=dict(schema='crispembed-moedern-authored-q8-storage-v1',completed=False)
    try:result=fixture(a.quantizer,a.out)
    except Exception as error:result['error']=type(error).__name__+': '+str(error);raise
    finally:a.out.mkdir(parents=True,exist_ok=True);(a.out/'fixture.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
