#!/usr/bin/env python3
"""Quantize a reviewed MoEdern F32 GGUF to Q8_0 or Q4_0 (diagnostic speed arm).

  quantize_gguf.py IN-F32.gguf OUT.gguf q8_0|q4_0

Only the large blk.* matmul weights the engine's quantizable() accepts are
quantized: attention q/k/v/output, routed and shared experts, conv in/out.
Embeddings (hash table and its projection), the router, the dense head, norms,
biases and the depthwise conv are copied as F32 byte-for-byte. A tensor whose
row length is not a multiple of the 32-wide block also stays F32. Metadata is
copied unchanged plus a provenance key. Loading the result needs
CRISPEMBED_MOEDERN_FAST=1 and CRISPEMBED_MOEDERN_QUANT=1.
"""
from __future__ import annotations

import hashlib
import json
import sys

QUANTIZABLE = ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight', 'ffn_up_exps.weight',
               'ffn_gate_exps.weight', 'ffn_down_exps.weight', 'ffn_up_shexp.weight', 'ffn_gate_shexp.weight',
               'ffn_down_shexp.weight', 'conv_in.weight', 'conv_out.weight')
BLOCK = 32


def quantizable(name: str) -> bool:
    return name.startswith('blk.') and any(name.endswith('.' + s) for s in QUANTIZABLE)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def quantize(src: str, dst: str, kind: str) -> dict:
    import numpy as np
    import gguf
    qtype = {'q8_0': gguf.GGMLQuantizationType.Q8_0, 'q4_0': gguf.GGMLQuantizationType.Q4_0}[kind]
    reader = gguf.GGUFReader(src)
    arch = reader.fields['general.architecture'].contents()
    writer = gguf.GGUFWriter(dst, arch=arch)
    for field in reader.fields.values():
        if field.name.startswith('GGUF.') or field.name == 'general.architecture':
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
    writer.add_string(arch + '.crispembed_quantization', kind)
    writer.add_string(arch + '.crispembed_quantized_from_sha256', sha256(src))
    records = []
    for t in reader.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError('F32 input required: ' + t.name)
        a = np.asarray(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
        if quantizable(t.name) and a.shape[-1] % BLOCK == 0:
            q = gguf.quants.quantize(a, qtype)
            err = float(np.max(np.abs(gguf.quants.dequantize(q, qtype).reshape(a.shape) - a)))
            writer.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=qtype)
            records.append(dict(name=t.name, type=kind.upper(), max_abs_weight_error=err))
        else:
            writer.add_tensor(t.name, a, raw_dtype=gguf.GGMLQuantizationType.F32)
            records.append(dict(name=t.name, type='F32'))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dict(input_sha256=sha256(src), output_sha256=sha256(dst), quantization=kind,
                quantized=sum(r['type'] != 'F32' for r in records), kept_f32=sum(r['type'] == 'F32' for r in records),
                tensors=records)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('q8_0', 'q4_0'):
        sys.exit(__doc__)
    report = quantize(*sys.argv[1:])
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}))
