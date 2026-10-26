#!/usr/bin/env python3
"""Lossless experimental F32 converter. Original reference must already be frozen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contract import ARCH, digest, read_contract, tensor_schema, validate_checkpoint


def convert(checkpoint, variant, outfile, original_report, original_report_sha256):
    # Deliberately enforce original-first before framework/model loading.
    if not original_report_sha256 or digest(original_report) != original_report_sha256:
        raise ValueError('reviewed original receipt SHA required')
    report = json.loads(Path(original_report).read_text())
    declaration = read_contract()['variants'][variant]
    if (report.get('schema') != 'crispembed.moedern-colbert.original.v1' or
            report.get('completed') is not True or report.get('variant') != variant or
            report.get('revision') != declaration['original']['revision'] or
            report.get('source_contract_sha256') != digest(Path(__file__).resolve().parents[2]/'docs/moedern-colbert-source-contract.json')):
        raise ValueError('original receipt identity/completion mismatch')
    identities, inventory = validate_checkpoint(checkpoint, variant)
    if report['checkpoint_files'] != identities:
        raise ValueError('original checkpoint identities differ')
    import numpy as np
    import gguf
    from safetensors.numpy import load_file

    cfg = declaration['config']
    data = {file: load_file(str(Path(checkpoint)/file)) for file in inventory}
    converted, records = [], []
    for row in tensor_schema(cfg):
        a = data[row['file']][row['source']]
        if a.dtype != np.float32 or list(a.shape) != row['shape'] or not np.isfinite(a).all():
            raise ValueError('invalid source dtype/shape/finiteness')
        if row['transform'] == 'expert_transpose':
            a = a.transpose(0, 2, 1)
        elif row['transform'] == 'depthwise_transpose':
            a = a[:, 0, :].T
        a = np.ascontiguousarray(a, dtype='<f4')
        import hashlib
        records.append({**row, 'gguf_shape_numpy': list(a.shape),
                        'gguf_data_sha256': hashlib.sha256(a.tobytes()).hexdigest()})
        converted.append((row['target'], a))
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(outfile), arch=ARCH)
    w.add_name(f'SauerkrautLM-MoEdernColBERT-{variant}-strict-F32')
    for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                'vocab_size', 'hash_projections', 'max_position_embeddings', 'num_routed_experts',
                'num_shared_experts', 'expert_dim', 'top_k']:
        w.add_uint32(f'{ARCH}.{key}', cfg[key])
    w.add_uint32(f'{ARCH}.head_dim', cfg['hidden_size']//cfg['num_attention_heads'])
    w.add_uint32(f'{ARCH}.out_dim', 128)
    for key, value in [('query_marker_token_id', 30000), ('document_marker_token_id', 30001), ('mask_token_id', 4)]:
        w.add_uint32(f'{ARCH}.{key}', value)
    for key, value in [('rope_theta', cfg['rope_theta']), ('norm_eps', cfg['norm_eps']),
                       ('hash_norm_eps', 1e-5), ('router_renorm_eps', 1e-9)]:
        w.add_float32(f'{ARCH}.{key}', value)
    if 'conv' in cfg['layer_types']:
        dilations = iter(cfg['conv_dilations'])
        values = [next(dilations) if k == 'conv' else 0 for k in cfg['layer_types']]
        for key, value in [('layer_kinds', [int(k == 'conv') for k in cfg['layer_types']]),
                           ('conv_dilations_per_layer', values)]:
            w.add_key_value(f'{ARCH}.{key}', value, gguf.GGUFValueType.ARRAY,
                            sub_type=gguf.GGUFValueType.UINT32)
        w.add_uint32(f'{ARCH}.conv_kernel_size', cfg['conv_kernel_size'])
        w.add_uint32(f'{ARCH}.conv_use_gate_norm', int(cfg['conv_use_gate_norm']))
    w.add_string(f'{ARCH}.hidden_activation', 'gelu_erf')
    w.add_string(f'{ARCH}.quant_recipe', 'f32')
    w.add_string('general.license', 'apache-2.0')
    w.add_string(f'{ARCH}.source_revision', declaration['original']['revision'])
    w.add_string(f'{ARCH}.tokenizer_sidecar_git_blob_sha1', declaration['original']['files']['tokenizer.json']['git_blob_sha1'])
    w.add_string(f'{ARCH}.tokenizer_sidecar_sha256', identities['tokenizer.json']['sha256'])
    for name, a in converted:
        w.add_tensor(name, a, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    # Independently read back every actual written payload; transpose metadata alone
    # cannot establish tensor-byte preservation.
    reader = gguf.GGUFReader(str(outfile))
    if len(reader.tensors) != len(converted):
        raise ValueError('GGUF tensor count differs')
    actual = {t.name: t for t in reader.tensors}
    for name, expected in converted:
        t = actual[name]
        if t.tensor_type != gguf.GGMLQuantizationType.F32 or not np.array_equal(t.data.reshape(expected.shape), expected):
            raise ValueError(f'GGUF roundtrip mismatch: {name}')
    receipt = {'schema': 'crispembed.moedern-colbert.conversion.v1', 'variant': variant,
               'completed': True, 'original_report_sha256': original_report_sha256,
               'gguf_sha256': digest(outfile), 'gguf_bytes': outfile.stat().st_size,
               'tensor_count': len(records), 'tensors': records,
               'tokenizer_required': True, 'tokenizer': identities['tokenizer.json']}
    outfile.with_suffix('.conversion.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--variant', choices=['1.5M', '27M', '115M'], required=True)
    p.add_argument('--outfile', type=Path, required=True)
    p.add_argument('--original-report', type=Path, required=True)
    p.add_argument('--original-report-sha256', required=True)
    a = p.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        p.error('model access is hosted-only')
    convert(a.checkpoint, a.variant, a.outfile, a.original_report, a.original_report_sha256)
