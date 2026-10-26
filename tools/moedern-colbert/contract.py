"""Dependency-free source identities and strict original-to-GGUF tensor schema.

Independent mapping from the pinned Apache-2.0 author sources. No model payloads,
framework imports or inference are needed to use these controls.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'docs/moedern-colbert-source-contract.json'
PANEL = ROOT / 'tests/fixtures/moedern-colbert/panel-v1.json'
ARCH = 'modernbert_hash_moe_v10'


def read_contract():
    return json.loads(CONTRACT.read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_file(path, identity):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != identity['bytes']:
        raise ValueError(f'file kind/size mismatch: {path.name}')
    if identity.get('sha256') and digest(path) != identity['sha256']:
        raise ValueError(f'SHA256 mismatch: {path.name}')
    if identity.get('git_blob_sha1'):
        h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != identity['git_blob_sha1']:
            raise ValueError(f'Git blob mismatch: {path.name}')
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def tensor_schema(cfg):
    """Exact original numpy-order shapes; transform only after shape validation."""
    h, p, v = cfg['hidden_size'], cfg['hash_projections'], cfg['vocab_size']
    e, he = cfg['num_routed_experts'], cfg['expert_dim']
    kv = cfg['num_key_value_heads'] * (h // cfg['num_attention_heads'])
    if (cfg['model_type'] != 'modernbert_hash_moe_hybrid' or
            cfg['hidden_activation'] != 'gelu' or cfg['num_shared_experts'] != 1 or
            len(cfg['layer_types']) != cfg['num_hidden_layers'] or
            h % cfg['num_attention_heads'] or
            cfg['num_attention_heads'] % cfg['num_key_value_heads'] or
            not 0 < cfg['top_k'] <= e):
        raise ValueError('unsupported architecture')
    rows = []

    def add(src, dst, shape, transform='identity', file='model.safetensors'):
        rows.append({'file': file, 'source': src, 'target': dst,
                     'shape': list(shape), 'transform': transform})

    add('embeddings.hash_embeddings.weight', 'token_embd.hash.weight', (v, p))
    add('embeddings.projection.weight', 'token_embd.proj.weight', (h, p))
    add('embeddings.norm.weight', 'token_embd.norm.weight', (h,))
    add('embeddings.norm.bias', 'token_embd.norm.bias', (h,))
    for i, kind in enumerate(cfg['layer_types']):
        src, dst = f'layers.{i}', f'blk.{i}'
        if kind == 'conv':
            add(src+'.mod.in_norm.weight', dst+'.conv_norm.weight', (h,))
            add(src+'.mod.in_proj.weight', dst+'.conv_in.weight', (3*h, h))
            if cfg['conv_use_gate_norm']:
                for gate in ['b', 'c']:
                    add(src+f'.mod.gate_norm_{gate}.weight', dst+f'.conv_gate_{gate}_norm.weight', (h,))
            add(src+'.mod.conv.weight', dst+'.conv_dw.weight', (h, 1, cfg['conv_kernel_size']), 'depthwise_transpose')
            add(src+'.mod.out_proj.weight', dst+'.conv_out.weight', (h, h))
        elif kind == 'attn_moe':
            add(src+'.attention.norm.weight', dst+'.attn_norm.weight', (h,))
            for name, out, width in [('Q', 'q', h), ('K', 'k', kv), ('V', 'v', kv), ('O', 'output', h)]:
                add(src+f'.attention.{name}_proj.weight', dst+f'.attn_{out}.weight', (width, h))
            add(src+'.ffn_norm.weight', dst+'.ffn_norm.weight', (h,))
            add(src+'.ffn.router.weight', dst+'.ffn_gate_inp.weight', (e, h))
            for name, shape in [('up', (e, h, he)), ('gate', (e, h, he)), ('down', (e, he, h))]:
                add(src+f'.ffn.W_{name}', dst+f'.ffn_{name}_exps.weight', shape, 'expert_transpose')
                add(src+f'.ffn.shared_{name}.weight', dst+f'.ffn_{name}_shexp.weight', (h, he) if name == 'down' else (he, h))
        else:
            raise ValueError('unsupported layer kind')
    add('norm.weight', 'output_norm.weight', (h,))
    add('linear.weight', 'dense_head.weight', (128, h), file='1_Dense/model.safetensors')
    if len({r['target'] for r in rows}) != len(rows):
        raise ValueError('duplicate target')
    return rows


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def safetensors_inventory(path):
    """Read bounded header only; reject gaps, aliases, extra bytes and non-F32."""
    path = Path(path)
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('truncated safetensors')
        n = struct.unpack('<Q', raw)[0]
        if not 2 <= n <= 16*1024*1024 or n + 8 > path.stat().st_size:
            raise ValueError('invalid safetensors header size')
        header = json.loads(f.read(n), object_pairs_hook=_unique_pairs)
    rows, end = {}, 0
    for name, value in sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                              key=lambda item: item[1]['data_offsets'][0]):
        shape, offsets = value['shape'], value['data_offsets']
        if (value['dtype'] != 'F32' or not shape or
                any(type(x) is not int or x <= 0 for x in shape) or
                len(offsets) != 2 or any(type(x) is not int for x in offsets) or
                offsets[0] != end or offsets[1]-offsets[0] != 4*math.prod(shape)):
            raise ValueError(f'invalid F32 tensor: {name}')
        end = offsets[1]
        rows[name] = {'shape': shape, 'dtype': 'F32', 'data_offsets': offsets}
    if not rows or 8+n+end != path.stat().st_size:
        raise ValueError('safetensors payload size mismatch')
    return rows


def validate_checkpoint(directory, variant):
    model = read_contract()['variants'][variant]
    directory = Path(directory)
    identities = {}
    for name, identity in model['original']['files'].items():
        identities[name] = verify_file(directory/name, identity)
    if json.loads((directory/'config.json').read_text()) != model['config']:
        raise ValueError('configuration mismatch')
    if json.loads((directory/'1_Dense/config.json').read_text()) != model['dense_config']:
        raise ValueError('Dense configuration mismatch')
    schema, inventory = tensor_schema(model['config']), {}
    for file in ['model.safetensors', '1_Dense/model.safetensors']:
        actual = safetensors_inventory(directory/file)
        expected = {r['source']: r['shape'] for r in schema if r['file'] == file}
        if {k: v['shape'] for k, v in actual.items()} != expected:
            raise ValueError(f'exact tensor set/shape mismatch: {file}')
        inventory[file] = actual
    return identities, inventory
