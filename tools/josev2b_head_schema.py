"""Strict schema derived from the pinned publisher's JointSchemaHead class.

No model/framework import or payload download. Shapes use PyTorch/NumPy order;
scalar parameters remain rank zero until the explicitly recorded GGUF adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'mayflowergmbh/Josev-2B'
REVISION = '6f0a31c8ed683aba79097eb3f48df2c9ab560afd'
ORIGINAL_MODULE_SHA = '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'
HEAD_CONFIG = dict(hidden_size=2048, width=512, routing_layers=2, layers=4, heads=8, feedforward=2048)


def tensor_schema(config=None):
    """Exact names, including MHA biases, three decoder norms and scalar scales."""
    if config is not None and config != HEAD_CONFIG:
        raise ValueError('Only the pinned Josev joint-head configuration is supported')
    h, w, f = 2048, 512, 2048
    rows = {}
    def add(name, *shape):
        if name in rows:
            raise ValueError('Duplicate expected tensor')
        rows[name] = list(shape)
    def norm(name, width):
        add(name+'.weight', width); add(name+'.bias', width)
    def linear(name, output, input_, bias=True):
        add(name+'.weight', output, input_)
        if bias:
            add(name+'.bias', output)
    def mha(name):
        add(name+'.in_proj_weight', 3*w, w)
        add(name+'.in_proj_bias', 3*w)
        linear(name+'.out_proj', w, w)
    norm('hidden_norm', h)
    for name in ['memory_projection', 'question_projection', 'option_question_projection',
                 'global_projection', 'option_context_projection', 'option_lexical_projection']:
        linear(name, w, h, False)
    add('type_embedding.weight', 3, w)
    for i in range(2):
        p = f'evidence_layers.{i}.'
        norm(p+'query_norm', w); norm(p+'memory_norm', w)
        mha(p+'attention'); norm(p+'feedforward_norm', w)
        linear(p+'feedforward.0', f, w); linear(p+'feedforward.3', w, f)
    norm('option_summary_norm', w)
    for i in range(4):
        p = f'layers.{i}.'
        mha(p+'self_attn'); mha(p+'multihead_attn')
        linear(p+'linear1', f, w); linear(p+'linear2', w, f)
        for j in range(1, 4):
            norm(p+f'norm{j}', w)
    norm('field_norm', w); norm('option_norm', w)
    linear('residual_scorer.0', w, 4*w); linear('residual_scorer.3', 1, w)
    for name in ['prior_logit_scale', 'joint_logit_scale', 'residual_gate']:
        add(name)
    return rows


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate safetensors JSON key')
        result[key] = value
    return result


def inventory(path):
    """Bounded header and exact contiguous BF16/F32 payload inventory, not inference."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 8 < path.stat().st_size <= 128*1024**2:
        raise ValueError('Bounded regular head checkpoint required')
    with path.open('rb') as stream:
        prefix = stream.read(8)
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 1024**2 or 8+length > path.stat().st_size:
            raise ValueError('Invalid head header size')
        header = json.loads(stream.read(length), object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError('Invalid head header object')
    rows, end = {}, 0
    tensors = [(name, row) for name, row in header.items() if name != '__metadata__']
    for name, row in tensors:
        if (not isinstance(row, dict) or not {'shape','data_offsets','dtype'} <= set(row)
                or not isinstance(row['data_offsets'], list) or len(row['data_offsets']) != 2
                or any(type(x) is not int for x in row['data_offsets'])):
            raise ValueError('Invalid head tensor descriptor')
    for name, row in sorted(tensors, key=lambda item: item[1]['data_offsets'][0]):
        shape, offsets, dtype = row['shape'], row['data_offsets'], row['dtype']
        if (not isinstance(shape, list) or any(type(x) is not int or x <= 0 for x in shape)
                or dtype not in ('BF16', 'F32') or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets) or offsets[0] != end
                or offsets[1]-offsets[0] != math.prod(shape)*(2 if dtype == 'BF16' else 4)):
            raise ValueError('Invalid head dtype, shape or contiguous tensor extent: '+name)
        end = offsets[1]
        rows[name] = dict(shape=shape, dtype=dtype, offsets=offsets)
    if not rows or 8+length+end != path.stat().st_size:
        raise ValueError('Head checkpoint has missing or trailing payload bytes')
    return 8+length, rows


def checked_schema(rows):
    if {name: row['shape'] for name, row in rows.items()} != tensor_schema():
        raise ValueError('Exact 122-tensor joint-head schema required')
    return rows


def checked_readback(shape, payload, expected_numpy_shape, expected_sha256):
    """Check GGUF ne dimensions and float bits independently of numeric equality."""
    if list(shape) != list(reversed(expected_numpy_shape)):
        raise ValueError('Written head dimensions differ from declared original adapter')
    if (len(payload) != 4*math.prod(expected_numpy_shape)
            or hashlib.sha256(payload).hexdigest() != expected_sha256):
        raise ValueError('Written head F32 payload bytes differ from actual loaded original')


def reference(path, digest, source):
    """Require an explicit reviewed receipt and immutable original worker source."""
    if not digest or sha256(path) != digest:
        raise ValueError('Exact reviewed original receipt SHA required')
    if not isinstance(source, str) or len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('Full immutable original worker source required')
    report = json.loads(Path(path).read_text())
    raw = subprocess.check_output(['git', 'show', source+':docs/josev2b-source-contract.json'], cwd=ROOT)
    cfg = json.loads(raw)
    if (cfg['repository'] != MODEL or cfg['revision'] != REVISION
            or cfg['files']['joint_schema_model.py']['sha256'] != ORIGINAL_MODULE_SHA
            or report.get('schema') != cfg['schema'] or report.get('status') != 'completed'
            or report.get('conclusive') is not True or report.get('native_outputs') is not False
            or report.get('git_commit') != source or report.get('model_revision') != REVISION
            or report.get('contract_sha256') != hashlib.sha256(raw).hexdigest()
            or report.get('panel_sha256') != cfg['panel']['sha256']
            or len(report.get('cases', [])) != cfg['panel']['cases']):
        raise ValueError('Original completion/source/model/panel differs')
    original_panel = json.loads(subprocess.check_output(
        ['git','show',source+':'+cfg['panel']['path']],cwd=ROOT))
    if ([case.get('id') for case in report['cases']] != [case['id'] for case in original_panel['cases']]
            or any(case.get('observer_bit_exact') is not True or case.get('repeat_bit_exact') is not True
                   for case in report['cases'])):
        raise ValueError('Complete neutral original case panel required')
    for name, expected in report.get('source', {}).items():
        measured = subprocess.check_output(['git', 'show', source+':'+name], cwd=ROOT)
        if hashlib.sha256(measured).hexdigest() != expected:
            raise ValueError('Original driver source differs')
    required = {'docs/josev2b-source-contract.json', cfg['panel']['path'],
                'tools/dump_josev2b_reference.py', 'tools/josev2b_contract.py',
                'tools/josev2b_environment.py', cfg['environment_lock']['path'],
                cfg['environment_lock']['requirements_path']}
    if set(report.get('source', {})) != required:
        raise ValueError('Original driver source coverage differs')
    weights = report.get('weights', {})
    if (weights.get('head_identity_exact') is not True or weights.get('head_tensor_count') != 122
            or weights.get('head_state_elements') != 30472708
            or weights.get('head_checkpoint_sha256') != cfg['files']['joint_head.safetensors']['sha256']):
        raise ValueError('Complete actual loaded-head identity proof required')
    proof = weights.get('head_tensors', {})
    if {name: row.get('shape') for name, row in proof.items()} != tensor_schema():
        raise ValueError('Actual loaded original head inventory required')
    for name, row in proof.items():
        if (row.get('checkpoint_dtype') not in ('bfloat16', 'float32', 'torch.bfloat16', 'torch.float32', 'BF16', 'F32')
                or row.get('loaded_dtype') not in ('float32', 'torch.float32', 'F32')
                or row.get('checkpoint_promoted_f32_sha256') != row.get('loaded_f32_sha256')
                or not isinstance(row.get('loaded_f32_sha256'), str)
                or len(row['loaded_f32_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['loaded_f32_sha256'])):
            raise ValueError('Promoted checkpoint/actual loaded original head mismatch: '+name)
    return report, cfg, proof
