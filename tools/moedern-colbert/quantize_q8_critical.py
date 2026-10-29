#!/usr/bin/env python3
"""Hosted actual-C-quantizer Q8_0 package; all3 reviewed MoEdern packed F32 sources.

Only metadata/profile admission changes; no Python weight quantizer or padding.
"""
import argparse
import hashlib
import json
import math
import numbers
import operator
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'tools'))
from quantize_gguf import quantizable
from quantize_q4k_padded import PROFILES, publication
from q8_source_contract import verify
from gguf_file_type_canonicalize import parse, span, sha

FORMAT = 'crispembed-moedern-q8_0-f32-critical-v1'
RECIPE = 'q8_0_f32_critical_v1'
Q8_0_FILE_TYPE = 7
Q8_COUNTS = {'1.5M': 20, '27M': 20, '115M': 52}
ALLOWED_METADATA = {'general.file_type', 'general.quantization_version',
                    'moedern.packed.format', 'modernbert_hash_moe_v10.quant_recipe'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def q8_file_type(value):
    if isinstance(value, bool):
        return False
    try:
        return operator.index(value) == Q8_0_FILE_TYPE
    except TypeError:
        return False


def plan(name, shape):
    require(isinstance(name, str) and name and isinstance(shape, (list, tuple)) and 1 <= len(shape) <= 4
            and all(isinstance(x, numbers.Integral) and not isinstance(x, bool) and 0 < x <= 30002 for x in shape),
            'bounded positive semantic axes required')
    shape = list(map(int, shape))
    selected = quantizable(name) and len(shape) >= 2
    require(not selected or shape[0] % 32 == 0, 'Q8_0 role must have32-aligned semantic row; no fallback')
    count = math.prod(shape)
    return dict(name=name, shape=shape, type='Q8_0' if selected else 'F32',
                type_id=8 if selected else 0, bytes=count // 32 * 34 if selected else count * 4,
                semantic_parameters=count)


def command(quantizer, source, output, rows):
    require(any(r['type'] == 'Q8_0' for r in rows), 'genuine Q8 roles required')
    args = [str(quantizer), str(source), str(output), 'q8_0',
            '--metadata-string', 'moedern.packed.format=' + FORMAT,
            '--metadata-string', 'modernbert_hash_moe_v10.quant_recipe=' + RECIPE]
    for row in rows:
        if row['type'] == 'F32':
            args += ['--tensor-type', row['name'] + '=f32']
    return args


def metadata_readback(before, after):
    a, b = before['metadata'], after['metadata']
    require(set(b) == set(a) | {'general.file_type', 'general.quantization_version'}, 'metadata key set changed')
    for key in set(a) - ALLOWED_METADATA:
        require(a[key]['kind'] == b[key]['kind'] and a[key]['raw_sha256'] == b[key]['raw_sha256'],
                'critical raw metadata changed: ' + key)
    for key, value in [('moedern.packed.format', FORMAT), ('modernbert_hash_moe_v10.quant_recipe', RECIPE)]:
        require(key in a and a[key]['kind'] == 8 and b[key]['kind'] == 8 and b[key]['value'] == value,
                'existing typed string profile override differs')
    require(b['general.file_type']['kind'] == 4 and q8_file_type(b['general.file_type']['value'])
            and b['general.quantization_version']['kind'] == 4 and b['general.quantization_version']['value'] == 2,
            'standard Q8_0 UINT32 header7/QNT_VERSION2 required')


def quantize(source, output, variant, quantizer):
    kaggle = (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()
              and os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch', 'Interactive'))
    require(os.environ.get('GITHUB_ACTIONS') == 'true' or kaggle, 'hosted CPU execution only')
    require(variant in PROFILES, 'explicit reviewed variant required')
    source, output, quantizer = Path(source), Path(output), Path(quantizer)
    pin = publication(variant); cfg = PROFILES[variant]
    require(source.is_file() and not source.is_symlink() and source.stat().st_size == pin['bytes'] and sha(source) == pin['sha256'],
            'immutable published packed F32 source differs')
    require(quantizer.is_file() and not quantizer.is_symlink() and not output.exists() and not output.is_symlink(), 'real quantizer/fresh output required')
    before = parse(source, storage_types={0: (1, 4)})
    require(before['metadata']['general.architecture']['value'] == 'modernbert_hash_moe_v10'
            and before['metadata']['modernbert_hash_moe_v10.source_revision']['value'] == cfg['revision']
            and before['metadata']['moedern.packed.format']['value'] == cfg['format']
            and before['metadata']['modernbert_hash_moe_v10.quant_recipe']['value'] == 'f32', 'strict original F32 source profile differs')
    rows = [plan(name, t['stored_shape']) for name, t in before['tensors'].items()]
    require(len(rows) == cfg['tensors'] and sum(r['semantic_parameters'] for r in rows) == cfg['parameters']
            and sum(r['type'] == 'Q8_0' for r in rows) == Q8_COUNTS[variant], 'exact original semantic/role inventory differs')
    # All source validation above is stdlib and precedes optional array imports.
    import numpy as np
    import gguf
    original = gguf.GGUFReader(source)
    source_by_name = {t.name: t for t in original.tensors}
    require(set(source_by_name) == set(before['tensors']), 'source reader names disagree')
    with source.open('rb') as f:
        for row in rows:
            t = before['tensors'][row['name']]
            row['original_f32_sha256'] = span(f, t['data_offset'], t['bytes'])
            values = np.asarray(source_by_name[row['name']].data).reshape(-1)
            require(values.dtype == np.dtype('<f4'), 'exact F32 source reader required')
            for start in range(0, len(values), 262144):
                require(np.isfinite(values[start:start+262144]).all(), 'nonfinite source weight')
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(output.suffix + '.quantize.log')
    require(not log.exists() and not log.is_symlink(), 'fresh quantizer log required')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    with log.open('wb') as stream:
        subprocess.run(command(quantizer, source, output, rows), stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=1200, env=env)
    require('(fallback ' not in log.read_text(encoding='utf-8'), 'quantizer fallback is not Q8 qualification')
    after = parse(output, storage_types={0: (1, 4), 8: (32, 34)})
    metadata_readback(before, after)
    require(set(after['tensors']) == set(before['tensors']), 'output tensor names differ')
    result = gguf.GGUFReader(output); result_by_name = {t.name: t for t in result.tensors}
    with output.open('rb') as f:
        for row in rows:
            tensor = after['tensors'][row['name']]
            require(tensor['type'] == row['type_id'] and tensor['stored_shape'] == row['shape']
                    and tensor['bytes'] == row['bytes'], 'actual Q8/F32 type/axes/extent differs: ' + row['name'])
            row['stored_payload_sha256'] = span(f, tensor['data_offset'], tensor['bytes'])
            if row['type'] == 'F32':
                require(row['stored_payload_sha256'] == row['original_f32_sha256'], 'critical F32 payload changed')
                continue
            encoded = np.asarray(result_by_name[row['name']].data).reshape(-1, row['shape'][0] // 32 * 34)
            values = np.asarray(source_by_name[row['name']].data).reshape(-1, row['shape'][0])
            maximum, sum_square, count = 0., 0., 0
            for start in range(0, len(encoded), 256):
                decoded = gguf.quants.dequantize(encoded[start:start+256], gguf.GGMLQuantizationType.Q8_0)
                require(decoded.shape == values[start:start+256].shape and np.isfinite(decoded).all(), 'finite Q8 row readback required')
                delta = decoded.astype('float64') - values[start:start+256].astype('float64')
                maximum = max(maximum, float(np.abs(delta).max())); sum_square += float(np.sum(delta * delta)); count += delta.size
            row.update(max_absolute_weight_error=maximum, rms_weight_error=math.sqrt(sum_square / count),
                       weight_error_is_not_output_quality=True)
    require(output.stat().st_size < source.stat().st_size and sha(source) == pin['sha256'], 'source mutated or Q8 package not smaller')
    return dict(schema='crispembed-moedern-q8-f32-critical-v1', completed=True, variant=variant, recipe=RECIPE,
                source=pin, output_sha256=sha(output), output_bytes=output.stat().st_size, general_file_type=7,
                actual_q8_tensors=Q8_COUNTS[variant], critical_f32_tensors=len(rows)-Q8_COUNTS[variant], tensors=rows,
                quantizer_sha256=sha(quantizer), quantizer_log_sha256=sha(log), unchanged_critical_payloads=True,
                unchanged_semantic_and_physical_shapes=True, allowed_metadata_changes=sorted(ALLOWED_METADATA),
                raw_other_metadata_byte_exact=True, full_f32_qualified=False, quantized_publication_qualified=False,
                quality_required='all5483 rows/eight Ref and independent panels/all300 rankings/all public clients after F32 blockers resolved')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'output', 'quantizer', 'report']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--variant', choices=list(PROFILES), required=True)
    p.add_argument('--reviewed-source', required=True)
    a = p.parse_args(); result = dict(schema='crispembed-moedern-q8-f32-critical-v1', completed=False, variant=a.variant)
    try:
        closure = verify(a.reviewed_source)
        result = quantize(a.input, a.output, a.variant, a.quantizer)
        result['source_closure'] = closure
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.report.with_name(a.report.name + '.pending')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); temporary.replace(a.report)
