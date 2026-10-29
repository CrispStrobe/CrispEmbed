#!/usr/bin/env python3
"""Predeclared protected Q8/Q4 storage; actual C quantizer, no original forwards.

Historical A/B/C converters and receipts remain immutable. All source values are
finite-checked; F32 payloads must match, F16 uses the independent integer oracle,
and packed scales must be finite. The C writer owns variable-length metadata.
"""
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import quantize_josev2b_mixed as previous
from gguf_file_type_canonicalize import parse

old = previous.old
SOURCE = old.SOURCE
PARAMETERS = old.PARAMETERS
FORMAT = 'crispembed-josev-packed-mixed-v2'
TYPES = {0: 'F32', 1: 'F16', 8: 'Q8_0', 12: 'Q4_K'}
COUNTS = {
    'q8-all': {0: 292, 1: 0, 8: 150, 12: 0},
    'q8-f32-down': {0: 316, 1: 0, 8: 126, 12: 0},
    'q4-last-f32-down': {0: 316, 1: 124, 8: 0, 12: 2},
    'q4-last-f16-down': {0: 292, 1: 148, 8: 0, 12: 2},
}
require = old.require
hosted = old.hosted
digest = old.digest
schema = old.schema


def recipe(rows, identity):
    require(identity in COUNTS, 'unknown predeclared precision recipe')
    projections = old.recipe(rows)  # Exact frozen 150 names and widths.
    plan = {}
    for name in rows:
        if projections[name] == 0:
            kind = 0
        elif identity == 'q8-all':
            kind = 8
        elif identity == 'q8-f32-down':
            kind = 0 if '.mlp.down_proj.' in name else 8
        elif '.mlp.down_proj.' in name and identity == 'q4-last-f32-down':
            kind = 0
        elif int(name.split('.')[2]) == 23 and any(
                token in name for token in ('.mlp.gate_proj.', '.mlp.up_proj.')):
            kind = 12
        else:
            kind = 1
        plan[name] = kind
    require({t: sum(v == t for v in plan.values()) for t in TYPES} == COUNTS[identity],
            'predeclared exact role counts differ')
    return plan


def checked_header(path, expected):
    value = parse(path)
    require(set(value['tensors']) == set(expected) and len(expected) == 442,
            'exact 442 tensor inventory required')
    require(sum(math.prod(s) for s in expected.values()) == PARAMETERS,
            'exact semantic parameter count required')
    for name, row in value['tensors'].items():
        require(row['stored_shape'] + [1] * (4 - row['storage_rank']) ==
                expected[name] + [1] * (4 - len(expected[name])),
                'semantic axes differ: ' + name)
    return value


def scalar(header, name, kind, expected):
    row = header['metadata'].get(name, {})
    require(row.get('kind') == kind and row.get('value') == expected,
            'typed metadata differs: ' + name)


def packed_hash(stream, row):
    kind = row['type']
    if kind != 8:
        return old.hash_range(stream, row['data_offset'], row['bytes'], q4k=kind == 12)
    import hashlib
    stream.seek(row['data_offset'])
    h = hashlib.sha256()
    left = row['bytes']
    while left:
        data = stream.read(min(left, 34 * 4096))
        require(data and len(data) % 34 == 0, 'truncated Q8_0 storage blocks')
        require(all(math.isfinite(scale) for (scale,) in struct.iter_unpack('<e32x', data)),
                'nonfinite Q8_0 scale')
        h.update(data)
        left -= len(data)
    return h.hexdigest()


def verify_conversion(source, output, expected, identity, *, vector=False):
    a, b = checked_header(source, expected), checked_header(output, expected)
    wanted = recipe(expected, identity)
    for header in (a, b):
        scalar(header, 'general.architecture', 8, 'josev2b_systemone')
        scalar(header, 'general.license', 8, 'apache-2.0')
    scalar(a, 'josev2b.packed.format', 8, old.OLD_FORMAT)
    scalar(a, 'general.file_type', 4, 0)
    scalar(b, 'josev2b.packed.format', 8, FORMAT)
    scalar(b, 'general.file_type', 4, 7 if identity.startswith('q8-') else 14)
    scalar(b, 'general.quantization_version', 4, 2)
    changed = {'general.file_type', 'general.quantization_version', 'josev2b.packed.format'}
    require(set(b['metadata']) == set(a['metadata']) | {'general.file_type', 'general.quantization_version'},
            'metadata names differ')
    for name in set(a['metadata']) - changed:
        require(a['metadata'][name]['raw_sha256'] == b['metadata'][name]['raw_sha256'],
                'component/tokenizer metadata changed: ' + name)
    records = []
    with Path(source).open('rb') as left, Path(output).open('rb') as right:
        for name in expected:
            x, y = a['tensors'][name], b['tensors'][name]
            require(x['type'] == 0 and y['type'] == wanted[name],
                    'actual recipe type differs: ' + name)
            original, half = previous.source_hash_and_half(
                left, x['data_offset'], x['bytes'], right if y['type'] == 1 else None,
                y['data_offset'], vector=vector)
            actual = packed_hash(right, y)
            require(y['type'] != 0 or actual == original, 'protected F32 bytes changed: ' + name)
            require(y['type'] != 1 or actual == half, 'independent F16 rounding differs: ' + name)
            records.append(dict(name=name, semantic_shape=expected[name],
                                source_storage_rank=x['storage_rank'], output_storage_rank=y['storage_rank'],
                                type=TYPES[y['type']], bytes=y['bytes'], source_f32_sha256=original,
                                stored_sha256=actual, critical_f32=not old.projection(name),
                                independent_ieee_f16_exact=y['type'] == 1))
    require(b['bytes'] < a['bytes'] and sum(r['critical_f32'] for r in records) == 292,
            'compression or critical inventory differs')
    return dict(recipe=identity, tensors=records,
                counts={TYPES[t]: n for t, n in COUNTS[identity].items()}, parameters=PARAMETERS,
                output_bytes=b['bytes'], general_file_type=b['metadata']['general.file_type']['value'],
                critical292_f32_byte_exact=True, independent_f16_rounding_exact=True,
                finite_source_and_packed_scales=True, semantic_shapes_unchanged=True,
                profile_stamped_by_c_quantizer=True, publication_qualified=False)


def quantize(source, output, executable, identity):
    hosted()
    source, output, executable = map(Path, (source, output, executable))
    require(identity in COUNTS and executable.is_file() and not output.exists() and not output.is_symlink(),
            'fresh reviewed recipe/quantizer required')
    require(source.is_file() and not source.is_symlink() and source.stat().st_size == SOURCE['bytes'] and
            digest(source) == SOURCE['sha256'], 'immutable published F32 differs')
    expected = schema()
    wanted = recipe(expected, identity)
    method = 'q8_0' if identity.startswith('q8-') else 'q4_k'
    target = 8 if method == 'q8_0' else 12
    args = [str(executable), str(source), str(output), method,
            '--metadata-string', 'josev2b.packed.format=' + FORMAT]
    for name, kind in wanted.items():
        if kind != target:
            args.extend(['--tensor-type', name + '=' + TYPES[kind].lower()])
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.quantize.log')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    from ci_process import run
    process = run(args, log, env=env, timeout=2400)
    log.with_name(log.name + '.process.json').write_text(json.dumps(process, indent=2) + '\n')
    require(not process['timed_out'] and process['exit'] == 0, 'actual C quantizer rejected; logs retained')
    require('(fallback ' not in log.read_text(), 'quantizer fallback forbidden')
    result = verify_conversion(source, output, expected, identity, vector=True)
    return dict(schema='crispembed-josev-protected-conversion-v1', completed=True, source=SOURCE,
                output_sha256=digest(output), quantizer_sha256=digest(executable),
                quantizer_log_sha256=digest(log), quantizer_process=process, **result)
