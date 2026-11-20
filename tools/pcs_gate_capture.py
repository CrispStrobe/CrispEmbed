#!/usr/bin/env python3
"""Explicit future PCS diagnostics. No build, inference, CI dispatch or scorer changes.

prepare creates a test-only source overlay; fetch records an immutable existing-Hub
input; provenance and analyze read retained files. Never run fetch/model gates as
part of source-only validation. Original pcs_parity.py remains the acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HOOKS = (
    ('    const int N = (int)token_ids.size();', '\n    pcs_gate_trace::chunk(N);'),
    ('    std::vector<int> token_ids = ctx->tokenizer.tokenize(input);',
     '\n    pcs_gate_trace::request(input, token_ids, ctx->tokenizer.id_to_token);'),
    ('            seg_argmax[t] = (logit1 > logit0) ? 1 : 0;',
     '\n            pcs_gate_trace::segmentation(t, logit0, logit1, p_boundary, result.sbd_preds[t], seg_argmax[t]);'),
    ('                result.cap_preds[t][c] = logit > 0;',
     '\n                pcs_gate_trace::casing(t, c, logit, result.cap_preds[t][c], inp[d] != 0);'),
)
PREFIX = '#include "pcs_gate_trace.h"\n'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def overlay(source):
    for anchor, insertion in HOOKS:
        if source.count(anchor) != 1:
            raise ValueError('PCS observer anchor missing or ambiguous: ' + anchor.strip())
        source = source.replace(anchor, anchor + insertion)
    return PREFIX + source


def prepare(source, output):
    text = overlay(Path(source).read_text())
    # CMake regenerates this task-owned file when its inputs change.
    Path(output).write_text(text)


def fetch_model(repo, filename, revision, expected_sha256, output):
    if not re.fullmatch('[0-9a-f]{40}', revision) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Full immutable model revision and predeclared SHA-256 required')
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + '.origin.json').exists():
        raise FileExistsError(output)
    from huggingface_hub import hf_hub_download  # only this explicit future command uses the existing provider
    cached = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision))
    if sha(cached) != expected_sha256:
        raise ValueError('Downloaded model SHA-256 differs; preserve the failed input, do not repin')
    with cached.open('rb') as source, output.open('xb') as target:
        shutil.copyfileobj(source, target)
    write_new(output.with_suffix(output.suffix + '.origin.json'), {
        'repository': repo, 'filename': filename, 'revision': revision,
        'sha256': sha(output), 'bytes': output.stat().st_size,
    })


def provenance(model, origin, binary, build, output):
    model, binary, build = Path(model), Path(binary), Path(build)
    identity = json.loads(Path(origin).read_text())
    if not re.fullmatch('[0-9a-f]{40}', identity['revision']) or sha(model) != identity['sha256']:
        raise ValueError('Model/revision does not match immutable origin receipt')
    cache = (build / 'CMakeCache.txt').read_text()
    commands = json.loads((build / 'compile_commands.json').read_text())
    if not any(Path(c['file']).name == 'pcs_gate_observed.cpp' for c in commands):
        raise ValueError('No diagnostic PCS compile command')
    observed = build / 'pcs_gate_observed.cpp'
    if observed.read_text() != overlay((ROOT / 'src/pcs.cpp').read_text()):
        raise ValueError('Observed PCS source differs from the declared adapter')
    compiler = re.search(r'^CMAKE_CXX_COMPILER:FILEPATH=(.+)$', cache, re.M)
    if not compiler:
        raise ValueError('C++ compiler identity missing')
    compiler = Path(compiler.group(1))
    cpu = Path('/proc/cpuinfo').read_text()
    # Only build/diagnostic controls: never collect authentication or general environment values.
    environment = {k: os.environ.get(k) for k in (
        'CRISPEMBED_BUILD_MODE', 'PCS_GATE_CAPTURE', 'OMP_NUM_THREADS', 'GGML_N_THREADS')}
    write_new(output, {
        'schema': 'crispembed.pcs.provenance.v1', 'model': identity,
        'binary': {'name': binary.name, 'sha256': sha(binary), 'bytes': binary.stat().st_size},
        'compiler': {'sha256': sha(compiler), 'version': subprocess.check_output(
            [str(compiler), '--version'], text=True)},
        'machine': platform.machine(), 'cpuinfo': cpu, 'environment': environment,
        'cmake_cache': cache, 'cmake_cache_sha256': sha(build / 'CMakeCache.txt'),
        'compile_commands': commands, 'compile_commands_sha256': sha(build / 'compile_commands.json'),
        'observed_source_sha256': sha(observed),
        'build_libraries': [{'name': str(p.relative_to(build)), 'sha256': sha(p)}
                            for p in sorted(build.rglob('*.so*')) if p.is_file() and not p.is_symlink()],
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'sources_sha256': {n: sha(ROOT / n) for n in (
            'src/pcs.cpp', 'tests/pcs_parity.py', 'tests/punct_ab.cpp', 'tests/pcs_gate_trace.h',
            'tools/pcs_gate_capture.py', 'tests/regression/pcs/blueprint_ref.txt')},
        'limits': 'Build-library inventory is not proof of dynamic loader selection. Capture neutrality and historical model identity remain separate gates.',
    })


def f32(value):
    if not re.fullmatch('[0-9a-f]{8}', value):
        raise ValueError('Invalid F32 bit pattern')
    number = struct.unpack('!f', bytes.fromhex(value))[0]
    return number


def analyze(trace, reference):
    sys.path.insert(0, str(ROOT / 'tests'))
    from pcs_parity import parse_ref
    refs, _ = parse_ref(reference)
    requests, tokens, segments, caps, chunks = {}, {}, {}, {}, {}
    build, isa = None, []
    for line in Path(trace).read_text().splitlines():
        row = line.split('\t')
        kind = row[0]
        if kind == 'build' and len(row) == 3 and build is None:
            build = {'cxx': int(row[1]), 'compiler': bytes.fromhex(row[2]).decode()}
        elif kind == 'isa' and len(row) == 2:
            isa.append(row[1])
        elif kind == 'request' and len(row) == 4:
            key = int(row[1])
            if key in requests:
                raise ValueError('Duplicate request')
            requests[key] = (int(row[2]), bytes.fromhex(row[3]).decode())
        elif kind == 'token' and len(row) == 5:
            key = (int(row[1]), int(row[2]))
            if key in tokens:
                raise ValueError('Duplicate token')
            tokens[key] = (int(row[3]), bytes.fromhex(row[4]).decode())
        elif kind == 'chunk' and len(row) == 4:
            chunks.setdefault(int(row[1]), []).append((int(row[2]), int(row[3])))
        elif kind == 'seg' and len(row) == 8:
            key = (int(row[1]), int(row[2]))
            if key in segments:
                raise ValueError('Duplicate segmentation row')
            segments[key] = row[3:]
        elif kind == 'cap' and len(row) == 7:
            key = (int(row[1]), int(row[2]), int(row[3]))
            if key in caps:
                raise ValueError('Duplicate casing row')
            caps[key] = row[4:]
        else:
            raise ValueError('Unknown, malformed or duplicate build record')
    if build is None or set(requests) != set(range(len(refs))):
        raise ValueError('Missing build or request records')
    expected_tokens = {(r, t) for r, ref in enumerate(refs) for t in range(len(ref['ids']))}
    if set(tokens) != expected_tokens or set(segments) != expected_tokens:
        raise ValueError('Token/segmentation inventory incomplete or misaligned')
    if set(caps) != {(r, t, c) for r, t in expected_tokens for c in range(16)}:
        raise ValueError('Casing inventory incomplete or misaligned')
    rows, mismatches, mappings, seg_nonfinite = [], [], [], []
    flat_offset = 0
    for r, ref in enumerate(refs):
        if len(ref['tokens']) != len(ref['ids']) or len(ref['cap']) != len(ref['ids']) or any(
                len(bits) != 16 or set(bits) - {'0', '1'} for bits in ref['cap']):
            raise ValueError('Malformed reference token/casing inventory')
        if requests[r] != (len(ref['ids']), ref['text']):
            raise ValueError('Native request differs from reference input')
        start = 0
        starts = set()
        for offset, count in chunks.get(r, []):
            if offset != start or count <= 0:
                raise ValueError('Chunk inventory is not contiguous')
            starts.add(offset)
            start += count
        if start != len(ref['ids']):
            raise ValueError('Chunk inventory incomplete')
        for t, (tid, piece) in enumerate(zip(ref['ids'], ref['tokens'])):
            if tokens[r, t] != (tid, piece):
                mappings.append({'request': r, 'token': t, 'reference_id': tid, 'reference_piece': piece,
                                 'native_id': tokens[r, t][0], 'native_piece': tokens[r, t][1]})
            seg = segments[r, t]
            a, b, probability = [f32(v) for v in seg[:3]]
            threshold, hard = int(seg[3]), int(seg[4])
            # Match the actual C++ 0.05f, including its rounded F32 tie value.
            if threshold not in (0, 1) or hard not in (0, 1) or hard != int(b > a) or threshold != int(probability > f32('3d4ccccd')):
                raise ValueError('Captured segmentation decisions disagree with actual logits/probability')
            if not all(math.isfinite(x) for x in (a, b, probability)):
                seg_nonfinite.append({'request': r, 'token': t, 'f32_bits': seg[:3]})
            for c in range(16):
                raw, bit, conditioning = caps[r, t, c]
                value, bit, conditioning = f32(raw), int(bit), int(conditioning)
                expected_initial = 1 if t in starts else int(segments[r, t - 1][4])
                if bit not in (0, 1) or conditioning != expected_initial or bit != int(value > 0):
                    raise ValueError('Casing sign or conditioning does not match observed native decision')
                significant = c < len(piece) and not (c == 0 and piece.startswith('▁'))
                row = {'request': r, 'token': t, 'flat_token': flat_offset + t, 'slot': c,
                       'native_id': tokens[r, t][0], 'native_piece': tokens[r, t][1],
                       'reference_id': tid, 'reference_piece': piece, 'significant': significant,
                       'logit_f32_bits': raw, 'finite': math.isfinite(value),
                       'margin_to_zero': value if math.isfinite(value) else None,
                       'absolute_magnitude': abs(value) if math.isfinite(value) else None,
                       'actual_bit': bit, 'reference_bit': int(ref['cap'][t][c]),
                       'sentence_initial': conditioning, 'sbd_logit_bits': seg[:2]}
                rows.append(row)
                if significant and bit != row['reference_bit']:
                    mismatches.append(row)
        flat_offset += len(ref['ids'])
    return {'schema': 'crispembed.pcs.case_capture.v1', 'build': build, 'compiled_isa_macros': isa,
            'requests': len(requests), 'tokens': len(tokens), 'casing_cells': len(rows),
            'token_mapping_mismatches': mappings, 'significant_casing_mismatches': mismatches,
            'first_known_casing_mismatch': mismatches[0] if mismatches else None,
            'all_values_finite': not seg_nonfinite and all(x['finite'] for x in rows),
            'nonfinite_segmentation_rows': seg_nonfinite, 'rows': rows,
            'trace_sha256': sha(trace), 'reference_sha256': sha(reference),
            'limits': 'Native capture only; does not supply original logits, prove observer neutrality or repair/waive the original decoded-text gate.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    a = commands.add_parser('prepare')
    a.add_argument('--source', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = commands.add_parser('fetch-model')
    for name in ('repo', 'filename', 'revision', 'expected-sha256', 'output'):
        a.add_argument('--' + name, required=True)
    a = commands.add_parser('provenance')
    for name in ('model', 'origin', 'binary', 'build', 'output'):
        a.add_argument('--' + name, type=Path, required=True)
    a = commands.add_parser('analyze')
    for name in ('trace', 'reference', 'output'):
        a.add_argument('--' + name, type=Path, required=True)
    args = vars(p.parse_args())
    command = args.pop('command')
    if command == 'analyze':
        output = args.pop('output')
        report = analyze(**args)
        write_new(output, report)
        return int(bool(report['token_mapping_mismatches'] or report['significant_casing_mismatches'] or not report['all_values_finite']))
    {'prepare': prepare, 'fetch-model': fetch_model, 'provenance': provenance}[command](**args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
