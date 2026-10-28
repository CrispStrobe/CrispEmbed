#!/usr/bin/env python3
"""Hosted-only retained-input F32 GGML boundary attribution; no model forward."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/wasm-f32-operators'), str(ROOT / 'tools'), str(Path(__file__).parent)]
from contract import *
from adapter import write_pair, write_gguf, collect
from wasm_layer3_attribution_v1 import layer_rows
from successor_wasm_trace_contract import (gguf, FIXTURE_ID, FIXTURE_BYTES, FIXTURE_SHA,
    INITIAL_RUN, INITIAL_SOURCE, NATIVE_REPORT_SHA)
from successor_assets import api, artifact
from josev2b_composed_panel import metrics

SPEC = ROOT / 'docs/wasm-f32-operators-v1.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def hosted():
    require(os.environ.get('GITHUB_ACTIONS') == 'true' and
            git(ROOT, 'rev-parse', 'HEAD') == os.environ.get('GITHUB_SHA'), 'Hosted exact source required')
    spec = json.loads(SPEC.read_text())
    require(spec['base'] == BASE and spec['ggml'] == GGML and spec['trace_report_sha256'] == TRACE_REPORT and
            spec['gates'] == {'max_absolute': .002, 'minimum_row_cosine': .99999} and
            spec['operators'] == {k: list(v) for k, v in OPERATORS.items()}, 'Declared F32 protocol differs')
    require(not git(ROOT, 'status', '--porcelain', '--', str(SPEC.relative_to(ROOT)), *spec['source_sha256']),
            'Dirty diagnostic source/protocol')
    for name, expected in spec['source_sha256'].items():
        require(digest(ROOT / name) == expected, 'Diagnostic dependency SHA differs: ' + name)
    return spec


def command(args, out, name, timeout=1800):
    log = out / (name + '.log')
    with log.open('wb') as stream:
        process = subprocess.run(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    save(out / 'review' / (name + '-command.json'), dict(argv=list(map(str, args)), exit=process.returncode,
        log=log.name, sha256=digest(log)))
    require(process.returncode == 0, 'Hosted command failed: ' + name)


def service(run, source, conclusion):
    row = api('actions/runs/' + str(run))
    require((row['head_sha'], row['path'], row['status'], row['conclusion'], row['event'], row['run_attempt']) ==
            (source, '.github/workflows/ovisocr2-metadata.yml', 'completed', conclusion, 'workflow_dispatch', 1),
            'Frozen service identity differs')
    return {k: row[k] for k in ['id', 'head_sha', 'path', 'status', 'conclusion', 'event', 'run_attempt']}


def download(run, name, path, expected):
    row = artifact(run, name, path, 32 * 1024**2)
    require((row['id'], row['size_in_bytes'], row['digest']) ==
            (expected['id'], expected['bytes'], 'sha256:' + expected['sha256']), 'Frozen artifact identity differs')
    require(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], 'Frozen ZIP SHA differs')
    return row


def prepare(out):
    hosted()
    out.mkdir(parents=True, exist_ok=False)
    proof = dict(source=os.environ['GITHUB_SHA'], protocol_sha256=digest(SPEC),
                 qualification=False, services={}, artifacts={})
    current = api('actions/runs/' + os.environ['GITHUB_RUN_ID'])
    require(current['head_sha'] == os.environ['GITHUB_SHA'] and current['run_attempt'] == int(os.environ['GITHUB_RUN_ATTEMPT']),
            'Current hosted source/attempt differs')
    proof['service'] = {k: current[k] for k in ['id', 'head_sha', 'path', 'event', 'run_attempt']}
    proof['services']['trace'] = service(TRACE_RUN, TRACE_SOURCE, 'success')
    prefix = 'wasm-layer3-attribution-v1-'
    proof['artifacts']['review'] = download(TRACE_RUN, prefix + 'review-' + str(TRACE_RUN), out / 'old-review.zip', REVIEW)
    review = safe_zip(out / 'old-review.zip', 16 * 1024**2)
    # upload LCA is the producer's layer3 root; report is under review/.
    name = 'review/layer3.json'
    require(name in review and sha(review[name]) == TRACE_REPORT, 'Exact old trace report differs')
    old = json.loads(review[name])
    require(old['source'] == TRACE_SOURCE and old['base'] == BASE and old['ggml'] == GGML and
            old['model_sha256'] == MODEL and old['state'] == 'attribution-complete' and old['qualification'] is False,
            'Old trace scope/source differs')
    save(out / 'review/historical-subnodes.json', dict(trace_report_sha256=TRACE_REPORT, subnodes=old['subnodes']))
    proof['artifacts']['raw'] = download(TRACE_RUN, prefix + 'raw-' + str(TRACE_RUN), out / 'old-raw.zip', RAW)
    files = safe_zip(out / 'old-raw.zip', 96 * 1024**2)
    bins = json.loads(review['review/build-artifacts.json'])
    proof['raw_authentication'] = authenticate_files(files, old['raw_members'], bins)
    # Preserve the actual old compiler records before selecting new build flags.
    (out / 'old-native-compile.json').write_bytes(review['review/compile-native.json'])
    (out / 'old-wasm-compile.json').write_bytes(review['review/compile-wasm.json'])
    proof['retained_native_ggml_exports'] = False
    # Original receipt's dynamic symbol listing has no ggml API: do not silently
    # link a new ABI to the retained .so. New builds must prove own byte anchors.
    symbols = review['review/native-symbols.txt'].decode()
    require(not any(line.split() and line.split()[-1] in ['ggml_norm', 'ggml_init', 'ggml_backend_cpu_init']
                    for line in symbols.splitlines()), 'Retained API capability changed; review linkage first')
    proof['services']['fixture'] = service(INITIAL_RUN, INITIAL_SOURCE, 'failure')
    fixture_pin = dict(id=FIXTURE_ID, bytes=FIXTURE_BYTES, sha256=FIXTURE_SHA)
    proof['artifacts']['fixture'] = download(INITIAL_RUN, 'successor-minilm-fixture-' + str(INITIAL_RUN), out / 'fixture.zip', fixture_pin)
    fixture = safe_zip(out / 'fixture.zip', 48 * 1024**2)
    models = [n for n in fixture if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')]
    require(len(fixture) == 2 and len(models) == 1 and sha(fixture[models[0]]) == MODEL and
            sha(fixture['consumer.json']) == NATIVE_REPORT_SHA, 'Frozen original fixture differs')
    model = fixture[models[0]]
    meta, tensors = gguf(model)
    require(meta.get('general.architecture') == 'bert', 'Actual model architecture differs')
    proof['epsilon'] = epsilon(meta)
    by_name = {row['name']: row for row in tensors}
    weights = {}
    for name in WEIGHTS:
        # Exact ordered get_any aliases from source580, not a suffix search.
        if name.startswith('attn_output_norm.'):
            suffix = name.split('.', 1)[1]
            aliases = [f'enc.3.ln1.{suffix}', f'blk.3.{name}', f'blk.3.attn_norm.{suffix}']
        elif name.startswith('layer_output_norm.'):
            suffix = name.split('.', 1)[1]
            aliases = [f'enc.3.ln2.{suffix}', f'blk.3.{name}', f'blk.3.ffn_norm.{suffix}']
        else:
            stem = {'attn_q.bias': 'attn.q.bias', 'attn_k.bias': 'attn.k.bias', 'attn_v.bias': 'attn.v.bias',
                    'attn_output.bias': 'attn.o.bias', 'ffn_up.bias': 'ffn.fc1.bias', 'ffn_down.bias': 'ffn.fc2.bias'}[name]
            aliases = ['enc.3.' + stem, 'blk.3.' + name]
        actual = next((n for n in aliases if n in by_name), None)
        require(actual, 'Missing source-mapped F32 model coefficient: ' + name)
        row = by_name[actual]
        require(row['type'] == 0 and row['shape'] == [1536 if name == 'ffn_up.bias' else 384], 'Exact F32 coefficient shape/type differs')
        raw = model[row['data_offset']:row['data_offset'] + row['shape'][0] * 4]
        require(len(raw) == row['shape'][0] * 4 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)), 'Nonfinite coefficient')
        weights[name] = dict(bytes=raw, actual_name=actual, sha256=sha(raw))
    packets = {}
    for axis, member in STAGES.items():
        stages = layer_rows(files[member])
        for name, (op, a, b, c, target) in OPERATORS.items():
            key = axis + '/' + name
            folder = out / 'browser' / 'packets' / axis / name
            folder.mkdir(parents=True)
            packet = dict(op=op, width=1536 if target in ['fc1', 'gelu'] else 384, rows=12,
                          epsilon=proof['epsilon']['value'], inputs={}, expected_sha256=stages['layer3_' + target]['sha256'])
            for label, source in [('a', a), ('b', b), ('c', c)]:
                if source is None:
                    continue
                if source.startswith('w:'):
                    value = weights[source[2:]]
                    raw = value['bytes']
                    packet['inputs'][label] = dict(file=str((folder / (label + '.f32')).relative_to(out / 'browser')),
                        bytes=len(raw), sha256=sha(raw), source=value['actual_name'])
                else:
                    row = stages['layer3_' + source]
                    begin = row['data_offset']
                    raw = files[member][begin:begin + math.prod(row['shape']) * 4]
                    require(sha(raw) == row['sha256'], 'Literal retained stage bytes differ')
                    packet['inputs'][label] = dict(file=str((folder / (label + '.f32')).relative_to(out / 'browser')),
                        bytes=len(raw), sha256=sha(raw), source='layer3_' + source)
                (folder / (label + '.f32')).write_bytes(raw)
            packet['b_count'] = packet['inputs'].get('b', {}).get('bytes', 0) // 4
            packets[key] = packet
    save(out / 'browser/packets.json', packets)
    proof['coefficient_sources'] = {k: {x: v for x, v in row.items() if x != 'bytes'} for k, row in weights.items()}
    proof['packets_sha256'] = digest(out / 'browser/packets.json')
    save(out / 'review/prepared.json', proof)
    save(out / 'review/packets.json', packets)


def prepared(out):
    proof = json.loads((out / 'review/prepared.json').read_text())
    require(proof['source'] == os.environ['GITHUB_SHA'] and proof['protocol_sha256'] == digest(SPEC) and
            proof['service']['id'] == int(os.environ['GITHUB_RUN_ID']) and
            proof['service']['run_attempt'] == int(os.environ['GITHUB_RUN_ATTEMPT']) and
            proof['qualification'] is False and proof['packets_sha256'] == digest(out / 'browser/packets.json') and
            digest(out / 'old-review.zip') == REVIEW['sha256'] and digest(out / 'old-raw.zip') == RAW['sha256'] and
            digest(out / 'fixture.zip') == FIXTURE_SHA, 'Authenticated preparation chain changed')
    packets = json.loads((out / 'browser/packets.json').read_text())
    require(set(packets) == {axis + '/' + name for axis in AXES for name in OPERATORS}, 'Exact48 packet coverage differs')
    return proof


def arithmetic_flags(row):
    argv = row.get('arguments') or shlex.split(row['command'])
    return sorted(arg for arg in argv if arg == '-pthread' or arg.startswith(('-D', '-m', '-f', '-O')))


def compile_profile(out):
    observed = {}
    for label in ['native', 'wasm']:
        old = json.loads((out / ('old-' + label + '-compile.json')).read_text())
        new = json.loads((out / (label + '-build/compile_commands.json')).read_text())
        rows = []
        for basename in ['ggml-cpu.c', 'ggml-cpu.cpp', 'vec.cpp', 'ops.cpp']:
            a = [r for r in old if Path(r['file']).name == basename]
            b = [r for r in new if Path(r['file']).name == basename]
            require(len(a) == len(b) == 1 and arithmetic_flags(a[0]) == arithmetic_flags(b[0]), 'Exact old/new compiler profile differs: ' + label + '/' + basename)
            rows.append(dict(file=basename, old=a[0], current=b[0], arithmetic_flags=arithmetic_flags(b[0])))
        observed[label] = rows
    return observed


def build(out):
    hosted()
    prepared(out)
    source = out / 'source'
    command(['git', 'worktree', 'add', '--detach', source, BASE], out, 'source-worktree')
    command(['git', '-C', source, 'submodule', 'update', '--init', '--depth', '1', 'ggml'], out, 'source-submodule')
    require(git(source, 'rev-parse', 'HEAD') == BASE and git(source / 'ggml', 'rev-parse', 'HEAD') == GGML and
            not git(source, 'status', '--porcelain') and not git(source / 'ggml', 'status', '--porcelain'), 'Frozen source closure differs')
    for label in ['native', 'wasm']:
        prefix = ['emcmake'] if label == 'wasm' else []
        command(prefix + ['cmake', '-S', ROOT / 'tests/wasm-f32-operators', '-B', out / (label + '-build'),
                         '-DCMAKE_BUILD_TYPE=Release', '-DFROZEN_GGML=' + str(source / 'ggml')], out, label + '-configure')
        command(['cmake', '--build', out / (label + '-build'), '--parallel', '2'], out, label + '-build', 1800)
    save(out / 'review/compile-profile.json', compile_profile(out))
    for label in ['native', 'wasm']:
        shutil.copy2(out / (label + '-build/compile_commands.json'), out / 'review' / ('compile-' + label + '.json'))
    destination = out / 'browser/module'
    destination.mkdir()
    for path in (out / 'wasm-build').glob('wasm-f32-operators.*'):
        if path.suffix in ['.js', '.wasm']:
            shutil.copy2(path, destination)
    require((destination / 'wasm-f32-operators.js').is_file() and (destination / 'wasm-f32-operators.wasm').is_file(), 'Actual browser module pair missing')
    save(out / 'review/build-artifacts.json', {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=digest(p))
        for folder in [out / 'native-build', destination] for p in folder.rglob('*')
        if p.is_file() and ('.so' in p.name or p.suffix in ['.js', '.wasm'])})
    ref_controls(out)


def verified_build(out):
    hosted()
    prepared(out)
    declared = json.loads((out / 'review/build-artifacts.json').read_text())
    for name, row in declared.items():
        path = out / name
        require(path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'Actual operator binary/module SHA changed')
    return declared


def verify(out):
    verified_build(out)


def ref_controls(out):
    folder = out / 'ref-controls'
    folder.mkdir()
    source = folder / 'values.f32'
    source.write_bytes(struct.pack('<4f', 1, 2, 3, 4))
    original = folder / 'original.gguf'
    write_gguf(original, [dict(name='stage', path=source, shape=[2, 2], sha256=digest(source))])
    results = {}
    for name, values, shape, code in [('positive', [1, 2, 3, 4], [2, 2], 0),
                                     ('numeric', [1, 2, 3.01, 4], [2, 2], 1),
                                     ('wrong-shape', [1, 2, 3, 4], [1, 4], 2)]:
        raw = folder / (name + '.f32')
        raw.write_bytes(struct.pack('<4f', *values))
        packet = folder / (name + '.gguf')
        write_gguf(packet, [dict(name='stage', path=raw, shape=shape, sha256=digest(raw))])
        result = subprocess.run([out / 'native-build/wasm-f32-diff', original, packet, folder / (name + '.json')], capture_output=True)
        (folder / (name + '.log')).write_bytes(result.stdout + result.stderr)
        require(result.returncode == code, 'Actual Ref control failed: ' + name)
        results[name] = dict(exit=result.returncode, report_sha256=digest(folder / (name + '.json')))
    save(out / 'review/ref-controls.json', results)


def native(out):
    binaries = verified_build(out)
    libraries = list((out / 'native-build').glob('libwasm-f32-operators.so'))
    require(len(libraries) == 1, 'One exact standalone native operator library required')
    lib = C.CDLL(str(libraries[0]))
    mappings = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if '/' in line and 'libggml' in line.split()[-1]}
    require(mappings, 'Actual GGML provider mappings missing')
    for name in mappings:
        path = Path(name).resolve()
        require(path.is_relative_to((out / 'native-build').resolve()), 'Actual GGML provider escaped frozen build')
        key = str(path.relative_to(out.resolve()))
        require(key in binaries and digest(path) == binaries[key]['sha256'], 'Actual loaded GGML provider hash differs')
    save(out / 'review/native-mappings.json', dict(providers=sorted(mappings), all_exact_frozen_build=True))
    function = lib.crispembed_f32_operator
    function.argtypes = [C.c_int] * 4 + [C.c_float] + [C.POINTER(C.c_float)] * 4 + [C.c_int] * 2
    function.restype = C.c_int
    control_input = (C.c_float * 4608)(*([1.0] * 4608))
    control_output = (C.c_float * 4608)()
    require(function(0, 384, 12, 0, 1e-12, control_input, None, None, control_output, 4608, 1) == 4608 and
            all(x == 0 for x in control_output), 'Real native F32 norm positive control')
    require(function(0, 385, 12, 0, 1e-12, control_input, None, None, control_output, 4608, 1) < 0,
            'Real native wrong-shape control')
    control_input[0] = float('nan')
    require(function(0, 384, 12, 0, 1e-12, control_input, None, None, control_output, 4608, 1) < 0,
            'Real native nonfinite control')
    control_input[0] = 1
    require(function(0, 384, 12, 0, 1e-12, control_input, None, None, control_output, 4608, 1) == 4608 and
            all(x == 0 for x in control_output), 'Real native failure recovery control')
    save(out / 'review/native-controls.json', dict(positive=True, wrong_shape_rejected=True,
                                                  nonfinite_rejected=True, recovery=True))
    packets = json.loads((out / 'browser/packets.json').read_text())
    report = {}
    for thread in [1, 2]:
        backend = 'native' + str(thread)
        report[backend] = {}
        for key, packet in packets.items():
            arrays = {}
            for label, row in packet['inputs'].items():
                raw = (out / 'browser' / row['file']).read_bytes()
                require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Packet bytes changed')
                arrays[label] = (C.c_float * (len(raw) // 4)).from_buffer_copy(raw)
            count = packet['width'] * 12
            output = (C.c_float * count)()
            n = function(packet['op'], packet['width'], 12, packet['b_count'], packet['epsilon'],
                         arrays['a'], arrays.get('b'), arrays.get('c'), output, count, thread)
            require(n == count, 'Actual native operator failure: ' + key + '/' + str(n))
            raw = C.string_at(output, count * 4)
            require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)), 'Nonfinite native output')
            repeated = (C.c_float * count)()
            require(function(packet['op'], packet['width'], 12, packet['b_count'], packet['epsilon'],
                             arrays['a'], arrays.get('b'), arrays.get('c'), repeated, count, thread) == count and
                    C.string_at(repeated, count * 4) == raw, 'Actual native repeat differs')
            require(all(sha(C.string_at(arrays[label], row['bytes'])) == row['sha256']
                        for label, row in packet['inputs'].items()), 'Actual native operator modified input')
            path = out / 'raw' / backend / (key + '.f32')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            report[backend][key] = dict(file=str(path.relative_to(out)), sha256=sha(raw), bytes=len(raw),
                shape=[12, packet['width']], input_hashes={k: r['sha256'] for k, r in packet['inputs'].items()})
            report[backend][key]['repeat_bit_exact'] = True
    save(out / 'review/native.json', report)


def compare(out):
    hosted()
    prepared(out)
    packets = json.loads((out / 'browser/packets.json').read_text())
    native = json.loads((out / 'review/native.json').read_text())
    browser = json.loads((out / 'browser/report.json').read_text())
    require(browser['completed'] is True and not browser['page_errors'] and browser['isolated'] is True and
            browser['pthreads'] is True, 'Actual isolated browser execution incomplete')
    outputs = dict(native, wasm=browser['outputs'])
    for backend, rows in outputs.items():
        for key, row in rows.items():
            path = out / row['file']
            require(not path.is_symlink() and path.is_file() and path.stat().st_size == row['bytes'] and
                    digest(path) == row['sha256'], 'Actual raw output hash differs before own-anchor barrier')
    # This mandatory64-anchor barrier precedes every cross-backend metric.
    anchors = own_anchors(packets, outputs)
    save(out / 'review/own-anchors.json', anchors)
    pairs, independent = [], {}
    import numpy as np
    for key, packet in packets.items():
        a, b = native['native1'][key], browser['outputs'][key]
        for label, row in [('native', a), ('wasm', b)]:
            path = out / row['file']
            require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'] and
                    row['input_hashes'] == {k: v['sha256'] for k, v in packet['inputs'].items()}, 'Actual replay input/output hash differs')
        name = key.replace('/', '.')
        shape = [12, packet['width']]
        pair = dict(name=name, original=dict(path=str(out / a['file']), shape=shape, sha256=a['sha256']),
                    native=dict(path=str(out / b['file']), shape=shape, sha256=b['sha256']))
        pairs.append(pair)
        independent[name] = metrics(np.memmap(pair['original']['path'], dtype='<f4', mode='r', shape=tuple(shape)),
                                    np.memmap(pair['native']['path'], dtype='<f4', mode='r', shape=tuple(shape)))
    adapters = write_pair(out / 'control.gguf', out / 'replay.gguf', pairs)
    diff = out / 'native-build/wasm-f32-diff'
    result = subprocess.run([diff, out / 'control.gguf', out / 'replay.gguf', out / 'review/ref.json'], capture_output=True)
    (out / 'ref.log').write_bytes(result.stdout + result.stderr)
    cpp = collect(result.returncode, out / 'review/ref.json', independent)
    save(out / 'review/operators.json', dict(schema='crispembed-wasm-f32-operators-v1',
        state='attribution-complete', completed=True, source=os.environ['GITHUB_SHA'],
        service=json.loads((out / 'review/prepared.json').read_text())['service'], base=BASE, ggml=GGML,
        original_run=TRACE_RUN, own_anchors=anchors, compared_same_input_boundaries=len(pairs),
        cpp=cpp, independent=independent, adapters=adapters, qualification=False,
        historical_natural_boundaries=json.loads((out / 'review/historical-subnodes.json').read_text()),
        all_boundaries_within_frozen_budgets=all(row['passed'] for row in independent.values()),
        gate_failures=[name for name, row in independent.items() if not row['passed']],
        production_candidate=False, norm_emulation_executed=False,
        limits='One retained12-token sentence,16 F32 boundaries,three exact input axes; no model forward or final quality qualification.'))
    save(out / 'review/raw-manifest.json', {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=digest(p))
        for folder in [out / 'raw', out / 'browser/packets', out / 'browser/module'] for p in sorted(folder.rglob('*')) if p.is_file()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'build', 'native', 'verify', 'compare'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        globals()[args.phase](args.output)
    except Exception as error:
        save(args.output / 'review' / ('rejection-' + args.phase + '.json'),
             dict(phase=args.phase, error=str(error), qualification=False))
        raise


if __name__ == '__main__':
    main()
