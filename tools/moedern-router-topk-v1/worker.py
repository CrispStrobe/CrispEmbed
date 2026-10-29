#!/usr/bin/env python3
"""Hosted-only authored CPU router audit; imports no model/front-end packages."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PROFILE, identity, packet, read, require, rows, sha, verify, write
sys.path.insert(0, str(ROOT / 'tools'))
from ci_process import run


def metrics(np, a, b):
    require(a.shape == b.shape and a.dtype == b.dtype == np.float32 and a.ndim == 2, 'Exact F32 comparator axes required')
    x, y = a.astype(np.float64), b.astype(np.float64)
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite operator outputs')
    d = y - x
    aa, bb = (x * x).sum(1), (y * y).sum(1)
    cosine = []
    for at in range(len(x)):
        cosine.append(float((x[at] * y[at]).sum() / (aa[at] ** .5 * bb[at] ** .5))
                      if aa[at] > 1e-18 and bb[at] > 1e-18 else (1. if aa[at] <= 1e-18 and bb[at] <= 1e-18 else 0.))
    result = dict(elements=a.size, shape=list(a.shape), max_absolute=float(abs(d).max()),
                  mean_absolute=float(abs(d).mean()), mean_error=float(d.mean()), rms=float((d * d).mean() ** .5),
                  min_row_cosine=min(cosine), original_l2=float((x * x).sum() ** .5), native_l2=float((y * y).sum() ** .5))
    for label, values in [('original', x), ('native', y)]:
        result.update({label + '_min': float(values.min()), label + '_max': float(values.max()),
                       label + '_abs_min': float(abs(values).min()), label + '_abs_max': float(abs(values).max())})
    result['passed'] = result['max_absolute'] <= .002 and result['min_row_cosine'] >= .99999
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed-source', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    a = p.parse_args()
    review, raw = a.out / 'review', a.out / 'raw'
    review.mkdir(parents=True, exist_ok=True)
    raw.mkdir(exist_ok=True)
    report = dict(schema='crispembed-moedern-router-topk-operator-v1', source=a.reviewed_source, completed=False,
                  model_access=False, production_qualified=False, old_wheel_identity_established=False,
                  actual_failed_documents_replayed=False, phase='source', comparisons=[], reference_controls={}, commands={})
    def save():
        write(review / 'operator.json', report)
    def command(name, argv, expected=(0,)):
        result = run(argv, review / (name + '.log'), cwd=ROOT, timeout=180)
        report['commands'][name] = dict(result, argv=list(map(str, argv)), log=name + '.log', **identity(review / (name + '.log')))
        save()
        require(not result['timed_out'] and result['exit'] in expected, 'Operator child rejected: ' + name)
        return result['exit']
    save()
    try:
        cfg = verify(a.reviewed_source)
        authored = rows(cfg)
        require(sys.version_info[:3] == (3, 12, 11), 'New cp312 audit profile required')
        versions = read(Path(__file__).with_name('wheels.json'))
        installed = {name: importlib.metadata.version(name) for name in versions if name != 'uv'}
        require(all(installed[n] == pin['version'] for n, pin in versions.items() if n != 'uv'), 'Installed locked versions differ')
        # Author input bits and all packet bytes are frozen BEFORE importing Torch/NumPy.
        report['input_freeze'] = dict(protocol_sha256=sha(PROFILE), rows=authored, original_first=True)
        for mode, values in authored.items():
            packet(raw / (mode + '.input'), [struct.unpack('<8f', struct.pack('<8I', *r['bits'])) for r in values])
        write(review / 'input-freeze.json', report['input_freeze'])
        report['phase'] = 'original-operators'
        save()
        import numpy as np
        import torch
        require(torch.__version__ == '2.8.0+cpu' and torch.version.cuda is None, 'Exact CPU Torch required')
        require(re.fullmatch('[a-f0-9]{40}', torch.version.git_version or ''), 'Record actual Torch build Git ID')
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        report['runtime'] = dict(python=sys.version, packages=installed, torch_git=torch.version.git_version,
                                 torch_config=torch.__config__.show(), platform=platform.platform(),
                                 cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
                                 cpu_capability=torch.backends.cpu.get_cpu_capability(), interop_threads=1,
                                 deterministic_algorithms=True, float32_matmul_precision='highest')
        mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                         if '/' in line and any(x in line for x in ['libstdc++', 'libtorch', 'libc10'])})
        report['runtime']['mapped_libraries'] = {str(Path(x).resolve()): identity(x) for x in mapped if Path(x).is_file()}
        references = {}
        continuity = []
        for mode, values in authored.items():
            base = np.asarray([r['bits'] for r in values], dtype='<u4').view('<f4')
            for threads in cfg['threads']:
                torch.set_num_threads(threads)
                for layout in cfg['layouts']:
                    if layout == 'contiguous':
                        x = torch.from_numpy(base.copy())
                    elif layout == 'column_stride2':
                        x = torch.empty((len(base), 16), dtype=torch.float32)[:, ::2]
                        x.copy_(torch.from_numpy(base))
                    elif layout == 'row_stride2':
                        x = torch.empty((len(base) * 2, 8), dtype=torch.float32)[::2, :]
                        x.copy_(torch.from_numpy(base))
                    else:
                        x = torch.from_numpy(base.T.copy()).T
                    require(x.numpy().tobytes() == base.tobytes(), 'Logical stride input bits changed')
                    for repeat in range(cfg['repeats']):
                        probs = torch.softmax(x, dim=-1) if mode == 'logits' else x
                        selected, indices = probs.topk(2, dim=-1, largest=True, sorted=True)
                        weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-9)
                        tag = f'{mode}_t{threads}_{layout}_r{repeat}'
                        arrays = {'probabilities': probs.detach().numpy().copy(), 'values': selected.numpy().copy(),
                                  'weights': weights.numpy().copy(), 'indices': indices.numpy().astype('<i4')}
                        if mode not in references:
                            references[mode] = arrays
                        exact = {name: value.tobytes() == references[mode][name].tobytes() for name, value in arrays.items()}
                        for name, value in arrays.items():
                            (raw / (tag + '.' + name)).write_bytes(value.tobytes())
                        continuity.append(dict(arm=tag, input_stride=list(x.stride()), probability_stride=list(probs.stride()),
                                               exact_baseline=exact, passed=all(exact.values())))
        report['original_controls'] = continuity
        report['original_observations_complete_before_cpp'] = True
        save()
        # No native execution occurs until every original control above is retained.
        sys.path.insert(0, str(ROOT / 'tools/moedern-native-v2'))
        import diff_adapter as adapter
        def spec(path, array):
            path.write_bytes(array.astype('<f4', copy=False).tobytes())
            return dict(path=str(path), shape=list(array.shape), sha256=sha(path))
        def compare(name, original, native):
            pairs, independent = [], {}
            for stage in original:
                key = name + '_' + stage
                independent[key] = metrics(np, original[stage], native[stage])
                pairs.append(dict(name=key, original=spec(raw / (key + '.original.f32'), original[stage]),
                                  native=spec(raw / (key + '.native.f32'), native[stage])))
            left, right = raw / (name + '.original.gguf'), raw / (name + '.native.gguf')
            adapters = adapter.write_pair(left, right, pairs)
            output = review / (name + '.ref.json')
            code = command(name + '_ref', [a.build / 'router-topk-ref', left, right, output], (0, 1))
            cpp = adapter.collect(code, output, independent)
            return dict(actual_ref=cpp, independent=independent, adapters=adapters,
                        passed=all(v['passed'] for v in independent.values()))
        small = np.arange(16, dtype=np.float32).reshape(2, 8) / 16
        report['reference_controls']['positive'] = compare('control_positive', {'test': small}, {'test': small})['passed']
        bad = small.copy()
        bad[0, 0] += .01
        report['reference_controls']['numeric_rejection'] = not compare('control_negative', {'test': small}, {'test': bad})['passed']
        adapter.write_gguf(raw / 'wrong-shape.gguf', [dict(name='control_positive_test', **spec(raw / 'shape.f32', small.reshape(4, 4)))])
        code = command('control_shape', [a.build / 'router-topk-ref', raw / 'control_positive.original.gguf',
                                        raw / 'wrong-shape.gguf', review / 'wrong-shape.ref.json'], (2,))
        report['reference_controls']['shape_rejection'] = code == 2
        # Packet rejection is before graph compute; neither case contains model data.
        for name, data in [('truncated', b'MRTOPK1\0'), ('nonfinite', b'MRTOPK1\0' + struct.pack('<I8f', 1, float('nan'), 0, 0, 0, 0, 0, 0, 0))]:
            path = raw / (name + '.input')
            path.write_bytes(data)
            report['reference_controls'][name + '_rejection'] = command(name, [a.build / 'router-topk-operator', path, raw / name, 'probabilities', '1'], (2,)) == 2
        report['phase'] = 'same-input-cpp-operators'
        for mode, reference in references.items():
            arms = [(mode, raw / (mode + '.input'), mode)]
            if mode == 'logits':
                path = raw / 'logits_torch_probabilities.input'
                packet(path, reference['probabilities'].tolist())
                arms.append(('logits_torch_probabilities', path, 'probabilities'))
            for arm, source, operator_mode in arms:
                for threads in cfg['threads']:
                    for repeat in range(cfg['repeats']):
                        name = f'{arm}_t{threads}_r{repeat}'
                        prefix = raw / name
                        command(name, [a.build / 'router-topk-operator', source, prefix, operator_mode, str(threads)])
                        n = len(reference['probabilities'])
                        native = {'probabilities': np.fromfile(str(prefix) + '.probabilities.f32', dtype='<f4').reshape(n, 8)}
                        record = dict(arm=name, input=identity(source), row_ids=[r['name'] for r in authored[mode]],
                                      same_probabilities=operator_mode == 'probabilities', candidates={})
                        expected_routes = np.zeros((n, 8), dtype=np.float32)
                        np.put_along_axis(expected_routes, reference['indices'].astype(np.int64), reference['weights'], axis=1)
                        for candidate in ['source', 'ggml']:
                            indices = np.fromfile(str(prefix) + '.' + candidate + '_indices.i32', dtype='<i4').reshape(n, 2)
                            require(((indices >= 0) & (indices < 8)).all() and (indices[:, 0] != indices[:, 1]).all(), 'Invalid expert indices')
                            values = np.fromfile(str(prefix) + '.' + candidate + '_values.f32', dtype='<f4').reshape(n, 2)
                            weights = np.fromfile(str(prefix) + '.' + candidate + '_weights.f32', dtype='<f4').reshape(n, 2)
                            routes = np.zeros((n, 8), dtype=np.float32)
                            np.put_along_axis(routes, indices.astype(np.int64), weights, axis=1)
                            measured = compare(name + '_' + candidate,
                                               dict(probabilities=reference['probabilities'], values=reference['values'], weights=reference['weights'], routes=expected_routes),
                                               dict(probabilities=native['probabilities'], values=values, weights=weights, routes=routes))
                            per_row = []
                            for at, row in enumerate(authored[mode]):
                                probs = reference['probabilities'][at]
                                order = np.sort(probs)[::-1]
                                per_row.append(dict(id=row['name'], input_bits=row['bits'],
                                                    original_probability_bits=probs.view('<u4').tolist(),
                                                    native_probability_bits=native['probabilities'][at].view('<u4').tolist(),
                                                    cutoff_margin=float(np.float64(order[1]) - np.float64(order[2])),
                                                    cutoff_tie=bool(order[1] == order[2]),
                                                    original_indices=reference['indices'][at].tolist(), native_indices=indices[at].tolist(),
                                                    indices_exact=bool(np.array_equal(indices[at], reference['indices'][at])),
                                                    value_bits_exact=values[at].tobytes() == reference['values'][at].tobytes(),
                                                    weights_bits_exact=weights[at].tobytes() == reference['weights'][at].tobytes()))
                            exact = all(r['indices_exact'] and r['value_bits_exact'] and r['weights_bits_exact'] for r in per_row)
                            record['candidates'][candidate] = dict(rows=per_row, exact_indices_values_weights=exact,
                                                                   comparison=measured, passed=exact and measured['passed'])
                        report['comparisons'].append(record)
                        save()
        report['all_original_layout_thread_repeat_exact'] = all(r['passed'] for r in continuity)
        report['all_fixed_comparisons_passed'] = (report['all_original_layout_thread_repeat_exact']
                                                and all(report['reference_controls'].values())
                                                and all(c['passed'] for r in report['comparisons'] for c in r['candidates'].values()))
        report['source_topk_same_probabilities_exact'] = all(r['candidates']['source']['passed'] for r in report['comparisons'] if r['same_probabilities'])
        report.update(completed=True, phase='completed', status='passed' if report['all_fixed_comparisons_passed'] else 'operator_difference_retained')
        save()
        return 0 if report['all_fixed_comparisons_passed'] else 1
    except Exception as e:
        report.update(error=type(e).__name__ + ': ' + str(e), phase='structural-rejection')
        save()
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
