#!/usr/bin/env python3
"""Bounded real-model PP-OCRv6 empty-frame and alternating-shape A/B."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
DEVICE = os.environ.get('DEVICE', 'cpu')
MODELS = {
    'det': ('cstr/PP-OCRv6-medium-det-GGUF', '62f655124d0b0bc71c7379ae598efe7807de1b97',
            'PP-OCRv6_medium_det-f16.gguf', '39b80b016b7bc10ae3e17614ed5ebe6d58855859b88ab292127638935e673678'),
    'rec': ('cstr/PP-OCRv6_medium_rec-GGUF', 'c67c83efd46f8efc4a0183fc26afbc06b6235ae7',
            'PP-OCRv6_medium_rec-f16.gguf', '8aeac742d9a7325f3f487f5c05cb69389e4642f333857e4e95b95891a263f2e7'),
}


def call(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(repo, revision, filename, digest):
    target = SCRATCH / filename
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'CrispEmbed-ci/1'}), timeout=90) as src, target.open('wb') as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            if sha(target) != digest:
                raise ValueError('model hash mismatch: ' + filename)
            return target
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    raise AssertionError('unreachable')


def metrics(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or differently shaped arrays')
    a, b = a.astype(np.float64), b.astype(np.float64)
    delta = a - b
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {'count': int(a.size), 'max_abs': float(np.max(np.abs(delta))),
            'rms': float(np.sqrt(np.mean(delta * delta))), 'graph_norm': an, 'scalar_norm': bn,
            'cosine': float(np.dot(a, b) / (an * bn)) if an and bn else None}


def boxes_iou(a, b):
    ax, ay, aw, ah = a['geometry'][:4]
    bx, by, bw, bh = b['geometry'][:4]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def parse(stdout):
    rows = {}
    for line in stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == 'CASE':
            _, idx, h, w, n = parts
            rows[int(idx)] = {'height': int(h), 'width': int(w), 'declared_boxes': int(n), 'boxes': []}
        elif parts and parts[0] == 'BOX':
            idx, index = int(parts[1]), int(parts[2])
            assert index == len(rows[idx]['boxes'])
            values = list(map(float, parts[3:16]))
            text = bytes.fromhex(parts[16] if len(parts) > 16 else '').decode('utf-8', errors='strict')
            rows[idx]['boxes'].append({'geometry': values, 'text': text})
    assert sorted(rows) == list(range(6)), rows
    for row in rows.values():
        assert row['declared_boxes'] == len(row['boxes'])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if DEVICE not in ('cpu', 'gpu'):
        raise ValueError('unknown backend')
    gpu = None
    if DEVICE == 'gpu':
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], text=True).strip()
        if not gpu:
            raise RuntimeError('actual CUDA GPU required')
    models = {key: download(*entry) for key, entry in MODELS.items()}
    build = SCRATCH / 'build'
    options = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF',
               '-DGGML_BLAS=OFF', '-DGGML_OPENMP=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF']
    options += shlex.split(os.environ.get('CMAKE_EXTRA', '-DGGML_CUDA=OFF'))
    call(['cmake', '-S', ROOT, '-B', build, *options])
    call(['cmake', '--build', build, '--parallel', '2', '--target', 'test-ppocrv6-det-policy', 'test-ppocrv6-det-live'])
    call([build / 'test-ppocrv6-det-policy'])
    image_root = SCRATCH / 'images'
    image_root.mkdir(exist_ok=True)
    source_image = ROOT / 'tests/regression/images/fox.png'
    fox = Image.open(source_image).convert('RGB')
    assert fox.size == (800, 200)
    paths = [image_root / name for name in ('white-a.png', 'fox-a.png', 'white-b.png', 'fox-b.png', 'fox-again.png', 'black-b.png')]
    Image.new('RGB', (800, 200), 'white').save(paths[0])
    fox.save(paths[1]); Image.new('RGB', (320, 192), 'white').save(paths[2])
    fox.resize((320, 80), Image.Resampling.LANCZOS).save(image_root / 'line.png')
    page = Image.new('RGB', (320, 192), 'white'); page.paste(Image.open(image_root / 'line.png'), (0, 56)); page.save(paths[3])
    fox.save(paths[4]); Image.new('RGB', (320, 192), 'black').save(paths[5])
    modes = {
        'scalar': {'CRISPEMBED_PPOCRV6_DET_SCALAR': '1'},
        'default': {},
        'zero': {'CRISPEMBED_PPOCRV6_DET_ACCEPT_EMPTY': '0', 'CRISPEMBED_PPOCRV6_DET_GRAPH_REBUILD': '0'},
        'empty_only': {'CRISPEMBED_PPOCRV6_DET_ACCEPT_EMPTY': '1'},
        'rebuild_only': {'CRISPEMBED_PPOCRV6_DET_GRAPH_REBUILD': '1'},
        'both': {'CRISPEMBED_PPOCRV6_DET_ACCEPT_EMPTY': '1', 'CRISPEMBED_PPOCRV6_DET_GRAPH_REBUILD': '1'},
        'compare': {'CRISPEMBED_PPOCRV6_DET_ACCEPT_EMPTY': '1', 'CRISPEMBED_PPOCRV6_DET_GRAPH_REBUILD': '1',
                    'CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE': '1'},
    }
    keys = ['CRISPEMBED_PPOCRV6_DET_SCALAR', 'CRISPEMBED_PPOCRV6_DET_ACCEPT_EMPTY',
            'CRISPEMBED_PPOCRV6_DET_GRAPH_REBUILD', 'CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE',
            'CRISPEMBED_PPOCRV6_FORCE_CPU', 'CRISPEMBED_PPOCRV6_DET_GPU',
            'CRISPEMBED_PPOCRV6_DET_GPU_LOAD', 'CRISPEMBED_PPOCRV6_DET_F32_WEIGHTS']
    rows, timing, dirs = {}, {}, {}
    report = {'schema': 'ppocrv6-issue164-live-v1', 'source_commit': source, 'backend': DEVICE, 'gpu': gpu,
              'models': {k: {'repository': v[0], 'revision': v[1], 'filename': v[2], 'sha256': v[3]} for k, v in MODELS.items()},
              'input_sha256': {p.name: sha(p) for p in paths}, 'modes': {}, 'comparisons': [], 'passed': False}
    try:
        for mode, changes in modes.items():
            env = {k: v for k, v in os.environ.items() if k not in keys}
            env.update(changes, CRISPEMBED_PPOCRV6_DET_BENCH='1', CRISPEMBED_PPOCRV6_DET_GPU='1' if DEVICE == 'gpu' else '0')
            if DEVICE == 'cpu': env['CRISPEMBED_PPOCRV6_FORCE_CPU'] = '1'
            directory = SCRATCH / mode
            directory.mkdir(exist_ok=True)
            dirs[mode] = directory
            start = time.monotonic()
            result = subprocess.run([str(build / 'test-ppocrv6-det-live'), str(models['det']), str(models['rec']),
                                     str(directory), *map(str, paths)], cwd=ROOT, env=env, text=True,
                                    capture_output=True, timeout=1200)
            (OUT / (mode + '.stdout.txt')).write_text(result.stdout)
            # The log itself remains on the remote artifact; no machine paths enter the report.
            (OUT / (mode + '.stderr.txt')).write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'{mode}: probe exit {result.returncode}')
            rows[mode] = parse(result.stdout)
            for index, row in rows[mode].items():
                expected_width = 800 if index in (0, 1, 4) else 320
                assert (row['height'], row['width']) == (192, expected_width), (mode, index, 'stale/wrong probability shape')
            benches = re.findall(r'\[ppocrv6-det-bench\] preprocess_ms=([0-9.]+) graph_ms=([0-9.]+) total_ms=([0-9.]+) boxes=(\d+) accepted=([01])', result.stderr)
            if len(benches) != 6:
                raise ValueError(f'{mode}: missing detector timing records: {len(benches)}')
            timing[mode] = [{'preprocess_ms': float(a), 'graph_ms': float(b), 'total_ms': float(c), 'boxes': int(d), 'accepted': int(e)} for a,b,c,d,e in benches]
            if mode != 'scalar':
                if DEVICE == 'gpu' and 'persistent GGML detector graph ready (CUDA' not in result.stderr:
                    raise ValueError('GPU mode did not execute a CUDA detector graph')
                if DEVICE == 'cpu' and 'persistent GGML detector graph ready (CPU' not in result.stderr:
                    raise ValueError('CPU mode did not execute a CPU detector graph')
            report['modes'][mode] = {'elapsed_seconds': time.monotonic()-start, 'cases': list(rows[mode].values()),
                                     'timings': timing[mode], 'graph_builds': result.stderr.count('persistent GGML detector graph ready')}
        # Literal zero must preserve both original results and actual map bytes.
        assert rows['default'] == rows['zero']
        for index in range(6):
            assert (dirs['default'] / f'{index}-prob.bin').read_bytes() == (dirs['zero'] / f'{index}-prob.bin').read_bytes()
        for mode in ('default', 'both', 'compare'):
            first = np.fromfile(dirs[mode] / '1-prob.bin', dtype=np.float32)
            repeated = np.fromfile(dirs[mode] / '4-prob.bin', dtype=np.float32)
            replay = metrics(first, repeated)
            report['comparisons'].append({'mode': mode, 'case': 'same-text-after-shape-switch', 'probability': replay})
            assert replay['max_abs'] <= 1e-6 and rows[mode][1] == rows[mode][4], (mode, 'replay drift', replay)
        # Original behavior and its literal-zero controls must match; each gate has its own A/B.
        assert [x['accepted'] for x in timing['default']] == [0, 1, 0, 0, 1, 0]
        assert [x['accepted'] for x in timing['zero']] == [0, 1, 0, 0, 1, 0]
        assert [x['accepted'] for x in timing['empty_only']] == [1, 1, 0, 0, 1, 0]
        assert [x['accepted'] for x in timing['rebuild_only']] == [0, 1, 0, 1, 1, 0]
        assert [x['accepted'] for x in timing['both']] == [1, 1, 1, 1, 1, 1]
        assert [x['accepted'] for x in timing['compare']] == [0] * 6
        assert report['modes']['default']['graph_builds'] == 1
        assert report['modes']['both']['graph_builds'] == 4
        assert report['modes']['compare']['graph_builds'] == 4
        for index in (0, 2, 5):
            assert rows['scalar'][index]['declared_boxes'] == 0, ('blank baseline detected text', index)
        for index in (1, 3, 4):
            assert rows['scalar'][index]['declared_boxes'] > 0, ('text baseline returned no text', index)
        for mode in ('default', 'zero', 'empty_only', 'rebuild_only', 'both', 'compare'):
            for index in range(6):
                baseline, candidate = rows['scalar'][index], rows[mode][index]
                assert len(baseline['boxes']) == len(candidate['boxes']), (mode,index,'box count')
                unmatched = list(candidate['boxes'])
                for box in baseline['boxes']:
                    best = max(unmatched, key=lambda other: boxes_iou(box, other))
                    overlap = boxes_iou(box, best)
                    assert overlap >= 0.90 and box['text'] == best['text'], (mode,index,'decoded/geometry',overlap,box,best)
                    unmatched.remove(best)
                a = np.fromfile(dirs[mode] / f'{index}-prob.bin', dtype=np.float32)
                b = np.fromfile(dirs['scalar'] / f'{index}-prob.bin', dtype=np.float32)
                row = {'mode': mode, 'case': index, 'probability': metrics(a,b),
                       'threshold_disagreements': int(np.count_nonzero((a >= 0.2) != (b >= 0.2)))}
                report['comparisons'].append(row)
        for index in range(6):
            pairs = [('graph_stage0','backbone_stage0'), ('graph_med_adjust0','med_adjust0'),
                     ('graph_head_down','head_down'), ('graph_head_final_pre','head_final_pre'),
                     ('graph_head_final',None)]
            for graph, cpu in pairs:
                a = np.fromfile(dirs['compare'] / f'{index}-{graph}.bin', dtype=np.float32)
                b = np.fromfile(dirs['compare'] / (f'{index}-{cpu}.bin' if cpu else f'{index}-prob.bin'), dtype=np.float32)
                report['comparisons'].append({'mode':'compare_stages','case':index,'stage':graph,'metrics':metrics(a,b)})
        report['passed'] = True
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('PASS: medium F16 detector/recognizer, empty frames, alternating shapes, scalar/graph maps and text')


if __name__ == '__main__':
    main()
