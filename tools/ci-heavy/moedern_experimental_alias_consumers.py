#!/usr/bin/env python3
"""Hosted fresh-cache and decoded consumer A/B for both experimental MoEdern aliases."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
MODELS = {
    '27m': {
        'f32': ('moedern-27m', 'moedern-27m-text-f32.gguf', 'e836acbb507ebf9d26ad6a3f5e5712487f1b90f1430b1174acb07de07e85ce18'),
        'q4': ('moedern-27m-q4k-experimental', 'moedern-27m-text-q4_k-mix.gguf', '5d31ac7e5495690e9f3bbe428166764bfee980b3697916ae71366bf717f70d62'),
    },
    '115m': {
        'f32': ('moedern-115m', 'moedern-115m-text-f32.gguf', '4150412f66caaa058c90ad0ed6cc4e318a0ea146f4ce9e59de62f83975903766'),
        'q4': ('moedern-115m-q4k-experimental', 'moedern-115m-text-q4_k-mix.gguf', '01d0e507430ba2f839709648272e3e3e622e02cc98aafac5b179e36b145db110'),
    },
}
GATES = ('PUBLIC_TEXT', 'PACKED', 'TEXT_COMPOSED', 'TEXT_FRONTEND', 'COLBERT', 'FAMILY', 'FAST', 'QUANT', 'Q4_K_PADDED', 'Q4_K_MIXED')
CASES = ('q-de', 'q-empty', 'q-markers', 'd-de', 'd-unicode', 'd-punctuation')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args, *, env, logfile, allow_failure=False):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    logfile.write_text(result.stderr)
    if not allow_failure and result.returncode:
        raise RuntimeError(f'{logfile.name}: exit {result.returncode}')
    return result


def validate_result(row, role):
    assert row['role'] == role
    n = len(row['ids'])
    assert n > 0 and all(len(row[k]) == n for k in ('supplied_mask', 'effective_mask', 'scoring_mask'))
    assert all(v in (0, 1) for k in ('supplied_mask', 'effective_mask', 'scoring_mask') for v in row[k])
    vectors = np.asarray(row['selected_vectors'], dtype=np.float64)
    assert vectors.size == sum(row['scoring_mask']) * 128 and vectors.size > 0 and np.isfinite(vectors).all()
    vectors = vectors.reshape(-1, 128)
    norms = np.linalg.norm(vectors, axis=1)
    assert (norms > 0.99).all() and (norms < 1.01).all()
    return vectors


def compare(a, b):
    assert a.shape == b.shape
    dots = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    delta = a - b
    return {'rows': int(a.shape[0]), 'min_cosine': float(dots.min()), 'mean_cosine': float(dots.mean()),
            'max_abs': float(np.abs(delta).max()), 'rms': float(np.sqrt(np.mean(delta * delta))),
            'f32_norm': float(np.linalg.norm(a)), 'q4_norm': float(np.linalg.norm(b))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    build = SCRATCH / 'build'
    cache = SCRATCH / 'cache'
    cache.mkdir(exist_ok=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = {'schema': 'moedern-experimental-alias-consumers-v1', 'source_commit': source,
              'scope': 'Six frozen raw-text cases, three-document within-precision ranking; no SciFact or model-quality qualification.',
              'models': {}, 'passed': False}
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_MOEDERN_')}
    env.update({'CRISPEMBED_CACHE_DIR': str(cache)})
    env.pop('CRISPEMBED_OFFLINE', None)
    for gate in GATES:
        env['CRISPEMBED_MOEDERN_' + gate] = '1'
    try:
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                        '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_LLAMAFILE=ON'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'crispembed-cli', '--parallel', '2'], check=True)
        cli = build / 'crispembed'
        panel = json.loads((ROOT / 'tests/fixtures/moedern-colbert/panel-v1.json').read_text())
        cases = {row['id']: row for row in panel['cases'] if row['id'] in CASES}
        assert set(cases) == set(CASES)
        for model, variants in MODELS.items():
            report['models'][model] = {'variants': {}, 'comparisons': {}, 'rankings': {}}
            results, vectors = {}, {}
            for kind, (alias, filename, expected) in variants.items():
                directory = OUT / model / kind
                directory.mkdir(parents=True, exist_ok=True)
                results[kind], vectors[kind] = {}, {}
                for cid in CASES:
                    case = cases[cid]
                    result = command([cli, '-m', alias, '--moedern', case['role'], case['text'], '-t', '2'],
                                     env=env, logfile=directory / (cid + '.stderr.txt'))
                    row = json.loads(result.stdout)
                    vectors[kind][cid] = validate_result(row, case['role'])
                    results[kind][cid] = row
                    (directory / (cid + '.json')).write_text(json.dumps(row, ensure_ascii=False) + '\n')
                matches = list(cache.rglob(filename))
                assert len(matches) == 1 and sha(matches[0]) == expected, (alias, 'immutable model SHA mismatch')
                offline = dict(env, CRISPEMBED_OFFLINE='1')
                replay = command([cli, '-m', alias, '--moedern', cases['q-de']['role'], cases['q-de']['text'], '-t', '2'],
                                 env=offline, logfile=directory / 'offline.stderr.txt')
                assert json.loads(replay.stdout) == results[kind]['q-de'], (alias, 'offline alias drift')
                explicit = command([cli, '-m', matches[0], '--moedern', cases['q-de']['role'], cases['q-de']['text'], '-t', '2'],
                                   env=offline, logfile=directory / 'explicit-path.stderr.txt')
                assert json.loads(explicit.stdout) == results[kind]['q-de'], (alias, 'alias/path drift')
                score = command([cli, '-m', alias, '--moedern-score', cases['q-de']['text'], cases['d-de']['text'], '-t', '2'],
                                env=offline, logfile=directory / 'score.stderr.txt')
                actual_score = float(json.loads(score.stdout)['score'])
                independent_score = float(np.max(vectors[kind]['q-de'] @ vectors[kind]['d-de'].T, axis=1).sum())
                assert np.isfinite(actual_score) and abs(actual_score - independent_score) <= 0.001, (alias, actual_score, independent_score)
                report['models'][model]['variants'][kind] = {'alias': alias, 'filename': filename, 'sha256': expected,
                    'offline_equal': True, 'explicit_path_equal': True, 'score': actual_score, 'independent_score': independent_score}
                if kind == 'q4':
                    denied = dict(offline, CRISPEMBED_MOEDERN_Q4_K_MIXED='0')
                    failure = command([cli, '-m', alias, '--moedern', 'query', cases['q-de']['text'], '-t', '2'],
                                      env=denied, logfile=directory / 'gate-off.stderr.txt', allow_failure=True)
                    assert failure.returncode != 0 and ('gate' in failure.stderr.lower() or 'q4' in failure.stderr.lower()), (alias, 'gate-off did not fail')
                    recovered = command([cli, '-m', alias, '--moedern', 'query', cases['q-de']['text'], '-t', '2'],
                                        env=offline, logfile=directory / 'recovery.stderr.txt')
                    assert json.loads(recovered.stdout) == results[kind]['q-de'], (alias, 'gate recovery changed output')
                    report['models'][model]['variants'][kind]['gate_off_rejected_and_recovered'] = True
            for cid in CASES:
                a, b = results['f32'][cid], results['q4'][cid]
                for key in ('role', 'ids', 'supplied_mask', 'effective_mask', 'scoring_mask', 'expansion_positions'):
                    assert a[key] == b[key], (model, cid, key)
                metric = compare(vectors['f32'][cid], vectors['q4'][cid])
                assert metric['min_cosine'] > 0.90, (model, cid, 'coarse consumer sanity', metric)
                report['models'][model]['comparisons'][cid] = metric
            for kind in ('f32', 'q4'):
                ranks = {}
                for query in ('q-de', 'q-empty', 'q-markers'):
                    scores = {doc: float(np.max(vectors[kind][query] @ vectors[kind][doc].T, axis=1).sum())
                              for doc in ('d-de', 'd-unicode', 'd-punctuation')}
                    ranks[query] = {'scores': scores, 'order': sorted(scores, key=lambda d: (-scores[d], d))}
                report['models'][model]['rankings'][kind] = ranks
            report['models'][model]['top1_agreement'] = sum(
                report['models'][model]['rankings']['f32'][q]['order'][0] == report['models'][model]['rankings']['q4'][q]['order'][0]
                for q in ('q-de', 'q-empty', 'q-markers'))
        report['passed'] = True
    except BaseException as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        traceback.print_exc()
        raise
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('PASS: both experimental aliases, fresh-cache hashes, online/offline and direct-path consumers')


if __name__ == '__main__':
    main()
