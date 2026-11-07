#!/usr/bin/env python3
"""Hosted Python-versus-CLI checks on the published experimental MoEdern Q4 aliases."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
MODELS = {
    '27m': ('moedern-27m-q4k-experimental', 'moedern-27m-text-q4_k-mix.gguf',
            '5d31ac7e5495690e9f3bbe428166764bfee980b3697916ae71366bf717f70d62'),
    '115m': ('moedern-115m-q4k-experimental', 'moedern-115m-text-q4_k-mix.gguf',
             '01d0e507430ba2f839709648272e3e3e622e02cc98aafac5b179e36b145db110'),
}
CASES = ('q-de', 'q-empty', 'q-markers', 'd-de', 'd-unicode', 'd-punctuation')
GATES = ('PUBLIC_TEXT', 'PACKED', 'TEXT_COMPOSED', 'TEXT_FRONTEND', 'COLBERT', 'FAMILY', 'FAST', 'QUANT', 'Q4_K_PADDED', 'Q4_K_MIXED')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cli_json(args, env, logfile):
    r = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    logfile.write_text(r.stderr)
    if r.returncode:
        raise RuntimeError(f'{logfile.name}: exit {r.returncode}')
    return json.loads(r.stdout)


def compare_rows(client, cli):
    for key in ('role', 'ids', 'supplied_mask', 'effective_mask', 'scoring_mask', 'expansion_positions'):
        if client[key] != cli[key]:
            raise AssertionError(f'client/CLI {key} differs')
    a, b = client['selected_vectors'], cli['selected_vectors']
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in a + b):
        raise AssertionError('selected vectors malformed')
    worst = max(abs(x - y) for x, y in zip(a, b))
    if worst > 1e-7:
        raise AssertionError(f'client/CLI vectors differ: {worst}')
    return worst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    build, cache = SCRATCH / 'build', SCRATCH / 'cache'
    cache.mkdir(exist_ok=True)
    for name in list(os.environ):
        if name.startswith('CRISPEMBED_MOEDERN_'):
            del os.environ[name]
    for gate in GATES:
        os.environ['CRISPEMBED_MOEDERN_' + gate] = '1'
    os.environ['CRISPEMBED_CACHE_DIR'] = str(cache)
    os.environ.pop('CRISPEMBED_OFFLINE', None)
    report = {'schema': 'moedern-q4-python-consumer-v1',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Six frozen cases and one MaxSim pair, Python versus CLI on the same model; no SciFact, original-quality or GPU claim.',
              'models': {}, 'passed': False}
    try:
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                        '-DCRISPEMBED_BUILD_SHARED=ON', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF',
                        '-DCRISPEMBED_NATIVE=OFF', '-DGGML_LLAMAFILE=ON'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'crispembed-shared', 'crispembed-cli', '--parallel', '2'], check=True)
        lib, cli = build / 'libcrispembed.so', build / 'crispembed'
        sys.path.insert(0, str(ROOT / 'python'))
        from crispembed import CrispMoEdern
        panel = json.loads((ROOT / 'tests/fixtures/moedern-colbert/panel-v1.json').read_text())
        cases = {c['id']: c for c in panel['cases'] if c['id'] in CASES}
        assert set(cases) == set(CASES)
        for name, (alias, filename, expected) in MODELS.items():
            d = OUT / name
            d.mkdir(exist_ok=True)
            # The CLI resolves the alias into a fresh cache; the public Python client then resolves it offline.
            first = cases[CASES[0]]
            first_cli = cli_json([cli, '-m', alias, '--moedern', first['role'], first['text'], '-t', '2'], os.environ.copy(), d / 'download.stderr')
            found = list(cache.rglob(filename))
            assert len(found) == 1 and sha(found[0]) == expected, 'model digest differs'
            os.environ['CRISPEMBED_OFFLINE'] = '1'
            result = {'alias': alias, 'sha256': expected, 'cases': {}}
            report['models'][name] = result
            with CrispMoEdern(alias, lib_path=str(lib), auto_download=False, offline=True) as model:
                for cid in CASES:
                    c = cases[cid]
                    ref = first_cli if cid == CASES[0] else cli_json(
                        [cli, '-m', alias, '--moedern', c['role'], c['text'], '-t', '2'], os.environ.copy(), d / (cid + '.stderr'))
                    got = model.encode(c['text'], c['role'])
                    result['cases'][cid] = {'tokens': len(ref['ids']), 'selected_rows': len(ref['selected_vectors']) // 128,
                                            'max_abs_client_vs_cli': compare_rows(got, ref)}
                q, doc = cases['q-de']['text'], cases['d-de']['text']
                client_score = model.score(q, doc)
                cli_score = cli_json([cli, '-m', alias, '--moedern-score', q, doc, '-t', '2'], os.environ.copy(), d / 'score.stderr')['score']
                assert abs(client_score - cli_score) <= 1e-6
                result.update(client_score=client_score, cli_score=cli_score, offline_alias=True)
            os.environ.pop('CRISPEMBED_OFFLINE', None)
        report['passed'] = True
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        raise
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
