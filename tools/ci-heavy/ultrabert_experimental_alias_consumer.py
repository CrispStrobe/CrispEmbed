#!/usr/bin/env python3
"""Hosted immutable alias, public CLI and cache smoke for experimental UltraBERT."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
ALIAS = 'ultrabert-q4k-experimental'
NAME = 'ultrabert-embed-150m-q4_k-last-four-ffn.gguf'
EXPECTED = '087c9be0fa50ba57db74e650889521c985d306863c94aabf731764538eb4533c'
TEXTS = ['How do plants turn light into energy?', 'Pflanzen wandeln Licht in chemische Energie um.',
         'A recipe for apple pie with cinnamon.']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args, env, label, allow_failure=False):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    (OUT / (label + '.stderr.txt')).write_text(result.stderr)
    if result.returncode and not allow_failure:
        raise RuntimeError(f'{label}: exit {result.returncode}')
    return result


def validate(rows):
    assert len(rows) == len(TEXTS)
    vectors = []
    for text, row in zip(TEXTS, rows):
        assert row['text'] == text
        v = np.asarray(row['embedding'], dtype=np.float64)
        assert v.shape == (768,) and np.isfinite(v).all() and np.linalg.norm(v) > 0
        vectors.append(v)
    return np.stack(vectors)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    build, cache = SCRATCH / 'build', SCRATCH / 'cache'
    cache.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_ULTRABERT')}
    env.pop('CRISPEMBED_OFFLINE', None)
    env.update(CRISPEMBED_CACHE_DIR=str(cache), CRISPEMBED_ULTRABERT='1', CRISPEMBED_ULTRABERT_QUANT='1',
               CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN='1', CRISPEMBED_ULTRABERT_TRUNCATE='1')
    report = {'schema': 'ultrabert-experimental-alias-consumer-v1',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'alias': ALIAS, 'filename': NAME, 'sha256': EXPECTED,
              'scope': 'Three-text CLI and cache smoke; no F32, original, SciFact, GPU or model quality qualification.',
              'passed': False}
    try:
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                        '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_LLAMAFILE=ON'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'crispembed-cli', '--parallel', '2'], check=True)
        cli = build / 'crispembed'
        args = [cli, '-m', ALIAS, '--json', '-t', '2', *TEXTS]
        online = run(args, env, 'online')
        rows = json.loads(online.stdout)
        vectors = validate(rows)
        (OUT / 'online.json').write_text(json.dumps(rows, ensure_ascii=False) + '\n')
        matches = list(cache.rglob(NAME))
        assert len(matches) == 1 and sha(matches[0]) == EXPECTED, 'immutable model SHA mismatch'
        offline = dict(env, CRISPEMBED_OFFLINE='1')
        assert json.loads(run(args, offline, 'offline').stdout) == rows, 'offline alias drift'
        assert json.loads(run([cli, '-m', matches[0], '--json', '-t', '2', *TEXTS], offline, 'direct').stdout) == rows, 'alias/path drift'
        disabled = dict(offline, CRISPEMBED_ULTRABERT_QUANT='0')
        negative = run(args, disabled, 'gate-off', allow_failure=True)
        assert negative.returncode != 0, 'quantized gate-off unexpectedly accepted'
        assert json.loads(run(args, offline, 'recovery').stdout) == rows, 'gate recovery changed output'
        norms = np.linalg.norm(vectors, axis=1)
        cos = (vectors @ vectors.T) / (norms[:, None] * norms[None, :])
        report.update(passed=True, dimensions=768, row_norms=norms.tolist(), pairwise_cosine=cos.tolist(),
                      offline_equal=True, explicit_path_equal=True, gate_off_rejected_and_recovered=True)
    except BaseException as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        traceback.print_exc()
        raise
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('PASS: experimental alias, actual CLI, fresh-cache SHA and offline/direct-path recovery')


if __name__ == '__main__':
    main()
