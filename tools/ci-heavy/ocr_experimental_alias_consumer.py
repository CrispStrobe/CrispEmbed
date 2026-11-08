#!/usr/bin/env python3
"""Bounded hosted CLI/cache smoke for a published experimental OCR alias."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
MODELS = {
    'minicpm': ('minicpm-ocr-q4_k-mix-experimental', 'minicpm-ocr-1.0-q4_k-mix.gguf',
                '5ba6cd538fd166a0243f08d49101fe1d07d8f353f7b554078b42d057c5c511ed',
                {'CRISPEMBED_MINICPM_OCR': '1', 'CRISPEMBED_QWEN35_OCR': '1', 'CRISPEMBED_QWEN35_QUANT': '1'},
                'CRISPEMBED_MINICPM_OCR'),
    'unlimited': ('unlimited-ocr2-q4_k-mix-experimental', 'unlimited-ocr-2-q4_k-mix.gguf',
                  'b71f158d36713e1c7ee39b7fb48193aa87a8592bca666f92caeae41093da493a',
                  {'CRISPEMBED_UNLIMITED_OCR2': '1'}, 'CRISPEMBED_UNLIMITED_OCR2'),
    'unlimited-q8': ('unlimited-ocr2-q8_0-experimental', 'unlimited-ocr-2-q8_0-sh16.gguf',
                     'db317f1066a4653dfa81dc481529c3f5259ab26a612fee02a5c835cde0fd7f48',
                     {'CRISPEMBED_UNLIMITED_OCR2': '1'}, 'CRISPEMBED_UNLIMITED_OCR2'),
    'got': ('got-ocr2-q4_k-experimental', 'got-ocr2-q4_k.gguf',
            'ba276f2e246273af7d1bcd58f35ac7bae09c30340b923ed4440effe82d5e7613',
            {'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE': '1'}, 'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE'),
    'got-q8': ('got-ocr2-q8_0-experimental', 'got-ocr2-q8_0-r2.gguf',
               '15920499f65b71bcfbf4d44d0e92ffbaf7f7c4523ad13f7b7bef160dc5edf098',
               {'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE': '1'}, 'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE'),
    'got-mix': ('got-ocr2-q8-q4k-mix-experimental', 'got-ocr2-q8-q4k-mix.gguf',
                '5ccd3d291e52cb9a0f923ccd1b7f70f7c25469382bbe7492e7373ac1bd2b73b5',
                {'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE': '1'}, 'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args, env, label):
    p = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    (OUT / (label + '.stderr.txt')).write_text(p.stderr)
    (OUT / (label + '.stdout.txt')).write_text(p.stdout)
    return p


def decoded(result):
    if result.returncode != 0:
        raise RuntimeError(f'CLI failed with exit {result.returncode}')
    row = json.loads(result.stdout)
    if not isinstance(row, dict) or not isinstance(row.get('text'), str) or not row['text'].strip():
        raise ValueError('CLI returned no OCR text')
    return row


def check_gate_rejected(result, key):
    expected = {'minicpm': 'failed to load OCR model',
                'unlimited': 'requires CRISPEMBED_UNLIMITED_OCR2=1',
                'got': 'CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1'}
    if expected[key.split('-')[0]] not in result.stderr:
        raise AssertionError('gate-off failure lacked the expected engine diagnostic')
    try:
        decoded(result)
    except (ValueError, RuntimeError):
        return
    raise AssertionError('disabled engine gate unexpectedly produced OCR text')


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        raise SystemExit('choose ' + ', '.join(MODELS))
    key = sys.argv[1]
    alias, filename, expected, gates, gate = MODELS[key]
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    build, cache = SCRATCH / 'build', SCRATCH / 'cache'
    cache.mkdir(exist_ok=True)
    image = ROOT / 'tests/regression/images/fox.png'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CRISPEMBED_MINICPM_', 'CRISPEMBED_QWEN35_', 'CRISPEMBED_UNLIMITED_'))}
    env.pop('CRISPEMBED_OFFLINE', None)
    env.update(gates, CRISPEMBED_CACHE_DIR=str(cache))
    report = {'schema': 'ocr-experimental-alias-consumer-v1',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'alias': alias, 'filename': filename, 'sha256': expected,
              'image': 'tests/regression/images/fox.png', 'image_sha256': sha(image), 'max_tokens': 32,
              'scope': 'Single bounded image CLI/cache smoke; no EOS, original/F16 comparison, GPU or quality qualification.',
              'passed': False}
    try:
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                        '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_LLAMAFILE=ON'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'crispembed-cli', '--parallel', '2'], check=True)
        cli = build / 'crispembed'
        def args(model):
            result = [cli, '-m', model, '--ocr', image, '--ocr-max-tokens', '32', '--json', '-t', '2']
            if key.startswith('got'):
                result.extend(['--accept-license', 'tongyi-qianwen-research'])
            return result
        online = decoded(run(args(alias), env, 'online'))
        matches = list(cache.rglob(filename))
        assert len(matches) == 1 and sha(matches[0]) == expected, 'immutable model SHA mismatch'
        offline = dict(env, CRISPEMBED_OFFLINE='1')
        assert decoded(run(args(alias), offline, 'offline')) == online, 'offline alias drift'
        assert decoded(run(args(matches[0]), offline, 'direct')) == online, 'alias/path drift'
        negative = run(args(alias), dict(offline, **{gate: '0'}), 'gate-off')
        # Recognition can fail while the CLI returns zero: require no decoded text.
        check_gate_rejected(negative, key)
        report.update(passed=True, decoded=online, offline_equal=True, explicit_path_equal=True,
                      gate_off_rejected=True)
    except BaseException as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        traceback.print_exc()
        raise
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
