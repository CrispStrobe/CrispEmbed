#!/usr/bin/env python3
"""Hosted public CLI and offline checks for both experimental Josev aliases."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
MODELS = {
    'Q8_0': ('josev2b-systemone-q8_0-experimental', 'josev2b-systemone-q8_0.gguf', 'fc1663bbd050938ff0bdb1b20ca52bd9ff7579dca63c0f7f9a15b3c3b7e36b85'),
    'Q4_K_MIX': ('josev2b-systemone-q4k-mix-experimental', 'josev2b-systemone-q4_k-down8.gguf', 'adc7fa473febfef3fac10eac6fa0bebb6f68932675d1613e24706a3b74d5e582'),
}
GATES = ('PACKED', 'COMPOSED', 'BACKBONE', 'HEAD', 'TEXT_FRONTEND', 'ANSWER_SERIALIZER', 'QUANT')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args, env, logfile, allow_failure=False):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    logfile.write_text(result.stderr)
    if result.returncode and not allow_failure:
        raise RuntimeError(f'{logfile.name}: exit {result.returncode}')
    return result


def compare(response, gold):
    assert response['model'] == gold['model'] and response['usage'] == gold['usage']
    assert set(response['answers']) == set(gold['answers'])
    errors = []
    for key, original in gold['answers'].items():
        actual = response['answers'][key]
        assert actual['type'] == original['type'] and set(actual) == set(original)
        if actual['type'] == 'noul':
            assert 0 <= actual['noul'] <= 1 and math.isfinite(actual['noul'])
            assert (actual['noul'] >= 0.5) == (original['noul'] >= 0.5)
            errors.append(abs(actual['noul'] - original['noul']))
        else:
            assert set(actual['probabilities']) == set(original['probabilities'])
            for name, value in actual['probabilities'].items():
                assert math.isfinite(value) and 0 <= value <= 1
                errors.append(abs(value - original['probabilities'][name]))
            assert abs(sum(actual['probabilities'].values()) - 1) < 0.002
            assert max(actual['probabilities'], key=actual['probabilities'].get) == max(original['probabilities'], key=original['probabilities'].get)
            assert math.isfinite(actual['confidence']) and 0 <= actual['confidence'] <= 1
            errors.append(abs(actual['confidence'] - original['confidence']))
            if actual['type'] == 'choice':
                assert actual['choice'] == original['choice']
            else:
                assert actual['type'] == 'score' and actual['legend'] == original['legend'] and math.isfinite(actual['score'])
                errors.append(abs(actual['score'] - original['score']))
    return {'typed_json_exact': response == gold, 'max_numeric_abs': max(errors, default=0.0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    build, cache = SCRATCH / 'build', SCRATCH / 'cache'
    cache.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_JOSEV_')}
    env.pop('CRISPEMBED_OFFLINE', None)
    env.update(CRISPEMBED_CACHE_DIR=str(cache), CRISPEMBED_QWEN35_OCR='1', CRISPEMBED_QWEN35_QUANT='1')
    for gate in GATES:
        env['CRISPEMBED_JOSEV_' + gate] = '1'
    report = {'schema': 'josev-experimental-alias-consumers-v1',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Four frozen typed requests through the published CLI; no independent task panel, GPU or model-quality qualification.',
              'models': {}, 'passed': False}
    try:
        subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                        '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_LLAMAFILE=ON'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'crispembed-cli', '--parallel', '2'], check=True)
        cli = build / 'crispembed'
        frozen = json.loads((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_text())
        reference = json.loads((ROOT / 'tests/results/josev2b/original-reference.json').read_text())
        assert frozen['panel_sha256'] == reference['panel_sha256'] and len(frozen['records']) == len(reference['cases']) == 4
        gold = {row['id']: row['official_response'] for row in reference['cases']}
        for variant, (alias, filename, expected) in MODELS.items():
            directory = OUT / variant
            directory.mkdir(exist_ok=True)
            responses, case_reports = {}, {}
            for record in frozen['records']:
                cid = record['id']
                request = json.dumps(record['request'], ensure_ascii=False, separators=(',', ':'))
                result = run([cli, '-m', alias, '--systemone', request, '-t', '2'], env, directory / (cid + '.stderr.txt'))
                response = json.loads(result.stdout)
                responses[cid] = response
                (directory / (cid + '.json')).write_text(json.dumps(response, ensure_ascii=False) + '\n')
                case_reports[cid] = compare(response, gold[cid])
            matches = list(cache.rglob(filename))
            assert len(matches) == 1 and sha(matches[0]) == expected, (alias, 'immutable SHA mismatch')
            first = frozen['records'][0]
            request = json.dumps(first['request'], ensure_ascii=False, separators=(',', ':'))
            offline = dict(env, CRISPEMBED_OFFLINE='1')
            args = [cli, '-m', alias, '--systemone', request, '-t', '2']
            assert json.loads(run(args, offline, directory / 'offline.stderr.txt').stdout) == responses[first['id']], 'offline output drift'
            assert json.loads(run([cli, '-m', matches[0], '--systemone', request, '-t', '2'], offline,
                                  directory / 'direct.stderr.txt').stdout) == responses[first['id']], 'alias/path drift'
            denied = dict(offline, CRISPEMBED_JOSEV_QUANT='0')
            assert run(args, denied, directory / 'gate-off.stderr.txt', allow_failure=True).returncode != 0, 'quantized gate-off accepted'
            assert json.loads(run(args, offline, directory / 'recovery.stderr.txt').stdout) == responses[first['id']], 'gate recovery drift'
            report['models'][variant] = {'alias': alias, 'filename': filename, 'sha256': expected,
                'cases': case_reports, 'offline_equal': True, 'explicit_path_equal': True, 'gate_off_rejected_and_recovered': True}
        report['passed'] = True
    except BaseException as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        traceback.print_exc()
        raise
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('PASS: both experimental aliases, frozen typed requests, fresh-cache SHA and offline/direct-path recovery')


if __name__ == '__main__':
    main()
