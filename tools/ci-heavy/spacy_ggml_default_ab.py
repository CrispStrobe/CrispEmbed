#!/usr/bin/env python3
"""Hosted Linux x86-64/AVX2 A/B for the spaCy ggml+SGEMM default flip.

Builds the real standalone tools/spacy target twice (default: ggml engine with
tiled SGEMM; and -DCRISPEMBED_SPACY_SGEMM=OFF), then:

1. model-free CTest controls;
2. the unchanged frozen development gates (tests/spacy_parity.py and
   tests/spacy_text_parity.py, atol 2e-4 / rtol 2e-5) and the C ABI controls,
   once with the new default engine and once with CRISPEMBED_SPACY_GGML=0;
3. decoded-output A/B on 1,776 raw UD German GSD r2.14 dev+test sentences
   against an original-spaCy oracle generated on this runner from the
   authenticated wheel (no native output is ever the reference);
4. the F16 file through the scalar engine, the ggml engine and the opt-in
   CRISPEMBED_SPACY_GGML_UPCAST path, plus original spaCy replayed with the
   same F16 weights to attribute any decoded change.

The consumed native acceptance split and the W6/W7 grammar panels are never
run. The published model bundle and capsule are unchanged; timings are one
hosted CPU, one native thread unless noted, and diagnostic only.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import spacy_w7_publish as common

ROOT = common.ROOT
BUNDLE_REVISION = '659dae33997d00296ae12b76510fa4dfe26540e0'
QUDS_REFERENCE = 'dcf2ec9b4d6fba05fa5eec2634c6654266332cc9'
GSD = {
    'de_gsd-ud-dev.conllu': ('a9950b9e1bad458b3a1ffb912b75dde6db6e85810c2bcc6553c4ea3228f54ea1', 903192),
    'de_gsd-ud-test.conllu': ('f64867239949b842aab306d508b0f0e0c81c9edebfbb19629390adafc6d1c629', 1205820),
}


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    assert sys.platform == 'linux' and platform.machine() == 'x86_64'
    flags = Path('/proc/cpuinfo').read_text().split('flags', 1)[1].splitlines()[0].split()
    assert 'avx2' in flags
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-ggml-default'
    scratch.mkdir()
    out = Path(os.environ['HEAVY_OUT'])
    report = {'schema': 'crispembed.spacy.ggml-default-ab.v1', 'source': os.environ.get('GITHUB_SHA'),
              'scope': 'Linux x86-64/AVX2 A/B of the ggml+SGEMM default vs the scalar reference engine',
              'consumed_native_or_grammar_heldout_replayed': False, 'stage': 'preflight', 'passed': False}

    def save():
        (out / 'ggml-default-ab.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, env=None, check=True):
        report['stage'] = label
        save()
        with (out / (label + '.log')).open('w') as log:
            return subprocess.run([str(x) for x in command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  check=check, env=env)

    try:
        run('apt', ['sudo', 'apt-get', 'update', '-qq'])
        run('pcre', ['sudo', 'apt-get', 'install', '-y', 'libpcre2-dev'])
        bundle = scratch / 'bundle'
        bundle.mkdir()
        for name, pin in common.load_distribution().BUNDLE_FILES.items():
            common.download(f'https://huggingface.co/{common.REPOSITORY}/resolve/{BUNDLE_REVISION}/bundle/{name}',
                            bundle / name, pin['sha256'], pin['size_bytes'])
        wheel = scratch / common.WHEEL
        common.download('https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/'
                        + common.WHEEL, wheel, common.WHEEL_SHA, 14639490)
        reference = scratch / 'reference-venv'
        run('venv', [sys.executable, '-m', 'venv', reference])
        python = reference / 'bin' / 'python'
        run('reference-install', [python, '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes',
                                  '-r', ROOT / 'tools/spacy/requirements-reference.txt'])
        quds = scratch / 'quds'
        run('quds-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                           'https://github.com/CrispStrobe/quds_office_suite.git', quds])
        subprocess.run(['git', 'sparse-checkout', 'set', 'tools/proofing/spacy'], cwd=quds, check=True)
        subprocess.run(['git', 'checkout', '--detach', QUDS_REFERENCE], cwd=quds, check=True)

        builds = {}
        for name, extra in (('default', []), ('sgemm-off', ['-DCRISPEMBED_SPACY_SGEMM=OFF'])):
            build = scratch / ('build-' + name)
            run('configure-' + name, ['cmake', '-S', ROOT / 'tools/spacy', '-B', build,
                                      '-DCMAKE_BUILD_TYPE=Release', *extra])
            run('build-' + name, ['cmake', '--build', build, '--parallel', '4'])
            run('ctest-' + name, ['ctest', '--test-dir', build, '--output-on-failure'])
            builds[name] = build / 'libcrispembed-spacy.so'
        report['libraries_sha256'] = {k: common.digest(v) for k, v in builds.items()}
        report['cpu'] = Path('/proc/cpuinfo').read_text().split('model name', 1)[1].splitlines()[0].strip(' :\t')
        report['compiler'] = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]

        base_env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_SPACY')}
        base_env['CRISPEMBED_SPACY'] = '1'
        gates = {}
        for mode, ggml in (('ggml-default', None), ('scalar', '0')):
            env = dict(base_env)
            if ggml is not None:
                env['CRISPEMBED_SPACY_GGML'] = ggml
            gates[mode] = {}
            for script in ('spacy_parity.py', 'spacy_text_parity.py', 'test_spacy_text_api.py'):
                target = out / f'gate-{mode}-{script[:-3]}.json'
                args = [python, ROOT / 'tests' / script, '--library', builds['default'], '--bundle', bundle,
                        '--report', target]
                if script != 'test_spacy_text_api.py':
                    args += ['--quds-root', quds]
                result = run(f'gate-{mode}-{script[:-3]}', args, env=env, check=False)
                data = json.loads(target.read_text()) if target.is_file() else {}
                gates[mode][script] = {'exit_code': result.returncode, 'passed': data.get('passed', result.returncode == 0),
                                       'counts': data.get('counts')}
        report['development_gates'] = gates

        corpus_texts = []
        for name, (sha, size) in GSD.items():
            path = scratch / name
            common.download('https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/r2.14/' + name,
                            path, sha, size)
            corpus_texts += [line[9:].strip() for line in path.read_text(encoding='utf-8').splitlines()
                             if line.startswith('# text = ')]
        corpus = scratch / 'corpus-gsd.json'
        corpus.write_text(json.dumps({'source': 'UD_German-GSD r2.14 dev+test raw text (CC BY-SA 4.0); input only',
                                      'texts': corpus_texts}, ensure_ascii=False), encoding='utf-8')
        oracle = scratch / 'oracle-gsd.json'
        run('oracle', [python, ROOT / 'tools/spacy/ggml_ab_oracle.py', '--wheel', wheel, '--corpus', corpus,
                       '--output', oracle])
        f16 = scratch / 'f16.gguf'
        run('quantize-f16', [python, ROOT / 'models/quantize-spacy-gguf.py', '--input', bundle / 'model.gguf',
                             '--output', f16, '--type', 'f16'])
        report['f16_sha256'] = common.digest(f16)

        configs = [
            ('scalar-f32', 'default', {'CRISPEMBED_SPACY_GGML': '0'}, None),
            ('ggml-sgemm-f32', 'default', {}, None),
            ('ggml-sgemm-f32-t4', 'default', {'CRISPEMBED_SPACY_THREADS': '4'}, None),
            ('ggml-nosgemm-f32', 'sgemm-off', {}, None),
            ('scalar-f16', 'default', {'CRISPEMBED_SPACY_GGML': '0', 'CRISPEMBED_SPACY_QUANT': '1'}, f16),
            ('ggml-sgemm-f16', 'default', {'CRISPEMBED_SPACY_QUANT': '1'}, f16),
            ('ggml-sgemm-f16-upcast', 'default', {'CRISPEMBED_SPACY_QUANT': '1',
                                                  'CRISPEMBED_SPACY_GGML_UPCAST': '1'}, f16),
        ]
        ab = {}
        for label, build, env_extra, model in configs:
            target = out / f'ab-{label}.json'
            args = [python, ROOT / 'tests/spacy_ggml_ab.py', '--library', builds[build], '--bundle', bundle,
                    '--oracle', oracle, '--label', label, '--report', target, '--repeats', '3']
            for key, value in env_extra.items():
                args += ['--env', f'{key}={value}']
            if model:
                args += ['--model', model]
            run('ab-' + label, args, env=base_env, check=False)
            data = json.loads(target.read_text()) if target.is_file() else {'admitted': False}
            ab[label] = {k: data.get(k) for k in ('admitted', 'cases', 'exact_cases', 'tokens', 'field_agreement')}
            ab[label]['timing'] = data.get('timing')
            ab[label]['differences'] = data.get('differences', [])[:20]
        report['gsd_ab'] = ab
        run('replay-f16', [python, ROOT / 'tools/spacy/quant_oracle_replay.py', '--wheel', wheel, '--gguf', f16,
                           '--oracle', oracle, '--output', out / 'replay-f16.json',
                           '--native-report', out / 'ab-scalar-f16.json'])
        report['original_spacy_with_f16_weights'] = json.loads((out / 'replay-f16.json').read_text())
        report['corpus'] = {'sentences': len(corpus_texts),
                            'oracle_sha256': hashlib.sha256(oracle.read_bytes()).hexdigest()}
        exact = ab['ggml-sgemm-f32']['exact_cases'] == ab['ggml-sgemm-f32']['cases'] == len(corpus_texts)
        report['default_candidate_exact'] = exact
        report['passed'] = exact and all(item['passed'] for mode in gates.values() for item in mode.values())
        report['stage'] = 'complete'
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()
        (out / 'summary.md').write_text('# spaCy ggml+SGEMM default A/B (Linux x86-64/AVX2)\n\n```json\n' + json.dumps(
            {k: report.get(k) for k in ('stage', 'passed', 'default_candidate_exact', 'failure')}, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()
