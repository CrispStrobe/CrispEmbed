#!/usr/bin/env python3
"""Hosted Linux A/B for version-2 spaCy pipelines (English/French sm).

Converts the pinned official wheel with models/convert-spacy-pipeline-to-gguf.py
in the hash-locked reference environment, builds tools/spacy, then compares
spacy_smoke (C ABI only) with original spaCy on the raw text of a UD r2.14 test
set: exact tokens, tag/pos/morph/lemma/dep/head/sentence starts and entities.
Also runs tests/test_spacy_v2_controls.py. Usage (heavy-cpu.yml):

    args: --model en_core_web_sm
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import spacy_w7_publish as common

ROOT = common.ROOT
MODELS = {
    'en_core_web_sm': ('UD_English-EWT', 'en_ewt-ud-test.conllu',
                       '5e0044aa9b786bfb81dbd5687400583a61635a65839e747b59b2ec907c8ab662', 1765008),
    'fr_core_news_sm': ('UD_French-GSD', 'fr_gsd-ud-test.conllu',
                        '158386a42ece1fca4fbcfa905b1a584d3e375415f4e0ffa7b3de20b1ea2d768a', 651369),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-v2'
    scratch.mkdir(parents=True)
    report = {'schema': 'crispembed.spacy.v2_ab.v1', 'model': args.model, 'source': os.environ.get('GITHUB_SHA'),
              'run': os.environ.get('GITHUB_RUN_ID'), 'passed': False}

    def save():
        (out / 'v2-ab.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

    def run(label, command, **kwargs):
        with (out / f'{label}.log').open('w') as log:
            return subprocess.run([str(c) for c in command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  check=True, **kwargs)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('convert_v2', ROOT / 'models/convert-spacy-pipeline-to-gguf.py')
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        wheel_name = f'{args.model}-3.8.0-py3-none-any.whl'
        sha, size = converter.WHEELS[wheel_name]
        wheel = scratch / wheel_name
        report['wheel'] = common.download(
            f'https://github.com/explosion/spacy-models/releases/download/{args.model}-3.8.0/{wheel_name}',
            wheel, sha, size)
        venv = scratch / 'reference'
        run('venv', [sys.executable, '-m', 'venv', venv])
        python = venv / 'bin/python'
        run('reference-install', [python, '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes',
                                  '-r', ROOT / 'tools/spacy/requirements-reference.txt'])
        bundle = scratch / 'bundle'
        run('convert', [python, ROOT / 'models/convert-spacy-pipeline-to-gguf.py', '--wheel', wheel,
                        '--output', bundle])
        manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
        report['bundle_files'] = manifest['files']
        run('pip-model', [python, '-m', 'pip', 'install', '--no-deps', wheel])
        repo, name, tb_sha, tb_size = MODELS[args.model]
        treebank = scratch / name
        common.download(f'https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.14/{name}',
                        treebank, tb_sha, tb_size)
        texts = [line[9:].strip() for line in treebank.read_text(encoding='utf-8').splitlines()
                 if line.startswith('# text = ')]
        corpus = scratch / 'corpus.json'
        corpus.write_text(json.dumps({'texts': texts}, ensure_ascii=False), encoding='utf-8')
        lines = scratch / 'texts.txt'
        lines.write_text('\n'.join(texts) + '\n', encoding='utf-8', newline='\n')
        run('oracle', [python, ROOT / 'tools/spacy/pipeline_oracle.py', '--model', args.model, '--corpus', corpus,
                       '--oracle', scratch / 'oracle.json', '--ner-oracle', scratch / 'ner-oracle.json'])
        run('apt', ['sudo', 'apt-get', 'install', '-y', '-qq', 'libpcre2-dev'])
        build = scratch / 'build'
        run('configure', ['cmake', '-S', ROOT / 'tools/spacy', '-B', build, '-DCMAKE_BUILD_TYPE=Release'])
        run('build', ['cmake', '--build', build, '--parallel', '4'])
        run('ctest', ['ctest', '--test-dir', build, '--output-on-failure'])
        library = build / 'libcrispembed-spacy.so'
        smoke_out = out / 'smoke.jsonl'
        with smoke_out.open('wb') as stdout, (out / 'smoke.log').open('w') as log:
            code = subprocess.run([str(build / 'spacy-smoke'), str(bundle / 'model.gguf'),
                                   manifest['files']['model.gguf']['sha256'], str(bundle / 'resources.json'),
                                   manifest['files']['resources.json']['sha256'], str(lines)],
                                  stdout=stdout, stderr=log).returncode
        assert code == 0, f'spacy_smoke exited {code}'
        subprocess.run([sys.executable, ROOT / 'tests/spacy_smoke_compare.py', '--smoke', smoke_out,
                        '--oracle', scratch / 'oracle.json', '--ner-oracle', scratch / 'ner-oracle.json',
                        '--report', out / 'compare.json'], check=False)
        report['comparison'] = json.loads((out / 'compare.json').read_text(encoding='utf-8'))
        controls = subprocess.run([sys.executable, ROOT / 'tests/test_spacy_v2_controls.py', '--library', library,
                                   '--bundle', bundle], capture_output=True, text=True)
        report['controls'] = {'returncode': controls.returncode, 'output': controls.stdout[-2000:]}
        comparison = report['comparison']
        report['passed'] = (comparison['exact_cases'] == comparison['cases'] and not comparison['differences']
                            and controls.returncode == 0)
        assert report['passed'], 'v2 A/B failed'
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {str(exc)[:1500]}'
        raise
    finally:
        save()
        (out / 'summary.md').write_text('# spaCy v2 A/B\n\n```json\n' + json.dumps(
            {k: report.get(k) for k in ('model', 'passed', 'failure')}, indent=2) + '\n```\n'
            + json.dumps({k: v for k, v in (report.get('comparison') or {}).items() if k != 'differences'}) + '\n',
            encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
