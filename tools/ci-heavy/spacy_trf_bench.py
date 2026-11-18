#!/usr/bin/env python3
"""Hosted benchmark + exactness A/B for spaCy transformer pipelines.

On a quiet hosted runner: converts fr_dep_news_trf (pinned wheel), builds
tools/spacy with CRISPEMBED_SPACY_TRF and CRISPEMBED_SPACY_BLAS (Accelerate on
macOS, OpenBLAS on Linux), produces the original-spaCy oracle for UD French GSD
test (416 sentences, and the same text as 7 long documents), then for every
native configuration (reference vs CRISPEMBED_SPACY_TRF_FUSED, BLAS off/on,
1 and 4 threads) records decoded exactness via the C ABI and wall time, next to
PyTorch timings for the original pipeline on the same machine.

Developer tooling; the reference environment here is not hash-locked (torch and
spacy-curated-transformers are recorded by version in the report).
"""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import spacy_w7_publish as common

ROOT = common.ROOT
WHEEL = 'fr_dep_news_trf-3.8.0-py3-none-any.whl'
WHEEL_SHA = 'df243e539bbc365390744a411aa574fb03eb2ee2f948cf6149ad722c59ab419b'
TREEBANK = ('UD_French-GSD', 'fr_gsd-ud-test.conllu',
            '158386a42ece1fca4fbcfa905b1a584d3e375415f4e0ffa7b3de20b1ea2d768a', 651369)


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-trf-bench'
    scratch.mkdir(parents=True)
    report = {'schema': 'crispembed.spacy.trf_bench.v1', 'source': os.environ.get('GITHUB_SHA'),
              'run': os.environ.get('GITHUB_RUN_ID'), 'platform': platform.platform(),
              'cpu_count': os.cpu_count(), 'native': {}, 'pytorch': {}}

    def save():
        (out / 'trf-bench.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

    def run(label, command, **kwargs):
        with (out / f'{label}.log').open('w') as log:
            return subprocess.run([str(c) for c in command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  check=True, **kwargs)

    try:
        venv = scratch / 'reference'
        run('venv', [sys.executable, '-m', 'venv', venv])
        python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        run('pip', [python, '-m', 'pip', 'install', '-q', 'spacy==3.8.18', 'spacy-curated-transformers>=0.2.2,<1.0.0',
                    'torch', 'gguf==0.19.0', 'numpy'])
        report['reference_versions'] = subprocess.check_output(
            [str(python), '-c', 'import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in '
             '("spacy", "spacy-curated-transformers", "curated-transformers", "curated-tokenizers", "torch")}))'],
            text=True).strip()
        wheel = scratch / WHEEL
        common.download(f'https://github.com/explosion/spacy-models/releases/download/fr_dep_news_trf-3.8.0/{WHEEL}',
                        wheel, WHEEL_SHA, 397740844)
        run('pip-model', [python, '-m', 'pip', 'install', '-q', '--no-deps', wheel])
        bundle = scratch / 'bundle'
        run('convert', [python, ROOT / 'models/convert-spacy-pipeline-to-gguf.py', '--wheel', wheel, '--output', bundle])
        manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
        report['model_gguf'] = manifest['files']['model.gguf']
        repo, name, sha, size = TREEBANK
        treebank = scratch / name
        common.download(f'https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.14/{name}',
                        treebank, sha, size)
        sentences = [line[9:].strip() for line in treebank.read_text(encoding='utf-8').splitlines()
                     if line.startswith('# text = ')]
        corpora = {'sentences': sentences, 'documents': [' '.join(sentences[i:i + 60])
                                                          for i in range(0, len(sentences), 60)]}
        for label, texts in corpora.items():
            corpus = scratch / f'{label}.json'
            corpus.write_text(json.dumps({'texts': texts}, ensure_ascii=False), encoding='utf-8')
            (scratch / f'{label}.txt').write_text('\n'.join(texts) + '\n', encoding='utf-8', newline='\n')
            run(f'oracle-{label}', [python, ROOT / 'tools/spacy/pipeline_oracle.py', '--model', 'fr_dep_news_trf',
                                    '--corpus', corpus, '--oracle', scratch / f'{label}-oracle.json',
                                    '--ner-oracle', scratch / f'{label}-ner.json'])
            timings = {}
            for threads in (1, 4):
                code = ('import json,sys,time,spacy,torch\n'
                        f'torch.set_num_threads({threads})\n'
                        'nlp=spacy.load("fr_dep_news_trf")\n'
                        f'texts=json.load(open(r"{corpus}"))["texts"]\n'
                        'list(nlp.pipe(texts[:2]))\n'
                        't=time.perf_counter(); docs=list(nlp.pipe(texts)); dt=time.perf_counter()-t\n'
                        'print(json.dumps({"seconds": dt, "tokens": sum(len(d) for d in docs)}))\n')
                result = json.loads(subprocess.check_output([str(python), '-c', code], text=True,
                                                            stderr=subprocess.DEVNULL).strip().splitlines()[-1])
                timings[f'threads_{threads}'] = result
            report['pytorch'][label] = timings
            save()
        if sys.platform.startswith('linux'):
            run('apt', ['sudo', 'apt-get', 'install', '-y', '-qq', 'libpcre2-dev', 'libopenblas-dev', 'pkg-config'])
        build = scratch / 'build'
        run('configure', ['cmake', '-S', ROOT / 'tools/spacy', '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                          '-DCRISPEMBED_SPACY_TRF=ON', '-DCRISPEMBED_SPACY_BLAS=ON', '-Wno-dev'])
        run('build', ['cmake', '--build', build, '--parallel', '4'])
        smoke = build / 'spacy-smoke'
        for label in corpora:
            for fused in ('0', '1'):
                for blas in ('0', '1'):
                    for threads in ('1', '4'):
                        key = f'{label}/fused={fused}/blas={blas}/threads={threads}'
                        env = {**os.environ, 'CRISPEMBED_SPACY_TRF_FUSED': fused, 'CRISPEMBED_SPACY_TRF_BLAS': blas,
                               'CRISPEMBED_SPACY_THREADS': threads}
                        target = out / f'smoke-{label}-{fused}{blas}{threads}.jsonl'
                        started = time.perf_counter()
                        with target.open('wb') as stdout:
                            code = subprocess.run([str(smoke), str(bundle / 'model.gguf'),
                                                   manifest['files']['model.gguf']['sha256'],
                                                   str(bundle / 'resources.json'),
                                                   manifest['files']['resources.json']['sha256'],
                                                   str(scratch / f'{label}.txt')],
                                                  stdout=stdout, stderr=subprocess.DEVNULL, env=env).returncode
                        seconds = time.perf_counter() - started
                        compare = out / f'compare-{label}-{fused}{blas}{threads}.json'
                        subprocess.run([sys.executable, ROOT / 'tests/spacy_smoke_compare.py', '--smoke', target,
                                        '--oracle', scratch / f'{label}-oracle.json', '--ner-oracle',
                                        scratch / f'{label}-ner.json', '--report', compare],
                                       stdout=subprocess.DEVNULL, check=False)
                        data = json.loads(compare.read_text(encoding='utf-8')) if compare.exists() else {}
                        report['native'][key] = {'returncode': code, 'seconds_including_load': round(seconds, 2),
                                                 'exact_cases': data.get('exact_cases'), 'cases': data.get('cases'),
                                                 'tokens': data.get('tokens')}
                        target.unlink()
                        save()
        report['all_exact'] = all(v['exact_cases'] == v['cases'] for v in report['native'].values())
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {str(exc)[:1500]}'
        raise
    finally:
        save()
        lines = ['# spaCy trf bench', '', f"{report.get('platform')} cpus={report.get('cpu_count')}", '']
        lines += [f'- pytorch {k}: {v}' for k, v in report.get('pytorch', {}).items()]
        lines += [f'- {k}: {v}' for k, v in report.get('native', {}).items()]
        (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
