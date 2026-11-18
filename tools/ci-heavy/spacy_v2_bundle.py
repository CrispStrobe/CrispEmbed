#!/usr/bin/env python3
"""Hosted distribution of native spaCy v2 language bundles (en_core_web_sm, fr_core_news_sm).

One Hugging Face model repository per language holds only the model bundle;
the language-independent native library is the capsule already published in
the German repository (tools/spacy/language_bundles.json names both). Steps
(.github/workflows/spacy-language-bundle.yml runs them in order):

  build    Linux x86-64, CPython 3.12.14. Authenticates the pinned official
           wheel, converts it with models/convert-spacy-pipeline-to-gguf.py in
           the hash-locked reference environment, requires model.gguf,
           resources.json and both notices to equal the tracked pins (notices
           also byte-equal to the wheel), and writes an original-spaCy oracle
           for the raw UD test-set texts. With --publish (main only): one Hub
           commit of bundle/*, the exact immutable tree, then a fresh
           empty-cache HTTP download of every file with digest and length.
  consume  Any desktop platform. Takes the bundle (published revision, or the
           dry-run payload), downloads the newest v2-capable library capsule of
           this platform from the library repository at its immutable revision,
           verifies and installs it, then runs the v2 admission controls and a
           ctypes-only analysis of every oracle text against the installed
           library bytes, compared token by token and entity by entity.
  card     Renders bundle-catalog.json and README.md from the build receipt and
           every consume receipt; with --publish, one Hub commit and fresh
           readback. A failed consumer is listed as failed, never omitted.

Upload alone is never success.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform as host
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

import spacy_w7_publish as common

ROOT = common.ROOT
CATALOG = json.loads((ROOT / 'tools/spacy/language_bundles.json').read_text(encoding='utf-8'))
LIBRARY_REPOSITORY = CATALOG['library_repository']
BUNDLE_FILES = ('model.gguf', 'resources.json', 'manifest.json', 'LICENSE', 'LICENSES_SOURCES')
PLATFORMS = {
    'linux-x86_64': ('Linux', 'x86_64'),
    'linux-arm64': ('Linux', 'aarch64'),
    'macos-arm64': ('Darwin', 'arm64'),
    'windows-x86_64': ('Windows', 'AMD64'),
}
HUB_VERSION = 'huggingface_hub==0.36.0'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hub_api():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', HUB_VERSION], check=True)
    from huggingface_hub import HfApi
    token = os.environ.get('HF_PUBLISH_TOKEN')
    assert token, 'Hosted Hub write credential unavailable'
    return HfApi(token=token)


def publish_commit(api, repository, uploads, message, scratch):
    """One Hub commit, the exact expected immutable tree, then fresh-cache readback."""
    from huggingface_hub import CommitOperationAdd
    pins = {path: {'sha256': common.digest(local), 'bytes': local.stat().st_size} for path, local in uploads.items()}
    before = set(api.list_repo_files(repository))
    commit = api.create_commit(repo_id=repository, repo_type='model', commit_message=message,
                               operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
                                           for path, local in uploads.items()])
    revision = commit.oid
    assert re.fullmatch('[0-9a-f]{40}', revision), revision
    tree = set(api.list_repo_files(repository, revision=revision))
    assert tree == before | set(uploads), f'Unexpected immutable hosted tree: {sorted(tree)}'
    fresh = scratch / f'fresh-{revision[:12]}'
    fresh.mkdir()
    assert not list(fresh.iterdir())
    readback = {}
    for path, pin in pins.items():
        local = fresh / path
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{repository}/resolve/{revision}/' + urllib.parse.quote(path, safe='/')
        readback[path] = common.download(url, local, pin['sha256'], pin['bytes'])
    return {'repository': repository, 'revision': revision, 'tree': sorted(tree), 'pins': pins,
            'fresh_readback': readback}, fresh


def published_bundle(api, repository, pins, scratch):
    """An existing commit whose bundle/* equals these pins, read back fresh; else None.

    Prefers the commit that introduced the files (consumers pin the bundle
    commit, not a later card commit); falls back to the current head.
    """
    head = api.model_info(repository).sha
    if not set(pins) <= set(api.list_repo_files(repository, revision=head)):
        return None
    candidates = []
    for commit in api.list_repo_commits(repository):  # newest first
        if not set(pins) <= set(api.list_repo_files(repository, revision=commit.commit_id)):
            break
        candidates.append(commit.commit_id)
    for revision in ([candidates[-1]] if candidates else []) + [head]:
        fresh = scratch / f'fresh-existing-{revision[:12]}'
        fresh.mkdir(exist_ok=True)
        readback = {}
        try:
            for path, pin in pins.items():
                local = fresh / path
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    local.unlink()
                url = f'https://huggingface.co/{repository}/resolve/{revision}/' + urllib.parse.quote(path, safe='/')
                readback[path] = common.download(url, local, pin['sha256'], pin['bytes'])
        except AssertionError:
            continue
        tree = sorted(api.list_repo_files(repository, revision=revision))
        return {'repository': repository, 'revision': revision, 'head_at_check': head,
                'reused_existing_commit': True, 'tree': tree, 'pins': pins, 'fresh_readback': readback}, fresh
    return None


def build(args, entry, out, scratch, report, run):
    assert sys.platform == 'linux' and host.machine() == 'x86_64', 'Linux x86-64 only'
    assert host.python_version() == '3.12.14', 'Canonical conversion needs CPython 3.12.14'
    wheel_pin = entry['wheel']
    wheel = scratch / wheel_pin['name']
    report['wheel'] = common.download(wheel_pin['url'], wheel, wheel_pin['sha256'], wheel_pin['bytes'])
    venv = scratch / 'reference'
    run('venv', [sys.executable, '-m', 'venv', venv])
    python = venv / 'bin/python'
    run('reference-install', [python, '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes',
                              '-r', ROOT / 'tools/spacy/requirements-reference.txt'])
    converted = scratch / 'converted'
    run('convert', [python, ROOT / 'models/convert-spacy-pipeline-to-gguf.py', '--wheel', wheel,
                    '--output', converted])
    assert sorted(p.name for p in converted.iterdir()) == sorted(BUNDLE_FILES), sorted(converted.iterdir())
    for name, pin in entry['expected_files'].items():
        got = {'sha256': common.digest(converted / name), 'bytes': (converted / name).stat().st_size}
        assert got == pin, f'{name} differs from the tracked pin: {got}'
    package = args.model
    with zipfile.ZipFile(wheel) as archive:
        for notice in ('LICENSE', 'LICENSES_SOURCES'):
            assert archive.read(f'{package}/{package}-{entry["version"]}/{notice}') == \
                (converted / notice).read_bytes(), f'{notice} is not verbatim'
    manifest = json.loads((converted / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['architecture'] == 'spacy_pipeline_v2' and manifest['format_version'] == 2
    assert manifest['component_order'] == entry['components'], manifest['component_order']
    for name, pin in entry['expected_files'].items():
        assert manifest['files'][name] == {'sha256': pin['sha256'], 'size_bytes': pin['bytes']}, name
    payload = out / 'payload'
    (payload / 'bundle').mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copyfile(converted / name, payload / 'bundle' / name)
    report['bundle_files'] = {f'bundle/{name}': {'sha256': common.digest(payload / 'bundle' / name),
                                                 'bytes': (payload / 'bundle' / name).stat().st_size}
                              for name in BUNDLE_FILES}
    report['converter_sha256'] = manifest['converter_sha256']
    # Original-spaCy oracle on the raw test-set text (input only; not redistributed).
    run('pip-model', [python, '-m', 'pip', 'install', '--no-deps', wheel])
    treebank = entry['treebank']
    conllu = scratch / treebank['file']
    report['treebank'] = common.download(
        f"https://raw.githubusercontent.com/UniversalDependencies/{treebank['repository']}/r2.14/{treebank['file']}",
        conllu, treebank['sha256'], treebank['bytes'])
    texts = [line[9:].strip() for line in conllu.read_text(encoding='utf-8').splitlines()
             if line.startswith('# text = ')]
    assert texts and not any(not t or '\n' in t or '\r' in t for t in texts)
    oracle = out / 'oracle'
    oracle.mkdir()
    corpus = scratch / 'corpus.json'
    corpus.write_text(json.dumps({'texts': texts}, ensure_ascii=False), encoding='utf-8')
    (oracle / 'texts.txt').write_text('\n'.join(texts) + '\n', encoding='utf-8', newline='\n')
    run('oracle', [python, ROOT / 'tools/spacy/pipeline_oracle.py', '--model', args.model, '--corpus', corpus,
                   '--oracle', oracle / 'oracle.json', '--ner-oracle', oracle / 'ner-oracle.json'])
    report['oracle'] = {name: common.digest(oracle / name) for name in ('texts.txt', 'oracle.json', 'ner-oracle.json')}
    report['oracle_texts'] = len(texts)
    if not args.publish:
        report['dry_run'] = True
        return
    api = hub_api()
    report['publisher'] = api.whoami()['name']
    api.create_repo(entry['repository'], repo_type='model', private=False, exist_ok=True)
    reused = published_bundle(api, entry['repository'], report['bundle_files'], scratch)
    if reused:
        # Re-runs (e.g. to add consumers once more capsules exist) keep the
        # already published, byte-identical bundle commit instead of an
        # empty commit; it is still read back fresh and verified.
        report['publication'], fresh = reused
    else:
        report['publication'], fresh = publish_commit(
            api, entry['repository'], {path: payload / path for path in report['bundle_files']},
            f"Publish {args.model} {entry['version']} native spaCy v2 bundle from CrispEmbed {report['source'][:12]}",
            scratch)
    fresh_manifest = json.loads((fresh / 'bundle/manifest.json').read_text(encoding='utf-8'))
    for name, pin in entry['expected_files'].items():
        assert fresh_manifest['files'][name] == {'sha256': pin['sha256'], 'size_bytes': pin['bytes']}
    report['revision'] = report['publication']['revision']


def v2_capsule(platform_name, scratch):
    """Newest published capsule of this platform whose source contains the v2 runtime.

    The Hub catalog at main is only an index; the capsule is fetched from its
    immutable revision and authenticated by the archive digest it lists.
    """
    url = f'https://huggingface.co/{LIBRARY_REPOSITORY}/resolve/main/hub-catalog.json'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}),
                                timeout=60) as response:
        catalog = json.load(response)
    candidates = []
    for package in catalog['packages']:
        match = re.fullmatch(rf'packages/{re.escape(platform_name)}/([0-9a-f]{{40}})/[^/]+', package.get('path') or '')
        if package['platform'] != platform_name or not match:
            continue
        commit = match.group(1)
        has_v2 = subprocess.run(['git', 'cat-file', '-e', f'{commit}:src/spacy_v2.cpp'], cwd=ROOT,
                                capture_output=True).returncode == 0
        if has_v2:
            when = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', commit], cwd=ROOT, text=True))
            candidates.append((when, commit, package))
    assert candidates, f'No published v2-capable {platform_name} capsule in {LIBRARY_REPOSITORY}'
    _, commit, package = max(candidates, key=lambda c: c[0])
    archive = scratch / package['path'].rsplit('/', 1)[-1]
    download = common.download(f"https://huggingface.co/{LIBRARY_REPOSITORY}/resolve/{package['revision']}/"
                               + package['path'], archive, package['sha256'], package['bytes'])
    return package, commit, archive, download


def consume(args, entry, out, scratch, report, run):
    system, machine = PLATFORMS[args.platform]
    assert host.system() == system and host.machine() == machine, \
        f'Real {args.platform} runner required (got {host.system()}/{host.machine()})'
    receipt = json.loads((args.build_dir / 'build.json').read_text(encoding='utf-8'))
    assert receipt['passed'] and receipt['model'] == args.model
    bundle = scratch / 'bundle'
    bundle.mkdir()
    if receipt.get('revision'):
        report['bundle_revision'] = receipt['revision']
        report['bundle_downloads'] = {}
        for path, pin in receipt['bundle_files'].items():
            url = f"https://huggingface.co/{entry['repository']}/resolve/{receipt['revision']}/{path}"
            report['bundle_downloads'][path] = common.download(url, bundle / path.split('/', 1)[1], pin['sha256'],
                                                               pin['bytes'])
    else:
        report['bundle_revision'] = None
        for path, pin in receipt['bundle_files'].items():
            local = args.build_dir / 'payload' / path
            assert common.digest(local) == pin['sha256'] and local.stat().st_size == pin['bytes']
            shutil.copyfile(local, bundle / path.split('/', 1)[1])
    for name, pin in entry['expected_files'].items():
        assert common.digest(bundle / name) == pin['sha256'], name
    pd = _load('spacy_platform_distribution', ROOT / 'tools/spacy/platform_distribution.py')
    package, commit, archive, download = v2_capsule(args.platform, scratch)
    report['library'] = {'capsule_path': package['path'], 'capsule_revision': package['revision'],
                         'capsule_sha256': package['sha256'], 'capsule_bytes': package['bytes'],
                         'library_sha256': package['library_sha256'], 'source_commit': commit,
                         'download': download}
    report['library']['archive_verification'] = pd.verify_archive(archive, package['sha256'])
    cache = scratch / 'capsule-cache'
    report['library']['install'] = pd.install(archive, package['sha256'], cache)
    target = pd.PLATFORMS[args.platform]
    installed = cache / package['sha256']
    library = installed / 'native' / package['library_sha256'] / target['library']
    assert library.is_file() and common.digest(library) == package['library_sha256'], library
    german = installed / 'bundle'
    controls = subprocess.run([sys.executable, '-X', 'faulthandler', ROOT / 'tests/test_spacy_v2_controls.py',
                               '--library', library, '--bundle', bundle, '--german-bundle', german],
                              capture_output=True, text=True, cwd=ROOT)
    (out / 'controls.log').write_text(controls.stdout + controls.stderr, encoding='utf-8')
    try:
        control_results = json.loads(controls.stdout)
    except ValueError:
        control_results = None
    report['controls'] = {'returncode': controls.returncode, 'results': control_results,
                          'passed': controls.returncode == 0 and bool(control_results) and all(control_results.values())}
    save_smoke = out / 'smoke.jsonl'
    smoke = subprocess.run([sys.executable, '-X', 'faulthandler', ROOT / 'tests/spacy_ctypes_smoke.py',
                            '--library', library, '--bundle', bundle, '--texts', args.build_dir / 'oracle/texts.txt',
                            '--output', save_smoke], capture_output=True, text=True, cwd=ROOT)
    (out / 'smoke.log').write_text(smoke.stdout + smoke.stderr, encoding='utf-8')
    assert smoke.returncode == 0, f'ctypes smoke exited {smoke.returncode}: {smoke.stderr[-800:]}'
    subprocess.run([sys.executable, ROOT / 'tests/spacy_smoke_compare.py', '--smoke', save_smoke,
                    '--oracle', args.build_dir / 'oracle/oracle.json',
                    '--ner-oracle', args.build_dir / 'oracle/ner-oracle.json', '--report', out / 'compare.json'],
                   check=False, cwd=ROOT)
    comparison = json.loads((out / 'compare.json').read_text(encoding='utf-8'))
    report['comparison'] = {k: v for k, v in comparison.items() if k != 'differences'}
    report['comparison']['first_differences'] = comparison['differences'][:5]
    exact = comparison['exact_cases'] == comparison['cases'] and not comparison['differences']
    report['comparison']['passed'] = exact
    assert report['controls']['passed'], f'Admission controls failed: {control_results}'
    assert exact, 'Native analysis differs from original spaCy'


def card(args, entry, out, scratch, report):
    receipt = json.loads((args.build_dir / 'build.json').read_text(encoding='utf-8'))
    assert receipt['passed'] and receipt['model'] == args.model
    assert receipt.get('revision') or not args.publish, 'Publishing a card needs a published bundle'
    libraries = []
    for path in sorted(args.consume_dir.glob('*/consume.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('model') != args.model:
            continue
        if 'bundle_revision' in item:
            assert item['bundle_revision'] == receipt.get('revision'), f'{path}: consumer used another bundle'
        library = item.get('library') or {}
        libraries.append({'platform': item['platform'], 'library_sha256': library.get('library_sha256'),
                          'capsule_path': library.get('capsule_path'),
                          'capsule_revision': library.get('capsule_revision'),
                          'source_commit': library.get('source_commit'), 'run': item['run'],
                          'controls_passed': bool((item.get('controls') or {}).get('passed')),
                          'comparison': item.get('comparison'), 'passed': bool(item['passed']),
                          'failure': item.get('failure')})
    report['libraries'] = libraries
    model = {key: entry[key] for key in ('language', 'language_name', 'version', 'wheel', 'license', 'components',
                                         'entity_labels', 'treebank', 'qualification')}
    model['name'] = args.model
    catalog = {
        'schema': 'crispembed.spacy.language_bundle.v1',
        'repository': entry['repository'],
        'library_repository': LIBRARY_REPOSITORY,
        'model': model,
        'bundle': {'revision': receipt.get('revision'), 'source_commit': receipt['source'], 'run': receipt['run'],
                   'files': receipt['bundle_files']},
        'libraries': libraries,
        'not_qualified': [
            'Grammar checking or any proofing claim for this language.',
            'Library builds not listed above, including every capsule built before CrispEmbed PR #244 '
            '(they reject spacy_pipeline_v2 bundles).',
            'iOS and Android SDK runs with this bundle; quantized variants; texts over 1 MiB or 2,048 tokens per call.',
        ],
    }
    payload = out / 'card-payload'
    payload.mkdir()
    (payload / 'bundle-catalog.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n',
                                                 encoding='utf-8', newline='\n')
    renderer = _load('spacy_bundle_card', ROOT / 'tools/spacy/bundle_card.py')
    (payload / 'README.md').write_text(renderer.render(catalog), encoding='utf-8', newline='\n')
    report['catalog'] = catalog
    if not args.publish:
        report['dry_run'] = True
        return
    api = hub_api()
    report['publication'], _ = publish_commit(
        api, entry['repository'], {name: payload / name for name in ('bundle-catalog.json', 'README.md')},
        'Model card and bundle catalog (known native libraries)', scratch)
    report['revision'] = report['publication']['revision']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'consume', 'card'])
    parser.add_argument('--model', choices=sorted(CATALOG['bundles']), required=True)
    parser.add_argument('--platform', choices=sorted(PLATFORMS))
    parser.add_argument('--build-dir', type=Path, help='consume/card: the build step output')
    parser.add_argument('--consume-dir', type=Path, help='card: directory of consume outputs (one per platform)')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    if args.publish:
        assert args.command != 'consume', 'consume never writes to the Hub'
        assert os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Hub writes only from main'
    entry = CATALOG['bundles'][args.model]
    out = Path(os.environ['HEAVY_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH']) / f'spacy-v2-{args.command}'
    scratch.mkdir(parents=True)
    report = {'schema': f'crispembed.spacy.v2_bundle.{args.command}.v1', 'model': args.model, 'passed': False,
              'platform': args.platform, 'publish': args.publish,
              'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'ref': os.environ.get('GITHUB_REF'), 'run': os.environ.get('GITHUB_RUN_ID'),
              'repository': entry['repository'], 'grammar_qualified': False}
    report_path = out / f'{args.command}.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
                               newline='\n')

    def run(label, command):
        with (out / f'{label}.log').open('w') as log:
            subprocess.run([str(c) for c in command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        if args.command == 'build':
            build(args, entry, out, scratch, report, run)
        elif args.command == 'consume':
            assert args.platform and args.build_dir
            consume(args, entry, out, scratch, report, run)
        else:
            assert args.build_dir and args.consume_dir
            card(args, entry, out, scratch, report)
        report['passed'] = True
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {str(exc)[:1500]}'
        raise
    finally:
        save()
        summary = {k: report.get(k) for k in ('model', 'platform', 'passed', 'revision', 'bundle_revision',
                                              'failure')}
        if report.get('comparison'):
            summary['comparison'] = {k: v for k, v in report['comparison'].items() if k != 'first_differences'}
        if report.get('library'):
            summary['library_sha256'] = report['library']['library_sha256']
        (out / 'summary.md').write_text(f'# spaCy v2 bundle {args.command}\n\n```json\n'
                                        + json.dumps(summary, indent=2, ensure_ascii=False) + '\n```\n',
                                        encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
