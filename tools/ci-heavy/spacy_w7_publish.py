#!/usr/bin/env python3
"""Publish only authenticated frozen W7 inputs; fresh HTTP readback precedes success.

Runs on hosted Linux. Restores model data through the exact developer converter;
never builds/loads a native library or executes either consumed linguistic panel.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'cstr/crispembed-spacy-de-core-news-sm-3.8.0'
PACKAGE = 'spacy-de_core_news_sm-3.8.0-linux-x86_64-avx2-f32.tar.gz'
PACKAGE_SHA = '4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01'
WHEEL = 'de_core_news_sm-3.8.0-py3-none-any.whl'
WHEEL_SHA = 'fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_distribution():
    spec = importlib.util.spec_from_file_location('spacy_distribution', ROOT / 'tools/spacy/distribution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download(url, destination, expected_sha, expected_bytes):
    request = urllib.request.Request(url, headers={'Cache-Control': 'no-cache', 'User-Agent': 'CrispEmbed-W7-readback/1'})
    with urllib.request.urlopen(request, timeout=180) as response, Path(destination).open('xb') as output:
        assert response.status == 200
        shutil.copyfileobj(response, output)
        commit = response.headers.get('X-Repo-Commit')
    assert Path(destination).stat().st_size == expected_bytes, 'Downloaded length differs'
    assert digest(destination) == expected_sha, 'Downloaded digest differs'
    return {'url': url, 'sha256': expected_sha, 'bytes': expected_bytes, 'http_status': 200,
            'response_repo_commit': commit, 'fresh_empty_cache': True}


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true', 'Hosted execution only'
    assert os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert sys.platform == 'linux' and platform.python_version() == '3.12.14'
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-w7-publish'
    scratch.mkdir()
    output = Path(os.environ['HEAVY_OUT'])
    report = {'schema': 'crispembed.spacy.immutable_publication.v1', 'passed': False,
              'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'repository': REPOSITORY, 'native_rebuilt_or_loaded': False,
              'native_acceptance_or_W6_replayed': False, 'scope': 'Frozen Linux x86-64/AVX2 CPU/F32 package only'}

    def save():
        (output / 'publication.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        dist = load_distribution()
        payload = scratch / 'payload'
        (payload / 'bundle').mkdir(parents=True)
        wheel = scratch / WHEEL
        report['source_wheel'] = download('https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/' + WHEEL,
                                          wheel, WHEEL_SHA, 14639490)
        reference = scratch / 'reference-environment'
        subprocess.run([sys.executable, '-m', 'venv', str(reference)], check=True)
        python = reference / 'bin/python'
        with (output / 'reference-install.log').open('w') as log:
            subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes',
                            '-r', str(ROOT / 'tools/spacy/requirements-reference.txt')], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        # Converter refuses existing destinations; the output becomes the payload's bundle after verification.
        bundle = scratch / 'converted-bundle'
        with (output / 'conversion.log').open('w') as log:
            subprocess.run([str(python), str(ROOT / 'models/convert-spacy-to-gguf.py'),
                            '--wheel', str(wheel), '--output', str(bundle)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        for name, pin in dist.BUNDLE_FILES.items():
            assert digest(bundle / name) == pin['sha256'] and (bundle / name).stat().st_size == pin['size_bytes']
            shutil.copyfile(bundle / name, payload / 'bundle' / name)
        native = payload / 'native-linux-x86_64-avx2-f32.tar.gz'
        shutil.copyfile(ROOT / 'docs/validation/native-spacy-20261117/acceptance-candidate-linux-x86_64.tar.gz', native)
        assert digest(native) == dist.NATIVE_ARCHIVE_SHA256
        built = dist.build(native, bundle, payload / PACKAGE)
        assert digest(payload / PACKAGE) == PACKAGE_SHA and (payload / PACKAGE).stat().st_size == 7808129, built
        (payload / 'distribution.json').write_bytes(dist._manifest_bytes())
        shutil.copyfile(ROOT / 'docs/validation/native-spacy-w7-20261118/operational-protocol.json',
                        payload / 'operational-protocol.json')
        (payload / 'README.md').write_text('''---
license: mit
language: de
tags: [spacy, gguf, crispembed]
---
# CrispEmbed native German spaCy — frozen F32 profile

Converted from the authenticated official de_core_news_sm 3.8.0 wheel. All
source/model/native notices are retained in the bundle and native/package
archives. No original training corpus is included. Native scope is the exact
Linux x86-64/AVX2 CPU/F32 candidate and six declared external dependency builds;
this is not general Linux portability, macOS, mobile, grammar accuracy,
quantization or app activation qualification. Provider defaults remain OFF.

Source: https://github.com/CrispStrobe/CrispEmbed/pull/237
Use immutable commit resolve URLs and authenticate every download before parsing.
The archive digest is 4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01.
''')
        pins = {p.relative_to(payload).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                for p in sorted(payload.rglob('*')) if p.is_file()}
        (payload / 'artifacts.json').write_text(json.dumps({'schema': 'crispembed.spacy.artifacts.v1',
            'source': report['source'], 'profile': dist.manifest()['profile'], 'files': pins}, indent=2) + '\n')
        pins['artifacts.json'] = {'sha256': digest(payload / 'artifacts.json'),
                                  'bytes': (payload / 'artifacts.json').stat().st_size}
        report['payload_pins'] = pins
        save()
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'], check=True)
        from huggingface_hub import HfApi
        token = os.environ.get('HF_PUBLISH_TOKEN') or os.environ.get('HF_TOKEN_2')
        assert token, 'Hosted Hub write credential unavailable'
        api = HfApi(token=token)
        report['publisher'] = api.whoami()['name']
        api.create_repo(REPOSITORY, repo_type='model', private=False, exist_ok=True)
        published = api.upload_folder(repo_id=REPOSITORY, repo_type='model', folder_path=str(payload),
                                      commit_message='Publish authenticated immutable spaCy3.8.0 Linux F32 W7 capsule')
        revision = published.oid
        assert re.fullmatch('[0-9a-f]{40}', revision)
        report['immutable_revision'] = revision
        remote_tree = sorted(api.list_repo_files(REPOSITORY, revision=revision))
        assert set(remote_tree) == set(pins) | {'.gitattributes'}, 'Unexpected immutable hosted tree'
        report['immutable_tree'] = remote_tree
        save()
        fresh = scratch / 'fresh-empty-http-cache'
        fresh.mkdir()
        assert not list(fresh.iterdir())
        readback = {}
        for name, pin in pins.items():
            local = fresh / name
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/{REPOSITORY}/resolve/{revision}/' + urllib.parse.quote(name, safe='/')
            readback[name] = download(url, local, pin['sha256'], pin['bytes'])
        assert json.loads((fresh / 'artifacts.json').read_text())['files'] == {k: v for k, v in pins.items() if k != 'artifacts.json'}
        archive = fresh / PACKAGE
        report['fresh_readback'] = readback
        report['archive_verification'] = dist.verify_archive(archive, PACKAGE_SHA)
        cache = scratch / 'fresh-empty-install-cache'
        cache.mkdir(mode=0o700)
        report['fresh_install'] = dist.install(archive, PACKAGE_SHA, cache)
        report['installed_tree'] = dist.verify_tree(cache / PACKAGE_SHA, PACKAGE_SHA)
        report['passed'] = True
        save()
        (output / 'summary.md').write_text(f'# Immutable spaCy Linux F32 publication\n\nRevision `{revision}`. '
            f'All {len(pins)} files freshly downloaded/hash-verified; closed archive, atomic installer and immutable tree pass. '
            'No native inference, new grammar qualification or platform promotion.\n')
        print(json.dumps({'passed': True, 'repository': REPOSITORY, 'revision': revision,
                          'package_sha256': PACKAGE_SHA, 'fresh_files': len(readback)}))
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        save()
        raise


if __name__ == '__main__':
    main()
