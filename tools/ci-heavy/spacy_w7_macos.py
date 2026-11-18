#!/usr/bin/env python3
"""Real ARM64 Mac component qualification; signing/app evidence is separate."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import spacy_w7_publish as common

ROOT = common.ROOT
PCRE2 = 'b2bd4254b379b9d7dc9a3dda060a7e27009ccdff'
ORACLE = 'dcf2ec9b4d6fba05fa5eec2634c6654266332cc9'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Real ARM64 macOS required; no emulation'
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-w7-macos'
    scratch.mkdir()
    out = Path(os.environ['HEAVY_OUT'])
    report = {'schema': 'crispembed.spacy.macos.component.v1', 'passed': False,
              'scope': 'macOS ARM64 native CPU/F32 component; unsigned development candidate',
              'consumed_acceptance_or_W6_replayed': False, 'signed_app_qualified': False,
              'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'model_revision': args.revision, 'architecture': platform.machine(), 'system': platform.platform(),
              'pcre2_commit': PCRE2, 'development_oracle_commit': ORACLE}

    def save():
        (out / 'macos-component.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, cwd=ROOT):
        with (out / (label + '.log')).open('w') as f:
            subprocess.run([str(x) for x in command], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)

    try:
        url = f'https://huggingface.co/{common.REPOSITORY}/resolve/{args.revision}/artifacts.json'
        manifest = json.load(urllib.request.urlopen(url, timeout=60))
        dist = common.load_distribution()
        bundle = scratch / 'bundle'
        bundle.mkdir()
        report['fresh_model_downloads'] = {}
        for name, pin in dist.BUNDLE_FILES.items():
            key = 'bundle/' + name
            assert manifest['files'][key] == {'sha256': pin['sha256'], 'bytes': pin['size_bytes']}
            report['fresh_model_downloads'][name] = common.download(
                f'https://huggingface.co/{common.REPOSITORY}/resolve/{args.revision}/' + urllib.parse.quote(key),
                bundle / name, pin['sha256'], pin['size_bytes'])
        pcre = scratch / 'pcre2'
        run('pcre2-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/PCRE2Project/pcre2.git', pcre])
        run('pcre2-checkout', ['git', 'checkout', '--detach', PCRE2], pcre)
        pcre_build, prefix = scratch / 'pcre2-build', scratch / 'pcre2-install'
        run('pcre2-configure', ['cmake', '-S', pcre, '-B', pcre_build, '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0',
            '-DBUILD_SHARED_LIBS=OFF', '-DPCRE2_STATIC_PIC=ON', '-DPCRE2_BUILD_PCRE2_8=ON',
            '-DPCRE2_BUILD_PCRE2_16=OFF', '-DPCRE2_BUILD_PCRE2_32=OFF', '-DPCRE2_BUILD_PCRE2GREP=OFF',
            '-DPCRE2_BUILD_TESTS=OFF', '-DCMAKE_INSTALL_PREFIX=' + str(prefix)])
        run('pcre2-build', ['cmake', '--build', pcre_build, '--parallel', '2'])
        run('pcre2-install', ['cmake', '--install', pcre_build])
        build, install = scratch / 'native-build', scratch / 'native-install'
        run('native-configure', ['cmake', '-S', ROOT / 'tools/spacy', '-B', build,
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0',
            '-DSPACY_PCRE2_INCLUDE_DIR=' + str(prefix / 'include'), '-DSPACY_PCRE2_LIBRARY=' + str(prefix / 'lib/libpcre2-8.a'),
            '-DCMAKE_INSTALL_NAME_DIR=@rpath', '-DCMAKE_INSTALL_RPATH=@loader_path', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON'])
        run('native-build', ['cmake', '--build', build, '--parallel', '2'])
        run('ctest', ['ctest', '--test-dir', build, '--output-on-failure'])
        run('native-install', ['cmake', '--install', build, '--prefix', install])
        library = install / 'lib/libcrispembed-spacy.dylib'
        file_type = subprocess.check_output(['file', str(library)], text=True)
        assert 'arm64' in file_type and 'x86_64' not in file_type
        dependencies = subprocess.check_output(['otool', '-L', str(library)], text=True)
        assert 'libpcre2' not in dependencies and 'Python' not in dependencies
        report['library'] = {'sha256': common.digest(library), 'bytes': library.stat().st_size,
                             'file': file_type, 'dependencies': dependencies}
        report['compiler'] = subprocess.check_output(['clang++', '--version'], text=True)
        report['xcode'] = subprocess.check_output(['xcodebuild', '-version'], text=True)
        report['cpu'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        oracle = scratch / 'quds-reference'
        run('oracle-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                             'https://github.com/CrispStrobe/quds_office_suite.git', oracle])
        run('oracle-sparse', ['git', 'sparse-checkout', 'set', 'tools/proofing/spacy'], oracle)
        run('oracle-checkout', ['git', 'checkout', '--detach', ORACLE], oracle)
        run('numpy-install', [sys.executable, '-m', 'pip', 'install', 'numpy==2.4.6'])
        for script, name in [('spacy_parity.py', 'prepared'), ('spacy_text_parity.py', 'text')]:
            run(name + '-parity', [sys.executable, ROOT / 'tests' / script, '--library', library, '--bundle', bundle,
                                   '--quds-root', oracle, '--report', out / (name + '-parity.json')])
            result = json.loads((out / (name + '-parity.json')).read_text())
            assert result['passed']
        run('text-api', [sys.executable, ROOT / 'tests/test_spacy_text_api.py', '--library', library,
                         '--bundle', bundle, '--report', out / 'text-api.json'])
        os.environ['CRISPEMBED_SPACY'] = '1'
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_spacy_text_api import NativeAPI, validate_bundle
        api = NativeAPI(library, bundle, validate_bundle(bundle))
        latencies, rss, fds = [], [], []
        data = 'Eine ruhige Schildkröte wartet.'.encode()
        for index in range(28):
            started = time.perf_counter()
            context, error = api.open()
            assert context, error
            document = None
            try:
                status, document, error = api.call(context, data, analyze=True)
                assert status == 0 and document, error
                tokens = api.tokens(document)
                assert tokens and tokens[0]['text'] == b'Eine' and tokens[-1]['text'] == b'.'
                for token in tokens:
                    assert data[token['byte_start']:token['byte_end']] == token['text']
            finally:
                if document:
                    api.lib.crispembed_spacy_document_free(document)
                api.lib.crispembed_spacy_free(context)
            gc.collect()
            if index >= 3:
                latencies.append(time.perf_counter() - started)
                rss.append(int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True)) * 1024)
                fds.append(len(os.listdir('/dev/fd')))
        assert max(rss) - rss[0] <= 32 * 1024 * 1024 and fds[-1] <= fds[0]
        report['finite_lifecycle_sample'] = {'warmups': 3, 'measured_context_document_pairs_created_and_freed': 25,
            'median_open_analyze_copy_free_seconds': statistics.median(latencies), 'times_seconds': latencies,
            'rss_bytes': rss, 'fds': fds, 'rss_growth_limit': 32 * 1024 * 1024,
            'limits': 'Single process/short authored operational input; not a leak proof or cross-host speed claim.'}
        report['passed'] = True
        save()
        payload = scratch / 'macos-payload'
        payload.mkdir()
        files = {str(p.relative_to(install)): p.read_bytes() for p in install.rglob('*') if p.is_file()}
        archive = payload / 'spacy-native-macos-arm64-f32-unsigned.tar.gz'
        archive.write_bytes(dist._pack(files))
        report['unsigned_native_archive'] = {'sha256': common.digest(archive), 'bytes': archive.stat().st_size,
            'files': {n: {'sha256': __import__('hashlib').sha256(b).hexdigest(), 'bytes': len(b)} for n, b in files.items()}}
        save()
        shutil.copyfile(out / 'macos-component.json', payload / 'macos-component.json')
        for name in ('prepared-parity.json', 'text-parity.json', 'text-api.json'):
            shutil.copyfile(out / name, payload / name)
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'], check=True)
        from huggingface_hub import HfApi
        api_hub = HfApi(token=os.environ.get('HF_PUBLISH_TOKEN') or os.environ.get('HF_TOKEN_2'))
        path = 'macos-candidates/' + report['source']
        commit = api_hub.upload_folder(repo_id=common.REPOSITORY, folder_path=str(payload), path_in_repo=path,
                                      commit_message='Retain unsigned real-ARM64 component; signing/app qualification pending').oid
        fresh = scratch / 'fresh-native-readback'
        fresh.mkdir()
        verified = {}
        for file in payload.iterdir():
            url = f'https://huggingface.co/{common.REPOSITORY}/resolve/{commit}/{path}/{file.name}'
            verified[file.name] = common.download(url, fresh / file.name, common.digest(file), file.stat().st_size)
        (out / 'macos-hosted-artifacts.json').write_text(json.dumps({'passed': True, 'revision': commit,
            'files': verified, 'signed_app_qualified': False}, indent=2) + '\n')
        (out / 'summary.md').write_text('# Real ARM64 macOS component\n\nDevelopment parity/API and finite lifecycle sample pass. '
            'Unsigned candidate freshly downloaded and hashed. Signing, sandbox and actual app remain separate gates.\n')
    except Exception as exc:
        report['passed'] = False
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        save()
        raise


if __name__ == '__main__':
    main()
