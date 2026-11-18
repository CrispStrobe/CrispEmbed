#!/usr/bin/env python3
"""Build, test, package and (optionally) publish a v2 native spaCy capsule.

Hosted only (heavy-cpu.yml). Without --publish this is a dry run: it builds the
library from the checked-out source, runs ctest, the development parity panels
and the text-API controls against the library *installed from the capsule*,
and writes receipts, but uploads nothing.

With --publish (main only) it also uploads the capsule, its receipt and the
merged model card in ONE Hub commit, then checks success the W7 way: immutable
commit, complete expected tree, fresh empty-cache HTTP download with digest and
length for every uploaded file, closed-archive verification and a new atomic
installation from the downloaded bytes. Upload alone is never success.

`--card-only --publish` re-renders and publishes only the model card.

Never runs the consumed acceptance or W6/W7 grammar panels; a new build's
grammar claim is always "not qualified".
"""
from __future__ import annotations

import argparse
import hashlib
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

import spacy_w7_publish as common

ROOT = common.ROOT
PCRE2 = 'b2bd4254b379b9d7dc9a3dda060a7e27009ccdff'
ORACLE = 'dcf2ec9b4d6fba05fa5eec2634c6654266332cc9'
BUNDLE_REVISION = '659dae33997d00296ae12b76510fa4dfe26540e0'
HUB_CATALOG = 'hub-catalog.json'
RUNNERS = {
    'linux-x86_64': ('Linux', 'x86_64'),
    'linux-arm64': ('Linux', 'aarch64'),
    'macos-arm64': ('Darwin', 'arm64'),
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog_with(entry, base):
    catalog = json.loads(json.dumps(base))
    if entry:
        catalog['packages'] = [p for p in catalog['packages'] if p['platform'] != entry['platform']]
        catalog['packages'].append(entry)
        catalog['packages'].sort(key=lambda p: (p.get('path') is None, p['platform']))
    return catalog


def current_hub_catalog(tracked):
    """Merge base: the Hub's last published catalog (not a trust input)."""
    url = f'https://huggingface.co/{common.REPOSITORY}/resolve/main/{HUB_CATALOG}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}),
                                    timeout=60) as response:
            remote = json.load(response)
        if remote.get('schema') == tracked['schema']:
            return remote
    except Exception:  # first publication or unreachable: use the tracked catalog
        pass
    return tracked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', choices=sorted(RUNNERS))
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--card-only', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert args.card_only or args.platform, '--platform is required unless --card-only'
    if args.publish:
        assert os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Hub writes only from main'
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-platform'
    scratch.mkdir()
    pd = _load('spacy_platform_distribution', ROOT / 'tools/spacy/platform_distribution.py')
    card = _load('spacy_model_card', ROOT / 'tools/spacy/model_card.py')
    tracked = json.loads((ROOT / 'tools/spacy/hub_catalog.json').read_text())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = {'schema': 'crispembed.spacy.platform_package.v1', 'passed': False, 'source': source,
              'ref': os.environ.get('GITHUB_REF'), 'run': os.environ.get('GITHUB_RUN_ID'),
              'platform': args.platform, 'publish': args.publish, 'card_only': args.card_only,
              'grammar_qualified': False, 'consumed_acceptance_or_grammar_panels_replayed': False}

    def save():
        (out / 'platform-package.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, cwd=ROOT, env=None):
        with (out / (label + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, env=env)

    try:
        payload = scratch / 'payload'
        payload.mkdir()
        entry = None
        if not args.card_only:
            system, machine = RUNNERS[args.platform]
            assert host.system() == system and host.machine() == machine, \
                f'Real {args.platform} runner required (got {host.system()}/{host.machine()}); no emulation'
            report['runner'] = {'system': host.platform(), 'python': sys.version}
            bundle = scratch / 'bundle'
            bundle.mkdir()
            report['bundle_downloads'] = {}
            dist = common.load_distribution()
            for name, pin in dist.BUNDLE_FILES.items():
                url = f'https://huggingface.co/{common.REPOSITORY}/resolve/{BUNDLE_REVISION}/bundle/{name}'
                report['bundle_downloads'][name] = common.download(url, bundle / name, pin['sha256'], pin['size_bytes'])
            # Static PCRE2 so the capsule needs only the platform C/C++ runtime.
            pcre, pcre_build, prefix = scratch / 'pcre2', scratch / 'pcre2-build', scratch / 'pcre2-install'
            run('pcre2-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                                'https://github.com/PCRE2Project/pcre2.git', pcre])
            run('pcre2-checkout', ['git', 'checkout', '--detach', PCRE2], pcre)
            darwin = system == 'Darwin'
            apple = ['-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0'] if darwin else []
            run('pcre2-configure', ['cmake', '-S', pcre, '-B', pcre_build, '-DCMAKE_BUILD_TYPE=Release', *apple,
                '-DBUILD_SHARED_LIBS=OFF', '-DPCRE2_STATIC_PIC=ON', '-DPCRE2_BUILD_PCRE2_8=ON',
                '-DPCRE2_BUILD_PCRE2_16=OFF', '-DPCRE2_BUILD_PCRE2_32=OFF', '-DPCRE2_BUILD_PCRE2GREP=OFF',
                '-DPCRE2_BUILD_TESTS=OFF', '-DCMAKE_INSTALL_PREFIX=' + str(prefix)])
            run('pcre2-build', ['cmake', '--build', pcre_build, '--parallel', '4'])
            run('pcre2-install', ['cmake', '--install', pcre_build])
            static_pcre = next(p for p in [prefix / 'lib/libpcre2-8.a', prefix / 'lib64/libpcre2-8.a'] if p.exists())
            build, install = scratch / 'native-build', scratch / 'native-install'
            link = (['-DCMAKE_INSTALL_NAME_DIR=@rpath', '-DCMAKE_INSTALL_RPATH=@loader_path',
                     '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON'] if darwin
                    else ['-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc'])
            configure = ['cmake', '-S', ROOT / 'tools/spacy', '-B', build, '-DCMAKE_BUILD_TYPE=Release', *apple,
                         '-DSPACY_PCRE2_INCLUDE_DIR=' + str(prefix / 'include'),
                         '-DSPACY_PCRE2_LIBRARY=' + str(static_pcre), *link]
            run('native-configure', configure)
            run('native-build', ['cmake', '--build', build, '--parallel', '4'])
            run('ctest', ['ctest', '--test-dir', build, '--output-on-failure'])
            run('native-install', ['cmake', '--install', build, '--prefix', install])
            cache_flags = {line.split(':', 1)[0]: line.split('=', 1)[1]
                           for line in (build / 'CMakeCache.txt').read_text().splitlines()
                           if re.match(r'^(GGML_|CRISPEMBED_|CMAKE_BUILD_TYPE|CMAKE_SHARED_LINKER_FLAGS|CMAKE_OSX_)', line)
                           and '=' in line}
            compiler = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
            library = install / 'lib' / pd.PLATFORMS[args.platform]['library']
            if darwin:
                dependencies = {'otool_L': subprocess.check_output(['otool', '-L', str(library)], text=True),
                                'bundled': False, 'pcre2': 'static ' + PCRE2}
                assert 'libpcre2' not in dependencies['otool_L'] and 'Python' not in dependencies['otool_L']
            else:
                dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
                needed = re.findall(r'\(NEEDED\).*\[(.+?)\]', dynamic)
                symbols = subprocess.check_output(['objdump', '-T', str(library)], text=True)
                floors = {}
                for family, version in re.findall(r'\b(GLIBC|GLIBCXX|CXXABI)_([0-9.]+)\b', symbols):
                    key = tuple(int(x) for x in version.split('.'))
                    if family not in floors or key > floors[family][0]:
                        floors[family] = (key, version)
                dependencies = {'elf_needed': needed, 'bundled': False, 'pcre2': 'static ' + PCRE2,
                                'symbol_version_floors': {k: v[1] for k, v in floors.items()},
                                'static_runtime': ['libstdc++', 'libgcc']}
                assert not [n for n in needed if 'pcre' in n or 'stdc++' in n or 'python' in n.lower()], needed
            capsule = payload / pd.archive_name(args.platform)
            built = pd.build(args.platform, install, bundle, capsule, source_commit=source,
                             build_info={'cmake_cache': cache_flags, 'compiler': compiler,
                                         'pcre2_commit': PCRE2, 'runner': host.platform()},
                             system_dependencies=dependencies)
            report['capsule'] = built
            report['archive_verification'] = pd.verify_archive(capsule, built['archive_sha256'])
            cache = scratch / 'install-cache'
            report['install'] = pd.install(capsule, built['archive_sha256'], cache)
            installed = cache / built['archive_sha256']
            installed_library = installed / built['entrypoints']['library']
            installed_bundle = installed / 'bundle'
            save()
            # Qualify the exact bytes inside the capsule, not the build tree.
            oracle = scratch / 'quds-reference'
            run('oracle-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                                 'https://github.com/CrispStrobe/quds_office_suite.git', oracle])
            run('oracle-sparse', ['git', 'sparse-checkout', 'set', 'tools/proofing/spacy'], oracle)
            run('oracle-checkout', ['git', 'checkout', '--detach', ORACLE], oracle)
            run('numpy-install', [sys.executable, '-m', 'pip', 'install', 'numpy==2.4.6'])
            report['parity'] = {}
            for script, name in [('spacy_parity.py', 'prepared'), ('spacy_text_parity.py', 'text')]:
                target = out / f'{name}-parity.json'
                subprocess.run([sys.executable, str(ROOT / 'tests' / script), '--library', str(installed_library),
                                '--bundle', str(installed_bundle), '--quds-root', str(oracle),
                                '--report', str(target)], cwd=ROOT, check=False,
                               stdout=(out / f'{name}-parity.log').open('w'), stderr=subprocess.STDOUT)
                result = json.loads(target.read_text())
                report['parity'][name] = {'passed': result['passed'], 'counts': result.get('counts')}
            subprocess.run([sys.executable, str(ROOT / 'tests/test_spacy_text_api.py'), '--library',
                            str(installed_library), '--bundle', str(installed_bundle),
                            '--report', str(out / 'text-api.json')], cwd=ROOT, check=False,
                           stdout=(out / 'text-api.log').open('w'), stderr=subprocess.STDOUT)
            api = json.loads((out / 'text-api.json').read_text())
            report['text_api'] = {'passed': api.get('passed'), 'controls': len(api.get('controls', api.get('cases', [])))}
            save()
            assert all(item['passed'] for item in report['parity'].values()), 'Development parity failed'
            assert report['text_api']['passed'], 'Text API controls failed'
            counts = report['parity']['text']['counts'] or {}
            run_id = os.environ.get('GITHUB_RUN_ID')
            prefix_path = f'packages/{args.platform}/{source}'
            entry = {
                'platform': args.platform, 'status': 'published', 'revision': None,
                'path': f'{prefix_path}/{capsule.name}', 'sha256': built['archive_sha256'],
                'bytes': built['archive_size_bytes'], 'manifest_sha256': built['manifest_sha256'],
                'library_sha256': built['library_sha256'], 'schema': pd.SCHEMA,
                'requirements': (f"{host.platform()} build; dependencies: "
                                 f"{dependencies.get('elf_needed') or 'system libc++/libSystem only'}"
                                 + (f"; symbol floors {dependencies['symbol_version_floors']}"
                                    if dependencies.get('symbol_version_floors') else '')),
                'analysis': (f"Hosted development parity on this exact library (run {run_id}): "
                             f"{counts.get('cases')} cases, {counts.get('tokens')} tokens, "
                             f"{counts.get('numeric_stages')} numerical stages exact; text-API controls pass. "
                             "Component evidence only; not an app qualification."),
                'grammar': 'Not qualified for this build',
                'receipt': f'{prefix_path}/receipt.json',
            }
            report['catalog_entry'] = entry
        if not args.publish:
            report['passed'] = True
            report['dry_run'] = True
            save()
            (out / 'summary.md').write_text('# spaCy platform capsule dry run\n\nBuilt, packaged, installed and '
                                            'tested from capsule bytes. Nothing was uploaded.\n')
            return
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'], check=True)
        from huggingface_hub import CommitOperationAdd, HfApi
        token = os.environ.get('HF_PUBLISH_TOKEN') or os.environ.get('HF_TOKEN_2')
        assert token, 'Hosted Hub write credential unavailable'
        hub = HfApi(token=token)

        def publish(uploads, message):
            """One Hub commit, then immutable-tree and fresh-cache readback."""
            pins = {path: {'sha256': common.digest(local), 'bytes': local.stat().st_size}
                    for path, local in uploads.items()}
            before = set(hub.list_repo_files(common.REPOSITORY))
            commit = hub.create_commit(
                repo_id=common.REPOSITORY, repo_type='model', commit_message=message,
                operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
                            for path, local in uploads.items()])
            revision = commit.oid
            assert re.fullmatch('[0-9a-f]{40}', revision)
            tree = set(hub.list_repo_files(common.REPOSITORY, revision=revision))
            assert tree == before | set(uploads), 'Unexpected immutable hosted tree'
            fresh = scratch / f'fresh-{revision[:12]}'
            fresh.mkdir()
            readback = {}
            for path, pin in pins.items():
                local = fresh / path
                local.parent.mkdir(parents=True, exist_ok=True)
                url = (f'https://huggingface.co/{common.REPOSITORY}/resolve/{revision}/'
                       + urllib.parse.quote(path, safe='/'))
                readback[path] = common.download(url, local, pin['sha256'], pin['bytes'])
            return revision, fresh, {'revision': revision, 'tree_file_count': len(tree),
                                     'pins': pins, 'fresh_readback': readback}

        if entry:
            receipt = payload / 'receipt.json'
            receipt.write_text(json.dumps(report, indent=2) + '\n')
            revision, fresh, report['package_publication'] = publish(
                {entry['path']: payload / pd.archive_name(args.platform), entry['receipt']: receipt},
                f'Publish {args.platform} spaCy capsule built from CrispEmbed {source[:12]}')
            downloaded = fresh / entry['path']
            report['fresh_archive_verification'] = pd.verify_archive(downloaded, entry['sha256'])
            fresh_cache = scratch / 'fresh-install-cache'
            report['fresh_install'] = pd.install(downloaded, entry['sha256'], fresh_cache)
            report['fresh_installed_tree'] = pd.verify_tree(fresh_cache / entry['sha256'], entry['sha256'])
            entry['revision'] = revision
            report['catalog_entry'] = entry
            save()
        catalog = catalog_with(entry, current_hub_catalog(tracked))
        (payload / HUB_CATALOG).write_text(json.dumps(catalog, indent=2) + '\n')
        (payload / 'README.md').write_text(card.render(catalog))
        revision, _, report['card_publication'] = publish(
            {HUB_CATALOG: payload / HUB_CATALOG, 'README.md': payload / 'README.md'},
            'Update model card and platform catalog')
        report['passed'] = True
        save()
        (out / 'summary.md').write_text('# spaCy Hub publication\n\n' + json.dumps(
            {k: report[k]['revision'] for k in ('package_publication', 'card_publication') if k in report}) + '\n')
        print(json.dumps({'passed': True, 'card_revision': revision,
                          'package_revision': report.get('package_publication', {}).get('revision')}))
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        save()
        raise


if __name__ == '__main__':
    main()
