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
NER_RUNTIME_OWNER_DEV_SHA256 = '1b65fce5864703d971880cf038a67f2ffd1ef99cad2cd49f9dc122d7882cdaa1'
RUNNERS = {
    'linux-x86_64': ('Linux', 'x86_64'),
    'linux-arm64': ('Linux', 'aarch64'),
    'macos-arm64': ('Darwin', 'arm64'),
    'windows-x86_64': ('Windows', 'AMD64'),
}
LEGACY_SCHEMA = 'crispembed.spacy.distribution.v1'


def pe_imports(path):
    """DLL names imported by a PE image (stdlib only; no dumpbin needed)."""
    data = Path(path).read_bytes()
    pe = int.from_bytes(data[0x3C:0x40], 'little')
    assert data[pe:pe + 4] == b'PE\0\0', 'Not a PE image'
    coff = pe + 4
    sections_count = int.from_bytes(data[coff + 2:coff + 4], 'little')
    optional_size = int.from_bytes(data[coff + 16:coff + 18], 'little')
    optional = coff + 20
    assert int.from_bytes(data[optional:optional + 2], 'little') == 0x20B, 'Expected PE32+'
    import_rva = int.from_bytes(data[optional + 120:optional + 124], 'little')
    sections = []
    for i in range(sections_count):
        at = optional + optional_size + 40 * i
        sections.append((int.from_bytes(data[at + 12:at + 16], 'little'),
                         int.from_bytes(data[at + 8:at + 12], 'little'),
                         int.from_bytes(data[at + 20:at + 24], 'little')))

    def offset(rva):
        for start, size, raw in sections:
            if start <= rva < start + size:
                return raw + rva - start
        raise AssertionError('RVA outside sections')

    names, at = [], offset(import_rva)
    while True:
        name_rva = int.from_bytes(data[at + 12:at + 16], 'little')
        if not name_rva:
            return names
        start = offset(name_rva)
        names.append(data[start:data.index(b'\0', start)].decode('ascii'))
        at += 20


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog_with(entry, base):
    catalog = json.loads(json.dumps(base))
    if entry:
        # A new build replaces earlier v2/pending rows for its platform but never
        # the frozen v1 W7 capsule: that row carries the only grammar claim.
        catalog['packages'] = [p for p in catalog['packages']
                               if p['platform'] != entry['platform'] or p.get('schema') == LEGACY_SCHEMA]
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
    parser.add_argument('--ner', action='store_true',
                        help='Convert ner.gguf canonically on CPython 3.12.14 and (with --publish) publish it')
    parser.add_argument('--oracle-dir', type=Path,
                        help='Original-spaCy oracle (spacy_mobile_sdk.py oracle) for the spacy_smoke comparison')
    parser.add_argument('--linux-runtime', choices=['shared', 'static'], default='shared',
                        help='Linux only: link libstdc++/libgcc dynamically (default) or statically')
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert sum(map(bool, (args.card_only, args.platform, args.ner))) == 1, \
        'choose exactly one of --platform, --ner or --card-only'
    if args.publish:
        assert os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Hub writes only from main'
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-platform'
    scratch.mkdir()
    pd = _load('spacy_platform_distribution', ROOT / 'tools/spacy/platform_distribution.py')
    card = _load('spacy_model_card', ROOT / 'tools/spacy/model_card.py')
    tracked = json.loads((ROOT / 'tools/spacy/hub_catalog.json').read_text(encoding='utf-8'))
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = {'schema': 'crispembed.spacy.platform_package.v1', 'passed': False, 'source': source,
              'ref': os.environ.get('GITHUB_REF'), 'run': os.environ.get('GITHUB_RUN_ID'),
              'platform': args.platform, 'publish': args.publish, 'card_only': args.card_only,
              'grammar_qualified': False, 'consumed_acceptance_or_grammar_panels_replayed': False}

    def save():
        (out / 'platform-package.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

    def run(label, command, cwd=ROOT, env=None):
        with (out / (label + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, env=env)

    try:
        payload = scratch / 'payload'
        payload.mkdir()
        entry = None
        ner = None
        if args.ner:
            assert sys.platform == 'linux' and host.python_version() == '3.12.14', 'Canonical NER conversion needs CPython 3.12.14'
            wheel = scratch / common.WHEEL
            report['source_wheel'] = common.download(
                'https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/' + common.WHEEL,
                wheel, common.WHEEL_SHA, 14639490)
            reference = scratch / 'reference-environment'
            run('reference-venv', [sys.executable, '-m', 'venv', reference])
            run('reference-install', [reference / 'bin/python', '-m', 'pip', 'install', '--only-binary=:all:',
                                      '--require-hashes', '-r', ROOT / 'tools/spacy/requirements-reference.txt'])
            converted = scratch / 'ner-converted'
            run('ner-conversion', [reference / 'bin/python', ROOT / 'models/convert-spacy-ner-to-gguf.py',
                                   '--wheel', wheel, '--output', converted])
            manifest = json.loads((converted / 'ner-manifest.json').read_text(encoding='utf-8'))
            assert manifest['dev_environment'] is False and manifest['reference_python'] == '3.12.14'
            assert sorted(p.name for p in converted.iterdir()) == ['ner-manifest.json', 'ner.gguf']
            pin = manifest['files']['ner.gguf']
            assert common.digest(converted / 'ner.gguf') == pin['sha256']
            assert (converted / 'ner.gguf').stat().st_size == pin['size_bytes']
            (payload / 'ner').mkdir()
            for name in ('ner.gguf', 'ner-manifest.json'):
                shutil.copyfile(converted / name, payload / 'ner' / name)
            ner = {'status': 'published', 'revision': None, 'source_commit': source,
                   'files': {f'ner/{name}': {'sha256': common.digest(payload / 'ner' / name),
                                             'bytes': (payload / 'ner' / name).stat().st_size}
                             for name in ('ner.gguf', 'ner-manifest.json')},
                   'reproduces_runtime_owner_dev_sha256': pin['sha256'] == NER_RUNTIME_OWNER_DEV_SHA256,
                   'qualification': ('Opt-in (CRISPEMBED_SPACY_NER=1, crispembed_spacy_attach_ner). '
                                     'Runtime-owner receipts: 1,776/1,776 UD GSD sentences and 1,758/1,758 '
                                     'entities identical to original spaCy on macOS arm64 and Linux AVX2 '
                                     '(run 44295825163). Not qualified for any Quds grammar use.')}
            report['ner'] = ner
            save()
        if args.platform:
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
            windows = system == 'Windows'
            apple = ['-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0'] if darwin else []
            # Visual Studio is multi-config: name the configuration everywhere.
            msvc = ['-A', 'x64'] if windows else []  # default /MD: dynamic MSVC runtime
            config = ['--config', 'Release'] if windows else []
            run('pcre2-configure', ['cmake', '-S', pcre, '-B', pcre_build, '-DCMAKE_BUILD_TYPE=Release', *apple, *msvc,
                '-DBUILD_SHARED_LIBS=OFF', '-DPCRE2_STATIC_PIC=ON', '-DPCRE2_BUILD_PCRE2_8=ON',
                '-DPCRE2_BUILD_PCRE2_16=OFF', '-DPCRE2_BUILD_PCRE2_32=OFF', '-DPCRE2_BUILD_PCRE2GREP=OFF',
                '-DPCRE2_BUILD_TESTS=OFF', '-DCMAKE_INSTALL_PREFIX=' + str(prefix)])
            run('pcre2-build', ['cmake', '--build', pcre_build, '--parallel', '4', *config])
            run('pcre2-install', ['cmake', '--install', pcre_build, *config])
            static_pcre = next(p for p in [prefix / 'lib/libpcre2-8.a', prefix / 'lib64/libpcre2-8.a',
                                           prefix / 'lib/pcre2-8-static.lib', prefix / 'lib/pcre2-8.lib']
                               if p.exists())
            build, install = scratch / 'native-build', scratch / 'native-install'
            link = (['-DCMAKE_INSTALL_NAME_DIR=@rpath', '-DCMAKE_INSTALL_RPATH=@loader_path',
                     '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON'] if darwin
                    else (['-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc']
                          if args.linux_runtime == 'static' else []))
            if windows:
                link = ['-DSPACY_PCRE2_STATIC=ON']
            configure = ['cmake', '-S', ROOT / 'tools/spacy', '-B', build, '-DCMAKE_BUILD_TYPE=Release', *apple, *msvc,
                         '-DSPACY_PCRE2_INCLUDE_DIR=' + str(prefix / 'include'),
                         '-DSPACY_PCRE2_LIBRARY=' + str(static_pcre), *link]
            run('native-configure', configure)
            run('native-build', ['cmake', '--build', build, '--parallel', '4', *config])
            run('ctest', ['ctest', '--test-dir', build, '--output-on-failure', *(['-C', 'Release'] if windows else [])])
            run('native-install', ['cmake', '--install', build, '--prefix', install, *config])
            cache_flags = {line.split(':', 1)[0]: line.split('=', 1)[1]
                           for line in (build / 'CMakeCache.txt').read_text(encoding='utf-8', errors='replace').splitlines()
                           if re.match(r'^(GGML_|CRISPEMBED_|SPACY_PCRE2_STATIC|CMAKE_BUILD_TYPE|CMAKE_SHARED_LINKER_FLAGS|CMAKE_OSX_|CMAKE_MSVC_|CMAKE_CXX_COMPILER_ID|CMAKE_CXX_COMPILER_VERSION|CMAKE_GENERATOR)', line)
                           and '=' in line}
            if windows:
                compiler = (f"{cache_flags.get('CMAKE_CXX_COMPILER_ID', 'MSVC')} "
                            f"{cache_flags.get('CMAKE_CXX_COMPILER_VERSION', '')}").strip()
            else:
                compiler = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
            target = pd.PLATFORMS[args.platform]
            library = install / target['library_dir'] / target['library']
            if windows:
                imports = pe_imports(library)
                dependencies = {'pe_imports': imports, 'bundled': False, 'pcre2': 'static ' + PCRE2,
                                'msvc_runtime': 'dynamic (MSVCP140/VCRUNTIME140, Visual C++ 2015-2022 redistributable)'}
                assert not [n for n in imports if 'pcre' in n.lower() or 'python' in n.lower()], imports
            elif darwin:
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
                                'static_runtime': ['libstdc++', 'libgcc'] if args.linux_runtime == 'static' else []}
                assert not [n for n in needed if 'pcre' in n or 'python' in n.lower()], needed
                if args.linux_runtime == 'static':
                    assert not [n for n in needed if 'stdc++' in n], needed
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
            # The Quds development fixtures contain a case directory named
            # `nul`, a reserved Windows device name git cannot check out there;
            # Windows relies on the decoded original-spaCy comparison below.
            quds_parity = not windows
            oracle = scratch / 'quds-reference'
            if quds_parity:
                run('oracle-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                                     'https://github.com/CrispStrobe/quds_office_suite.git', oracle])
                run('oracle-sparse', ['git', 'sparse-checkout', 'set', 'tools/proofing/spacy'], oracle)
                run('oracle-checkout', ['git', 'checkout', '--detach', ORACLE], oracle)
            else:
                report['quds_parity_unrun_reason'] = (
                    "Quds development fixtures include a case directory named 'nul', a reserved Windows device "
                    "name that cannot be checked out; decoded exactness is verified with spacy_smoke instead.")
            run('numpy-install', [sys.executable, '-m', 'pip', 'install', 'numpy==2.4.6'])
            def check(label, command):
                with (out / f'{label}.log').open('w') as log:
                    # faulthandler keeps a native crash diagnosable in the log.
                    code = subprocess.run([sys.executable, '-X', 'faulthandler', *map(str, command)],
                                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                target = out / f'{label}.json'
                return code, (json.loads(target.read_text(encoding='utf-8')) if target.exists() else None)

            report['text_api'] = {}
            code, api = check('text-api', [ROOT / 'tests/test_spacy_text_api.py', '--library', installed_library,
                                           '--bundle', installed_bundle, '--report', out / 'text-api.json'])
            report['text_api'] = {'returncode': code, 'passed': bool(api and api.get('status') == 'pass'),
                                  'controls': (api or {}).get('tests_run')}
            save()
            with (out / 'enable-api.log').open('w') as log:
                enable = subprocess.run([sys.executable, '-X', 'faulthandler', str(ROOT / 'tests/test_spacy_enable_api.py'),
                                         '--library', str(installed_library), '--bundle', str(installed_bundle)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={k: v for k, v in os.environ.items()
                                             if k not in ('CRISPEMBED_SPACY', 'CRISPEMBED_SPACY_NER')})
            report['enable_api'] = {'returncode': enable.returncode, 'passed': enable.returncode == 0}
            assert report['enable_api']['passed'], 'Process-wide enable API controls failed'
            # spacy_smoke (C ABI only) on copies of the installed library bytes,
            # compared token by token and entity by entity with original spaCy.
            if args.oracle_dir:
                mobile = _load('spacy_mobile_sdk', ROOT / 'tools/ci-heavy/spacy_mobile_sdk.py')
                smoke_built = next(p for p in build.rglob('spacy-smoke' + ('.exe' if windows else ''))
                                   if p.is_file())
                smoke_dir = scratch / 'smoke'
                smoke_dir.mkdir()
                smoke = smoke_dir / smoke_built.name
                shutil.copyfile(smoke_built, smoke)
                os.chmod(smoke, 0o755)
                shutil.copyfile(installed_library, smoke_dir / installed_library.name)
                assert common.digest(smoke_dir / installed_library.name) == built['library_sha256']
                ner_dir = scratch / 'smoke-models'
                ner_dir.mkdir()
                _, ner_model = mobile.model_files(ner_dir)
                texts = mobile.corpus_lines(scratch / 'smoke-texts.txt')
                smoke_env = dict(os.environ)
                smoke_env['LD_LIBRARY_PATH' if system == 'Linux' else 'DYLD_LIBRARY_PATH'] = str(smoke_dir)
                smoke_out = out / 'smoke.jsonl'
                with smoke_out.open('wb') as stdout, (out / 'smoke.log').open('w') as log:
                    code = subprocess.run([str(smoke), *map(str, mobile.smoke_arguments(installed_bundle, ner_model,
                                                                                          texts))],
                                          stdout=stdout, stderr=log, env=smoke_env).returncode
                assert code == 0, f'spacy_smoke exited {code}'
                comparison = mobile.compare(run, out, smoke_out, args.oracle_dir, 'smoke')
                report['original_spacy_smoke'] = comparison
                save()
                assert comparison['passed'], 'spacy_smoke output differs from original spaCy'
            elif not quds_parity:
                raise AssertionError('Windows needs --oracle-dir: the Quds parity panels cannot run there')
            report['parity'] = {}
            for script, name in ([('spacy_parity.py', 'prepared'), ('spacy_text_parity.py', 'text')]
                                 if quds_parity else []):
                code, result = check(f'{name}-parity', [ROOT / 'tests' / script, '--library', installed_library,
                                                        '--bundle', installed_bundle, '--quds-root', oracle,
                                                        '--report', out / f'{name}-parity.json'])
                report['parity'][name] = {'returncode': code, 'passed': bool(result and result['passed']),
                                          'counts': (result or {}).get('counts')}
                save()
            save()
            assert all(item['passed'] for item in report['parity'].values()), 'Development parity failed'
            assert report['text_api']['passed'], 'Text API controls failed'
            counts = report['parity']['text']['counts'] or {} if quds_parity else {}
            run_id = os.environ.get('GITHUB_RUN_ID')
            smoke_result = report.get('original_spacy_smoke')
            smoke_text = (f"spacy_smoke vs original spaCy: {smoke_result['exact_cases']}/{smoke_result['cases']} "
                          f"development texts, {smoke_result['tokens']} tokens, "
                          f"{smoke_result['entities']['matched']}/{smoke_result['entities']['oracle']} entities exact"
                          if smoke_result else None)
            prefix_path = f'packages/{args.platform}/{source}'
            entry = {
                'platform': args.platform, 'status': 'published', 'revision': None,
                'path': f'{prefix_path}/{capsule.name}', 'sha256': built['archive_sha256'],
                'bytes': built['archive_size_bytes'], 'manifest_sha256': built['manifest_sha256'],
                'library_sha256': built['library_sha256'], 'schema': pd.SCHEMA,
                'requirements': (f"{host.platform()} build; dependencies: "
                                 f"{dependencies.get('elf_needed') or dependencies.get('pe_imports') or 'system libc++/libSystem only'}"
                                 + (f"; symbol floors {dependencies['symbol_version_floors']}"
                                    if dependencies.get('symbol_version_floors') else '')),
                'analysis': ((f"Hosted development parity on this exact library (run {run_id}): "
                              f"{counts.get('cases')} cases, {counts.get('tokens')} tokens, "
                              f"{counts.get('numeric_stages')} numerical stages exact; "
                              if quds_parity else f"Hosted run {run_id} on this exact library: ")
                             + (f"{smoke_text}; " if smoke_text else '')
                             + "text-API and enable-API controls pass. Component evidence only; not an app "
                               "qualification."),
                'grammar': 'Not qualified for this build',
                'receipt': f'{prefix_path}/receipt.json',
            }
            report['catalog_entry'] = entry
        if not args.publish:
            # Dry-run outputs travel as a short-lived workflow artifact only, for
            # local consumer checks; they are never a publication.
            for item in payload.rglob('*'):
                if item.is_file():
                    target = out / 'dry-run-payload' / item.relative_to(payload)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item, target)
            report['passed'] = True
            report['dry_run'] = True
            save()
            (out / 'summary.md').write_text('# spaCy distribution dry run\n\nNothing was uploaded.\n', encoding='utf-8', newline='\n')
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
            receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
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
        if ner:
            revision, fresh, report['ner_publication'] = publish(
                {path: payload / path for path in ner['files']},
                f'Publish opt-in NER model converted on CPython 3.12.14 from CrispEmbed {source[:12]}')
            ner['revision'] = revision
            report['ner'] = ner
            save()
        catalog = catalog_with(entry, current_hub_catalog(tracked))
        if ner:
            catalog['ner'] = ner
        (payload / HUB_CATALOG).write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8', newline='\n')
        (payload / 'README.md').write_text(card.render(catalog), encoding='utf-8', newline='\n')
        revision, _, report['card_publication'] = publish(
            {HUB_CATALOG: payload / HUB_CATALOG, 'README.md': payload / 'README.md'},
            'Update model card and platform catalog')
        report['passed'] = True
        save()
        (out / 'summary.md').write_text('# spaCy Hub publication\n\n' + json.dumps(
            {k: report[k]['revision'] for k in ('package_publication', 'ner_publication', 'card_publication')
             if k in report}) + '\n', encoding='utf-8', newline='\n')
        print(json.dumps({'passed': True, 'card_revision': revision,
                          'package_revision': report.get('package_publication', {}).get('revision')}))
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        save()
        raise


if __name__ == '__main__':
    main()
