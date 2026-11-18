#!/usr/bin/env python3
"""Mobile SDKs for the native German spaCy runtime (iOS, Android).

Mobile apps must ship the native code inside the signed app; only model files
may be downloaded. These SDK archives are therefore build inputs for app
developers, not installable capsules. Each target is run, not just built:

  oracle          (Linux x86-64) original spaCy 3.8.18 annotations and entities
                  for the 60-text development corpus, from the hash-locked
                  reference environment (it resolves on Linux x86-64 only).
  ios             (macOS arm64) static PCRE2 + crispembed-spacy for iphoneos
                  and iphonesimulator, one combined static archive per slice,
                  an XCFramework, and spacy_smoke run inside a booted iOS
                  Simulator against the oracle (analysis + NER).
  android-build   (Linux) NDK builds for arm64-v8a and x86_64 (API 24).
  android-finish  compares the emulator's x86_64 spacy_smoke output with the
                  oracle and packages both ABIs. arm64-v8a is build-only.

With --publish (main only) the SDK archive and its receipt go to the Hub in one
commit, followed by fresh-cache readback and a catalog/model-card update.
Never runs the consumed acceptance or grammar panels; grammar is never claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform as host
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse

import spacy_w7_publish as common
import spacy_w7_platform_package as plat

ROOT = common.ROOT
CORPUS = ROOT / 'tests/fixtures/spacy-ggml-ab/corpus-de.json'
NER_REVISION = '8b5d7084bab236fff9921ef5e397069163e83738'
NER_SHA = '1b65fce5864703d971880cf038a67f2ffd1ef99cad2cd49f9dc122d7882cdaa1'
NER_BYTES = 6135456
IOS_DEPLOYMENT = '15.0'
ANDROID_API = '24'
GGML_ARCHIVES = ('libggml.a', 'libggml-base.a', 'libggml-cpu.a')


def digest(path):
    return common.digest(path)


class Runner:
    def __init__(self, out):
        self.out = out

    def __call__(self, label, command, cwd=ROOT, env=None, capture=None):
        """Log to <label>.log; with `capture`, stdout goes to that file instead."""
        with (self.out / f'{label}.log').open('w') as log:
            if capture is None:
                return subprocess.run([str(c) for c in command], cwd=cwd, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, check=True)
            with open(capture, 'wb') as stdout:
                return subprocess.run([str(c) for c in command], cwd=cwd, env=env, stdout=stdout,
                                      stderr=log, check=True)


def model_files(scratch):
    """Bundle (pinned revision) and ner.gguf, both hash-checked."""
    bundle = scratch / 'bundle'
    bundle.mkdir(exist_ok=True)
    for name, pin in common.load_distribution().BUNDLE_FILES.items():
        if not (bundle / name).exists():
            common.download(f'https://huggingface.co/{common.REPOSITORY}/resolve/{plat.BUNDLE_REVISION}/bundle/{name}',
                            bundle / name, pin['sha256'], pin['size_bytes'])
    ner = scratch / 'ner.gguf'
    if not ner.exists():
        common.download(f'https://huggingface.co/{common.REPOSITORY}/resolve/{NER_REVISION}/ner/ner.gguf',
                        ner, NER_SHA, NER_BYTES)
    return bundle, ner


def smoke_arguments(bundle, ner, texts):
    return [bundle / 'model.gguf', digest(bundle / 'model.gguf'), bundle / 'resources.json',
            digest(bundle / 'resources.json'), texts, ner, NER_SHA]


def corpus_lines(path):
    texts = json.loads(CORPUS.read_text(encoding='utf-8'))['texts']
    assert not any('\n' in t or '\r' in t for t in texts)
    path.write_text('\n'.join(texts) + '\n', encoding='utf-8')
    return path


def build_pcre2(run, scratch, label, cmake_flags):
    source = scratch / 'pcre2'
    if not source.exists():
        run('pcre2-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                            'https://github.com/PCRE2Project/pcre2.git', source])
        run('pcre2-checkout', ['git', 'checkout', '--detach', plat.PCRE2], source)
    build, prefix = scratch / f'pcre2-build-{label}', scratch / f'pcre2-{label}'
    run(f'pcre2-configure-{label}', ['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release', *cmake_flags,
                                     '-DBUILD_SHARED_LIBS=OFF', '-DPCRE2_STATIC_PIC=ON', '-DPCRE2_BUILD_PCRE2_8=ON',
                                     '-DPCRE2_BUILD_PCRE2_16=OFF', '-DPCRE2_BUILD_PCRE2_32=OFF',
                                     '-DPCRE2_BUILD_PCRE2GREP=OFF', '-DPCRE2_BUILD_TESTS=OFF',
                                     '-DCMAKE_INSTALL_PREFIX=' + str(prefix)])
    run(f'pcre2-build-{label}', ['cmake', '--build', build, '--parallel', '4'])
    run(f'pcre2-install-{label}', ['cmake', '--install', build])
    library = next(p for p in [prefix / 'lib/libpcre2-8.a', prefix / 'lib64/libpcre2-8.a'] if p.exists())
    return prefix / 'include', library


def build_spacy(run, scratch, label, cmake_flags, pcre_include, pcre_library, targets):
    build = scratch / f'spacy-build-{label}'
    run(f'spacy-configure-{label}', ['cmake', '-S', ROOT / 'tools/spacy', '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                                     *cmake_flags, '-DSPACY_PCRE2_INCLUDE_DIR=' + str(pcre_include),
                                     '-DSPACY_PCRE2_LIBRARY=' + str(pcre_library)])
    for target in targets:
        run(f'spacy-build-{label}-{target}', ['cmake', '--build', build, '--parallel', '4', '--target', target])
    return build


def notices(destination):
    licenses = destination / 'licenses'
    licenses.mkdir(parents=True)
    shutil.copyfile(ROOT / 'LICENSE', licenses / 'CrispEmbed-LICENSE')
    shutil.copyfile(ROOT / 'ggml/LICENSE', licenses / 'ggml-LICENSE')
    shutil.copyfile(ROOT / 'tools/spacy/THIRD_PARTY_NOTICES', licenses / 'THIRD_PARTY_NOTICES')
    shutil.copytree(ROOT / 'tools/spacy/notices', licenses / 'notices')
    shutil.copyfile(ROOT / 'src/spacy_pipeline.h', destination / 'spacy_pipeline.h')


def deterministic_tar(source, archive):
    def clean(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        return info
    with tarfile.open(archive, 'w:gz', format=tarfile.USTAR_FORMAT, compresslevel=9) as tar:
        for path in sorted(source.rglob('*')):
            tar.add(path, arcname=path.relative_to(source).as_posix(), recursive=False, filter=clean)
    return {'sha256': digest(archive), 'bytes': archive.stat().st_size}


def compare(run, out, smoke_output, oracle_dir, label):
    report = out / f'compare-{label}.json'
    subprocess.run([sys.executable, ROOT / 'tests/spacy_smoke_compare.py', '--smoke', smoke_output,
                    '--oracle', oracle_dir / 'oracle.json', '--ner-oracle', oracle_dir / 'ner-oracle.json',
                    '--report', report], check=False)
    data = json.loads(report.read_text())
    data['passed'] = data['exact_cases'] == data['cases'] and not data['differences']
    return data


def publish_sdk(scratch, out, report, target, archive, receipt_data):
    """One Hub commit (archive + receipt), fresh readback, then catalog/card."""
    assert os.environ.get('GITHUB_REF') == 'refs/heads/main', 'Hub writes only from main'
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'], check=True)
    from huggingface_hub import CommitOperationAdd, HfApi
    token = os.environ.get('HF_PUBLISH_TOKEN')
    assert token, 'Hosted Hub write credential unavailable'
    hub = HfApi(token=token)
    card = plat._load('spacy_model_card', ROOT / 'tools/spacy/model_card.py')
    tracked = json.loads((ROOT / 'tools/spacy/hub_catalog.json').read_text())
    source = report['source']
    prefix = f'sdk/{target}/{source}'
    receipt = scratch / 'receipt.json'
    receipt.write_text(json.dumps(receipt_data, indent=2) + '\n')

    def commit(uploads, message):
        before = set(hub.list_repo_files(common.REPOSITORY))
        result = hub.create_commit(repo_id=common.REPOSITORY, repo_type='model', commit_message=message,
                                   operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(l))
                                               for p, l in uploads.items()])
        revision = result.oid
        assert re.fullmatch('[0-9a-f]{40}', revision)
        assert set(hub.list_repo_files(common.REPOSITORY, revision=revision)) == before | set(uploads)
        fresh = scratch / f'fresh-{revision[:12]}'
        readback = {}
        for path, local in uploads.items():
            target_path = fresh / path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/{common.REPOSITORY}/resolve/{revision}/' + urllib.parse.quote(path, safe='/')
            readback[path] = common.download(url, target_path, digest(local), local.stat().st_size)
        return revision, readback

    uploads = {f'{prefix}/{archive.name}': archive, f'{prefix}/receipt.json': receipt}
    revision, readback = commit(uploads, f'Publish {target} spaCy SDK built from CrispEmbed {source[:12]}')
    report['publication'] = {'revision': revision, 'fresh_readback': readback}
    entry = {'target': target, 'status': 'published', 'revision': revision, 'path': f'{prefix}/{archive.name}',
             'sha256': digest(archive), 'bytes': archive.stat().st_size, 'receipt': f'{prefix}/receipt.json',
             'contents': receipt_data['contents'], 'verification': receipt_data['verification'],
             'grammar': 'Not qualified'}
    catalog = plat.current_hub_catalog(tracked)
    catalog['sdks'] = [s for s in catalog.get('sdks', []) if s['target'] != target] + [entry]
    catalog['sdks'].sort(key=lambda s: s['target'])
    payload = scratch / 'card'
    payload.mkdir()
    (payload / plat.HUB_CATALOG).write_text(json.dumps(catalog, indent=2) + '\n')
    (payload / 'README.md').write_text(card.render(catalog))
    card_revision, card_readback = commit({plat.HUB_CATALOG: payload / plat.HUB_CATALOG,
                                           'README.md': payload / 'README.md'}, 'Update model card and SDK catalog')
    report['card_publication'] = {'revision': card_revision, 'fresh_readback': card_readback}
    report['catalog_entry'] = entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['oracle', 'ios', 'android-build', 'android-finish'])
    parser.add_argument('--oracle-dir', type=Path)
    parser.add_argument('--run-dir', type=Path, help='android: directory shared with the emulator step')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / f'spacy-mobile-{args.command}'
    scratch.mkdir(parents=True, exist_ok=True)
    run = Runner(out)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = {'schema': 'crispembed.spacy.mobile_sdk.v1', 'command': args.command, 'source': source,
              'run': os.environ.get('GITHUB_RUN_ID'), 'passed': False, 'grammar_qualified': False,
              'consumed_acceptance_or_grammar_panels_replayed': False}
    report_path = out / f'mobile-{args.command}.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    try:
        if args.command == 'oracle':
            assert sys.platform == 'linux' and host.machine() == 'x86_64'
            target = args.oracle_dir
            target.mkdir(parents=True, exist_ok=True)
            wheel = scratch / common.WHEEL
            common.download('https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/'
                            + common.WHEEL, wheel, common.WHEEL_SHA, 14639490)
            venv = scratch / 'reference'
            run('venv', [sys.executable, '-m', 'venv', venv])
            python = venv / 'bin/python'
            run('reference-install', [python, '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes',
                                      '-r', ROOT / 'tools/spacy/requirements-reference.txt'])
            run('oracle', [python, ROOT / 'tools/spacy/ggml_ab_oracle.py', '--wheel', wheel, '--corpus', CORPUS,
                           '--output', target / 'oracle.json'])
            run('ner-oracle', [python, ROOT / 'tests/spacy_ner_ab.py', 'oracle', '--wheel', wheel,
                               '--corpus', CORPUS, '--output', target / 'ner-oracle.json'])
            report['oracle_sha256'] = {name: digest(target / name) for name in ('oracle.json', 'ner-oracle.json')}
        elif args.command == 'ios':
            assert sys.platform == 'darwin' and host.machine() == 'arm64'
            bundle, ner = model_files(scratch)
            texts = corpus_lines(scratch / 'texts.txt')
            slices = {}
            for label, sdk in (('ios-arm64', 'iphoneos'), ('ios-arm64-simulator', 'iphonesimulator')):
                sysroot = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-path'], text=True).strip()
                flags = ['-DCMAKE_SYSTEM_NAME=iOS', '-DCMAKE_OSX_ARCHITECTURES=arm64',
                         '-DCMAKE_OSX_DEPLOYMENT_TARGET=' + IOS_DEPLOYMENT, '-DCMAKE_OSX_SYSROOT=' + sysroot]
                include, pcre = build_pcre2(run, scratch, label, flags)
                targets = ['crispembed-spacy'] + (['spacy-smoke'] if sdk == 'iphonesimulator' else [])
                build = build_spacy(run, scratch, label, flags + ['-DCRISPEMBED_SPACY_STATIC=ON',
                                                                   '-DBUILD_TESTING=OFF'],
                                    include, pcre, targets)
                parts = [next(build.rglob('libcrispembed-spacy.a'))]
                parts += [next(build.rglob(name)) for name in GGML_ARCHIVES]
                parts.append(pcre)
                combined = scratch / f'{label}/libcrispembed-spacy.a'
                combined.parent.mkdir(parents=True)
                run(f'libtool-{label}', ['xcrun', 'libtool', '-static', '-o', combined, *parts])
                slices[label] = {'archive': combined, 'build': build, 'members': [p.name for p in parts]}
            # Run inside a booted simulator: same static library, real iOS runtime.
            devices = json.loads(subprocess.check_output(['xcrun', 'simctl', 'list', 'devices', 'available', '-j']))
            udid = next(d['udid'] for runtime, items in sorted(devices['devices'].items(), reverse=True)
                        if 'iOS' in runtime for d in items if d['name'].startswith('iPhone'))
            subprocess.run(['xcrun', 'simctl', 'boot', udid], check=False)
            run('simctl-bootstatus', ['xcrun', 'simctl', 'bootstatus', udid, '-b'])
            smoke = next(slices['ios-arm64-simulator']['build'].rglob('spacy-smoke'))
            output = out / 'ios-simulator-smoke.jsonl'
            run('simulator-smoke', ['xcrun', 'simctl', 'spawn', udid, smoke,
                                    *smoke_arguments(bundle, ner, texts)], capture=output)
            report['simulator'] = {'udid_runtime': next(r for r, items in devices['devices'].items()
                                                        if any(d['udid'] == udid for d in items)),
                                   'comparison': compare(run, out, output, args.oracle_dir, 'ios-simulator')}
            save()
            assert report['simulator']['comparison']['passed'], 'iOS Simulator output differs from original spaCy'
            sdk_root = scratch / 'sdk'
            headers = scratch / 'headers'
            headers.mkdir()
            shutil.copyfile(ROOT / 'src/spacy_pipeline.h', headers / 'spacy_pipeline.h')
            xcframework = sdk_root / 'CrispEmbedSpacy.xcframework'
            sdk_root.mkdir()
            run('xcframework', ['xcodebuild', '-create-xcframework',
                                '-library', slices['ios-arm64']['archive'], '-headers', headers,
                                '-library', slices['ios-arm64-simulator']['archive'], '-headers', headers,
                                '-output', xcframework])
            notices(sdk_root)
            archive = scratch / 'crispembed-spacy-ios-xcframework.tar.gz'
            report['archive'] = deterministic_tar(sdk_root, archive)
            report['contents'] = ('CrispEmbedSpacy.xcframework (ios-arm64 + ios-arm64-simulator static, '
                                  f'iOS {IOS_DEPLOYMENT}+; crispembed-spacy + ggml + PCRE2), spacy_pipeline.h, licenses')
            report['verification'] = (f"iOS Simulator ({report['simulator']['udid_runtime'].split('.')[-1]}) run "
                                      f"{report['run']}: 60/60 development texts, "
                                      f"{report['simulator']['comparison']['tokens']} tokens and "
                                      f"{report['simulator']['comparison']['entities']['matched']} entities identical "
                                      'to original spaCy. Device slice built, not run on hardware.')
            if args.publish:
                publish_sdk(scratch, out, report, 'ios', archive, {k: report[k] for k in (
                    'source', 'run', 'archive', 'contents', 'verification', 'simulator')})
        elif args.command == 'android-build':
            ndk = Path(os.environ.get('ANDROID_NDK_LATEST_HOME') or os.environ['ANDROID_NDK_HOME'])
            run_dir = args.run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            bundle, ner = model_files(scratch)
            for name in ('model.gguf', 'resources.json'):
                shutil.copyfile(bundle / name, run_dir / name)
            shutil.copyfile(ner, run_dir / 'ner.gguf')
            corpus_lines(run_dir / 'texts.txt')
            (run_dir / 'hashes.env').write_text(
                f"MODEL_SHA={digest(bundle / 'model.gguf')}\nRESOURCES_SHA={digest(bundle / 'resources.json')}\n"
                f"NER_SHA={NER_SHA}\n")
            report['ndk'] = ndk.name
            for abi in ('arm64-v8a', 'x86_64'):
                flags = ['-DCMAKE_TOOLCHAIN_FILE=' + str(ndk / 'build/cmake/android.toolchain.cmake'),
                         '-DANDROID_ABI=' + abi, '-DANDROID_PLATFORM=android-' + ANDROID_API]
                include, pcre = build_pcre2(run, scratch, abi, flags)
                build = build_spacy(run, scratch, abi, flags + ['-DBUILD_TESTING=OFF'],
                                    include, pcre, ['crispembed-spacy', 'spacy-smoke'])
                target = run_dir / abi
                target.mkdir()
                shutil.copyfile(next(build.rglob('libcrispembed-spacy.so')), target / 'libcrispembed-spacy.so')
                shutil.copyfile(next(build.rglob('spacy-smoke')), target / 'spacy-smoke')
                readelf = next(ndk.rglob('llvm-readelf'))
                dynamic = subprocess.check_output([str(readelf), '-d', str(target / 'libcrispembed-spacy.so')], text=True)
                report.setdefault('elf_needed', {})[abi] = re.findall(r'\(NEEDED\).*\[(.+?)\]', dynamic)
                assert not [n for n in report['elf_needed'][abi] if 'pcre' in n], report['elf_needed'][abi]
        elif args.command == 'android-finish':
            run_dir = args.run_dir
            build_report = json.loads((out / 'mobile-android-build.json').read_text())
            output = run_dir / 'emulator-smoke-x86_64.jsonl'
            assert output.exists() and output.stat().st_size > 0, 'Emulator produced no smoke output'
            report['emulator'] = {'abi': 'x86_64', 'api': (run_dir / 'emulator-api.txt').read_text().strip(),
                                  'comparison': compare(run, out, output, args.oracle_dir, 'android-emulator')}
            save()
            assert report['emulator']['comparison']['passed'], 'Android emulator output differs from original spaCy'
            sdk_root = scratch / 'sdk'
            for abi in ('arm64-v8a', 'x86_64'):
                (sdk_root / 'jniLibs' / abi).mkdir(parents=True)
                shutil.copyfile(run_dir / abi / 'libcrispembed-spacy.so', sdk_root / 'jniLibs' / abi / 'libcrispembed-spacy.so')
            notices(sdk_root)
            archive = scratch / 'crispembed-spacy-android-jnilibs.tar.gz'
            report['archive'] = deterministic_tar(sdk_root, archive)
            report['elf_needed'] = build_report.get('elf_needed')
            report['contents'] = (f'jniLibs/{{arm64-v8a,x86_64}}/libcrispembed-spacy.so (API {ANDROID_API}+, '
                                  f'NDK {build_report.get("ndk")}, static PCRE2, static libc++), '
                                  'spacy_pipeline.h, licenses')
            report['verification'] = (f"Android emulator x86_64 (API {report['emulator']['api']}) run {report['run']}: "
                                      f"60/60 development texts, {report['emulator']['comparison']['tokens']} tokens and "
                                      f"{report['emulator']['comparison']['entities']['matched']} entities identical to "
                                      'original spaCy. arm64-v8a built from the same source, not run.')
            if args.publish:
                publish_sdk(scratch, out, report, 'android', archive, {k: report[k] for k in (
                    'source', 'run', 'archive', 'contents', 'verification', 'emulator', 'elf_needed')})
        report['passed'] = True
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
