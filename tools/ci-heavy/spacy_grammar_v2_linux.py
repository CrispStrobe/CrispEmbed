#!/usr/bin/env python3
"""Linux x86-64 cross-binary replay of the Quds grammar v2 sealed qualification.

The one-shot qualification ran on the app-bundled macOS library (ccc96620...).
This replays the same frozen inputs (244-case panel + 977-sentence UD German-GSD
test corpus) through the qualified Linux library (ea89842e..., installed from
the immutable W7 capsule) and compares annotations, planner decisions, all
arms and corrected rechecks case by case with the retained macOS report.
It is not a new qualification and changes no native bytes.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import spacy_w7_publish as common

ROOT = common.ROOT
REVISION = '659dae33997d00296ae12b76510fa4dfe26540e0'
PCRE2 = 'b2bd4254b379b9d7dc9a3dda060a7e27009ccdff'
LINUX_LIBRARY = 'ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977'
QUDS_COMMIT = 'b7f04f12a8ca591c0e3df630027f39c1de192346'
EVIDENCE = 'docs/validation/native-spacy-grammar-v2-20261120'
RECORDED_SHA = 'c3336e439c5c7fed86e6ea6ed716169c4f141659260bcc2b185e73b7472e0861'


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    assert sys.platform == 'linux' and platform.machine() == 'x86_64'
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'grammar-v2'
    scratch.mkdir()
    out = Path(os.environ['HEAVY_OUT'])
    report = {'schema': 'crispembed.spacy.grammar-v2.linux-replay.v1', 'passed': False,
              'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'quds_commit': QUDS_COMMIT, 'recorded_macos_report_sha256': RECORDED_SHA,
              'native_bytes_changed': False, 'stage': 'preflight'}

    def save():
        (out / 'linux-replay-execution.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, cwd=ROOT, env=None):
        report['stage'] = label
        save()
        with (out / (label + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=cwd, stdout=log,
                           stderr=subprocess.STDOUT, check=True, env=env)

    try:
        pcre = scratch / 'pcre2'
        run('pcre-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                          'https://github.com/PCRE2Project/pcre2.git', pcre])
        run('pcre-checkout', ['git', 'checkout', '--detach', PCRE2], pcre)
        run('pcre-configure', ['cmake', '-S', pcre, '-B', scratch / 'pcre-build',
            '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DPCRE2_BUILD_PCRE2_16=OFF',
            '-DPCRE2_BUILD_PCRE2_32=OFF', '-DPCRE2_BUILD_TESTS=OFF', '-DPCRE2_BUILD_PCRE2GREP=OFF',
            '-DCMAKE_INSTALL_PREFIX=' + str(scratch / 'pcre-install')])
        run('pcre-build', ['cmake', '--build', scratch / 'pcre-build', '--parallel', '2'])
        run('pcre-install', ['cmake', '--install', scratch / 'pcre-build'])
        os.environ['LD_LIBRARY_PATH'] = str(scratch / 'pcre-install/lib')
        package = scratch / common.PACKAGE
        report['fresh_package'] = common.download(
            f'https://huggingface.co/{common.REPOSITORY}/resolve/{REVISION}/{common.PACKAGE}',
            package, common.PACKAGE_SHA, 7808129)
        cache = scratch / 'installed'
        cache.mkdir(mode=0o700)
        installation = common.load_distribution().install(package, common.PACKAGE_SHA, cache)
        library = Path(installation['entrypoints']['library'])
        bundle = Path(installation['entrypoints']['bundle'])
        assert common.digest(library) == LINUX_LIBRARY, 'unexpected Linux library bytes'
        report['library_sha256'] = LINUX_LIBRARY
        quds = scratch / 'quds'
        run('quds-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                          'https://github.com/CrispStrobe/quds_office_suite.git', quds])
        run('quds-checkout', ['git', 'checkout', '--detach', QUDS_COMMIT], quds)
        evidence = quds / EVIDENCE
        recorded = scratch / 'qualification-macos.json'
        recorded.write_bytes(gzip.decompress((evidence / 'qualification-macos.json.gz').read_bytes()))
        assert common.digest(recorded) == RECORDED_SHA, 'recorded macOS report differs'
        flutter = scratch / 'flutter'
        run('flutter-clone', ['git', 'clone', '--depth', '1', '--branch', '3.47.8',
                             'https://github.com/flutter/flutter.git', flutter])
        env = dict(os.environ, PATH=str(flutter / 'bin') + os.pathsep + os.environ['PATH'],
                   PUB_CACHE=str(scratch / 'pub-cache'), CRISPEMBED_SPACY='1')
        run('flutter-pub-get', [flutter / 'bin/flutter', 'pub', 'get'], quds, env)
        run('linux-replay', [flutter / 'bin/dart', 'run',
            'packages/quds_proofing_spacy/tool/evaluate_grammar_v2.dart',
            '--root', quds, '--phase', 'replay',
            '--panel', evidence / 'qualification-panel.json',
            '--corpus', evidence / 'clean-corpus-gsd-test.json',
            '--recorded', recorded, '--library', library,
            '--library-sha256', LINUX_LIBRARY, '--bundle', bundle,
            '--report', out / 'linux-replay.json'], quds, env)
        result = json.loads((out / 'linux-replay.json').read_text())
        report['case_differences'] = len(result.get('case_differences', []))
        report['cases_compared'] = result.get('cases_compared')
        report['per_family'] = result.get('per_family')
        report['passed'] = result['passed']
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()
        (out / 'summary.md').write_text('# Quds grammar v2 Linux cross-binary replay\n\n' +
            json.dumps({k: report.get(k) for k in ('stage', 'passed', 'cases_compared', 'case_differences', 'failure')}, indent=2) +
            '\n\nReplay of the sealed macOS qualification inputs on the Linux library; not a new qualification.\n')


if __name__ == '__main__':
    main()
