#!/usr/bin/env python3
"""One-shot new authored panel; never reruns the consumed W6 panel."""
from __future__ import annotations

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


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    assert os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert sys.platform == 'linux' and platform.machine() == 'x86_64'
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'w7-new-coverage'
    scratch.mkdir()
    out = Path(os.environ['HEAVY_OUT'])
    fixture = ROOT / 'docs/validation/native-spacy-w7-20261118/owner-execution/coverage-freeze'
    freeze = json.loads((fixture / 'freeze.json').read_text())
    for name, sha in freeze['new_files_sha256'].items():
        assert common.digest(fixture / name) == sha
    report = {'schema': 'crispembed.spacy.w7.coverage.execution.v1', 'passed': False,
              'scope': 'New finite authored panel, Linux x86-64/AVX2 CPU/F32 only',
              'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'quds_source': freeze['quds_commit'], 'new_panel_sha256': freeze['panel_sha256'],
              'consumed_W6_rerun': False, 'native_acceptance_rerun': False,
              'new_panel_started': False, 'python': sys.version, 'stage': 'preflight'}

    def save():
        (out / 'coverage-execution.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, cwd=ROOT):
        report['stage'] = label
        save()
        with (out / (label + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        # PCRE2 is the same source revision as the frozen candidate's dependency.
        # Different compiler/system bytes are captured, not silently called the
        # original six dependency identities or universal Linux portability.
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
        dist = common.load_distribution()
        installation = dist.install(package, common.PACKAGE_SHA, cache)
        library = Path(installation['entrypoints']['library'])
        bundle = Path(installation['entrypoints']['bundle'])
        report['installation'] = installation
        resolved = subprocess.check_output(['ldd', str(library)], text=True)
        report['actual_dependencies'] = {'resolution': resolved, 'pcre2_source': PCRE2,
            'original_six_dependency_identity_claim': False, 'files': {}}
        for line in resolved.splitlines():
            values = line.split()
            path = values[2] if '=>' in values and len(values) > 2 else values[0] if values else ''
            if path.startswith('/') and Path(path).is_file():
                report['actual_dependencies']['files'][Path(path).name] = common.digest(path)
        quds = scratch / 'quds'
        run('quds-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                          'https://github.com/CrispStrobe/quds_office_suite.git', quds])
        run('quds-checkout', ['git', 'checkout', '--detach', freeze['quds_commit']], quds)
        assert common.digest(quds / 'packages/quds_proofing_spacy/tool/evaluate_grammar.dart') == freeze['source_evaluator_sha256']
        run('numpy', [sys.executable, '-m', 'pip', 'install', 'numpy==2.4.6'])
        # New host first proves the same frozen W1 DEVELOPMENT stages and text;
        # this is not the consumed native/W6 held-out split.
        for script, label in [('spacy_parity.py', 'prepared-development'), ('spacy_text_parity.py', 'text-development')]:
            run(label, [sys.executable, ROOT / 'tests' / script, '--library', library, '--bundle', bundle,
                        '--quds-root', quds, '--report', out / (label + '.json')])
        flutter = scratch / 'flutter'
        run('flutter-clone', ['git', 'clone', '--depth', '1', '--branch', '3.47.8',
                             'https://github.com/flutter/flutter.git', flutter])
        os.environ['PATH'] = str(flutter / 'bin') + os.pathsep + os.environ['PATH']
        os.environ['PUB_CACHE'] = str(scratch / 'pub-cache')
        run('flutter-pub-get', [flutter / 'bin/flutter', 'pub', 'get'], quds)
        target = quds / 'docs/validation/native-spacy-w7-extension-20261118'
        target.mkdir()
        for name in ('panel.json', 'protocol.json', 'freeze.json'):
            shutil.copyfile(fixture / name, target / name)
        driver = quds / 'packages/quds_proofing_spacy/tool/evaluate_w7_coverage.dart'
        shutil.copyfile(fixture / driver.name, driver)
        # Compile and verify source closure before consuming the new panel.
        run('compile-new-harness', [flutter / 'bin/dart', 'compile', 'exe', driver,
                                   '-o', scratch / 'evaluate-w7'], quds)
        seal_helper = driver.with_name('seal_w7.dart')
        seal_helper.write_text("import 'dart:convert';\nimport 'dart:io';\nimport 'evaluate_w7_coverage.dart' as e;\n"
            "void main(List<String> args) { final root=args[0]; final p=jsonDecode(File('$root/" +
            "docs/validation/native-spacy-w7-extension-20261118/protocol.json').readAsStringSync()); "
            "e.verifyBaseline(root,p); stdout.write(jsonEncode(e.sourceHashes(root))); }\n")
        source_map = json.loads(subprocess.check_output([str(flutter / 'bin/dart'), str(seal_helper), str(quds)], cwd=quds, text=True))
        seal = {'schema': 'quds.spacy.grammar.candidate.v1', 'qualification_panel_sha256': freeze['panel_sha256'],
                'native_library_sha256': 'ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977',
                'source_sha256': source_map, 'candidate_commit': freeze['quds_commit'],
                'new_harness_source_commit': report['source']}
        seal_path = target / 'candidate-seal.json'
        seal_path.write_text(json.dumps(seal, indent=2) + '\n')
        shutil.copyfile(seal_path, out / 'new-panel-seal.json')
        shutil.copyfile(fixture / 'freeze.json', out / 'new-panel-freeze.json')
        # The committed freeze predates this dispatch; the exclusive claim in
        # the unchanged scorer is created before the first model opening.
        report['new_panel_started'] = True
        save()
        os.environ['CRISPEMBED_SPACY'] = '1'
        run('new-panel', [scratch / 'evaluate-w7', '--root', quds, '--phase', 'qualification',
            '--panel', target / 'panel.json', '--seal', seal_path, '--library', library,
            '--bundle', bundle, '--report', out / 'new-panel-result.json'], quds)
        result = json.loads((out / 'new-panel-result.json').read_text())
        report['summary'] = result['summary']
        report['passed'] = result['passed']
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()
        (out / 'summary.md').write_text('# W7 new finite coverage panel\n\n' +
            json.dumps({key: report.get(key) for key in ('stage', 'passed', 'new_panel_started', 'summary', 'failure')}, indent=2) +
            '\n\nOriginal W6 held-out split was not rerun; no population-level grammar or app activation claim.\n')


if __name__ == '__main__':
    main()
