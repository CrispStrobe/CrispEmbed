#!/usr/bin/env python3
"""Hosted CPU-only quantization assessment, preserving accepted F32 artifacts."""
from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import spacy_w7_publish as common

ROOT = common.ROOT
sys.path.insert(0, str(ROOT / 'tools/spacy'))
import experimental_quantization as experiment


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    assert os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert sys.platform == 'linux' and platform.machine() == 'x86_64'
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'w7-quantization'
    scratch.mkdir()
    out = Path(os.environ['HEAVY_OUT'])
    revision = '659dae33997d00296ae12b76510fa4dfe26540e0'
    report = {'schema': 'crispembed.spacy.quantization.assessment.v1', 'execution_passed': False,
              'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'Linux x86-64 CPU experimental storage/dequantization; scalar F32 compute unchanged',
              'baseline_replaced': False, 'production_sources_modified': False,
              'consumed_acceptance_or_W6_replayed': False, 'Kaggle_kernel_IDs': [],
              'Kaggle_used': False, 'reason_no_Kaggle': 'This standalone CPU backend has no real GPU inference path.',
              'qualified_candidates': [], 'candidates': [], 'stage': 'preflight'}

    def save():
        (out / 'quantization-assessment.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, cwd=ROOT):
        report['stage'] = label
        save()
        with (out / (label + '.log')).open('w') as log:
            subprocess.run([str(x) for x in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        # Install only hosted build prerequisites; no shared-host compilation.
        run('apt', ['sudo', 'apt-get', 'update', '-qq'])
        run('pcre', ['sudo', 'apt-get', 'install', '-y', 'libpcre2-dev'])
        run('numpy', [sys.executable, '-m', 'pip', 'install', 'numpy==2.4.6'])
        bundle = scratch / 'baseline-bundle'
        bundle.mkdir()
        report['fresh_baseline_files'] = {}
        for name, pin in common.load_distribution().BUNDLE_FILES.items():
            report['fresh_baseline_files'][name] = common.download(
                f'https://huggingface.co/{common.REPOSITORY}/resolve/{revision}/bundle/{name}',
                bundle / name, pin['sha256'], pin['size_bytes'])
        quds = scratch / 'quds-reference'
        run('quds-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
                          'https://github.com/CrispStrobe/quds_office_suite.git', quds])
        run('quds-sparse', ['git', 'sparse-checkout', 'set', 'tools/proofing/spacy'], quds)
        run('quds-checkout', ['git', 'checkout', '--detach', 'dcf2ec9b4d6fba05fa5eec2634c6654266332cc9'], quds)
        generated = scratch / 'experimental-source'
        report['generated_sources_sha256'] = experiment.generate(ROOT, generated)
        report['unchanged_production_sources'] = {name: common.digest(ROOT / name) for name in (
            'src/spacy_pipeline.cpp', 'src/spacy_numeric.cpp', 'src/spacy_lexical.cpp', 'src/spacy_decode.cpp',
            'tools/quantize.cpp', 'tests/spacy_parity.py', 'tests/spacy_text_parity.py', 'src/crispembed_diff.h')}
        build = scratch / 'build'
        run('configure', ['cmake', '-S', generated, '-B', build, '-DCMAKE_BUILD_TYPE=Release'])
        run('build', ['cmake', '--build', build, '--parallel', '2'])
        run('model-free-contracts', ['ctest', '--test-dir', build, '--output-on-failure'])
        library = build / 'libcrispembed-spacy.so'
        report['experimental_library'] = {'sha256': common.digest(library), 'bytes': library.stat().st_size}
        report['compiler'] = subprocess.check_output(['c++', '--version'], text=True)
        report['cpu'] = Path('/proc/cpuinfo').read_text().split('model name', 1)[-1].splitlines()[0]
        report['dependencies'] = subprocess.check_output(['ldd', str(library)], text=True)
        # The private alternate library must preserve the F32 development gate
        # with the experimental storage gate OFF before a candidate is opened.
        os.environ.pop('CRISPEMBED_SPACY_QUANTIZED', None)
        for script, label in [('spacy_parity.py', 'f32-prepared'), ('spacy_text_parity.py', 'f32-text')]:
            run(label, [sys.executable, ROOT / 'tests' / script, '--library', library,
                '--bundle', bundle, '--quds-root', quds, '--report', out / (label + '.json')])
        payload = scratch / 'candidate-artifacts'
        payload.mkdir()
        os.environ['CRISPEMBED_SPACY'] = '1'
        quantizer = build / 'crispembed-quantize'
        quantizer_sha = common.digest(quantizer)
        for recipe in ('f16', 'q8_0', 'q4_0'):
            candidate_bundle = scratch / (recipe + '-bundle')
            candidate_bundle.mkdir()
            for name in ('resources.json', 'LICENSE', 'LICENSES_SOURCES'):
                shutil.copyfile(bundle / name, candidate_bundle / name)
            model = candidate_bundle / 'model.gguf'
            run(recipe + '-quantize', [quantizer, bundle / 'model.gguf', model, recipe,
                                       '--metadata-string', 'spacy.precision=' + recipe])
            proof = {'recipe': recipe, 'source_f32_sha256': common.digest(bundle / 'model.gguf'),
                'candidate_sha256': common.digest(model), 'candidate_bytes': model.stat().st_size,
                'quantizer_binary_sha256': quantizer_sha, 'quantizer_generated_source_sha256': report['generated_sources_sha256']['quantize.cpp'],
                'experimental_generated_loader_sha256': report['generated_sources_sha256']['spacy_pipeline.cpp'],
                'production_loader_sha256': experiment.PIPELINE_SHA, 'source_commit': report['source']}
            manifest = json.loads((bundle / 'manifest.json').read_text())
            manifest['files']['model.gguf'] = {'sha256': proof['candidate_sha256'], 'size_bytes': proof['candidate_bytes']}
            (candidate_bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            proof_path = out / (recipe + '-proof.json')
            proof_path.write_text(json.dumps(proof, indent=2) + '\n')
            # Prove actual native loader rejects each candidate with the
            # default/OFF spellings before explicitly opting into the experiment.
            lib = C.CDLL(str(library))
            lib.crispembed_spacy_create.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]
            lib.crispembed_spacy_create.restype = C.c_void_p
            for gate in (None, '', '0', 'false', 'OFF', 'No'):
                if gate is None:
                    os.environ.pop('CRISPEMBED_SPACY_QUANTIZED', None)
                else:
                    os.environ['CRISPEMBED_SPACY_QUANTIZED'] = gate
                error = C.create_string_buffer(1024)
                assert not lib.crispembed_spacy_create(os.fsencode(model), proof['candidate_sha256'].encode(), error, len(error))
                assert b'storage requires explicit opt-in' in error.value, error.value
            result_dir = out / recipe
            run(recipe + '-parity', [sys.executable, ROOT / 'tools/spacy/quantized_parity.py',
                '--library', library, '--baseline-bundle', bundle, '--bundle', candidate_bundle,
                '--quds-root', quds, '--candidate-proof', proof_path, '--diff-library', build / 'libspacy-diff.so',
                '--output', result_dir])
            result = json.loads((result_dir / 'summary.json').read_text())
            report['candidates'].append({**proof, **result, 'gate_OFF_controls_passed': 6})
            if result['qualified']:
                report['qualified_candidates'].append(recipe)
            # Preserve rejected candidates for independent analysis, never
            # replace the accepted model, alias or Linux package.
            shutil.copyfile(model, payload / (recipe + '.gguf'))
        report['execution_passed'] = True
        save()
        shutil.copyfile(out / 'quantization-assessment.json', payload / 'quantization-assessment.json')
        for recipe in ('f16', 'q8_0', 'q4_0'):
            for name in ('prepared.json', 'text.json', 'summary.json'):
                shutil.copyfile(out / recipe / name, payload / (recipe + '-' + name))
        (payload / 'README.md').write_text('# Experimental quantization attempts\n\n'
            'Qualification is recorded per candidate in quantization-assessment.json. '
            'These are development diagnostics, not replacements for the accepted F32 capsule or app distributions.\n')
        run('hub-install', [sys.executable, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'])
        from huggingface_hub import HfApi
        hub = HfApi(token=os.environ.get('HF_PUBLISH_TOKEN') or os.environ.get('HF_TOKEN_2'))
        prefix = 'quantization-attempts/' + report['source']
        commit = hub.upload_folder(repo_id=common.REPOSITORY, folder_path=str(payload), path_in_repo=prefix,
                                   commit_message='Retain experimental quantization attempts; accepted F32 unchanged').oid
        fresh = scratch / 'fresh-readback'
        fresh.mkdir()
        receipt = {'revision': commit, 'files': {}}
        for file in payload.iterdir():
            receipt['files'][file.name] = common.download(
                f'https://huggingface.co/{common.REPOSITORY}/resolve/{commit}/{prefix}/{file.name}',
                fresh / file.name, common.digest(file), file.stat().st_size)
        (out / 'quantization-hosted-artifacts.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()
        (out / 'summary.md').write_text('# W7 experimental quantization\n\n' + json.dumps({key: report.get(key)
            for key in ('stage', 'execution_passed', 'qualified_candidates', 'failure')}, indent=2) +
            '\n\nF32 capsule remains accepted and unchanged. No GPU, consumed held-out split, app activation or speed claim.\n')


if __name__ == '__main__':
    main()
