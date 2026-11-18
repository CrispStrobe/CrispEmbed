#!/usr/bin/env python3
"""One gated AVX2/F32 candidate; full unchanged development gates precede timing."""
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import spacy_w7_publish as common

ROOT = common.ROOT
sys.path.insert(0, str(ROOT / 'tools/spacy'))
import experimental_quantization as experiment


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true'
    assert os.environ.get('GITHUB_REPOSITORY') == 'CrispStrobe/CrispEmbed'
    assert sys.platform == 'linux' and platform.machine() == 'x86_64'
    assert 'avx2' in Path('/proc/cpuinfo').read_text().split('flags', 1)[1].splitlines()[0].split()
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'w7-optimized-f32'
    scratch.mkdir()
    out = Path(os.environ['HEAVY_OUT'])
    report = {'schema': 'crispembed.spacy.w7.optimized-f32.v1', 'execution_passed': False,
        'source': os.environ['GITHUB_SHA'], 'scope': 'Private Linux x86-64/AVX2 F32 affine reduction candidate only',
        'production_F32_sources_or_capsule_changed': False, 'default_gate': 'OFF',
        'consumed_native_W6_W7_heldout_replayed': False, 'speed_claim': False,
        'promoted_profile': False, 'Kaggle_kernel_IDs': [], 'stage': 'preflight'}

    def save():
        (out / 'optimized-f32.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command):
        report['stage'] = label
        save()
        with (out / (label + '.log')).open('w') as log:
            return subprocess.run([str(x) for x in command], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        run('apt', ['sudo', 'apt-get', 'update', '-qq'])
        run('pcre', ['sudo', 'apt-get', 'install', '-y', 'libpcre2-dev'])
        run('numpy', [sys.executable, '-m', 'pip', 'install', 'numpy==2.4.6'])
        bundle = scratch / 'bundle'
        bundle.mkdir()
        report['fresh_bundle'] = {}
        for name, pin in common.load_distribution().BUNDLE_FILES.items():
            report['fresh_bundle'][name] = common.download(
                f'https://huggingface.co/{common.REPOSITORY}/resolve/659dae33997d00296ae12b76510fa4dfe26540e0/bundle/{name}',
                bundle / name, pin['sha256'], pin['size_bytes'])
        quds = scratch / 'reference'
        run('reference-clone', ['git', 'clone', '--filter=blob:none', '--no-checkout',
            'https://github.com/CrispStrobe/quds_office_suite.git', quds])
        subprocess.run(['git', 'sparse-checkout', 'set', 'tools/proofing/spacy'], cwd=quds, check=True)
        subprocess.run(['git', 'checkout', '--detach', 'dcf2ec9b4d6fba05fa5eec2634c6654266332cc9'], cwd=quds, check=True)
        generated = scratch / 'source'
        report['generated_sources_sha256'] = experiment.generate(ROOT, generated, optimized=True)
        build = scratch / 'build'
        run('configure', ['cmake', '-S', generated, '-B', build, '-DCMAKE_BUILD_TYPE=Release'])
        run('build', ['cmake', '--build', build, '--parallel', '2'])
        run('model-free-controls', ['ctest', '--test-dir', build, '--output-on-failure'])
        library = build / 'libcrispembed-spacy.so'
        report['library_sha256'] = common.digest(library)
        report['compiler'] = subprocess.check_output(['c++', '--version'], text=True)
        report['cpu'] = Path('/proc/cpuinfo').read_text().split('model name', 1)[1].splitlines()[0]
        # Never touch the accepted source/binary, model, reference or budgets.
        # The same private library keeps its exact scalar alternative OFF.
        os.environ.pop('CRISPEMBED_SPACY_QUANTIZED', None)
        outcomes = {}
        for mode in ('scalar', 'fast'):
            os.environ['CRISPEMBED_SPACY_FAST_DOT'] = '1' if mode == 'fast' else '0'
            outcomes[mode] = {}
            for script, label in [('spacy_parity.py', 'prepared'), ('spacy_text_parity.py', 'text')]:
                target = out / (mode + '-' + label + '.json')
                report['stage'] = mode + '-' + label
                save()
                with (out / (mode + '-' + label + '.log')).open('w') as log:
                    result = subprocess.run([sys.executable, str(ROOT / 'tests' / script),
                        '--library', str(library), '--bundle', str(bundle), '--quds-root', str(quds),
                        '--report', str(target)], stdout=log, stderr=subprocess.STDOUT)
                assert target.is_file(), 'Comparator did not retain an actual report'
                data = json.loads(target.read_text())
                outcomes[mode][label] = {'passed': data['passed'], 'counts': data['counts'],
                    'exit_code': result.returncode, 'report_sha256': common.digest(target)}
            if mode == 'scalar':
                assert all(item['passed'] for item in outcomes[mode].values()), 'Scalar F32 control failed'
        report['development_gates'] = outcomes
        report['optimized_development_passed'] = all(item['passed'] for item in outcomes['fast'].values())
        if report['optimized_development_passed']:
            sys.path.insert(0, str(ROOT / 'tests'))
            from test_spacy_text_api import NativeAPI, validate_bundle
            os.environ['CRISPEMBED_SPACY'] = '1'
            api = NativeAPI(library, bundle, validate_bundle(bundle))
            context, error = api.open()
            assert context, error
            # New operational timing input; not a consumed grammar/acceptance
            # example, and not an independently labeled linguistic benchmark.
            data = ('Die freundlichen Hunde warten ruhig vor dem Haus. ' * 8).encode()
            latencies = {'scalar': [], 'fast': []}
            baseline_tokens = None
            start_rss = None
            rss = []
            try:
                for index in range(13):
                    modes = ('scalar', 'fast') if index % 2 == 0 else ('fast', 'scalar')
                    for mode in modes:
                        os.environ['CRISPEMBED_SPACY_FAST_DOT'] = '1' if mode == 'fast' else '0'
                        started = time.perf_counter()
                        document = None
                        try:
                            status, document, error = api.call(context, data, analyze=True)
                            assert status == 0 and document, error
                            tokens = api.tokens(document)
                            assert tokens
                            if baseline_tokens is None:
                                baseline_tokens = tokens
                            assert tokens == baseline_tokens, 'Operational decoded outputs changed'
                        finally:
                            if document:
                                api.lib.crispembed_spacy_document_free(document)
                        if index >= 3:
                            latencies[mode].append(time.perf_counter() - started)
                    if index == 2:
                        start_rss = int(Path('/proc/self/status').read_text().split('VmRSS:', 1)[1].split()[0]) * 1024
                    if index >= 3:
                        rss.append(int(Path('/proc/self/status').read_text().split('VmRSS:', 1)[1].split()[0]) * 1024)
            finally:
                api.lib.crispembed_spacy_free(context)
            medians = {mode: statistics.median(values) for mode, values in latencies.items()}
            reduction = 1 - medians['fast'] / medians['scalar']
            report['same_context_operational_timing'] = {'warmup_pairs': 3, 'measured_pairs': 10,
                'tokens_per_call': len(baseline_tokens), 'latencies_seconds': latencies,
                'median_seconds': medians, 'median_reduction_fraction': reduction,
                'rss_growth_bytes': max(rss) - start_rss,
                'all_copied_token_records_exact_between_modes': True,
                'predeclared_candidate_signal': 'At least10% median reduction with <=32MiB sampled growth; not broad performance qualification.',
                'candidate_signal_passed': reduction >= 0.10 and max(rss) - start_rss <= 32 * 1024**2,
                'limits': 'One authored operational input on one hosted CPU, fixed1native thread, alternating order; not cross-host/app/grammar proof.'}
            report['diagnostic_speed_signal'] = report['same_context_operational_timing']['candidate_signal_passed']
        else:
            report['timing_UNRUN_reason'] = 'Fast candidate fails unchanged numerical/decoded development gate; no speed claim.'
        report['execution_passed'] = True
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        raise
    finally:
        save()
        (out / 'summary.md').write_text('# W7 gated AVX2/F32 development candidate\n\n' + json.dumps({key: report.get(key)
            for key in ('stage', 'execution_passed', 'optimized_development_passed', 'speed_claim', 'failure')}, indent=2) +
            '\n\nAccepted F32 capsule unchanged; default OFF. No consumed held-out split, app, grammar or release promotion.\n')


if __name__ == '__main__':
    main()
