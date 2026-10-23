"""Hosted-only checks for a probe-free PosFormer runtime correction.

The pinned CrispMath helpers supply the unchanged corpus, native public-API
benchmark and independent exported-weight reference. Source identities are
validated against the actual upstream commits, never rewritten in reports.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

BASE = '11e6d598521976f38081934106b55095b46b40e3'
APP = '23f1470df0a0853234880ac7e0c6135fd4c02f59'
ORIGINAL_SOURCE = 'bc88ab2f68cfe992d3cc78df47d488771a237ae3ae3eb7af4d927383fa8e7f2c'
CORRECT_IDS = {'0111fa141bb73b48', '02c39c1be9d660b7', '0276c02c9b9222e9',
               '0333d9584ff7c0d0', '002ae6d5dd4173e4', '02da6f52e30f674d',
               '032278982233fefa'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


@contextlib.contextmanager
def expected_bridge(module, source):
    previous = module.BRIDGE_SOURCE
    module.BRIDGE_SOURCE = source
    try:
        yield
    finally:
        module.BRIDGE_SOURCE = previous


def compare(before, after, provenance):
    import compare_handwriting_encoders as checks
    from audit_handwriting_vocabulary import FROZEN_MANIFEST_SHA256
    source = provenance.get('bridge_source')
    require(isinstance(source, str) and re.fullmatch('[0-9a-f]{40}', source) and source != BASE,
            'Missing corrected source commit')
    require(provenance.get('baseline_source') == BASE and provenance.get('app_source') == APP,
            'Unpinned baseline or app harness')
    require(provenance.get('baseline_source_sha256') == ORIGINAL_SOURCE, 'Baseline source changed')
    with expected_bridge(checks, BASE):
        left, left_matches = checks.validate(before, 'default')
    with expected_bridge(checks, source):
        right, right_matches = checks.validate(after, 'default')
    for report, field in ((before, 'baseline_library_sha256'), (after, 'production_library_sha256')):
        require(report.get('bridge_patch_sha256') is None, 'Production normal API used a diagnostic patch')
        require(report.get('source') == source, 'Incorrect workflow source identity')
        require(report.get('corpus_manifest_sha256') == FROZEN_MANIFEST_SHA256, 'Frozen manifest changed')
        require(re.fullmatch('[0-9a-f]{64}', provenance.get(field, '')) and
                report.get('library_sha256') == provenance[field], 'Wrong normal-API library identity')
    require(provenance['baseline_library_sha256'] != provenance['production_library_sha256'],
            'Production library unchanged')
    require(provenance.get('production_probes_absent') is True, 'Probes shipped in production library')
    for key in ('corpus', 'model', 'model_sha256', 'scoring', 'threads'):
        require(before[key] == after[key], 'Paired ' + key + ' changed')
    left_ids = {row['id'] for row in left if row['status'] == 'exact_match'}
    right_ids = {row['id'] for row in right if row['status'] == 'exact_match'}
    require(left_ids == CORRECT_IDS and left_matches == 7, 'Known baseline outputs changed')
    require(left_ids <= right_ids, 'Previously correct original-model case regressed')
    rows = []
    for a, b in zip(left, right):
        for key in ('id', 'reference_latex', 'image_sha256', 'image_width', 'image_height'):
            require(a[key] == b[key], 'Paired ' + key + ' changed')
        rows.append({'id': a['id'], 'reference_latex': a['reference_latex'],
                     'baseline_latex': a['latex'], 'corrected_latex': b['latex'],
                     'baseline_status': a['status'], 'corrected_status': b['status'],
                     'token_output_equal': checks.tokens(a['latex']) == checks.tokens(b['latex'])})
    return {'format': 'crispembed.posformer-production-runtime-validation',
            'provenance': provenance, 'corpus_manifest_sha256': FROZEN_MANIFEST_SHA256,
            'samples': 50, 'baseline_exact_matches': left_matches, 'corrected_exact_matches': right_matches,
            'baseline_correct_ids': sorted(left_ids), 'corrected_correct_ids': sorted(right_ids),
            'original_correct_ids_preserved': True,
            'token_output_differences': sum(not row['token_output_equal'] for row in rows),
            'runtime_failures': 0, 'cases': rows,
            'limits': ['Frozen original-Q8 regression only; unchanged scoring and references.',
                       'Seven preserved matches do not establish general handwriting reliability.',
                       'Independent bounded exported-FP32 reference is a separate hosted test fixture.']}


def source_provenance(args):
    import ctypes
    root = Path(args.root)
    source = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(subprocess.check_output(['git', '-C', args.app_root, 'rev-parse', 'HEAD'], text=True).strip() == APP,
            'App harness checkout changed')
    require(subprocess.check_output(['git', '-C', args.baseline_root, 'rev-parse', 'HEAD'], text=True).strip() == BASE,
            'Baseline checkout changed')
    production = root / 'src/posformer_ocr.cpp'
    tracked = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:src/posformer_ocr.cpp'])
    require(production.read_bytes() == tracked, 'Modified production source')
    require(b'_probe' not in tracked and b'POSFORMER_REFERENCE' not in tracked, 'Production diagnostics found')
    lib = ctypes.CDLL(str(Path(args.production_library).resolve()))
    require(not hasattr(lib, 'crispembed_posformer_input_norm_probe') and
            not hasattr(lib, 'crispembed_posformer_pool_ceil_probe'), 'Production probe symbol exported')
    baseline = subprocess.check_output(['git', '-C', args.baseline_root, 'show',
                                        BASE + ':src/posformer_ocr.cpp'])
    require(hashlib.sha256(baseline).hexdigest() == ORIGINAL_SOURCE, 'Baseline source changed')
    write(args.output, {'bridge_source': source, 'baseline_source': BASE, 'app_source': APP,
                        'baseline_source_sha256': ORIGINAL_SOURCE,
                        'production_source_sha256': sha(production),
                        'production_library_sha256': sha(args.production_library),
                        'baseline_library_sha256': sha(args.baseline_library),
                        'production_probes_absent': True, 'validation_tool_sha256': sha(__file__)})


def prepare(args):
    from prepare_handwriting_reference_bridge import instrument
    provenance = load(args.provenance)
    root = Path(args.root)
    path = root / 'src/posformer_ocr.cpp'
    require(sha(path) == provenance['production_source_sha256'], 'Unexpected production source')
    # Probes are extracted from the pinned, previously measured diagnostic patch.
    # Only their C ABI wrappers are added; the runtime helpers remain untouched.
    diagnostic = Path(args.app_root) / 'tool/patches/posformer-forward-repair.patch'
    added = '\n'.join(line[1:] for line in diagnostic.read_text().splitlines()
                      if line.startswith('+') and not line.startswith('+++'))
    probes = []
    for name in ('pool_ceil', 'input_norm'):
        token = 'crispembed_posformer_' + name + '_probe'
        start = added.rfind('extern "C"', 0, added.index(token))
        end = added.index('\n}', added.index(token)) + 2
        require(start >= 0, 'Probe source anchor missing')
        probes.append(added[start:end])
    text = instrument(path.read_text() + '\n' + '\n\n'.join(probes) + '\n')
    path.write_text(text)
    provenance.update({'original_source_sha256': provenance['production_source_sha256'],
                       'instrumented_source_sha256': sha(path),
                       'instrumentation_tool_sha256': sha(Path(args.app_root) / 'tool/prepare_handwriting_reference_bridge.py'),
                       'probe_fixture_patch_sha256': sha(diagnostic),
                       'test_only': True, 'mathematical_change': 'none: production helpers unchanged'})
    write(args.output, provenance)


def reference(args):
    import handwriting_reference_parity as reference
    provenance = load(args.provenance)
    expected = provenance.copy()
    require(expected.get('test_only') is True, 'Reference requires separate test-only library')
    require(expected.get('production_probes_absent') is True, 'Production library contains probes')
    require(sha(Path(args.root) / 'src/posformer_ocr.cpp') == expected['instrumented_source_sha256'],
            'Instrumented source identity changed')
    require(sha(args.library) != expected['production_library_sha256'], 'Test-only library equals production')
    expected['library_sha256'] = sha(args.library)
    write(args.provenance, expected)

    def validate(actual, library_hash, instrumentation_hash, _norm_fixture_hash, forward_fixture_hash):
        require(actual == expected, 'Reference source provenance changed')
        require(actual['library_sha256'] == library_hash, 'Reference library identity mismatch')
        require(actual['instrumentation_tool_sha256'] == instrumentation_hash,
                'Reference instrumentation identity mismatch')
        require(actual['probe_fixture_patch_sha256'] == forward_fixture_hash, 'Probe fixture changed')
        require(actual['production_source_sha256'] != actual['instrumented_source_sha256'],
                'Reference instrumentation missing')
    # This adapter changes provenance validation only, never the independent
    # constructor, tensor mapping, preprocessing, inference math or tolerances.
    reference.validate_provenance = validate
    reference.run(SimpleNamespace(model=args.model, manifest=args.manifest, library=args.library,
                                  reference=args.reference, provenance=args.provenance,
                                  encoder=args.encoder, output=args.output))
    report = load(args.output)
    require(report.get('samples') == 5, 'Incomplete bounded reference')
    for row in report['cases']:
        require(row.get('preprocessing', {}).get('matches') is True, 'Preprocessing mismatch')
        require(row.get('first_encoder_divergence') is None, 'Encoder reference divergence')
        require(row.get('decoder') and all(stage.get('matches') is True
                for step in row['decoder'] for stage in step['stages'].values()),
                'Decoder reference divergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['provenance', 'compare', 'prepare', 'reference'])
    for name in ('root', 'app-root', 'baseline-root', 'baseline-library', 'production-library',
                 'baseline', 'corrected', 'provenance', 'library', 'model', 'manifest', 'reference', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--encoder', choices=['default', 'scalar'])
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.app_root) / 'tool'))
    if args.operation == 'provenance':
        source_provenance(args)
    elif args.operation == 'compare':
        write(args.output, compare(load(args.baseline), load(args.corrected), load(args.provenance)))
    elif args.operation == 'prepare':
        prepare(args)
    else:
        reference(args)


if __name__ == '__main__':
    main()
