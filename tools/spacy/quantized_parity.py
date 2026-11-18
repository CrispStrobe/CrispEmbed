#!/usr/bin/env python3
"""Candidate-only admission adapter around unchanged development scorers.

First authenticates the accepted F32 bundle and frozen reference with the
original validator. Only the candidate's model hash/size changes, bound to the
quantizer proof. No original manifests, assertions, budgets or decoded scorers
are edited. Each compared stage additionally enters crispembed_diff.h.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
import spacy_parity as numeric
import spacy_text_parity as text


def reference_gguf(path, values):
    name = b'stage'
    dims = tuple(reversed(values.shape)) or (1,)
    header = struct.pack('<IIQQQ', 0x46554747, 3, 1, 0, len(name)) + name
    header += struct.pack('<I', len(dims))
    header += b''.join(struct.pack('<Q', n) for n in dims)
    header += struct.pack('<IQ', 0, 0)
    header += b'\0' * (-len(header) % 32)
    path.write_bytes(header + values.astype('<f4').tobytes())


def main():
    parser = argparse.ArgumentParser()
    for arg in ('library', 'baseline-bundle', 'bundle', 'quds-root', 'candidate-proof', 'diff-library', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    args.reference = None
    assert not args.output.exists(), 'Never overwrite a candidate report'
    args.output.mkdir()
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.bundle = args.baseline_bundle
    original_validate = numeric.validate_inputs
    baseline = original_validate(baseline_args)
    proof = json.loads(args.candidate_proof.read_text())
    assert proof['source_f32_sha256'] == baseline[2]['files']['model.gguf']['sha256']
    assert proof['recipe'] in ('f16', 'q8_0', 'q4_0')
    model = args.bundle / 'model.gguf'
    assert numeric.sha256(model) == proof['candidate_sha256'] and model.stat().st_size == proof['candidate_bytes']
    candidate = json.loads((args.bundle / 'manifest.json').read_text())
    expected = json.loads(json.dumps(baseline[2]))
    expected['files']['model.gguf'] = {'sha256': proof['candidate_sha256'], 'size_bytes': proof['candidate_bytes']}
    assert candidate == expected
    for name, pin in candidate['files'].items():
        item = args.bundle / name
        assert not item.is_symlink() and item.stat().st_size == pin['size_bytes'] and numeric.sha256(item) == pin['sha256']
    numeric.validate_inputs = lambda supplied: (baseline[0], baseline[1], candidate, baseline[3])
    bridge = C.CDLL(str(args.diff_library.resolve()))
    bridge.spacy_diff.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_float), C.c_size_t, C.POINTER(C.c_double)]
    bridge.spacy_diff.restype = C.c_int
    original_compare = numeric.compare_stage
    scratch = args.output / 'current-reference.gguf'
    calls = 0

    def compare(name, original, actual):
        nonlocal calls
        result = original_compare(name, original, actual)
        reference_gguf(scratch, original)
        values = actual.astype('<f4').ravel()
        metrics = (C.c_double * 7)()
        status = bridge.spacy_diff(os.fsencode(scratch), b'stage', values.ctypes.data_as(C.POINTER(C.c_float)), values.size, metrics)
        scratch.unlink()
        assert status == 0
        result['crispembed_diff'] = dict(zip(('max_abs', 'mean_abs', 'rms', 'cos_min', 'cos_global', 'actual_norm', 'reference_norm'), metrics))
        calls += 1
        return result

    numeric.compare_stage = compare
    os.environ['CRISPEMBED_SPACY_QUANTIZED'] = '1'
    reports = {}
    for label, qualify in [('prepared', numeric.qualify), ('text', text.qualify)]:
        result = qualify(args)
        result['experimental_candidate_proof'] = proof
        result['original_f32_bundle_validated_before_candidate'] = True
        result['baseline_is_not_replaced'] = True
        result['scope'] = 'Experimental storage/dequantization on frozen DEVELOPMENT oracle only; original scorer and numerical budgets'
        result['crispembed_diff_stage_calls_cumulative'] = calls
        (args.output / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        reports[label] = {'passed': result['passed'], 'counts': result['counts']}
    (args.output / 'summary.json').write_text(json.dumps({'recipe': proof['recipe'], 'reports': reports,
        'crispembed_diff_calls': calls, 'qualified': all(r['passed'] for r in reports.values()),
        'acceptance_or_W6_replayed': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
