#!/usr/bin/env python3
"""IDs/text/EOS comparison of the OvisOCR2 graph-decoder path with the frozen first-F16 receipt.

Reads the unchanged eleven-case producer report (tools/ci-heavy/ovisocr2_quality_run.py
--phase f16) written under CRISPEMBED_QWEN35_OCR_GRAPH_DECODER=1, and compares every case
with tests/results/qwen35-ocr/ovisocr2-f16-quality.json (producer.native_cases.f16), the
frozen default-path (decoder_cpu) ground truth.

Contract (owner-approved, 11 November 2026, for the graph path only): input hashes, token
IDs, raw/processed text, stop ID, finish reason, stop kind and cap must be exact. Selected
confidences are recorded with their drift but are not required to be exact (the graph
path reorders float reductions). The frozen default-path reference and its exact-
confidence guards are unchanged and remain testable with the gate set to 0.

Unlike the frozen-guard check this does not stop at the first difference: every
differing case is listed so a failed flip can be reported precisely.

    HEAVY_OUT=... ovisocr2_graph_quality_check.py --matmul {native,f32,exact,default}
Writes HEAVY_OUT/graph-quality-comparison.json; exits 1 if any case differs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json'
EXACT_KEYS = ('input_rgb_sha256', 'input_png_sha256', 'raw_text', 'processed_text',
              'token_ids', 'stop_id', 'finish_reason', 'stop_kind', 'cap')


def as_list(value):
    if isinstance(value, str) and value[:1] == '[':
        return json.loads(value)
    return value


def compare(current, frozen):
    actual = current.get('native_cases', {}).get('f16', {})
    old = frozen['native_cases']['f16']
    result = {'cases': {}, 'differing_cases': [], 'missing_cases': sorted(set(old) - set(actual)),
              'extra_cases': sorted(set(actual) - set(old))}
    for name, before in sorted(old.items()):
        row = actual.get(name)
        if row is None:
            continue
        diffs = [k for k in EXACT_KEYS if as_list(row.get(k)) != as_list(before.get(k))]
        ids, fids = as_list(row.get('token_ids')) or [], as_list(before['token_ids'])
        conf, fconf = as_list(row.get('confidences')) or [], as_list(before['confidences'])
        valid = (len(conf) == len(ids) and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                                               for v in conf))
        common = min(len(conf), len(fconf))
        rec = {'token_count': len(ids), 'frozen_token_count': len(fids), 'exact_fields_differing': diffs,
               'ids_text_eos_exact': not diffs, 'confidences_valid': valid,
               'confidences_exact': conf == fconf,
               'max_selected_confidence_abs_drift': max((abs(a - b) for a, b in zip(conf, fconf)), default=None),
               'mean_confidence': row.get('mean_confidence'), 'frozen_mean_confidence': before.get('mean_confidence'),
               'first_id_divergence': next((i for i in range(min(len(ids), len(fids))) if ids[i] != fids[i]),
                                           None if len(ids) == len(fids) else min(len(ids), len(fids))),
               'wall_seconds': row.get('wall_seconds'), 'token_ids': ids, 'confidences': conf}
        if row.get('mean_confidence') is not None and before.get('mean_confidence') is not None:
            rec['mean_confidence_abs_drift'] = abs(row['mean_confidence'] - before['mean_confidence'])
        if diffs:
            rec['raw_text'] = row.get('raw_text')
            rec['frozen_raw_text'] = before.get('raw_text')
        if diffs or not valid:
            result['differing_cases'].append(name)
        result['cases'][name] = rec
    result['token_count'] = sum(r['token_count'] for r in result['cases'].values())
    result['max_selected_confidence_abs_drift'] = max(
        (r['max_selected_confidence_abs_drift'] or 0.0 for r in result['cases'].values()), default=None)
    result['screens_equal'] = {k: current.get(k) == frozen.get(k)
                               for k in ('f16_paired_synthetic', 'f16_paired_documents')}
    result['scores_equal'] = current.get('scores', {}).get('f16') == frozen.get('scores', {}).get('f16')
    cabi = current.get('cabi_smoke', {}).get('f16', {})
    native = actual.get('dev-hello', {})
    result['cabi_text_matches_native'] = bool(cabi) and cabi.get('raw_text') == native.get('raw_text')
    # The graph path reorders float reductions, so exact confidences on every case would
    # mean the gate did not reach the decoder.
    result['graph_path_observed'] = any(not r['confidences_exact'] for r in result['cases'].values())
    result['passed'] = (len(result['cases']) == 11 and not result['differing_cases'] and not result['missing_cases']
                        and all(result['screens_equal'].values()) and result['scores_equal']
                        and result['cabi_text_matches_native'] and result['graph_path_observed'])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--matmul', choices=('native', 'f32', 'exact', 'default'), required=True,
                   help='arm label; default = gates unset (graph decoder, native since 11 November 2026)')
    args = p.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    current = json.loads((out / 'ovisocr2-quality.json').read_text())
    frozen = json.loads(FROZEN.read_text())['producer']
    result = compare(current, frozen)
    env = {k: os.environ.get(k) for k in ('CRISPEMBED_QWEN35_OCR_GRAPH_DECODER', 'CRISPEMBED_QWEN35_GRAPH_MATMUL',
                                         'CRISPEMBED_QWEN35_VISION_THREADED_ATTN',
                                         'CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION')}
    result.update(schema='crispembed-ovisocr2-graph-quality-v1', matmul=args.matmul, environment=env,
                  source_commit=os.environ.get('GITHUB_SHA'), run_id=os.environ.get('GITHUB_RUN_ID'),
                  producer_status=current.get('status'),
                  contract='IDs/text/EOS/stop/cap exact vs frozen first-F16; confidences recorded with drift')
    (out / 'graph-quality-comparison.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=1))
    for name, r in result['cases'].items():
        print(name, r['token_count'], 'exact' if r['ids_text_eos_exact'] else 'DIFF ' + ','.join(r['exact_fields_differing']),
              'conf_drift', r['max_selected_confidence_abs_drift'])
    sys.exit(0 if result['passed'] else 1)
