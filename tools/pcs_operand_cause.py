#!/usr/bin/env python3
"""Causal arithmetic checks of qualified PCS captures; never runs a model.

Uses the original MatMul's captured Q/K operands and the already-executed
reference-input native arm. Float64 algebra separates input perturbations from
local arithmetic. Original F32 observations and budgets remain authoritative.
No checkpoint, graph producer, inference runtime or conversion code is loaded.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def head_operands(query, key, score_shape):
    if len(score_shape) != 3:
        raise ValueError('QK scores must have canonical [head,query,key] axes')
    heads, queries, keys = score_shape
    if not (0 < heads <= 32 and 0 < queries <= 64 and 0 < keys <= 64):
        raise ValueError('Captured operator exceeds the bounded analysis scope')
    if query.ndim != 2 or key.ndim != 2 or query.shape[0] != queries or key.shape[0] != keys:
        raise ValueError('Q/K token axes disagree with captured scores')
    if query.shape[1] != key.shape[1] or query.shape[1] % heads:
        raise ValueError('Q/K channel and head axes disagree')
    width = query.shape[1] // heads
    if not 0 < width <= 256:
        raise ValueError('Head width exceeds bounded scope')
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(key)):
        raise ValueError('Nonfinite captured operand')
    return (query.astype(np.float64).reshape(queries, heads, width).transpose(1, 0, 2),
            key.astype(np.float64).reshape(keys, heads, width).transpose(1, 0, 2))


def split_qk(query_ref, key_ref, query_native, key_native, scores_ref, scores_native, scores_replay):
    shape = scores_ref.shape
    if scores_native.shape != shape or scores_replay.shape != shape:
        raise ValueError('Actual/reference/replay score shapes differ')
    qr, kr = head_operands(query_ref, key_ref, shape)
    qn, kn = head_operands(query_native, key_native, shape)
    if not all(np.all(np.isfinite(x)) for x in [scores_ref, scores_native, scores_replay]):
        raise ValueError('Nonfinite captured result')
    r, n, replay = [x.astype(np.float64) for x in [scores_ref, scores_native, scores_replay]]
    kt = lambda value: value.swapaxes(-1, -2)
    dq, dk = qn - qr, kn - kr
    q_effect = dq @ kt(kr)
    k_effect = qr @ kt(dk)
    cross = dq @ kt(dk)
    propagated = q_effect + k_effect + cross
    actual_difference = n - r
    local_at_reference = replay - r
    input_dependent_local = (n - replay) - propagated
    closure = actual_difference - (local_at_reference + q_effect + k_effect + cross + input_dependent_local)
    return {'actualDifference': actual_difference, 'referenceInputLocalError': local_at_reference,
            'queryPerturbationEffect': q_effect, 'keyPerturbationEffect': k_effect,
            'crossEffect': cross, 'totalOperandEffect': propagated,
            'inputDependentArithmeticRemainder': input_dependent_local,
            'algebraClosureError': closure,
            'referenceFloat32VersusFloat64Residual': r - qr @ kt(kr),
            'nativeFloat32VersusFloat64Residual': n - qn @ kt(kn)}


def describe_point(parts, index):
    return {name: float(array[index]) for name, array in parts.items()}


def embedding_split(original, stored):
    names = ['emb.token', 'emb.position', 'emb.type', 'emb.sum']
    if any(original[n].shape != stored[n].shape for n in names):
        raise ValueError('Embedding capture shape differs')
    ot, op, oy = [original[n] for n in names[:3]]
    st, sp, sy = [stored[n] for n in names[:3]]
    if not np.array_equal(oy, sy):
        raise ValueError('Type embedding differs; cannot use a two-table intervention')
    original_sum = (ot + oy) + op
    stored_sum = (st + sy) + sp
    if original_sum.tobytes() != original['emb.sum'].tobytes() or stored_sum.tobytes() != stored['emb.sum'].tobytes():
        raise ValueError('Two original graph additions do not reproduce captured sums')
    token_effect = st.astype(np.float64) - ot
    position_effect = sp.astype(np.float64) - op
    observed = stored_sum.astype(np.float64) - original_sum
    rounding = observed - token_effect - position_effect
    index = np.unravel_index(np.argmax(np.abs(observed)), observed.shape)
    token_only = (st + oy) + op
    position_only = (ot + oy) + sp
    return {'coordinate': list(map(int, index)), 'observedSumDifference': float(observed[index]),
            'tokenTableContribution': float(token_effect[index]),
            'positionTableContribution': float(position_effect[index]),
            'additionRoundingRemainder': float(rounding[index]),
            'maxAbsoluteObservedSumDifference': float(np.max(np.abs(observed))),
            'tokenOnlyCounterfactualSum': float(token_only[index]),
            'positionOnlyCounterfactualSum': float(position_only[index]),
            'originalSum': float(original_sum[index]), 'storedSum': float(stored_sum[index]),
            'originalAndStoredGraphSumsReproducedBitExactly': True,
            'boundary': 'pre-normalization embedding sum only; not downstream model or decoded causality'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', required=True, type=Path)
    parser.add_argument('--admission', required=True, type=Path)
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--head-biases', required=True, type=Path)
    parser.add_argument('--original-summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = args.panel
    run = json.loads((panel / 'run.json').read_text())
    admission = json.loads(args.admission.read_text())
    qualification = json.loads(args.qualification.read_text())
    if run['source'] != admission['source'] or run['source'] != qualification['measuredSource']:
        raise ValueError('Mixed source identities')
    if not qualification['freshSnapshotQualification'] or not admission['checks']['same214StoredTensors']:
        raise ValueError('Captures are not admitted for operand attribution')
    summary = json.loads((panel / 'reference/summary.json').read_text())
    old_summary = json.loads(args.original_summary.read_text())
    if old_summary['source']['sha256']['model.onnx'] != summary['inputs_sha256']['model.onnx']:
        raise ValueError('Original bias source model differs')
    if run['absolute_budget'] != 0.0001 or run['arms']['materialized-replay']['mode'] != 'independent-input replay, not composed quality':
        raise ValueError('Changed budget or missing actual reference-input intervention')
    from gguf import GGUFReader
    readers = {name: GGUFReader(str(path)) for name, path in {
        'original': panel / 'reference/original.gguf', 'reference': panel / 'reference/stored.gguf',
        'native': panel / 'materialized-on/capture.gguf',
        'replay': panel / 'materialized-replay/capture.gguf'}.items()}
    arrays = {name: {t.name: t.data for t in reader.tensors} for name, reader in readers.items()}
    rows = list(csv.DictReader((panel / 'materialized-on-vs-stored.stdout').open(), delimiter='\t'))
    original_nodes = {output: node for node in summary['original_operators'] for output in node['outputs']}
    report = {'source': run['source'], 'newModelOrCaptureRuns': 0, 'weightsRegenerated': False,
              'budget': run['absolute_budget'], 'localInputs': 'existing admitted operator captures only',
              'newModelQualityPassClaim': False, 'qkLayers': [], 'biasParameters': []}
    for row in rows:
        stage = row['stage']
        if not stage.endswith('.attention.qk'):
            continue
        layer = int(stage.split('.')[1]);prefix = f'layer.{layer}.'
        node = original_nodes[summary['capture_map'][stage]]
        if node['op'] != 'MatMul' or len(node['inputs']) != 2:
            raise ValueError('Captured QK does not name the actual original binary MatMul')
        ref, native, replay = [arrays[name] for name in ['reference', 'native', 'replay']]
        parts = split_qk(ref[prefix + 'q.bias'], ref[prefix + 'k.bias'],
                         native[prefix + 'q.bias'], native[prefix + 'k.bias'],
                         ref[stage], native[stage], replay[stage])
        flat = int(row['worst_index']);index = np.unravel_index(flat, ref[stage].shape)
        point = describe_point(parts, index)
        report['qkLayers'].append({'stage': stage, 'originalNode': node,
            'originalBudgetStatusUnchanged': row['result'], 'originalReportedMaxAbsolute': float(row['max_abs']),
            'coordinate': {'head': int(index[0]), 'queryFullToken': int(index[1]), 'keyFullToken': int(index[2])},
            'atReportedWorst': point,
            'maxAbsoluteAlgebraClosure': float(np.max(np.abs(parts['algebraClosureError']))),
            'maxActualReferenceInputLocalError': float(np.max(np.abs(parts['referenceInputLocalError']))),
            'maxOperandPropagationEffect': float(np.max(np.abs(parts['totalOperandEffect'])))})
    report['embeddingParameterBoundary'] = embedding_split(arrays['original'], arrays['reference'])
    mapped = {item['original']: item for item in summary['tensor_mapping']}
    with np.load(args.head_biases) as biases:
        for name in biases.files:
            if name not in mapped:
                raise ValueError('Original retained head bias lacks actual tensor mapping')
            value = np.ascontiguousarray(biases[name], dtype='<f4');entry = mapped[name]
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            report['biasParameters'].append({'original': name, 'native': entry['native'],
                'shape': list(value.shape), 'originalSha256': digest, 'storedDequantizedSha256': entry['sha256'],
                'bytesIdentical': digest == entry['sha256'],
                'exclusionBoundary': 'Direct parameter representation only; not affine-input or accumulation errors'})
    required = ['head.tc.fc1.weight', 'head.tc.fc2.weight']
    report['missingForUniqueDecodedTensorAttribution'] = {
        'requiredCasingProjectionNames': required,
        'projectionPayloadsInCaptures': {name: any(name in values for values in arrays.values()) for name in required},
        'availableEvidence': 'activation arrays, selected embedding rows, tensor hashes/shapes and four original head biases',
        'limit': 'No single tensor/conversion is identified as the unique cause of the final decoded error. '
                 'This panel cannot execute a parameter-held-fixed classifier/encoder intervention without missing weight bytes.'}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    failed = [entry for entry in report['qkLayers'] if entry['originalBudgetStatusUnchanged'] == 'FAIL']
    print(json.dumps({'qkLayers': len(report['qkLayers']), 'retainedQKBudgetFailures': len(failed),
        'firstFailedOperatorAttribution': failed[0] if failed else None,
        'embeddingParameterBoundary': report['embeddingParameterBoundary'],
        'biasParameters': report['biasParameters']}, indent=2))


if __name__ == '__main__':
    main()
