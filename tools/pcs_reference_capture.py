#!/usr/bin/env python3
"""Hosted-only readback of immutable original PCS graph; no reference replacement."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from pcs_parity import parse_ref


def main():
    out = ROOT / 'pcs-reference-evidence'
    out.mkdir(exist_ok=True)
    repo = '1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase'
    rev = 'd1769a597ce8dfaa070d436bc67d4ee761f58884'
    expected = {'model.onnx': 'c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89',
                'sp.model': '7f944d0be93b275f62e1913fd409f378ddbba108e57fe4a9cb47e8c047f6bef1'}
    paths = {}
    for name, digest in expected.items():
        p = Path(hf_hub_download(repo, name, revision=rev))
        with p.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError('Immutable original hash differs: ' + name)
        paths[name] = str(p)
    report = {'source': {'repo': repo, 'revision': rev, 'sha256': expected}, 'cases': []}
    graph = onnx.load(paths['model.onnx'])
    nodes = {output: node for node in graph.graph.node for output in node.output}
    (out / 'graph.json').write_text(json.dumps([{'name': n.name, 'op': n.op_type,
        'inputs': list(n.input), 'outputs': list(n.output)} for n in graph.graph.node], indent=2))
    outputs = [o.name for o in graph.graph.output]
    cap = outputs[2]
    # Walk the actual cap path through its small classifier, stopping at the
    # encoder/head concatenation. Capture graph values, never recode the head.
    capture = []
    visited = set()
    def walk(value):
        if value in visited or value not in nodes:
            return
        visited.add(value)
        n = nodes[value]
        if n.op_type in ('Add', 'MatMul', 'Gemm', 'Relu', 'Concat', 'Sigmoid'):
            capture.append(value)
        if n.op_type == 'Concat':
            capture.extend(x for x in n.input if x in nodes and nodes[x].op_type not in ('Cast', 'Unsqueeze'))
            return
        if n.op_type not in ('Constant', 'Shape', 'Gather'):
            for x in n.input:
                walk(x)
    walk(cap)
    # Extended stage inventory for the earliest differing encoder operation.
    capture.extend(n.output[0] for n in graph.graph.node if n.op_type == 'LayerNormalization')
    capture = list(dict.fromkeys(capture))
    head_weights = {x.name: onnx.numpy_helper.to_array(x) for x in graph.graph.initializer
                    if '_cap_head' in x.name or '_seg_head' in x.name}
    np.savez(out / 'original-head-weights.npz', **head_weights)
    (out / 'capture-names.json').write_text(json.dumps(capture, indent=2))
    for name in capture:
        graph.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    observed = out / 'observed.onnx'
    onnx.save(graph, observed)
    del graph
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    baseline = ort.InferenceSession(paths['model.onnx'], sess_options=opts, providers=['CPUExecutionProvider'])
    observer = ort.InferenceSession(str(observed), sess_options=opts, providers=['CPUExecutionProvider'])
    sp = spm.SentencePieceProcessor(model_file=paths['sp.model'])
    refs, _ = parse_ref(ROOT / 'tests/regression/pcs/blueprint_ref.txt')
    for ri, ref in enumerate(refs):
        ids = sp.encode(ref['text'])
        if ids != ref['ids']:
            raise ValueError('Original token IDs differ from frozen reference')
        feed = {'input_ids': np.array([[sp.bos_id()] + ids + [sp.eos_id()]], dtype=np.int64)}
        plain = baseline.run(None, feed)
        seen = observer.run(None, feed)
        if any(a.tobytes() != b.tobytes() for a, b in zip(plain, seen)):
            raise ValueError('Original ONNX observer changes existing predictions')
        expected_heads = [np.array(ref['pre']), np.array(ref['post']),
            np.array([[c == '1' for c in row] for row in ref['cap']]), np.array([c == '1' for c in ref['seg']])]
        if any(not np.array_equal(a[0, 1:-1], b) for a, b in zip(plain, expected_heads)):
            raise ValueError('Original outputs differ from frozen reference')
        arrays = {str(i): array for i, array in enumerate(seen[len(plain):])}
        np.savez(out / ('case-' + str(ri) + '.npz'), **arrays)
        report['cases'].append({'index': ri, 'observer_predictions_exact': True,
            'all_original_heads_match_frozen_reference': True,
            'captures': [{'name': name, 'shape': list(a.shape),
                'token3': a[0, 4].tolist() if a.ndim == 3 and a.shape[1] == len(ids)+2 else None}
                for name, a in zip(capture, seen[len(plain):])]})
        (out / 'summary.json').write_text(json.dumps(report, indent=2))
    observed.unlink()  # The large original remains in the provider cache; reports only.
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
