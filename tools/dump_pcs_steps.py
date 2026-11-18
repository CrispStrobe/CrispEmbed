#!/usr/bin/env python3
"""Observe the pinned original ONNX operators, using the existing PCS dumper.

One selected original corpus case. Original and same-stored-weight graphs use
identical nodes/attributes/inputs; no hand-written Python inference substitutes
for the original. Model/expanded graph files remain hosted, never uploaded.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

from pcs_steps import inventory, write_archive


def canonical(array):
    # These named ONNX values all have a leading batch axis of exactly one.
    if array.shape[0] != 1:
        raise ValueError('PCS reference batch is not one')
    return array[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--stored-model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', type=int, default=3)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import sentencepiece as spm
    from onnx import helper, TensorProto
    from pcs_weight_replay import map_stored_weights
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tests'))
    from pcs_parity import parse_ref
    refs, _ = parse_ref(root / 'tests/regression/pcs/blueprint_ref.txt')
    ref = refs[args.case]
    args.output.mkdir(exist_ok=False)
    model = args.model_dir / 'model.onnx'
    expected = {'model.onnx': 'c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89',
                'sp.model': '7f944d0be93b275f62e1913fd409f378ddbba108e57fe4a9cb47e8c047f6bef1'}
    for name, digest in expected.items():
        with (args.model_dir / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Pinned original input changed: ' + name)
    with args.stored_model.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != 'faee4bf8469a502362b5967a2cfeb7a1732539fbd999c63b489d00efff3531da':
            raise ValueError('Pinned stored model changed')
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.model_dir / 'sp.model'))
    ids = tokenizer.encode(ref['text'])
    if ids != ref['ids']:
        raise ValueError('Frozen original token IDs changed')
    full = [tokenizer.bos_id()] + ids + [tokenizer.eos_id()]
    feed = {'input_ids': np.array([full], dtype=np.int64)}
    captures = inventory()
    graph = onnx.load(str(model))
    names = {value for node in graph.graph.node for value in node.output}
    if set(captures.values()) - names:
        raise ValueError('Missing original operator: ' + repr(set(captures.values()) - names))
    operators = [{'name': node.name, 'op': node.op_type, 'inputs': list(node.input),
                  'outputs': list(node.output),
                  'attributes': {a.name: str(helper.get_attribute_value(a)) for a in node.attribute}}
                 for node in graph.graph.node if any(value in captures.values() for value in node.output)]
    report = {'case': args.case, 'text': ref['text'], 'full_input_ids': full,
              'capture_map': captures,
              'original_operators': operators, 'inputs_sha256': expected, 'arms': {}}
    (args.output / 'input.txt').write_text(ref['text'] + '\n')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    # Freeze the unobserved original predictions before appending any outputs.
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    baseline = session.run(None, feed)
    expected_heads = [np.array(ref['pre']), np.array(ref['post']),
                      np.array([[c == '1' for c in row] for row in ref['cap']]),
                      np.array([c == '1' for c in ref['seg']])]
    if any(not np.array_equal(actual[0, 1:-1], expected) for actual, expected in zip(baseline, expected_heads)):
        raise ValueError('Unobserved original no longer matches frozen heads')
    del session
    def append_outputs():
        for key, name in captures.items():
            dtype = TensorProto.INT64 if key.startswith('input.') else TensorProto.FLOAT
            graph.graph.output.append(helper.make_tensor_value_info(name, dtype, None))
    append_outputs()
    for arm in ['original', 'stored']:
        if arm == 'stored':
            # save_model externalizes and clears raw initializer bytes in-place.
            # Reload the immutable original before mapping the second arm.
            del graph
            gc.collect()
            graph = onnx.load(str(model))
            rows = {'emb.tok_emb.weight': full,
                    'emb.pos_emb.weight': report['position_ids'], 'report': []}
            report['tensor_mapping'] = map_stored_weights(graph, args.stored_model, args.output, rows)
            report['embedding_tensor_differences'] = rows['report']
            plain = args.output / 'stored-unobserved.onnx'
            onnx.save_model(graph, str(plain), save_as_external_data=True,
                            all_tensors_to_one_file=True, location='stored-unobserved-weights.bin', size_threshold=1024)
            session = ort.InferenceSession(str(plain), sess_options=options, providers=['CPUExecutionProvider'])
            stored_baseline = session.run(None, feed)
            del session, graph
            gc.collect()
            graph = onnx.load(str(plain))
            append_outputs()
        path = args.output / (arm + '.onnx')
        onnx.save_model(graph, str(path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=arm + '-weights.bin', size_threshold=1024)
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        observed = session.run(None, feed)
        predictions = observed[:4]
        unobserved = baseline if arm == 'original' else stored_baseline
        if any(a.tobytes() != b.tobytes() for a, b in zip(predictions, unobserved)):
            raise ValueError(arm + ' observer changes predictions')
        tensors = {name: canonical(value).astype(np.int32 if name.startswith('input.') else np.float32)
                   for name, value in zip(captures, observed[4:])}
        # Native heads operate after removing BOS/EOS. Encoder captures retain
        # them. Explicitly align these different boundaries rather than flatten
        # different token sequences into an apparently matching element count.
        for name in tensors:
            if name.startswith('head.'):
                tensors[name] = tensors[name][1:-1]
        tensors['input.ids'] = np.asarray(full, dtype=np.int32)
        if arm == 'original':
            report['position_ids'] = tensors['input.positions'].tolist()
            report['type_ids'] = tensors['input.types'].tolist()
        np.savez(args.output / (arm + '.npz'), **tensors)
        write_archive(args.output / (arm + '.gguf'), tensors, {'pcs.arm': arm, 'pcs.text': ref['text']})
        report['arms'][arm] = {'capture_count': len(tensors), 'observer_predictions_exact': True,
                              'frozen_head_counts': [int(np.all(a[0, 1:-1] == b, axis=-1).sum()) if b.ndim == 2
                                                     else int(np.sum(a[0, 1:-1] == b)) for a, b in zip(predictions, expected_heads)],
                              'prediction_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in predictions]}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        del session, observed, tensors
        gc.collect()
    print(json.dumps({'case': args.case, 'arms': report['arms'],
                      'embedding_tensor_differences': report['embedding_tensor_differences']}, indent=2))


if __name__ == '__main__':
    main()
