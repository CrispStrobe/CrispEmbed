#!/usr/bin/env python3
"""Replay immutable PCS weights in the pinned original graph; reports only.

This does not create a model for distribution or change any quantization recipe.
All tensor replacements are mapped by actual original operators and shape-checked.

    python tools/pcs_weight_replay.py [--model STORED.gguf]

Without `--model` it replays `.pcs-models/pcs-xlmr-base-q4_k-imatrix.gguf`.
"""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request

def gate_exit_code(report):
    cases = report['cases']
    return int(len(cases) != 6 or any(not c['decoded_exact'] or
        any(value != c['tokens'] for value in c['head_counts'].values()) for c in cases))


# Reanalysis verifies preserved outputs without another model run or imports.
if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--verify-report':
    retained = json.loads(Path(sys.argv[2]).read_text())
    code = gate_exit_code(retained)
    print('PASS' if code == 0 else 'FAIL: retained stored-weight replay is quality-negative')
    raise SystemExit(code)


import gguf
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
import sentencepiece as spm
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from pcs_parity import parse_ref


def map_stored_weights(graph, model_path, out, selected_rows=None):
    nodes = {n.name: n for n in graph.graph.node}
    initializers = {i.name: i for i in graph.graph.initializer}
    reader = gguf.GGUFReader(str(model_path))
    tensors = {t.name: t for t in reader.tensors}
    (out / 'inventory.json').write_text(json.dumps({
        'original': [{'name': i.name, 'shape': list(i.dims), 'type': i.data_type} for i in graph.graph.initializer],
        'native': [{'name': t.name, 'shape': list(map(int, t.shape)), 'type': str(t.tensor_type)} for t in reader.tensors]}, indent=2))
    mapped, used = {}, set()
    manifest = []

    def replace(name, source, transpose=False):
        t = tensors[source]
        array = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
        if transpose:
            array = array.T
        dst = initializers[name]
        # GGUF omits trailing singleton GGML axes, e.g. [768,1] is stored
        # as a vector while ONNX retains the [1,768] embedding table.
        if tuple(x for x in array.shape if x != 1) == tuple(x for x in dst.dims if x != 1):
            array = array.reshape(tuple(dst.dims))
        if list(array.shape) != list(dst.dims) or dst.data_type != TensorProto.FLOAT:
            raise ValueError('Original/native tensor geometry/type differs: ' + name + ' / ' + source +
                             ' native=' + str(array.shape) + ' original=' + str(list(dst.dims)) +
                             ' original_type=' + str(dst.data_type))
        if selected_rows is not None and source.startswith('emb.'):
            original = numpy_helper.to_array(dst)
            rows = selected_rows.get(source)
            left, right = (original[rows], array[rows]) if rows is not None else (original, array)
            delta = left.astype(np.float64) - right.astype(np.float64)
            selected_rows['report'].append({'name': source, 'rows': None if rows is None else rows,
                'shape': list(left.shape), 'max_absolute': float(np.max(np.abs(delta))),
                'rms': float(np.sqrt(np.mean(delta * delta))),
                'original_max_magnitude': float(np.max(np.abs(left))),
                'stored_max_magnitude': float(np.max(np.abs(right))),
                'bytes_equal': left.tobytes() == right.tobytes()})
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if name in mapped and mapped[name] != digest:
            raise ValueError('Conflicting original alias replacement')
        mapped[name] = digest
        used.add(source)
        manifest.append({'original': name, 'native': source, 'type': str(t.tensor_type),
                         'shape': list(array.shape), 'sha256': digest})
        dst.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(array), name=name))

    for original, native in [('word_embeddings', 'tok_emb'), ('position_embeddings', 'pos_emb'),
                             ('token_type_embeddings', 'type_emb')]:
        node = nodes['/bert_model/embeddings/' + original + '/Gather']
        replace(node.input[0], 'emb.' + native + '.weight')
    def norm(prefix, native):
        node = nodes[prefix + '/LayerNormalization']
        replace(node.input[1], native + '.weight')
        replace(node.input[2], native + '.bias')
    def linear(prefix, native):
        node = nodes[prefix + '/MatMul']
        replace(node.input[1], native + '.weight', True)
        bias = [x for x in nodes[prefix + '/Add'].input if x in initializers]
        if len(bias) != 1:
            raise ValueError('Ambiguous bias')
        replace(bias[0], native + '.bias')
    norm('/bert_model/embeddings/LayerNorm', 'emb.ln')
    for layer in range(12):
        prefix = '/bert_model/encoder/layer.' + str(layer)
        native = 'enc.' + str(layer)
        for op, key in [('query', 'q'), ('key', 'k'), ('value', 'v')]:
            linear(prefix + '/attention/self/' + op, native + '.attn.' + key)
        linear(prefix + '/attention/output/dense', native + '.attn.out')
        norm(prefix + '/attention/output/LayerNorm', native + '.attn.ln')
        linear(prefix + '/intermediate/dense', native + '.ffn.up')
        linear(prefix + '/output/dense', native + '.ffn.down')
        norm(prefix + '/output/LayerNorm', native + '.ffn.ln')
    for key, original in [('post', '_punct_head_post'), ('pre', '_punct_head_pre'),
                          ('sbd', '_seg_head'), ('tc', '_cap_head')]:
        for index in range(2):
            linear('/_decoder/' + original + '/_linears.' + str(index),
                   'head.' + key + '.fc' + str(index + 1))
    replace(nodes['/_decoder/_punct_emb/Gather'].input[0], 'head.post_emb.weight')
    if used != set(tensors):
        raise ValueError('Incomplete native tensor replacement: ' + repr(set(tensors) - used))
    return manifest


def main():
    # `--model PATH` replays another stored PCS GGUF (default: the published
    # q4_k-imatrix download location used by the original blocker record).
    model = ROOT / '.pcs-models/pcs-xlmr-base-q4_k-imatrix.gguf'
    out = ROOT / 'pcs-weight-replay'
    if len(sys.argv) == 3 and sys.argv[1] == '--model':
        model = Path(sys.argv[2]).resolve()
        out = ROOT / ('pcs-weight-replay-' + model.stem)
    out.mkdir(exist_ok=True)
    repo = '1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase'
    revision = 'd1769a597ce8dfaa070d436bc67d4ee761f58884'
    paths = {name: Path(hf_hub_download(repo, name, revision=revision))
             for name in ['model.onnx', 'sp.model', 'config.yaml']}
    expected = {'model.onnx': 'c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89',
                'sp.model': '7f944d0be93b275f62e1913fd409f378ddbba108e57fe4a9cb47e8c047f6bef1'}
    for name, digest in expected.items():
        with paths[name].open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                raise ValueError('Original hash differs')
    graph = onnx.load(str(paths['model.onnx']))
    nodes = {n.name: n for n in graph.graph.node}
    initializers = {i.name: i for i in graph.graph.initializer}
    manifest = map_stored_weights(graph, model, out)
    capture = ['/_decoder/_cap_head/_linears.1/Add_output_0', '/_decoder/Concat_4_output_0']
    capture += [n.output[0] for n in graph.graph.node if n.op_type == 'LayerNormalization']
    for name in capture:
        graph.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    report = {'original': {'repository': repo, 'revision': revision, 'sha256': expected},
              'model': (json.loads(Path(str(model) + '.origin.json').read_text())
                        if Path(str(model) + '.origin.json').exists()
                        else {'path': model.name, 'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}),
              'tensor_replacements': manifest, 'all_native_tensors_mapped': True, 'cases': []}
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    onnx.save_model(graph, str(out / 'replay.onnx'), save_as_external_data=True,
                    all_tensors_to_one_file=True, location='replay-weights.bin', size_threshold=1024)
    del graph, initializers, nodes
    gc.collect()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(out / 'replay.onnx'), sess_options=opts, providers=['CPUExecutionProvider'])
    sp = spm.SentencePieceProcessor(model_file=str(paths['sp.model']))
    cfg = yaml.safe_load(paths['config.yaml'].read_text())
    url = 'https://raw.githubusercontent.com/1-800-BAD-CODE/punctuators/e0723a4f5dfe5f4fac8571a916ea03571b87edd3/punctuators/collectors/pcs_collector.py'
    collector_source = urllib.request.urlopen(url).read()
    (out / 'pcs_collector.py').write_bytes(collector_source)
    report['collector'] = {'url': url, 'sha256': hashlib.sha256(collector_source).hexdigest()}
    spec = importlib.util.spec_from_file_location('frozen_original_pcs_collector', out / 'pcs_collector.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    refs, _ = parse_ref(ROOT / 'tests/regression/pcs/blueprint_ref.txt')
    for ri, ref in enumerate(refs):
        ids = sp.encode(ref['text'])
        if ids != ref['ids']:
            raise ValueError('Original tokenizer IDs differ')
        values = session.run(None, {'input_ids': np.array([[sp.bos_id()] + ids + [sp.eos_id()]], dtype=np.int64)})
        pre, post, caps, seg = [a[0, 1:-1] for a in values[:4]]
        collector = module.PunctCapSegResultCollector(apply_sbd=True, overlap=16, sp_model=sp)
        collector.collect(ids=ids, pre_preds=[cfg['pre_labels'][i] if i else None for i in pre],
                          post_preds=[cfg['post_labels'][i] if i else None for i in post],
                          sbd_preds=seg.tolist(), cap_preds=caps.tolist(), idx=0)
        decoded = ' '.join(collector.produce())
        np.savez(out / ('case-' + str(ri) + '.npz'), **{str(i): a for i, a in enumerate(values)})
        report['cases'].append({'index': ri, 'decoded': decoded, 'reference': ref['punc'],
            'decoded_exact': decoded == ref['punc'], 'head_counts': {
                'post': int(np.equal(post, ref['post']).sum()), 'pre': int(np.equal(pre, ref['pre']).sum()),
                'seg': int(np.equal(seg, [v == '1' for v in ref['seg']]).sum()),
                'cap_significant': sum(''.join('1' if c else '0' for c in a)[1 if p.startswith('▁') else 0:len(p)] == b[1 if p.startswith('▁') else 0:len(p)]
                                       for a, b, p in zip(caps, ref['cap'], ref['tokens']))},
            'tokens': len(ids), 'row37_logits': values[4][0, 4].tolist() if ri == 3 else None,
            'row37_conditioning': float(values[5][0, 4, -1]) if ri == 3 else None})
        (out / 'summary.json').write_text(json.dumps(report, indent=2))
    (out / 'capture-names.json').write_text(json.dumps(capture, indent=2))
    print(json.dumps(report, indent=2))
    # Diagnostic graph/expanded weights stay task-transient on the hosted runner.
    return gate_exit_code(report)


if __name__ == '__main__':
    raise SystemExit(main())
