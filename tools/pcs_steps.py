#!/usr/bin/env python3
"""Shared PCS operator inventory: names correspond to actual ONNX nodes.

No expected arithmetic is implemented here. Native-only quantities are named
separately; in particular the native embedding partial sum uses another order.
"""
from __future__ import annotations

EMBED = {
    'input.positions': '/bert_model/embeddings/Add_output_0',
    'input.types': '/bert_model/Expand_output_0',
    'emb.token': '/bert_model/embeddings/word_embeddings/Gather_output_0',
    'emb.type': '/bert_model/embeddings/token_type_embeddings/Gather_output_0',
    'emb.position': '/bert_model/embeddings/position_embeddings/Gather_output_0',
    'emb.sum': '/bert_model/embeddings/Add_2_output_0',
    'emb.output': '/bert_model/embeddings/LayerNorm/LayerNormalization_output_0',
}
LAYER = {
    'q.matmul': 'attention/self/query/MatMul', 'q.bias': 'attention/self/query/Add',
    'k.matmul': 'attention/self/key/MatMul', 'k.bias': 'attention/self/key/Add',
    'v.matmul': 'attention/self/value/MatMul', 'v.bias': 'attention/self/value/Add',
    'attention.qk': 'attention/self/MatMul',
    'attention.probabilities': 'attention/self/Softmax',
    'attention.context': 'attention/self/Reshape_3',
    'attention.matmul': 'attention/output/dense/MatMul',
    'attention.bias': 'attention/output/dense/Add',
    'attention.residual': 'attention/output/Add',
    'attention.output': 'attention/output/LayerNorm/LayerNormalization',
    'ffn.up.matmul': 'intermediate/dense/MatMul', 'ffn.up.bias': 'intermediate/dense/Add',
    'ffn.gelu': 'intermediate/intermediate_act_fn/Mul_1',
    'ffn.down.matmul': 'output/dense/MatMul', 'ffn.down.bias': 'output/dense/Add',
    'ffn.residual': 'output/Add', 'output': 'output/LayerNorm/LayerNormalization',
}


def inventory(layers=12):
    result = dict(EMBED)
    for layer in range(layers):
        for key, node in LAYER.items():
            result[f'layer.{layer}.{key}'] = f'/bert_model/encoder/layer.{layer}/{node}_output_0'
    for native, original in [('post', '_punct_head_post'), ('pre', '_punct_head_pre')]:
        for key, suffix in [('fc1.matmul', '_linears.0/MatMul'), ('fc1.bias', '_linears.0/Add'),
                            ('relu', '_act/Relu'), ('fc2.matmul', '_linears.1/MatMul'), ('output', '_linears.1/Add')]:
            result[f'head.{native}.{key}'] = f'/_decoder/{original}/{suffix}_output_0'
    for native, original, concat in [('sbd', '_seg_head', 'Concat'), ('cap', '_cap_head', 'Concat_4')]:
        result[f'head.{native}.input'] = f'/_decoder/{concat}_output_0'
        for key, suffix in [('fc1.bias', '_linears.0/Add'), ('relu', '_act/Relu'), ('output', '_linears.1/Add')]:
            result[f'head.{native}.{key}'] = f'/_decoder/{original}/{suffix}_output_0'
    return result


def write_archive(path, tensors, metadata):
    """Same GGUF reference format as tools/dump_reference.py; no torch import."""
    import gguf
    import numpy as np
    writer = gguf.GGUFWriter(str(path), 'crispembed-ref')
    for key, value in metadata.items():
        writer.add_string(key, str(value))
    for name, array in tensors.items():
        array = np.asarray(array)
        writer.add_tensor(name, np.ascontiguousarray(array, dtype=np.int32 if array.dtype == np.int32 else np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
