#!/usr/bin/env python3
"""Dump per-layer ref_points and pos_enc from Python for comparison."""
import numpy as np
import onnx, onnxruntime as ort
from PIL import Image
import os

ONNX_PATH = '/mnt/storage/models/docling-layout-heron/model.onnx'
IMAGE_PATH = '/tmp/layout_test_640.png'
OUTPUT = '/tmp/refpoints-per-layer.gguf'

img = Image.open(IMAGE_PATH).convert('RGB')
img_u = np.expand_dims(np.array(img).astype(np.uint8).transpose(2, 0, 1), 0)
sizes = np.array([[640, 640]], dtype=np.int64)

model = onnx.load(ONNX_PATH)
g = model.graph

# Find sigmoid nodes that produce refined ref_points after each decoder layer
# Pattern: at end of each decoder layer, there's a Sigmoid that produces new ref_points
# Also find query_pos_head outputs (linear_12-like) for each layer

# Find all Sigmoid nodes (these produce ref_points)
sigmoid_nodes = [(i, node) for i, node in enumerate(g.node) if node.op_type == 'Sigmoid']
print(f'Found {len(sigmoid_nodes)} Sigmoid nodes')

# Find all nodes named like "linear_12" pattern — these are query_pos_head outputs
# The query_pos_head bias is reused across layers, so look for Add nodes with qpos bias
targets = []
for node in g.node:
    if node.op_type == 'Add':
        for inp in node.input:
            if 'query_pos_head.layers.1.bias' in inp:
                targets.append(node.output[0])
                print(f'  qpos output: {node.output[0]} (from {list(node.input)})')

# Also find all Sigmoid outputs that could be ref_points
# The initial ref_points come from sigmoid applied to gather output
# Each refinement: sigmoid(inv_sigmoid(prev_ref) + bbox_delta)
# Look for Sigmoid nodes whose output feeds into Unsqueeze/Slice patterns
for node in g.node:
    if node.op_type == 'Sigmoid':
        out = node.output[0]
        targets.append(out)

# Add all targets as graph outputs
existing = {o.name for o in model.graph.output}
for name in targets:
    if name not in existing:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

sess = ort.InferenceSession(model.SerializeToString())
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})

print(f'\nCaptured {len(results)} outputs')
# Find ref_points candidates (shape [1, 300, 4] or [300, 4])
# and pos_enc candidates (shape [1, 300, 256])
import gguf
writer = gguf.GGUFWriter(OUTPUT, 'refpoints-per-layer')
count = 0
for name, data in zip(output_names, results):
    d = data.astype(np.float32)
    shape = d.shape
    if shape in ((1, 300, 4), (300, 4)):
        flat = d.reshape(300, 4).flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f'  REF_POINTS {name:30s} {str(shape):20s} [{d.min():.6f}, {d.max():.6f}]')
        count += 1
    elif shape == (1, 300, 256):
        flat = d.reshape(300, 256).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f'  POS_ENC    {name:30s} {str(shape):20s} [{d.min():.6f}, {d.max():.6f}]')
        count += 1

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f'\nWrote {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.1f} MB, {count} tensors)')
