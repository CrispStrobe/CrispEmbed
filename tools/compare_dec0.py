#!/usr/bin/env python3
"""Compare C++ decoder layer 0 outputs with Python reference."""
import numpy as np, onnx, onnxruntime as ort
from PIL import Image
import os

ONNX_PATH = '/mnt/storage/models/docling-layout-heron/model.onnx'
IMAGE_PATH = '/tmp/layout_test_640.png'

img = Image.open(IMAGE_PATH).convert('RGB')
img_u = np.expand_dims(np.array(img).astype(np.uint8).transpose(2, 0, 1), 0)
sizes = np.array([[640, 640]], dtype=np.int64)

model = onnx.load(ONNX_PATH)
targets = ['layer_norm_2', 'layer_norm_3', 'layer_norm_4', 'layer_norm_5',
           'linear_13', 'linear_14', 'linear_15', 'linear_16', 'linear_17',
           'linear_18', 'linear_19', 'linear_20', 'concat_5', 'stack_2',
           'add_2545', 'add_2801']
for name in targets:
    model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

sess = ort.InferenceSession(model.SerializeToString())
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})

refs = {}
for name, data in zip(output_names, results):
    refs[name] = data.astype(np.float32)

# Print key tensors
for name in targets:
    if name in refs:
        d = refs[name]
        print(f'{name:25s} {str(d.shape):25s} [{d.min():.4f}, {d.max():.4f}]')

# Print initial ref_points
if 'stack_2' in refs:
    rp = refs['stack_2'].reshape(300, 4)
    print(f'\nInitial ref_points [0:3]:')
    for i in range(3):
        print(f'  q{i}: cx={rp[i,0]:.6f} cy={rp[i,1]:.6f} w={rp[i,2]:.6f} h={rp[i,3]:.6f}')

# Self-attn Q/K/V
for name, label in [('linear_13', 'Q'), ('linear_14', 'K'), ('linear_15', 'V')]:
    if name in refs:
        d = refs[name].reshape(300, 256)
        print(f'\nDec0 {label}: [{d.min():.4f}, {d.max():.4f}]')
        print(f'  [0,:4]: {d[0,:4]}')

# Self-attn out_proj
if 'linear_16' in refs:
    d = refs['linear_16'].reshape(300, 256)
    print(f'\nDec0 self-attn out_proj: [{d.min():.4f}, {d.max():.4f}]')
    print(f'  [0,:4]: {d[0,:4]}')

# Norm1
if 'layer_norm_3' in refs:
    d = refs['layer_norm_3'].reshape(300, 256)
    print(f'\nDec0 norm1: [{d.min():.4f}, {d.max():.4f}]')
    print(f'  [0,:4]: {d[0,:4]}')
