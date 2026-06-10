#!/usr/bin/env python3
"""Dump all 6 decoder layers' norm1/norm2/norm3 + final scores in col-major for crispembed_diff."""
import numpy as np
import onnx, onnxruntime as ort
from PIL import Image
import os

ONNX_PATH = '/mnt/storage/models/docling-layout-heron/model.onnx'
IMAGE_PATH = '/tmp/layout_test_640.png'
OUTPUT = '/tmp/dec0-diff-ref.gguf'

img = Image.open(IMAGE_PATH).convert('RGB')
img_u = np.expand_dims(np.array(img).astype(np.uint8).transpose(2, 0, 1), 0)
sizes = np.array([[640, 640]], dtype=np.int64)

model = onnx.load(ONNX_PATH)

# All decoder norms: layer_norm_3..layer_norm_20 (3 per layer × 6 layers = 18)
# layer_norm_2 = enc_proj, layer_norm_3/4/5 = dec0 norm1/2/3, etc.
targets = ['layer_norm_2']
for i in range(6):
    for j in range(3):
        targets.append(f'layer_norm_{3 + i*3 + j}')

# Also add: dec0 intermediates, initial queries/pos, ref_points, final scores
targets += [
    'gather_2', 'linear_12', 'sigmoid',  # initial queries, pos_enc, ref_points
    'concat_4', 'linear_17',  # memory, dec0 values
    'linear_13', 'linear_14', 'linear_15', 'linear_16',  # dec0 Q/K/V/sa_out
    'add_2545', 'linear_18', 'linear_19', 'linear_20', 'add_2801',  # dec0 cross-attn
    'scores',  # final detection scores [1, 300]
]

existing = {o.name for o in model.graph.output}
for name in targets:
    if name not in existing:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

sess = ort.InferenceSession(model.SerializeToString())
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})

refs = {}
for name, data in zip(output_names, results):
    refs[name] = data.astype(np.float32)

import gguf
writer = gguf.GGUFWriter(OUTPUT, 'dec-all-layers-ref')

for name in targets:
    if name not in refs:
        print(f'  MISSING: {name}')
        continue
    d = refs[name]

    # Convert to col-major [D, N] for crispembed_diff
    if name.startswith('layer_norm_') and name != 'layer_norm_2':
        flat = d.reshape(-1, 256).T.flatten()
    elif name == 'layer_norm_2':
        flat = d.reshape(-1, 256).T.flatten()
    elif name in ('concat_4', 'linear_17'):
        flat = d.reshape(-1, 256).T.flatten()
    elif name in ('linear_13', 'linear_14', 'linear_15'):
        flat = d.reshape(300, 256).T.flatten()
    elif name == 'linear_16':
        flat = d.reshape(300, 256).T.flatten()
    elif name in ('gather_2', 'linear_12', 'linear_20', 'add_2545', 'add_2801'):
        flat = d.reshape(-1, 256).T.flatten()
    elif name == 'linear_18':
        flat = d.reshape(300, 192).T.flatten()
    elif name == 'linear_19':
        flat = d.reshape(300, 96).T.flatten()
    elif name == 'sigmoid':
        flat = d.reshape(300, 4).flatten()
    elif name == 'scores':
        flat = d.flatten()  # [300]
    else:
        flat = d.flatten()

    writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    print(f'  {name:25s} {str(d.shape):25s} [{d.min():.4f}, {d.max():.4f}]')

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f'\nWrote {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.1f} MB, {len(targets)} tensors)')
