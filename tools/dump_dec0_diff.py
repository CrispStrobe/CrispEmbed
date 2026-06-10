#!/usr/bin/env python3
"""Dump decoder layer 0 intermediates in C++ column-major layout for crispembed_diff.

All tensors are stored as [D, N] column-major (ggml convention) so they can be
compared directly with C++ data via crispembed_diff::Ref::compare().
"""
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

targets = [
    'layer_norm_2',   # enc_proj (query source) [1, 8400, 256]
    'concat_4',       # decoder memory [1, 8400, 256]
    'linear_17',      # dec0 value projection [1, 8400, 256]
    'linear_13',      # dec0 Q [300, 1, 256]
    'linear_14',      # dec0 K [300, 1, 256]
    'linear_15',      # dec0 V [300, 1, 256]
    'linear_16',      # dec0 self-attn out_proj [300, 256]
    'add_2545',       # dec0 self-attn residual+out [1, 300, 256]
    'layer_norm_3',   # dec0 norm1 [1, 300, 256]
    'linear_18',      # dec0 sampling offsets [1, 300, 192]
    'linear_19',      # dec0 attention weights [1, 300, 96]
    'linear_20',      # dec0 cross-attn output_proj [1, 300, 256]
    'add_2801',       # dec0 cross-attn residual [1, 300, 256]
    'layer_norm_4',   # dec0 norm2 [1, 300, 256]
    'layer_norm_5',   # dec0 norm3 [1, 300, 256]
    'concat_5',       # dec0 sampled values [8, 32, 300, 12]
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
writer = gguf.GGUFWriter(OUTPUT, 'dec0-diff-ref')

for name in targets:
    if name not in refs:
        print(f'  MISSING: {name}')
        continue
    d = refs[name]
    print(f'  {name:25s} {str(d.shape):25s} [{d.min():.4f}, {d.max():.4f}]')

    # Convert to [D, N] column-major for crispembed_diff
    # Python tensors are row-major [batch, N, D] or [N, batch, D] etc.
    if name in ('concat_4', 'linear_17'):
        # [1, 8400, 256] → [256, 8400]
        flat = d.reshape(-1, 256).T.flatten()  # [256, 8400] col-major
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name in ('layer_norm_2',):
        flat = d.reshape(-1, 256).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name in ('linear_13', 'linear_14', 'linear_15'):
        # [300, 1, 256] → [256, 300]
        flat = d.reshape(300, 256).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name == 'linear_16':
        # [300, 256] → [256, 300]
        flat = d.reshape(300, 256).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name in ('layer_norm_3', 'layer_norm_4', 'layer_norm_5',
                  'add_2545', 'add_2801', 'linear_20'):
        # [1, 300, 256] → [256, 300]
        flat = d.reshape(300, 256).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name in ('linear_18',):
        # [1, 300, 192] → [192, 300]
        flat = d.reshape(300, 192).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name in ('linear_19',):
        # [1, 300, 96] → [96, 300]
        flat = d.reshape(300, 96).T.flatten()
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif name == 'concat_5':
        # [8, 32, 300, 12] — store flat
        writer.add_tensor(name, d.flatten(), raw_dtype=gguf.GGMLQuantizationType.F32)
    else:
        writer.add_tensor(name, d.flatten(), raw_dtype=gguf.GGMLQuantizationType.F32)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f'\nWrote {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.1f} MB)')
