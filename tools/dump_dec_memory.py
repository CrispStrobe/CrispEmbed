#!/usr/bin/env python3
"""Dump decoder memory (concat_4), value projection, and dec0 norm outputs for diff."""
import numpy as np
import onnx, onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import os

ONNX_PATH = '/mnt/storage/models/docling-layout-heron/model.onnx'
IMAGE_PATH = '/tmp/layout_test_640.png'
OUTPUT = '/tmp/dec-memory-ref.gguf'

def make_test_image():
    img = Image.new('RGB', (640, 640), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    except:
        font = ImageFont.load_default()
    draw.text((50, 50), 'Section Header', fill=(0,0,0), font=font)
    draw.text((50, 100), 'Body text here.', fill=(40,40,40), font=font)
    draw.rectangle([50, 200, 300, 400], outline=(0,0,0), width=2)
    return img

def main():
    img = Image.open(IMAGE_PATH).convert('RGB') if os.path.exists(IMAGE_PATH) else make_test_image()
    img_u = np.expand_dims(np.array(img).astype(np.uint8).transpose(2, 0, 1), 0)
    sizes = np.array([[640, 640]], dtype=np.int64)

    model = onnx.load(ONNX_PATH)
    g = model.graph

    # Key tensors to capture:
    targets = [
        'concat_4',       # decoder memory [1, 8400, 256]
        'linear_17',      # dec0 value projection [1, 8400, 256]
        'layer_norm_3',   # dec0 norm1
        'layer_norm_4',   # dec0 norm2
        'layer_norm_5',   # dec0 norm3
        'linear_18',      # dec0 sampling offsets [1, 300, 192]
        'linear_19',      # dec0 attention weights [1, 300, 96]
        'linear_20',      # dec0 cross-attn output [1, 300, 256]
        'linear_16',      # dec0 self-attn out_proj
        'linear_13',      # dec0 Q
        'linear_14',      # dec0 K
        'linear_15',      # dec0 V
        'view_237',       # values reshaped [1, 8400, 8, 32]
        'view_238',       # offsets reshaped [1, 300, 8, 12, 2]
        'view_239',       # attn weights reshaped [1, 300, 8, 12]
        'concat_5',       # dec0 sampled values concatenated [8, 32, 300, 12]
        'grid_sampler',   # dec0 grid_sampler level 0
        'grid_sampler_1', # dec0 grid_sampler level 1
        'grid_sampler_2', # dec0 grid_sampler level 2
        'split_with_sizes_1_split_0',  # dec0 grid coords level 0
        'split_with_sizes_1_split_1',  # dec0 grid coords level 1
        'split_with_sizes_1_split_2',  # dec0 grid coords level 2
        'add_2545',       # dec0 self-attn residual
        'add_2801',       # dec0 cross-attn residual
        'stack_2',        # initial ref_points [1, 300, 4]
    ]

    existing = {o.name for o in g.output}
    for name in targets:
        if name not in existing:
            g.output.append(onnx.helper.make_empty_tensor_value_info(name))

    sess = ort.InferenceSession(model.SerializeToString())
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})

    name_to_data = {}
    for name, data in zip(output_names, results):
        name_to_data[name] = data

    import gguf
    writer = gguf.GGUFWriter(OUTPUT, 'dec-memory-ref')
    count = 0
    for name in sorted(name_to_data.keys()):
        data = name_to_data[name]
        if data.dtype in (np.float32, np.float64):
            # Store in the layout C++ expects: [D, N] column-major
            # Python tensors are [batch, N, D] or [batch, H, W, ...] row-major
            arr = data.astype(np.float32)
            print(f"  {name:45s} {str(arr.shape):30s} [{arr.min():.4f}, {arr.max():.4f}]")
            writer.add_tensor(name, arr.flatten(), raw_dtype=gguf.GGMLQuantizationType.F32)
            count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.1f} MB, {count} tensors)")


if __name__ == '__main__':
    main()
