#!/usr/bin/env python3
"""Trace the AIFI block in the ONNX graph and dump all intermediates."""

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import os, sys

ONNX_PATH = '/mnt/storage/models/docling-layout-heron/model.onnx'
IMAGE_PATH = '/tmp/layout_test_640.png'
OUTPUT_PATH = '/tmp/aifi-full-ref.gguf'


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
    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH).convert('RGB')
    else:
        img = make_test_image()
        img.save(IMAGE_PATH)

    img_u = np.expand_dims(np.array(img).astype(np.uint8).transpose(2, 0, 1), 0)
    sizes = np.array([[640, 640]], dtype=np.int64)

    model = onnx.load(ONNX_PATH)
    g = model.graph

    # Find all consumers of split_split_0 (Q weight after in_proj_weight Split)
    print("=== Consumers of split_split_0 (Q weight) ===")
    for node in g.node:
        if 'split_split_0' in node.input:
            print(f"  {node.op_type:15s} {node.name}")
            print(f"    inputs:  {list(node.input)}")
            print(f"    outputs: {list(node.output)}")

    # Find all consumers of split_1_split_0 (Q bias)
    print("\n=== Consumers of split_1_split_0 (Q bias) ===")
    for node in g.node:
        if 'split_1_split_0' in node.input:
            print(f"  {node.op_type:15s} {node.name}")
            print(f"    inputs:  {list(node.input)}")
            print(f"    outputs: {list(node.output)}")

    # More broadly: find all nodes that reference the AIFI weight prefix
    prefix = 'model.encoder.encoder.0'
    print(f"\n=== All nodes referencing {prefix} ===")
    for node in g.node:
        for inp in node.input:
            if prefix in inp:
                print(f"  {node.op_type:15s} {node.name}")
                print(f"    inputs:  {list(node.input)}")
                print(f"    outputs: {list(node.output)}")
                break

    # Find the ip5 conv output: getitem_6
    print("\n=== Tracing from getitem_6 (ip5) forward ===")
    # Build forward map: input_name -> list of (node, output_names)
    fwd = {}
    for node in g.node:
        for inp in node.input:
            if inp not in fwd:
                fwd[inp] = []
            fwd[inp].append(node)

    # BFS from getitem_6
    visited = set()
    queue = ['getitem_6']
    targets = []
    count = 0
    while queue and count < 80:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        targets.append(name)
        count += 1
        if name in fwd:
            for node in fwd[name]:
                for out in node.output:
                    if out not in visited:
                        queue.append(out)
                        # Print the chain
                        print(f"  {node.op_type:15s} {name:35s} -> {out}")

    print(f"\n{len(targets)} tensors in forward trace from ip5")

    # Add all targets as graph outputs
    existing = {o.name for o in g.output}
    for name in targets:
        if name not in existing:
            g.output.append(onnx.helper.make_empty_tensor_value_info(name))

    # Run inference
    print("\nRunning inference...")
    sess = ort.InferenceSession(model.SerializeToString())
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})

    name_to_data = {}
    for name, data in zip(output_names, results):
        name_to_data[name] = data

    # Print shapes and ranges
    print(f"\nCaptured tensors ({len(name_to_data)}):")
    for name in targets:
        if name in name_to_data:
            d = name_to_data[name]
            if d.dtype in (np.float32, np.float64):
                print(f"  {name:40s} {str(d.shape):25s} [{d.min():.4f}, {d.max():.4f}]")
            else:
                print(f"  {name:40s} {str(d.shape):25s} dtype={d.dtype}")

    # Save to GGUF
    import gguf
    writer = gguf.GGUFWriter(OUTPUT_PATH, 'aifi-full-ref')
    count = 0
    for name in targets:
        if name in name_to_data:
            data = name_to_data[name]
            if data.dtype in (np.float32, np.float64):
                writer.add_tensor(name, data.astype(np.float32).flatten(),
                                  raw_dtype=gguf.GGMLQuantizationType.F32)
                count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB, {count} tensors)")


if __name__ == '__main__':
    main()
