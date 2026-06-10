#!/usr/bin/env python3
"""Trace the offset → grid pipeline in decoder layer 0 of the ONNX model."""

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import os

ONNX_PATH = '/mnt/storage/models/docling-layout-heron/model.onnx'
IMAGE_PATH = '/tmp/layout_test_640.png'


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

    # Build maps
    out_to_node = {}
    for node in g.node:
        for out in node.output:
            out_to_node[out] = node

    # Find the grid coordinate computation for decoder layer 0
    # We know:
    # - linear_18: sampling offsets [1, 300, 192]
    # - view_238: reshape to [1, 300, 8, 12, 2]
    # - split_with_sizes_1_split_0/1/2: grid coords [8, 300, 4, 2] per level

    # Trace the path from linear_18/view_238 to split_with_sizes_1
    # Find the SplitWithSizes node that produces split_with_sizes_1_split_0
    for node in g.node:
        if 'split_with_sizes_1_split_0' in node.output:
            print("Split node for grid coords:")
            print(f"  op: {node.op_type}")
            print(f"  inputs: {list(node.input)}")
            print(f"  outputs: {list(node.output)}")

            # Trace back to find how the grid was computed
            grid_input = node.input[0]
            print(f"\n  Grid input: {grid_input}")

            # Trace back through the chain
            chain = []
            current = grid_input
            for _ in range(20):
                if current not in out_to_node:
                    break
                n = out_to_node[current]
                chain.append((current, n))
                # Pick the first non-weight input to trace further
                for inp in n.input:
                    if inp in out_to_node:
                        current = inp
                        break
                else:
                    break

            print("\n  Trace back from grid to offsets:")
            for name, n in chain:
                attrs = {}
                for attr in n.attribute:
                    if attr.type == 1:  # FLOAT
                        attrs[attr.name] = attr.f
                    elif attr.type == 2:  # INT
                        attrs[attr.name] = attr.i
                    elif attr.type == 7:  # INTS
                        attrs[attr.name] = list(attr.ints)
                print(f"    {n.op_type:15s} {str(list(n.input)):60s} -> {name}")
                if attrs:
                    print(f"      attrs: {attrs}")
            break

    # Now add ALL intermediate outputs between linear_18 and grid_sampler_0
    # to capture the actual values
    targets = set()
    # Also trace other branches of the SplitWithSizes input
    for name, n in chain:
        targets.add(name)
        for inp in n.input:
            if inp in out_to_node:
                targets.add(inp)
    # Add the grid coords
    targets.add('split_with_sizes_1_split_0')
    targets.add('split_with_sizes_1_split_1')
    targets.add('split_with_sizes_1_split_2')
    # Also add the reference points and offsets
    targets.add('linear_18')
    targets.add('view_238')
    targets.add('linear_19')
    targets.add('view_239')

    # Add graph outputs
    existing = {o.name for o in g.output}
    for name in targets:
        if name not in existing:
            g.output.append(onnx.helper.make_empty_tensor_value_info(name))

    # Run inference
    sess = ort.InferenceSession(model.SerializeToString())
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(None, {'images': img_u, 'orig_target_sizes': sizes})

    name_to_data = {}
    for name, data in zip(output_names, results):
        name_to_data[name] = data

    # Print all captured intermediate tensors
    print(f"\nCaptured intermediate tensors:")
    for name in sorted(name_to_data.keys()):
        d = name_to_data[name]
        if d.dtype in (np.float32, np.float64):
            print(f"  {name:45s} {str(d.shape):30s} [{d.min():.6f}, {d.max():.6f}]")
        else:
            print(f"  {name:45s} {str(d.shape):30s} dtype={d.dtype}")

    # Check specific values to understand the offset formula
    if 'view_238' in name_to_data:
        offsets = name_to_data['view_238']  # [1, 300, 8, 12, 2]
        print(f"\nSampling offsets (view_238): {offsets.shape}")
        print(f"  Range: [{offsets.min():.4f}, {offsets.max():.4f}]")
        # Show first query, first head, all level*point pairs
        print(f"  Query 0, Head 0:")
        for lp in range(12):
            lv = lp // 4
            pt = lp % 4
            dx, dy = offsets[0, 0, 0, lp]
            print(f"    lv={lv} pt={pt}: dx={dx:.4f} dy={dy:.4f}")

    if 'split_with_sizes_1_split_0' in name_to_data:
        grid_l0 = name_to_data['split_with_sizes_1_split_0']  # [8, 300, 4, 2]
        print(f"\nGrid coords level 0 (split_with_sizes_1_split_0): {grid_l0.shape}")
        print(f"  Range: [{grid_l0.min():.4f}, {grid_l0.max():.4f}]")
        print(f"  Head 0, Query 0:")
        for pt in range(4):
            gx, gy = grid_l0[0, 0, pt]
            print(f"    pt={pt}: gx={gx:.6f} gy={gy:.6f}")
            # Convert to pixel coords (align_corners=False, level 0 is 80x80)
            px = (gx + 1) * 80 / 2 - 0.5
            py = (gy + 1) * 80 / 2 - 0.5
            print(f"           pixel: px={px:.4f} py={py:.4f}")

    if 'split_with_sizes_1_split_2' in name_to_data:
        grid_l2 = name_to_data['split_with_sizes_1_split_2']  # [8, 300, 4, 2]
        print(f"\nGrid coords level 2 (split_with_sizes_1_split_2): {grid_l2.shape}")
        print(f"  Range: [{grid_l2.min():.4f}, {grid_l2.max():.4f}]")
        print(f"  Head 0, Query 0:")
        for pt in range(4):
            gx, gy = grid_l2[0, 0, pt]
            print(f"    pt={pt}: gx={gx:.6f} gy={gy:.6f}")
            px = (gx + 1) * 20 / 2 - 0.5
            py = (gy + 1) * 20 / 2 - 0.5
            print(f"           pixel: px={px:.4f} py={py:.4f}")


if __name__ == '__main__':
    main()
