#!/usr/bin/env python3
"""Dump per-stage reference activations for RT-DETRv2 layout detection.

Runs the ONNX model on a test image, captures backbone/encoder/decoder
intermediates, writes to GGUF for comparison with the C++ implementation.

Usage:
    python tools/dump_layout_reference.py \
        --onnx /path/to/model.onnx \
        --output /path/to/layout-ref.gguf \
        [--image /path/to/test.png]
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def make_test_image(h=640, w=640):
    """Create a synthetic document image."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    except:
        font = ImageFont.load_default()
    draw.text((50, 50), 'Section Header', fill=(0,0,0), font=font)
    draw.text((50, 100), 'Body text here.', fill=(40,40,40), font=font)
    draw.rectangle([50, 200, 300, 400], outline=(0,0,0), width=2)
    return np.array(img)


def preprocess(img, target_h=640, target_w=640):
    """Preprocess image for RT-DETRv2: resize + uint8 CHW."""
    from PIL import Image
    pil = Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(pil).astype(np.uint8)
    return np.expand_dims(arr.transpose(2, 0, 1), 0)  # [1, 3, H, W]


def preprocess_float(img, target_h=640, target_w=640,
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Preprocess for C++ comparison: float32 CHW normalized."""
    from PIL import Image
    pil = Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - mean[c]) / std[c]
    return arr.transpose(2, 0, 1)  # [3, H, W]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image", default=None)
    args = p.parse_args()

    import onnxruntime as ort
    import onnx

    # Load image
    if args.image:
        from PIL import Image
        img = np.array(Image.open(args.image).convert("RGB"))
    else:
        img = make_test_image()
    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    # Preprocess
    img_uint8 = preprocess(img)
    img_float = preprocess_float(img)
    print(f"Input uint8: {img_uint8.shape}, float: {img_float.shape}")

    # Load ONNX model and add intermediate outputs
    model = onnx.load(args.onnx)
    g = model.graph

    # Find key intermediate tensor names by tracing the graph
    intermediates = {}

    # Run full model first to get final output
    sess = ort.InferenceSession(args.onnx)
    sizes = np.array([[640, 640]], dtype=np.int64)
    labels, boxes, scores = sess.run(None, {'images': img_uint8, 'orig_target_sizes': sizes})

    intermediates["final_labels"] = labels[0].astype(np.int32)
    intermediates["final_boxes"] = boxes[0].astype(np.float32)
    intermediates["final_scores"] = scores[0].astype(np.float32)
    print(f"Detections (>0.3): {(scores[0] > 0.3).sum()}")

    # Now extract intermediate outputs by adding them to the ONNX graph
    # Find backbone stage outputs, encoder outputs, decoder layer outputs
    node_outputs = {}
    for i, node in enumerate(g.node):
        for out in node.output:
            node_outputs[out] = (i, node.op_type)

    # Key stages to capture:
    # 1. After each backbone stage (Conv+Relu after the last block)
    # 2. After input_proj
    # 3. After encoder (FPN/PAN/AIFI)
    # 4. After each decoder layer
    # 5. Final classification and bbox outputs

    # Add intermediate outputs to the model
    # We'll capture specific named nodes
    target_names = set()

    # Find backbone stage outputs by searching for specific patterns
    for node in g.node:
        # Backbone outputs: the last Relu in each stage
        if node.op_type == 'Relu':
            out = node.output[0]
            # These will be captured but we need to find the right ones
            pass

    # Simpler: capture ALL Relu outputs and filter by shape later
    # Or: run with all intermediate outputs enabled
    # For now, just capture the encoder final outputs and decoder outputs

    # Use onnxruntime's ability to request arbitrary intermediate tensors
    # by specifying them as output names

    # Find the input_proj outputs (after the 1x1 convs that reduce to 256ch)
    target_outputs = []
    for node in g.node:
        # Look for nodes whose inputs reference encoder.input_proj
        for inp in node.input:
            if 'encoder.input_proj' in inp and node.op_type in ('Conv', 'Add'):
                target_outputs.append(node.output[0])
                break
        # Decoder layer norm outputs (after each layer)
        for inp in node.input:
            if 'decoder.layers.0.norm3' in inp and node.op_type == 'LayerNormalization':
                target_outputs.append(node.output[0])
                break
            if 'decoder.layers.5.norm3' in inp and node.op_type == 'LayerNormalization':
                target_outputs.append(node.output[0])
                break

    # Add these as graph outputs
    shape_info = onnx.helper.make_tensor_value_info
    for name in target_outputs:
        model.graph.output.append(
            onnx.helper.make_empty_tensor_value_info(name))

    # Run with intermediates
    try:
        sess2 = ort.InferenceSession(model.SerializeToString())
        all_outputs = sess2.run(None, {'images': img_uint8, 'orig_target_sizes': sizes})
        output_names = [o.name for o in sess2.get_outputs()]

        for name, data in zip(output_names, all_outputs):
            if name in ('labels', 'boxes', 'scores'):
                continue
            intermediates[f"onnx_{name}"] = data[0].astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
            print(f"  {name}: {data.shape}")
    except Exception as e:
        print(f"Intermediate extraction failed: {e}")

    # Also store the preprocessed input (float32) for C++ comparison
    intermediates["input_float"] = img_float.astype(np.float32)

    # Write GGUF
    import gguf
    writer = gguf.GGUFWriter(args.output, "layout-ref")
    writer.add_string("layout.ref.onnx", str(args.onnx))
    writer.add_string("layout.ref.image", args.image or "synthetic")

    for name, arr in intermediates.items():
        if isinstance(arr, np.ndarray):
            writer.add_tensor(name, arr.astype(np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB, {len(intermediates)} tensors)")
    for name in sorted(intermediates.keys()):
        arr = intermediates[name]
        if isinstance(arr, np.ndarray):
            print(f"  {name}: {arr.shape} [{arr.min():.4f}, {arr.max():.4f}]")


if __name__ == "__main__":
    main()
