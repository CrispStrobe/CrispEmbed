#!/usr/bin/env python3
"""Convert face detection/recognition ONNX models to GGUF format.

Supports:
  - SCRFD (det_10g.onnx) — face detection (Apache 2.0)
  - AuraFace (glintr100.onnx) — face recognition 512-D (Apache 2.0)
  - SFace (face_recognition_sface_2021dec.onnx) — face recognition 128-D (Apache 2.0)

These are CNN models (ResNet/MobileFaceNet) stored in ONNX format.
We extract all weight tensors and store them in GGUF with their
original names, preserving the conv/bn/fc structure.

    pip install onnx numpy gguf
    python models/convert-face-to-gguf.py \
        --onnx det_10g.onnx \
        --output scrfd-det-10g.gguf \
        --model-type detection

Model types:
  detection   — SCRFD/YuNet (outputs bboxes + landmarks)
  recognition — AuraFace/SFace (outputs embedding vector)
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def main():
    p = argparse.ArgumentParser(description="Convert face ONNX to GGUF")
    p.add_argument("--onnx", required=True, help="Input ONNX model path")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--model-type", choices=["detection", "recognition"],
                   default="recognition", help="Model type")
    p.add_argument("--model-name", default=None, help="Model name for metadata")
    args = p.parse_args()

    print(f"Loading ONNX: {args.onnx}")
    model = onnx.load(args.onnx)
    graph = model.graph

    # Analyze model
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
              for i in graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
               for o in graph.output]

    from collections import Counter
    ops = Counter(n.op_type for n in graph.node)

    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {len(outputs)} tensors")
    print(f"  Nodes: {len(graph.node)}")
    print(f"  Ops: {dict(ops.most_common(10))}")

    # Determine architecture from ops
    has_conv = ops.get("Conv", 0) > 0
    has_bn = ops.get("BatchNormalization", 0) > 0
    n_conv = ops.get("Conv", 0)

    # Detect output embedding dim for recognition models
    embed_dim = 0
    if args.model_type == "recognition" and outputs:
        for name, shape in outputs:
            if len(shape) == 2 and shape[1] > 0:
                embed_dim = shape[1]
                break

    print(f"  Type: {args.model_type}")
    if embed_dim:
        print(f"  Embedding dim: {embed_dim}")
    print(f"  Conv layers: {n_conv}")
    if has_bn:
        print(f"  BatchNorm layers: {ops['BatchNormalization']}")

    # Extract weight tensors from initializers
    weights = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        weights[init.name] = arr

    print(f"  Weight tensors: {len(weights)}")

    # Try to fold BatchNorm into Conv where possible
    # Pattern: Conv → BatchNorm
    # Folded conv: w_new = w * gamma / sqrt(var + eps)
    #              b_new = (b - mean) * gamma / sqrt(var + eps) + beta
    bn_folded = 0
    if has_bn:
        # Build node adjacency
        output_to_node = {}
        for node in graph.node:
            for out in node.output:
                output_to_node[out] = node

        for node in graph.node:
            if node.op_type != "BatchNormalization":
                continue
            bn_input = node.input[0]
            if bn_input not in output_to_node:
                continue
            conv_node = output_to_node[bn_input]
            if conv_node.op_type != "Conv":
                continue

            # Get BN params
            scale_name = node.input[1]  # gamma
            bias_name = node.input[2]   # beta
            mean_name = node.input[3]
            var_name = node.input[4]

            if not all(n in weights for n in [scale_name, bias_name, mean_name, var_name]):
                continue

            gamma = weights[scale_name]
            beta = weights[bias_name]
            mean = weights[mean_name]
            var = weights[var_name]

            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f

            # Get conv weight
            conv_w_name = conv_node.input[1]
            if conv_w_name not in weights:
                continue

            conv_w = weights[conv_w_name]
            conv_b = weights.get(conv_node.input[2] if len(conv_node.input) > 2 else None, None)

            # Fold: w_new[oc] = w[oc] * gamma[oc] / sqrt(var[oc] + eps)
            inv_std = gamma / np.sqrt(var + eps)
            n_oc = conv_w.shape[0]

            # Reshape for broadcasting: [OC, 1, 1, 1] for 4D conv weights
            shape = [n_oc] + [1] * (conv_w.ndim - 1)
            weights[conv_w_name] = conv_w * inv_std.reshape(shape)

            # Fold bias
            if conv_b is not None:
                new_bias = (conv_b - mean) * inv_std + beta
            else:
                new_bias = -mean * inv_std + beta

            # Store folded bias under conv bias name
            if len(conv_node.input) > 2:
                weights[conv_node.input[2]] = new_bias
            else:
                # Create new bias name — append _bias suffix
                bias_key = conv_w_name + "_bias"
                weights[bias_key] = new_bias

            # Remove BN params from weights (they're folded)
            for k in [scale_name, bias_name, mean_name, var_name]:
                if k in weights:
                    del weights[k]
            bn_folded += 1

    if bn_folded:
        print(f"  BatchNorm folded into Conv: {bn_folded}")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="cnn")

    # Metadata
    writer.add_string("cnn.model_type", args.model_type)
    writer.add_string("cnn.model_name", args.model_name or Path(args.onnx).stem)
    writer.add_uint32("cnn.num_conv_layers", n_conv)
    if embed_dim:
        writer.add_uint32("cnn.embedding_dim", embed_dim)

    # Input shape
    if inputs and len(inputs[0][1]) == 4:
        _, c, h, w = inputs[0][1]
        if h > 0:
            writer.add_uint32("cnn.input_height", h)
        if w > 0:
            writer.add_uint32("cnn.input_width", w)
        writer.add_uint32("cnn.input_channels", c or 3)

    # Output info
    writer.add_uint32("cnn.num_outputs", len(outputs))
    for i, (name, shape) in enumerate(outputs):
        writer.add_string(f"cnn.output.{i}.name", name)

    # Precompute unfoldable BN as scale+shift weight tensors
    # BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    #       = scale * x + shift
    import copy
    node_descs_updated = []
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            gamma_name = node.input[1]
            beta_name = node.input[2]
            mean_name = node.input[3]
            var_name = node.input[4]
            if all(n in weights for n in [gamma_name, beta_name, mean_name, var_name]):
                gamma = weights[gamma_name]
                beta = weights[beta_name]
                mean = weights[mean_name]
                var = weights[var_name]
                eps = 1e-5
                for attr in node.attribute:
                    if attr.name == "epsilon": eps = attr.f
                inv_std = gamma / np.sqrt(var + eps)
                scale = inv_std
                shift = beta - mean * inv_std
                # Store as new tensors
                sc_name = f"bn_scale_{node.output[0].replace('/', '_')}"
                sh_name = f"bn_shift_{node.output[0].replace('/', '_')}"
                weights[sc_name] = scale.astype(np.float32)
                weights[sh_name] = shift.astype(np.float32)
                # Remove original BN params
                for k in [gamma_name, beta_name, mean_name, var_name]:
                    weights.pop(k, None)
                # Replace BN node with Mul+Add in graph description
                # (the graph replayer handles "BNPrecomputed" as mul(scale)+add(shift))
                attrs = ""
                desc = f"BNPrecomputed:{node.input[0]},{sc_name},{sh_name}:{node.output[0]}"
                node_descs_updated.append(desc)
                bn_folded += 1
                continue
        # Original node
        attrs = ""
        if node.op_type == "Conv":
            stride = [1, 1]; pads = [0, 0, 0, 0]; group = 1
            for a in node.attribute:
                if a.name == "strides": stride = list(a.ints)
                if a.name == "pads": pads = list(a.ints)
                if a.name == "group": group = a.i
            attrs = f"[s{stride[0]}p{pads[0]}g{group}]"
        elif node.op_type in ("AveragePool", "MaxPool"):
            kernel = [1, 1]; stride = [1, 1]; pads = [0, 0, 0, 0]
            for a in node.attribute:
                if a.name == "kernel_shape": kernel = list(a.ints)
                if a.name == "strides": stride = list(a.ints)
                if a.name == "pads": pads = list(a.ints)
            attrs = f"[k{kernel[0]}s{stride[0]}p{pads[0]}]"
        desc = f"{node.op_type}{attrs}:{','.join(node.input)}:{','.join(node.output)}"
        node_descs_updated.append(desc)
    node_descs = node_descs_updated

    # Store all weight tensors
    stored = 0
    for name, arr in sorted(weights.items()):
        # Clean up name for GGUF (replace special chars)
        clean_name = name.replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "")
        writer.add_tensor(clean_name, arr)
        stored += 1

    print(f"  Stored {stored} tensors in GGUF")

    # Store the ONNX graph topology (already built in node_descs above)
    writer.add_string("cnn.graph_nodes", "|".join(node_descs))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB, {stored} tensors)")


if __name__ == "__main__":
    main()
