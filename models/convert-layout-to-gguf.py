#!/usr/bin/env python3
"""Convert document layout detection ONNX model to GGUF format.

Supports RT-DETRv2-based layout models (e.g., docling-layout-heron).
Uses the same ONNX graph replay approach as convert-face-to-gguf.py —
stores all weight tensors + a serialized graph topology string, which
the C++ runtime replays node-by-node using ggml ops.

Architecture: ResNet-50 backbone + hybrid encoder (FPN/PAN) +
transformer decoder → bbox + class predictions (300 queries, 17 classes).

Usage:
    pip install gguf numpy onnx
    python models/convert-layout-to-gguf.py \
        --onnx /path/to/model.onnx \
        --output layout-heron-f32.gguf
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def main():
    p = argparse.ArgumentParser(description="Convert layout detection ONNX to GGUF")
    p.add_argument("--onnx", required=True, help="Input ONNX model path")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--model-name", default=None, help="Model name for metadata")
    p.add_argument("--fp16", action="store_true", help="Store weights in FP16")
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
    print(f"  Ops: {dict(ops.most_common(15))}")

    # Extract weight tensors
    weights = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init).astype(np.float32)
        weights[init.name] = arr

    print(f"  Weight tensors: {len(weights)}")
    total_params = sum(np.prod(v.shape) for v in weights.values())
    print(f"  Total params: {total_params:,}")

    # Try to fold BatchNorm into Conv where possible
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    bn_folded = 0
    for node in graph.node:
        if node.op_type != "BatchNormalization":
            continue
        bn_input = node.input[0]
        if bn_input not in output_to_node:
            continue
        conv_node = output_to_node[bn_input]
        if conv_node.op_type != "Conv":
            continue

        gamma = weights.get(node.input[1])
        beta = weights.get(node.input[2])
        mean = weights.get(node.input[3])
        var = weights.get(node.input[4])
        if gamma is None or beta is None or mean is None or var is None:
            continue

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        conv_w_name = conv_node.input[1]
        if conv_w_name not in weights:
            continue

        conv_w = weights[conv_w_name]
        conv_b = weights.get(conv_node.input[2] if len(conv_node.input) > 2 else None)

        inv_std = gamma / np.sqrt(var + eps)
        n_oc = conv_w.shape[0]
        shape = [n_oc] + [1] * (conv_w.ndim - 1)
        weights[conv_w_name] = conv_w * inv_std.reshape(shape)

        if conv_b is not None:
            new_bias = (conv_b - mean) * inv_std + beta
        else:
            new_bias = -mean * inv_std + beta

        if len(conv_node.input) > 2:
            weights[conv_node.input[2]] = new_bias
        else:
            bias_key = conv_w_name + "_bias"
            weights[bias_key] = new_bias

        for k in [node.input[1], node.input[2], node.input[3], node.input[4]]:
            weights.pop(k, None)
        bn_folded += 1

    if bn_folded:
        print(f"  BatchNorm folded into Conv: {bn_folded}")

    # Fold Conv → Mul → Add patterns (decomposed BN from ONNX export)
    # Pattern: Conv(x, w) → Mul(result, scale) → Add(result, shift)
    mul_folded = 0
    for node in graph.node:
        if node.op_type != 'Mul':
            continue
        mul_input = node.input[0]
        scale_name = node.input[1]
        if mul_input not in output_to_node or scale_name not in weights:
            continue
        conv_node = output_to_node[mul_input]
        if conv_node.op_type != 'Conv':
            continue

        # Find the Add after this Mul
        mul_out = node.output[0]
        add_node = None
        for n2 in graph.node:
            if n2.op_type == 'Add' and mul_out in n2.input:
                add_node = n2
                break
        if not add_node:
            continue

        # Get scale and shift
        scale = weights[scale_name].squeeze()  # [OC]
        shift_name = [n for n in add_node.input if n != mul_out and n in weights]
        if not shift_name:
            continue
        shift = weights[shift_name[0]].squeeze()  # [OC]

        # Fold into conv weight: w_new = w * scale
        conv_w_name = conv_node.input[1]
        if conv_w_name not in weights:
            continue
        conv_w = weights[conv_w_name]
        n_oc = conv_w.shape[0]
        shape = [n_oc] + [1] * (conv_w.ndim - 1)
        weights[conv_w_name] = conv_w * scale.reshape(shape)

        # Create bias: shift (= beta - mean * gamma/sqrt(var))
        bias_key = conv_w_name.replace('.weight', '.bias')
        if conv_w_name.endswith('.weight'):
            weights[bias_key] = shift.astype(np.float32)
        else:
            weights[conv_w_name + '_bias'] = shift.astype(np.float32)

        # Remove scale and shift from weights
        weights.pop(scale_name, None)
        weights.pop(shift_name[0], None)
        mul_folded += 1

    if mul_folded:
        print(f"  Conv-Mul-Add (decomposed BN) folded: {mul_folded}")

    # Transpose weights that are stored in PyTorch (out, in) convention.
    # The C++ cpu_linear uses a single MatMul convention (in, out) for all weights.
    # Two patterns need transposing:
    # 1. Gemm(transB=1): weight stored as (out, in), transposed at runtime → transpose here
    # 2. Split+Transpose+MatMul: weight stored as (out, in), split then transposed → transpose here
    transposed = 0

    # Collect Gemm(transB=1) weight names (DECODER only — encoder uses ggml graph)
    gemm_transB_weights = set()
    for node in graph.node:
        if node.op_type == 'Gemm':
            attrs = {a.name: a for a in node.attribute}
            if attrs.get('transB') and attrs['transB'].i == 1:
                for inp in node.input:
                    if inp in weights and weights[inp].ndim == 2:
                        # Only transpose decoder weights (encoder uses ggml_mul_mat)
                        if 'decoder' in inp:
                            gemm_transB_weights.add(inp)

    # Collect Split+Transpose weight names (in_proj_weight patterns, decoder only)
    split_weights = set()
    for node in graph.node:
        if node.op_type == 'Split':
            for inp in node.input:
                if inp in weights and 'weight' in inp and weights[inp].ndim == 2:
                    if 'decoder' in inp:
                        split_weights.add(inp)

    # NOTE: decoder Conv2d(1x1) weights (decoder.input_proj) are NOT transposed.
    # They are used via cpu_linear which handles the (out, in) convention correctly
    # when the weight ne[0] matches in_d. The transposition only applies to
    # Gemm(transB=1) and Split+Transpose patterns.
    conv_decoder_weights = set()  # empty — no Conv2d transposition

    # Transpose decoder weights from (out, in) → (in, out)
    for name in gemm_transB_weights | split_weights | conv_decoder_weights:
        if name in weights and weights[name].ndim == 2:
            old_shape = weights[name].shape
            weights[name] = weights[name].T.copy()
            transposed += 1

    if transposed:
        print(f"  Transposed {transposed} weights to MatMul (in, out) convention")

    # Rename val_XXXX weights: trace each MatMul node to find the named bias
    # it pairs with, then rename the weight accordingly.
    renamed = 0
    for node in graph.node:
        if node.op_type not in ('MatMul', 'Gemm'):
            continue
        # Find which input is a val_XXXX initializer
        val_inputs = [inp for inp in node.input if inp.startswith('val_') and inp in weights]
        if not val_inputs:
            continue
        val_name = val_inputs[0]
        # Find the Add node that uses this MatMul's output + a named bias
        matmul_out = node.output[0]
        for add_node in graph.node:
            if add_node.op_type != 'Add':
                continue
            if matmul_out not in add_node.input:
                continue
            # The other input should be a named bias
            named_inputs = [inp for inp in add_node.input
                           if inp != matmul_out and inp in weights
                           and not inp.startswith('val_')]
            if named_inputs:
                bias_name = named_inputs[0]
                # Derive weight name from bias name: replace .bias with .weight
                weight_name = bias_name.replace('.bias', '.weight')
                if val_name in weights and weight_name != bias_name:
                    weights[weight_name] = weights.pop(val_name)
                    renamed += 1
            break

    if renamed:
        print(f"  Renamed {renamed} val_XXXX → named weights")

    # Build graph topology string for graph replayer
    node_descs = []
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            # Check if it was folded
            gamma_name = node.input[1]
            if gamma_name not in weights:
                # Was folded — skip
                continue
            # Not folded — store as precomputed scale+shift
            gamma = weights.get(gamma_name)
            beta = weights.get(node.input[2])
            mean = weights.get(node.input[3])
            var = weights.get(node.input[4])
            if gamma is not None and beta is not None and mean is not None and var is not None:
                eps = 1e-5
                for attr in node.attribute:
                    if attr.name == "epsilon": eps = attr.f
                inv_std = gamma / np.sqrt(var + eps)
                sc_name = f"bn_scale_{node.output[0].replace('/', '_')}"
                sh_name = f"bn_shift_{node.output[0].replace('/', '_')}"
                weights[sc_name] = inv_std.astype(np.float32)
                weights[sh_name] = (beta - mean * inv_std).astype(np.float32)
                for k in [gamma_name, node.input[2], node.input[3], node.input[4]]:
                    weights.pop(k, None)
                desc = f"BNPrecomputed:{node.input[0]};{sc_name};{sh_name}:{node.output[0]}"
                node_descs.append(desc)
                continue

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
        elif node.op_type == "Reshape":
            # Store target shape if it's a constant
            if len(node.input) > 1 and node.input[1] in weights:
                shape_arr = weights[node.input[1]].astype(np.int64)
                attrs = f"[{','.join(str(int(x)) for x in shape_arr)}]"
        elif node.op_type == "Transpose":
            perm = []
            for a in node.attribute:
                if a.name == "perm": perm = list(a.ints)
            if perm:
                attrs = f"[{','.join(str(p) for p in perm)}]"
        elif node.op_type in ("Split",):
            axis = 0; num_outputs = 0
            for a in node.attribute:
                if a.name == "axis": axis = a.i
                if a.name == "num_outputs": num_outputs = a.i
            attrs = f"[axis{axis}n{num_outputs}]"
        elif node.op_type == "Unsqueeze":
            axes = []
            for a in node.attribute:
                if a.name == "axes": axes = list(a.ints)
            if axes:
                attrs = f"[{','.join(str(a) for a in axes)}]"
        elif node.op_type == "GridSample":
            mode = "bilinear"; padding = "zeros"; align = 0
            for a in node.attribute:
                if a.name == "mode": mode = a.s.decode() if isinstance(a.s, bytes) else str(a.s)
                if a.name == "padding_mode": padding = a.s.decode() if isinstance(a.s, bytes) else str(a.s)
                if a.name == "align_corners": align = a.i
            attrs = f"[{mode},{padding},{align}]"
        elif node.op_type == "LayerNormalization":
            axis = -1; eps = 1e-5
            for a in node.attribute:
                if a.name == "axis": axis = a.i
                if a.name == "epsilon": eps = a.f
            attrs = f"[axis{axis}eps{eps:.0e}]"

        desc = f"{node.op_type}{attrs}:{';'.join(node.input)}:{';'.join(node.output)}"
        node_descs.append(desc)

    graph_str = "|".join(node_descs)
    print(f"  Graph topology: {len(node_descs)} nodes, {len(graph_str)} chars")

    # Write GGUF
    writer = gguf.GGUFWriter(str(args.output), arch="layout")

    writer.add_string("general.architecture", "layout")
    writer.add_string("general.name", args.model_name or "docling-layout-heron")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "docling-project/docling-layout-heron-onnx")

    # Model metadata
    writer.add_string("layout.model_type", "rt_detr_v2")
    writer.add_uint32("layout.input_height", 640)
    writer.add_uint32("layout.input_width", 640)
    writer.add_uint32("layout.num_classes", 17)
    writer.add_uint32("layout.num_queries", 300)

    # Preprocessing (ImageNet + rescale to [0,1])
    writer.add_array("layout.image_mean", [0.485, 0.456, 0.406])
    writer.add_array("layout.image_std", [0.229, 0.224, 0.225])

    # Class labels
    labels = ["caption", "footnote", "formula", "list_item", "page_footer",
              "page_header", "picture", "section_header", "table", "text",
              "title", "document_index", "code", "checkbox_selected",
              "checkbox_unselected", "form", "key_value_region"]
    writer.add_array("layout.labels", labels)

    # Graph topology
    writer.add_string("layout.graph_nodes", graph_str)

    # Write tensors
    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_written = 0
    for name, arr in weights.items():
        data = arr.astype(dtype_np)
        # Flatten conv weights to 2D for quantization
        if data.ndim == 4:
            data = data.reshape(data.shape[0], -1)
        total_written += data.size
        # Shorten names that exceed GGUF 64-char limit
        if len(name) >= 64:
            name = name.replace("attention_weights", "attn_wts")
            name = name.replace("sampling_offsets", "samp_offs")
            name = name.replace("model.decoder.decoder", "m.dec.dec")
            name = name.replace("model.backbone.res_layers", "m.bb.rl")
            name = name.replace("model.encoder", "m.enc")
        writer.add_tensor(name, data, raw_dtype=dtype_gguf)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nWritten: {args.output}")
    print(f"  Tensors: {len(weights)}")
    print(f"  Parameters: {total_written:,}")
    print(f"  File size: {output_size:.1f} MB")
    print(f"  Format: {'FP16' if args.fp16 else 'FP32'}")


if __name__ == "__main__":
    main()
