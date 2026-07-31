#!/usr/bin/env python3
"""Convert Paddle PIR PP-LCNet text-line orientation weights to GGUF.

The current Paddle export is a JSON PIR program plus a combined
``inference.pdiparams`` stream.  Paddle writes parameters in lexical variable
name order; each record contains a DenseTensor version, empty LoD, a protobuf
descriptor, and raw tensor bytes.  The graph supplies the names and shapes,
so this converter needs no Paddle import and keeps the runtime dependency-free.

Large inputs/outputs belong on /Volumes/backups/ai/crispembed-gguf/.  The
default precision policy keeps BatchNorm, depthwise convolutions, and the
classifier head in F32; ordinary pointwise convolutions may be F16.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import gguf
import numpy as np


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def graph_parameters(program):
    params = {}
    for node in walk(program):
        if node.get("#") != "p":
            continue
        args = node.get("A", [])
        name = args[-1] if args and isinstance(args[-1], str) else None
        if not name:
            continue
        outputs = node.get("O", {})
        output = outputs if isinstance(outputs, dict) and "TT" in outputs else next(
            (x for x in outputs if isinstance(x, dict) and "TT" in x), {})
        d = output.get("TT", {}).get("D", [])
        shape = d[1] if len(d) > 1 and isinstance(d[1], list) else None
        if shape is not None:
            params[name] = tuple(int(x) for x in shape)
    return dict(sorted(params.items()))


def read_varint(data, pos):
    value = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
    raise ValueError("truncated protobuf varint")


def descriptor_shape(desc):
    """Read only TensorDesc dims; unknown protobuf fields are skipped."""
    pos = 0
    dims = []
    while pos < len(desc):
        tag, pos = read_varint(desc, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = read_varint(desc, pos)
            if field == 2:
                dims.append(value)
        elif wire == 2:
            size, pos = read_varint(desc, pos)
            end = pos + size
            if field == 2:
                while pos < end:
                    value, pos = read_varint(desc, pos)
                    dims.append(value)
            else:
                pos = end
        else:
            raise ValueError(f"unsupported TensorDesc wire type {wire}")
    return tuple(dims)


def read_records(path: Path, names):
    records = []
    with path.open("rb") as stream:
        for index, (name, expected_shape) in enumerate(names.items()):
            outer_version = struct.unpack("<I", stream.read(4))[0]
            if outer_version != 0:
                raise ValueError(f"{name}: unsupported DenseTensor version {outer_version}")
            lod_level = struct.unpack("<Q", stream.read(8))[0]
            for _ in range(lod_level):
                size = struct.unpack("<Q", stream.read(8))[0]
                stream.seek(size, 1)
            tensor_version = struct.unpack("<I", stream.read(4))[0]
            if tensor_version != 0:
                raise ValueError(f"{name}: unsupported tensor version {tensor_version}")
            desc_size = struct.unpack("<i", stream.read(4))[0]
            desc = stream.read(desc_size)
            shape = descriptor_shape(desc)
            if shape != expected_shape:
                raise ValueError(f"record {index} {name}: graph shape {expected_shape} != stream shape {shape}")
            count = int(np.prod(shape, dtype=np.int64))
            # This export is F32.  Keep the size check explicit so a future
            # FP16/ BF16 export cannot silently shift all subsequent records.
            raw = stream.read(count * 4)
            if len(raw) != count * 4:
                raise ValueError(f"{name}: truncated tensor payload")
            records.append((name, shape, np.frombuffer(raw, dtype="<f4").copy()))
        if stream.read(1):
            raise ValueError("parameter stream has trailing bytes")
    return records


def critical(name):
    return "batch_norm" in name or "depthwise" in name or name.startswith("linear_0.") or name.startswith("conv2d_31.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fp32", action="store_true", help="keep all tensors in F32")
    args = ap.parse_args()
    doc = json.loads((args.model_dir / "inference.json").read_text())
    params = graph_parameters(doc["program"])
    records = read_records(args.model_dir / "inference.pdiparams", params)

    writer = gguf.GGUFWriter(str(args.output), arch="pplcnet_orientation")
    writer.add_string("general.name", "PP-LCNet_x1_0_textline_ori")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source", "PaddlePaddle/PP-LCNet_x1_0_textline_ori")
    writer.add_string("pplcnet.kind", "textline_orientation")
    writer.add_uint32("pplcnet.input_height", 80)
    writer.add_uint32("pplcnet.input_width", 160)
    writer.add_uint32("pplcnet.num_classes", 2)
    writer.add_array("pplcnet.labels", ["0_degree", "180_degree"])
    writer.add_uint32("pplcnet.parameter_count", len(records))
    for name, shape, values in records:
        # Keep the original PIR variable name in the artifact.  The native
        # mapper can consequently be audited against inventory.json.
        out_name = "ori." + name
        keep_f32 = args.fp32 or critical(name) or len(shape) <= 1
        raw = values if keep_f32 else values.astype(np.float16)
        raw_dtype = gguf.GGMLQuantizationType.F32 if keep_f32 else gguf.GGMLQuantizationType.F16
        writer.add_tensor(out_name, raw.reshape(shape), raw_dtype=raw_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} ({len(records)} tensors, {args.output.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()
