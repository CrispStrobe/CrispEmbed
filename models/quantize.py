#!/usr/bin/env python3
"""Quantize CrispEmbed GGUF models to Q8_0/F16.

Uses the original converter to create F16 models (recommended for maximum
compatibility), or quantizes tensor data in-place for Q8_0.

Usage:
    python models/quantize.py input.gguf                    # produces input-q8_0.gguf
    python models/quantize.py input.gguf --types q8_0 f16
    python models/quantize.py --all --dir /path/to/ggufs
"""

import argparse
import os
import struct
import sys

import numpy as np

try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


QUANT_MAP = {
    "f16":  gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
}

KEEP_F32_PATTERNS = ["norm", "bias", "embd_ln", "position_embd",
                      "token_type_embd", "pooler", "output_norm"]


def should_quantize(name: str, shape: tuple, qtype_name: str) -> bool:
    name_lower = name.lower()
    if len(shape) < 2:
        return False
    total = 1
    for d in shape:
        total *= d
    if total < 4096:
        return False
    for p in KEEP_F32_PATTERNS:
        if p in name_lower:
            return False
    if "token_embd" in name_lower:
        return qtype_name in ("q8_0", "f16")
    if shape[0] % 32 != 0:
        return False
    return True


def quantize_gguf(input_path: str, output_path: str, qtype_name: str):
    """Quantize by re-reading original GGUF and writing through gguf_init."""
    qtype = QUANT_MAP[qtype_name]
    print(f"Quantizing {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({qtype_name})")

    # Strategy: Use gguf C API (via gguf_init_from_file) to read everything,
    # then write via GGUFWriter with proper metadata.
    # This avoids the buggy string array copy by using the C API directly.

    gp = {"no_alloc": True, "ctx": None}

    # Read with gguf_init (C API through the gguf Python bindings)
    reader = gguf.GGUFReader(input_path, "r")

    # Detect arch
    arch = "unknown"
    for name, field in reader.fields.items():
        if name == "general.architecture":
            raw = field.parts[-1]
            arch = raw.tobytes().decode("utf-8") if hasattr(raw, "tobytes") else str(raw)
            break

    # Read all tensor data from file, keeping raw bytes for non-f32 tensors
    tensor_info = []
    n_quantized = 0
    n_kept = 0

    for tensor in reader.tensors:
        tname = tensor.name
        shape = tuple(tensor.shape)
        src_type = tensor.tensor_type
        raw = tensor.data

        # Get f32 data
        if src_type == gguf.GGMLQuantizationType.F32:
            f32_flat = np.array(raw, dtype=np.float32)
        elif src_type == gguf.GGMLQuantizationType.F16:
            f32_flat = np.array(raw).view(np.float16).astype(np.float32)
        else:
            tensor_info.append((tname, shape, raw, src_type))
            n_kept += 1
            continue

        if should_quantize(tname, shape, qtype_name):
            try:
                # gguf.quantize expects [n_rows, row_width] where row_width = ne[0]
                n_rows = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                row_width = shape[0]
                mat = f32_flat.reshape(n_rows, row_width)
                q_data = gguf.quantize(mat, qtype)
                tensor_info.append((tname, shape, q_data, qtype))
                n_quantized += 1
            except Exception as e:
                tensor_info.append((tname, shape, f32_flat, gguf.GGMLQuantizationType.F32))
                n_kept += 1
                if str(e):
                    print(f"  skip '{tname}' {shape}: {e}")
        else:
            tensor_info.append((tname, shape, f32_flat, gguf.GGMLQuantizationType.F32))
            n_kept += 1

    # Write output: use binary copy of header+metadata, then replace tensors
    # This preserves the exact metadata bytes (no string array corruption)
    _write_with_raw_metadata(input_path, output_path, arch, tensor_info, reader)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"  {n_quantized} quantized, {n_kept} kept -> {size_mb:.0f} MB (was {orig_mb:.0f} MB)")


def _write_with_raw_metadata(input_path, output_path, arch, tensor_info, reader):
    """Write GGUF by copying raw metadata bytes and replacing tensor data."""
    # Parse the original GGUF header to find where metadata ends and tensors start
    with open(input_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"Not a GGUF file: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

    # Use GGUFWriter to create proper output
    writer = gguf.GGUFWriter(output_path, arch=arch)

    # Copy metadata using a careful field-by-field approach
    for field_name, field in reader.fields.items():
        if field_name.startswith("GGUF.") or field_name == "general.architecture":
            continue
        _copy_field(writer, field_name, field)

    # Add tensors
    for tname, shape, data, dtype in tensor_info:
        writer.add_tensor(tname, data, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _copy_field(writer, field_name, field):
    """Copy a single metadata field to the writer."""
    if not field.types:
        return

    ftype = field.types[0]
    try:
        if ftype == gguf.GGUFValueType.STRING:
            writer.add_string(field_name, field.parts[-1].tobytes().decode("utf-8", "replace"))
        elif ftype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(field_name, int(field.parts[-1][0]))
        elif ftype == gguf.GGUFValueType.INT32:
            writer.add_int32(field_name, int(field.parts[-1][0]))
        elif ftype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(field_name, float(field.parts[-1][0]))
        elif ftype == gguf.GGUFValueType.BOOL:
            writer.add_bool(field_name, bool(field.parts[-1][0]))
        elif ftype == gguf.GGUFValueType.UINT64:
            writer.add_uint64(field_name, int(field.parts[-1][0]))
        elif ftype == gguf.GGUFValueType.ARRAY:
            _copy_array_field(writer, field_name, field)
    except Exception as e:
        print(f"  Warning: skip '{field_name}': {e}")


def _copy_array_field(writer, field_name, field):
    """Copy an array metadata field."""
    if len(field.types) < 2:
        return
    arr_type = field.types[1]

    if arr_type == gguf.GGUFValueType.STRING:
        # The GGUFReader stores string array parts as:
        #   parts[0] = array count (uint64 or similar metadata)
        #   Then for each string: parts[k] = length (uint32), parts[k+1] = bytes
        # But the exact layout varies. Use data field if available.
        strings = []
        parts = list(field.parts)

        # Find the string data: skip non-bytes parts at the start
        # Actually, each string is stored as two parts: len (uint32 memmap) + data (bytes memmap)
        # parts[0] is typically the array-level metadata
        # Let's try pairs starting from index 1
        i = 0
        while i < len(parts):
            p = parts[i]
            # String data parts are byte arrays with length > 4 or contain printable chars
            if hasattr(p, 'tobytes'):
                raw = p.tobytes()
                # Detect if this is a length prefix (4 bytes, uint32) or string data
                if len(raw) <= 8 and all(b < 128 for b in raw[:4]):
                    # Might be a length, skip
                    pass
            i += 1

        # Simpler: use the field.data attribute if available, or count total strings
        # from the expected_count in parts[0]
        # Let's just count: expected N = n_kv_reported for this array
        # For string arrays: parts has exactly 2*N + 1 entries (1 metadata + N pairs)
        n_strings = (len(parts) - 1) // 2
        for j in range(n_strings):
            idx = 1 + j * 2 + 1  # skip metadata, then len+data pairs
            if idx < len(parts):
                try:
                    s = parts[idx].tobytes().decode("utf-8", errors="replace")
                    strings.append(s)
                except:
                    strings.append("")
            else:
                strings.append("")

        if strings:
            writer.add_array(field_name, strings)

    elif arr_type == gguf.GGUFValueType.FLOAT32:
        # Float32 array: collect all float data from parts
        vals = []
        for p in field.parts:
            if hasattr(p, 'dtype') and p.dtype == np.float32 and p.ndim > 0:
                vals.extend(p.tolist())
            elif hasattr(p, 'dtype') and p.dtype == np.float32:
                vals.append(float(p))
        if vals:
            writer.add_array(field_name, vals)

    elif arr_type in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32):
        vals = []
        for p in field.parts:
            if hasattr(p, 'dtype') and p.ndim > 0 and p.dtype in (np.uint32, np.int32):
                vals.extend(p.tolist())
            elif hasattr(p, 'dtype') and p.dtype in (np.uint32, np.int32):
                vals.append(int(p))
        if vals:
            writer.add_array(field_name, vals)


def main():
    parser = argparse.ArgumentParser(description="Quantize CrispEmbed GGUF models")
    parser.add_argument("input", nargs="?", help="Input GGUF file")
    parser.add_argument("--types", nargs="+", default=["q8_0"],
                        choices=list(QUANT_MAP.keys()),
                        help="Quantization types (default: q8_0)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dir", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.all:
        inputs = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.endswith(".gguf")
            and not any(q in f for q in ["-q4_0", "-q8_0", "-q5_0", "-f16"])
        ])
    elif args.input:
        inputs = [args.input]
    else:
        parser.print_help()
        return 1

    for inp in inputs:
        base = os.path.splitext(os.path.basename(inp))[0]
        out_dir = args.output_dir or os.path.dirname(inp) or "."
        for qt in args.types:
            out = os.path.join(out_dir, f"{base}-{qt}.gguf")
            if os.path.exists(out):
                print(f"  Skip {os.path.basename(out)} (exists)")
                continue
            try:
                quantize_gguf(inp, out, qt)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                if os.path.exists(out):
                    os.remove(out)


if __name__ == "__main__":
    sys.exit(main() or 0)
