#!/usr/bin/env python3
"""Shared GGUF read/write core for the llama.cpp -> CrispEmbed merge scripts.

Hand-rolled (no `gguf` dependency for the byte-copy path) so quantized tensor
data is copied byte-for-byte with no re-quantization — the `gguf` package's
writer makes raw quantized round-tripping fiddly. Both
`merge-llamacpp-qwen2vl-gguf.py` and `merge-llamacpp-smolvlm-gguf.py` build on
this, differing only in their per-architecture tensor-name + metadata maps
(the "family dispatch"). Extracted verbatim from the proven Qwen2-VL merge.
"""

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── GGUF constants ───────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
ALIGNMENT = 32

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35

# (block_size_in_elements, bytes_per_block)
GGML_TYPE_META = {
    GGML_TYPE_F32: (1, 4), GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q4_0: (32, 18), GGML_TYPE_Q4_1: (32, 20),
    GGML_TYPE_Q5_0: (32, 22), GGML_TYPE_Q5_1: (32, 24),
    GGML_TYPE_Q8_0: (32, 34), GGML_TYPE_Q8_1: (32, 36),
    GGML_TYPE_Q2_K: (256, 84), GGML_TYPE_Q3_K: (256, 110),
    GGML_TYPE_Q4_K: (256, 144), GGML_TYPE_Q5_K: (256, 176),
    GGML_TYPE_Q6_K: (256, 210), GGML_TYPE_Q8_K: (256, 292),
    GGML_TYPE_I8: (1, 1), GGML_TYPE_I16: (1, 2), GGML_TYPE_I32: (1, 4),
    GGML_TYPE_I64: (1, 8), GGML_TYPE_F64: (1, 8), GGML_TYPE_BF16: (1, 2),
    GGML_TYPE_IQ2_XXS: (256, 66), GGML_TYPE_IQ2_XS: (256, 74),
    GGML_TYPE_IQ3_XXS: (256, 98), GGML_TYPE_IQ1_S: (256, 50),
    GGML_TYPE_IQ4_NL: (32, 18), GGML_TYPE_IQ3_S: (256, 110),
    GGML_TYPE_IQ2_S: (256, 82), GGML_TYPE_IQ4_XS: (256, 136),
    GGML_TYPE_IQ1_M: (256, 56), GGML_TYPE_TQ1_0: (256, 54),
    GGML_TYPE_TQ2_0: (256, 82),
}

GGML_TYPE_NAME = {
    GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16", GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1", GGML_TYPE_Q5_0: "Q5_0", GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0", GGML_TYPE_Q8_1: "Q8_1", GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K", GGML_TYPE_Q4_K: "Q4_K", GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K", GGML_TYPE_Q8_K: "Q8_K", GGML_TYPE_BF16: "BF16",
    GGML_TYPE_I8: "I8", GGML_TYPE_I16: "I16", GGML_TYPE_I32: "I32",
    GGML_TYPE_I64: "I64", GGML_TYPE_F64: "F64",
}


def tensor_nbytes(shape: List[int], dtype: int) -> int:
    if dtype not in GGML_TYPE_META:
        raise ValueError(f"Unknown GGML type: {dtype}")
    block_size, bytes_per_block = GGML_TYPE_META[dtype]
    n_elements = 1
    for d in shape:
        n_elements *= d
    if block_size == 1:
        return n_elements * bytes_per_block
    ne0 = shape[0] if shape else 1
    n_blocks_row = (ne0 + block_size - 1) // block_size
    n_rows = n_elements // ne0 if ne0 > 0 else 1
    return n_blocks_row * n_rows * bytes_per_block


def align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


# ── Reader ──────────────────────────────────────────────────────────

@dataclass
class TensorInfo:
    name: str
    shape: List[int]      # ggml ne order
    dtype: int
    offset: int
    nbytes: int


@dataclass
class GGUFFile:
    version: int
    n_tensors: int
    n_kv: int
    metadata: Dict[str, Any]
    metadata_types: Dict[str, int]
    tensors: List[TensorInfo]
    tensor_data_offset: int
    path: str


def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_value(f, vtype: int):
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif vtype == GGUF_TYPE_ARRAY:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, arr_type) for _ in range(arr_len)]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf(path: str) -> GGUFFile:
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path} (magic={magic:#x})")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata, metadata_types = {}, {}
        for _ in range(n_kv):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            metadata[key] = _read_value(f, vtype)
            metadata_types[key] = vtype

        tensors = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append(TensorInfo(name, shape, dtype, offset,
                                      tensor_nbytes(shape, dtype)))
        tensor_data_offset = align_offset(f.tell())

    return GGUFFile(version, n_tensors, n_kv, metadata, metadata_types,
                    tensors, tensor_data_offset, path)


def read_tensor_data(gguf_file: GGUFFile, tensor: TensorInfo) -> bytes:
    with open(gguf_file.path, "rb") as f:
        f.seek(gguf_file.tensor_data_offset + tensor.offset)
        return f.read(tensor.nbytes)


# ── Writer ──────────────────────────────────────────────────────────

def _write_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _write_value(f, vtype: int, value):
    if vtype == GGUF_TYPE_UINT8:
        f.write(struct.pack("<B", value))
    elif vtype == GGUF_TYPE_INT8:
        f.write(struct.pack("<b", value))
    elif vtype == GGUF_TYPE_UINT16:
        f.write(struct.pack("<H", value))
    elif vtype == GGUF_TYPE_INT16:
        f.write(struct.pack("<h", value))
    elif vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", value))
    elif vtype == GGUF_TYPE_INT32:
        f.write(struct.pack("<i", value))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", value))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(struct.pack("<B", 1 if value else 0))
    elif vtype == GGUF_TYPE_STRING:
        _write_string(f, value)
    elif vtype == GGUF_TYPE_UINT64:
        f.write(struct.pack("<Q", value))
    elif vtype == GGUF_TYPE_INT64:
        f.write(struct.pack("<q", value))
    elif vtype == GGUF_TYPE_FLOAT64:
        f.write(struct.pack("<d", value))
    elif vtype == GGUF_TYPE_ARRAY:
        arr_type, arr_vals = value
        f.write(struct.pack("<I", arr_type))
        f.write(struct.pack("<Q", len(arr_vals)))
        for v in arr_vals:
            _write_value(f, arr_type, v)
    else:
        raise ValueError(f"Cannot write GGUF value type: {vtype}")


def _write_kv(f, key: str, vtype: int, value):
    _write_string(f, key)
    f.write(struct.pack("<I", vtype))
    _write_value(f, vtype, value)


# out_tensors item: (name, TensorInfo, source) where source is either a
# GGUFFile (read raw bytes from) or the sentinel tuple ("__BYTES__", bytes).
OutTensor = Tuple[str, TensorInfo, Any]


def write_combined_gguf(out_path: str,
                        out_tensors: List[OutTensor],
                        metadata: List[Tuple[str, int, Any]],
                        progress_every: int = 50):
    """Write a combined GGUF v3. Metadata is a list of (key, gguf_type, value).
    Tensor data is copied byte-for-byte from each source (no re-quantization)."""
    import os
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(out_tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        for key, vtype, value in metadata:
            _write_kv(f, key, vtype, value)

        # Offsets (aligned) for the tensor-info table.
        tensor_offsets, offset = [], 0
        for _, info, _ in out_tensors:
            tensor_offsets.append(offset)
            offset = align_offset(offset + info.nbytes)

        for i, (name, info, _) in enumerate(out_tensors):
            _write_string(f, name)
            f.write(struct.pack("<I", len(info.shape)))
            for d in info.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", info.dtype))
            f.write(struct.pack("<Q", tensor_offsets[i]))

        pos = f.tell()
        aligned = align_offset(pos)
        if aligned > pos:
            f.write(b"\x00" * (aligned - pos))

        for i, (name, info, source) in enumerate(out_tensors):
            if isinstance(source, tuple) and source and source[0] == "__BYTES__":
                data = source[1]
            else:
                data = read_tensor_data(source, info)
            if len(data) != info.nbytes:
                raise ValueError(f"{name}: data {len(data)}B != declared {info.nbytes}B")
            f.write(data)
            pad = align_offset(len(data)) - len(data)
            if pad > 0:
                f.write(b"\x00" * pad)
            if (i + 1) % progress_every == 0 or i == len(out_tensors) - 1:
                dn = GGML_TYPE_NAME.get(info.dtype, f"type{info.dtype}")
                print(f"  [{i+1}/{len(out_tensors)}] {name} {list(info.shape)} "
                      f"{dn} ({info.nbytes/1024/1024:.1f} MB)")

    return os.path.getsize(out_path)


# ── llama.cpp import helpers ────────────────────────────────────────

def llama_unpermute_qk_rows(data: bytes, out_rows: int, n_head: int, head_dim: int) -> bytes:
    """Undo llama.cpp's q/k permute (convert_hf_to_gguf LlamaModel.permute),
    converting its interleaved-RoPE weight layout back to HF rotate_half layout
    — which is what CrispEmbed's converters emit and their RoPE expects.

    Applies to any llama.cpp arch=llama/qwen2 LLM imported into a CrispEmbed
    HF-layout loader; without it the decoder emits fluent-but-repetitive garbage
    (never a crash — shapes are identical, only row order differs). The permute
    reorders OUTPUT rows only, so this is a byte-exact row-shuffle on any dtype
    (Q8_0 rows are independently quantized). Works for weights (row = one matmul
    row) AND biases (row = one element). q uses n_head; k uses n_head_kv."""
    if n_head * head_dim != out_rows:
        raise ValueError(f"n_head*head_dim {n_head*head_dim} != out_rows {out_rows}")
    row = len(data) // out_rows
    hd2 = head_dim // 2
    src = memoryview(data)
    out = bytearray(len(data))
    for h in range(n_head):
        for s in range(2):
            for d in range(hd2):
                hf = h * head_dim + s * hd2 + d
                lla = h * head_dim + d * 2 + s
                out[hf * row:(hf + 1) * row] = src[lla * row:(lla + 1) * row]
    return bytes(out)
