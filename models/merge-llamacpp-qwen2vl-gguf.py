#!/usr/bin/env python3
"""Merge llama.cpp split Qwen2-VL GGUFs (LLM + mmproj) into a single CrispEmbed GGUF.

llama.cpp exports Qwen2-VL as two separate files:
  - LLM GGUF: blk.N.* tensors + token_embd + output_norm + output
  - mmproj GGUF: v.blk.N.* tensors + v.patch_embd + v.post_ln + mm.*

This script reads both, renames tensors to CrispEmbed convention, merges
metadata, and writes a single combined GGUF v3 file.

Tensor data is copied byte-for-byte (no re-quantization).

Usage:
    python merge-llamacpp-qwen2vl-gguf.py \
        --llm german-ocr-3.1-F16.gguf \
        --mmproj mmproj-german-ocr-3.1-F16.gguf \
        --output german-ocr-3.1-crispembed.gguf
"""

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── GGUF constants ───────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT8    = 0
GGUF_TYPE_INT8     = 1
GGUF_TYPE_UINT16   = 2
GGUF_TYPE_INT16    = 3
GGUF_TYPE_UINT32   = 4
GGUF_TYPE_INT32    = 5
GGUF_TYPE_FLOAT32  = 6
GGUF_TYPE_BOOL     = 7
GGUF_TYPE_STRING   = 8
GGUF_TYPE_ARRAY    = 9
GGUF_TYPE_UINT64   = 10
GGUF_TYPE_INT64    = 11
GGUF_TYPE_FLOAT64  = 12

# GGML tensor types (subset that matters for reading)
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q4_1  = 3
GGML_TYPE_Q5_0  = 6
GGML_TYPE_Q5_1  = 7
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q8_1  = 9
GGML_TYPE_Q2_K  = 10
GGML_TYPE_Q3_K  = 11
GGML_TYPE_Q4_K  = 12
GGML_TYPE_Q5_K  = 13
GGML_TYPE_Q6_K  = 14
GGML_TYPE_Q8_K  = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS  = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S   = 19
GGML_TYPE_IQ4_NL  = 20
GGML_TYPE_IQ3_S   = 21
GGML_TYPE_IQ2_S   = 22
GGML_TYPE_IQ4_XS  = 23
GGML_TYPE_I8    = 24
GGML_TYPE_I16   = 25
GGML_TYPE_I32   = 26
GGML_TYPE_I64   = 27
GGML_TYPE_F64   = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16  = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35

# Block sizes and type sizes for computing tensor byte counts.
# For quantized types: (block_size_in_elements, bytes_per_block)
GGML_TYPE_META = {
    GGML_TYPE_F32:   (1, 4),
    GGML_TYPE_F16:   (1, 2),
    GGML_TYPE_Q4_0:  (32, 18),    # 32 elements -> 16 bytes data + 2 bytes scale
    GGML_TYPE_Q4_1:  (32, 20),    # 32 elements -> 16 bytes data + 2 scale + 2 min
    GGML_TYPE_Q5_0:  (32, 22),    # 32 elements -> 4 bytes high bits + 16 bytes low + 2 scale
    GGML_TYPE_Q5_1:  (32, 24),
    GGML_TYPE_Q8_0:  (32, 34),    # 32 elements -> 32 bytes data + 2 bytes scale
    GGML_TYPE_Q8_1:  (32, 36),
    GGML_TYPE_Q2_K:  (256, 84),
    GGML_TYPE_Q3_K:  (256, 110),
    GGML_TYPE_Q4_K:  (256, 144),
    GGML_TYPE_Q5_K:  (256, 176),
    GGML_TYPE_Q6_K:  (256, 210),
    GGML_TYPE_Q8_K:  (256, 292),
    GGML_TYPE_I8:    (1, 1),
    GGML_TYPE_I16:   (1, 2),
    GGML_TYPE_I32:   (1, 4),
    GGML_TYPE_I64:   (1, 8),
    GGML_TYPE_F64:   (1, 8),
    GGML_TYPE_BF16:  (1, 2),
    GGML_TYPE_IQ2_XXS: (256, 66),
    GGML_TYPE_IQ2_XS:  (256, 74),
    GGML_TYPE_IQ3_XXS: (256, 98),
    GGML_TYPE_IQ1_S:   (256, 50),
    GGML_TYPE_IQ4_NL:  (32, 18),
    GGML_TYPE_IQ3_S:   (256, 110),
    GGML_TYPE_IQ2_S:   (256, 82),
    GGML_TYPE_IQ4_XS:  (256, 136),
    GGML_TYPE_IQ1_M:   (256, 56),
    GGML_TYPE_TQ1_0:   (256, 54),
    GGML_TYPE_TQ2_0:   (256, 82),
}

GGML_TYPE_NAME = {
    GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0", GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0", GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0", GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K", GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K", GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K", GGML_TYPE_Q8_K: "Q8_K",
    GGML_TYPE_BF16: "BF16", GGML_TYPE_I8: "I8",
    GGML_TYPE_I16: "I16", GGML_TYPE_I32: "I32",
    GGML_TYPE_I64: "I64", GGML_TYPE_F64: "F64",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS", GGML_TYPE_IQ2_XS: "IQ2_XS",
    GGML_TYPE_IQ3_XXS: "IQ3_XXS", GGML_TYPE_IQ1_S: "IQ1_S",
    GGML_TYPE_IQ4_NL: "IQ4_NL", GGML_TYPE_IQ3_S: "IQ3_S",
    GGML_TYPE_IQ2_S: "IQ2_S", GGML_TYPE_IQ4_XS: "IQ4_XS",
    GGML_TYPE_IQ1_M: "IQ1_M",
    GGML_TYPE_TQ1_0: "TQ1_0", GGML_TYPE_TQ2_0: "TQ2_0",
}


def tensor_nbytes(shape: List[int], dtype: int) -> int:
    """Compute the byte size of a tensor given its shape and GGML type."""
    if dtype not in GGML_TYPE_META:
        raise ValueError(f"Unknown GGML type: {dtype}")
    block_size, bytes_per_block = GGML_TYPE_META[dtype]
    n_elements = 1
    for d in shape:
        n_elements *= d
    if block_size == 1:
        return n_elements * bytes_per_block
    # Quantized: blocks along the innermost (first) dimension
    # n_blocks = ceil(ne[0] / block_size) * product(ne[1:])
    ne0 = shape[0] if shape else 1
    n_blocks_row = (ne0 + block_size - 1) // block_size
    n_rows = n_elements // ne0 if ne0 > 0 else 1
    return n_blocks_row * n_rows * bytes_per_block


def align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


# ── GGUF Reader ──────────────────────────────────────────────────────

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: int
    offset: int        # offset within the tensor data section
    nbytes: int        # byte count of tensor data

@dataclass
class GGUFFile:
    version: int
    n_tensors: int
    n_kv: int
    metadata: Dict[str, Any]
    metadata_types: Dict[str, int]  # original GGUF type ids
    tensors: List[TensorInfo]
    tensor_data_offset: int  # file offset where tensor data starts
    path: str


def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_value(f, vtype: int):
    """Read a single GGUF metadata value of the given type."""
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
    """Read GGUF header, metadata, and tensor info (but not tensor data)."""
    with open(path, "rb") as f:
        # Header
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path} (magic={magic:#x})")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Metadata KV pairs
        metadata = {}
        metadata_types = {}
        for _ in range(n_kv):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = _read_value(f, vtype)
            metadata[key] = value
            metadata_types[key] = vtype

        # Tensor info entries
        tensors = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack("<Q", f.read(8))[0])
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            nbytes = tensor_nbytes(shape, dtype)
            tensors.append(TensorInfo(name, shape, dtype, offset, nbytes))

        # Tensor data starts after alignment
        tensor_data_offset = align_offset(f.tell())

    return GGUFFile(
        version=version,
        n_tensors=n_tensors,
        n_kv=n_kv,
        metadata=metadata,
        metadata_types=metadata_types,
        tensors=tensors,
        tensor_data_offset=tensor_data_offset,
        path=path,
    )


def read_tensor_data(gguf_file: GGUFFile, tensor: TensorInfo) -> bytes:
    """Read raw tensor bytes from a GGUF file."""
    with open(gguf_file.path, "rb") as f:
        f.seek(gguf_file.tensor_data_offset + tensor.offset)
        return f.read(tensor.nbytes)


# ── GGUF Writer ──────────────────────────────────────────────────────

def _write_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _write_kv(f, key: str, vtype: int, value):
    """Write a single KV pair."""
    _write_string(f, key)
    f.write(struct.pack("<I", vtype))
    _write_value(f, vtype, value)


def _write_value(f, vtype: int, value):
    """Write a single metadata value."""
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
        # value must be (element_type, list_of_values)
        arr_type, arr_vals = value
        f.write(struct.pack("<I", arr_type))
        f.write(struct.pack("<Q", len(arr_vals)))
        for v in arr_vals:
            _write_value(f, arr_type, v)
    else:
        raise ValueError(f"Cannot write GGUF value type: {vtype}")


# ── Tensor name mapping ─────────────────────────────────────────────

# LLM block pattern: blk.N.suffix -> llm.layers.N.new_suffix
LLM_BLOCK_MAP = {
    "attn_q.weight":       "attn.q.weight",
    "attn_k.weight":       "attn.k.weight",
    "attn_v.weight":       "attn.v.weight",
    "attn_output.weight":  "attn.o.weight",
    "attn_q.bias":         "attn.q.bias",
    "attn_k.bias":         "attn.k.bias",
    "attn_v.bias":         "attn.v.bias",
    "attn_norm.weight":    "attn_norm.weight",
    "ffn_gate.weight":     "ffn_gate.weight",
    "ffn_up.weight":       "ffn_up.weight",
    "ffn_down.weight":     "ffn_down.weight",
    "ffn_norm.weight":     "ffn_norm.weight",
}

# Vision block pattern: v.blk.N.suffix -> vis.blocks.N.new_suffix
VIS_BLOCK_MAP = {
    "attn_q.weight":    "attn.q.weight",
    "attn_k.weight":    "attn.k.weight",
    "attn_v.weight":    "attn.v.weight",
    "attn_out.weight":  "attn.proj.weight",
    "attn_q.bias":      "attn.q.bias",
    "attn_k.bias":      "attn.k.bias",
    "attn_v.bias":      "attn.v.bias",
    "attn_out.bias":    "attn.proj.bias",
    "ln1.weight":       "norm1.weight",
    "ln1.bias":         "norm1.bias",
    "ln2.weight":       "norm2.weight",
    "ln2.bias":         "norm2.bias",
    "ffn_up.weight":    "mlp.fc1.weight",
    "ffn_up.bias":      "mlp.fc1.bias",
    "ffn_down.weight":  "mlp.fc2.weight",
    "ffn_down.bias":    "mlp.fc2.bias",
}

# Non-block tensor mappings
GLOBAL_MAP = {
    # LLM global
    "token_embd.weight":  "llm.embed_tokens.weight",
    "output_norm.weight":  "llm.norm.weight",
    "output.weight":       "llm.lm_head.weight",
    # Vision global
    "v.patch_embd.weight":     "vis.patch_embed.proj.weight",
    "v.patch_embd.weight.1":   "vis.patch_embed.proj_t.weight",
    "v.post_ln.weight":        "vis.merger.ln_q.weight",
    "v.post_ln.bias":          "vis.merger.ln_q.bias",
    # Projector
    "mm.0.weight":   "proj.mlp1.weight",
    "mm.0.bias":     "proj.mlp1.bias",
    "mm.2.weight":   "proj.mlp2.weight",
    "mm.2.bias":     "proj.mlp2.bias",
}

_LLM_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_VIS_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(.+)$")


def map_tensor_name(name: str) -> Optional[str]:
    """Map a llama.cpp tensor name to CrispEmbed convention.

    Returns the new name, or None if the tensor should be skipped.
    """
    # Check global map first
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]

    # Vision block
    m = _VIS_BLK_RE.match(name)
    if m:
        layer = m.group(1)
        suffix = m.group(2)
        if suffix in VIS_BLOCK_MAP:
            return f"vis.blocks.{layer}.{VIS_BLOCK_MAP[suffix]}"
        print(f"  WARNING: unmapped vision block tensor: {name}")
        return None

    # LLM block
    m = _LLM_BLK_RE.match(name)
    if m:
        layer = m.group(1)
        suffix = m.group(2)
        if suffix in LLM_BLOCK_MAP:
            return f"llm.layers.{layer}.{LLM_BLOCK_MAP[suffix]}"
        print(f"  WARNING: unmapped LLM block tensor: {name}")
        return None

    print(f"  WARNING: unmapped tensor: {name}")
    return None


# ── Metadata mapping ─────────────────────────────────────────────────

def build_output_metadata(
    llm_gguf: GGUFFile,
    mmproj_gguf: GGUFFile,
) -> List[Tuple[str, int, Any]]:
    """Build the output GGUF metadata from source GGUFs.

    Returns a list of (key, gguf_type, value) tuples.
    """
    lm = llm_gguf.metadata
    vm = mmproj_gguf.metadata

    kv: List[Tuple[str, int, Any]] = []

    def add_str(k, v):
        kv.append((k, GGUF_TYPE_STRING, v))

    def add_u32(k, v):
        kv.append((k, GGUF_TYPE_UINT32, int(v)))

    def add_f32(k, v):
        kv.append((k, GGUF_TYPE_FLOAT32, float(v)))

    def add_bool(k, v):
        kv.append((k, GGUF_TYPE_BOOL, bool(v)))

    def add_arr_u32(k, vals):
        kv.append((k, GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT32, [int(x) for x in vals])))

    def add_arr_i32(k, vals):
        kv.append((k, GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, [int(x) for x in vals])))

    def add_arr_f32(k, vals):
        kv.append((k, GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, [float(x) for x in vals])))

    def add_arr_str(k, vals):
        kv.append((k, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, list(vals))))

    # ── General ──
    add_str("general.architecture", "qwen2vl")
    if "general.name" in lm:
        add_str("general.name", lm["general.name"])

    # ── LLM config ──
    # Map from llama.cpp qwen2vl.* keys
    def llm_val(lcpp_key, default=None):
        """Get from LLM metadata, trying llama.cpp naming."""
        if lcpp_key in lm:
            return lm[lcpp_key]
        return default

    n_layers = llm_val("qwen2vl.block_count")
    if n_layers is not None:
        add_u32("qwen2vl.num_hidden_layers", n_layers)

    hidden_size = llm_val("qwen2vl.embedding_length")
    if hidden_size is not None:
        add_u32("qwen2vl.hidden_size", hidden_size)

    inter_size = llm_val("qwen2vl.feed_forward_length")
    if inter_size is not None:
        add_u32("qwen2vl.intermediate_size", inter_size)

    n_heads = llm_val("qwen2vl.attention.head_count")
    if n_heads is not None:
        add_u32("qwen2vl.num_attention_heads", n_heads)

    n_kv_heads = llm_val("qwen2vl.attention.head_count_kv")
    if n_kv_heads is not None:
        add_u32("qwen2vl.num_key_value_heads", n_kv_heads)

    rope_theta = llm_val("qwen2vl.rope.freq_base")
    if rope_theta is not None:
        add_f32("qwen2vl.rope_theta", rope_theta)

    # Additional LLM metadata (pass through common keys)
    for src_key, dst_key, converter in [
        ("qwen2vl.attention.layer_norm_rms_epsilon", "qwen2vl.rms_norm_eps", float),
        ("qwen2vl.context_length", "qwen2vl.max_position_embeddings", int),
    ]:
        v = llm_val(src_key)
        if v is not None:
            if converter == float:
                add_f32(dst_key, v)
            else:
                add_u32(dst_key, v)

    # Tie word embeddings — check if output.weight is absent
    has_output = any(t.name == "output.weight" for t in llm_gguf.tensors)
    add_bool("qwen2vl.tie_word_embeddings", not has_output)

    # mRoPE sections (pass through from llama.cpp if present)
    rope_sections = llm_val("qwen2vl.rope.dimension_sections")
    if rope_sections is not None:
        if isinstance(rope_sections, list):
            add_arr_u32("qwen2vl.rope_sections", rope_sections)

    # ── Vision config ──
    def vis_val(key, default=None):
        if key in vm:
            return vm[key]
        return default

    vis_layers = vis_val("clip.vision.block_count")
    if vis_layers is not None:
        add_u32("qwen2vl.vision.num_hidden_layers", vis_layers)
        # Also write as depth for compat with existing engine
        add_u32("qwen2vl.vision.depth", vis_layers)

    # Infer vision hidden size from tensor shapes (should be 1280 for Qwen2-VL)
    vis_hidden = None
    for t in mmproj_gguf.tensors:
        if t.name == "v.blk.0.ln1.weight":
            vis_hidden = t.shape[0]
            break
    if vis_hidden is None:
        vis_hidden = vis_val("clip.vision.embedding_length", 1280)
    add_u32("qwen2vl.vision.hidden_size", vis_hidden)

    vis_heads = vis_val("clip.vision.attention.head_count")
    if vis_heads is not None:
        add_u32("qwen2vl.vision.num_attention_heads", vis_heads)
        # Also write as num_heads for compat
        add_u32("qwen2vl.vision.num_heads", vis_heads)

    patch_size = vis_val("clip.vision.patch_size")
    if patch_size is not None:
        add_u32("qwen2vl.vision.spatial_patch_size", patch_size)
        add_u32("qwen2vl.vision.patch_size", patch_size)

    # spatial_merge_size is standard 2 for Qwen2-VL
    add_u32("qwen2vl.vision.spatial_merge_size", 2)

    # temporal_patch_size (standard 2 for Qwen2-VL)
    add_u32("qwen2vl.vision.temporal_patch_size", 2)

    # Vision intermediate size (from tensor shapes or clip metadata)
    vis_inter = vis_val("clip.vision.feed_forward_length")
    if vis_inter is None:
        # Infer from fc1 weight shape
        for t in mmproj_gguf.tensors:
            if t.name == "v.blk.0.ffn_up.weight":
                vis_inter = t.shape[1] if len(t.shape) > 1 else t.shape[0]
                break
    if vis_inter is not None:
        add_u32("qwen2vl.vision.intermediate_size", vis_inter)

    # in_channels (standard 3)
    add_u32("qwen2vl.vision.in_channels", 3)

    # Merger output size — infer from mm.2.weight output dim
    for t in mmproj_gguf.tensors:
        if t.name == "mm.2.weight":
            # mm.2 is the second linear in projector: shape = [out_hidden, in]
            # In GGUF, shape[1] is the output dimension (row-major: [out, in])
            out_hidden = t.shape[1] if len(t.shape) > 1 else t.shape[0]
            add_u32("qwen2vl.vision.out_hidden_size", out_hidden)
            break

    # ── Vocab size ──
    # From tokenizer data if present
    vocab_size = llm_val("qwen2vl.vocab_size")
    if vocab_size is None:
        # Try to infer from token_embd shape
        for t in llm_gguf.tensors:
            if t.name == "token_embd.weight":
                # shape = [hidden_size, vocab_size] in GGUF row-major
                vocab_size = t.shape[1] if len(t.shape) > 1 else t.shape[0]
                break
    if vocab_size is not None:
        add_u32("qwen2vl.vocab_size", vocab_size)

    # ── Pass through tokenizer metadata ──
    for key, val in lm.items():
        if key.startswith("tokenizer."):
            vtype = llm_gguf.metadata_types[key]
            if vtype == GGUF_TYPE_ARRAY:
                # Determine array element type and pass through
                if isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], str):
                        add_arr_str(key, val)
                    elif isinstance(val[0], float):
                        add_arr_f32(key, val)
                    elif isinstance(val[0], int):
                        add_arr_u32(key, val)
                    elif isinstance(val[0], bool):
                        # Bool arrays — store as uint8
                        kv.append((key, GGUF_TYPE_ARRAY,
                                   (GGUF_TYPE_UINT8, [1 if x else 0 for x in val])))
                    else:
                        print(f"  WARNING: skipping tokenizer array {key} "
                              f"(unknown element type)")
                # Empty array — skip
            else:
                kv.append((key, vtype, val))

    # ── Pass through vision special tokens from LLM metadata ──
    for key in ["qwen2vl.image_token_id", "qwen2vl.video_token_id",
                "qwen2vl.vision_start_token_id", "qwen2vl.vision_end_token_id"]:
        v = lm.get(key)
        if v is not None:
            add_u32(key, v)

    # ── Image preprocessor defaults ──
    # llama.cpp doesn't always carry these; use Qwen2-VL defaults
    if "qwen2vl.vision.image_mean" not in {k for k, _, _ in kv}:
        add_arr_f32("qwen2vl.vision.image_mean", [0.48145466, 0.4578275, 0.40821073])
    if "qwen2vl.vision.image_std" not in {k for k, _, _ in kv}:
        add_arr_f32("qwen2vl.vision.image_std", [0.26862954, 0.26130258, 0.27577711])

    return kv


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge llama.cpp split Qwen2-VL GGUFs into CrispEmbed format")
    parser.add_argument("--llm", required=True,
                        help="Path to llama.cpp LLM GGUF (e.g., german-ocr-3.1-F16.gguf)")
    parser.add_argument("--mmproj", required=True,
                        help="Path to llama.cpp mmproj GGUF (e.g., mmproj-german-ocr-3.1-F16.gguf)")
    parser.add_argument("--output", required=True,
                        help="Output CrispEmbed GGUF path")
    args = parser.parse_args()

    # ── Read source GGUFs ──
    print(f"Reading LLM:    {args.llm}")
    llm = read_gguf(args.llm)
    print(f"  Version {llm.version}, {llm.n_tensors} tensors, {llm.n_kv} KV pairs")

    print(f"Reading mmproj: {args.mmproj}")
    mmproj = read_gguf(args.mmproj)
    print(f"  Version {mmproj.version}, {mmproj.n_tensors} tensors, {mmproj.n_kv} KV pairs")

    # ── Build tensor list with mapped names ──
    out_tensors: List[Tuple[str, TensorInfo, GGUFFile]] = []  # (new_name, info, source)
    skipped = []

    # Process LLM tensors
    for t in llm.tensors:
        new_name = map_tensor_name(t.name)
        if new_name is not None:
            out_tensors.append((new_name, t, llm))
        else:
            skipped.append(t.name)

    # Process mmproj tensors
    for t in mmproj.tensors:
        new_name = map_tensor_name(t.name)
        if new_name is not None:
            out_tensors.append((new_name, t, mmproj))
        else:
            skipped.append(t.name)

    # Handle tied embeddings: if no lm_head, engine uses tie_word_embeddings flag
    has_lm_head = any(name == "llm.lm_head.weight" for name, _, _ in out_tensors)

    print(f"\nMapped {len(out_tensors)} tensors, skipped {len(skipped)}")
    if skipped:
        for s in skipped[:10]:
            print(f"  skipped: {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # ── Build metadata ──
    metadata = build_output_metadata(llm, mmproj)
    print(f"\nMetadata: {len(metadata)} KV pairs")

    # Print summary
    n_llm = sum(1 for n, _, _ in out_tensors if n.startswith("llm."))
    n_vis = sum(1 for n, _, _ in out_tensors if n.startswith("vis."))
    n_proj = sum(1 for n, _, _ in out_tensors if n.startswith("proj."))
    print(f"  LLM tensors:       {n_llm}")
    print(f"  Vision tensors:    {n_vis}")
    print(f"  Projector tensors: {n_proj}")

    # ── Write output GGUF ──
    print(f"\nWriting: {args.output}")

    with open(args.output, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(out_tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        # Metadata KV pairs
        for key, vtype, value in metadata:
            _write_kv(f, key, vtype, value)

        # Tensor info entries
        # First pass: compute offsets with alignment
        tensor_offsets = []
        offset = 0
        for new_name, info, source in out_tensors:
            tensor_offsets.append(offset)
            offset += info.nbytes
            offset = align_offset(offset)

        # Write tensor info
        for i, (new_name, info, source) in enumerate(out_tensors):
            _write_string(f, new_name)
            f.write(struct.pack("<I", len(info.shape)))
            for d in info.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", info.dtype))
            f.write(struct.pack("<Q", tensor_offsets[i]))

        # Align to start of tensor data
        pos = f.tell()
        aligned = align_offset(pos)
        if aligned > pos:
            f.write(b"\x00" * (aligned - pos))

        # Write tensor data
        total_bytes = 0
        for i, (new_name, info, source) in enumerate(out_tensors):
            data = read_tensor_data(source, info)
            f.write(data)
            total_bytes += len(data)

            # Pad to alignment
            pad = align_offset(len(data)) - len(data)
            if pad > 0:
                f.write(b"\x00" * pad)

            if (i + 1) % 50 == 0 or i == len(out_tensors) - 1:
                dtype_name = GGML_TYPE_NAME.get(info.dtype, f"type{info.dtype}")
                print(f"  [{i+1}/{len(out_tensors)}] {new_name} "
                      f"{list(info.shape)} {dtype_name} "
                      f"({info.nbytes / 1024 / 1024:.1f} MB)")

    fsize = os.path.getsize(args.output)
    print(f"\nDone: {args.output}")
    print(f"  Size: {fsize / 1024 / 1024:.1f} MB "
          f"({fsize / 1024 / 1024 / 1024:.2f} GB)")
    print(f"  Tensors: {len(out_tensors)}")
    print(f"  Metadata: {len(metadata)} KV pairs")


if __name__ == "__main__":
    main()
