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
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Shared GGUF read/write core — the family-dispatch foundation, shared with
# merge-llamacpp-smolvlm-gguf.py. See gguf_merge_core.py.
import gguf_merge_core as core
from gguf_merge_core import (
    GGUF_MAGIC, GGUF_VERSION, ALIGNMENT,
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, GGUF_TYPE_UINT64, GGUF_TYPE_INT64,
    GGUF_TYPE_FLOAT64,
    GGML_TYPE_META, GGML_TYPE_NAME,
    TensorInfo, GGUFFile, read_gguf, read_tensor_data,
    tensor_nbytes, align_offset,
)

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
    """Keep the llama.cpp-native tensor name.

    The qwen2vl_ocr loader reads llama.cpp-native names directly (`v.blk.*`,
    `blk.*`, `token_embd`, `output_norm`, `mm.*`, `v.post_ln`, `v.patch_embd`).
    An earlier version of this script renamed everything to `vis.blocks.*` /
    `llm.layers.*`, which the current loader does NOT read — its output crashed
    on load (vision misdetected → SIGSEGV). Identity is correct; the only
    structural transform (concatenating the split temporal patch embedding) is
    handled in main().
    """
    if name == "v.patch_embd.weight.1":
        return None  # folded into v.patch_embd.weight
    return name
    # (legacy remap retained below for reference; unreachable)
    if name in GLOBAL_MAP:  # noqa
        return GLOBAL_MAP[name]
    m = _VIS_BLK_RE.match(name)
    if m:
        layer = m.group(1)
        suffix = m.group(2)
        if suffix in VIS_BLOCK_MAP:
            return f"vis.blocks.{layer}.{VIS_BLOCK_MAP[suffix]}"
        return None
    m = _LLM_BLK_RE.match(name)
    if m:
        layer = m.group(1)
        suffix = m.group(2)
        if suffix in LLM_BLOCK_MAP:
            return f"llm.layers.{layer}.{LLM_BLOCK_MAP[suffix]}"
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

    # ── Vision special tokens ──
    # Pass through from LLM metadata if present, else use the fixed Qwen2/2.5-VL
    # IDs. llama.cpp GGUFs do NOT carry these, and CrispEmbed's vision-text splice
    # needs qwen2vl.image_token_id to find the <|image_pad|> positions — without
    # it the splice looks for token 0, never fires, and the image is silently
    # dropped (model replies "text not visible").
    _tok_defaults = {
        "qwen2vl.image_token_id": 151655,        # <|image_pad|>
        "qwen2vl.vision_start_token_id": 151652,  # <|vision_start|>
        "qwen2vl.vision_end_token_id": 151653,    # <|vision_end|>
    }
    for key in ["qwen2vl.image_token_id", "qwen2vl.video_token_id",
                "qwen2vl.vision_start_token_id", "qwen2vl.vision_end_token_id"]:
        v = lm.get(key)
        if v is None:
            v = _tok_defaults.get(key)
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

    # Combine the split temporal patch embedding. llama.cpp stores the Qwen2-VL
    # Conv3d patch as two [out,in,H,W] slices (v.patch_embd.weight + .weight.1,
    # temporal_patch_size=2); the loader expects one weight flattening to
    # [out, in*T*H*W]. Stack in PyTorch Conv3d order [out,in,T,H,W] → flatten.
    import numpy as np

    p0 = next((x for x in out_tensors if x[0] == "v.patch_embd.weight"), None)
    p1_src = next((t for t in mmproj.tensors if t.name == "v.patch_embd.weight.1"), None)
    if p0 is not None and p1_src is not None:
        info0 = p0[1]
        # The concat only reorders whole elements, so a width-correct integer
        # view is byte-exact for ANY unquantized dtype (F16/F32/BF16) — the old
        # hardcoded float16 silently corrupted F32 patch embeddings.
        block_size, bytes_per_elem = GGML_TYPE_META.get(info0.dtype, (0, 0))
        _elem_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
        if block_size != 1 or bytes_per_elem not in _elem_dtype:
            sys.exit(f"error: cannot split quantized patch embed (dtype "
                     f"{GGML_TYPE_NAME.get(info0.dtype, info0.dtype)}); "
                     f"convert the mmproj to F16/F32 first")
        edt = _elem_dtype[bytes_per_elem]
        np_shape = list(reversed(info0.shape))  # ggml ne → numpy (out,in,H,W)
        a0 = np.frombuffer(read_tensor_data(mmproj, info0), dtype=edt).reshape(np_shape)
        a1 = np.frombuffer(read_tensor_data(mmproj, p1_src), dtype=edt).reshape(np_shape)
        comb = np.stack([a0, a1], axis=2).reshape(np_shape[0], -1)  # (out, in*T*H*W)
        comb_bytes = np.ascontiguousarray(comb).tobytes()
        comb_info = TensorInfo(
            name="v.patch_embd.weight",
            shape=[comb.shape[1], comb.shape[0]],  # ggml ne = [in*T*H*W, out]
            dtype=info0.dtype,
            offset=0,
            nbytes=len(comb_bytes),
        )
        out_tensors = [x for x in out_tensors if x[0] != "v.patch_embd.weight"]
        out_tensors.append(("v.patch_embd.weight", comb_info, ("__BYTES__", comb_bytes)))
        print(f"  patch embed: concatenated 2 temporal slices -> {comb_info.shape} ({len(comb_bytes)/1e6:.1f} MB)")

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

    fsize = core.write_combined_gguf(args.output, out_tensors, metadata)
    print(f"\nDone: {args.output}")
    print(f"  Size: {fsize / 1024 / 1024:.1f} MB "
          f"({fsize / 1024 / 1024 / 1024:.2f} GB)")
    print(f"  Tensors: {len(out_tensors)}")
    print(f"  Metadata: {len(metadata)} KV pairs")


if __name__ == "__main__":
    main()
