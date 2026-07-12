#!/usr/bin/env python3
"""Merge a llama.cpp SmolVLM (Idefics3) split GGUF pair (LLM + mmproj) into a
single CrispEmbed "smoldocling"-arch GGUF.

llama.cpp exports SmolVLM as:
  - LLM GGUF   (arch=llama):  blk.N.* + token_embd + output_norm + output,
                              tokenizer.ggml.* (gpt2 BPE), llama.* config
  - mmproj GGUF (projector=idefics3): v.blk.N.* (SigLIP) + v.patch_embd +
                              v.position_embd + v.post_ln + mm.model.fc

CrispEmbed's `smoldocling_ocr` loader consumes ONE combined GGUF with a
different tensor-name convention (`vis.layers.N.*`, `llm.layers.N.*`,
`connector.proj.weight`) and `smoldocling.*` metadata. This script performs
that translation, copying tensor data byte-for-byte (no re-quantization).

Grounded in the real ggml-org/SmolVLM-256M-Instruct-GGUF files + the native
convert-smoldocling-to-gguf.py target format — not guessed. Verified points:
  - SigLIP FFN is name-INVERTED in llama.cpp's clip export (`ffn_down`=fc1,
    `ffn_up`=fc2); mapped by output dim, like the Qwen2-VL ViT.
  - The 4D Conv2d patch weight flattens to 2D by a pure shape relabel
    (C-order contiguous), so even F32 data copies raw.
  - The gpt2 BPE tokenizer passes through as `tokenizer.ggml.*` (the loader
    reads that as a fallback) — no tokenizer rewrite needed.

    python merge-llamacpp-smolvlm-gguf.py \
        --llm SmolVLM-256M-Instruct-Q8_0.gguf \
        --mmproj mmproj-SmolVLM-256M-Instruct-f16.gguf \
        --output smolvlm-256m-crispembed.gguf
"""
import argparse
import sys

import gguf_merge_core as core

# SmolVLM/Idefics3 <image> placeholder token (constant across the 256M/500M/2B
# family; absent from the llama.cpp GGUF metadata, so we inject it).
SMOLVLM_IMAGE_TOKEN_ID = 49190


def _first(md, *keys, default=None):
    for k in keys:
        if k in md:
            return md[k]
    return default


def map_vision_name(name, tinfo_by_name):
    """llama.cpp mmproj (SigLIP/idefics3) name -> CrispEmbed vis.* / connector.*.
    fc1/fc2 resolved by output dim (llama.cpp inverts ffn_up/ffn_down). Returns
    (new_name, shape_override_or_None); shape override flattens the 4D patch."""
    # Globals
    if name == "v.patch_embd.weight":
        t = tinfo_by_name[name]
        out = t.shape[-1]                      # ggml ne last dim = out channels
        in_flat = 1
        for d in t.shape[:-1]:
            in_flat *= d                       # in*kh*kw (C-order contiguous)
        return "vis.patch_embed.weight", [in_flat, out]
    if name == "v.patch_embd.bias":
        return "vis.patch_embed.bias", None
    if name == "v.position_embd.weight":
        return "vis.pos_embed.weight", None
    if name == "v.post_ln.weight":
        return "vis.post_ln.weight", None
    if name == "v.post_ln.bias":
        return "vis.post_ln.bias", None
    if name == "mm.model.fc.weight":
        return "connector.proj.weight", None
    # Blocks: v.blk.N.<suffix>
    if name.startswith("v.blk."):
        rest = name[len("v.blk."):]
        li, _, suffix = rest.partition(".")
        base = {
            "ln1.weight": "ln1.weight", "ln1.bias": "ln1.bias",
            "ln2.weight": "ln2.weight", "ln2.bias": "ln2.bias",
            "attn_q.weight": "attn.q.weight", "attn_q.bias": "attn.q.bias",
            "attn_k.weight": "attn.k.weight", "attn_k.bias": "attn.k.bias",
            "attn_v.weight": "attn.v.weight", "attn_v.bias": "attn.v.bias",
            "attn_out.weight": "attn.out.weight", "attn_out.bias": "attn.out.bias",
        }.get(suffix)
        if base is not None:
            return f"vis.layers.{li}.{base}", None
        # FFN — map by output dim, not name (llama.cpp inverts up/down).
        if suffix.startswith("ffn_up.") or suffix.startswith("ffn_down."):
            field = "weight" if suffix.endswith(".weight") else "bias"
            up = tinfo_by_name.get(f"v.blk.{li}.ffn_up.{field}")
            dn = tinfo_by_name.get(f"v.blk.{li}.ffn_down.{field}")
            # out dim: weight ne=[in,out]->shape[-1]; bias ne=[out]->shape[0]
            up_out = up.shape[-1] if field == "weight" else up.shape[0]
            dn_out = dn.shape[-1] if field == "weight" else dn.shape[0]
            this_is = "ffn_up" if suffix.startswith("ffn_up.") else "ffn_down"
            fc1_src = "ffn_up" if up_out >= dn_out else "ffn_down"  # fc1 = larger out
            fc = "fc1" if this_is == fc1_src else "fc2"
            return f"vis.layers.{li}.mlp.{fc}.{field}", None
    return None, None


def llama_unpermute_qk_rows(data: bytes, out_rows: int, n_head: int, head_dim: int) -> bytes:
    """Undo llama.cpp's q/k permute (convert_hf_to_gguf LlamaModel.permute),
    converting its interleaved-RoPE weight layout back to HF rotate_half layout
    — which is what CrispEmbed's converter emits and the loader's RoPE expects.

    The permute only reorders OUTPUT rows (the ne1 dim); each row's bytes are
    untouched, so this works byte-exact on any dtype incl. Q8_0 (rows are
    independently quantized). Forward maps HF (h,s,d)->llama (h,d,s); we invert:
    HF_row(h,s,d) data comes from llama_row(h,d,s)."""
    if out_rows % n_head != 0:
        raise ValueError(f"out_rows {out_rows} not divisible by n_head {n_head}")
    row = len(data) // out_rows
    hd2 = head_dim // 2
    if head_dim * n_head != out_rows:
        raise ValueError(f"n_head*head_dim {n_head*head_dim} != out_rows {out_rows}")
    src = memoryview(data)
    out = bytearray(len(data))
    for h in range(n_head):
        for s in range(2):
            for d in range(hd2):
                hf = h * head_dim + s * hd2 + d
                lla = h * head_dim + d * 2 + s
                out[hf * row:(hf + 1) * row] = src[lla * row:(lla + 1) * row]
    return bytes(out)


def map_llm_name(name):
    """llama.cpp LLM (arch=llama) name -> CrispEmbed llm.* name (FFN standard)."""
    if name == "token_embd.weight":
        return "llm.embed.weight"
    if name == "output_norm.weight":
        return "llm.norm.weight"
    if name == "output.weight":
        return "llm.lm_head.weight"
    if name.startswith("blk."):
        rest = name[len("blk."):]
        li, _, suffix = rest.partition(".")
        base = {
            "attn_norm.weight": "attn_norm.weight",
            "ffn_norm.weight": "ffn_norm.weight",
            "attn_q.weight": "attn.q.weight", "attn_k.weight": "attn.k.weight",
            "attn_v.weight": "attn.v.weight", "attn_output.weight": "attn.o.weight",
            "ffn_gate.weight": "ffn.gate.weight", "ffn_up.weight": "ffn.up.weight",
            "ffn_down.weight": "ffn.down.weight",
        }.get(suffix)
        if base is not None:
            return f"llm.layers.{li}.{base}"
    return None


def build_metadata(llm, mmproj):
    lm, vm = llm.metadata, mmproj.metadata
    kv = []

    def u32(k, v):
        kv.append((k, core.GGUF_TYPE_UINT32, int(v)))

    def f32(k, v):
        kv.append((k, core.GGUF_TYPE_FLOAT32, float(v)))

    def s(k, v):
        kv.append((k, core.GGUF_TYPE_STRING, v))

    s("general.architecture", "smoldocling")
    if "general.name" in lm:
        s("general.name", lm["general.name"])
    s("general.source", "merged from llama.cpp SmolVLM (LLM + idefics3 mmproj)")

    # Vision (from clip.*)
    u32("smoldocling.vision.hidden_size", _first(vm, "clip.vision.embedding_length", default=768))
    u32("smoldocling.vision.num_heads", _first(vm, "clip.vision.attention.head_count", default=12))
    u32("smoldocling.vision.num_layers", _first(vm, "clip.vision.block_count", default=12))
    u32("smoldocling.vision.patch_size", _first(vm, "clip.vision.patch_size", default=16))
    u32("smoldocling.vision.image_size", _first(vm, "clip.vision.image_size", default=512))
    u32("smoldocling.vision.intermediate_size",
        _first(vm, "clip.vision.feed_forward_length", default=3072))
    u32("smoldocling.connector.scale_factor",
        _first(vm, "clip.vision.projector.scale_factor", "clip.vision.projector.scale", default=4))

    # LLM (from llama.*)
    u32("smoldocling.hidden_size", _first(lm, "llama.embedding_length", default=576))
    u32("smoldocling.num_attention_heads", _first(lm, "llama.attention.head_count", default=9))
    u32("smoldocling.num_key_value_heads", _first(lm, "llama.attention.head_count_kv", default=3))
    u32("smoldocling.num_hidden_layers", _first(lm, "llama.block_count", default=30))
    u32("smoldocling.intermediate_size", _first(lm, "llama.feed_forward_length", default=1536))
    u32("smoldocling.head_dim", _first(lm, "llama.attention.key_length",
                                       "llama.rope.dimension_count", default=64))
    f32("smoldocling.rms_norm_eps", _first(lm, "llama.attention.layer_norm_rms_epsilon", default=1e-5))
    f32("smoldocling.rope_theta", _first(lm, "llama.rope.freq_base", default=100000.0))
    u32("smoldocling.vocab_size", _first(lm, "llama.vocab_size", default=49280))
    u32("smoldocling.image_token_id",
        _first(lm, "smoldocling.image_token_id", default=SMOLVLM_IMAGE_TOKEN_ID))

    # Tokenizer: pass through the gpt2 BPE data verbatim (the loader reads the
    # tokenizer.ggml.* fallback). Scalars keep their type; arrays lost their
    # element type on read, so re-infer it (str->STRING, int->INT32, ...).
    def _elem_type(vals):
        x = vals[0]
        if isinstance(x, bool):
            return core.GGUF_TYPE_BOOL          # before int (bool subclasses int)
        if isinstance(x, str):
            return core.GGUF_TYPE_STRING
        if isinstance(x, float):
            return core.GGUF_TYPE_FLOAT32
        if isinstance(x, int):
            return core.GGUF_TYPE_INT32
        raise ValueError(f"unhandled tokenizer array element type: {type(x)}")

    for key, val in lm.items():
        if not key.startswith("tokenizer."):
            continue
        vtype = llm.metadata_types[key]
        if vtype == core.GGUF_TYPE_ARRAY:
            if not val:
                continue  # skip empty arrays
            kv.append((key, core.GGUF_TYPE_ARRAY, (_elem_type(val), val)))
        else:
            kv.append((key, vtype, val))
    return kv


def main():
    ap = argparse.ArgumentParser(description="Merge llama.cpp SmolVLM -> CrispEmbed smoldocling GGUF")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    print(f"Reading LLM:    {a.llm}")
    llm = core.read_gguf(a.llm)
    print(f"  {llm.n_tensors} tensors, {llm.n_kv} KV")
    print(f"Reading mmproj: {a.mmproj}")
    mmproj = core.read_gguf(a.mmproj)
    print(f"  {mmproj.n_tensors} tensors, {mmproj.n_kv} KV")

    proj = mmproj.metadata.get("clip.projector_type")
    if proj != "idefics3":
        print(f"  WARNING: expected clip.projector_type=idefics3, got {proj!r}")

    vis_by_name = {t.name: t for t in mmproj.tensors}
    out_tensors, skipped = [], []

    # Vision + connector
    for t in mmproj.tensors:
        new_name, shape_override = map_vision_name(t.name, vis_by_name)
        if new_name is None:
            skipped.append(t.name)
            continue
        info = t
        if shape_override is not None:
            info = core.TensorInfo(t.name, shape_override, t.dtype, t.offset, t.nbytes)
        out_tensors.append((new_name, info, mmproj))

    # LLM. q/k projections must be un-permuted from llama.cpp's interleaved-RoPE
    # layout back to HF rotate_half layout (what the loader's RoPE expects).
    n_head = _first(llm.metadata, "llama.attention.head_count", default=9)
    n_head_kv = _first(llm.metadata, "llama.attention.head_count_kv", default=3)
    head_dim = _first(llm.metadata, "llama.attention.key_length",
                      "llama.rope.dimension_count", default=64)
    n_qk_fixed = 0
    for t in llm.tensors:
        new_name = map_llm_name(t.name)
        if new_name is None:
            skipped.append(t.name)
            continue
        is_q = new_name.endswith(".attn.q.weight")
        is_k = new_name.endswith(".attn.k.weight")
        if is_q or is_k:
            nh = n_head if is_q else n_head_kv
            data = core.read_tensor_data(llm, t)
            data = llama_unpermute_qk_rows(data, t.shape[-1], nh, head_dim)
            out_tensors.append((new_name, t, ("__BYTES__", data)))
            n_qk_fixed += 1
        else:
            out_tensors.append((new_name, t, llm))
    print(f"  un-permuted {n_qk_fixed} q/k projections (llama.cpp -> HF RoPE layout)")

    metadata = build_metadata(llm, mmproj)

    n_vis = sum(1 for n, _, _ in out_tensors if n.startswith("vis."))
    n_conn = sum(1 for n, _, _ in out_tensors if n.startswith("connector."))
    n_llm = sum(1 for n, _, _ in out_tensors if n.startswith("llm."))
    print(f"\nMapped {len(out_tensors)} tensors "
          f"(vis={n_vis}, connector={n_conn}, llm={n_llm}), skipped {len(skipped)}")
    for sname in skipped:
        print(f"  skipped: {sname}")
    print(f"Metadata: {len(metadata)} KV pairs")

    print(f"\nWriting: {a.output}")
    size = core.write_combined_gguf(a.output, out_tensors, metadata)
    print(f"\nDone: {a.output}  ({size/1024/1024:.1f} MB, {len(out_tensors)} tensors)")


if __name__ == "__main__":
    main()
