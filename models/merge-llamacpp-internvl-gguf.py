#!/usr/bin/env python3
"""Merge a llama.cpp InternVL2.5/3 split GGUF pair (LLM + mmproj) into a single
CrispEmbed "internvl2"-arch GGUF.

llama.cpp exports InternVL as:
  - LLM GGUF   (arch=qwen2):    blk.N.* (separate Q/K/V + biases) + token_embd
                               + output_norm + output, tokenizer.ggml.* (gpt2)
  - mmproj GGUF (projector=internvl): InternViT v.blk.N.* (separate attn_q/k/v,
                               layer-scale ls1/ls2) + v.class_embd + v.patch_embd
                               + v.position_embd + MLP connector mm.model.mlp.*

CrispEmbed's `internvl2_ocr` loader consumes ONE combined GGUF with a different
convention: FUSED vision `attn_qkv`, `vis.*`/`v.blk.*` names, `l.blk.*` LLM,
`v.proj.*` connector, and `internvl2.*` metadata (incl. dynamic-tiling preproc).
This script performs that translation, copying tensor data byte-for-byte.

Grounded in the real ggml-org/InternVL2_5-1B-GGUF files + the native
convert-internvl2-to-gguf.py target format — not guessed. Notable transforms:
  - VISION QKV FUSE: the mmproj splits attn_q/k/v; the loader wants a fused
    attn_qkv — re-concat [q;k;v] (byte concat; vision has no RoPE so no permute).
  - LLM q/k UN-PERMUTE: arch=qwen2 permutes q/k (weights AND biases) for its
    interleaved RoPE; undo it to HF layout (core.llama_unpermute_qk_rows).
  - ViT FFN fc1/fc2 mapped by OUTPUT dim (here ffn_up=fc1, the inverse of
    SmolVLM — proving why name-based mapping is wrong).
  - 4-D Conv2d patch -> 2-D flatten (pure shape relabel, byte-identical).
  - Dynamic-tiling preproc metadata injected per InternVL2.5 defaults.

    python merge-llamacpp-internvl-gguf.py \
        --llm InternVL2_5-1B-Q8_0.gguf \
        --mmproj mmproj-InternVL2_5-1B-f16.gguf \
        --output internvl2_5-1b-crispembed.gguf
"""
import argparse
import sys

import gguf_merge_core as core

INTERNVL_IMAGE_TOKEN = "<IMG_CONTEXT>"  # id injected from the LLM vocab


def _first(md, *keys, default=None):
    for k in keys:
        if k in md:
            return md[k]
    return default


def main():
    ap = argparse.ArgumentParser(description="Merge llama.cpp InternVL -> CrispEmbed internvl2 GGUF")
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

    lm, vm = llm.metadata, mmproj.metadata
    proj = vm.get("clip.projector_type")
    if proj != "internvl":
        print(f"  WARNING: expected clip.projector_type=internvl, got {proj!r}")

    vis = {t.name: t for t in mmproj.tensors}
    n_head = int(_first(lm, "qwen2.attention.head_count", default=14))
    n_kv = int(_first(lm, "qwen2.attention.head_count_kv", default=2))
    head_dim = int(_first(lm, "qwen2.attention.key_length", default=0)) or \
        (int(_first(lm, "qwen2.embedding_length", default=896)) // n_head)

    out_tensors, skipped = [], []
    n_fused = 0

    # ── Vision tower + connector ──
    for t in mmproj.tensors:
        n = t.name
        if n == "v.patch_embd.weight":
            out = t.shape[-1]
            in_flat = 1
            for d in t.shape[:-1]:
                in_flat *= d
            info = core.TensorInfo(n, [in_flat, out], t.dtype, t.offset, t.nbytes)
            out_tensors.append(("v.patch_embed.weight", info, mmproj))
        elif n == "v.patch_embd.bias":
            out_tensors.append(("v.patch_embed.bias", t, mmproj))
        elif n == "v.class_embd":
            out_tensors.append(("v.class_embedding", t, mmproj))
        elif n == "v.position_embd.weight":
            out_tensors.append(("v.position_embedding", t, mmproj))
        elif n == "mm.model.mlp.0.weight":
            out_tensors.append(("v.proj.norm.weight", t, mmproj))
        elif n == "mm.model.mlp.0.bias":
            out_tensors.append(("v.proj.norm.bias", t, mmproj))
        elif n == "mm.model.mlp.1.weight":
            out_tensors.append(("v.proj.fc1.weight", t, mmproj))
        elif n == "mm.model.mlp.1.bias":
            out_tensors.append(("v.proj.fc1.bias", t, mmproj))
        elif n == "mm.model.mlp.3.weight":
            out_tensors.append(("v.proj.fc2.weight", t, mmproj))
        elif n == "mm.model.mlp.3.bias":
            out_tensors.append(("v.proj.fc2.bias", t, mmproj))
        elif n.startswith("v.blk."):
            rest = n[len("v.blk."):]
            li, _, suffix = rest.partition(".")
            simple = {
                "ln1.weight": "norm1.weight", "ln1.bias": "norm1.bias",
                "ln2.weight": "norm2.weight", "ln2.bias": "norm2.bias",
                "ls1.weight": "ls1", "ls2.weight": "ls2",
                "attn_out.weight": "attn_proj.weight", "attn_out.bias": "attn_proj.bias",
            }.get(suffix)
            if simple is not None:
                out_tensors.append((f"v.blk.{li}.{simple}", t, mmproj))
            elif suffix.startswith(("attn_q.", "attn_k.", "attn_v.")):
                # Fuse Q,K,V -> attn_qkv exactly once per (layer, field).
                field = "weight" if suffix.endswith(".weight") else "bias"
                if not suffix.startswith("attn_q."):
                    continue  # handled when we hit attn_q.<field>
                q = vis[f"v.blk.{li}.attn_q.{field}"]
                k = vis[f"v.blk.{li}.attn_k.{field}"]
                v = vis[f"v.blk.{li}.attn_v.{field}"]
                data = (core.read_tensor_data(mmproj, q) + core.read_tensor_data(mmproj, k)
                        + core.read_tensor_data(mmproj, v))
                if field == "weight":
                    in_dim = q.shape[0]
                    out_dim = q.shape[-1] + k.shape[-1] + v.shape[-1]
                    shape = [in_dim, out_dim]
                else:
                    shape = [q.shape[0] + k.shape[0] + v.shape[0]]
                info = core.TensorInfo("attn_qkv", shape, q.dtype, 0, len(data))
                out_tensors.append((f"v.blk.{li}.attn_qkv.{field}", info, ("__BYTES__", data)))
                n_fused += 1
            elif suffix.startswith(("ffn_up.", "ffn_down.")):
                field = "weight" if suffix.endswith(".weight") else "bias"
                up = vis[f"v.blk.{li}.ffn_up.{field}"]
                dn = vis[f"v.blk.{li}.ffn_down.{field}"]
                up_out = up.shape[-1] if field == "weight" else up.shape[0]
                dn_out = dn.shape[-1] if field == "weight" else dn.shape[0]
                this_up = suffix.startswith("ffn_up.")
                fc1_is_up = up_out >= dn_out          # fc1 = larger (intermediate) out
                fc = "fc1" if (this_up == fc1_is_up) else "fc2"
                out_tensors.append((f"v.blk.{li}.ffn_{fc}.{field}", t, mmproj))
            else:
                skipped.append(n)
        else:
            skipped.append(n)

    # ── LLM: rename to l.*. q/k un-permute is arch-dependent: llama.cpp permutes
    # q/k ONLY for interleaved-RoPE arches (llama); NEOX-RoPE arches (qwen2) are
    # stored in HF layout already, so un-permuting them would SCRAMBLE them. ──
    llm_arch = lm.get("general.architecture", "")
    needs_unpermute = llm_arch in ("llama", "mistral", "gemma")  # NORMAL-rope arches
    n_unperm = 0
    for t in llm.tensors:
        n = t.name
        if n == "token_embd.weight":
            out_tensors.append(("l.embed_tokens.weight", t, llm))
        elif n == "output_norm.weight":
            out_tensors.append(("l.output_norm.weight", t, llm))
        elif n == "output.weight":
            out_tensors.append(("l.lm_head.weight", t, llm))
        elif n.startswith("blk."):
            rest = n[len("blk."):]
            li, _, suffix = rest.partition(".")
            base = {
                "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
                "attn_output.weight": "attn_o.weight",
                "ffn_gate.weight": "ffn_gate.weight", "ffn_up.weight": "ffn_up.weight",
                "ffn_down.weight": "ffn_down.weight",
                "attn_v.weight": "attn_v.weight", "attn_v.bias": "attn_v.bias",
            }.get(suffix)
            if base is not None:
                out_tensors.append((f"l.blk.{li}.{base}", t, llm))
            elif suffix.startswith(("attn_q.", "attn_k.")):
                is_q = suffix.startswith("attn_q.")
                dst = ("attn_q" if is_q else "attn_k") + ("." + suffix.split(".", 1)[1])
                if needs_unpermute:
                    nh = n_head if is_q else n_kv
                    rows = t.shape[-1] if suffix.endswith(".weight") else t.shape[0]
                    data = core.llama_unpermute_qk_rows(core.read_tensor_data(llm, t), rows, nh, head_dim)
                    out_tensors.append((f"l.blk.{li}.{dst}",
                                        core.TensorInfo(n, t.shape, t.dtype, 0, len(data)),
                                        ("__BYTES__", data)))
                    n_unperm += 1
                else:
                    out_tensors.append((f"l.blk.{li}.{dst}", t, llm))
            else:
                skipped.append(n)
        else:
            skipped.append(n)

    # ── Metadata ──
    ARCH = "internvl2"
    kv = []

    def u32(k, v):
        kv.append((k, core.GGUF_TYPE_UINT32, int(v)))

    def f32(k, v):
        kv.append((k, core.GGUF_TYPE_FLOAT32, float(v)))

    def s(k, v):
        kv.append((k, core.GGUF_TYPE_STRING, v))

    def b(k, v):
        kv.append((k, core.GGUF_TYPE_BOOL, bool(v)))

    s("general.architecture", ARCH)
    if "general.name" in lm:
        s("general.name", lm["general.name"])
    s("general.source", "merged from llama.cpp InternVL (qwen2 LLM + internvl mmproj)")

    vis_hidden = int(_first(vm, "clip.vision.embedding_length", default=1024))
    vis_patch = int(_first(vm, "clip.vision.patch_size", default=14))
    vis_image = int(_first(vm, "clip.vision.image_size", default=448))
    downsample = 0.5
    n_patches = (vis_image // vis_patch) ** 2
    n_merged = int(n_patches * downsample ** 2)
    merge_dim = int(vis_hidden / (downsample ** 2))
    u32(f"{ARCH}.vision.num_hidden_layers", int(_first(vm, "clip.vision.block_count", default=24)))
    u32(f"{ARCH}.vision.hidden_size", vis_hidden)
    u32(f"{ARCH}.vision.intermediate_size", int(_first(vm, "clip.vision.feed_forward_length", default=4096)))
    u32(f"{ARCH}.vision.num_attention_heads", int(_first(vm, "clip.vision.attention.head_count", default=16)))
    u32(f"{ARCH}.vision.patch_size", vis_patch)
    u32(f"{ARCH}.vision.image_size", vis_image)
    f32(f"{ARCH}.vision.layer_norm_eps", float(_first(vm, "clip.vision.attention.layer_norm_epsilon", default=1e-6)))
    b(f"{ARCH}.vision.qkv_bias", True)
    f32(f"{ARCH}.downsample_ratio", downsample)
    s(f"{ARCH}.ps_version", "v2")
    u32(f"{ARCH}.vision.num_merged_tokens", n_merged)
    u32(f"{ARCH}.vision.merge_dim", merge_dim)
    u32(f"{ARCH}.max_dynamic_patch", 12)
    u32(f"{ARCH}.min_dynamic_patch", 1)
    b(f"{ARCH}.use_thumbnail", True)
    # image mean/std (InternVL ImageNet stats; the mmproj carries them).
    for dst, src in ((f"{ARCH}.vision.image_mean", "clip.vision.image_mean"),
                     (f"{ARCH}.vision.image_std", "clip.vision.image_std")):
        if src in vm:
            kv.append((dst, core.GGUF_TYPE_ARRAY, (core.GGUF_TYPE_FLOAT32, [float(x) for x in vm[src]])))

    u32(f"{ARCH}.vocab_size", int(_first(lm, "qwen2.vocab_size",
                                         default=len(lm.get("tokenizer.ggml.tokens", [])) or 151674)))
    u32(f"{ARCH}.hidden_size", int(_first(lm, "qwen2.embedding_length", default=896)))
    u32(f"{ARCH}.intermediate_size", int(_first(lm, "qwen2.feed_forward_length", default=4864)))
    u32(f"{ARCH}.num_hidden_layers", int(_first(lm, "qwen2.block_count", default=24)))
    u32(f"{ARCH}.num_attention_heads", n_head)
    u32(f"{ARCH}.num_key_value_heads", n_kv)
    u32(f"{ARCH}.max_position_embeddings", int(_first(lm, "qwen2.context_length", default=32768)))
    f32(f"{ARCH}.rms_norm_eps", float(_first(lm, "qwen2.attention.layer_norm_rms_epsilon", default=1e-6)))
    f32(f"{ARCH}.rope_theta", float(_first(lm, "qwen2.rope.freq_base", default=1000000.0)))
    b(f"{ARCH}.tie_word_embeddings", not any(n == "l.lm_head.weight" for n, _, _ in out_tensors))
    s(f"{ARCH}.hidden_act", "silu")

    # image_token_id from the LLM vocab (<IMG_CONTEXT>).
    toks = lm.get("tokenizer.ggml.tokens", [])
    if INTERNVL_IMAGE_TOKEN in toks:
        u32(f"{ARCH}.image_token_id", toks.index(INTERNVL_IMAGE_TOKEN))
    else:
        print(f"  WARNING: {INTERNVL_IMAGE_TOKEN} not in vocab")

    # Tokenizer: pass through the gpt2 BPE arrays verbatim (loader reads
    # tokenizer.ggml.* directly). Arrays lost element type on read -> re-infer.
    def _elem(vals):
        x = vals[0]
        if isinstance(x, bool):
            return core.GGUF_TYPE_BOOL
        if isinstance(x, str):
            return core.GGUF_TYPE_STRING
        if isinstance(x, float):
            return core.GGUF_TYPE_FLOAT32
        return core.GGUF_TYPE_INT32
    for key, val in lm.items():
        if not key.startswith("tokenizer."):
            continue
        vtype = llm.metadata_types[key]
        if vtype == core.GGUF_TYPE_ARRAY:
            if val:
                kv.append((key, core.GGUF_TYPE_ARRAY, (_elem(val), val)))
        else:
            kv.append((key, vtype, val))

    n_vis = sum(1 for n, _, _ in out_tensors if n.startswith("v.") and not n.startswith("v.proj"))
    n_proj = sum(1 for n, _, _ in out_tensors if n.startswith("v.proj"))
    n_llm = sum(1 for n, _, _ in out_tensors if n.startswith("l."))
    print(f"\nMapped {len(out_tensors)} tensors (vis={n_vis}, proj={n_proj}, llm={n_llm}); "
          f"fused {n_fused} vision QKV, un-permuted {n_unperm} LLM q/k; skipped {len(skipped)}")
    for sk in skipped:
        print(f"  skipped: {sk}")
    print(f"Metadata: {len(kv)} KV pairs")

    print(f"\nWriting: {a.output}")
    size = core.write_combined_gguf(a.output, out_tensors, kv)
    print(f"\nDone: {a.output}  ({size/1024/1024:.1f} MB, {len(out_tensors)} tensors)")


if __name__ == "__main__":
    main()
