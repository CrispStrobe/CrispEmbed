#!/usr/bin/env python3
"""Regression test for the llama.cpp InternVL2.5/3 -> CrispEmbed merge.

Validated end-to-end on 2026-07-12: a stock ggml-org/InternVL2_5-1B-GGUF pair,
merged by merge-llamacpp-internvl-gguf.py, loads in CrispEmbed's internvl2
engine and OCRs correctly, and the diff-harness intermediates match the native
converter to 6 decimals. This test guards the structural transforms WITHOUT any
download:

  1. VISION QKV FUSE — the mmproj splits attn_q/k/v; the loader wants a fused
     attn_qkv. Re-concat [q;k;v] (byte concat); separate q/k/v must be gone.
  2. ARCH-CONDITIONAL q/k un-permute — arch=qwen2 uses NEOX RoPE (NOT permuted),
     so q/k copy VERBATIM; un-permuting them scrambles the LLM (was THE bug).
     (arch=llama IS permuted — covered by test_mmproj_smolvlm.py.)
  3. ViT FFN fc1/fc2 by OUTPUT dim — here ffn_up=fc1 (the INVERSE of SmolVLM),
     proving name-based mapping is wrong.
  4. Connector mm.model.mlp.{0,1,3} -> v.proj.{norm,fc1,fc2}; layer-scale
     ls1/ls2 (drop .weight); class/position embedding rename; patch flatten;
     dynamic-tiling metadata; gpt2 tokenizer passthrough; arch=internvl2.

Run:  ~/miniconda3/bin/python tests/test_mmproj_internvl.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")
MERGE = os.path.join(MODELS, "merge-llamacpp-internvl-gguf.py")

F16 = GGMLQuantizationType.F16
F32 = GGMLQuantizationType.F32

V_HID, V_HEADS, V_LAYERS, V_INTER, V_PATCH, V_IN = 8, 2, 2, 16, 4, 3
L_HID, L_HEADS, L_KV, L_LAYERS, L_INTER, HEAD_DIM = 8, 2, 1, 2, 16, 4
VOCAB = 24
IMG_TOKEN_ID = 20  # index of <IMG_CONTEXT> in the fixture vocab


def _f(qt, *shape):
    n = int(np.prod(shape))
    npt = np.float16 if qt == F16 else np.float32
    return (np.arange(n, dtype=np.float32).reshape(shape) * 0.01 + 0.05).astype(npt)


def build_mmproj(path):
    w = GGUFWriter(path, "clip")
    w.add_type("clip-vision")
    w.add_name("tiny-internvl")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "internvl")
    w.add_uint32("clip.vision.projection_dim", L_HID)
    w.add_uint32("clip.vision.image_size", V_PATCH * 2)
    w.add_uint32("clip.vision.patch_size", V_PATCH)
    w.add_uint32("clip.vision.embedding_length", V_HID)
    w.add_uint32("clip.vision.feed_forward_length", V_INTER)
    w.add_uint32("clip.vision.block_count", V_LAYERS)
    w.add_uint32("clip.vision.attention.head_count", V_HEADS)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_uint32("clip.vision.projector.scale_factor", 2)
    w.add_array("clip.vision.image_mean", [0.485, 0.456, 0.406])
    w.add_array("clip.vision.image_std", [0.229, 0.224, 0.225])
    w.add_file_type(1)
    n_pos = (V_PATCH * 2 // V_PATCH) ** 2 + 1  # +1 cls
    w.add_tensor("v.patch_embd.weight", _f(F16, V_HID, V_IN, V_PATCH, V_PATCH), raw_dtype=F16)
    w.add_tensor("v.patch_embd.bias", _f(F32, V_HID), raw_dtype=F32)
    w.add_tensor("v.class_embd", _f(F32, V_HID, 1, 1), raw_dtype=F32)
    w.add_tensor("v.position_embd.weight", _f(F32, V_HID, n_pos, 1), raw_dtype=F32)
    # MLP connector: mlp.0=LayerNorm(merge_dim), mlp.1=Linear(merge_dim->llm),
    # mlp.3=Linear(llm->llm). merge_dim = V_HID * scale^2 = 8*4 = 32.
    merge_dim = V_HID * 4
    w.add_tensor("mm.model.mlp.0.weight", _f(F32, merge_dim), raw_dtype=F32)
    w.add_tensor("mm.model.mlp.0.bias", _f(F32, merge_dim), raw_dtype=F32)
    w.add_tensor("mm.model.mlp.1.weight", _f(F16, L_HID, merge_dim), raw_dtype=F16)
    w.add_tensor("mm.model.mlp.1.bias", _f(F32, L_HID), raw_dtype=F32)
    w.add_tensor("mm.model.mlp.3.weight", _f(F16, L_HID, L_HID), raw_dtype=F16)
    w.add_tensor("mm.model.mlp.3.bias", _f(F32, L_HID), raw_dtype=F32)
    for b in range(V_LAYERS):
        p = f"v.blk.{b}."
        for nm in ("attn_q", "attn_k", "attn_v", "attn_out"):
            w.add_tensor(p + nm + ".weight", _f(F16, V_HID, V_HID), raw_dtype=F16)
            w.add_tensor(p + nm + ".bias", _f(F32, V_HID), raw_dtype=F32)
        for nm in ("ln1", "ln2"):
            w.add_tensor(p + nm + ".weight", _f(F32, V_HID), raw_dtype=F32)
            w.add_tensor(p + nm + ".bias", _f(F32, V_HID), raw_dtype=F32)
        w.add_tensor(p + "ls1.weight", _f(F32, V_HID), raw_dtype=F32)
        w.add_tensor(p + "ls2.weight", _f(F32, V_HID), raw_dtype=F32)
        # InternVL INVERTS opposite to SmolVLM: ffn_up=fc1 (out=inter).
        w.add_tensor(p + "ffn_up.weight", _f(F16, V_INTER, V_HID), raw_dtype=F16)   # out=inter -> fc1
        w.add_tensor(p + "ffn_up.bias", _f(F32, V_INTER), raw_dtype=F32)
        w.add_tensor(p + "ffn_down.weight", _f(F16, V_HID, V_INTER), raw_dtype=F16)  # out=hid -> fc2
        w.add_tensor(p + "ffn_down.bias", _f(F32, V_HID), raw_dtype=F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def build_llm(path):
    w = GGUFWriter(path, "qwen2")
    w.add_name("tiny-internvl")
    w.add_uint32("qwen2.block_count", L_LAYERS)
    w.add_uint32("qwen2.embedding_length", L_HID)
    w.add_uint32("qwen2.feed_forward_length", L_INTER)
    w.add_uint32("qwen2.attention.head_count", L_HEADS)
    w.add_uint32("qwen2.attention.head_count_kv", L_KV)
    w.add_uint32("qwen2.attention.key_length", HEAD_DIM)
    w.add_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("qwen2.rope.freq_base", 1e6)
    w.add_uint32("qwen2.context_length", 4096)
    w.add_uint32("qwen2.vocab_size", VOCAB)
    w.add_file_type(1)
    toks = [f"t{i}" for i in range(VOCAB)]
    toks[IMG_TOKEN_ID] = "<IMG_CONTEXT>"
    w.add_tokenizer_model("gpt2")
    w.add_token_list(toks)
    w.add_token_merges(["t 0", "t 1"])
    w.add_tensor("token_embd.weight", _f(F16, VOCAB, L_HID), raw_dtype=F16)
    w.add_tensor("output_norm.weight", _f(F32, L_HID), raw_dtype=F32)
    w.add_tensor("output.weight", _f(F16, VOCAB, L_HID), raw_dtype=F16)
    for b in range(L_LAYERS):
        p = f"blk.{b}."
        w.add_tensor(p + "attn_q.weight", _f(F16, L_HEADS * HEAD_DIM, L_HID), raw_dtype=F16)
        w.add_tensor(p + "attn_q.bias", _f(F32, L_HEADS * HEAD_DIM), raw_dtype=F32)
        w.add_tensor(p + "attn_k.weight", _f(F16, L_KV * HEAD_DIM, L_HID), raw_dtype=F16)
        w.add_tensor(p + "attn_k.bias", _f(F32, L_KV * HEAD_DIM), raw_dtype=F32)
        w.add_tensor(p + "attn_v.weight", _f(F16, L_KV * HEAD_DIM, L_HID), raw_dtype=F16)
        w.add_tensor(p + "attn_v.bias", _f(F32, L_KV * HEAD_DIM), raw_dtype=F32)
        w.add_tensor(p + "attn_output.weight", _f(F16, L_HID, L_HEADS * HEAD_DIM), raw_dtype=F16)
        w.add_tensor(p + "attn_norm.weight", _f(F32, L_HID), raw_dtype=F32)
        w.add_tensor(p + "ffn_gate.weight", _f(F16, L_INTER, L_HID), raw_dtype=F16)
        w.add_tensor(p + "ffn_up.weight", _f(F16, L_INTER, L_HID), raw_dtype=F16)
        w.add_tensor(p + "ffn_down.weight", _f(F16, L_HID, L_INTER), raw_dtype=F16)
        w.add_tensor(p + "ffn_norm.weight", _f(F32, L_HID), raw_dtype=F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main():
    fails = []

    def check(cond, msg):
        print(("  ok  " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as d:
        llm = os.path.join(d, "llm.gguf")
        mmproj = os.path.join(d, "mmproj.gguf")
        out = os.path.join(d, "crisp.gguf")
        build_llm(llm)
        build_mmproj(mmproj)
        r = subprocess.run([sys.executable, MERGE, "--llm", llm, "--mmproj", mmproj,
                            "--output", out], capture_output=True, text=True)
        check(r.returncode == 0, f"merge exits 0 (rc={r.returncode})")
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            return 1

        rr = GGUFReader(out)
        md = {f.name: f.contents() for f in rr.fields.values()}
        tn = {t.name: t for t in rr.tensors}

        check(md.get("general.architecture") == "internvl2", "arch == internvl2")
        check(int(md.get("internvl2.image_token_id", -1)) == IMG_TOKEN_ID,
              f"image_token_id == {IMG_TOKEN_ID} (<IMG_CONTEXT>)")
        # dynamic-tiling metadata
        n_patches = (V_PATCH * 2 // V_PATCH) ** 2
        check(int(md.get("internvl2.vision.num_merged_tokens", -1)) == int(n_patches * 0.25),
              "num_merged_tokens = n_patches * downsample^2")
        check(int(md.get("internvl2.vision.merge_dim", -1)) == V_HID * 4, "merge_dim = hid/downsample^2")
        check(int(md.get("internvl2.max_dynamic_patch", -1)) == 12, "max_dynamic_patch injected")
        check(md.get("internvl2.use_thumbnail") in (True, 1), "use_thumbnail injected")

        # 1. Vision QKV fused; separate q/k/v gone.
        check("v.blk.0.attn_qkv.weight" in tn, "vision attn_qkv fused (weight)")
        check("v.blk.0.attn_qkv.bias" in tn, "vision attn_qkv fused (bias)")
        check(not any(n.startswith("v.blk.0.attn_q.") for n in tn), "separate vision attn_q gone")
        qkv = tn.get("v.blk.0.attn_qkv.weight")
        if qkv is not None:
            check(int(qkv.shape[-1]) == 3 * V_HID, f"fused qkv out == 3*hid ({3*V_HID})")

        # 2. qwen2 q/k copied VERBATIM (NOT un-permuted).
        src = GGUFReader(llm)
        st = {t.name: t for t in src.tensors}
        for tag in ("attn_q", "attn_k"):
            got = np.array(tn[f"l.blk.0.{tag}.weight"].data)
            exp = np.array(st[f"blk.0.{tag}.weight"].data)
            check(np.array_equal(got, exp), f"qwen2 l.blk.0.{tag} copied verbatim (no un-permute)")

        # 3. fc1/fc2 by output dim (InternVL: fc1 out=inter).
        fc1 = tn.get("v.blk.0.ffn_fc1.weight")
        fc2 = tn.get("v.blk.0.ffn_fc2.weight")
        if fc1 is not None and fc2 is not None:
            check(int(fc1.shape[-1]) == V_INTER, f"vis fc1 out == intermediate ({V_INTER})")
            check(int(fc2.shape[-1]) == V_HID, f"vis fc2 out == hidden ({V_HID})")

        # 4. Connector, layer-scale, class/pos, patch flatten, tokenizer.
        for n in ["v.proj.norm.weight", "v.proj.fc1.weight", "v.proj.fc2.weight",
                  "v.class_embedding", "v.position_embedding",
                  "v.blk.0.ls1", "v.blk.0.ls2", "v.blk.0.norm1.weight",
                  "l.embed_tokens.weight", "l.lm_head.weight"]:
            check(n in tn, f"native name present: {n}")
        check(not any(n.startswith("mm.model.mlp") for n in tn), "raw mm.model.mlp.* gone")
        pe = tn.get("v.patch_embed.weight")
        if pe is not None:
            check(sorted(int(x) for x in pe.shape) == sorted([V_IN * V_PATCH * V_PATCH, V_HID]),
                  f"patch flattened to [{V_IN*V_PATCH*V_PATCH}, {V_HID}] {list(pe.shape)}")
        check(len(md.get("tokenizer.ggml.tokens", [])) == VOCAB, "tokenizer.ggml.tokens passed through")

    print("\n" + ("ALL PASS" if not fails else f"FAILED ({len(fails)}): " + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
