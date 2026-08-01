#!/usr/bin/env python3
"""Regression test for the llama.cpp SmolVLM (Idefics3) -> CrispEmbed merge.

Validated end-to-end on 2026-07-12: a stock ggml-org/SmolVLM-256M-Instruct
llama.cpp GGUF pair, merged by merge-llamacpp-smolvlm-gguf.py, loads in
CrispEmbed's smoldocling engine and OCRs correctly on Metal. This test guards
the merge's structural transforms WITHOUT any model download:

  1. q/k un-permute (llama.cpp interleaved-RoPE layout -> HF rotate_half). This
     was THE bug: without it the LLM degenerates ("The The [ ["). Tested
     directly as an exact inverse of llama.cpp's forward permute.
  2. SigLIP FFN fc1/fc2 name inversion (mapped by output dim, not name).
  3. 4D Conv2d patch -> 2D flatten (pure shape relabel, byte-identical).
  4. Native tensor names (vis.layers.*, llm.layers.*, connector.proj) +
     smoldocling.* metadata + gpt2 tokenizer passthrough + arch=smoldocling.

Run:  ~/miniconda3/bin/python tests/test_mmproj_smolvlm.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")
MERGE = os.path.join(MODELS, "merge-llamacpp-smolvlm-gguf.py")

# Load the hyphenated merge module to unit-test its un-permute helper.
_spec = importlib.util.spec_from_file_location("smolvlm_merge", MERGE)
smolvlm_merge = importlib.util.module_from_spec(_spec)
sys.path.insert(0, MODELS)  # so it can `import gguf_merge_core`
_spec.loader.exec_module(smolvlm_merge)

F16 = GGMLQuantizationType.F16
F32 = GGMLQuantizationType.F32

# Tiny SmolVLM-shaped dims.
V_HID, V_HEADS, V_LAYERS, V_INTER, V_PATCH, V_IN = 8, 2, 2, 16, 4, 3
SCALE = 2                       # pixel-shuffle scale
L_HID, L_HEADS, L_KV, L_LAYERS, L_INTER, HEAD_DIM = 8, 2, 1, 2, 16, 4
VOCAB = 20
CONN_IN = V_HID * SCALE * SCALE  # 32


def _field_contents(field):
    if hasattr(field, "contents"):
        return field.contents()
    type_names = [getattr(t, "name", str(t)) for t in field.types]
    if type_names and type_names[0] == "STRING":
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if type_names and type_names[0] == "ARRAY":
        if len(type_names) > 1 and type_names[1] == "STRING":
            return [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        return [field.parts[i][0].item() for i in field.data]
    value = field.parts[field.data[0]][0]
    return value.item() if hasattr(value, "item") else value


def _f(qt, *shape):
    n = int(np.prod(shape))
    npt = np.float16 if qt == F16 else np.float32
    return (np.arange(n, dtype=np.float32).reshape(shape) * 0.01).astype(npt)


def llama_forward_permute(w, n_head, head_dim):
    """Mirror llama.cpp convert_hf_to_gguf LlamaModel.permute (HF -> llama)."""
    out = w.shape[0]
    return (w.reshape(n_head, 2, head_dim // 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(out, *w.shape[1:]))


def test_unpermute_inverse():
    """un-permute must exactly invert llama.cpp's forward permute, on raw bytes,
    for both the q (n_head) and k (n_head_kv) shapes."""
    fails = []
    for tag, nh, out in (("q", L_HEADS, L_HEADS * HEAD_DIM), ("k", L_KV, L_KV * HEAD_DIM)):
        hf = (np.arange(out * L_HID, dtype=np.float32) * 0.5).reshape(out, L_HID)
        llama = llama_forward_permute(hf, nh, HEAD_DIM)          # what llama.cpp stores
        recovered = smolvlm_merge.llama_unpermute_qk_rows(
            llama.tobytes(), out, nh, HEAD_DIM)
        rec = np.frombuffer(recovered, dtype=np.float32).reshape(out, L_HID)
        ok = np.array_equal(rec, hf)
        print(f"  {'ok  ' if ok else 'FAIL'} un-permute inverts forward permute ({tag}, n_head={nh})")
        if not ok:
            fails.append(f"unpermute-{tag}")
    return fails


def build_mmproj(path):
    w = GGUFWriter(path, "clip")
    w.add_type("clip-vision")
    w.add_name("tiny-smolvlm")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "idefics3")
    w.add_uint32("clip.vision.projection_dim", L_HID)
    w.add_uint32("clip.vision.image_size", V_PATCH * 2)
    w.add_uint32("clip.vision.patch_size", V_PATCH)
    w.add_uint32("clip.vision.embedding_length", V_HID)
    w.add_uint32("clip.vision.feed_forward_length", V_INTER)
    w.add_uint32("clip.vision.block_count", V_LAYERS)
    w.add_uint32("clip.vision.attention.head_count", V_HEADS)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_uint32("clip.vision.projector.scale_factor", SCALE)
    w.add_file_type(1)
    # Patch: 4D conv (out, in, H, W); position; post_ln; connector.
    w.add_tensor("v.patch_embd.weight", _f(F32, V_HID, V_IN, V_PATCH, V_PATCH), raw_dtype=F32)
    w.add_tensor("v.patch_embd.bias", _f(F32, V_HID), raw_dtype=F32)
    w.add_tensor("v.position_embd.weight", _f(F32, (V_PATCH * 2 // V_PATCH) ** 2, V_HID), raw_dtype=F32)
    w.add_tensor("v.post_ln.weight", _f(F32, V_HID), raw_dtype=F32)
    w.add_tensor("v.post_ln.bias", _f(F32, V_HID), raw_dtype=F32)
    w.add_tensor("mm.model.fc.weight", _f(F16, L_HID, CONN_IN), raw_dtype=F16)
    for b in range(V_LAYERS):
        p = f"v.blk.{b}."
        for nm in ("attn_q", "attn_k", "attn_v", "attn_out"):
            w.add_tensor(p + nm + ".weight", _f(F16, V_HID, V_HID), raw_dtype=F16)
            w.add_tensor(p + nm + ".bias", _f(F32, V_HID), raw_dtype=F32)
        for nm in ("ln1", "ln2"):
            w.add_tensor(p + nm + ".weight", _f(F32, V_HID), raw_dtype=F32)
            w.add_tensor(p + nm + ".bias", _f(F32, V_HID), raw_dtype=F32)
        # llama.cpp INVERTS: ffn_down = fc1 (out=inter), ffn_up = fc2 (out=hid).
        # numpy is (out, in); GGUFWriter stores ne=(in, out).
        w.add_tensor(p + "ffn_down.weight", _f(F16, V_INTER, V_HID), raw_dtype=F16)  # out=inter -> fc1
        w.add_tensor(p + "ffn_down.bias", _f(F32, V_INTER), raw_dtype=F32)
        w.add_tensor(p + "ffn_up.weight", _f(F16, V_HID, V_INTER), raw_dtype=F16)    # out=hid -> fc2
        w.add_tensor(p + "ffn_up.bias", _f(F32, V_HID), raw_dtype=F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def build_llm(path):
    w = GGUFWriter(path, "llama")
    w.add_name("tiny-smolvlm")
    w.add_uint32("llama.block_count", L_LAYERS)
    w.add_uint32("llama.embedding_length", L_HID)
    w.add_uint32("llama.feed_forward_length", L_INTER)
    w.add_uint32("llama.attention.head_count", L_HEADS)
    w.add_uint32("llama.attention.head_count_kv", L_KV)
    w.add_uint32("llama.attention.key_length", HEAD_DIM)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_float32("llama.rope.freq_base", 100000.0)
    w.add_uint32("llama.vocab_size", VOCAB)
    w.add_file_type(1)
    # gpt2 BPE tokenizer arrays (loader reads tokenizer.ggml.* fallback).
    w.add_tokenizer_model("gpt2")
    w.add_token_list([f"tok{i}" for i in range(VOCAB)])
    w.add_token_merges(["t o", "to k"])
    w.add_tensor("token_embd.weight", _f(F16, VOCAB, L_HID), raw_dtype=F16)
    w.add_tensor("output_norm.weight", _f(F32, L_HID), raw_dtype=F32)
    w.add_tensor("output.weight", _f(F16, VOCAB, L_HID), raw_dtype=F16)
    for b in range(L_LAYERS):
        p = f"blk.{b}."
        # numpy is (out, in); GGUFWriter stores ne=(in, out). q/k/v out = heads*head_dim.
        w.add_tensor(p + "attn_q.weight", _f(F16, L_HEADS * HEAD_DIM, L_HID), raw_dtype=F16)
        w.add_tensor(p + "attn_k.weight", _f(F16, L_KV * HEAD_DIM, L_HID), raw_dtype=F16)
        w.add_tensor(p + "attn_v.weight", _f(F16, L_KV * HEAD_DIM, L_HID), raw_dtype=F16)
        w.add_tensor(p + "attn_output.weight", _f(F16, L_HID, L_HEADS * HEAD_DIM), raw_dtype=F16)
        w.add_tensor(p + "attn_norm.weight", _f(F32, L_HID), raw_dtype=F32)
        w.add_tensor(p + "ffn_gate.weight", _f(F16, L_INTER, L_HID), raw_dtype=F16)
        w.add_tensor(p + "ffn_up.weight", _f(F16, L_INTER, L_HID), raw_dtype=F16)
        w.add_tensor(p + "ffn_down.weight", _f(F16, L_HID, L_INTER), raw_dtype=F16)
        w.add_tensor(p + "ffn_norm.weight", _f(F32, L_HID), raw_dtype=F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_merge_structure():
    fails = []

    def check(cond, msg):
        print(("  ok  " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
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
            return fails

        rr = GGUFReader(out)
        md = {f.name: _field_contents(f) for f in rr.fields.values()}
        tn = {t.name: t for t in rr.tensors}

        check(md.get("general.architecture") == "smoldocling", "arch == smoldocling")
        check(int(md.get("smoldocling.image_token_id", -1)) == 49190,
              "image_token_id injected == 49190")
        check(int(md.get("smoldocling.connector.scale_factor", -1)) == SCALE,
              f"connector.scale_factor == {SCALE}")
        check(int(md.get("smoldocling.num_hidden_layers", -1)) == L_LAYERS, "llm num_hidden_layers")
        for n in ["vis.patch_embed.weight", "vis.pos_embed.weight", "vis.post_ln.weight",
                  "vis.layers.0.attn.q.weight", "connector.proj.weight",
                  "llm.embed.weight", "llm.lm_head.weight", "llm.layers.0.attn.q.weight",
                  "llm.layers.0.ffn.gate.weight"]:
            check(n in tn, f"native name present: {n}")
        # No llama.cpp-native names leaked through.
        leaked = [n for n in tn if n.startswith("v.blk.") or n.startswith("blk.") or n == "mm.model.fc.weight"]
        check(not leaked, f"no llama.cpp-native names leaked ({leaked[:2]})")

        # fc1/fc2 by output dim: fc1 out=inter, fc2 out=hid.
        fc1 = tn.get("vis.layers.0.mlp.fc1.weight")
        fc2 = tn.get("vis.layers.0.mlp.fc2.weight")
        if fc1 is not None and fc2 is not None:
            check(int(fc1.shape[-1]) == V_INTER, f"vis fc1 out == intermediate ({V_INTER})")
            check(int(fc2.shape[-1]) == V_HID, f"vis fc2 out == hidden ({V_HID})")

        # Patch flatten: 4D -> 2D [in*H*W, out].
        pe = tn.get("vis.patch_embed.weight")
        if pe is not None:
            check(sorted(int(x) for x in pe.shape) == sorted([V_IN * V_PATCH * V_PATCH, V_HID]),
                  f"patch flattened to [{V_IN*V_PATCH*V_PATCH}, {V_HID}] shape {list(pe.shape)}")

        # q/k un-permuted: merged q/k bytes must equal un-permute(fixture bytes).
        src = GGUFReader(llm)
        srct = {t.name: t for t in src.tensors}
        q_src = np.array(srct["blk.0.attn_q.weight"].data)
        q_exp = smolvlm_merge.llama_unpermute_qk_rows(
            q_src.tobytes(), L_HEADS * HEAD_DIM, L_HEADS, HEAD_DIM)
        q_got = np.array(tn["llm.layers.0.attn.q.weight"].data).tobytes()
        check(q_got == q_exp, "merged llm q weight is un-permuted")
        # v is NOT permuted (must be copied verbatim).
        v_got = np.array(tn["llm.layers.0.attn.v.weight"].data)
        v_src = np.array(srct["blk.0.attn_v.weight"].data)
        check(np.array_equal(v_got, v_src), "merged llm v weight copied verbatim (not permuted)")

        # Tokenizer passthrough.
        check(len(md.get("tokenizer.ggml.tokens", [])) == VOCAB, "tokenizer.ggml.tokens passed through")
        check("tokenizer.ggml.merges" in md, "tokenizer.ggml.merges passed through")
    return fails


def main():
    fails = []
    print("[unit] q/k un-permute inverse")
    fails += test_unpermute_inverse()
    print("\n[merge] synthetic SmolVLM fixtures -> CrispEmbed smoldocling GGUF")
    fails += test_merge_structure()
    print("\n" + ("ALL PASS" if not fails else f"FAILED ({len(fails)}): " + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
