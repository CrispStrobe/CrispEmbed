#!/usr/bin/env python3
"""Regression test for the llama.cpp <-> CrispEmbed Qwen2-VL mmproj interop.

Guards the bug classes fixed on 2026-07-12 when the reverse-interop
(llama.cpp mmproj -> CrispEmbed loadable GGUF) was brought to a working
state. All four bugs were silent — the merge/export scripts produced a
file that loaded but mis-rendered — so a schema-level regression test is
the cheapest tripwire.

Bug classes guarded (see LEARNINGS.md "reverse mmproj interop"):
  1. Identity tensor naming. `map_tensor_name` must keep the native
     llama.cpp names (`v.blk.*`, `blk.*`, `token_embd`, `mm.*`). An
     earlier version remapped to `vis.blocks.*`/`llm.layers.*`, which the
     qwen2vl_ocr loader does NOT read -> vision misdetected -> SIGSEGV.
  2. Vision special-token injection. llama.cpp GGUFs carry no
     `qwen2vl.image_token_id`/`vision_start`/`vision_end`. The splice needs
     them to find the <|image_pad|> positions; missing -> image silently
     dropped ("text not visible"). The merge script must inject the fixed
     Qwen2/2.5-VL defaults. `vision_start/end` default to 0 in the loader
     header, so writing them is load-bearing (not just belt-and-suspenders).
  3. Split temporal patch embedding. llama.cpp stores the Conv3d patch as
     two [out,in,H,W] slices (`v.patch_embd.weight` + `.weight.1`); the
     loader expects one weight flattening to [in*T*H*W, out]. The merge
     must concatenate and drop the `.weight.1` slice.
  4. Export inverse maps. The export script's C->L maps must be the exact
     inverse of the merge L->C maps (validated by its --self-test) and its
     real do_export() path must run on a merged GGUF.

Fully self-contained: synthesizes tiny llama.cpp-shaped LLM + mmproj GGUFs
in a temp dir (no model download, no inference, no C++ build). Run:

    ~/miniconda3/bin/python tests/test_mmproj_interop.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")
MERGE = os.path.join(MODELS, "merge-llamacpp-qwen2vl-gguf.py")
EXPORT = os.path.join(MODELS, "export-mmproj-llamacpp.py")

# ── Tiny model dimensions (shapes must be self-consistent; values arbitrary) ──
VIS_HIDDEN = 8
VIS_HEADS = 2
VIS_INTER = 16
VIS_BLOCKS = 2
VIS_PATCH = 14           # spatial patch H=W
VIS_IN = 3               # in_channels
VIS_TEMPORAL = 2         # temporal_patch_size
PROJ_DIM = 12            # merger out_hidden
LLM_HIDDEN = 8
LLM_INTER = 16
LLM_LAYERS = 2
LLM_HEADS = 2
VOCAB = 40

F16 = GGMLQuantizationType.F16
F32 = GGMLQuantizationType.F32
_NP = {F16: np.float16, F32: np.float32}


def _f16(*shape):
    # Deterministic, order-sensitive fill so a mis-concatenation is visible.
    n = int(np.prod(shape))
    return (np.arange(n, dtype=np.float32).reshape(shape) * 0.001).astype(np.float16)


def _patch(qt, *shape):
    # The temporal patch slices, in the tensor's real dtype (F16 or F32) — a
    # distinct ramp per slice so a wrong-width read/concat is visible.
    n = int(np.prod(shape))
    return (np.arange(n, dtype=np.float32).reshape(shape) * 0.0007 + 0.1).astype(_NP[qt])


def build_mmproj(path, patch_qt=F16):
    """A minimal llama.cpp Qwen2-VL mmproj GGUF (arch=clip)."""
    w = GGUFWriter(path, "clip")
    w.add_type("clip-vision")
    w.add_name("tiny-qwen2vl")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen2vl_merger")
    w.add_uint32("clip.vision.projection_dim", PROJ_DIM)
    w.add_uint32("clip.vision.image_size", VIS_PATCH * VIS_TEMPORAL)
    w.add_uint32("clip.vision.patch_size", VIS_PATCH)
    w.add_uint32("clip.vision.embedding_length", VIS_HIDDEN)
    w.add_uint32("clip.vision.feed_forward_length", VIS_INTER)
    w.add_uint32("clip.vision.block_count", VIS_BLOCKS)
    w.add_uint32("clip.vision.attention.head_count", VIS_HEADS)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.image_mean",
                [0.48145467, 0.45782751, 0.40821072])
    w.add_array("clip.vision.image_std",
                [0.26862955, 0.26130259, 0.27577710])
    w.add_file_type(1)  # F16

    # Split temporal patch: two [out,in,H,W] slices (numpy order). The patch
    # dtype is parameterized (F16 vs F32) to guard the merge/export dtype width.
    w.add_tensor("v.patch_embd.weight",
                 _patch(patch_qt, VIS_HIDDEN, VIS_IN, VIS_PATCH, VIS_PATCH), raw_dtype=patch_qt)
    w.add_tensor("v.patch_embd.weight.1",
                 _patch(patch_qt, VIS_HIDDEN, VIS_IN, VIS_PATCH, VIS_PATCH) + _NP[patch_qt](1.0),
                 raw_dtype=patch_qt)
    w.add_tensor("v.post_ln.weight", _f16(VIS_HIDDEN), raw_dtype=F16)
    w.add_tensor("v.post_ln.bias", _f16(VIS_HIDDEN), raw_dtype=F16)
    # Projector (qwen2vl_merger): mm.0 (in->inter), mm.2 (inter->out).
    merge_in = VIS_HIDDEN * 4  # spatial_merge_size**2 * hidden
    w.add_tensor("mm.0.weight", _f16(merge_in, merge_in), raw_dtype=F16)
    w.add_tensor("mm.0.bias", _f16(merge_in), raw_dtype=F16)
    w.add_tensor("mm.2.weight", _f16(PROJ_DIM, merge_in), raw_dtype=F16)
    w.add_tensor("mm.2.bias", _f16(PROJ_DIM), raw_dtype=F16)
    for b in range(VIS_BLOCKS):
        p = f"v.blk.{b}."
        for nm in ("attn_q", "attn_k", "attn_v", "attn_out"):
            w.add_tensor(p + nm + ".weight", _f16(VIS_HIDDEN, VIS_HIDDEN), raw_dtype=F16)
            w.add_tensor(p + nm + ".bias", _f16(VIS_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ln1.weight", _f16(VIS_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ln1.bias", _f16(VIS_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ln2.weight", _f16(VIS_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ln2.bias", _f16(VIS_HIDDEN), raw_dtype=F16)
        # ffn_up = fc1 (hidden->inter), ffn_down = fc2 (inter->hidden).
        w.add_tensor(p + "ffn_up.weight", _f16(VIS_INTER, VIS_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ffn_up.bias", _f16(VIS_INTER), raw_dtype=F16)
        w.add_tensor(p + "ffn_down.weight", _f16(VIS_HIDDEN, VIS_INTER), raw_dtype=F16)
        w.add_tensor(p + "ffn_down.bias", _f16(VIS_HIDDEN), raw_dtype=F16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def build_llm(path):
    """A minimal llama.cpp Qwen2-VL LLM GGUF, deliberately WITHOUT any
    vision special-token metadata (that's what the merge must inject)."""
    w = GGUFWriter(path, "qwen2vl")
    w.add_name("tiny-qwen2vl")
    w.add_uint32("qwen2vl.block_count", LLM_LAYERS)
    w.add_uint32("qwen2vl.embedding_length", LLM_HIDDEN)
    w.add_uint32("qwen2vl.feed_forward_length", LLM_INTER)
    w.add_uint32("qwen2vl.attention.head_count", LLM_HEADS)
    w.add_uint32("qwen2vl.attention.head_count_kv", LLM_HEADS)
    w.add_float32("qwen2vl.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("qwen2vl.rope.freq_base", 1e6)
    w.add_uint32("qwen2vl.context_length", 128)
    w.add_array("qwen2vl.rope.dimension_sections", [2, 1, 1, 0])
    w.add_file_type(1)

    w.add_tensor("token_embd.weight", _f16(VOCAB, LLM_HIDDEN), raw_dtype=F16)
    w.add_tensor("output_norm.weight", _f16(LLM_HIDDEN), raw_dtype=F16)
    w.add_tensor("output.weight", _f16(VOCAB, LLM_HIDDEN), raw_dtype=F16)
    for b in range(LLM_LAYERS):
        p = f"blk.{b}."
        for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
            w.add_tensor(p + nm + ".weight", _f16(LLM_HIDDEN, LLM_HIDDEN), raw_dtype=F16)
        for nm in ("attn_q", "attn_k", "attn_v"):
            w.add_tensor(p + nm + ".bias", _f16(LLM_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "attn_norm.weight", _f16(LLM_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ffn_gate.weight", _f16(LLM_INTER, LLM_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ffn_up.weight", _f16(LLM_INTER, LLM_HIDDEN), raw_dtype=F16)
        w.add_tensor(p + "ffn_down.weight", _f16(LLM_HIDDEN, LLM_INTER), raw_dtype=F16)
        w.add_tensor(p + "ffn_norm.weight", _f16(LLM_HIDDEN), raw_dtype=F16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _run(argv):
    r = subprocess.run([sys.executable] + argv, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


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


def _read(path):
    r = GGUFReader(path)
    md = {f.name: _field_contents(f) for f in r.fields.values()}
    tensors = {t.name: t for t in r.tensors}
    return md, tensors


def main():
    fails = []

    def check(cond, msg):
        print(("  ok  " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # Run the whole merge<->export scenario for each patch dtype so the
    # dtype-width handling (F32 patch embed, not just F16) is guarded.
    for patch_qt in (F16, F32):
        print("\n" + "=" * 60)
        print(f"SCENARIO: patch_embd dtype = {patch_qt.name}")
        print("=" * 60)
        _scenario(patch_qt, check)

    print("\n" + ("ALL PASS" if not fails else f"FAILED ({len(fails)}): " + "; ".join(fails)))
    return 0 if not fails else 1


def _scenario(patch_qt, check):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        llm = os.path.join(d, "llm.gguf")
        mmproj = os.path.join(d, "mmproj.gguf")
        merged = os.path.join(d, "crisp.gguf")
        mmproj2 = os.path.join(d, "mmproj2.gguf")
        build_llm(llm)
        build_mmproj(mmproj, patch_qt=patch_qt)

        # ── 1. Merge llama.cpp split GGUFs -> CrispEmbed combined GGUF ──
        print("\n[merge] llama.cpp LLM + mmproj -> CrispEmbed combined")
        rc, out = _run([MERGE, "--llm", llm, "--mmproj", mmproj, "--output", merged])
        check(rc == 0, f"merge exits 0 (rc={rc})")
        if rc != 0:
            print(out)
            return
        md, tensors = _read(merged)
        names = set(tensors)

        # Bug 1: identity naming — native names kept, legacy remap absent.
        check("v.blk.0.attn_q.weight" in names, "native vision name v.blk.0.attn_q.weight kept")
        check("blk.0.attn_q.weight" in names, "native LLM name blk.0.attn_q.weight kept")
        check("token_embd.weight" in names, "native token_embd.weight kept")
        check("mm.0.weight" in names, "native projector mm.0.weight kept")
        check("v.post_ln.weight" in names, "native v.post_ln.weight kept")
        legacy = [n for n in names if n.startswith("vis.blocks.") or n.startswith("llm.layers.")]
        check(not legacy, f"no legacy vis.blocks.*/llm.layers.* names (found {legacy[:2]})")
        check(md.get("general.architecture") == "qwen2vl", "general.architecture == qwen2vl")

        # Bug 2: vision special-token injection (absent in source LLM).
        check(int(md.get("qwen2vl.image_token_id", -1)) == 151655,
              "qwen2vl.image_token_id injected == 151655")
        check(int(md.get("qwen2vl.vision_start_token_id", -1)) == 151652,
              "qwen2vl.vision_start_token_id injected == 151652")
        check(int(md.get("qwen2vl.vision_end_token_id", -1)) == 151653,
              "qwen2vl.vision_end_token_id injected == 151653")

        # Bug 3: split temporal patch concatenated into one tensor.
        check("v.patch_embd.weight" in names, "v.patch_embd.weight present")
        check("v.patch_embd.weight.1" not in names, "v.patch_embd.weight.1 dropped (folded)")
        pe = tensors.get("v.patch_embd.weight")
        if pe is not None:
            # ggml ne = [in*T*H*W, out]; GGUFReader.shape is numpy order (out, in*T*H*W).
            expect_in = VIS_IN * VIS_TEMPORAL * VIS_PATCH * VIS_PATCH
            got = list(pe.shape)
            check(sorted(got) == sorted([expect_in, VIS_HIDDEN]),
                  f"patch_embd concatenated shape {got} == [{VIS_HIDDEN}, {expect_in}] (some order)")

        # tie_word_embeddings must be False (output.weight present).
        check(md.get("qwen2vl.tie_word_embeddings") in (False, 0),
              "tie_word_embeddings == False (output.weight present)")

        # ── 2. Export inverse-map self-test (no reference download) ──
        print("\n[export --self-test] inverse maps round-trip vs the fixture mmproj")
        rc, out = _run([EXPORT, "--self-test", mmproj])
        check(rc == 0 and "SELF-TEST PASS" in out, f"export --self-test PASS (rc={rc})")
        if "SELF-TEST PASS" not in out:
            print(out)

        # ── 3. Real do_export() path on the merged combined GGUF ──
        print("\n[export] combined CrispEmbed GGUF -> mmproj (real production path)")
        rc, out = _run([EXPORT, "--in", merged, "--out", mmproj2])
        check(rc == 0, f"export --in/--out exits 0 (rc={rc})")
        if rc == 0:
            emd, etensors = _read(mmproj2)
            check(emd.get("clip.projector_type") == "qwen2vl_merger",
                  "exported clip.projector_type == qwen2vl_merger")
            check(int(emd.get("clip.vision.block_count", -1)) == VIS_BLOCKS,
                  f"exported clip.vision.block_count == {VIS_BLOCKS}")
            check("v.blk.0.attn_q.weight" in etensors,
                  "exported mmproj has native v.blk.0.attn_q.weight")
            # Full vision-tower round-trip: EVERY vision tensor of the original
            # mmproj (both split patch slices included) must reappear in mmproj2
            # byte-identically — merge and export are exact inverses on vision.
            _, otensors = _read(mmproj)
            ov = {n: t for n, t in otensors.items()
                  if n.startswith("v.") or n.startswith("mm.")}
            missing = [n for n in ov if n not in etensors]
            check(not missing, f"round-trip preserves all vision tensor names (missing {missing[:3]})")
            diffs = [n for n in ov if n in etensors
                     and not np.array_equal(np.array(etensors[n].data), np.array(ov[n].data))]
            check(not diffs, f"round-trip byte-identical for all {len(ov)} vision tensors (differ: {diffs[:3]})")
        else:
            print(out)


if __name__ == "__main__":
    sys.exit(main())
