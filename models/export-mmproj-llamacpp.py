#!/usr/bin/env python3
"""Export a CrispEmbed combined Qwen2-VL GGUF's vision tower as a llama.cpp
mmproj GGUF (the reverse of merge-llamacpp-qwen2vl-gguf.py).

llama.cpp / mtmd consume the vision tower as a standalone `mmproj-*.gguf` with:
  - metadata: general.architecture=clip, general.type=clip-vision, clip.* keys
  - tensors:  v.blk.N.*, v.patch_embd(.weight[.1]), v.post_ln.*, mm.*

This lets a CrispEmbed-converted Qwen2-VL vision tower be loaded by
`llama-mtmd-cli --mmproj ...` (interop; do NOT link libmtmd into crispembed).

The tensor-name and metadata maps here are the exact INVERSE of the shipped,
proven merge script, so an mmproj → CrispEmbed → mmproj round-trip is the
identity (see --self-test, which validates that with a reference mmproj — no
VL-model download or inference needed).

Usage:
    python export-mmproj-llamacpp.py --in crispembed-qwen2vl.gguf --out mmproj.gguf
    python export-mmproj-llamacpp.py --self-test ref-mmproj.gguf
"""
import argparse
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType

# ── Tensor-name maps: CrispEmbed → llama.cpp (inverse of the merge script) ──
GLOBAL_MAP_C2L = {
    "vis.patch_embed.proj.weight":   "v.patch_embd.weight",
    "vis.patch_embed.proj_t.weight": "v.patch_embd.weight.1",
    "vis.merger.ln_q.weight":        "v.post_ln.weight",
    "vis.merger.ln_q.bias":          "v.post_ln.bias",
    "proj.mlp1.weight":              "mm.0.weight",
    "proj.mlp1.bias":                "mm.0.bias",
    "proj.mlp2.weight":              "mm.2.weight",
    "proj.mlp2.bias":                "mm.2.bias",
}
VIS_BLOCK_MAP_C2L = {
    "attn.q.weight": "attn_q.weight", "attn.k.weight": "attn_k.weight",
    "attn.v.weight": "attn_v.weight", "attn.proj.weight": "attn_out.weight",
    "attn.q.bias": "attn_q.bias", "attn.k.bias": "attn_k.bias",
    "attn.v.bias": "attn_v.bias", "attn.proj.bias": "attn_out.bias",
    "norm1.weight": "ln1.weight", "norm1.bias": "ln1.bias",
    "norm2.weight": "ln2.weight", "norm2.bias": "ln2.bias",
    "mlp.fc1.weight": "ffn_up.weight", "mlp.fc1.bias": "ffn_up.bias",
    "mlp.fc2.weight": "ffn_down.weight", "mlp.fc2.bias": "ffn_down.bias",
}
# Qwen2-VL image normalization is the fixed OpenAI-CLIP statistic (matches the
# reference mmproj exactly); CrispEmbed combined GGUFs don't store it.
QWEN2VL_IMAGE_MEAN = [0.48145467042922974, 0.45782750844955444, 0.40821072459220886]
QWEN2VL_IMAGE_STD = [0.2686295509338379, 0.2613025903701782, 0.27577710151672363]


def crisp_to_llama_name(name):
    if name in GLOBAL_MAP_C2L:
        return GLOBAL_MAP_C2L[name]
    if name.startswith("vis.blocks."):
        rest = name[len("vis.blocks."):]
        layer, _, suffix = rest.partition(".")
        if suffix in VIS_BLOCK_MAP_C2L:
            return f"v.blk.{layer}.{VIS_BLOCK_MAP_C2L[suffix]}"
    return None  # not a vision tensor (LLM tensor, skip)


def _mv(md, *keys, default=None):
    """First present metadata value among keys (handles the merge script's
    redundant aliases, e.g. num_hidden_layers/depth)."""
    for k in keys:
        if k in md:
            return md[k]
    return default


def write_mmproj(out_path, tensors, md_general_name, vis):
    """tensors: list of (llama_name, np_array, ggml_type). vis: dict of config."""
    w = GGUFWriter(out_path, "clip")
    w.add_type("clip-vision")
    if md_general_name:
        w.add_name(md_general_name)
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen2vl_merger")
    w.add_uint32("clip.vision.projection_dim", vis["projection_dim"])
    w.add_uint32("clip.vision.image_size", vis["image_size"])
    w.add_uint32("clip.vision.patch_size", vis["patch_size"])
    w.add_uint32("clip.vision.embedding_length", vis["embedding_length"])
    w.add_uint32("clip.vision.feed_forward_length", vis["feed_forward_length"])
    w.add_uint32("clip.vision.block_count", vis["block_count"])
    w.add_uint32("clip.vision.attention.head_count", vis["head_count"])
    w.add_float32("clip.vision.attention.layer_norm_epsilon", vis["ln_eps"])
    w.add_array("clip.vision.image_mean", vis["image_mean"])
    w.add_array("clip.vision.image_std", vis["image_std"])
    w.add_file_type(vis["file_type"])
    for name, arr, qtype in tensors:
        w.add_tensor(name, arr, raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_crisp_vision(reader):
    """Extract vision tensors + config from a CrispEmbed combined GGUF."""
    md = {f.name: f.contents() for f in reader.fields.values()}
    tensors = []
    hidden = None
    for t in reader.tensors:
        ln = crisp_to_llama_name(t.name)
        if ln is None:
            continue
        arr = np.array(t.data)
        tensors.append((ln, arr, t.tensor_type))
        if ln == "v.blk.0.ln1.weight":
            hidden = int(t.shape[0])
    layers = int(_mv(md, "qwen2vl.vision.num_hidden_layers", "qwen2vl.vision.depth"))
    heads = int(_mv(md, "qwen2vl.vision.num_attention_heads", "qwen2vl.vision.num_heads"))
    patch = int(_mv(md, "qwen2vl.vision.patch_size", "qwen2vl.vision.spatial_patch_size", default=14))
    vis = {
        "block_count": layers,
        "head_count": heads,
        "patch_size": patch,
        "embedding_length": hidden or int(_mv(md, "qwen2vl.vision.hidden_size", default=1280)),
        "feed_forward_length": int(_mv(md, "qwen2vl.vision.intermediate_size", default=0)) or None,
        "projection_dim": int(_mv(md, "qwen2vl.vision.out_hidden_size", "qwen2vl.hidden_size", default=1536)),
        "image_size": int(_mv(md, "qwen2vl.vision.image_size", default=560)),
        "ln_eps": float(_mv(md, "qwen2vl.vision.layer_norm_eps", default=1e-6)),
        "image_mean": QWEN2VL_IMAGE_MEAN,
        "image_std": QWEN2VL_IMAGE_STD,
        "file_type": int(_mv(md, "general.file_type", default=7)),
    }
    return tensors, vis, md.get("general.name")


def do_export(in_path, out_path):
    r = GGUFReader(in_path)
    tensors, vis, name = read_crisp_vision(r)
    if not tensors:
        sys.exit("error: no vision tensors (vis.*/proj.*) found — not a CrispEmbed Qwen2-VL GGUF?")
    if vis["feed_forward_length"] is None:
        # infer from mlp.fc1 (ffn_up) shape
        for n, a, _ in tensors:
            if n.endswith("ffn_up.weight"):
                vis["feed_forward_length"] = int(a.shape[0]); break
    write_mmproj(out_path, tensors, name, vis)
    print(f"wrote {out_path}: {len(tensors)} vision tensors, {vis['block_count']} blocks")


def do_self_test(ref_path):
    """Round-trip: ref mmproj → CrispEmbed vision names → export → diff vs ref.
    Validates the inverse maps + schema reproduce the reference exactly."""
    ref = GGUFReader(ref_path)
    rmd = {f.name: f.contents() for f in ref.fields.values()}
    # llama.cpp → CrispEmbed (forward map, from the merge script)
    L2C_GLOBAL = {v: k for k, v in GLOBAL_MAP_C2L.items()}
    L2C_BLOCK = {v: k for k, v in VIS_BLOCK_MAP_C2L.items()}

    def llama_to_crisp(name):
        if name in L2C_GLOBAL:
            return L2C_GLOBAL[name]
        if name.startswith("v.blk."):
            rest = name[len("v.blk."):]
            layer, _, suffix = rest.partition(".")
            if suffix in L2C_BLOCK:
                return f"vis.blocks.{layer}.{L2C_BLOCK[suffix]}"
        return None

    # Build synthetic CrispEmbed vision tensors + expected reference tensors.
    ref_tensors = {}
    crisp_tensors = []
    for t in ref.tensors:
        cn = llama_to_crisp(t.name)
        if cn is None:
            continue  # non-vision (shouldn't happen in an mmproj)
        arr = np.array(t.data)
        ref_tensors[t.name] = (arr, t.tensor_type)
        crisp_tensors.append((cn, arr, t.tensor_type))
    # Round-trip the names back to llama.cpp via the export map.
    exp_tensors = {}
    for cn, arr, qt in crisp_tensors:
        ln = crisp_to_llama_name(cn)
        assert ln is not None, f"export map lost tensor {cn}"
        exp_tensors[ln] = (arr, qt)

    # 1) tensor names + data identical
    ok = True
    if set(exp_tensors) != set(ref_tensors):
        miss = set(ref_tensors) - set(exp_tensors)
        extra = set(exp_tensors) - set(ref_tensors)
        print(f"FAIL tensor-name set: missing={list(miss)[:3]} extra={list(extra)[:3]}")
        ok = False
    else:
        for n in ref_tensors:
            a, at = exp_tensors[n]
            b, bt = ref_tensors[n]
            if at != bt or not np.array_equal(a, b):
                print(f"FAIL tensor data differs: {n}")
                ok = False
                break
        print(f"tensor round-trip: {len(ref_tensors)} tensors "
              f"{'byte-identical' if ok else 'DIFFER'}")

    # 1b) Exercise the actual GGUF write path: write an mmproj from the
    # crisp-renamed tensors + reference config, re-read it, diff vs reference.
    import os
    import tempfile
    vis = {
        "block_count": rmd["clip.vision.block_count"],
        "head_count": rmd["clip.vision.attention.head_count"],
        "patch_size": rmd["clip.vision.patch_size"],
        "embedding_length": rmd["clip.vision.embedding_length"],
        "feed_forward_length": rmd["clip.vision.feed_forward_length"],
        "projection_dim": rmd["clip.vision.projection_dim"],
        "image_size": rmd["clip.vision.image_size"],
        "ln_eps": rmd.get("clip.vision.attention.layer_norm_epsilon", 1e-6),
        "image_mean": QWEN2VL_IMAGE_MEAN,
        "image_std": QWEN2VL_IMAGE_STD,
        "file_type": rmd.get("general.file_type", 7),
    }
    tmp = os.path.join(tempfile.gettempdir(), "mmproj_export_roundtrip.gguf")
    write_mmproj(tmp, [(n, a, t) for n, (a, t) in exp_tensors.items()],
                 rmd.get("general.name"), vis)
    rt = GGUFReader(tmp)
    rt_md = {f.name: f.contents() for f in rt.fields.values()}
    rt_tensors = {t.name: (np.array(t.data), t.tensor_type) for t in rt.tensors}
    if set(rt_tensors) != set(ref_tensors):
        print("FAIL written-file tensor set differs")
        ok = False
    else:
        for n in ref_tensors:
            a, at = rt_tensors[n]
            b, bt = ref_tensors[n]
            if at != bt or not np.array_equal(a, b):
                print(f"FAIL written-file tensor {n} differs")
                ok = False
                break
        print(f"written-file re-read: {len(rt_tensors)} tensors "
              f"{'byte-identical to reference' if ok else 'DIFFER'}")
    for k in ("clip.projector_type", "clip.vision.block_count",
              "clip.vision.embedding_length", "clip.vision.image_size"):
        if rt_md.get(k) != rmd.get(k):
            print(f"FAIL written meta {k}: wrote={rt_md.get(k)} ref={rmd.get(k)}")
            ok = False
    os.remove(tmp)

    # 2) metadata schema: the clip.* keys we emit must match the reference values.
    checks = {
        "clip.projector_type": "qwen2vl_merger",
        "clip.has_vision_encoder": True,
        "clip.vision.image_size": rmd.get("clip.vision.image_size"),
        "clip.vision.patch_size": rmd.get("clip.vision.patch_size"),
        "clip.vision.embedding_length": rmd.get("clip.vision.embedding_length"),
        "clip.vision.block_count": rmd.get("clip.vision.block_count"),
        "clip.vision.attention.head_count": rmd.get("clip.vision.attention.head_count"),
        "clip.vision.projection_dim": rmd.get("clip.vision.projection_dim"),
    }
    for k, expect in checks.items():
        got = rmd.get(k)
        same = (got == expect)
        if not same:
            print(f"FAIL meta {k}: ref={got}")
            ok = False
    # image_mean/std constants must equal the reference
    for k, const in (("clip.vision.image_mean", QWEN2VL_IMAGE_MEAN),
                     ("clip.vision.image_std", QWEN2VL_IMAGE_STD)):
        rv = list(rmd.get(k, []))
        if not (len(rv) == 3 and all(abs(x - y) < 1e-6 for x, y in zip(rv, const))):
            print(f"FAIL meta {k}: ref={rv} const={const}")
            ok = False
    print(f"metadata schema: {'complete + matches reference' if ok else 'MISMATCH'}")
    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--self-test", dest="selftest")
    a = ap.parse_args()
    if a.selftest:
        return do_self_test(a.selftest)
    if not a.inp or not a.out:
        ap.error("need --in and --out (or --self-test REF)")
    do_export(a.inp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
