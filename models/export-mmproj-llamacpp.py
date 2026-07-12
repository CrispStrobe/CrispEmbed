#!/usr/bin/env python3
"""Export a CrispEmbed combined Qwen2-VL GGUF's vision tower as a llama.cpp
mmproj GGUF (the reverse of merge-llamacpp-qwen2vl-gguf.py).

llama.cpp / mtmd consume the vision tower as a standalone `mmproj-*.gguf` with:
  - metadata: general.architecture=clip, general.type=clip-vision, clip.* keys
  - tensors:  v.blk.N.*, v.patch_embd(.weight[.1]), v.post_ln.*, mm.*

This lets a CrispEmbed-converted Qwen2-VL vision tower be loaded by
`llama-mtmd-cli --mmproj ...` (interop; do NOT link libmtmd into crispembed).

Naming: the merge script keeps llama.cpp-native tensor names (`v.blk.*`,
`v.patch_embd`, `v.post_ln`, `mm.*`) verbatim in the combined GGUF — those
are exactly the names the qwen2vl_ocr loader reads. So the vision tensor
names are IDENTITY between a combined CrispEmbed GGUF and the mmproj; the
only structural transform is splitting the temporal patch embedding back
into its two slices (the inverse of the merge's concatenation) and lifting
`qwen2vl.vision.*` config into `clip.vision.*`.

    python export-mmproj-llamacpp.py --in crispembed-qwen2vl.gguf --out mmproj.gguf
    python export-mmproj-llamacpp.py --self-test ref-mmproj.gguf
"""
import argparse
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

# Qwen2-VL image normalization is the fixed OpenAI-CLIP statistic (matches the
# reference mmproj exactly); CrispEmbed combined GGUFs don't store it.
QWEN2VL_IMAGE_MEAN = [0.48145467042922974, 0.45782750844955444, 0.40821072459220886]
QWEN2VL_IMAGE_STD = [0.2686295509338379, 0.2613025903701782, 0.27577710151672363]

# GGML types the temporal patch can be split in-place. GGUFReader decodes
# these to a real float ndarray we can reshape/slice losslessly; BF16 and the
# quantized types decode ambiguously, so we keep those combined (see below).
_SPLITTABLE = {GGMLQuantizationType.F16, GGMLQuantizationType.F32}

# The combined patch's llama.cpp name and its split second slice.
PATCH_NAME = "v.patch_embd.weight"
PATCH_NAME_1 = "v.patch_embd.weight.1"


def is_vision_tensor(name):
    """A tensor belongs to the vision tower / projector (not the LLM)."""
    return name.startswith("v.") or name.startswith("mm.")


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


def split_temporal_patch(combined, in_ch, temporal, patch):
    """Inverse of the merge concatenation. `combined` is the numpy patch in
    (out, in*T*H*W) order (GGUFReader data order); returns a list of `temporal`
    slices each shaped (out, in, H, W) — the llama.cpp split form. Returns None
    if the element count doesn't match (caller keeps the combined tensor)."""
    out = combined.shape[0]
    if combined.size != out * in_ch * temporal * patch * patch:
        return None
    resh = combined.reshape(out, in_ch, temporal, patch, patch)
    return [np.ascontiguousarray(resh[:, :, t]) for t in range(temporal)]


def read_crisp_vision(reader):
    """Extract vision tensors + config from a CrispEmbed combined GGUF.

    Vision tensor names are kept identity (they already are llama.cpp-native);
    the concatenated temporal patch is split back into two slices."""
    md = {f.name: f.contents() for f in reader.fields.values()}
    patch = int(_mv(md, "qwen2vl.vision.patch_size",
                    "qwen2vl.vision.spatial_patch_size", default=14))
    in_ch = int(_mv(md, "qwen2vl.vision.in_channels", default=3))
    temporal = int(_mv(md, "qwen2vl.vision.temporal_patch_size", default=2))

    tensors = []
    hidden = None
    ffn_up = None
    proj_out = None
    for t in reader.tensors:
        if not is_vision_tensor(t.name):
            continue
        qt = t.tensor_type
        if t.name == PATCH_NAME:
            data = np.array(t.data)
            slices = None
            if qt in _SPLITTABLE:
                slices = split_temporal_patch(data, in_ch, temporal, patch)
            if slices is None:
                # Unsplittable (BF16/quantized or shape mismatch): keep combined.
                if qt not in _SPLITTABLE:
                    print(f"  note: patch dtype {qt.name} not splittable — keeping combined")
                tensors.append((PATCH_NAME, data, qt))
            else:
                # Preserve GGUFReader's decoded dtype (F16->float16, F32->float32).
                tensors.append((PATCH_NAME, slices[0].astype(data.dtype), qt))
                for i, s in enumerate(slices[1:], start=1):
                    tensors.append((f"{PATCH_NAME}.{i}", s.astype(data.dtype), qt))
            continue
        if t.name == PATCH_NAME_1:
            continue  # already-split slice (a combined GGUF shouldn't have it)
        tensors.append((t.name, np.array(t.data), qt))
        if t.name == "v.blk.0.ln1.weight":
            hidden = int(t.shape[-1])       # ne last dim = vector length
        if t.name == "v.blk.0.ffn_up.weight":
            ffn_up = int(t.shape[-1])       # ne = [in, out] -> out
        if t.name == "mm.2.weight":
            proj_out = int(t.shape[-1])     # ne = [in, out] -> out

    layers = int(_mv(md, "qwen2vl.vision.num_hidden_layers", "qwen2vl.vision.depth"))
    heads = int(_mv(md, "qwen2vl.vision.num_attention_heads", "qwen2vl.vision.num_heads"))
    vis = {
        "block_count": layers,
        "head_count": heads,
        "patch_size": patch,
        "embedding_length": int(_mv(md, "qwen2vl.vision.hidden_size", default=0)) or hidden or 1280,
        "feed_forward_length": int(_mv(md, "qwen2vl.vision.intermediate_size", default=0)) or ffn_up,
        "projection_dim": int(_mv(md, "qwen2vl.vision.out_hidden_size",
                                  "qwen2vl.hidden_size", default=0)) or proj_out or 1536,
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
        sys.exit("error: no vision tensors (v.*/mm.*) found — not a CrispEmbed Qwen2-VL GGUF?")
    if vis["feed_forward_length"] is None:
        for n, a, _ in tensors:
            if n.endswith("ffn_up.weight"):
                vis["feed_forward_length"] = int(a.shape[0]); break
    write_mmproj(out_path, tensors, name, vis)
    n_blk = sum(1 for n, _, _ in tensors if n.startswith("v.blk."))
    print(f"wrote {out_path}: {len(tensors)} vision tensors, {vis['block_count']} blocks "
          f"({n_blk} block tensors)")


def do_self_test(ref_path):
    """Validate the risky transform (temporal patch concat<->split) and the
    identity naming against a reference llama.cpp mmproj — no VL-model download
    or inference needed.

    Steps: read ref mmproj (which has the split patch weight + weight.1);
    concat the two slices exactly as the merge script would; split the combined
    back with export's logic; assert the recovered slices are byte-identical to
    the originals and that every vision tensor name is recognized."""
    ref = GGUFReader(ref_path)
    rmd = {f.name: f.contents() for f in ref.fields.values()}
    patch = int(rmd.get("clip.vision.patch_size", 14))
    in_ch = 3
    temporal = 2

    by_name = {t.name: t for t in ref.tensors}
    ok = True

    # 1) Every ref tensor is classified as vision (an mmproj is vision-only).
    unclassified = [t.name for t in ref.tensors if not is_vision_tensor(t.name)]
    if unclassified:
        print(f"FAIL unclassified vision tensors: {unclassified[:3]}")
        ok = False
    else:
        print(f"tensor classification: all {len(by_name)} mmproj tensors recognized as vision")

    # 2) Temporal patch concat -> split round-trip is byte-identical.
    p0 = by_name.get(PATCH_NAME)
    p1 = by_name.get(PATCH_NAME_1)
    if p0 is None or p1 is None:
        print(f"WARN reference has no split patch ({PATCH_NAME}[.1]); skipping patch round-trip")
    else:
        a0 = np.array(p0.data)  # (out, in, H, W)
        a1 = np.array(p1.data)
        # merge concat: stack on temporal axis -> (out, in, T, H, W) -> (out, in*T*H*W)
        combined = np.stack([a0, a1], axis=2).reshape(a0.shape[0], -1)
        slices = split_temporal_patch(combined, in_ch, temporal, patch)
        if slices is None:
            print("FAIL patch split returned None (shape mismatch)")
            ok = False
        elif not (np.array_equal(slices[0], a0) and np.array_equal(slices[1], a1)):
            print("FAIL patch concat->split not identity")
            ok = False
        else:
            print(f"temporal patch: concat->split byte-identical ({a0.shape} x{temporal})")

    # 3) clip.* config keys the export re-emits are all readable from the ref.
    needed = ["clip.vision.block_count", "clip.vision.attention.head_count",
              "clip.vision.patch_size", "clip.vision.embedding_length",
              "clip.vision.projection_dim", "clip.vision.image_size"]
    missing = [k for k in needed if k not in rmd]
    if missing:
        print(f"FAIL reference missing clip.* keys: {missing}")
        ok = False
    else:
        print("metadata schema: all required clip.* keys present in reference")

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
