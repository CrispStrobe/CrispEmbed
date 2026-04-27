#!/usr/bin/env python3
"""BidirLM-Omni vision-tower parity test.

Compares CrispEmbed's vision tower against the HF reference. The HF model
is loaded as a stand-alone BidirLMOmniVisionModel (no decoder, no audio)
so weight loading is fast.

    python tests/test_bidirlm_vision.py \
        --model BidirLM/BidirLM-Omni-2.5B-Embedding \
        --gguf  $CRISPEMBED_CACHE_DIR/bidirlm-omni-2.5b-f16.gguf \
        --image $CRISPEMBED_BIDIRLM_IMAGE

Reports two numbers:
  * `image_embeds`        — final-merger output, per-token cosine averaged.
  * `deepstack[k]`        — same, per deepstack hook.

Pass criterion: per-token cosine ≥ 0.99 on each. Raw mean L1 difference is
also printed for diagnostics.
"""

import argparse
import sys
import numpy as np


def hf_vision_features(model_id: str, image):
    """Run BidirLMOmniVisionModel on one image. Returns (image_embeds, [deepstack...]).

    Both arrays are np.float32, shape (n_merged, out_hidden_size).
    """
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    vision_cls = get_class_from_dynamic_module(
        "modeling_bidirlm_omni.BidirLMOmniVisionModel", model_id,
    )
    vision = vision_cls(config.vision_config)
    vision.eval()

    # Load only visual.* weights.
    sd = {}
    try:
        st_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        raw = load_file(st_path)
    except Exception:
        idx_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        shards = {v for v in idx.values()}
        raw = {}
        for shard in shards:
            sp = hf_hub_download(repo_id=model_id, filename=shard)
            raw.update(load_file(sp))
    for k, v in raw.items():
        if k.startswith("visual."):
            sd[k[len("visual."):]] = v
    missing, unexpected = vision.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN: {len(missing)} missing visual tensors, e.g. {missing[:3]}")

    # Use the same preprocessor CrispEmbed will use.
    sys.path.insert(0, "python")
    from crispembed.image import preprocess_image
    pixel_values_np, grid_thw_np = preprocess_image(image, model_name=model_id)
    pixel_values = torch.from_numpy(pixel_values_np).to(torch.float32)
    grid_thw = torch.from_numpy(grid_thw_np).to(torch.long)

    with torch.no_grad():
        image_embeds, deepstack = vision(pixel_values, grid_thw=grid_thw)
    img = image_embeds.float().cpu().numpy()
    ds = [d.float().cpu().numpy() for d in deepstack]
    return img, ds, pixel_values_np, grid_thw_np


def crispembed_vision_features(gguf: str, pixel_values, grid_thw, lib_path=None):
    sys.path.insert(0, "python")
    from crispembed._binding import CrispEmbed

    ce = CrispEmbed(gguf, lib_path=lib_path) if lib_path else CrispEmbed(gguf)
    if not ce.has_vision:
        raise RuntimeError("CrispEmbed build lacks vision support.")

    # encode_image_raw normally takes a PIL image and runs the preprocessor;
    # for parity testing we want to ensure both paths see the SAME pixel_values
    # and grid_thw, so we go through the binding more directly.
    import ctypes
    n_patches = int(pixel_values.shape[0])
    n_images = int(grid_thw.shape[0])
    n_merged = ctypes.c_int(0)
    out_dim = ctypes.c_int(0)
    n_ds = ctypes.c_int(0)

    pv = np.ascontiguousarray(pixel_values, dtype=np.float32)
    gt = np.ascontiguousarray(grid_thw, dtype=np.int32)
    ptr = ce._lib.crispembed_encode_image_raw(
        ce._ctx,
        pv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_patches),
        gt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(n_images),
        ctypes.byref(n_merged),
        ctypes.byref(out_dim),
        ctypes.byref(n_ds),
    )
    if not ptr or n_merged.value <= 0:
        raise RuntimeError("encode_image_raw returned no data — vision tower missing in GGUF?")
    per_slab = n_merged.value * out_dim.value
    total = (1 + n_ds.value) * per_slab
    flat = np.ctypeslib.as_array(ptr, shape=(total,)).copy()
    img = flat[:per_slab].reshape(n_merged.value, out_dim.value)
    deepstack = [
        flat[(1 + k) * per_slab : (2 + k) * per_slab].reshape(n_merged.value, out_dim.value)
        for k in range(n_ds.value)
    ]
    return img, deepstack


def per_token_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine across rows."""
    if a.shape != b.shape:
        return -1.0
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float((a_n * b_n).sum(axis=-1).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model ID")
    p.add_argument("--gguf",  required=True, help="GGUF path")
    p.add_argument("--image", required=True, help="Image path")
    p.add_argument("--lib",   default=None, help="libcrispembed.{dylib,so} path")
    args = p.parse_args()

    print(f"Image: {args.image}")
    print("HF reference (vision-only path)...")
    hf_img, hf_ds, pv, gt = hf_vision_features(args.model, args.image)
    print(f"  HF image_embeds: {hf_img.shape}; deepstack count: {len(hf_ds)}")
    print(f"  pixel_values: {pv.shape} grid_thw: {gt.tolist()}")

    print("CrispEmbed (bidirlm_vision)...")
    ce_img, ce_ds = crispembed_vision_features(args.gguf, pv, gt, args.lib)
    print(f"  CE image_embeds: {ce_img.shape}; deepstack count: {len(ce_ds)}")

    print()
    ok = True
    if hf_img.shape != ce_img.shape:
        print(f"FAIL: image_embeds shape mismatch {hf_img.shape} vs {ce_img.shape}")
        return 1
    cos_img = per_token_cosine(hf_img, ce_img)
    max_d = float(np.max(np.abs(hf_img - ce_img)))
    print(f"image_embeds: cosine={cos_img:.6f}  max_diff={max_d:.6f}  {'PASS' if cos_img > 0.99 else 'FAIL'}")
    ok = ok and (cos_img > 0.99)

    for k in range(min(len(hf_ds), len(ce_ds))):
        a, b = hf_ds[k], ce_ds[k]
        if a.shape != b.shape:
            print(f"deepstack[{k}]: shape mismatch {a.shape} vs {b.shape}  FAIL")
            ok = False
            continue
        c = per_token_cosine(a, b)
        m = float(np.max(np.abs(a - b)))
        print(f"deepstack[{k}]: cosine={c:.6f}  max_diff={m:.6f}  {'PASS' if c > 0.99 else 'FAIL'}")
        ok = ok and (c > 0.99)

    if len(hf_ds) != len(ce_ds):
        print(f"WARN: deepstack count differs (HF={len(hf_ds)}, CE={len(ce_ds)})")
        ok = False

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
