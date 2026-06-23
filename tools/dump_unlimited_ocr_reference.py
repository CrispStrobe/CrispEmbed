#!/usr/bin/env python3
"""Dump Unlimited-OCR reference vision intermediates for crispembed-diff parity.

Vision pathway:

    image (1,3,1024,1024)
      -> sam_model  (SAM-ViT-B)     -> "sam_output"   (1024, 16, 16)
      -> vision_model (CLIP-L/14)   -> "clip_output"  (257, 1024) with CLS
      -> concat(clip[:,1:], sam.flat) -> "fused_features" (256, 2048)
      -> projector  (linear 2048->1280) -> "projector_output" (256, 1280)
      -> add image_newline per row  -> "vision_features" (272, 1280)
      -> append view_separator      -> total (273, 1280)

CLIP receives SAM output as patch_embeds (replacing its conv patch embedding).

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_unlimited_ocr_reference.py \
        --model-dir /mnt/storage/models/Unlimited-OCR \
        --image test.png \
        --output /mnt/volume1/tmp-overflow/unlimited-ocr-ref.gguf
"""

import argparse
from pathlib import Path
import numpy as np


def squeeze_leading(data: np.ndarray) -> np.ndarray:
    """Drop leading singleton dims (e.g. batch) so element count matches C++."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    return data


def as_numpy(x):
    """bf16/fp16/fp32 tensor (or tuple[0]) -> float32 numpy."""
    import torch
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    assert isinstance(x, torch.Tensor), f"unexpected hook output: {type(x)}"
    return x.detach().to(torch.float32).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                    help="HF model dir (safetensors + config + *.py)")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", required=True,
                    help="Output GGUF reference file")
    p.add_argument("--base-size", type=int, default=1024,
                    help="Global-view square size")
    args = p.parse_args()

    import importlib.util
    import sys
    import gc
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms
    from safetensors import safe_open

    torch.manual_seed(0)
    model_dir = Path(args.model_dir)
    st_path = str(model_dir / "model-00001-of-000001.safetensors")

    # ---- Load architecture code directly ----
    print("Importing deepencoder (vision modules)...", flush=True)
    spec = importlib.util.spec_from_file_location(
        "deepencoder", str(model_dir / "deepencoder.py"))
    de = importlib.util.module_from_spec(spec)
    sys.modules["deepencoder"] = de
    spec.loader.exec_module(de)

    # ---- Build modules ----
    sam = de.build_sam_vit_b()
    clip = de.build_clip_l()

    class Cfg(dict):
        def __getattr__(self, k):
            return self[k]

    projector = de.MlpProjector(
        Cfg(projector_type="linear", input_dim=2048, n_embed=1280))

    # ---- Load weights by prefix ----
    def load_prefix(module, prefix, strict=True):
        sd = {}
        with safe_open(st_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = f.get_tensor(k).float()
        module.load_state_dict(sd, strict=strict)
        module.eval().float()
        print(f"  loaded {len(sd)} tensors into {prefix}", flush=True)

    load_prefix(sam, "model.sam_model.")
    load_prefix(clip, "model.vision_model.", strict=False)
    load_prefix(projector, "model.projector.")

    # Load learned tokens
    with safe_open(st_path, framework="pt") as f:
        vsep = f.get_tensor("model.view_seperator").float()
        img_newline = f.get_tensor("model.image_newline").float()

    # ---- Preprocess ----
    img = Image.open(args.image).convert("RGB")
    img = ImageOps.exif_transpose(img)
    print(f"Image: {img.size[0]}x{img.size[1]}", flush=True)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    global_view = ImageOps.pad(
        img, (args.base_size, args.base_size),
        color=tuple(int(x * 255) for x in mean),
    )
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    image_ori = tf(global_view).unsqueeze(0).float()  # (1,3,base,base)
    print(f"image_ori: {tuple(image_ori.shape)}", flush=True)

    captures = {}

    def hook(name):
        def fn(module, inp, out):
            captures[name] = as_numpy(out)
        return fn

    # ---- Hook per-layer SAM hidden states ----
    # SAM blocks produce (B, H, W, C) spatial tensors. Hook each block to
    # capture intermediate hidden states for bisection debugging.
    # Use a pre-hook on blocks[0] to get hidden state AFTER patch_embed + pos_embed.
    def pre_hook_input(name):
        def fn(module, args):
            captures[name] = as_numpy(args[0])
        return fn
    sam.blocks[0].register_forward_pre_hook(pre_hook_input("sam_patch_embed"))
    for i, blk in enumerate(sam.blocks):
        blk.register_forward_hook(hook(f"sam_layer_{i}"))
    # Hook neck components
    sam.neck.register_forward_hook(hook("sam_neck"))

    # ---- Hook per-layer CLIP hidden states ----
    for i, layer in enumerate(clip.transformer.layers):
        layer.register_forward_hook(hook(f"clip_layer_{i}"))

    # ---- Run vision pathway ----
    print("Running SAM...", flush=True)
    with torch.no_grad():
        sam_out = sam(image_ori)  # (1, 1024, 16, 16)
    captures["sam_output_raw"] = as_numpy(sam_out)

    print("Running CLIP (with SAM patch_embeds)...", flush=True)
    with torch.no_grad():
        clip_out = clip(image_ori, sam_out)  # (1, 257, 1024)
    captures["clip_output"] = as_numpy(clip_out)

    print("Running fusion + projector...", flush=True)
    with torch.no_grad():
        # Fusion: concat CLIP (skip CLS) + SAM flattened
        clip_feats = clip_out[:, 1:]  # (1, 256, 1024)
        sam_flat = sam_out.flatten(2).permute(0, 2, 1)  # (1, 256, 1024)
        fused = torch.cat((clip_feats, sam_flat), dim=-1)  # (1, 256, 2048)
        captures["fused_features"] = as_numpy(fused)

        proj_out = projector(fused)  # (1, 256, 1280)
        captures["projector_output"] = as_numpy(proj_out)

    # ---- Build vision_features with image_newline ----
    # Reshape to (16, 16, 1280), add newline at end of each row → (16, 17, 1280)
    feats = proj_out.squeeze(0)  # (256, 1280)
    hw = feats.shape[0]
    h = w = int(hw ** 0.5)
    n_dim = feats.shape[1]
    feats_grid = feats.view(h, w, n_dim)
    feats_with_nl = torch.cat(
        [feats_grid, img_newline[None, None, :].expand(h, 1, n_dim)], dim=1
    )  # (16, 17, 1280)
    feats_flat = feats_with_nl.view(-1, n_dim)  # (272, 1280)
    # Append view_separator
    vision_features = torch.cat(
        [feats_flat, vsep[None, :]], dim=0
    )  # (273, 1280)
    captures["vision_features"] = as_numpy(vision_features)
    captures["view_separator"] = as_numpy(vsep)
    captures["image_newline"] = as_numpy(img_newline)

    # SAM output in token-major format (H*W, C) for comparison with C++
    so = np.squeeze(captures["sam_output_raw"])  # (1024, 16, 16)
    if so.ndim == 3:
        so = so.transpose(1, 2, 0).reshape(-1, so.shape[0])  # (256, 1024)
    captures["sam_output"] = so
    del captures["sam_output_raw"]

    # Convert SAM per-layer captures to token-major [N, C] format.
    # SAM blocks output (B, H, W, C) — squeeze batch, flatten spatial to (H*W, C).
    # patch_embed output is (B, H, W, C) too. Neck output is (B, C, H, W).
    for key in list(captures.keys()):
        if key.startswith("sam_layer_") or key == "sam_patch_embed":
            v = np.squeeze(captures[key])  # drop batch
            if v.ndim == 3:  # (H, W, C) → (H*W, C) token-major
                H, W, C = v.shape
                captures[key] = v.reshape(H * W, C)
        elif key == "sam_neck":
            v = np.squeeze(captures[key])  # (C, H, W)
            if v.ndim == 3:
                C, H, W = v.shape
                captures[key] = v.transpose(1, 2, 0).reshape(H * W, C)

    # ---- Write GGUF ----
    print(f"Writing reference GGUF to {args.output}...", flush=True)
    import gguf
    writer = gguf.GGUFWriter(args.output, arch="unlimited_ocr_ref")
    for name in sorted(captures.keys()):
        data = squeeze_leading(captures[name])
        writer.add_tensor(name, data,
                          raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"  {name}: shape={data.shape}, "
              f"range=[{data.min():.4f}, {data.max():.4f}]", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {len(captures)} stages to {args.output}", flush=True)


if __name__ == "__main__":
    main()
