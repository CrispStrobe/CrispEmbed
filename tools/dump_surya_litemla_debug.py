#!/usr/bin/env python3
"""Dump LiteMLA internals for stage 3 block 1 of surya detector.

Captures: QKV, aggregated QKV, Q/K/V split, attention output, projection,
context residual, local MBConv output, local residual.
"""
import sys
import numpy as np
import torch
import gguf

def main():
    from surya.detection.loader import DetectionModelLoader
    from PIL import Image

    print("Loading model...")
    loader = DetectionModelLoader()
    model = loader.model(device='cpu', dtype=torch.float32)
    processor = loader.processor()

    # Load test image
    img = np.asarray(Image.open('/tmp/test_doc.png').convert("RGB"))
    new_size = (processor.size["width"], processor.size["height"])
    pil = Image.fromarray(img).convert("RGB")
    pil.thumbnail(new_size, Image.Resampling.LANCZOS)
    pil = pil.resize(new_size, Image.Resampling.LANCZOS)
    arr = np.asarray(pil, dtype=np.uint8)
    processed = processor(arr)["pixel_values"][0]
    pixel_values = torch.from_numpy(processed).unsqueeze(0).float()

    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                activations[name] = out.detach().cpu().float()
        return hook

    # Hook into stage 3 block 0 (MBConv) output
    model.vit.stages[3].blocks[0].register_forward_hook(make_hook("s3_block0"))

    # Hook into stage 3 block 1 (EfficientVitBlock) internals
    blk1 = model.vit.stages[3].blocks[1]

    # Context module (LiteMLA)
    ctx_mod = blk1.context_module
    litemla = ctx_mod.main

    # QKV conv
    litemla.qkv.register_forward_hook(make_hook("b1_qkv"))

    # Aggregation
    litemla.aggreg[0][0].register_forward_hook(make_hook("b1_agg_dw"))
    litemla.aggreg[0][1].register_forward_hook(make_hook("b1_agg_pw"))

    # Projection
    litemla.proj.register_forward_hook(make_hook("b1_proj"))

    # Context module output (after residual)
    ctx_mod.register_forward_hook(make_hook("b1_context"))

    # Local module
    local_mod = blk1.local_module
    local_mod.main.inverted_conv.register_forward_hook(make_hook("b1_local_inv"))
    local_mod.main.depth_conv.register_forward_hook(make_hook("b1_local_dw"))
    local_mod.main.point_conv.register_forward_hook(make_hook("b1_local_pt"))
    local_mod.register_forward_hook(make_hook("b1_local"))

    # Full block output
    blk1.register_forward_hook(make_hook("b1_output"))

    print("Running forward pass...")
    with torch.inference_mode():
        output = model(pixel_values)

    print("\n=== Stage 3 Block 1 LiteMLA internals ===")
    writer = gguf.GGUFWriter('/tmp/surya-litemla-debug.gguf', 'surya-litemla-debug')

    for name in sorted(activations.keys()):
        t = activations[name]
        if t.dim() >= 4:
            t = t[0]
        elif t.dim() == 3:
            t = t[0]
        data = t.numpy().astype(np.float32)
        print(f"  {name}: {list(data.shape)} min={data.min():.4f} max={data.max():.4f} mean={data.mean():.4f}")
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nSaved to /tmp/surya-litemla-debug.gguf")

if __name__ == "__main__":
    main()
