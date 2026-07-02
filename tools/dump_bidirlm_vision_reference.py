#!/usr/bin/env python3
"""Dump a BidirLM-Omni vision-tower reference GGUF for test-bidirlm-vision-diff.

Loads ONLY the ``visual.*`` weights (lazy, via safetensors.safe_open — ~1.5 GB,
not the full ~5 GB model), runs the HF BidirLMOmniVisionModel on a fixed image
through the exact preprocessor CrispEmbed uses, and writes the input
(pixel_values, image_grid_thw) plus the reference outputs (image_embeds and each
deepstack slab) to a GGUF. The C++ diff harness feeds the stored pixel_values to
crispembed_encode_image_raw and compares its image_embeds/deepstack against these.

Usage:
  HF_HOME=... HF_MODULES_CACHE=... \
  python tools/dump_bidirlm_vision_reference.py \
      --model BidirLM/BidirLM-Omni-2.5B-Embedding \
      --image tests/regression/images/fox.png \
      --output /tmp/bidirlm-vision-ref.gguf
"""
import argparse
import sys

import numpy as np


def hf_vision_features(model_id, image):
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    vision_cls = get_class_from_dynamic_module(
        "modeling_bidirlm_omni.BidirLMOmniVisionModel", model_id
    )
    vision = vision_cls(config.vision_config)
    vision.eval()

    # Lazy: pull only visual.* tensors out of the (possibly sharded) safetensors.
    sd = {}
    try:
        st = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        shards = [st]
    except Exception:
        import json

        idx = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        shards = sorted({hf_hub_download(repo_id=model_id, filename=s) for s in set(wm.values())})
    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                if k.startswith("visual."):
                    sd[k[len("visual.") :]] = f.get_tensor(k).float()
    missing, _ = vision.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN: {len(missing)} missing visual tensors, e.g. {missing[:3]}")

    sys.path.insert(0, "python")
    from crispembed.image import preprocess_image

    pv_np, gt_np = preprocess_image(image, model_name=model_id)
    with torch.no_grad():
        image_embeds, deepstack = vision(
            torch.from_numpy(pv_np).float(), grid_thw=torch.from_numpy(gt_np).long()
        )
    img = image_embeds.float().cpu().numpy()
    ds = [d.float().cpu().numpy() for d in deepstack]
    return pv_np, gt_np, img, ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BidirLM/BidirLM-Omni-2.5B-Embedding")
    p.add_argument("--image", default="tests/regression/images/fox.png")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import gguf

    pv, gt, img, ds = hf_vision_features(args.model, args.image)
    print(f"pixel_values {pv.shape}  grid_thw {gt.tolist()}  image_embeds {img.shape}  deepstack {len(ds)}")

    w = gguf.GGUFWriter(args.output, "bidirlm_vision_ref")
    # Inputs (fed to crispembed_encode_image_raw by the C++ harness). grid_thw is
    # stored as f32 and rounded back to int on the C++ side.
    w.add_tensor("pixel_values", np.ascontiguousarray(pv, dtype=np.float32))
    w.add_tensor("image_grid_thw", np.ascontiguousarray(gt, dtype=np.float32))
    # Reference outputs.
    w.add_tensor("image_embeds", np.ascontiguousarray(img, dtype=np.float32))
    for k, d in enumerate(ds):
        w.add_tensor(f"deepstack.{k}", np.ascontiguousarray(d, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
