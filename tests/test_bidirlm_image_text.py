#!/usr/bin/env python3
"""BidirLM-Omni text+image (DeepStack) parity test.

Compares CrispEmbed's image-conditioned text encoder against the HF
reference BidirLMOmniModel.forward(input_ids, pixel_values, image_grid_thw):
  * HF computes inputs_embeds, splices image_embeds at <|image_pad|>
    placeholders, adds DeepStack features at the first 3 layers, and runs
    the bidirectional encoder with 3D MRoPE position ids.
  * CrispEmbed runs the same pipeline via crispembed_encode_text_with_image.

Both paths mean-pool the encoder output over the attention mask and L2
normalize. Pass criterion: cosine similarity ≥ 0.99 on every prompt.

Usage:
    python tests/test_bidirlm_image_text.py \
        --model BidirLM/BidirLM-Omni-2.5B-Embedding \
        --gguf  $CRISPEMBED_CACHE_DIR/bidirlm-omni-2.5b-f16.gguf \
        --image $CRISPEMBED_BIDIRLM_IMAGE

Caveat: the HF model is loaded in fp32 to avoid bf16-vs-f32 numerical
divergence between channels — we want to test our graph, not bf16
quantization. With f16/q4_k GGUFs the cosine still clears 0.99 but the
max-diff will be larger.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def hf_text_with_image(model_id: str, image, prompts: list[str]):
    """Run the full HF BidirLMOmniModel on each prompt with the given image.

    Returns:
        (np.ndarray (N, hidden_size) of L2-normalized mean-pooled embeddings,
         pixel_values np.ndarray, grid_thw np.ndarray, prompt token-id lists)
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    omni_cls = get_class_from_dynamic_module(
        "modeling_bidirlm_omni.BidirLMOmniModel", model_id,
    )
    model = omni_cls(config).to(torch.float32)
    model.eval()

    # Load full state dict (text + visual + audio).
    raw = {}
    try:
        st_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        raw = load_file(st_path)
    except Exception:
        idx_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        for shard in set(idx.values()):
            sp = hf_hub_download(repo_id=model_id, filename=shard)
            raw.update(load_file(sp))
    sd = {k: v.to(torch.float32) for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN: {len(missing)} missing tensors, e.g. {missing[:3]}")

    # Preprocess image with the same processor CrispEmbed uses.
    sys.path.insert(0, "python")
    from crispembed.image import preprocess_image
    pixel_values_np, grid_thw_np = preprocess_image(image, model_name=model_id)

    spatial_merge = config.vision_config.spatial_merge_size
    n_merged = (int(grid_thw_np[0, 0]) * int(grid_thw_np[0, 1]) * int(grid_thw_np[0, 2])) // (spatial_merge * spatial_merge)

    image_token_id = config.image_token_id
    vision_start_id = config.vision_start_token_id
    vision_end_id = config.vision_end_token_id

    pixel_values_pt = torch.from_numpy(pixel_values_np).to(torch.float32)
    grid_thw_pt = torch.from_numpy(grid_thw_np).to(torch.long)

    out_vecs = []
    out_token_id_seqs = []
    for p in prompts:
        # Build "<vision_start><image_pad>*n_merged<vision_end>{prompt}" — an
        # image-and-caption template the model was pretrained on.
        prefix_ids = tokenizer.encode(p, add_special_tokens=False)
        ids = [vision_start_id] + [image_token_id] * n_merged + [vision_end_id] + prefix_ids
        input_ids = torch.tensor([ids], dtype=torch.long)
        attn = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=pixel_values_pt,
                image_grid_thw=grid_thw_pt,
            )
        h = out.last_hidden_state[0]  # (T, H)
        m = attn[0].to(h.dtype)
        pooled = (h * m[:, None]).sum(0) / m.sum().clamp(min=1)
        v = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
        out_vecs.append(v)
        out_token_id_seqs.append(np.asarray(ids, dtype=np.int32))
    return np.stack(out_vecs), pixel_values_np, grid_thw_np, out_token_id_seqs


def crispembed_text_with_image(gguf, lib, pixel_values, grid_thw, token_id_seqs):
    """Run CrispEmbed's pre-tokenized encode_with_image_ids for each prompt.

    Using the token-id ABI (rather than encode_text_with_image) bypasses
    CrispEmbed's BPE tokenizer, so the parity test compares the multimodal
    graph alone — the tokenizer is a separately tested concern.
    """
    sys.path.insert(0, "python")
    from crispembed._binding import CrispEmbed

    ce = CrispEmbed(gguf, lib_path=lib) if lib else CrispEmbed(gguf)
    if not hasattr(ce._lib, "crispembed_encode_with_image_ids"):
        raise RuntimeError("libcrispembed lacks crispembed_encode_with_image_ids — "
                           "rebuild against the latest src/crispembed.h.")

    pv = np.ascontiguousarray(pixel_values, dtype=np.float32)
    gt = np.ascontiguousarray(grid_thw, dtype=np.int32)

    out = []
    for ids in token_id_seqs:
        v = ce.encode_with_image_ids(ids, pv, gt)
        if v.size == 0:
            raise RuntimeError(f"encode_with_image_ids returned no data (n_tokens={len(ids)})")
        out.append(v)
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--lib", default=None)
    args = p.parse_args()

    prompts = [
        "a photograph of a cat",
        "describe what you see in this image",
        "a small animal",
    ]

    if not Path(args.image).exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    print("HF reference (full BidirLMOmniModel)...")
    hf, pv, gt, token_id_seqs = hf_text_with_image(args.model, args.image, prompts)
    print(f"  HF embed shape: {hf.shape}; pixel_values: {pv.shape}; grid_thw: {gt.tolist()}")
    print(f"  token-id seq lengths: {[len(s) for s in token_id_seqs]}")

    print("CrispEmbed (encode_with_image_ids)...")
    ce = crispembed_text_with_image(args.gguf, args.lib, pv, gt, token_id_seqs)
    print(f"  CE embed shape: {ce.shape}")

    if hf.shape != ce.shape:
        print(f"FAIL: shape mismatch {hf.shape} vs {ce.shape}")
        return 1

    print()
    print(f"{'prompt':<50s} {'maxdiff':>10s} {'cosine':>10s}  {'status':>6s}")
    print("-" * 84)
    ok = True
    for prompt, h, c in zip(prompts, hf, ce):
        d = float(np.max(np.abs(h - c)))
        s = float(np.dot(h, c) / (np.linalg.norm(h) * np.linalg.norm(c) + 1e-12))
        passed = s > 0.99
        ok &= passed
        label = (prompt[:47] + "...") if len(prompt) > 50 else prompt
        print(f"{label:<50s} {d:>10.6f} {s:>10.6f}  {'PASS' if passed else 'FAIL':>6s}")

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
