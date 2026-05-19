#!/usr/bin/env python3
"""Convert a SigLIP / CLIP vision encoder to GGUF format.

Extracts the vision tower (ViT) from a CLIP-style model and stores it as
a standalone GGUF file that CrispEmbed can load for image embedding.

    pip install torch transformers gguf pillow
    python models/convert-siglip-to-gguf.py \
        --model google/siglip-base-patch16-384 \
        --output siglip-base.gguf

The text encoder is NOT included — use a separate CrispEmbed text model
for cross-modal text↔image retrieval (the text and vision encoders
project into the same space via a learned projection).

Supported models:
  - google/siglip-base-patch16-384 (Apache 2.0)
  - google/siglip2-base-patch16-384 (Apache 2.0)
  - openai/clip-vit-base-patch32 (MIT)
  - openai/clip-vit-large-patch14 (MIT)
  - Any HuggingFace SiglipModel / CLIPModel
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import torch


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Convert SigLIP/CLIP vision encoder to GGUF")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--output", required=True, help="Output GGUF path")
    args = p.parse_args()

    from transformers import AutoModel, AutoConfig

    print(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32,
                                       trust_remote_code=True)
    model.eval()
    sd = model.state_dict()

    # Determine model type
    model_type = config.model_type  # "siglip" or "clip"
    is_siglip = "siglip" in model_type.lower()
    is_clip = "clip" in model_type.lower()

    if not (is_siglip or is_clip):
        print(f"Warning: unknown model_type '{model_type}', assuming CLIP-like")

    vc = config.vision_config
    hidden = vc.hidden_size
    layers = vc.num_hidden_layers
    heads = vc.num_attention_heads
    inter = vc.intermediate_size
    image_size = vc.image_size
    patch_size = vc.patch_size
    n_patches = (image_size // patch_size) ** 2

    print(f"  Vision: hidden={hidden} layers={layers} heads={heads} inter={inter}")
    print(f"  Image: {image_size}×{image_size}, patch={patch_size}×{patch_size}, patches={n_patches}")

    # Extract vision encoder weights
    vpfx = "vision_model."  # HF prefix for vision tower

    writer = gguf.GGUFWriter(str(args.output), arch="vit")

    # Metadata
    writer.add_uint32("vit.hidden_size", hidden)
    writer.add_uint32("vit.num_hidden_layers", layers)
    writer.add_uint32("vit.num_attention_heads", heads)
    writer.add_uint32("vit.intermediate_size", inter)
    writer.add_uint32("vit.image_size", image_size)
    writer.add_uint32("vit.patch_size", patch_size)
    writer.add_uint32("vit.num_patches", n_patches)
    writer.add_uint32("vit.num_channels", vc.num_channels)
    writer.add_string("vit.model_type", "siglip" if is_siglip else "clip")
    writer.add_float32("vit.layer_norm_eps", getattr(vc, "layer_norm_eps", 1e-6))

    # Image normalization constants
    # SigLIP: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # CLIP: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    if is_siglip:
        writer.add_array("vit.image_mean", [0.5, 0.5, 0.5])
        writer.add_array("vit.image_std", [0.5, 0.5, 0.5])
    else:
        writer.add_array("vit.image_mean", [0.48145466, 0.4578275, 0.40821073])
        writer.add_array("vit.image_std", [0.26862954, 0.26130258, 0.27577711])

    # ── Patch embedding (Conv2D) ──
    # weight: [hidden, channels, patch_h, patch_w]
    patch_w = sd[f"{vpfx}embeddings.patch_embedding.weight"]
    writer.add_tensor("patch_embed.weight", f32(patch_w))
    if f"{vpfx}embeddings.patch_embedding.bias" in sd:
        writer.add_tensor("patch_embed.bias", f32(sd[f"{vpfx}embeddings.patch_embedding.bias"]))
    print("  patch_embed: ok")

    # ── Position embedding ──
    pos_embd = sd[f"{vpfx}embeddings.position_embedding.weight"]
    writer.add_tensor("position_embd.weight", f32(pos_embd))
    print(f"  position_embd: {list(pos_embd.shape)}")

    # ── CLS token (CLIP has it, SigLIP may not) ──
    cls_key = f"{vpfx}embeddings.class_embedding"
    if cls_key in sd:
        writer.add_tensor("cls_token", f32(sd[cls_key]))
        writer.add_bool("vit.has_cls_token", True)
        print("  cls_token: ok")
    else:
        writer.add_bool("vit.has_cls_token", False)

    # ── Pre-LN (CLIP) or Post-LN (SigLIP) ──
    pre_ln_key = f"{vpfx}pre_layrnorm.weight"  # CLIP has pre_layrnorm
    if pre_ln_key in sd:
        writer.add_tensor("pre_ln.weight", f32(sd[pre_ln_key]))
        writer.add_tensor("pre_ln.bias", f32(sd[pre_ln_key.replace(".weight", ".bias")]))
        print("  pre_ln: ok")

    # ── Transformer layers ──
    for i in range(layers):
        lpfx = f"{vpfx}encoder.layers.{i}."

        # Layer norm 1 (pre-attention)
        writer.add_tensor(f"enc.{i}.ln1.weight", f32(sd[f"{lpfx}layer_norm1.weight"]))
        writer.add_tensor(f"enc.{i}.ln1.bias", f32(sd[f"{lpfx}layer_norm1.bias"]))

        # Self-attention Q/K/V
        # SigLIP/CLIP fuse QKV into in_proj_weight [3H, H]
        in_proj_w_key = f"{lpfx}self_attn.in_proj_weight"  # fused
        q_proj_w_key = f"{lpfx}self_attn.q_proj.weight"    # separate

        if in_proj_w_key in sd:
            # Fused QKV → split
            qkv_w = sd[in_proj_w_key]
            H = qkv_w.shape[1]
            writer.add_tensor(f"enc.{i}.attn.q.weight", f32(qkv_w[:H]))
            writer.add_tensor(f"enc.{i}.attn.k.weight", f32(qkv_w[H:2*H]))
            writer.add_tensor(f"enc.{i}.attn.v.weight", f32(qkv_w[2*H:]))
            if f"{lpfx}self_attn.in_proj_bias" in sd:
                qkv_b = sd[f"{lpfx}self_attn.in_proj_bias"]
                writer.add_tensor(f"enc.{i}.attn.q.bias", f32(qkv_b[:H]))
                writer.add_tensor(f"enc.{i}.attn.k.bias", f32(qkv_b[H:2*H]))
                writer.add_tensor(f"enc.{i}.attn.v.bias", f32(qkv_b[2*H:]))
        elif q_proj_w_key in sd:
            # Separate Q/K/V
            writer.add_tensor(f"enc.{i}.attn.q.weight", f32(sd[f"{lpfx}self_attn.q_proj.weight"]))
            writer.add_tensor(f"enc.{i}.attn.k.weight", f32(sd[f"{lpfx}self_attn.k_proj.weight"]))
            writer.add_tensor(f"enc.{i}.attn.v.weight", f32(sd[f"{lpfx}self_attn.v_proj.weight"]))
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                bk = f"{lpfx}self_attn.{proj}.bias"
                if bk in sd:
                    p = proj[0]  # q, k, v
                    writer.add_tensor(f"enc.{i}.attn.{p}.bias", f32(sd[bk]))

        # Attention output projection
        writer.add_tensor(f"enc.{i}.attn.o.weight", f32(sd[f"{lpfx}self_attn.out_proj.weight"]))
        if f"{lpfx}self_attn.out_proj.bias" in sd:
            writer.add_tensor(f"enc.{i}.attn.o.bias", f32(sd[f"{lpfx}self_attn.out_proj.bias"]))

        # Layer norm 2 (pre-FFN)
        writer.add_tensor(f"enc.{i}.ln2.weight", f32(sd[f"{lpfx}layer_norm2.weight"]))
        writer.add_tensor(f"enc.{i}.ln2.bias", f32(sd[f"{lpfx}layer_norm2.bias"]))

        # FFN (MLP)
        writer.add_tensor(f"enc.{i}.ffn.fc1.weight", f32(sd[f"{lpfx}mlp.fc1.weight"]))
        writer.add_tensor(f"enc.{i}.ffn.fc1.bias", f32(sd[f"{lpfx}mlp.fc1.bias"]))
        writer.add_tensor(f"enc.{i}.ffn.fc2.weight", f32(sd[f"{lpfx}mlp.fc2.weight"]))
        writer.add_tensor(f"enc.{i}.ffn.fc2.bias", f32(sd[f"{lpfx}mlp.fc2.bias"]))

        print(f"  enc.{i}: ok")

    # ── Post-LayerNorm ──
    post_ln_key = f"{vpfx}post_layernorm.weight"
    if post_ln_key in sd:
        writer.add_tensor("post_ln.weight", f32(sd[post_ln_key]))
        writer.add_tensor("post_ln.bias", f32(sd[post_ln_key.replace(".weight", ".bias")]))
        print("  post_ln: ok")

    # ── SigLIP attention pooling head ──
    head_pfx = f"{vpfx}head."
    if f"{head_pfx}probe" in sd:
        writer.add_tensor("head.probe", f32(sd[f"{head_pfx}probe"]))
        writer.add_tensor("head.attn.in_proj.weight", f32(sd[f"{head_pfx}attention.in_proj_weight"]))
        writer.add_tensor("head.attn.in_proj.bias", f32(sd[f"{head_pfx}attention.in_proj_bias"]))
        writer.add_tensor("head.attn.o.weight", f32(sd[f"{head_pfx}attention.out_proj.weight"]))
        writer.add_tensor("head.attn.o.bias", f32(sd[f"{head_pfx}attention.out_proj.bias"]))
        writer.add_tensor("head.ln.weight", f32(sd[f"{head_pfx}layernorm.weight"]))
        writer.add_tensor("head.ln.bias", f32(sd[f"{head_pfx}layernorm.bias"]))
        writer.add_tensor("head.mlp.fc1.weight", f32(sd[f"{head_pfx}mlp.fc1.weight"]))
        writer.add_tensor("head.mlp.fc1.bias", f32(sd[f"{head_pfx}mlp.fc1.bias"]))
        writer.add_tensor("head.mlp.fc2.weight", f32(sd[f"{head_pfx}mlp.fc2.weight"]))
        writer.add_tensor("head.mlp.fc2.bias", f32(sd[f"{head_pfx}mlp.fc2.bias"]))
        writer.add_bool("vit.has_attn_pool", True)
        print("  attention_pool_head: ok")
    else:
        writer.add_bool("vit.has_attn_pool", False)

    # ── CLIP visual projection (maps to shared text-image space) ──
    vp_key = "visual_projection.weight"
    if vp_key in sd:
        writer.add_tensor("visual_proj.weight", f32(sd[vp_key]))
        writer.add_bool("vit.has_visual_proj", True)
        proj_dim = sd[vp_key].shape[0]
        writer.add_uint32("vit.projection_dim", proj_dim)
        print(f"  visual_projection: {list(sd[vp_key].shape)}")
    else:
        writer.add_bool("vit.has_visual_proj", False)

    # Write
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
