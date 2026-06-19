#!/usr/bin/env python3
"""Dump LightOnOCR-2-1B per-layer activations for crispembed-diff parity testing.

Loads weights via safetensors (no full model load — fits 8GB RAM).
Runs vision encoder manually layer by layer, dumps intermediates.

Usage:
    PYTHONNOUSERSITE=1 python3 tools/dump_lightonocr_reference.py \
        --model lightonai/LightOnOCR-2-1B \
        --image test.png \
        --output /tmp/lightonocr-ref.gguf \
        --max-vis-layers 2
"""
import argparse, gc, json, math, os, sys
import numpy as np
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-vis-layers", type=int, default=2)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from PIL import Image

    # Load config
    cfg_path = hf_hub_download(args.model, "config.json",
        cache_dir='/mnt/akademie_storage/huggingface/hub/')
    with open(cfg_path) as f:
        cfg = json.load(f)
    vc = cfg["vision_config"]
    patch_size = vc["patch_size"]
    hidden = vc["hidden_size"]
    n_heads = vc["num_attention_heads"]
    head_dim = vc.get("head_dim", hidden // n_heads)
    inter = vc["intermediate_size"]
    n_layers = vc["num_hidden_layers"]
    rope_theta = vc.get("rope_theta", 10000.0)
    max_size = vc.get("image_size", 1540)

    sf_path = hf_hub_download(args.model, "model.safetensors",
        cache_dir='/mnt/akademie_storage/huggingface/hub/')

    # Preprocess image (matching C++ and Pixtral processor)
    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    ratio = max(h / max_size, w / max_size)
    if ratio > 1:
        h = int(math.floor(h / ratio))
        w = int(math.floor(w / ratio))
    pw = (w - 1) // patch_size + 1
    ph = (h - 1) // patch_size + 1
    if pw % 2: pw += 1
    if ph % 2: ph += 1
    tw, th = pw * patch_size, ph * patch_size
    n_patches = ph * pw
    print(f"Image {img.size} → {tw}x{th}, {pw}x{ph} = {n_patches} patches")

    img_resized = img.resize((tw, th), Image.BILINEAR)
    pixels = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std_ = np.array([0.26862954, 0.26130258, 0.27577711])
    pixels = (pixels - mean) / std_
    pixels = pixels.transpose(2, 0, 1)  # CHW
    pixels_t = torch.from_numpy(pixels).float().unsqueeze(0)

    tensors = {}

    with safe_open(sf_path, framework="pt") as sf:
        # Patch conv
        conv_w = sf.get_tensor("model.vision_encoder.patch_conv.weight").float()
        patches = torch.nn.functional.conv2d(pixels_t, conv_w, stride=patch_size)
        x = patches.squeeze(0).reshape(hidden, -1).T  # (n_patches, hidden)
        tensors["patch_conv_out"] = x.numpy().copy()
        print(f"  patch_conv: {x.shape}, first5={x[0,:5].tolist()}")

        # ln_pre (RMSNorm)
        ln_w = sf.get_tensor("model.vision_encoder.ln_pre.weight").float()
        var = (x * x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + 1e-6) * ln_w
        tensors["ln_pre_out"] = x.numpy().copy()
        print(f"  ln_pre: first5={x[0,:5].tolist()}")

        # 2D RoPE (Pixtral PixtralRotaryEmbedding)
        dim = head_dim
        half = dim // 2
        freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs_h = torch.outer(torch.arange(ph).float(), freqs[::2])
        freqs_w = torch.outer(torch.arange(pw).float(), freqs[1::2])
        inv_freq = torch.cat([
            freqs_h[:, None, :].repeat(1, pw, 1),
            freqs_w[None, :, :].repeat(ph, 1, 1),
        ], dim=-1).reshape(-1, half)
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        cos_rope = inv_freq.cos()
        sin_rope = inv_freq.sin()
        tensors["rope_cos"] = cos_rope.numpy().copy()
        tensors["rope_sin"] = sin_rope.numpy().copy()
        print(f"  2D RoPE: {cos_rope.shape}")

        def rotate_half(t):
            h = t.shape[-1] // 2
            return torch.cat((-t[..., h:], t[..., :h]), dim=-1)

        def apply_rope(t, cos, sin):
            return t * cos + rotate_half(t) * sin

        def rmsnorm(t, w, eps=1e-6):
            var = (t * t).mean(dim=-1, keepdim=True)
            return t * torch.rsqrt(var + eps) * w

        # Vision transformer layers
        max_l = min(args.max_vis_layers, n_layers)
        for il in range(max_l):
            pfx = f"model.vision_encoder.transformer.layers.{il}"
            residual = x.clone()

            # Pre-attn RMSNorm
            norm_w = sf.get_tensor(f"{pfx}.attention_norm.weight").float()
            y = rmsnorm(x, norm_w)

            # Q/K/V
            q_w = sf.get_tensor(f"{pfx}.attention.q_proj.weight").float()
            k_w = sf.get_tensor(f"{pfx}.attention.k_proj.weight").float()
            v_w = sf.get_tensor(f"{pfx}.attention.v_proj.weight").float()
            Q = (y @ q_w.T).reshape(n_patches, n_heads, head_dim)
            K = (y @ k_w.T).reshape(n_patches, n_heads, head_dim)
            V = (y @ v_w.T).reshape(n_patches, n_heads, head_dim)

            # Apply 2D RoPE
            cos_r = cos_rope.unsqueeze(1)  # (T, 1, dim)
            sin_r = sin_rope.unsqueeze(1)
            Q = apply_rope(Q, cos_r, sin_r)
            K = apply_rope(K, cos_r, sin_r)

            tensors[f"vis_layer_{il}_q_rope"] = Q.reshape(n_patches, -1).detach().numpy().copy()
            tensors[f"vis_layer_{il}_k_rope"] = K.reshape(n_patches, -1).detach().numpy().copy()

            # Attention: (nh, T, hd) @ (nh, hd, T) = (nh, T, T) → softmax → @ V
            Q_p = Q.permute(1, 0, 2)  # (nh, T, hd)
            K_p = K.permute(1, 0, 2)
            V_p = V.permute(1, 0, 2)
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(Q_p, K_p.transpose(-2, -1)) * scale
            attn_w = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_w, V_p)
            attn_out = attn_out.permute(1, 0, 2).reshape(n_patches, hidden)

            # Output projection
            o_w = sf.get_tensor(f"{pfx}.attention.o_proj.weight").float()
            attn_out = attn_out @ o_w.T
            tensors[f"vis_layer_{il}_attn_out"] = attn_out.detach().numpy().copy()
            print(f"  vis_layer_{il}_attn_out: first10={attn_out[0,:10].tolist()}")
            x = residual + attn_out
            tensors[f"vis_layer_{il}_post_attn"] = x.detach().numpy().copy()
            print(f"  vis_layer_{il}_post_attn: first5={x[0,:5].tolist()}")

            # FFN: SiLU gate * up → down
            residual = x.clone()
            ffn_norm_w = sf.get_tensor(f"{pfx}.ffn_norm.weight").float()
            y = rmsnorm(x, ffn_norm_w)
            gate_w = sf.get_tensor(f"{pfx}.feed_forward.gate_proj.weight").float()
            up_w = sf.get_tensor(f"{pfx}.feed_forward.up_proj.weight").float()
            down_w = sf.get_tensor(f"{pfx}.feed_forward.down_proj.weight").float()
            gate = torch.nn.functional.silu(y @ gate_w.T)
            up = y @ up_w.T
            x = residual + (gate * up) @ down_w.T

            tensors[f"vis_layer_{il}_out"] = x.detach().numpy().copy()
            print(f"  vis_layer_{il}: first5={x[0,:5].tolist()}")

    # Write reference GGUF
    import gguf
    writer = gguf.GGUFWriter(args.output, arch="lightonocr-ref")
    for name, arr in tensors.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {args.output} ({os.path.getsize(args.output)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
