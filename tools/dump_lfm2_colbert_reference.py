#!/usr/bin/env python3
"""Dump LFM2.5-ColBERT reference for parity testing (memory-efficient).

Uses safetensors lazy loading — runs the backbone one layer at a time
to stay within 8GB RAM.

Usage:
    python tools/dump_lfm2_colbert_reference.py \
        --model /mnt/storage/models/LFM2.5-ColBERT-350M \
        --output /mnt/storage/gguf-models/lfm2-colbert-ref.gguf
"""
import argparse, sys, os, gc
from pathlib import Path
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", "-o", required=True)
    args = p.parse_args()

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = "query: The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = inputs["input_ids"]  # [1, T]
    T = input_ids.shape[1]
    print(f"Input: {T} tokens")

    st_path = os.path.join(args.model, "model.safetensors")

    # Load only what we need per layer from safetensors
    with safe_open(st_path, framework="pt", device="cpu") as sf:
        # Embedding
        embed_w = sf.get_tensor("embed_tokens.weight").float()
        hidden = embed_w[input_ids.squeeze(0)]  # [T, 1024]
        H = hidden.shape[1]
        print(f"Embedding: [{T}, {H}]")
        del embed_w; gc.collect()

        # Read layer types from config
        import json
        with open(os.path.join(args.model, "config.json")) as f:
            cfg = json.load(f)
        layer_types = cfg.get("block_layer_types", "ccaccaccacacacac")
        n_layers = cfg.get("num_hidden_layers", 16)
        n_heads = cfg.get("num_attention_heads", 16)
        n_kv = cfg.get("num_key_value_heads", 8)
        hd = H // n_heads
        eps = cfg.get("rms_norm_eps", 1e-5)
        rope_theta = cfg.get("rope_theta", 1000000.0)

        # Position IDs
        pos = torch.arange(T)

        # Run layers one at a time
        for li in range(n_layers):
            lt = layer_types[li] if li < len(layer_types) else 'c'
            prefix = f"layers.{li}"

            # Load layer weights
            op_norm = sf.get_tensor(f"{prefix}.operator_norm.weight").float()
            ffn_norm = sf.get_tensor(f"{prefix}.ffn_norm.weight").float()
            w1 = sf.get_tensor(f"{prefix}.feed_forward.w1.weight").float()
            w2 = sf.get_tensor(f"{prefix}.feed_forward.w2.weight").float()
            w3 = sf.get_tensor(f"{prefix}.feed_forward.w3.weight").float()

            # Operator norm
            def rms_norm(x, w):
                var = x.pow(2).mean(-1, keepdim=True)
                return x * torch.rsqrt(var + eps) * w

            normed = rms_norm(hidden, op_norm)

            if lt == 'a':
                # Attention layer
                q_w = sf.get_tensor(f"{prefix}.self_attn.q_proj.weight").float()
                k_w = sf.get_tensor(f"{prefix}.self_attn.k_proj.weight").float()
                v_w = sf.get_tensor(f"{prefix}.self_attn.v_proj.weight").float()
                o_w = sf.get_tensor(f"{prefix}.self_attn.out_proj.weight").float()

                Q = (normed @ q_w.T).view(T, n_heads, hd)
                K = (normed @ k_w.T).view(T, n_kv, hd)
                V = (normed @ v_w.T).view(T, n_kv, hd)

                # RoPE
                freqs = 1.0 / (rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
                t = pos.float().unsqueeze(1) * freqs.unsqueeze(0)  # [T, hd/2]
                cos_t = torch.cos(t); sin_t = torch.sin(t)
                def apply_rope(x):
                    x1 = x[..., :hd//2]; x2 = x[..., hd//2:]
                    c = cos_t.unsqueeze(1); s = sin_t.unsqueeze(1)
                    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
                Q = apply_rope(Q); K = apply_rope(K)

                # GQA repeat
                rep = n_heads // n_kv
                K = K.repeat_interleave(rep, dim=1)
                V = V.repeat_interleave(rep, dim=1)

                # Attention (bidirectional — no causal mask)
                Q = Q.transpose(0, 1); K = K.transpose(0, 1); V = V.transpose(0, 1)
                scale = 1.0 / (hd ** 0.5)
                attn = (Q @ K.transpose(-2, -1)) * scale
                attn = torch.softmax(attn, dim=-1)
                out = (attn @ V).transpose(0, 1).reshape(T, H)
                out = out @ o_w.T
                del q_w, k_w, v_w, o_w, Q, K, V, attn
            else:
                # Conv layer (ShortConv)
                conv_w = sf.get_tensor(f"{prefix}.conv.conv.weight").float()  # [1, 1, K]
                in_proj = sf.get_tensor(f"{prefix}.conv.in_proj.weight").float()
                out_proj = sf.get_tensor(f"{prefix}.conv.out_proj.weight").float()

                # in_proj: [3*H, H] → split into 3 parts
                projected = normed @ in_proj.T  # [T, 3*H]
                x_conv = projected[:, :H]
                gate = projected[:, H:2*H]
                skip = projected[:, 2*H:]

                # Depthwise 1D conv (kernel size K, causal for unidirectional, full for bidirectional)
                K_sz = conv_w.shape[2]
                # For bidirectional, use symmetric padding
                pad = K_sz // 2
                x_padded = torch.nn.functional.pad(x_conv.T.unsqueeze(0), (pad, pad))  # [1, H, T+2*pad]
                # Actually the conv weight is [1, 1, K] — depthwise per channel
                # Each channel convolved independently
                conv_out = torch.nn.functional.conv1d(
                    x_padded, conv_w.expand(H, 1, K_sz), groups=H)[:, :, :T]
                conv_out = conv_out.squeeze(0).T  # [T, H]

                # SiLU gate
                out = conv_out * torch.sigmoid(gate) * skip
                out = out @ out_proj.T
                del conv_w, in_proj, out_proj, projected

            hidden = hidden + out  # residual

            # FFN
            normed_ff = rms_norm(hidden, ffn_norm)
            ff_gate = normed_ff @ w1.T
            ff_up = normed_ff @ w3.T
            ff_out = (torch.sigmoid(ff_gate) * ff_gate) * ff_up  # SwiGLU
            ff_out = ff_out @ w2.T
            hidden = hidden + ff_out

            del op_norm, ffn_norm, w1, w2, w3, normed, normed_ff, out, ff_gate, ff_up, ff_out
            gc.collect()

            if li % 4 == 3:
                print(f"  layer {li}: range=[{hidden.min():.4f}, {hidden.max():.4f}]")

        # Final norm
        emb_norm = sf.get_tensor("embedding_norm.weight").float()
        hidden = rms_norm(hidden, emb_norm)
        print(f"After final norm: range=[{hidden.min():.4f}, {hidden.max():.4f}]")

    # ColBERT projection
    dense_path = os.path.join(args.model, "1_Dense", "model.safetensors")
    with safe_open(dense_path, framework="pt", device="cpu") as f:
        proj_w = f.get_tensor("linear.weight").float()  # [128, 1024]
    projected = hidden @ proj_w.T  # [T, 128]
    projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
    print(f"ColBERT output: [{T}, {proj_w.shape[0]}], range=[{projected.min():.4f}, {projected.max():.4f}]")

    # Save reference
    writer = gguf.GGUFWriter(args.output, "lfm2-colbert-ref")
    writer.add_uint32("ref.n_tokens", T)
    writer.add_uint32("ref.colbert_dim", proj_w.shape[0])
    writer.add_tensor("hidden_states", hidden.numpy().astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("colbert_output", projected.numpy().astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("input_ids", input_ids.squeeze(0).numpy().astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
