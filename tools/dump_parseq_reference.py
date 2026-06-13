#!/usr/bin/env python3
"""Dump PARSeq reference activations for parity testing.

Runs the PARSeq model (from .pt checkpoint) on a test image and dumps
per-layer intermediate activations to a GGUF file. The C++ diff test
compares its outputs against these reference values.

Usage:
    python tools/dump_parseq_reference.py \
        --checkpoint /mnt/storage/models/parseq-bb5792a6.pt \
        --image test.png \
        --output /tmp/parseq-ref.gguf
"""

import argparse
import math
import string
import sys
from pathlib import Path

import gguf
import numpy as np

# PARSeq charset
CHARSET = string.printable[:94]


def preprocess_image(img_path, img_h=32, img_w=128):
    """Load and preprocess image: resize to [img_w, img_h], normalize to [-1,1]."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: pip install Pillow")
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_w, img_h), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    # Normalize to [-1, 1]
    arr = arr * 2.0 - 1.0
    # HWC → CHW
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return arr[np.newaxis]  # [1, 3, H, W]


class PARSeqManual:
    """Manual PARSeq forward pass for per-layer activation dumping."""

    def __init__(self, state_dict):
        self.sd = {k: v if isinstance(v, np.ndarray) else v.numpy()
                   for k, v in state_dict.items()}
        self.embed_dim = self.sd["encoder.norm.weight"].shape[0]
        self.n_enc_layers = max(int(k.split(".")[2]) for k in self.sd
                                if k.startswith("encoder.blocks.")) + 1
        self.n_heads_enc = self.embed_dim // 64
        self.head_dim = 64
        self.max_label_len = self.sd["pos_queries"].shape[1]  # 26
        self.vocab_size = self.sd["head.weight"].shape[0]  # 95
        self.activations = {}

    def save(self, name, data):
        if data.ndim == 0:
            data = data.reshape(1)
        self.activations[name] = data.astype(np.float32).copy()

    def layer_norm(self, x, w, b, eps=1e-6):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w + b

    def gelu(self, x):
        return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

    def softmax(self, x, axis=-1):
        m = x.max(axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=axis, keepdims=True)

    def mha(self, q, k, v, n_heads, mask=None):
        """Multi-head attention. q,k,v: [B, T, D]."""
        B, Tq, D = q.shape
        Tk = k.shape[1]
        hd = D // n_heads
        # Reshape to [B, n_heads, T, hd]
        q = q.reshape(B, Tq, n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, n_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, n_heads, hd).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(hd)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B,nh,Tq,Tk]
        if mask is not None:
            scores = scores + mask
        attn = self.softmax(scores, axis=-1)
        out = np.matmul(attn, v)  # [B,nh,Tq,hd]
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, D)
        return out

    def linear(self, x, w, b=None):
        y = x @ w.T
        if b is not None:
            y = y + b
        return y

    def run_encoder(self, pixel_values):
        """ViT encoder. pixel_values: [1, 3, 32, 128]."""
        # Patch embedding: Conv2d [embed_dim, 3, 4, 8] stride [4, 8]
        w = self.sd["encoder.patch_embed.proj.weight"]  # [D, 3, 4, 8]
        b = self.sd["encoder.patch_embed.proj.bias"]  # [D]
        D = w.shape[0]

        # Manual conv2d with stride
        img = pixel_values[0]  # [3, 32, 128]
        ph, pw = w.shape[2], w.shape[3]
        oh = img.shape[1] // ph  # 32/4 = 8
        ow = img.shape[2] // pw  # 128/8 = 16
        patches = np.zeros((D, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                patch = img[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw]  # [3, 4, 8]
                for d in range(D):
                    patches[d, i, j] = (w[d] * patch).sum() + b[d]

        # Flatten to [1, N, D] — row-major: oh*ow tokens
        x = patches.reshape(D, -1).T[np.newaxis]  # [1, 128, D]
        self.save("patch_embed", x[0])

        # Add position embedding
        pos = self.sd["encoder.pos_embed"]  # [1, 128, D]
        x = x + pos
        self.save("pos_embed_added", x[0])

        # Encoder blocks
        for i in range(self.n_enc_layers):
            prefix = f"encoder.blocks.{i}"

            # Pre-LN + Self-attention
            x_norm = self.layer_norm(
                x,
                self.sd[f"{prefix}.norm1.weight"],
                self.sd[f"{prefix}.norm1.bias"],
            )

            qkv_w = self.sd[f"{prefix}.attn.qkv.weight"]  # [3*D, D]
            qkv_b = self.sd[f"{prefix}.attn.qkv.bias"]  # [3*D]
            qkv = self.linear(x_norm, qkv_w, qkv_b)  # [1, 128, 3*D]
            q, k, v = np.split(qkv, 3, axis=-1)

            attn_out = self.mha(q, k, v, self.n_heads_enc)
            proj_w = self.sd[f"{prefix}.attn.proj.weight"]
            proj_b = self.sd[f"{prefix}.attn.proj.bias"]
            attn_out = self.linear(attn_out, proj_w, proj_b)
            x = x + attn_out

            # Pre-LN + FFN
            x_norm = self.layer_norm(
                x,
                self.sd[f"{prefix}.norm2.weight"],
                self.sd[f"{prefix}.norm2.bias"],
            )
            fc1_out = self.linear(
                x_norm,
                self.sd[f"{prefix}.mlp.fc1.weight"],
                self.sd[f"{prefix}.mlp.fc1.bias"],
            )
            fc1_out = self.gelu(fc1_out)
            fc2_out = self.linear(
                fc1_out,
                self.sd[f"{prefix}.mlp.fc2.weight"],
                self.sd[f"{prefix}.mlp.fc2.bias"],
            )
            x = x + fc2_out

            self.save(f"enc_layer_{i}", x[0])

        # Final LayerNorm
        x = self.layer_norm(
            x,
            self.sd["encoder.norm.weight"],
            self.sd["encoder.norm.bias"],
        )
        self.save("enc_final", x[0])
        return x  # [1, 128, D]

    def run_decoder_step(self, memory, context_tokens, step, mask=None):
        """One decoder step. Returns logits for all positions up to step+1.

        memory: [1, 128, D] encoder output
        context_tokens: [1, T, D] embedded context (BOS + previous predictions)
        step: current decode step (0-indexed)
        mask: optional attention mask [T, T]
        """
        D = self.embed_dim
        pos_queries = self.sd["pos_queries"]  # [1, 26, D]

        # Use position queries up to step+1
        p = pos_queries[:, :step+1, :]  # [1, step+1, D]
        c = context_tokens[:, :step+1, :]  # [1, step+1, D]

        prefix = "decoder.layers.0"

        # Self-attention: norm_q(p) as Q, c as K/V
        p_norm = self.layer_norm(
            p,
            self.sd[f"{prefix}.norm_q.weight"],
            self.sd[f"{prefix}.norm_q.bias"],
        )

        # Self-attention QKV from in_proj
        sa_w = self.sd[f"{prefix}.self_attn.in_proj_weight"]  # [3*D, D]
        sa_b = self.sd[f"{prefix}.self_attn.in_proj_bias"]  # [3*D]
        # Q from p_norm, K/V from c
        q = self.linear(p_norm, sa_w[:D], sa_b[:D])
        k = self.linear(c, sa_w[D:2*D], sa_b[D:2*D])
        v = self.linear(c, sa_w[2*D:], sa_b[2*D:])

        dec_heads = D // 32
        sa_out = self.mha(q, k, v, dec_heads, mask=mask)
        sa_out = self.linear(
            sa_out,
            self.sd[f"{prefix}.self_attn.out_proj.weight"],
            self.sd[f"{prefix}.self_attn.out_proj.bias"],
        )
        h = p + sa_out

        # Cross-attention: norm1(h) as Q, memory as K/V
        h_norm = self.layer_norm(
            h,
            self.sd[f"{prefix}.norm1.weight"],
            self.sd[f"{prefix}.norm1.bias"],
        )

        ca_w = self.sd[f"{prefix}.cross_attn.in_proj_weight"]  # [3*D, D]
        ca_b = self.sd[f"{prefix}.cross_attn.in_proj_bias"]
        q = self.linear(h_norm, ca_w[:D], ca_b[:D])
        k = self.linear(memory, ca_w[D:2*D], ca_b[D:2*D])
        v = self.linear(memory, ca_w[2*D:], ca_b[2*D:])

        ca_out = self.mha(q, k, v, dec_heads)
        ca_out = self.linear(
            ca_out,
            self.sd[f"{prefix}.cross_attn.out_proj.weight"],
            self.sd[f"{prefix}.cross_attn.out_proj.bias"],
        )
        h = h + ca_out

        # FFN
        h_norm = self.layer_norm(
            h,
            self.sd[f"{prefix}.norm2.weight"],
            self.sd[f"{prefix}.norm2.bias"],
        )
        ff = self.linear(
            h_norm,
            self.sd[f"{prefix}.linear1.weight"],
            self.sd[f"{prefix}.linear1.bias"],
        )
        ff = self.gelu(ff)
        ff = self.linear(
            ff,
            self.sd[f"{prefix}.linear2.weight"],
            self.sd[f"{prefix}.linear2.bias"],
        )
        h = h + ff

        # Final norm + head
        h = self.layer_norm(
            h,
            self.sd["decoder.norm.weight"],
            self.sd["decoder.norm.bias"],
        )

        logits = self.linear(
            h,
            self.sd["head.weight"],
            self.sd["head.bias"],
        )  # [1, step+1, 95]

        return logits, h

    def run_ar_decode(self, memory, max_steps=25):
        """Autoregressive decode. Returns text and per-step activations."""
        D = self.embed_dim
        embed_w = self.sd["text_embed.embedding.weight"]  # [97, D]
        scale = math.sqrt(D)

        # Start with BOS token (id=0)
        token_ids = [0]
        text = ""

        for step in range(max_steps):
            # Embed context tokens
            ctx = np.array([embed_w[tid] for tid in token_ids], dtype=np.float32)
            ctx = ctx * scale
            ctx = ctx[np.newaxis]  # [1, step+1, D]

            # Build causal mask for AR: upper triangular
            T = step + 1
            mask = np.zeros((T, T), dtype=np.float32)
            for i in range(T):
                for j in range(i + 1, T):
                    mask[i, j] = -1e9
            # Broadcast to [1, 1, T, T] for MHA
            mask = mask[np.newaxis, np.newaxis]

            logits, h = self.run_decoder_step(memory, ctx, step, mask=mask)

            # Take logit at the last position (step)
            step_logits = logits[0, step]  # [95]
            pred = int(np.argmax(step_logits))

            self.save(f"dec_step_{step}_logits", step_logits)
            self.save(f"dec_step_{step}_hidden", h[0, step])

            # Map prediction to character
            # Head output: 0..93 = chars, 94 = EOS (which is vocab index 95 - 1)
            # Actually: head output dim is 95 (indices 0..94)
            # chars are at indices 0..93, EOS is 94
            eos_id = 94  # in head output space

            if pred == eos_id:
                break
            if pred < len(CHARSET):
                text += CHARSET[pred]
                # Token ID for embedding: char index + 1 (since 0=BOS)
                token_ids.append(pred + 1)
            else:
                break

        self.save("decoded_text_len", np.array([len(text)], dtype=np.float32))
        return text


def main():
    p = argparse.ArgumentParser(description="Dump PARSeq reference activations")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-enc-layers", type=int, default=12,
                   help="Max encoder layers to dump (default: all)")
    args = p.parse_args()

    # Load checkpoint
    sys.path = [pp for pp in sys.path if '.local' not in pp]
    for mod in list(sys.modules.keys()):
        if 'torch' in mod:
            del sys.modules[mod]
    import torch

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    model = PARSeqManual(sd)
    print(f"PARSeq: embed_dim={model.embed_dim}, "
          f"enc_layers={model.n_enc_layers}, heads={model.n_heads_enc}")

    # Preprocess image
    print(f"Loading image: {args.image}")
    pixel_values = preprocess_image(args.image)
    model.save("input_image", pixel_values[0])
    print(f"  Image shape: {pixel_values.shape}")

    # Run encoder
    print("Running encoder...")
    memory = model.run_encoder(pixel_values)
    print(f"  Encoder output: {memory.shape}")

    # Run decoder
    print("Running AR decode...")
    text = model.run_ar_decode(memory)
    print(f"  Decoded text: '{text}'")

    # Write GGUF
    print(f"\nWriting reference to {args.output}")
    writer = gguf.GGUFWriter(str(args.output), arch="parseq_ref")
    writer.add_uint32("parseq.embed_dim", model.embed_dim)
    writer.add_uint32("parseq.n_enc_layers", model.n_enc_layers)

    for name, data in sorted(model.activations.items()):
        writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    n_act = len(model.activations)
    out_size = Path(args.output).stat().st_size
    print(f"Wrote {n_act} activations ({out_size / 1024 / 1024:.1f} MB)")
    print(f"Decoded text: '{text}'")


if __name__ == "__main__":
    main()
