#!/usr/bin/env python3
"""Dump SwinIR-light per-stage reference activations for parity testing.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_swinir_reference.py \
        --model 001_classicalSR_DIV2K_s48w8_SwinIR-S_x4.pth \
        --output /tmp/swinir-ref.gguf [--size 64]

Pure-numpy reimplementation of SwinIR-light forward pass with intermediate
captures at each RSTB output and the final output.
"""

import argparse
import struct

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def conv2d(x, weight, bias=None, padding=0):
    """Vectorized Conv2D: x=[B,IC,H,W], weight=[OC,IC,KH,KW]."""
    B, IC, H, W = x.shape
    OC, IC_W, KH, KW = weight.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, PH, PW = x.shape
    OH, OW = PH - KH + 1, PW - KW + 1
    s = x.strides
    col = np.lib.stride_tricks.as_strided(
        x, shape=(B, IC, KH, KW, OH, OW),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]))
    col = col.reshape(B, IC * KH * KW, OH * OW)
    w = weight.reshape(OC, IC_W * KH * KW)
    out = np.einsum('ij,bjk->bik', w, col).reshape(B, OC, OH, OW)
    if bias is not None:
        out += bias.reshape(1, OC, 1, 1)
    return out


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over last dim: x=[..., C]."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * weight + bias


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def pixel_shuffle(x, scale):
    """PixelShuffle: [B, C*s*s, H, W] -> [B, C, H*s, W*s]."""
    B, C_total, H, W = x.shape
    C = C_total // (scale * scale)
    x = x.reshape(B, C, scale, scale, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # B, C, H, s, W, s
    x = x.reshape(B, C, H * scale, W * scale)
    return x


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Window attention helpers
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    """Partition into windows: [B, H, W, C] -> [B*nW, ws, ws, C]."""
    B, H, W, C = x.shape
    ws = window_size
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # B, nH, nW, ws, ws, C
    x = x.reshape(-1, ws, ws, C)
    return x


def window_reverse(windows, window_size, H, W):
    """Reverse window_partition: [B*nW, ws, ws, C] -> [B, H, W, C]."""
    ws = window_size
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    x = windows.reshape(B, nH, nW, ws, ws, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # B, nH, ws, nW, ws, C
    x = x.reshape(B, H, W, -1)
    return x


def cyclic_shift(x, shift_h, shift_w):
    """Roll along H and W dims: x=[B, H, W, C]."""
    x = np.roll(x, shift_h, axis=1)
    x = np.roll(x, shift_w, axis=2)
    return x


def window_mhsa(x, qkv_w, qkv_b, proj_w, proj_b, rpb_table, rpb_index,
                 n_heads, window_size, shift, attn_mask=None):
    """Window multi-head self-attention.

    x: [B, H*W, C]
    Returns: [B, H*W, C]
    """
    B, L, C = x.shape
    H = W = int(np.sqrt(L))
    ws = window_size
    head_dim = C // n_heads

    # Reshape to spatial
    x_2d = x.reshape(B, H, W, C)

    # Cyclic shift for odd blocks
    if shift:
        shift_size = ws // 2
        x_2d = cyclic_shift(x_2d, -shift_size, -shift_size)

    # Partition into windows
    x_win = window_partition(x_2d, ws)  # [B*nW, ws, ws, C]
    nW_total = x_win.shape[0]
    x_win = x_win.reshape(nW_total, ws * ws, C)  # [B*nW, ws*ws, C]

    # QKV projection
    qkv = x_win @ qkv_w.T + qkv_b  # [B*nW, ws*ws, 3*C]
    qkv = qkv.reshape(nW_total, ws * ws, 3, n_heads, head_dim)
    qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, B*nW, n_heads, ws*ws, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Attention scores
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B*nW, n_heads, ws*ws, ws*ws]

    # Relative position bias
    # rpb_index: [ws*ws, ws*ws], rpb_table: [num_entries, n_heads]
    rpb_idx = rpb_index.astype(int)  # [ws*ws, ws*ws]
    rpb = rpb_table[rpb_idx]  # [ws*ws, ws*ws, n_heads]
    rpb = rpb.transpose(2, 0, 1)  # [n_heads, ws*ws, ws*ws]
    attn = attn + rpb[np.newaxis, :, :, :]  # broadcast over batch

    # Apply attention mask for shifted windows
    if shift and attn_mask is not None:
        # attn_mask: [nW, ws*ws, ws*ws]
        nW = attn_mask.shape[0]
        attn = attn.reshape(-1, nW, n_heads, ws * ws, ws * ws)
        attn = attn + attn_mask[np.newaxis, :, np.newaxis, :, :]
        attn = attn.reshape(-1, n_heads, ws * ws, ws * ws)

    attn = softmax(attn, axis=-1)

    # Apply attention to values
    out = attn @ v  # [B*nW, n_heads, ws*ws, head_dim]
    out = out.transpose(0, 2, 1, 3).reshape(nW_total, ws * ws, C)

    # Output projection
    out = out @ proj_w.T + proj_b

    # Reverse windows
    out = out.reshape(nW_total, ws, ws, C)
    out = window_reverse(out, ws, H, W)  # [B, H, W, C]

    # Reverse cyclic shift
    if shift:
        out = cyclic_shift(out, shift_size, shift_size)

    out = out.reshape(B, H * W, C)
    return out


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def get(state, key):
    return state[key].astype(np.float32) if key in state else None


def swinir_forward(state, x, n_rstb=4, n_blocks=6, n_heads=6, window_size=8):
    """SwinIR-light forward pass with intermediates."""
    intermediates = {"input": x[0].copy()}

    # 1. conv_first
    fea = conv2d(x, get(state, "conv_first.weight"),
                 get(state, "conv_first.bias"), padding=1)
    conv_first_out = fea.copy()
    intermediates["conv_first"] = fea[0].copy()

    B, C, H, W = fea.shape

    # 2. patch_embed: [B,C,H,W] -> [B, H*W, C] + LayerNorm
    fea_seq = fea.transpose(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
    fea_seq = layer_norm(fea_seq,
                         get(state, "patch_embed.norm.weight"),
                         get(state, "patch_embed.norm.bias"))

    # 3. RSTB layers
    for i in range(n_rstb):
        rstb_input = fea_seq.copy()

        for j in range(n_blocks):
            sp = f"layers.{i}.residual_group.blocks.{j}"

            # LN1 -> window MHSA -> residual
            normed = layer_norm(fea_seq,
                                get(state, f"{sp}.norm1.weight"),
                                get(state, f"{sp}.norm1.bias"))

            shift = (j % 2 == 1)
            mask_key = f"{sp}.attn_mask"
            attn_mask = get(state, mask_key)

            attn_out = window_mhsa(
                normed,
                get(state, f"{sp}.attn.qkv.weight"),
                get(state, f"{sp}.attn.qkv.bias"),
                get(state, f"{sp}.attn.proj.weight"),
                get(state, f"{sp}.attn.proj.bias"),
                get(state, f"{sp}.attn.relative_position_bias_table"),
                get(state, f"{sp}.attn.relative_position_index"),
                n_heads, window_size, shift, attn_mask)

            fea_seq = fea_seq + attn_out

            # LN2 -> MLP -> residual
            normed2 = layer_norm(fea_seq,
                                 get(state, f"{sp}.norm2.weight"),
                                 get(state, f"{sp}.norm2.bias"))

            mlp_up = normed2 @ get(state, f"{sp}.mlp.fc1.weight").T + get(state, f"{sp}.mlp.fc1.bias")
            mlp_up = gelu(mlp_up)
            mlp_down = mlp_up @ get(state, f"{sp}.mlp.fc2.weight").T + get(state, f"{sp}.mlp.fc2.bias")

            fea_seq = fea_seq + mlp_down

        # patch_unembed: [B, H*W, C] -> [B, C, H, W]
        fea_spatial = fea_seq.reshape(B, H, W, C).transpose(0, 3, 1, 2)

        # Conv3x3 + residual to RSTB input
        conv_out = conv2d(fea_spatial,
                          get(state, f"layers.{i}.conv.weight"),
                          get(state, f"layers.{i}.conv.bias"), padding=1)

        # Add residual (rstb_input in spatial form)
        rstb_input_spatial = rstb_input.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        fea_spatial = conv_out + rstb_input_spatial

        intermediates[f"rstb_{i}"] = fea_spatial[0].copy()

        # patch_embed for next RSTB (reshape + LN)
        fea_seq = fea_spatial.transpose(0, 2, 3, 1).reshape(B, H * W, C)
        fea_seq = layer_norm(fea_seq,
                             get(state, "patch_embed.norm.weight"),
                             get(state, "patch_embed.norm.bias"))

    # 4. Final norm
    fea_seq = layer_norm(fea_seq,
                         get(state, "norm.weight"),
                         get(state, "norm.bias"))

    # 5. patch_unembed
    fea_spatial = fea_seq.reshape(B, H, W, C).transpose(0, 3, 1, 2)

    # 6. conv_after_body + global residual
    fea_spatial = conv2d(fea_spatial,
                         get(state, "conv_after_body.weight"),
                         get(state, "conv_after_body.bias"), padding=1)
    fea_spatial = fea_spatial + conv_first_out

    # 7. Upsample: Conv2d(60 -> 3*scale^2) + PixelShuffle
    up = conv2d(fea_spatial,
                get(state, "upsample.0.weight"),
                get(state, "upsample.0.bias"), padding=1)
    upsample_out_ch = get(state, "upsample.0.weight").shape[0]
    scale = int(np.sqrt(upsample_out_ch / 3))
    out = pixel_shuffle(up, scale)

    intermediates["output"] = out[0].copy()

    return out, intermediates


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------

def write_gguf(path, tensors):
    MAGIC = 0x46554747; VERSION = 3; TYPE_STRING = 8; TYPE_F32 = 0
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC)); f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING)); ws(f, "swinir_ref")
        offset = 0
        for name, data in tensor_list:
            ws(f, name); f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", TYPE_F32)); f.write(struct.pack("<Q", offset))
            nbytes_f32 = data.size * 4  # always F32
            offset += nbytes_f32; offset = (offset + 31) & ~31
        pos = f.tell(); aligned = (pos + 31) & ~31; f.write(b"\x00" * (aligned - pos))
        for name, data in tensor_list:
            f32 = data.astype(np.float32)
            f.write(f32.tobytes())
            pad = ((f32.nbytes + 31) & ~31) - f32.nbytes
            if pad > 0: f.write(b"\x00" * pad)
    print(f"Written {path}: {len(tensor_list)} tensors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=64, help="Input size (default 64)")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]

    # Convert all tensors to numpy
    sd = {}
    for k, v in state.items():
        sd[k] = v.numpy()

    # Detect config
    embed_dim = sd["conv_first.weight"].shape[0]
    n_rstb = 0
    while f"layers.{n_rstb}.residual_group.blocks.0.attn.qkv.weight" in sd:
        n_rstb += 1
    n_blocks = 0
    while f"layers.0.residual_group.blocks.{n_blocks}.attn.qkv.weight" in sd:
        n_blocks += 1
    n_heads = sd["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[1]

    print(f"SwinIR-light: embed_dim={embed_dim}, n_rstb={n_rstb}, n_blocks={n_blocks}, "
          f"n_heads={n_heads}")

    np.random.seed(42)
    inp = np.random.rand(1, 3, args.size, args.size).astype(np.float32)

    out, intermediates = swinir_forward(sd, inp, n_rstb, n_blocks, n_heads)

    print("\nIntermediate activations:")
    for name, data in intermediates.items():
        print(f"  {name:20s}  shape={str(list(data.shape)):30s}  mean={data.mean():.6f}")

    write_gguf(args.output, intermediates)


if __name__ == "__main__":
    main()
