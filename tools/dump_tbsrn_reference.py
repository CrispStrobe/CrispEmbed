#!/usr/bin/env python3
"""Dump TBSRN per-stage reference activations for parity testing.

Loads PaddleOCR TBSRN weights (.pdparams), runs forward pass in pure numpy
on a deterministic 16×64 input, captures intermediate activations at each
stage. Writes to GGUF for crispembed_diff comparison.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_tbsrn_reference.py \
        --model sr_telescope_train/best_accuracy.pdparams \
        --output /tmp/tbsrn-ref.gguf
"""

import argparse
import math
import pickle
import struct
import sys

import numpy as np


# ── Pure numpy ops ──────────────────────────────────────────────────────

def conv2d(x, weight, bias, stride=1, padding=0, groups=1):
    """Conv2D forward (im2col, vectorized): x=[B,IC,H,W], weight=[OC,IC/G,KH,KW]."""
    B, IC, H, W = x.shape
    OC, IC_G, KH, KW = weight.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, PH, PW = x.shape
    OH = (PH - KH) // stride + 1
    OW = (PW - KW) // stride + 1
    s = x.strides
    col = np.lib.stride_tricks.as_strided(
        x, shape=(B, IC, KH, KW, OH, OW),
        strides=(s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride))
    col = col.reshape(B, IC * KH * KW, OH * OW)
    if groups == 1:
        w = weight.reshape(OC, IC * KH * KW)
        out = np.einsum('ij,bjk->bik', w, col)
    else:
        oc_pg = OC // groups
        out = np.zeros((B, OC, OH * OW), dtype=np.float32)
        for g in range(groups):
            w_g = weight[g * oc_pg:(g + 1) * oc_pg].reshape(oc_pg, IC_G * KH * KW)
            c_g = col[:, g * IC_G * KH * KW:(g + 1) * IC_G * KH * KW]
            out[:, g * oc_pg:(g + 1) * oc_pg] = np.einsum('ij,bjk->bik', w_g, c_g)
    out = out.reshape(B, OC, OH, OW)
    if bias is not None:
        out += bias.reshape(1, OC, 1, 1)
    return out


def batchnorm2d(x, weight, bias, running_mean, running_var, eps=1e-5):
    """BatchNorm2D eval mode: x=[B,C,H,W]."""
    # Normalize using running stats
    mean = running_mean.reshape(1, -1, 1, 1)
    var = running_var.reshape(1, -1, 1, 1)
    w = weight.reshape(1, -1, 1, 1)
    b = bias.reshape(1, -1, 1, 1)
    return w * (x - mean) / np.sqrt(var + eps) + b


def mish(x):
    """mish activation: x * tanh(softplus(x))."""
    return x * np.tanh(np.log1p(np.exp(np.clip(x, -20, 20))))


def prelu(x, weight):
    """PReLU: max(0,x) + weight*min(0,x). weight shape (1,) or (C,)."""
    w = weight.reshape(1, -1, 1, 1) if weight.ndim == 1 and weight.size > 1 else weight
    return np.maximum(x, 0) + w * np.minimum(x, 0)


def pixel_shuffle(x, r):
    """PixelShuffle: [B, C*r*r, H, W] → [B, C, H*r, W*r]."""
    B, C_in, H, W = x.shape
    C_out = C_in // (r * r)
    x = x.reshape(B, C_out, r, r, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # B, C_out, H, r, W, r
    return x.reshape(B, C_out, H * r, W * r)


def layernorm(x, weight, bias, eps=1e-6):
    """LayerNorm on last dim: x=[B, T, D]."""
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    return weight * (x - mean) / (std + eps) + bias


def linear(x, weight, bias):
    """Linear: Paddle stores weight=[in, out], x=[..., in] → [..., out]."""
    return x @ weight + bias


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def positionalencoding2d(d_model, height, width):
    """2D sinusoidal positional encoding: [d_model, height, width]."""
    pe = np.zeros((d_model, height, width), dtype=np.float32)
    d_half = d_model // 2
    div_term = np.exp(np.arange(0, d_half, 2, dtype=np.float32) * -(math.log(10000.0) / d_half))
    pos_w = np.arange(width, dtype=np.float32)[:, None]   # [W, 1]
    pos_h = np.arange(height, dtype=np.float32)[:, None]   # [H, 1]
    # Width encoding: first d_half channels
    sin_w = np.sin(pos_w * div_term)  # [W, d_half/2]
    cos_w = np.cos(pos_w * div_term)  # [W, d_half/2]
    pe[0:d_half:2, :, :] = np.broadcast_to(sin_w.T[:, None, :], (d_half // 2, height, width))
    pe[1:d_half:2, :, :] = np.broadcast_to(cos_w.T[:, None, :], (d_half // 2, height, width))
    # Height encoding: second d_half channels
    sin_h = np.sin(pos_h * div_term)  # [H, d_half/2]
    cos_h = np.cos(pos_h * div_term)  # [H, d_half/2]
    pe[d_half::2, :, :]     = np.broadcast_to(sin_h.T[:, :, None], (d_half // 2, height, width))
    pe[d_half+1::2, :, :]   = np.broadcast_to(cos_h.T[:, :, None], (d_half // 2, height, width))
    return pe


def multi_head_attention(q, k, v, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo, n_heads):
    """Multi-head self-attention: q=k=v=[B, T, D]."""
    B, T, D = q.shape
    d_k = D // n_heads

    Q = linear(q, Wq, Bq)  # [B, T, D]
    K = linear(k, Wk, Bk)
    V = linear(v, Wv, Bv)

    Q = Q.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)  # [B, h, T, d_k]
    K = K.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(d_k)  # [B, h, T, T]
    attn = softmax(scores, axis=-1)
    out = attn @ V  # [B, h, T, d_k]
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)  # [B, T, D]
    return linear(out, Wo, Bo)


# ── Weight loader ───────────────────────────────────────────────────────

def load_pdparams(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get(state, key):
    if key in state:
        v = state[key]
        return v.astype(np.float32) if isinstance(v, np.ndarray) else v
    raise KeyError(f"Missing key: {key}")


# ── TBSRN forward with intermediates ────────────────────────────────────

def tbsrn_forward(state, x, prefix="transform."):
    """Run TBSRN forward pass, returning (output, dict_of_intermediates)."""
    intermediates = {}
    intermediates["input"] = x[0].copy()  # [3, H, W]

    p = prefix
    srb_nums = 5
    H_in, W_in = x.shape[2], x.shape[3]  # 16, 64

    # block1: Conv9(3→64) + PReLU
    x = conv2d(x, get(state, f"{p}block1.0.weight"), get(state, f"{p}block1.0.bias"), padding=4)
    x = prelu(x, get(state, f"{p}block1.1._weight"))
    block1_out = x.copy()
    intermediates["block1"] = x[0].copy()

    # block2..block6: RecurrentResidualBlock
    for i in range(srb_nums):
        blk = i + 2
        bp = f"{p}block{blk}"

        residual = conv2d(x, get(state, f"{bp}.conv1.weight"), get(state, f"{bp}.conv1.bias"), padding=1)
        residual = batchnorm2d(residual,
                               get(state, f"{bp}.bn1.weight"), get(state, f"{bp}.bn1.bias"),
                               get(state, f"{bp}.bn1._mean"), get(state, f"{bp}.bn1._variance"))
        residual = mish(residual)
        residual = conv2d(residual, get(state, f"{bp}.conv2.weight"), get(state, f"{bp}.conv2.bias"), padding=1)
        residual = batchnorm2d(residual,
                               get(state, f"{bp}.bn2.weight"), get(state, f"{bp}.bn2.bias"),
                               get(state, f"{bp}.bn2._mean"), get(state, f"{bp}.bn2._variance"))

        intermediates[f"srb{i}_pre_fe"] = residual[0].copy()

        # FeatureEnhancer
        B, C, H, W = residual.shape
        fe = f"{bp}.feature_enhancer"
        feat = residual.reshape(B, C, H * W)  # [B, 64, 1024]

        # Concat with PE2D → [B, 128, 1024]
        pe = positionalencoding2d(64, H, W).reshape(64, H * W)  # [64, 1024]
        pe_batch = np.broadcast_to(pe[None, :, :], (B, 64, H * W))
        feat = np.concatenate([feat, pe_batch], axis=1)  # [B, 128, 1024]

        # Transpose → [B, 1024, 128]
        feat = feat.transpose(0, 2, 1)
        origin = feat.copy()

        # MHA self-attention
        mha_out = multi_head_attention(
            feat, feat, feat,
            get(state, f"{fe}.multihead.linears.0.weight"), get(state, f"{fe}.multihead.linears.0.bias"),
            get(state, f"{fe}.multihead.linears.1.weight"), get(state, f"{fe}.multihead.linears.1.bias"),
            get(state, f"{fe}.multihead.linears.2.weight"), get(state, f"{fe}.multihead.linears.2.bias"),
            get(state, f"{fe}.multihead.linears.3.weight"), get(state, f"{fe}.multihead.linears.3.bias"),
            n_heads=4)
        feat = layernorm(origin + mha_out,
                         get(state, f"{fe}.mul_layernorm1.a_2"),
                         get(state, f"{fe}.mul_layernorm1.b_2"))
        origin2 = feat.copy()

        # FFN
        ffn_out = linear(feat, get(state, f"{fe}.pff.w_1.weight"), get(state, f"{fe}.pff.w_1.bias"))
        ffn_out = np.maximum(ffn_out, 0)  # ReLU
        ffn_out = linear(ffn_out, get(state, f"{fe}.pff.w_2.weight"), get(state, f"{fe}.pff.w_2.bias"))
        feat = layernorm(origin2 + ffn_out,
                         get(state, f"{fe}.mul_layernorm3.a_2"),
                         get(state, f"{fe}.mul_layernorm3.b_2"))

        # Output linear (128→64)
        feat = linear(feat, get(state, f"{fe}.linear.weight"), get(state, f"{fe}.linear.bias"))
        # [B, 1024, 64] → transpose → [B, 64, 1024] → reshape → [B, 64, H, W]
        residual = feat.transpose(0, 2, 1).reshape(B, C, H, W)

        x = x + residual
        intermediates[f"srb{i}"] = x[0].copy()

    # block7: Conv3(64→64) + BN
    blk7 = srb_nums + 2  # = 7
    final = conv2d(block1_out + x,
                   get(state, f"{p}block{blk7}.0.weight"),
                   get(state, f"{p}block{blk7}.0.bias"), padding=1)
    # Note: the original code does (block1 + block6) → block7
    # But looking at TBSRN source: block[srb_nums+3]( block["1"] + block[srb_nums+2] )
    # block7 = block[srb_nums+2], so its input is the output of block6 (= our x after the loop)
    # Then block8 = block[srb_nums+3], its input is (block1 + block7_output)
    # Wait, re-reading the source more carefully:
    #   block[str(srb_nums + 2)] runs on block[str(srb_nums + 1)]
    #   block[str(srb_nums + 3)] runs on (block["1"] + block[str(srb_nums + 2)])
    # So block7 runs on block6 output (= x), THEN block8 runs on (block1 + block7_output)

    # Redo: block7 input is x (block6 output), not block1+x
    final = conv2d(x, get(state, f"{p}block{blk7}.0.weight"),
                   get(state, f"{p}block{blk7}.0.bias"), padding=1)
    final = batchnorm2d(final,
                        get(state, f"{p}block{blk7}.1.weight"),
                        get(state, f"{p}block{blk7}.1.bias"),
                        get(state, f"{p}block{blk7}.1._mean"),
                        get(state, f"{p}block{blk7}.1._variance"))
    intermediates["block7"] = final[0].copy()

    # block8 input: block1 + block7
    upsample_input = block1_out + final

    # UpsampleBlock: Conv(64→256) + PixelShuffle(2) + mish
    blk8 = srb_nums + 3  # = 8
    up = conv2d(upsample_input,
                get(state, f"{p}block{blk8}.0.conv.weight"),
                get(state, f"{p}block{blk8}.0.conv.bias"), padding=1)
    up = pixel_shuffle(up, 2)
    up = mish(up)
    intermediates["upsample"] = up[0].copy()

    # Final conv: Conv(64→3, k=9, p=4)
    out = conv2d(up, get(state, f"{p}block{blk8}.1.weight"),
                 get(state, f"{p}block{blk8}.1.bias"), padding=4)
    out = np.tanh(out)
    intermediates["output"] = out[0].copy()

    return out, intermediates


# ── GGUF writer ─────────────────────────────────────────────────────────

def write_gguf(path, tensors):
    MAGIC = 0x46554747
    VERSION = 3
    TYPE_STRING = 8
    TYPE_F32 = 0

    def write_string(f, s):
        b = s.encode("utf-8")
        f.write(struct.pack("<Q", len(b)))
        f.write(b)

    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list)))
        f.write(struct.pack("<Q", 1))

        write_string(f, "general.architecture")
        f.write(struct.pack("<I", TYPE_STRING))
        write_string(f, "tbsrn_ref")

        offset = 0
        for name, data in tensor_list:
            write_string(f, name)
            f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", TYPE_F32))
            f.write(struct.pack("<Q", offset))
            nbytes = data.nbytes
            offset += nbytes
            offset = (offset + 31) & ~31

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for name, data in tensor_list:
            f.write(data.astype(np.float32).tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0:
                f.write(b"\x00" * pad)

    print(f"Written {path}: {len(tensor_list)} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="PaddleOCR .pdparams file")
    parser.add_argument("--output", required=True, help="Output reference GGUF")
    args = parser.parse_args()

    state = load_pdparams(args.model)
    print(f"Loaded: {sum(v.size for v in state.values() if isinstance(v, np.ndarray)):,} total params")

    # Deterministic input: 16×64 (TBSRN native LR size)
    np.random.seed(42)
    inp = np.random.rand(1, 3, 16, 64).astype(np.float32)

    out, intermediates = tbsrn_forward(state, inp)

    print(f"\nIntermediate activations:")
    for name, data in intermediates.items():
        shp = str(list(data.shape))
        print(f"  {name:20s}  shape={shp:30s}  mean={data.mean():.6f}  std={data.std():.6f}")

    write_gguf(args.output, intermediates)


if __name__ == "__main__":
    main()
