#!/usr/bin/env python3
"""Dump SCUNet per-stage reference activations for parity testing.

Usage:
    python tools/dump_scunet_reference.py \
        --model /mnt/storage/models/scunet_color_real_psnr.pth \
        --output /tmp/scunet-ref.gguf [--width 64] [--height 64]
"""
import argparse, sys, gc
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# ── Minimal SCUNet reimplementation ──

def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size * win_size, C)
    return windows

def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def wmsa(x, qkv_w, qkv_b, proj_w, proj_b, rpb, n_heads, win_size, shift):
    B, C, H, W = x.shape
    head_dim = C // n_heads

    # Pad to multiple of win_size
    pad_h = (win_size - H % win_size) % win_size
    pad_w = (win_size - W % win_size) % win_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    _, _, Hp, Wp = x.shape

    # Shift
    if shift:
        x = torch.roll(x, shifts=(-win_size // 2, -win_size // 2), dims=(2, 3))

    x = x.permute(0, 2, 3, 1)  # B, H, W, C
    windows = window_partition(x, win_size)  # (nW*B, win*win, C)
    nW = windows.shape[0]

    # QKV
    qkv = F.linear(windows, qkv_w, qkv_b)  # (nW*B, N, 3C)
    qkv = qkv.reshape(nW, win_size * win_size, 3, n_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, nW*B, heads, N, head_dim
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Attention
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale

    # Relative position bias
    # Build position index
    coords_h = torch.arange(win_size)
    coords_w = torch.arange(win_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, win, win
    coords_flat = coords.view(2, -1)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, N, N
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += win_size - 1
    rel[:, :, 1] += win_size - 1
    rel[:, :, 0] *= 2 * win_size - 1
    rel_idx = rel.sum(-1)  # N, N

    # rpb shape: [1, 2*win-1, 2*win-1] → flatten to [(2w-1)^2]
    rpb_flat = rpb.reshape(-1)
    bias = rpb_flat[rel_idx.view(-1)].view(win_size * win_size, win_size * win_size)
    attn = attn + bias.unsqueeze(0).unsqueeze(0)

    # Shift mask
    if shift:
        # Create attention mask for shifted windows
        mask = torch.zeros(1, Hp, Wp, 1)
        h_slices = (slice(0, -win_size), slice(-win_size, -win_size // 2), slice(-win_size // 2, None))
        w_slices = (slice(0, -win_size), slice(-win_size, -win_size // 2), slice(-win_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(mask, win_size)  # nW, N, 1
        mask_windows = mask_windows.squeeze(-1)  # nW, N
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, N, N
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn = attn + attn_mask.unsqueeze(1)  # broadcast over heads

    attn = F.softmax(attn, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(nW, win_size * win_size, C)
    out = F.linear(out, proj_w, proj_b)

    out = window_reverse(out, win_size, Hp, Wp)  # B, H, W, C
    if shift:
        out = torch.roll(out, shifts=(win_size // 2, win_size // 2), dims=(1, 2))
    if pad_h > 0 or pad_w > 0:
        out = out[:, :H, :W, :]
    return out.permute(0, 3, 1, 2)  # B, C, H, W


def swin_block(x, sd, prefix, n_heads, win_size, shift):
    B, C, H, W = x.shape
    # LN1 + WMSA
    y = x.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
    y = F.layer_norm(y, [C], sd[f'{prefix}.ln1.weight'], sd[f'{prefix}.ln1.bias'])
    y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

    y = wmsa(y, sd[f'{prefix}.msa.embedding_layer.weight'],
             sd[f'{prefix}.msa.embedding_layer.bias'],
             sd[f'{prefix}.msa.linear.weight'],
             sd[f'{prefix}.msa.linear.bias'],
             sd[f'{prefix}.msa.relative_position_params'],
             n_heads, win_size, shift)
    x = x + y

    # LN2 + MLP
    y = x.permute(0, 2, 3, 1).reshape(-1, C)
    y = F.layer_norm(y, [C], sd[f'{prefix}.ln2.weight'], sd[f'{prefix}.ln2.bias'])
    y = F.linear(y, sd[f'{prefix}.mlp.0.weight'], sd[f'{prefix}.mlp.0.bias'])
    y = F.gelu(y)
    y = F.linear(y, sd[f'{prefix}.mlp.2.weight'], sd[f'{prefix}.mlp.2.bias'])
    y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
    x = x + y
    return x


def conv_trans_block(x, sd, prefix, n_heads, win_size, shift):
    B, C, H, W = x.shape
    half = C // 2

    # Split via 1x1 conv
    y = F.conv2d(x, sd[f'{prefix}.conv1_1.weight'], sd[f'{prefix}.conv1_1.bias'])

    # Conv branch: first half
    conv_in = y[:, :half]
    conv_out = F.conv2d(conv_in, sd[f'{prefix}.conv_block.0.weight'], padding=1)
    conv_out = F.relu(conv_out)
    conv_out = F.conv2d(conv_out, sd[f'{prefix}.conv_block.2.weight'], padding=1)
    conv_out = conv_in + conv_out  # residual

    # Trans branch: second half
    trans_in = y[:, half:]
    trans_out = swin_block(trans_in, sd, f'{prefix}.trans_block', n_heads, win_size, shift)

    # Fuse
    y = torch.cat([conv_out, trans_out], dim=1)
    y = F.conv2d(y, sd[f'{prefix}.conv1_2.weight'], sd[f'{prefix}.conv1_2.bias'])
    return x + y  # residual


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    args = p.parse_args()

    sd = torch.load(args.model, map_location='cpu', weights_only=False)

    W, H = args.width, args.height
    np.random.seed(42)
    inp = np.random.rand(H, W, 3).astype(np.float32)
    x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
    print(f"Input: {W}x{H}")

    stages = {}
    stages["input"] = x.squeeze(0).numpy().copy()
    win = 8
    head_dim = 32

    # Head
    x = F.conv2d(x, sd['m_head.0.weight'], sd.get('m_head.0.bias'), padding=1)
    stages["head"] = x.squeeze(0).detach().numpy().copy()
    x1 = x.clone()

    # Encoder stages
    configs = [
        ("m_down1", 64, 4),
        ("m_down2", 128, 4),
        ("m_down3", 256, 4),
    ]
    skips = [x1]
    for stage_name, ch, n_blocks in configs:
        n_heads = (ch // 2) // head_dim
        for i in range(n_blocks):
            shift = (i % 2 == 1)
            x = conv_trans_block(x, sd, f'{stage_name}.{i}', n_heads, win, shift)
        # Downsample
        x = F.conv2d(x, sd[f'{stage_name}.{n_blocks}.weight'],
                     sd.get(f'{stage_name}.{n_blocks}.bias'), stride=2)
        stages[stage_name] = x.squeeze(0).detach().numpy().copy()
        skips.append(x.clone())
        print(f"  {stage_name}: {list(x.shape[1:])}")

    # Body
    n_heads = 256 // head_dim
    for i in range(4):
        shift = (i % 2 == 1)
        x = conv_trans_block(x, sd, f'm_body.{i}', n_heads, win, shift)
    stages["body"] = x.squeeze(0).detach().numpy().copy()
    print(f"  body: {list(x.shape[1:])}")

    # Decoder: skip connections are added BEFORE the upsample stage.
    # SCUNet forward: x = m_up3(x + x4), x = m_up2(x + x3), x = m_up1(x + x2)
    # where m_up{N} = [ConvTranspose2d, block, block, block, block]
    up_configs = [
        ("m_up3", 256, 4, skips[3]),  # skip = m_down3 output (512ch /8)
        ("m_up2", 128, 4, skips[2]),  # skip = m_down2 output (256ch /4)
        ("m_up1", 64, 4, skips[1]),   # skip = m_down1 output (128ch /2)
    ]
    for stage_name, ch, n_blocks, skip in up_configs:
        x = x + skip  # skip connection (same shape)
        # ConvTranspose2d upsample
        x = F.conv_transpose2d(x, sd[f'{stage_name}.0.weight'],
                               sd.get(f'{stage_name}.0.bias'), stride=2)
        n_heads = max(1, (ch // 2) // head_dim)
        for i in range(n_blocks):
            shift = (i % 2 == 1)
            x = conv_trans_block(x, sd, f'{stage_name}.{i+1}', n_heads, win, shift)
        stages[stage_name] = x.squeeze(0).detach().numpy().copy()
        print(f"  {stage_name}: {list(x.shape[1:])}")

    # Final skip + tail
    x = x + skips[0]  # head output (64ch full res)
    x = F.conv2d(x, sd['m_tail.0.weight'], sd.get('m_tail.0.bias'), padding=1)
    stages["output"] = x.squeeze(0).detach().numpy().copy()
    print(f"  output: {list(x.shape[1:])}, range=[{x.min():.4f}, {x.max():.4f}]")

    # Write GGUF
    writer = gguf.GGUFWriter(args.output, "scunet-reference")
    writer.add_uint32("scunet.ref.width", W)
    writer.add_uint32("scunet.ref.height", H)
    for name, arr in stages.items():
        writer.add_tensor(name, arr.astype(np.float32),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nReference: {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
