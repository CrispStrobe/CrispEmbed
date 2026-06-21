#!/usr/bin/env python
"""Self-consistent HAT reference for crispembed-diff (test-hat-diff), built from
the GGUF weights instead of the original .pth — the granite-llm-ref pattern.

Loads the gguf-dequantized weights into the torch HAT arch (reversing
convert-hat-to-gguf.py's name map), runs the forward, and writes a -ref.gguf with
input + per-RHAG stages + output. Proves the C++ runtime == torch on identical
weights (output cos ~0.99997). Upload the result to the model's HF repo.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_hat_reference_from_gguf.py \
        --gguf hat-sr-x4-f16.gguf --arch hat_arch.py \
        --output hat-ref.gguf --size 32
(arch = HAT architecture from github.com/XPixelGroup/HAT hat/archs/hat_arch.py)
"""
import argparse, sys, struct, math, types
import numpy as np
import torch, torch.nn as nn
import gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="converted HAT gguf (the C++ runtime model)")
    ap.add_argument("--arch", required=True, help="hat_arch.py from the HAT repo")
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=32, help="input HxW (multiple of window_size; small = faster scalar C++ diff)")
    args = ap.parse_args()

    # basicsr stub so hat_arch.py imports standalone
    class _ArchUtil:
        @staticmethod
        def to_2tuple(x): return (x, x) if isinstance(x, int) else x
        @staticmethod
        def trunc_normal_(t, std=0.02): nn.init.trunc_normal_(t, std=std)
    bs = types.ModuleType("basicsr"); bs.utils = types.ModuleType("basicsr.utils")
    bs.utils.registry = types.ModuleType("basicsr.utils.registry")
    bs.archs = types.ModuleType("basicsr.archs"); bs.archs.arch_util = _ArchUtil
    class _Reg:
        def register(self): return lambda c: c
    bs.utils.registry.ARCH_REGISTRY = _Reg()
    for n, mod in [("basicsr", bs), ("basicsr.utils", bs.utils),
                   ("basicsr.utils.registry", bs.utils.registry),
                   ("basicsr.archs", bs.archs), ("basicsr.archs.arch_util", bs.archs.arch_util)]:
        sys.modules[n] = mod
    ns = {}
    exec(open(args.arch).read(), ns)
    HAT = ns["HAT"]

    # gguf weights (dequantized, torch-shaped) → torch state_dict (reverse the converter map)
    r = gguf.GGUFReader(args.gguf)
    gg = {t.name: np.array(t.data, dtype=np.float32).reshape([int(s) for s in t.shape]) for t in r.tensors}
    state = {}
    def m(g, t):
        if g in gg: state[t] = torch.from_numpy(gg[g].copy())
    m("conv_first.weight", "conv_first.weight"); m("conv_first.bias", "conv_first.bias")
    m("patch_embed.norm.weight", "patch_embed.norm.weight"); m("patch_embed.norm.bias", "patch_embed.norm.bias")
    n_layers = 0
    while f"layer.{n_layers}.hab.0.attn.qkv.weight" in gg: n_layers += 1
    for li in range(n_layers):
        bi = 0
        while f"layer.{li}.hab.{bi}.attn.qkv.weight" in gg:
            sb, db = f"layers.{li}.residual_group.blocks.{bi}", f"layer.{li}.hab.{bi}"
            for g, t in [("attn.qkv.weight", "attn.qkv.weight"), ("attn.qkv.bias", "attn.qkv.bias"),
                         ("attn.proj.weight", "attn.proj.weight"), ("attn.proj.bias", "attn.proj.bias"),
                         ("attn.rpb", "attn.relative_position_bias_table"),
                         ("cab.conv1.weight", "conv_block.cab.0.weight"), ("cab.conv1.bias", "conv_block.cab.0.bias"),
                         ("cab.conv2.weight", "conv_block.cab.2.weight"), ("cab.conv2.bias", "conv_block.cab.2.bias"),
                         ("cab.ca_down.weight", "conv_block.cab.3.attention.1.weight"), ("cab.ca_down.bias", "conv_block.cab.3.attention.1.bias"),
                         ("cab.ca_up.weight", "conv_block.cab.3.attention.3.weight"), ("cab.ca_up.bias", "conv_block.cab.3.attention.3.bias"),
                         ("mlp.fc1.weight", "mlp.fc1.weight"), ("mlp.fc1.bias", "mlp.fc1.bias"),
                         ("mlp.fc2.weight", "mlp.fc2.weight"), ("mlp.fc2.bias", "mlp.fc2.bias"),
                         ("norm1.weight", "norm1.weight"), ("norm1.bias", "norm1.bias"),
                         ("norm2.weight", "norm2.weight"), ("norm2.bias", "norm2.bias"), ("conv_scale", "conv_scale")]:
                m(f"{db}.{g}", f"{sb}.{t}")
            bi += 1
        so, do = f"layers.{li}.residual_group.overlap_attn", f"layer.{li}.ocab"
        for g, t in [("qkv.weight", "qkv.weight"), ("qkv.bias", "qkv.bias"), ("proj.weight", "proj.weight"),
                     ("proj.bias", "proj.bias"), ("rpb", "relative_position_bias_table"),
                     ("norm1.weight", "norm1.weight"), ("norm1.bias", "norm1.bias"),
                     ("norm2.weight", "norm2.weight"), ("norm2.bias", "norm2.bias"),
                     ("mlp.fc1.weight", "mlp.fc1.weight"), ("mlp.fc1.bias", "mlp.fc1.bias"),
                     ("mlp.fc2.weight", "mlp.fc2.weight"), ("mlp.fc2.bias", "mlp.fc2.bias")]:
            m(f"{do}.{g}", f"{so}.{t}")
        m(f"layer.{li}.conv.weight", f"layers.{li}.conv.weight"); m(f"layer.{li}.conv.bias", f"layers.{li}.conv.bias")
    m("norm.weight", "norm.weight"); m("norm.bias", "norm.bias")
    m("conv_after_body.weight", "conv_after_body.weight"); m("conv_after_body.bias", "conv_after_body.bias")
    m("conv_before_upsample.weight", "conv_before_upsample.0.weight"); m("conv_before_upsample.bias", "conv_before_upsample.0.bias")
    nu = 0
    while f"upsample.{nu}.weight" in gg:
        m(f"upsample.{nu}.weight", f"upsample.{nu*2}.weight"); m(f"upsample.{nu}.bias", f"upsample.{nu*2}.bias"); nu += 1
    m("conv_last.weight", "conv_last.weight"); m("conv_last.bias", "conv_last.bias")

    # config from torch state shapes
    embed_dim = state["conv_first.weight"].shape[0]
    depths = [sum(f"layers.{i}.residual_group.blocks.{d}.norm1.weight" in state for d in range(64)) for i in range(n_layers)]
    heads = [state[f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table"].shape[1] for i in range(n_layers)]
    window_size = int((math.sqrt(state["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].shape[0]) + 1) / 2)
    ocab_rpb = state["layers.0.residual_group.overlap_attn.relative_position_bias_table"].shape[0]
    overlap_ratio = (int(math.sqrt(ocab_rpb)) - window_size + 1 - window_size) / window_size
    compress_ratio = embed_dim // state["layers.0.residual_group.blocks.0.conv_block.cab.0.weight"].shape[0]
    squeeze_factor = embed_dim // state["layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight"].shape[0]
    print(f"HAT cfg: embed={embed_dim} layers={n_layers} depths={depths} heads={heads} ws={window_size} x{2**nu} overlap={overlap_ratio}")

    model = HAT(img_size=args.size, in_chans=3, embed_dim=embed_dim, depths=tuple(depths), num_heads=tuple(heads),
                window_size=window_size, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                overlap_ratio=overlap_ratio, mlp_ratio=2., upscale=2**nu, upsampler='pixelshuffle', resi_connection='1conv')
    miss, unexp = model.load_state_dict(state, strict=False)
    miss = [k for k in miss if "relative_position_index" not in k and "attn_mask" not in k and k != "mean"]
    assert not unexp and not miss, f"weight key mismatch — unexpected={unexp[:3]} missing={miss[:3]}"
    model.eval()

    torch.manual_seed(42)
    inp = torch.rand(1, 3, args.size, args.size)
    inter = {"input": inp[0].numpy().copy()}
    with torch.no_grad():
        x = inp; model.mean = model.mean.type_as(x); x = (x - model.mean) * model.img_range
        x = model.conv_first(x); inter["conv_first"] = x[0].numpy().copy()
        residual = x; xs = (x.shape[2], x.shape[3])
        params = {'attn_mask': model.calculate_mask(xs).to(x.device),
                  'rpi_sa': model.relative_position_index_SA, 'rpi_oca': model.relative_position_index_OCA}
        x = model.patch_embed(x)
        for i, layer in enumerate(model.layers):
            x = layer(x, xs, params); inter[f"rhag_{i}"] = model.patch_unembed(x, xs)[0].numpy().copy()
        x = model.norm(x); x = model.patch_unembed(x, xs); x = model.conv_after_body(x) + residual
        inter["deep_features"] = x[0].numpy().copy()
        x = model.conv_before_upsample(x); x = model.conv_last(model.upsample(x))
        inter["output"] = np.clip((x / model.img_range + model.mean)[0].numpy(), 0, 1).copy()

    # write crispembed_diff gguf
    with open(args.output, "wb") as f:
        def ws(s): b = s.encode(); f.write(struct.pack("<Q", len(b))); f.write(b)
        tl = list(inter.items())
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(tl), 1))
        ws("general.architecture"); f.write(struct.pack("<I", 8)); ws("hat_ref")
        off = 0
        for n, d in tl:
            ws(n); f.write(struct.pack("<I", d.ndim))
            for s in d.shape: f.write(struct.pack("<Q", s))
            f.write(struct.pack("<IQ", 0, off)); off = (off + d.nbytes + 31) & ~31
        pos = f.tell(); f.write(b"\x00" * (((pos + 31) & ~31) - pos))
        for n, d in tl:
            f.write(d.astype(np.float32).tobytes()); f.write(b"\x00" * ((((d.nbytes + 31) & ~31) - d.nbytes)))
    print(f"wrote {args.output}: {len(inter)} tensors")


if __name__ == "__main__":
    main()
