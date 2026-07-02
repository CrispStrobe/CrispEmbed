#!/usr/bin/env python3
"""Dump a Restormer parity reference from a CrispEmbed GGUF (no .pth needed).

Reconstructs the PyTorch Restormer (arch reimpl in dump_restormer_reference.py)
with weights loaded from the converted GGUF, runs a deterministic 64x64 input,
and writes input+output to a ref GGUF for test-restormer-diff.

Weights: convert-restormer-to-gguf.py writes conv kernels RAW as numpy
(OC,IC,KH,KW) C-order, so raveling the on-disk bytes and reshaping to each torch
param's shape recovers the exact tensor. GGUF tensor names are the CrispEmbed
names; map them back to the reimpl module keys.

Usage:
    /Users/.../miniconda3/bin/python tools/dump_restormer_reference_from_gguf.py \
        --gguf restormer-denoise-f16.gguf --output restormer-ref.gguf --size 64
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "restormer_ref_arch", os.path.join(HERE, "dump_restormer_reference.py"))
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
Restormer = arch.Restormer
write_gguf = arch.write_gguf


# ── GGUF name (CrispEmbed) → reimpl module key ──────────────────────────

LEVELS = {  # gguf level prefix → reimpl attribute
    "enc.0": "encoder_level1", "enc.1": "encoder_level2",
    "enc.2": "encoder_level3", "latent": "latent",
    "dec.2": "decoder_level3", "dec.1": "decoder_level2",
    "dec.0": "decoder_level1", "refine": "refinement",
}
# within-block leaf renames (gguf → reimpl)
BLOCK_LEAF = {
    "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
    "attn.qkv.weight": "attn.qkv.weight", "attn.qkv.bias": "attn.qkv.bias",
    "attn.qkv_dw.weight": "attn.qkv_dwconv.weight",
    "attn.qkv_dw.bias": "attn.qkv_dwconv.bias",
    "attn.proj.weight": "attn.project_out.weight",
    "attn.proj.bias": "attn.project_out.bias",
    "attn.temperature": "attn.temperature",
    "ffn.in.weight": "ffn.project_in.weight", "ffn.in.bias": "ffn.project_in.bias",
    "ffn.dw.weight": "ffn.dwconv.weight", "ffn.dw.bias": "ffn.dwconv.bias",
    "ffn.out.weight": "ffn.project_out.weight", "ffn.out.bias": "ffn.project_out.bias",
}
TOP = {  # gguf top-level → reimpl key
    "patch_embed.weight": "patch_embed.weight", "patch_embed.bias": "patch_embed.bias",
    "down.0.weight": "down1_2.0.weight", "down.1.weight": "down2_3.0.weight",
    "down.2.weight": "down3_4.0.weight",
    "up.0.weight": "up4_3.0.weight", "up.1.weight": "up3_2.0.weight",
    "up.2.weight": "up2_1.0.weight",
    "reduce.2.weight": "reduce_chan_level3.weight",
    "reduce.1.weight": "reduce_chan_level2.weight",
    "reduce.2.bias": "reduce_chan_level3.bias",
    "reduce.1.bias": "reduce_chan_level2.bias",
    "output.weight": "output.weight", "output.bias": "output.bias",
}


def gguf_to_reimpl_key(name):
    if name in TOP:
        return TOP[name]
    for gp, rp in LEVELS.items():
        pref = gp + "."
        if name.startswith(pref):
            rest = name[len(pref):]              # "<b>.<leaf>"
            b, leaf = rest.split(".", 1)
            if leaf not in BLOCK_LEAF:
                return None
            return f"{rp}.{b}.{BLOCK_LEAF[leaf]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="converted restormer GGUF (weights)")
    ap.add_argument("--output", required=True, help="ref GGUF to write")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)

    def kv(key, default=None):
        f = r.fields.get(key)
        if f is None:
            return default
        # scalar or array of scalars
        vals = [f.parts[d] for d in f.data]
        out = [int(v[0]) if hasattr(v, "__len__") else int(v) for v in vals]
        return out if len(out) > 1 else out[0]

    dim = kv("restormer.dim", 48)
    num_blocks = kv("restormer.num_blocks", [4, 6, 6, 8])
    heads = kv("restormer.heads", [1, 2, 4, 8])
    n_refine = kv("restormer.num_refinement_blocks", 4)
    has_bias = bool(kv("restormer.bias", 0))
    # The GGUF stores a *lossy derived* ffn factor (e.g. 2.64583) that floors
    # int(dim*factor) one short of the real hidden width. Restormer's canonical
    # value is 2.66, which reproduces every level's stored FFN width exactly
    # (int(dim*2.66)*2 == 254/510/1020/2042). Use it.
    ffn_factor = 2.66
    ln_type = "WithBias" if has_bias else "BiasFree"
    print(f"config: dim={dim} blocks={num_blocks} heads={heads} refine={n_refine} "
          f"bias={has_bias} ffn={ffn_factor:.5f} ln={ln_type}")

    model = Restormer(dim=dim, num_blocks=num_blocks, heads=heads,
                      ffn_expansion_factor=ffn_factor, bias=has_bias,
                      ln_type=ln_type, num_refinement_blocks=n_refine)
    model.eval()
    tgt = model.state_dict()

    raw = {t.name: np.array(t.data) for t in r.tensors}
    new_sd, loaded, missing, mism = {}, 0, [], []
    for gname, arr in raw.items():
        rk = gguf_to_reimpl_key(gname)
        if rk is None or rk not in tgt:
            missing.append(gname); continue
        p = tgt[rk]
        flat = arr.reshape(-1)
        if flat.size != p.numel():
            mism.append((gname, rk, flat.size, p.numel())); continue
        new_sd[rk] = torch.from_numpy(flat.reshape(tuple(p.shape)).astype(np.float32))
        loaded += 1
    print(f"weights: loaded={loaded}/{len(tgt)} unmapped_gguf={len(missing)} mismatch={len(mism)}")
    if mism:
        print("  mismatch:", mism[:6])
    result = model.load_state_dict(new_sd, strict=False)
    still_missing = [k for k in result.missing_keys]
    if still_missing:
        print(f"  WARNING: {len(still_missing)} model params not loaded: {still_missing[:8]}")

    # Deterministic input, quantized to uint8 granularity so the C++ test's
    # f32→uint8→f32 round-trip is exact (isolates the parity check to the graph).
    rng = np.random.RandomState(42)
    S = args.size
    x = rng.rand(1, 3, S, S).astype(np.float32)
    x = np.round(x * 255.0) / 255.0
    with torch.no_grad():
        out, inter = model.forward_with_intermediates(torch.from_numpy(x))

    ref = {"input": inter["input"], "output": inter["output"]}
    write_gguf(args.output, ref)
    o = inter["output"]
    print(f"output stats: mean={o.mean():.4f} std={o.std():.4f} "
          f"min={o.min():.4f} max={o.max():.4f}")


if __name__ == "__main__":
    main()
