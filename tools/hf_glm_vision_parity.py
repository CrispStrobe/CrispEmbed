#!/usr/bin/env python3
"""Check a GLM-OCR vision reference GGUF against the REAL HF forward.

Written to arbitrate the layer-13 vision cliff (2026-08-07): the port and the
numpy dumper disagreed from layer 13 and both were suspects. Only the real
GlmOcrVisionModel can say which — and it cleared the dumper, localizing the
defect to the port's METAL path (CPU passes at cos_glob 1.000000, Metal reads
0.958/0.940). See the parity note in src/glm_ocr.cpp.

⚠ PATCH ORDER IS THE TRAP. HF derives its vision position ids assuming the
image processor's merge-BLOCK patch order, so feeding plain raster order makes
the real model disagree with a correct reference from layer 0 — which reads
exactly like a model bug and is not one. Run with GLM_BLOCK_ORDER=1 for the
apples-to-apples comparison; the raster arm is kept only to demonstrate the
trap.

Needs transformers >= 5.x (GlmOcrForConditionalGeneration) and the HF weights.

Usage:
  GLM_BLOCK_ORDER=1 HF_HOME=<cache> python tools/hf_glm_vision_parity.py \
      <hf-snapshot-dir> <ref.gguf>
"""
import os, sys
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrVisionModel
from safetensors import safe_open
from pathlib import Path
import gguf

SNAP = Path(sys.argv[1])
REF = sys.argv[2]

cfg = AutoConfig.from_pretrained(SNAP)
vcfg = cfg.vision_config
S = vcfg.image_size
P = vcfg.patch_size
Tp = vcfg.temporal_patch_size
print(f"vision: image_size={S} patch={P} temporal={Tp} depth={vcfg.depth} hidden={vcfg.hidden_size}")

# ── the harness's synthetic gradient image, verbatim ────────────────────
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
pixels = np.zeros((3, S, S), dtype=np.float32)
for c in range(3):
    for y in range(S):
        for x in range(S):
            val = float(y * S + x) / float(S * S)
            pixels[c, y, x] = (val - mean[c]) / std[c]

n_ph = n_pw = S // P
n_patches = n_ph * n_pw

# ── load vision weights only ────────────────────────────────────────────
vis = GlmOcrVisionModel._from_config(vcfg)
vis = vis.to(torch.float32).eval()
sd = {}
prefix = "model.visual."
for f in sorted(SNAP.glob("*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = sf.get_tensor(k).float()
missing, unexpected = vis.load_state_dict(sd, strict=False)
print(f"loaded {len(sd)} vision tensors; missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing sample:", missing[:6])
if unexpected:
    print("  unexpected sample:", unexpected[:6])

# ── patch tensor exactly as the dumper builds it ────────────────────────
pe_w = sd["patch_embed.proj.weight"].numpy()          # [D, C, T, P, P]
frames = np.stack([pixels] * Tp, axis=0)              # [T, C, P, P] per patch below
patch_dim = int(np.prod(pe_w.shape[1:]))
patches = np.zeros((n_patches, patch_dim), dtype=np.float32)
i = 0
for ph in range(n_ph):
    for pw in range(n_pw):
        blk = frames[:, :, ph * P:(ph + 1) * P, pw * P:(pw + 1) * P]   # [T,C,P,P]
        patches[i] = blk.flatten()
        i += 1

ORDER = os.environ.get("GLM_BLOCK_ORDER", "") == "1"
M = vcfg.spatial_merge_size
# processor order: patches grouped into (merge x merge) spatial blocks, block-raster
idx_raster = np.arange(n_patches).reshape(n_ph, n_pw)
blocks = idx_raster.reshape(n_ph // M, M, n_pw // M, M).transpose(0, 2, 1, 3).reshape(-1)
inv = np.argsort(blocks)


def unblock(a):
    return a[inv]


if ORDER:
    patches = patches[blocks]
hidden = torch.from_numpy(patches)
grid_thw = torch.tensor([[1, n_ph, n_pw]], dtype=torch.long)

# ── capture every block output ──────────────────────────────────────────
caps = {}
for li, blk in enumerate(vis.blocks):
    blk.register_forward_hook(
        lambda m, inp, out, li=li: caps.__setitem__(li, (out[0] if isinstance(out, tuple) else out).detach()))
pe_cap = {}
vis.patch_embed.register_forward_hook(
    lambda m, inp, out, d=pe_cap: d.__setitem__("pe", out.detach()))
pl_cap = {}
vis.post_layernorm.register_forward_hook(
    lambda m, inp, out, d=pl_cap: d.__setitem__("post", out.detach()))

with torch.no_grad():
    res = vis(hidden_states=hidden, grid_thw=grid_thw)
print("HF forward OK; blocks captured:", len(caps))

# ── compare against the reference GGUF (the numpy dumper's output) ──────
r = gguf.GGUFReader(REF)
ref = {t.name: np.asarray(t.data).astype(np.float32) for t in r.tensors}


def cmp(name, hf):
    if name not in ref:
        return None
    a = hf.reshape(-1).astype(np.float64)
    b = ref[name].reshape(-1).astype(np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return cos, float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(np.abs(a - b).max())


out = cmp("vis_patch_embed", pe_cap["pe"].numpy() if not ORDER else unblock(pe_cap["pe"].numpy()))
print("patch_embed vs ref:", out)
print(f"\n{'stage':<20} {'cos(HF,ref)':>12} {'|HF|':>12} {'|ref|':>12} {'max_abs':>10}")
print("-" * 72)
for li in sorted(caps):
    out = cmp(f"vis_layer_{li}", caps[li].numpy())
    if out:
        c, na, nb, m = out
        flag = "  <-- CLIFF" if c < 0.99 else ""
        print(f"vis_layer_{li:<11} {c:>12.6f} {na:>12.2f} {nb:>12.2f} {m:>10.4f}{flag}")
if "post" in pl_cap:
    v = pl_cap["post"].numpy()
    out = cmp("vis_post_norm", unblock(v) if ORDER else v)
    if out:
        c, na, nb, m = out
        print(f"{'vis_post_norm':<20} {c:>12.6f} {na:>12.2f} {nb:>12.2f} {m:>10.4f}")
out = cmp("vis_merger_output", res.pooler_output.numpy())
if out:
    c, na, nb, m = out
    print(f"{'vis_merger_output':<20} {c:>12.6f} {na:>12.2f} {nb:>12.2f} {m:>10.4f}")
out = cmp("vis_downsample", res.last_hidden_state.numpy())
if out:
    c, na, nb, m = out
    print(f"{'vis_downsample':<20} {c:>12.6f} {na:>12.2f} {nb:>12.2f} {m:>10.4f}")

np.save(Path(__file__).with_name("hf_vis_layers.npy"),
        {k: v.numpy() for k, v in caps.items()}, allow_pickle=True)
print("\nsaved HF layer captures")
