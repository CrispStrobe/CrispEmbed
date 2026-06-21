#!/usr/bin/env python3
# dat-ref.gguf: genuine reference from the REAL PyTorch DAT-light model, with
# weights reconstructed from dat-light-x2-f32.gguf (official gguf writer that
# kept torch key names + flattened 4D convs to 2D). Recovers torch tensor order
# by raveling the on-disk bytes and reshaping to each model param's shape.
import sys, types, importlib.util, math
import numpy as np, gguf

# ── mock basicsr registry + timm (DropPath/trunc_normal_ are no-ops in eval) ──
import torch, torch.nn as nn
_b = types.ModuleType("basicsr"); _bu = types.ModuleType("basicsr.utils")
_br = types.ModuleType("basicsr.utils.registry")
class _Reg:
    def register(self): return lambda c: c
_br.ARCH_REGISTRY = _Reg(); _bu.registry = _br; _b.utils = _bu
_ba = types.ModuleType("basicsr.archs"); _b.archs = _ba
sys.modules.update({"basicsr": _b, "basicsr.utils": _bu,
                    "basicsr.utils.registry": _br, "basicsr.archs": _ba})
_timm = types.ModuleType("timm"); _tm = types.ModuleType("timm.models")
_tl = types.ModuleType("timm.models.layers")
class DropPath(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, x): return x
def trunc_normal_(t, *a, **k): return t
_tl.DropPath = DropPath; _tl.trunc_normal_ = trunc_normal_
_tm.layers = _tl; _timm.models = _tm
sys.modules.update({"timm": _timm, "timm.models": _tm, "timm.models.layers": _tl})

spec = importlib.util.spec_from_file_location(
    "dat_arch", "/private/tmp/dat-src/basicsr/archs/dat_arch.py")
dat_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(dat_mod)
DAT = dat_mod.DAT

model = DAT(upscale=2, in_chans=3, img_size=64, img_range=1., depth=[18],
            embed_dim=60, num_heads=[6], expansion_factor=2,
            resi_connection='3conv', split_size=[8, 32], qkv_bias=True,
            upsampler='pixelshuffledirect')
model.eval()
tgt = model.state_dict()

r = gguf.GGUFReader("/private/tmp/sr/dat-light-x2-f32.gguf")
raw = {t.name: np.array(t.data) for t in r.tensors}

new_sd, missing, mismatch = {}, [], []
for name, p in tgt.items():
    if name not in raw:
        missing.append(name); new_sd[name] = p; continue
    want = tuple(p.shape)
    flat = raw[name].reshape(-1)
    if flat.size != p.numel():
        mismatch.append((name, flat.size, p.numel())); new_sd[name] = p; continue
    new_sd[name] = torch.from_numpy(flat.reshape(want).copy()).to(p.dtype)

print(f"params={len(tgt)} loaded={len(tgt)-len(missing)-len(mismatch)} "
      f"missing={len(missing)} mismatch={len(mismatch)}")
print("missing (first 8):", missing[:8])
print("mismatch (first 8):", mismatch[:8])
extra = [n for n in raw if n not in tgt]
print(f"gguf tensors not in model: {len(extra)}", extra[:5])

model.load_state_dict(new_sd, strict=False)

np.random.seed(42)
x = torch.from_numpy(np.random.rand(1, 3, 64, 64).astype(np.float32))

caps = {"input": x[0].numpy().copy()}
def hook(name):
    def h(m, i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        if isinstance(t, torch.Tensor): caps[name] = t.detach().cpu().numpy()
    return h
model.conv_first.register_forward_hook(hook("conv_first"))
for i, blk in enumerate(model.layers[0].blocks):
    blk.register_forward_hook(hook(f"block_{i}"))

with torch.no_grad():
    out = model(x)
caps["output"] = out[0].detach().cpu().numpy()
print("output shape", caps["output"].shape, "range",
      float(out.min()), float(out.max()))

# conv_first capture is [1,C,H,W] -> [C,H,W]; blocks are [1,H*W,C] (token layout)
w = gguf.GGUFWriter("/private/tmp/sr/dat-ref.gguf", "dat-reference")
w.add_uint32("dat.ref.size", 64)
for n, d in caps.items():
    d = np.asarray(d)
    if d.ndim == 4 and d.shape[0] == 1: d = d[0]
    w.add_tensor(n, np.ascontiguousarray(d, dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("stages:", {n: list(np.asarray(d).shape if np.asarray(d).ndim<4 else np.asarray(d)[0].shape) for n,d in caps.items()})
print("wrote dat-ref.gguf")
