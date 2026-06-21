#!/usr/bin/env python3
# adair-ref.gguf: genuine reference from the REAL PyTorch AdaIR model, weights
# reconstructed from adair-5d-f32.gguf (avoids the 345MB .ckpt). Recovers torch
# tensor order by raveling on-disk bytes and reshaping to each model param shape.
#
# AdaIR architecture (model.py) is the upstream net/model.py — fetch once:
#   curl -fsSL https://raw.githubusercontent.com/c-yn/AdaIR/main/net/model.py \
#     -o /private/tmp/adair_model.py
# Deps: torch + einops only (no basicsr/timm). gguf tensor names are 'net.<param>';
# strip the 'net.' prefix to match model.state_dict() keys.
import importlib.util
import numpy as np, gguf, torch

spec = importlib.util.spec_from_file_location("adair_model", "/private/tmp/adair_model.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

model = mod.AdaIR()          # defaults: dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8]
model.eval()
tgt = model.state_dict()

r = gguf.GGUFReader("/private/tmp/sr/adair-5d-f32.gguf")
raw = {}
for t in r.tensors:
    n = t.name[4:] if t.name.startswith("net.") else t.name   # strip 'net.'
    raw[n] = np.array(t.data)

new_sd, missing, mismatch = {}, [], []
for name, p in tgt.items():
    if name not in raw:
        missing.append(name); new_sd[name] = p; continue
    flat = raw[name].reshape(-1)
    if flat.size != p.numel():
        mismatch.append((name, flat.size, p.numel())); new_sd[name] = p; continue
    new_sd[name] = torch.from_numpy(flat.reshape(tuple(p.shape)).copy()).to(p.dtype)

print(f"params={len(tgt)} loaded={len(tgt)-len(missing)-len(mismatch)} "
      f"missing={len(missing)} mismatch={len(mismatch)}")
print("missing(8):", missing[:8]); print("mismatch(8):", mismatch[:8])
print("gguf-not-in-model:", [n for n in raw if n not in tgt][:5])
model.load_state_dict(new_sd, strict=False)

W = H = 64
np.random.seed(42)
inp = np.random.rand(H, W, 3).astype(np.float32)
x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
with torch.no_grad():
    out = model(x)
print("output", list(out.shape[1:]), "range", float(out.min()), float(out.max()))

w = gguf.GGUFWriter("/private/tmp/sr/adair-ref.gguf", "adair-reference")
w.add_tensor("input", x.squeeze(0).numpy().astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
w.add_tensor("output", out.squeeze(0).detach().numpy().astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote adair-ref.gguf")
