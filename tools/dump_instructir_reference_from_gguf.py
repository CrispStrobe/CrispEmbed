#!/usr/bin/env python3
# Generate instructir-ref.gguf from the f32 GGUF (self-consistent reference).
# Reuses the verified numpy/torch forward in tools/dump_instructir_reference.py;
# only the weight source changes (.pt + .npz  ->  the GGUF itself).
import sys, importlib.util
import numpy as np, torch, torch.nn.functional as F, gguf

REPO = "/Users/christianstrobele/code/CrispEmbed"
spec = importlib.util.spec_from_file_location("dref", REPO + "/tools/dump_instructir_reference.py")
dref = importlib.util.module_from_spec(spec); spec.loader.exec_module(dref)

GGUF = "/private/tmp/sr/instructir-f32.gguf"
OUT  = "/private/tmp/sr/instructir-ref.gguf"
TASK = 0  # INSTRUCTIR_DENOISE — must match test_instructir_diff.cpp

r = gguf.GGUFReader(GGUF)
sd = {}; task_emb = None
for t in r.tensors:
    arr = np.array(t.data, dtype=np.float32)  # gguf-py already gives PyTorch shape
    if t.name == "task_embeddings":
        task_emb = arr; continue
    sd[t.name] = torch.from_numpy(arr.copy())
assert task_emb is not None, "no task_embeddings tensor"
text_embd = torch.from_numpy(task_emb[TASK].copy()).float()  # [256]
print(f"loaded {len(sd)} weight tensors; task_emb {task_emb.shape}")

W = H = 64
np.random.seed(42)
inp = np.random.rand(H, W, 3).astype(np.float32)
x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
stages = {"input": x.squeeze(0).numpy().copy()}

x = F.conv2d(x, sd['intro.weight'], sd['intro.bias'], padding=1)
enc_blocks = [2, 2, 4, 8]; skips = []
for lvl in range(4):
    for i in range(enc_blocks[lvl]):
        x = dref.nafblock(x, sd, f'encoders.{lvl}.{i}')
    x = dref.icb(x, text_embd, sd, f'enc_cond.{lvl}')
    skips.append(x.clone())
    x = F.conv2d(x, sd[f'downs.{lvl}.weight'], sd[f'downs.{lvl}.bias'], stride=2)
for i in range(4):
    x = dref.nafblock(x, sd, f'middle_blks.{i}')
for lvl in range(4):
    x = F.conv2d(x, sd[f'ups.{lvl}.0.weight'], sd.get(f'ups.{lvl}.0.bias'))
    x = dref.pixel_shuffle(x, 2)
    x = x + skips[3 - lvl]
    for i in range(2):
        x = dref.nafblock(x, sd, f'decoders.{lvl}.{i}')
    x = dref.icb(x, text_embd, sd, f'dec_cond.{lvl}')
x = F.conv2d(x, sd['ending.weight'], sd['ending.bias'], padding=1)
x = x + torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
stages["output"] = x.squeeze(0).detach().numpy().copy()
print(f"output range [{x.min():.4f}, {x.max():.4f}]")

w = gguf.GGUFWriter(OUT, "instructir-reference")
w.add_uint32("instructir.ref.width", W); w.add_uint32("instructir.ref.height", H)
for n, a in stages.items():
    w.add_tensor(n, a.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {OUT}")
