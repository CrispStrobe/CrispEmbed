#!/usr/bin/env python3
# swinir-ref.gguf from the GGUF (self-consistent). Reuses the verified numpy
# forward in tools/dump_swinir_reference.py; rebuilds the PyTorch-keyed state by
# REVERSING the converter's rename map (gguf rstb.* -> pytorch layers.*).
import importlib.util, re
import numpy as np, gguf

REPO = "/Users/christianstrobele/code/CrispEmbed"
spec = importlib.util.spec_from_file_location("dref", REPO + "/tools/dump_swinir_reference.py")
dref = importlib.util.module_from_spec(spec); spec.loader.exec_module(dref)

r = gguf.GGUFReader("/private/tmp/sr/swinir-light-x4-f16.gguf")
# gguf-py returns t.data reversed vs t.shape (ne); reshape to ne = PyTorch order.
raw = {t.name: np.array(t.data).reshape([int(s) for s in t.shape]) for t in r.tensors}

def to_pt(g):
    m = re.match(r'rstb\.(\d+)\.block\.(\d+)\.(.*)', g)
    if m:
        i, j, rest = m.groups()
        rest = (rest.replace('attn.rpb_table', 'attn.relative_position_bias_table')
                    .replace('attn.rpb_index', 'attn.relative_position_index')
                    .replace('mlp.up', 'mlp.fc1').replace('mlp.down', 'mlp.fc2'))
        return f'layers.{i}.residual_group.blocks.{j}.{rest}'
    m = re.match(r'rstb\.(\d+)\.conv\.(.*)', g)
    if m: return f'layers.{m.group(1)}.conv.{m.group(2)}'
    if g.startswith('patch_norm'): return g.replace('patch_norm', 'patch_embed.norm')
    if g.startswith('upsample.'): return g.replace('upsample.', 'upsample.0.', 1)
    return g  # conv_first, norm, conv_after_body unchanged

sd = {to_pt(n): v for n, v in raw.items()}
print("conv_first.weight shape:", sd['conv_first.weight'].shape)
n_rstb = len({k.split('.')[1] for k in sd if k.startswith('layers.') and 'residual_group' in k})
n_blocks = len({k.split('.')[4] for k in sd if '.residual_group.blocks.' in k})
n_heads = int(sd['layers.0.residual_group.blocks.0.attn.relative_position_bias_table'].shape[1])
print(f"n_rstb={n_rstb} n_blocks={n_blocks} n_heads={n_heads}")

SIZE = 64
np.random.seed(42)
inp = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)
out, inter = dref.swinir_forward(sd, inp, n_rstb, n_blocks, n_heads)
print("stages:", {k: list(np.asarray(v).shape) for k, v in inter.items()})

w = gguf.GGUFWriter("/private/tmp/sr/swinir-ref.gguf", "swinir-reference")
w.add_uint32("swinir.ref.size", SIZE)
for n, d in inter.items():
    w.add_tensor(n, np.ascontiguousarray(d, dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote swinir-ref.gguf")
