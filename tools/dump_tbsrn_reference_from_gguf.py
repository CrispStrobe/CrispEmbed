#!/usr/bin/env python3
# tbsrn-ref.gguf from the model GGUF (self-consistent). Reuses the numpy forward
# in tools/dump_tbsrn_reference.py; rebuilds the paddle-keyed state by REVERSING
# the converter (convert-tbsrn-to-gguf.py) rename + Linear-transpose map.
import importlib.util
import numpy as np, gguf

REPO = "/Users/christianstrobele/code/CrispEmbed"
spec = importlib.util.spec_from_file_location("dref", REPO + "/tools/dump_tbsrn_reference.py")
dref = importlib.util.module_from_spec(spec); spec.loader.exec_module(dref)

r = gguf.GGUFReader("/private/tmp/sr/tbsrn-telescope-f16.gguf")
# custom struct writer (like swinir): reshape t.data to t.shape recovers paddle order.
raw = {t.name: np.array(t.data).reshape([int(s) for s in t.shape]).astype(np.float32)
       for t in r.tensors}

P = "transform."
SRB = 5
# gguf names the converter transposed (transpose=True, 2D Linear): un-transpose here.
TRANSPOSED = set()
for i in range(SRB):
    for j in range(4):
        TRANSPOSED.add(f"srb.{i}.fe.mha.linear{j}.weight")
    TRANSPOSED.add(f"srb.{i}.fe.ffn.w1.weight")
    TRANSPOSED.add(f"srb.{i}.fe.ffn.w2.weight")
    TRANSPOSED.add(f"srb.{i}.fe.linear.weight")

def to_paddle(g):
    """gguf tensor name -> paddle key consumed by dump_tbsrn_reference.tbsrn_forward."""
    if g == "block1.conv.weight":  return P + "block1.0.weight"
    if g == "block1.conv.bias":    return P + "block1.0.bias"
    if g == "block1.prelu.weight": return P + "block1.1._weight"
    if g.startswith("srb."):
        _, i, rest = g.split(".", 2)
        bp = f"{P}block{int(i)+2}"
        fe = f"{bp}.feature_enhancer"
        m = {
            "conv1.weight": f"{bp}.conv1.weight", "conv1.bias": f"{bp}.conv1.bias",
            "conv2.weight": f"{bp}.conv2.weight", "conv2.bias": f"{bp}.conv2.bias",
            "bn1.weight": f"{bp}.bn1.weight", "bn1.bias": f"{bp}.bn1.bias",
            "bn1.running_mean": f"{bp}.bn1._mean", "bn1.running_var": f"{bp}.bn1._variance",
            "bn2.weight": f"{bp}.bn2.weight", "bn2.bias": f"{bp}.bn2.bias",
            "bn2.running_mean": f"{bp}.bn2._mean", "bn2.running_var": f"{bp}.bn2._variance",
            "fe.ln1.weight": f"{fe}.mul_layernorm1.a_2", "fe.ln1.bias": f"{fe}.mul_layernorm1.b_2",
            "fe.ln3.weight": f"{fe}.mul_layernorm3.a_2", "fe.ln3.bias": f"{fe}.mul_layernorm3.b_2",
            "fe.ffn.w1.weight": f"{fe}.pff.w_1.weight", "fe.ffn.w1.bias": f"{fe}.pff.w_1.bias",
            "fe.ffn.w2.weight": f"{fe}.pff.w_2.weight", "fe.ffn.w2.bias": f"{fe}.pff.w_2.bias",
            "fe.linear.weight": f"{fe}.linear.weight", "fe.linear.bias": f"{fe}.linear.bias",
        }
        for j in range(4):
            m[f"fe.mha.linear{j}.weight"] = f"{fe}.multihead.linears.{j}.weight"
            m[f"fe.mha.linear{j}.bias"]   = f"{fe}.multihead.linears.{j}.bias"
        return m[rest]
    if g == "final_conv.weight": return P + "block7.0.weight"
    if g == "final_conv.bias":   return P + "block7.0.bias"
    if g == "final_bn.weight":   return P + "block7.1.weight"
    if g == "final_bn.bias":     return P + "block7.1.bias"
    if g == "final_bn.running_mean": return P + "block7.1._mean"
    if g == "final_bn.running_var":  return P + "block7.1._variance"
    if g == "upsample.conv.weight":  return P + "block8.0.conv.weight"
    if g == "upsample.conv.bias":    return P + "block8.0.conv.bias"
    if g == "output_conv.weight":    return P + "block8.1.weight"
    if g == "output_conv.bias":      return P + "block8.1.bias"
    raise KeyError(g)

state = {}
for n, v in raw.items():
    if n in TRANSPOSED and v.ndim == 2:
        v = v.T.copy()  # converter did [in,out]->[out,in]; dumper wants paddle [in,out]
    state[to_paddle(n)] = v

print("tensors:", len(state), " block1 conv:", state[P+"block1.0.weight"].shape,
      " mha lin0:", state[P+"block2.feature_enhancer.multihead.linears.0.weight"].shape)

np.random.seed(42)
inp = np.random.rand(1, 3, 16, 64).astype(np.float32)
out, inter = dref.tbsrn_forward(state, inp, prefix=P)
print("stages:", {k: list(np.asarray(v).shape) for k, v in inter.items()})

w = gguf.GGUFWriter("/private/tmp/sr/tbsrn-ref.gguf", "tbsrn-reference")
w.add_uint32("tbsrn.ref.size", 64)
for n, d in inter.items():
    w.add_tensor(n, np.ascontiguousarray(d, dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote tbsrn-ref.gguf")
