import sys, struct, math, types, functools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gguf

# arch_util stub (make_layer + to_2tuple) for standalone PAN_arch import
au = types.ModuleType("models"); au2 = types.ModuleType("models.archs")
arch_util = types.ModuleType("models.archs.arch_util")
def make_layer(block, n_layers): return nn.Sequential(*[block() for _ in range(n_layers)])
arch_util.make_layer = make_layer
def initialize_weights(*a, **k): pass
arch_util.initialize_weights = initialize_weights
sys.modules["models"]=au; sys.modules["models.archs"]=au2; sys.modules["models.archs.arch_util"]=arch_util
import models.archs.arch_util as arch_util  # noqa
ns = {"arch_util": arch_util, "functools": functools, "nn": nn, "F": F, "torch": torch}
exec(open("/private/tmp/pan_arch.py").read().replace("import models.archs.arch_util as arch_util","").replace("from models.archs.arch_util import","# "), ns)
PAN = ns["PAN"]

r = gguf.GGUFReader("/private/tmp/pan/pan-x4-f16.gguf")
gg = {t.name: np.array(t.data, dtype=np.float32).reshape([int(s) for s in t.shape]) for t in r.tensors}
state = {}
def m(g,t):
    if g in gg: state[t]=torch.from_numpy(gg[g].copy())
m("conv_first.weight","conv_first.weight"); m("conv_first.bias","conv_first.bias")
nb=0
while f"scpa.{nb}.conv1_a.weight" in gg: nb+=1
for i in range(nb):
    d=f"scpa.{i}"; p=f"SCPA_trunk.{i}"
    m(f"{d}.conv1_a.weight",f"{p}.conv1_a.weight"); m(f"{d}.conv1_b.weight",f"{p}.conv1_b.weight")
    m(f"{d}.k1.weight",f"{p}.k1.0.weight")
    m(f"{d}.paconv.k2.weight",f"{p}.PAConv.k2.weight"); m(f"{d}.paconv.k2.bias",f"{p}.PAConv.k2.bias")
    m(f"{d}.paconv.k3.weight",f"{p}.PAConv.k3.weight"); m(f"{d}.paconv.k4.weight",f"{p}.PAConv.k4.weight")
    m(f"{d}.conv3.weight",f"{p}.conv3.weight")
m("trunk_conv.weight","trunk_conv.weight"); m("trunk_conv.bias","trunk_conv.bias")
m("upconv1.weight","upconv1.weight"); m("upconv1.bias","upconv1.bias")
m("att1.weight","att1.conv.weight"); m("att1.bias","att1.conv.bias")
m("hrconv1.weight","HRconv1.weight"); m("hrconv1.bias","HRconv1.bias")
m("upconv2.weight","upconv2.weight"); m("upconv2.bias","upconv2.bias")
m("att2.weight","att2.conv.weight"); m("att2.bias","att2.conv.bias")
m("hrconv2.weight","HRconv2.weight"); m("hrconv2.bias","HRconv2.bias")
m("conv_last.weight","conv_last.weight"); m("conv_last.bias","conv_last.bias")

nf=state["conv_first.weight"].shape[0]; unf=state["upconv1.weight"].shape[0]
scale=4 if "upconv2.weight" in state else 2
print(f"PAN cfg nf={nf} unf={unf} nb={nb} scale={scale}")
model=PAN(in_nc=3,out_nc=3,nf=nf,unf=unf,nb=nb,scale=scale)
miss,unexp=model.load_state_dict(state,strict=False)
print(f"load: {len(unexp)} unexpected, {len(miss)} missing: {miss[:4]}")
assert not unexp and not miss
model.eval()
torch.manual_seed(42); SIZE=32
inp=torch.rand(1,3,SIZE,SIZE)
# Match the diff harness, which feeds the C++ engine a uint8-quantized input.
# Snap to the 1/255 grid so torch and C++ see identical pixels (idempotent under
# the test's round(x*255)/255).
inp=torch.round(inp*255.0)/255.0
with torch.no_grad(): out=model(inp)
inter={"input":inp[0].numpy().copy(),"output":np.clip(out[0].numpy(),0,1).copy()}
print("stages:",{k:list(v.shape) for k,v in inter.items()})
with open("/private/tmp/pan/pan-ref.gguf","wb") as f:
    def ws(s): b=s.encode(); f.write(struct.pack("<Q",len(b))); f.write(b)
    tl=list(inter.items()); f.write(struct.pack("<IIQQ",0x46554747,3,len(tl),1))
    ws("general.architecture"); f.write(struct.pack("<I",8)); ws("pan_ref")
    off=0
    for n,d in tl:
        ws(n); f.write(struct.pack("<I",d.ndim))
        for s in d.shape: f.write(struct.pack("<Q",s))
        f.write(struct.pack("<IQ",0,off)); off=(off+d.nbytes+31)&~31
    pos=f.tell(); f.write(b"\x00"*(((pos+31)&~31)-pos))
    for n,d in tl: f.write(d.astype(np.float32).tobytes()); f.write(b"\x00"*((((d.nbytes+31)&~31)-d.nbytes)))
print("wrote pan-ref.gguf")
