#!/usr/bin/env python3
# scunet-ref.gguf from the f32 GGUF (self-consistent). Reuses the verified
# forward in tools/dump_scunet_reference.py; weights come from the GGUF.
import importlib.util
import numpy as np, torch, torch.nn.functional as F, gguf

REPO = "/Users/christianstrobele/code/CrispEmbed"
spec = importlib.util.spec_from_file_location("dref", REPO + "/tools/dump_scunet_reference.py")
dref = importlib.util.module_from_spec(spec); spec.loader.exec_module(dref)

r = gguf.GGUFReader("/private/tmp/sr/scunet-color-f32.gguf")
sd = {t.name: torch.from_numpy(np.array(t.data, dtype=np.float32).copy()) for t in r.tensors}
print(f"loaded {len(sd)} tensors")

W = H = 64; win = 8; head_dim = 32
np.random.seed(42)
inp = np.random.rand(H, W, 3).astype(np.float32)
x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
stages = {"input": x.squeeze(0).numpy().copy()}

x = F.conv2d(x, sd['m_head.0.weight'], sd.get('m_head.0.bias'), padding=1)
stages["head"] = x.squeeze(0).numpy().copy()
skips = [x.clone()]
for sn, ch, nb in [("m_down1", 64, 4), ("m_down2", 128, 4), ("m_down3", 256, 4)]:
    nh = (ch // 2) // head_dim
    for i in range(nb):
        x = dref.conv_trans_block(x, sd, f'{sn}.{i}', nh, win, i % 2 == 1)
    x = F.conv2d(x, sd[f'{sn}.{nb}.weight'], sd.get(f'{sn}.{nb}.bias'), stride=2)
    stages[sn] = x.squeeze(0).numpy().copy(); skips.append(x.clone())
nh = 256 // head_dim
for i in range(4):
    x = dref.conv_trans_block(x, sd, f'm_body.{i}', nh, win, i % 2 == 1)
stages["body"] = x.squeeze(0).numpy().copy()
for sn, ch, nb, skip in [("m_up3", 256, 4, skips[3]), ("m_up2", 128, 4, skips[2]), ("m_up1", 64, 4, skips[1])]:
    x = x + skip
    x = F.conv_transpose2d(x, sd[f'{sn}.0.weight'], sd.get(f'{sn}.0.bias'), stride=2)
    nh = max(1, (ch // 2) // head_dim)
    for i in range(nb):
        x = dref.conv_trans_block(x, sd, f'{sn}.{i+1}', nh, win, i % 2 == 1)
    stages[sn] = x.squeeze(0).numpy().copy()
x = x + skips[0]
x = F.conv2d(x, sd['m_tail.0.weight'], sd.get('m_tail.0.bias'), padding=1)
stages["output"] = x.squeeze(0).numpy().copy()
print(f"output range [{x.min():.4f}, {x.max():.4f}]")

w = gguf.GGUFWriter("/private/tmp/sr/scunet-ref.gguf", "scunet-reference")
w.add_uint32("scunet.ref.width", W); w.add_uint32("scunet.ref.height", H)
for n, a in stages.items():
    w.add_tensor(n, a.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote scunet-ref.gguf")
