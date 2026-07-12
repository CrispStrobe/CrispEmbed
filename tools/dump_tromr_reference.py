#!/usr/bin/env python
"""Dump per-stage TrOMR reference activations for the crispembed_diff harness.

Loads the real Polyphonic-TrOMR PyTorch model (via forward hooks — not a
re-implementation) and dumps per-stage F32 intermediates to a GGUF the C++
tromr_ocr engine compares against.

Env quirks handled: timm pulls in `trio` (broken on this platform → stubbed);
the checkpoint needs the OLD x-transformers 0.29.2 (pass its dir via --xt); the
.pth was saved on CUDA (map_location=cpu).

Stages dumped (earliest first):
  input_tensor   (1,H,W)   preprocessed image (ToGray+Normalize) — STRUCTURAL GATE
  enc_backbone   (C,H',W')  ResNetV2 output (1024ch, /16)
  enc_context    (N+1, 256) ViT output (cls + patches) = decoder cross-attn memory
  dec_tok_emb    (L, 256)   rhythm+pitch+lift+pos embedding (teacher-forced)
  dec_layer{i}   (L, 256)   x_transformers layer outputs (0..11)
  logits_rhythm/pitch/lift/note  (L, V)   per-position head logits
  ids_rhythm/pitch/lift  (L,)   the greedy-decoded id streams (I32→F32)

Usage:
    USE_TF=0 python tools/dump_tromr_reference.py --repo <Polyphonic-TrOMR/tromr> \
        --xt <x-transformers-0.29.2 dir> --image <photo.jpg> --output tromr_ref.gguf
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Polyphonic-TrOMR/tromr dir")
    ap.add_argument("--xt", required=True, help="x-transformers 0.29.2 install dir")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.xt).resolve()))
    sys.modules["trio"] = types.ModuleType("trio")  # timm's async backend, broken here
    import gguf
    import cv2
    import torch

    _o = torch.load
    torch.load = lambda *a, **k: _o(*a, map_location="cpu", **{kk: vv for kk, vv in k.items() if kk != "map_location"})
    torch.manual_seed(0)

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from configs import getconfig
    from staff2score import StaffToScore

    conf = getconfig(str(Path(args.repo) / "workspace" / "config.yaml"))
    handler = StaffToScore(conf)
    model = handler.model.eval()

    # ---- preprocessing (staff2score.readimg) ----
    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        a = 255 - img[:, :, 3]
        img = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    nh = conf.max_height
    nw = int(nh / h * w) // conf.patch_size * conf.patch_size
    img = cv2.resize(img, (nw, nh))
    x = handler.transform(image=img)["image"][:1].float().unsqueeze(0)  # (1,1,H,W)
    print(f"input {tuple(x.shape)} range [{x.min():.3f},{x.max():.3f}]")

    stages = {}
    def store(name, t):
        stages[name] = np.ascontiguousarray(t.detach().cpu().float().numpy())

    store("input_tensor", x[0])

    # ---- encoder with hooks ----
    hooks = []
    cap = {}
    hooks.append(model.encoder.patch_embed.backbone.register_forward_hook(
        lambda m, i, o: cap.__setitem__("backbone", o)))
    with torch.no_grad():
        context = model.encoder(x)  # (1, N+1, 256) after ViT+norm
    for h in hooks:
        h.remove()
    store("enc_backbone", cap["backbone"][0])       # (1024, H', W')
    store("enc_context", context[0])                # (N+1, 256)
    print(f"enc_backbone {cap['backbone'].shape}  enc_context {tuple(context.shape)}")

    # ---- greedy decode (argmax) to get id streams ----
    dec = model.decoder
    start = torch.LongTensor([[conf.bos_token]])
    nonote = torch.LongTensor([[conf.nonote_token]])
    out_r, out_p, out_l = start, nonote, nonote
    mask = torch.ones_like(out_r, dtype=torch.bool)
    for _ in range(args.max_tokens):
        rp, pp, lp, npp, _ = dec.net(out_r[:, -conf.max_seq_len:], out_p[:, -conf.max_seq_len:],
                                     out_l[:, -conf.max_seq_len:], mask=mask, context=context)
        r = rp[:, -1].argmax(-1, keepdim=True)
        p = pp[:, -1].argmax(-1, keepdim=True)
        l = lp[:, -1].argmax(-1, keepdim=True)
        out_r = torch.cat([out_r, r], 1); out_p = torch.cat([out_p, p], 1); out_l = torch.cat([out_l, l], 1)
        mask = torch.nn.functional.pad(mask, (0, 1), value=True)
        if (out_r == conf.eos_token).any():
            break
    L = out_r.shape[1]
    print(f"decoded {L} tokens")

    # ---- teacher-forced pass with per-layer hooks (argmax id streams) ----
    layer_out = {}
    hh = []
    for i, (_norm, block, _res) in enumerate(dec.net.attn_layers.layers):
        pass
    # hook each residual output via forward hook on the whole attn_layers is hard;
    # instead hook each block and capture its residual-added output by wrapping.
    caps = {}
    def mk(i):
        def hook(m, inp, out):
            caps[i] = out[0] if isinstance(out, tuple) else out
        return hook
    for i, layer in enumerate(dec.net.attn_layers.layers):
        hh.append(layer[1].register_forward_hook(mk(i)))
    emb_cap = {}
    with torch.no_grad():
        # replicate ScoreTransformerWrapper.forward embedding
        net = dec.net
        emb = net.rhythm_emb(out_r) + net.pitch_emb(out_p) + net.lift_emb(out_l) + net.pos_emb(out_r)
        emb = net.project_emb(emb)
        emb_cap["tok"] = emb
        rp, pp, lp, npp, hx = net(out_r, out_p, out_l, mask=torch.ones_like(out_r, dtype=torch.bool), context=context)
    for h in hh:
        h.remove()
    store("dec_tok_emb", emb_cap["tok"][0])
    for i in sorted(caps):
        store(f"dec_block{i}", caps[i][0])
    store("logits_rhythm", rp[0]); store("logits_pitch", pp[0])
    store("logits_lift", lp[0]); store("logits_note", npp[0])

    # ---- write GGUF ----
    W = gguf.GGUFWriter(args.output, arch="tromr_ref")
    W.add_string("general.name", "tromr_reference")
    W.add_uint32("tromr.seq_len", L)
    for name, arr in stages.items():
        W.add_tensor(name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    for name, ids in (("ids_rhythm", out_r), ("ids_pitch", out_p), ("ids_lift", out_l)):
        W.add_tensor(name, ids[0].cpu().numpy().astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    W.write_header_to_file()
    W.write_kv_data_to_file()
    W.write_tensors_to_file()
    W.close()
    print(f"Written {args.output} ({len(stages)+3} tensors)")
    for n in stages:
        print(f"  {n:14s} {stages[n].shape}")


if __name__ == "__main__":
    main()
