#!/usr/bin/env python
"""Dump per-stage SMT (Sheet Music Transformer) reference activations.

Ground truth for the crispembed_diff harness. Loads the REAL SMT-plusplus
PyTorch model (not a re-implementation — a manual re-forward could share the
port's bug), runs one fixed image, and writes named F32 intermediates into a
GGUF archive that src/crispembed_diff.h can compare the C++ engine against.

Because SMT is tiny (21.4M / 85 MB) a full torch load is fine on 16 GB — no
layer-by-layer safe_open dance needed.

Stages dumped (earliest first — gate the input before trusting later cos):
  input_tensor   (C,H,W)   preprocessed image (invert+gray+[0,1]); STRUCTURAL GATE
  enc_stage{0,1,2}         ConvNext stage outputs (C,H',W')
  enc_output     (C,H',W') last_hidden_state (pre-pooler-LN) -> decoder
  mem_value      (HW,C)    cross-attn VALUE = raw flattened features
  mem_key        (HW,C)    cross-attn KEY   = features + 2D PE  (key != value!)
  dec_tok_emb    (L,C)     token embedding + 1D PE (teacher-forced seq)
  dec_layer{i}   (L,C)     decoder layer outputs (post-norm3)
  logits         (L,V)     per-position vocab logits (teacher forcing)
  token_ids      (L,)      the greedy-decoded sequence (I32)
metadata: decoded bekern string, seq length.

Usage:
    USE_TF=0 python tools/dump_smt_reference.py \
        --model-dir <smt-grandstaff dir> \
        --smt-repo  <SMT-plusplus clone> \
        --image     <score.png> \
        --output    smt_ref.gguf [--max-tokens 128]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with config.json + model.safetensors")
    ap.add_argument("--smt-repo", required=True, help="SMT-plusplus clone (for smt_model + preprocessing)")
    ap.add_argument("--image", required=True, help="input score image")
    ap.add_argument("--output", required=True, help="output ref GGUF")
    ap.add_argument("--reduce-ratio", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256, help="cap greedy decode length")
    args = ap.parse_args()

    import gguf
    import torch
    import cv2
    from safetensors import safe_open

    sys.path.insert(0, str(Path(args.smt_repo).resolve()))
    from smt_model.modeling_smt import SMTModelForCausalLM
    from smt_model.configuration_smt import SMTConfig
    from data_augmentation.data_augmentation import convert_img_to_tensor

    torch.manual_seed(0)
    device = "cpu"

    # ---- config / model ----
    with open(Path(args.model_dir) / "config.json") as f:
        cfg = json.load(f)
    # i2w keys arrive as JSON strings; predict() indexes by int
    i2w = {int(k): v for k, v in cfg.get("i2w", {}).items()}
    w2i = {k: int(v) for k, v in cfg.get("w2i", {}).items()}
    smt_cfg = SMTConfig(
        maxh=int(cfg["maxh"]), maxw=int(cfg["maxw"]), maxlen=int(cfg["maxlen"]),
        out_categories=int(cfg["out_categories"]), padding_token=int(cfg.get("padding_token", 0)),
        in_channels=int(cfg.get("in_channels", 1)), w2i=w2i, i2w=i2w,
        out_dir=cfg.get("out_dir", "SMIR"), d_model=int(cfg["d_model"]),
        dim_ff=int(cfg["dim_ff"]), num_dec_layers=int(cfg["num_dec_layers"]),
    )
    model = SMTModelForCausalLM(smt_cfg)

    # load weights verbatim (out_layer stays Conv1d [V,d,1] here — do NOT squeeze)
    sd = {}
    with safe_open(str(Path(args.model_dir) / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # positional_2D.pe / positional_1D.pe are non-persistent buffers -> expected missing
    missing = [m for m in missing if not m.endswith(".pe")]
    if missing:
        print(f"WARNING missing (non-PE): {missing[:8]}", file=sys.stderr)
    if unexpected:
        print(f"WARNING unexpected: {unexpected[:8]}", file=sys.stderr)
    model.eval().to(device)

    # ---- preprocessing (SMT-main data.py prepare_data + convert_img_to_tensor) ----
    # RGB (the HF dataset feeds np.array(PIL), NOT cv2 BGR); reduce_ratio=1.0;
    # width = min(w, 3056); height = max(h, 256); Grayscale -> ToTensor, NO invert
    # (SMT-main convert_img_to_tensor has no RandomInvert — inverting tanks accuracy).
    from PIL import Image
    img = np.array(Image.open(args.image).convert("RGB"))  # RGB HWC uint8
    W = min(int(np.ceil(img.shape[1] * args.reduce_ratio)), 3056)
    H = max(int(np.ceil(img.shape[0] * args.reduce_ratio)), 256)
    img = cv2.resize(img, (W, H))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0  # no invert
    x = torch.from_numpy(gray)[None, None].to(device)  # (1,1,H,W)
    print(f"input_tensor: {tuple(x.shape)}  range [{x.min():.3f},{x.max():.3f}]")

    stages = {}
    def store(name, t):
        a = t.detach().cpu().float().numpy()
        stages[name] = np.ascontiguousarray(a)

    def store_hwc(name, t):
        # feature map (C,H,W) -> (H*W, C) so each row is one spatial token's
        # channel vector (per-token cosine in the diff harness). Token order
        # t = h*W + w matches torch flatten(2,3) and ggml [W,H,C] flattening.
        C, H, W = t.shape
        store(name, t.reshape(C, H * W).permute(1, 0).contiguous())

    store("input_tensor", x[0])             # (C,H,W) — fed verbatim to the C++ encoder

    # ---- encoder with stage hooks ----
    hooks = []
    stage_out = {}
    for i, st in enumerate(model.encoder.encoder.stages):
        hooks.append(st.register_forward_hook(
            lambda m, inp, out, i=i: stage_out.__setitem__(i, out)))
    with torch.no_grad():
        enc_output = model.forward_encoder(x)   # (B,256,H',W')
    for h in hooks:
        h.remove()
    for i in sorted(stage_out):
        store_hwc(f"enc_stage{i}", stage_out[i][0])   # (H'W', C)
    store_hwc("enc_output", enc_output[0])            # (H'W', C)
    B, C, Hf, Wf = enc_output.shape
    print(f"enc_output: (C={C}, H'={Hf}, W'={Wf})  seq_len={Hf*Wf}")

    # ---- cross-attn memory (key != value) ----
    with torch.no_grad():
        pos_features = model.positional_2D(enc_output)
        mem_value = torch.flatten(enc_output, 2, 3).permute(2, 0, 1)   # (HW,B,C) raw
        mem_key = torch.flatten(pos_features, 2, 3).permute(2, 0, 1)   # (HW,B,C) +2D PE
    store("mem_value", mem_value[:, 0, :])   # (HW,C)
    store("mem_key", mem_key[:, 0, :])       # (HW,C)

    # ---- greedy decode to get the token sequence ----
    # cap the greedy loop (predict() otherwise runs up to maxlen=1281 steps)
    model.maxlen = min(model.maxlen, args.max_tokens + 1)
    with torch.no_grad():
        text_seq, _ = model.predict(x, convert_to_str=False)
    # rebuild the id sequence predict() walked (bos + argmax tokens, no eos)
    ids = [w2i["<bos>"]] + [w2i.get(tok, w2i.get("<unk>", 0)) if isinstance(tok, str) else int(tok)
                            for tok in text_seq]
    ids = ids[: args.max_tokens]
    decoded = "".join(text_seq[: args.max_tokens - 1]) if text_seq else ""
    print(f"decoded {len(ids)} tokens: {' '.join(map(str, text_seq[:20]))} ...")
    tokens_t = torch.tensor([ids], dtype=torch.long, device=device)

    # ---- teacher-forced decoder pass with per-layer hooks ----
    with torch.no_grad():
        emb = model.decoder.embedding(tokens_t).permute(0, 2, 1)       # (B,C,L)
        pe = model.decoder.positional_1D(emb, start=0)
        dec_tok_emb = pe.permute(2, 0, 1)                              # (L,B,C)
    store("dec_tok_emb", dec_tok_emb[:, 0, :])                          # (L,C)

    layer_out = {}
    hooks = []
    for i, lyr in enumerate(model.decoder.decoder.layers):
        hooks.append(lyr.register_forward_hook(
            lambda m, inp, out, i=i: layer_out.__setitem__(i, out[0])))  # out = (tgt, w_self, w_cross)
    with torch.no_grad():
        out = model.forward_decoder(enc_output, tokens_t)
    for h in hooks:
        h.remove()
    for i in sorted(layer_out):
        store(f"dec_layer{i}", layer_out[i][:, 0, :])                   # (L,C)
    logits = out.logits[0].permute(1, 0).contiguous()                  # (L,V)
    store("logits", logits)
    print(f"logits: {tuple(logits.shape)}")

    # ---- write GGUF ----
    w = gguf.GGUFWriter(args.output, arch="smt_ref")
    w.add_string("general.name", "smt_grandstaff_reference")
    w.add_string("smt.decoded", decoded)
    w.add_uint32("smt.seq_len", len(ids))
    w.add_uint32("smt.enc_h", Hf)
    w.add_uint32("smt.enc_w", Wf)
    for name, a in stages.items():
        w.add_tensor(name, a.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    # store as F32 (the crispembed_diff GGUF reader only decodes F32; its I32
    # branch checks a stale type id) — run_diff rounds back to int
    w.add_tensor("token_ids", np.array(ids, dtype=np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"\nWritten {args.output} ({size_kb:.0f} KB, {len(stages)+1} tensors)")
    for n in stages:
        print(f"  {n:14s} {stages[n].shape}")


if __name__ == "__main__":
    main()
