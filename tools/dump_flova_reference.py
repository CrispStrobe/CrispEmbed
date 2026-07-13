#!/usr/bin/env python
"""Dump per-stage Flova/omr_transformer reference activations for crispembed_diff.

Loads the real Donut VED (DonutSwin encoder + mBART decoder) and dumps per-stage
F32 intermediates the C++ flova_ocr engine compares against.

Stages:
  pixel_values   (3,H,W)     DonutProcessor output — STRUCTURAL GATE
  enc_stage{0..3} (N_s, C_s)  DonutSwin stage outputs (post-downsample seq)
  enc_output     (N, 1024)    encoder last_hidden_state (post final LayerNorm)
  dec_block{0..3}(L, 1024)    mBART decoder layer outputs (teacher-forced)
  logits         (L, 75)      LM-head logits
  ids            (L,)         greedy-decoded ids (start 56 … eos 54)

Usage:
    USE_TF=0 python tools/dump_flova_reference.py --repo Flova/omr_transformer \
        --image <sample.png> --output flova_ref.gguf
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Flova/omr_transformer")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    import gguf
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    proc = DonutProcessor.from_pretrained(args.repo)
    model = VisionEncoderDecoderModel.from_pretrained(args.repo).eval()
    EOS = proc.tokenizer.eos_token_id  # 54
    START = 56

    img = Image.open(args.image).convert("RGB")
    pv = proc(img, return_tensors="pt").pixel_values  # (1,3,583,409)
    print("pixel_values", tuple(pv.shape), "range", float(pv.min()), float(pv.max()))

    stages = {}
    def store(name, t):
        stages[name] = np.ascontiguousarray(t.detach().cpu().float().numpy())
    store("pixel_values", pv[0])

    enc = model.encoder
    dec = model.decoder  # MBartForCausalLM

    # ---- encoder with per-stage hooks ----
    caps = {}
    hooks = []
    for i, layer in enumerate(enc.encoder.layers):
        hooks.append(layer.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(i, out[0])))
    with torch.no_grad():
        enc_out = enc(pv).last_hidden_state  # (1, N, 1024)
    for h in hooks:
        h.remove()
    for i in sorted(caps):
        store(f"enc_stage{i}", caps[i][0])
    store("enc_output", enc_out[0])
    print("enc_output", tuple(enc_out.shape))

    # ---- greedy decode (argmax) to get ids ----
    with torch.no_grad():
        gen = model.generate(pv, max_length=args.max_tokens, num_beams=1,
                             decoder_start_token_id=START, eos_token_id=EOS, pad_token_id=55)
    ids = gen[0]
    print("decoded", ids.shape[0], "tokens:", repr(proc.tokenizer.decode(ids, skip_special_tokens=True)))
    store("ids", ids.float())

    # ---- teacher-forced decoder pass with per-layer hooks ----
    caps2 = {}
    hh = []
    for i, layer in enumerate(dec.model.decoder.layers):
        hh.append(layer.register_forward_hook(lambda m, inp, out, i=i: caps2.__setitem__(i, out[0])))
    with torch.no_grad():
        dout = dec(input_ids=ids.unsqueeze(0),
                   encoder_hidden_states=enc_out,
                   output_hidden_states=False)
    for h in hh:
        h.remove()
    for i in sorted(caps2):
        store(f"dec_block{i}", caps2[i][0])
    store("logits", dout.logits[0])
    print("logits", tuple(dout.logits.shape))

    W = gguf.GGUFWriter(args.output, arch="flova_ref")
    W.add_string("general.name", "flova_reference")
    W.add_uint32("flova.seq_len", int(ids.shape[0]))
    for name, arr in stages.items():
        W.add_tensor(name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    W.write_header_to_file()
    W.write_kv_data_to_file()
    W.write_tensors_to_file()
    W.close()
    print(f"Written {args.output} ({len(stages)} tensors)")
    for n in stages:
        print(f"  {n:14s} {stages[n].shape}")


if __name__ == "__main__":
    sys.exit(main() or 0)
