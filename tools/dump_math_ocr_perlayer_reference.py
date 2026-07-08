#!/usr/bin/env python
# Per-layer DeiT encoder reference for the crispembed_diff math_ocr harness.
#
# Companion to src/math_ocr.cpp's env-gated diff hooks (MATH_OCR_DIFF_REF) and
# tests/test_math_ocr_diff.cpp. Runs the HF encoder with output_hidden_states on
# the SAME synthetic barred image test_math_ocr_diff.cpp builds, and dumps
# enc_embed + enc_layer_0..N-1 + enc_output as (T,H) F32 tensors. crispembed_diff
# compares per token with row_dim=0.
#
# Usage:
#   pip install transformers torch numpy gguf safetensors sentencepiece
#   USE_TF=0 python tools/dump_math_ocr_perlayer_reference.py \
#       --model-dir microsoft/trocr-small-printed --output trocr-encoder-ref.gguf
#   # then, against a GGUF of the same model:
#   MATH_OCR_DIFF_REF=trocr-encoder-ref.gguf build/test-math-ocr-diff model.gguf <ignored.bin>
#
# NOTE (crispasr-crispembed-dev.md HARD RULE #3): per-layer cos on this SYNTHETIC
# input is diagnostic only — a degenerate/uniform image is ill-conditioned and
# makes quantization look far worse than it is. Always also judge the decoded
# output on a real crop (that is the acceptance test).
import argparse
import numpy as np
import torch
import gguf
from transformers import VisionEncoderDecoderModel


def main():
    p = argparse.ArgumentParser(description="Per-layer DeiT encoder reference dumper")
    p.add_argument("--model-dir", required=True, help="HF model id or local dir (VisionEncoderDecoder)")
    p.add_argument("--output", required=True, help="Output reference GGUF")
    p.add_argument("--image-size", type=int, default=384)
    args = p.parse_args()

    S = args.image_size
    # Identical to tests/test_math_ocr_diff.cpp: uniform gray 0.8, dark bar 0.1.
    gray = np.ones((S, S), np.float32) * 0.8
    gray[S // 2 - 2 : S // 2 + 2, S // 4 : 3 * S // 4] = 0.1
    norm = (gray - 0.5) / 0.5  # TrOCR image processor: mean=std=0.5
    pix = torch.tensor(np.stack([norm, norm, norm])[None])  # (1,3,S,S)

    print(f"loading {args.model_dir} …", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).eval()
    with torch.no_grad():
        out = model.encoder(pix, output_hidden_states=True)
    hs = out.hidden_states            # (embed, layer_0 … layer_{L-1}); len = L+1
    last = out.last_hidden_state       # post final LN
    L = len(hs) - 1
    print(f"encoder: {L} layers, hidden {tuple(hs[0].shape)}")

    w = gguf.GGUFWriter(args.output, arch="math_ocr_ref")
    w.add_string("general.name", "math_ocr_deit_perlayer_reference")

    def add(name, t):
        w.add_tensor(name, t[0].detach().cpu().numpy().astype(np.float32))  # (T,H)

    add("enc_embed", hs[0])
    for i in range(L):
        add(f"enc_layer_{i}", hs[i + 1])
    add("enc_output", last)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: enc_embed + enc_layer_0..{L - 1} + enc_output ({S}x{S} synthetic input)")


if __name__ == "__main__":
    main()
