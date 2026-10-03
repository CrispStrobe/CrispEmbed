---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-Encoder-230M
base_model_relation: quantized
pipeline_tag: fill-mask
tags: [gguf, crispembed, liquid, lfm2, bidirectional, masked-lm]
---

# LFM2.5-Encoder-230M — calibrated CrispEmbed GGUFs

Quantizations of [LiquidAI/LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M)
for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
The upstream LFM Open License v1.0 is included unchanged as LICENSE.
This is a bidirectional masked language model with 1024-dimensional token features,
not a fine-tuned retrieval model. Multiple masks are evaluated jointly.

## Files and measured reference agreement

CPU measurements against the original FP32 Python checkpoint, 15 probe cases:

| File | MB | Arithmetic | Minimum token cosine | Mask agreement |
|---|---:|---|---:|---:|
| LFM2.5-Encoder-230M-crisp-Q4_K-imatrix.gguf | 165.3 | FP32 matrix casts | 0.889690 | 12/15 |
| LFM2.5-Encoder-230M-q4_ops_down_q8.gguf | 209.9 | FP32 row-dot | 0.928912 | 14/15 |
| LFM2.5-Encoder-230M-q8_ops_down_f16.gguf | 330.2 | FP32 row-dot | 0.999818 | 14/15 |

These counts measure reference agreement, not downstream accuracy. Profiles were
selected on these probes. The 210 MB precise-mode result is a numerical/MLM screen;
full API contracts were also checked in ordinary mode. The 330 MB precise mode
passes API contracts and numerical tolerances, but still changes one low-margin
masked prediction. Larger precision profiles were not consistently better.
Use the [official F16 GGUF](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M-GGUF)
for strongest checkpoint parity (0.999989 minimum token cosine, 15/15 masks).
The base alias `lfm2-encoder-230m` in CrispEmbed continues to select official F16.

## Run

Build CrispEmbed from main at ef2cb170 or later (no release binary claim).
Download and run the 330 MB model with:

```sh
CRISPEMBED_LFM2_F32_DOT=1 crispembed -m lfm2-encoder-230m-mixed330 \
  --fill-mask "The capital of France is [MASK]." --json
```

The 210 MB alias is `lfm2-encoder-230m-mixed210`. The compact calibrated alias
is `lfm2-encoder-230m-q4k`; use `CRISPEMBED_LFM2_F32_MATMUL=1` to reproduce its
reported FP32-cast measurement. Without these opt-ins the usual GGML matmuls
also quantize activation operands, and parity is lower. No GPU parity or
uncontended speed advantage is claimed. The precision modes change inference
arithmetic, not stored weights. CPU is the default for this encoder.

Raw token features are available through `--raw-tokens`, Python
`encode_tokens(text, normalize=False)`, and the C/Rust/Dart APIs. Use raw features
for the tied MLM head; normalized or pooled features are not interchangeable.
The publisher's unchanged NumPy helper assumes floating-point weights; quantized
heads must be dequantized before projection.

## Reproduce

Calibration and mixed precision commands, tensor override patterns, tokenizer
checks, decoded predictions, norms and limitations are documented in
[CrispEmbed's encoder guide](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/lfm2-encoder.md).
`results/` contains the quantization, mixed-precision and arithmetic audit manifests.
The manifests include model SHA-256 hashes. Original FP32 and identical-Q8-weight
Python fixtures are hosted in
[cstr/crispembed-regression-fixtures](https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/tree/main/lfm2-encoder-230m).
Both precise Q8 modes pass all 297 layer checks against identical-weight Python
(minimum token cosine 0.999999707, 15/15 same-weight masks).

The 165 MB profile used 6,711 calibration tokens. The mixed models used a fresh
FP32 source and expanded independent calibration: 174 records / 185 samples /
12,065 tokens (maximum 1,149). Critical attention, ShortConv projection and FFN
down matrices are kept at Q8_0 (210 MB) or F16 (330 MB); 28 FFN gate/up matrices
remain calibrated Q4_K or Q8_0, respectively. The tied embedding/head is Q8_0;
49 F32 norm/kernel tensors are preserved. No training data or task fine-tuning
was added. Calibration records and reproduction tooling live in CrispEmbed.
