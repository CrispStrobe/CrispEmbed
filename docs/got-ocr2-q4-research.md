# GOT-OCR2 Q4: prior art, diagnosis and experiment plan

Status: research (7 November 2026). Nothing in this document changes a default, a published file
or an alias. GOT-OCR2 is covered by the Tongyi Qianwen RESEARCH LICENSE (research/evaluation
only; `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1`), so every experiment here is evaluation
work. Nothing below is offered as a commercial build.

## Problem

F16-r2 (`got-ocr2-f16`, Hub `c0f1aa7b`) is token-exact against the original
`stepfun-ai/GOT-OCR2_0` `chat()` pipeline on the frozen panel. The published Q4_K
(`got-ocr2-q4_k-experimental`) is not. On the **3-case** panel (fox / receipt_example /
simple_form; "83 / 20" in PLAN is the first divergent token on the two failing cases, not a
20-case panel) it is 1/3 token-exact. receipt_example diverges at token 83 (char ratio 0.8041)
and simple_form at token 20 (0.9679). Receipts: `tests/results/got-ocr2-f16/qualify-41588457181-orig-ref/`,
[quantized-models.md](quantized-models.md#got-ocr2-research-licence).

## 1. Prior art

| Project | Licence (as stated by that repo) | What is quantized | What stays high precision | Method | Reported quality |
|---|---|---|---|---|---|
| [mlx-community/GOT-OCR2_0-4bit](https://huggingface.co/mlx-community/GOT-OCR2_0-4bit) ([config](https://huggingface.co/mlx-community/GOT-OCR2_0-4bit/raw/main/config.json)) | card says apache-2.0. Not our licence reading; we keep the Tongyi research gate | all 169 language-model matrices, **including the tied embeddings / LM head** | **SAM vision tower and projector in bf16** ("by design", 96.7M of 560.5M params) | MLX affine, 4 bit, group 64, no calibration. Activations stay bf16 (weight-only dequant matmul) | 6 synthetic documents: CER vs bf16 **0.0116** at 4-bit; 8-bit byte-identical to bf16. Final-position logits of MLX bf16 vs PyTorch fp32: cosine 0.99979 |
| [MosRat/got.cpp](https://github.com/MosRat/got.cpp) + [got_decoder-Q4_K_M-GGUF](https://huggingface.co/MosRat/got_decoder-Q4_K_M-GGUF) (credited in the [official repo](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)) | MIT (code). The weights card names no licence | decoder only, as a stock `qwen2` GGUF with llama.cpp k-quants (Q2_K … Q8_0) | vision encoder runs **outside** llama.cpp in ONNX Runtime (Olive export, precision not stated) | stock `llama-quantize` presets (gguf-my-repo tag); no imatrix mentioned | none published |
| [Jerry666/GOT-OCR2_0-716M-BF16-GGUF](https://huggingface.co/Jerry666/GOT-OCR2_0-716M-BF16-GGUF) | — | nothing (BF16) | everything | — | none |
| llama.cpp mainline | MIT | no `got_ocr` architecture; the only path is the decoder-as-qwen2 route above | — | — | — |
| Transformers `GotOcr2ForConditionalGeneration` (≥ 4.49) | — | no published bitsandbytes/AWQ/GPTQ GOT-OCR2 checkpoint found | — | generic bnb/AWQ would apply; ONNX INT4 tooling (e.g. NVIDIA ModelOpt) excludes the LM head by default | none found |
| [Qwen/Qwen1.5-0.5B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF) (same decoder family, text only) | tongyi-qianwen-research | stock llama.cpp presets | per preset | — | wiki perplexity fp16 34.20, q8_0 34.22, **q4_k_m 34.27**, q3_k_m 38.25, q2_k 62.14 |

What llama.cpp's Q4_K_M actually does
([src/llama-quant.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp)):
the base type is Q4_K. The output tensor, and `token_embd` when it is tied (as in GOT/Qwen-0.5B), go to
**Q6_K**. `attn_v` and `ffn_down` go to **Q6_K** on the "use_more_bits" layers
`i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2`. For n = 24 those are layers
0,1,2,5,8,11,14,17,20,21,22,23. On CPU the activations of a K-quant matmul are quantized to
Q8_K (Q8_0 for Q8_0 weights); F16 weights take F16 activations.

Takeaways:
1. Nobody publishes a token-exact 4-bit GOT-OCR2. The one measured 4-bit build (MLX) reports
   CER 0.0116 vs bf16 on six documents. Do not expect all-token exactness from a 4-bit decoder
   on long receipts. The PLAN's alternative bar (> 0.94 with divergences documented) is realistic.
2. Every serious port keeps the **vision tower and projector unquantized** (MLX: bf16 by design;
   got.cpp: not in GGUF at all). Ours quantizes them to Q8_0.
3. The 0.5B Qwen decoder itself tolerates Q4_K_M well on text (ppl +0.2%).

## 2. Our recipe and runtime

The tensor census of the published `got-ocr2-q4_k.gguf` (Hub `5a0b0a1adc`) was read from the
GGUF header over an HTTP range request, without downloading the whole file:

| Group | Count | Type |
|---|---|---|
| decoder matrices `l.blk.*.{attn_q,k,v,o, ffn_gate,up,down}` | 168 | **Q4_K** (pure; no Q6_K layers) |
| decoder norms / q,k,v biases | 120 | F32 |
| `l.embed_tokens.weight` (151860×1024; **tied LM head**, no `l.lm_head`) | 1 | **Q8_0** (generic `is_embd` rule) |
| SAM `v.blk.*` qkv / proj / ffn | 48 | **Q8_0** (generic `is_vision_weight` floor) |
| SAM `v.blk.*.attn_rel_pos_{h,w}` (64×27 tables) | 24 | **Q8_0** |
| `v.pos_embed` (additive 4096×768 table) | 1 | **Q8_0** |
| neck convs, `net_2`, `net_3`, `v.projector` | 5 | **Q8_0** |
| `v.patch_embed`, LN weights | 27 | F16 |
| biases, LN biases | 76 | F32 |

Runtime (`src/got_ocr.cpp`): KV cache F16 (also in the exact F16-r2 path, so it is not the cause).
`flash_attn_ext`, NEOX RoPE θ = 1e6 (1-D; GOT has no 2-D RoPE), logits = `mul_mat(embed_tokens, x)`.
The logits therefore come from a Q8_0 matrix with Q8_0-quantized activations. On CPU every
quantized matmul quantizes its activation input (Q8_K for Q4_K, Q8_0 for Q8_0).

Stage evidence already on record (`qualify-41588457181-orig-ref/summary.md`, `probe-41387019092`,
the stage diff against the Python F32 reference):

- F16-r2: all 19 stages cos 1.000000.
- Published Q4_K: **vis_layer_0 already 0.999213 (max_abs 0.77)**, vis_layer_5 0.997745,
  vis_layer_11 0.998014 (max_abs 3.62), projector 0.999013. The decoder then reads 0.995236 at
  llm_layer_0.
- The Q8_0-from-F16 arm has **identical** vision numbers. The vision drift comes only from the Q8_0
  vision policy, not from Q4_K.

## 3. Diagnosis

Ranked by expected impact relative to cost:

1. **Quantized SAM tower, projector and additive tables (high impact, about +85 MB).** The vision
   error is visible from layer 0 and is the same at Q8_0 and Q4_K. Every one of the 256 image
   tokens the decoder attends to is perturbed. Prior art keeps this F16/bf16. MinerU2.5 showed the
   mechanism in this repo: Q8_0 weights with F32 activations stay ≥ 0.999, while the CPU
   quantized-activation matmul against outlier channels does not. `pos_embed` and `rel_pos` at Q8_0
   are additive tables, quantized for no real size gain.
2. **Pure Q4_K decoder, no Q6_K "more bits" layers, no imatrix (medium).** This is weaker than
   llama.cpp Q4_K_M, which is the configuration with the +0.2% perplexity evidence. Our audit puts
   the worst tensors at layer 0 (`ffn_down`, `attn_q`) and `ffn_down` everywhere. Those are exactly
   the tensors Q4_K_M protects.
3. **Attention at Q4_K (medium, cheap to test).** Attention mixes the image tokens into the text
   stream (cf. the InternVL2 attention finding in `tools/quantize.cpp`). Q8_0 attention costs about
   +25 MB here.
4. **Tied LM head at Q8_0 with Q8 activations (low).** It is already above llama.cpp's Q6_K, and MLX
   runs it at 4-bit with CER 0.0116. Test it once (embed F16, +150 MB) for attribution only.
5. **Runtime precision (low; F16-r2 is exact on the same runtime).** KV F16, flash attention and RoPE
   are shared with the exact path. The only runtime lever is activation quantization inside
   quantized matmuls. Keeping F32/F16 activations for the decoder would need a dequantize-to-F16
   weight path. That is follow-up only if the recipes plateau.

## 4. Experiment plan

Every recipe is produced from F16-r2 with existing quantizer flags (default policy unchanged) and
measured in one hosted lane: `tools/ci-heavy/got_ocr2_q4_recipe_search.sh` on `heavy-cpu.yml`.
Measurements per recipe, on fox / receipt_example / simple_form:

- **token-ID agreement** vs the original `chat()` pipeline: exact?, first divergent token
- **CER** and character ratio vs the original text
- **teacher-forced final-logit cosine** vs F16-r2 on F16-r2's own (original-exact) token stream:
  min / mean over steps, top-1 agreement. This uses the new diagnostic env vars
  `CRISPEMBED_GOT_OCR_FORCE_IDS` / `CRISPEMBED_GOT_OCR_LOGITS_OUT`, which are inert when unset.
- min stage cosine (19 stages) vs the Python F32 reference, and file size

| ID | Recipe (base `q4_k`) | Question it answers |
|---|---|---|
| E0 `q4k_repro` | default policy | does it reproduce the published 1/3 (control)? |
| E1 `dec_f16` | `--decoder-f16` | how much of the drift is vision Q8_0 alone? |
| E2 `vis_f16` | `--tensor-type 'v.*=f16'` | vision/projector/tables F16, decoder pure Q4_K |
| E3 `vis_f16_km` | E2 + Q6_K `attn_v`/`ffn_down` on layers 0,1,2,5,8,11,14,17,20,21,22,23 | Q4_K_M-equivalent decoder |
| E4 `vis_f16_attn_q8` | E2 + `l.blk.*.attn_?.weight=q8_0` | attention sensitivity |
| E5 `vis_f16_emb_f16` | E2 + `l.embed_tokens.weight=f16` | LM-head attribution |
| E6 `vis_f16_imat` | E2 + `--imatrix` | calibration effect |
| E7 `vis_f16_km_imat` | E3 + `--imatrix` | best expected 4-bit-class recipe |

The imatrix is collected by the GOT runtime (`crispembed_imatrix_install`, a no-op unless
`CRISPEMBED_IMATRIX_OUT` is set) on F16-r2 over six CC0 repository images that are **not** in the
panel (licences in `tests/regression/images/cc0/MANIFEST.json`).

**Done when** (unchanged from PLAN): a recipe is token-exact on all three cases, or the smallest
recipe with min teacher-forced logit cosine > 0.94 and documented divergences goes to the owner.
Publication, if any, follows the existing licence-gated publish lane and is not part of this lane.

Dispatch (research only; it never uploads):

```
gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/got_ocr2_q4_recipe_search.sh \
  -f isolated_lane=got-q4-search -f timeout_minutes=300 \
  -f pip="torch transformers>=4.49 accelerate pillow gguf numpy safetensors huggingface_hub"
```

Follow-ups if E2–E7 plateau: Q5_K decoder (`q5_k` base plus the same vision rule), and a gated
F16-activation decoder path (dequantize the weight to F16 per matmul) to separate weight error
from activation-quantization error.

## 5. Round-1 results (run 42375883471) — hypothesis 1 rejected

[Receipt](../tests/results/got-ocr2-q4/recipe-search-42375883471.txt). Panel: fox / receipt_example / simple_form.

| Recipe | Bytes | Exact | First div. (receipt/form) | CER (receipt/form) | min logit cos (receipt/form) |
|---|---|---|---|---|---|
| F16-r2 | 1.44 GB | 3/3 | — | 0 / 0 | 1.0 / 1.0 |
| published Q4_K / E0 | 445 MB | 1/3 | 83 / 20 | 0.34 / 0.05 | 0.46 / 0.80 |
| **E1 F16 decoder, Q8_0 vision** | 1.03 GB | **2/3** | — / 7 | **0 / 0.004** | **0.999 / 0.966** |
| E2 F16 vision, Q4_K decoder | 535 MB | 1/3 | 83 / 20 | 0.35 / 0.04 | 0.50 / 0.82 |
| E3 E2 + Q4_K_M Q6_K layers | 548 MB | 1/3 | 3 / 27 | 0.60 / 0.04 | 0.82 / 0.59 |
| E4 E2 + Q8_0 attention | 586 MB | 1/3 | 3 / 16 | 0.55 / 0.07 | 0.27 / 0.95 |
| E5 E2 + F16 embedding | 681 MB | 1/3 | 3 / 16 | 0.55 / 0.05 | 0.72 / 0.87 |
| E6/E7 + imatrix | 535/548 MB | 1/3 | 3 / 27, 3 / 7 | 0.55 / 0.02, 0.55 / 0.004 | 0.66 / 0.73, 0.56 / 0.77 |

Findings: the Q8_0 vision policy is **not** the cause (E1 with Q8_0 vision is near-exact; E2 with
F16 vision is no better than the published file). Every 4-bit-weight decoder fails the long,
digit-heavy receipt page regardless of Q4_K_M layers, Q8_0 attention, an F16 LM head or imatrix,
consistent with the prior art (no exact 4-bit GOT-OCR2 exists; MLX 4-bit CER 0.0116).
Open question: weight precision vs CPU activation quantization (F16-weight matmuls take F16
activations; quantized matmuls quantize activations to Q8_K/Q8_0).

**Round 2** (dispatched next): decoder base Q8_0 / Q6_K / Q5_K with vision at the default Q8_0
policy (`dec_q8_0`, `dec_q6_k`, `dec_q5_k`). If Q8_0 is exact, the answer is weight precision and
the smallest exact base is the candidate; if Q8_0 also drifts on the receipt, the remaining lever
is a gated F16-activation decoder path.

## 6. Round-2 results (run 42382978017) — weight precision confirmed

| recipe | size | exact | CER (fox / receipt / form) | min logit cos |
|---|---|---|---|---|
| f16 (reference) | — | 3/3 | 0 / 0 / 0 | 1.0 |
| published_q4k | — | 1/3 | 0 / 0.345 / 0.052 | 0.456 |
| **dec_q8_0** | **600 MB** | **3/3** | **0 / 0 / 0** | **0.984** |
| dec_q6_k | 503 MB | 2/3 | 0 / 0.356 / 0 | 0.937 |
| dec_q5_k | 449 MB | 1/3 | 0 / 0.002 / 0.070 | 0.107 |

Conclusion: CPU Q8 activation quantization is *not* the blocker — a Q8_0
decoder (which also quantizes activations to Q8_0) is token-exact on all three
pages. The failure is decoder weight precision below 8 bits on long greedy
decodes (the receipt page diverges at token 83 for both Q6_K and Q4_K).
An F16-activation path is therefore unnecessary.

Recommendation: ship a `q8_0` GOT-OCR2 (Q8_0 decoder + Q8_0 vision, ~600 MB)
as the non-F16 option; keep Q4_K labelled experimental (known drift on long
pages); do not ship Q6_K/Q5_K. Done-when for publication: Q8_0 re-run on the
full gold corpus with the standard OCR qualification lane, then publish via
the HF workflow with LICENSE/NOTICE (Tongyi Qianwen research licence).

## 7. Round-3 results (run 42392365607) — no mixed recipe below full Q8_0 is exact

| recipe | size | exact | CER (receipt / form) | min logit cos (receipt / form) |
|---|---|---|---|---|
| mlx_q4_1 (affine 4-bit, vision F16) | 555 MB | 1/3 | 0.421 / 0.017 | 0.739 / 0.602 |
| q8_ffn_gu_q4k (= q4k_attn_down_q8) | 530 MB | 1/3 | 0.067 / 0.026 | 0.878 / 0.936 |
| q8_ffn_q6k (attention Q8_0, FFN Q6_K) | 549 MB | 1/3 | 0.067 / 0.030 | 0.991 / 0.962 |
| q6k_down_v_q8 | 525 MB | 1/3 | 0.188 / 0.017 | 0.978 / 0.955 |
| q6k_attn_q8 | 527 MB | 1/3 | 0.414 / 0.039 | 0.980 / 0.973 |

Findings: the MLX-style affine Q4_1 is no better than Q4_K. Even Q6_K on the FFN
alone, with Q8_0 attention and down projection, breaks greedy exactness on both
long pages; every decoder matrix needs >= 8 bits. Best approximate 4-bit-FFN arm
is `q8_ffn_gu_q4k` (530 MB, receipt CER 0.067 vs 0.345 for the published Q4_K),
a candidate replacement for the experimental Q4 file. Exact recommendation is
unchanged: full Q8_0 decoder (600 MB).

## 8. Gold corpus (57 pages, Kaggle CPU, 7 November 2026)

Kaggle kernels `crispembed-got-gold-{f16,q8-0,q8-ffn-gu-q4k}` (`tools/kaggle/ocr-real-scans/kernel.py`,
modes `native-got-*`, manifest `docs/ocr-gold-corpus-v1.json` via
`tools/ocr-gold-corpus/to_text_manifest.py`), source `166dae4c`. Receipt:
[summary.json](../tests/results/got-ocr2-q4/gold-kaggle-20261107/summary.json). Two Wikisource
pages (`en-form-economic-journal-67`, `en-register-1758-16`) failed their sha256 pin in the F16
and Q8_0 sessions (served thumbnail bytes differed) and were not run; the table uses the 55 pages
common to all three arms.

| arm | size | mean CER | median CER | mean WER | token-exact vs F16 | cap hits | decode wall (55 pages) | peak RSS |
|---|---|---|---|---|---|---|---|---|
| F16-r2 | 1.44 GB | 0.363 | 0.220 | 0.518 | 55/55 | 3 | 3899 s | 3.57 GB |
| Q8_0 | 600 MB | 0.357 | 0.236 | 0.504 | 17/55 | 4 | 2830 s | 2.77 GB |
| q8_ffn_gu_q4k | 530 MB | 0.345 | 0.178 | 0.474 | 11/55 | 2 | 1294 s | 2.70 GB |

Findings:
- Token exactness does not survive real long pages even at Q8_0 (17/55); the 3/3 panel result does
  not generalize. The panel exactness bar is therefore not a usable publication criterion for any
  quantized GOT-OCR2.
- Against the gold text, both quantized arms are **on par** with F16 in aggregate (mean CER within
  ±0.02; per-page swings both ways, worst regression `de-degraded-freiligrath-46` +0.21 / +0.50).
  GOT-OCR2's own gold quality on this corpus is weak (F16 mean CER 0.36), so quantization noise is
  small relative to model error.
- q8_ffn_gu_q4k is 3x faster than F16 in decode wall time on the same panel.
- F16 is not verified against the original Python pipeline on these 55 pages (only on the 3-case
  panel); an original-reference arm would be a separate long Kaggle session.

Recommendation: replace the experimental Q4_K with `q8_ffn_gu_q4k` (still `-experimental`, quality
on par with F16 on the gold corpus, not token-exact), optionally also publish Q8_0 as experimental;
F16-r2 stays the default exact variant. Publication goes through the HF workflow with
LICENSE/NOTICE under the Tongyi Qianwen research licence.

## 9. Original pipeline on the gold corpus (Kaggle CPU, 8 November 2026)

Mode `ref-got` (`tools/kaggle/ocr-real-scans/kernel.py`, original `stepfun-ai/GOT-OCR2_0` `chat()` contract from
`tools/ci-heavy/got_ocr2_orig_probe.py`, F32 CPU, transformers 4.37.2 / torch 2.4.1), three page shards; all
57 pages completed (the pin-retry fix fetched both previously failing pages).
[Receipt](../tests/results/got-ocr2-q4/gold-orig-kaggle-20261108/original.json).

| arm | pages | token-exact vs original | mean CER (same pages) |
|---|---|---|---|
| original F32 | 57 | - | 0.351 (0.359 on the 55) |
| native F16-r2 | 55 | **42/55** | 0.363 |
| native Q8_0 | 55 | 17/55 | 0.357 |
| native q8_ffn_gu_q4k | 57 | 11/57 | 0.346 |

F16-r2 diverges from the original on 13 pages; most divergences are late (token >= 120) and CER-neutral, but
four are early: `fr-old-tachard-siam-102` (token 3), `de-table-meyers-b1-168` (12; native stops at 215 tokens,
original runs to 2972), `receipt-cord-test-20` (25), `de-table-rgbl-1867-403` (52; 458 vs 1542 tokens).
Open: attribute those four (F16 weights/KV vs the F32 original is the first suspect; an F32 native GGUF run
of the same pages separates precision from a pipeline difference). Gold quality is on par in aggregate.

## 10. F32 attribution of the four early divergences (Kaggle CPU, 8 November 2026)

Mode `native-got-f32` converts the original checkpoint at the probe's pinned revision to an F32 GGUF on the
runner; a same-host F16-r2 control ran the same four pages.
[Receipts](../tests/results/got-ocr2-q4/attr-f32-20261108/).

| page | original tokens | F16 first divergence | F32 first divergence |
|---|---|---|---|
| fr-old-tachard-siam-102 | 355 | 3 | 3 |
| de-table-meyers-b1-168 | 2972 | 12 (stops at 215) | 12 (stops at 215) |
| receipt-cord-test-20 | 81 | 25 | 25 |
| de-table-rgbl-1867-403 | 1542 | 52 (stops at 458) | 34 (stops at 1018) |

The F16 control reproduces the gold run exactly (deterministic). F32 diverges at the same tokens on three of
four pages, so **weight precision is not the cause**: a pipeline difference vs the original remains. The
3-case panel (small images) is exact, and these are large real pages, so the first suspect is image
preprocessing (the original resizes the PIL image to 1024x1024 with torchvision BICUBIC, antialiased on PIL
input); the second is the no_repeat_ngram_size=20 logits processor (two pages stop early).
Next (done-when: the earliest differing stage is named): dump native `pixels.f32`
(`CRISPEMBED_GOT_OCR_DUMP_DIR`) and the original `orig.<case>.pixels.npy`
(`tools/ci-heavy/got_ocr2_orig_probe.py`) for `fr-old-tachard-siam-102` and compare; if pixels match,
teacher-force the original IDs and compare per-step logits to find the first divergent step.
