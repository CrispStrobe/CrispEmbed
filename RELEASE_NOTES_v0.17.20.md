# CrispEmbed v0.17.20

Patch release, 21 November 2026. Changes since [v0.17.19](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.19),
PRs #219-#266.

## Default changes

| Change | PR | Opt out |
|---|---|---|
| `pcs` resolves to the mixed-precision `pcs-xlmr-base-q4_k-mixed.gguf` (q4_k-imatrix with `enc.0` and the truecase head at Q8_0, pinned by revision and SHA-256) | [#266](https://github.com/CrispStrobe/CrispEmbed/pull/266) | alias `pcs-q4k-imatrix` selects the previous file |
| `CRISPEMBED_UNLIMITED_OCR2_FUSED_STEP` is on: Unlimited-OCR-2 uses the fused decode step | [#265](https://github.com/CrispStrobe/CrispEmbed/pull/265) | `=0` keeps the unfused path |
| spaCy: ggml+SGEMM engine and fused transformer graph | [#240](https://github.com/CrispStrobe/CrispEmbed/pull/240), [#252](https://github.com/CrispStrobe/CrispEmbed/pull/252) | see `docs/environment-variables.md` |

- **PCS licence correction.** The PCS model entries are now Apache-2.0, matching upstream (#266).
- **PCS quality.** #260 traced the `I'm OK` -> `I'm oK` loss to summed 4-bit rounding in the
  q4_k-imatrix artifact, not to a runtime operator. `punct-parity` now has a strict q8_0
  ground-truth gate. The mixed file decodes 6/6 and passes 67/67 on all four heads (#266).
- **Unlimited-OCR-2 fused step.** IDs, text, EOS and confidences are bit-identical to the
  unfused path in 21/21 pairs on CPU and T4 CUDA. CUDA decode is 1.13-1.32x faster (#265).

## OCR

- OCR GPU benchmark for DeepSeek-OCR2 and Unlimited-OCR-2 on Kaggle T4 and CPU. Also fixes the
  `minicpm-ocr-graph` link failure and the CPU ccache hit rate in Kaggle kernels (#262).
- OvisOCR2 GPU: opt-in F32 device-matrix precision gates `CRISPEMBED_QWEN35_OCR_GPU_F32_VISION`
  and `_F32_DECODER`, default off. All arms are token- and text-exact (#258).
- LightOn4B: gated source-aligned parity evidence and a gated full-Q6 physical profile, default
  off with no new alias (#231, #234).
- OCR failure outcomes and complete native result records are preserved (#227). Worksheet cleanup,
  ruled-photo perspective and lossless binary PDF export are fixed (#235).

## spaCy pipelines

- Native German raw-text analysis with frozen acceptance (#236). English v2 and French v2 profiles
  are exact against original spaCy (#244).
- Transformer pipelines `fr_dep_news_trf`, `en_core_web_trf` and `de_dep_news_trf` are exact against
  original spaCy. Opt-in BLAS (#247, #253).
- Packages for Linux (glibc 2.34, x86_64 and arm64), Windows, iOS and Android, a process-wide enable
  API, and per-language bundle distribution with published-library consumers (#237-#239, #241,
  #245, #246, #248-#251). Linux cross-binary replay of the grammar v2 qualification (#259).
- Fixes a Windows ABI exception and bumps SentencePiece to v0.2.1 (#252).

## Arabic morphology

- Native Camel Morph MSA analyzer (a camel_tools port) with CC BY database bundles (#254).
- Memory-mapped binary proofing database and compact ABI (#255, #256). Opt-in context
  disambiguator (bigram HMM, ABI 3, default off) (#261).

## Core, bindings and CI

- `core_bert::pretokenize` mirrors CrispASR's optional `out_space_before` output, which fixes the
  sibling CrispASR build. Existing callers are unchanged (#263).
- Model registry lookups resolve all HF tree pages (#232). Source-bound and packaged desktop runtime
  candidates are added (#228, #229). Windows consumer checkouts keep their source bytes (#230).
- Main-health fix: the Arabic morph test exits through `clean_exit` (#257). Fixes for the CI selection
  tests, Kaggle ccache and the merge guard helper (#219, #233). Docs and records (#220, #225, #226, #264).

No pub.dev publication change.
