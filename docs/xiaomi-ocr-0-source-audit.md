# Xiaomi-OCR-0: initial source and checkpoint-header audit

This audit is not a conversion, native parity, quality or performance result.
The later original-only GPU run and its limitations are recorded below.

## Immutable inputs and terms

- [Xiaomi-OCR-0 checkpoint](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0/tree/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30), revision `bbd8cf8d21f256b3644b44cb860f3a7cecb93c30`: the Hub card declares Apache-2.0 and names Qwen/Qwen3.5-0.8B-Base as its base. The checkpoint repository does not contain a standalone LICENSE file; preserve the actual card and separately verify applicable terms before redistribution.
- [Base model](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68), revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`: its card declares Apache-2.0 and links its LICENSE.
- [Xiaomi source](https://github.com/xiaomi-research/Xiaomi-OCR-0/tree/b1a9bb4c07060c453785e9151715ccda6b96f0b1), commit `b1a9bb4c07060c453785e9151715ccda6b96f0b1`: Apache-2.0 with separately licensed [third-party notices](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/b1a9bb4c07060c453785e9151715ccda6b96f0b1/THIRD_PARTY_NOTICES.md). Optional layout components and evaluation datasets require their own license audit.
- The config records Transformers 5.8.1. The Apache-2.0 [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/cc832f9055ba11c8c55f918ab4bda9472b910d48/src/transformers/models/qwen3_5/modeling_qwen3_5.py) at the v5.8.1 tag is a candidate reference; freeze and verify the actually loaded files and environment before accepting reference arrays. The recorded version alone does not prove upstream used that exact execution environment.

The Hub tree declares `model.safetensors` as 1,746,937,856 bytes with LFS
SHA-256 `d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348`.
This is a service declaration, not a locally verified full-file hash. An HTTP
range read of the first 60,288 bytes captured the 8-byte header-length field
and the 60,280-byte safetensors JSON header; SHA-256 of that range is
`edc06f8f34bf602e58f6c5fa95c7967ecd158fa47db400ee2af775fcb737efd5`.
The header declares 488 BF16 tensors: 320 under `model.language_model`, 153
under `model.visual`, and 15 under `mtp`. `lm_head.weight` is absent; the
embedding is `[248320, 1024]`. This census observes header declarations only:
it does not validate payload bytes, official model schema or tied-head behavior.
The [complete header census](../tests/results/xiaomi-ocr-0/checkpoint-header.json)
retains each declared name, shape and byte range. The observed ranges are
contiguous, every declared byte count equals twice its element count, and the
last offset plus header length equals the Hub-declared file size. These checks
do not authenticate the unread weight payload.

The [cross-model schema comparison](../tests/results/xiaomi-ocr-0/ovis-meta-schema-comparison.json)
also compares those declarations with the existing OvisOCR2 Transformers
5.16.1 meta-model reference: all 473 shared names and shapes match; the 15
additional Xiaomi names are all `mtp.*`, and only `lm_head.weight` is absent.
This is evidence for a shared main-model layout, not a Xiaomi 5.8.1 live-schema
check, a payload comparison, or proof of tied-head execution.

The pinned revision's [Hub API sibling listing](../tests/results/xiaomi-ocr-0/hub-revision-file-list.json)
was checked separately from the converter's filtered download. It lists
`model.safetensors` and `processor_config.json`, but no
`generation_config.json`, `preprocessor_config.json`, shard index or standalone
`LICENSE`. The listing is a Hub declaration, not a payload hash, and is not
currently enforced by the converter's filtered local-snapshot check.

## Original-only GPU run

The upstream [Ollama guide](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/b1a9bb4c07060c453785e9151715ccda6b96f0b1/OLLAMA.md)
identifies llama.cpp commit `d89651a7b205c03c4a0b13cd0646d400dc929f79` as
another ggml-based implementation to investigate. Its pinned
[vision graph](https://github.com/ggml-org/llama.cpp/blob/d89651a7b205c03c4a0b13cd0646d400dc929f79/tools/mtmd/models/qwen3vl.cpp)
reorders patches before its blocks, but the final pre-block reshape restores
`n_embd` by `n_patches_x * n_patches_y`; it does not reduce block sequence
length by four. Its final merger does reshape to `n_embd * 4` and `n_pos / 4`.
This is a source-level observation, not a verified runtime comparison, and the
two implementations' GGUF tensor layouts are not assumed interchangeable.

The retained [first-run evidence](../tests/results/xiaomi-ocr-0/kaggle-v1-evidence.json)
records a full checkpoint SHA-256 matching the Hub declaration and a live
Transformers 5.8.1 meta schema: 473 main tensor names and shapes match, the
official schema has an additional `lm_head.weight` (the config declares tied
embeddings), and the checkpoint
has 15 additional MTP names. The original eager BF16 model produced raw text
on two example pages. Its third, dense-equations page failed with a GPU
out-of-memory error in vision softmax; therefore the run did not complete and
is not a native or quantization result. Its text hashes were preserved for a
separately labeled SDPA control, which still requires measurement. Pixel hashes
and original processor metadata for all three pages were retained. No output
IDs or per-stage numeric parity were established for the failed run.

The [second capture](../tests/results/xiaomi-ocr-0/kaggle-v2-evidence.json)
used SDPA on a Tesla T4 with the same BF16 checkpoint and Transformers 5.8.1.
The first two processor pixel hashes and decoded text hashes matched the eager
run; this is not token, logit or stage parity. The dense-equations page reached
the 512-new-token limit without EOS, so its transcription is incomplete. The
second timing of each original case was 15.633 s, 56.298 s and 92.785 s,
respectively; these are GPU blueprint observations, not native comparisons.
Peak allocated GPU memory was 10.416 GiB. The evidence includes generated IDs
and processor/grid metadata, but does not capture the loaded implementation
source hash or intermediate activation values. The [third hosted run](../tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json)
recorded the actually loaded model source SHA-256
`641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7`,
matching the source at the pinned Transformers commit. Its embedding and
output head were the same BF16 CUDA parameter and storage. It captured 31
original-only qianziwen input and prefill stages, with the first logit argmax
matching greedy generation. GGUF and NPZ reference artifacts were hash-checked
and preserved separately; they are not native or decoded-parity results. The
export profile now pins the capture and source hash, while the native pins and
alias remain unqualified.

The separate [F32 export/GPU control](../tests/results/xiaomi-ocr-0/kaggle-export-v1-report.json)
used the staged source closure at base commit `15adb672f71f8470a23db1284dafa102515adae9`,
including the uncommitted converter/profile bytes recorded in that report. The
3,421,929,920-byte experimental GGUF has SHA-256
`1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235`.
All 473 exported F32 tensor records round-tripped byte-for-byte; each value also
equalled the independently loaded checkpoint widened to F32 before the GGUF
weights were reloaded into the original BF16 model. On a Tesla T4 the frozen
qianziwen, receipt and dense-equations inputs produced the same IDs and raw
text hashes as the v3 original capture. The dense case remained capped at 512
tokens. This validates that export/reload control only: it is neither native
ggml parity nor evidence that any precision or speed target is met.
The converter was edited after that capture. A separate [v2 GPU export control](../tests/results/xiaomi-ocr-0/kaggle-export-v2-report.json)
then ran the current recorded converter bytes: its [manifest](../tests/results/xiaomi-ocr-0/kaggle-export-v2-manifest.json)
accounts for 488 source tensors as 473 main and 15 excluded MTP tensors, and
the GGUF SHA-256 is identical to v1. All 473 records passed the same checkpoint
and round-trip checks, and frozen IDs and text matched on all three inputs;
dense remained capped. This is still an original-model reload control, not
native inference or quality/speed parity.

The retained 3,421,929,920-byte GGUF was then re-verified offline, without a
model runtime, by [models/verify-gguf-export.py](../models/verify-gguf-export.py):
all 473 F32 payloads reproduce their manifest digests, no tensor is missing or
extra, the 15 `mtp.*` names are recorded as exclusions (488 = 473 + 15), and the
container metadata carries format 4, profile `xiaomi`, stop `[248046]` and the
checkpoint digest. The [verification record](../tests/results/xiaomi-ocr-0/kaggle-export-v1-verified.json)
keeps the counts. 244 matrices appear with reversed extents because ggml stores
`ne[0]` first; that is a header convention, not a payload change.

The first native stage has since been measured on a hosted Tesla T4 against the
[frozen qianziwen prefill](../tests/results/xiaomi-ocr-0/native-patch-v1-report.json).
A test-only probe loaded only `v.patch_embed.proj.weight` and its bias from the
F32 export and replayed the same F32 projection as `vision_cpu`, once on the
native CPU backend and once on a standalone ggml CUDA graph. The two native arms
agree far more tightly than the strict gate requires: maximum absolute
difference `2.384e-06`, minimum per-row cosine `0.99999999999993` over 1,984
tokens. Both arms nevertheless fail the strict `0.002`/`0.99999` comparison
against the archived activation (max `0.011677`, cos_min `0.9999973`).

Reference precision is a plausible contributor to the gap. The upstream
[vision patch embed](https://github.com/huggingface/transformers/blob/cc832f9055ba11c8c55f918ab4bda9472b910d48/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
casts the pixels to `self.proj.weight.dtype` (BF16 here) before the convolution,
so the archive holds a BF16-executed activation widened to F32. Rebuilding the
same projection from the retained operands reproduces the native F32 comparison
([attribution record](../tests/results/xiaomi-ocr-0/native-patch-v1-attribution.json)):
F32 operands and output give max `0.011677`, matching the hosted native
measurement; BF16 operands and a BF16 output give `99.99 %` bit-exact elements
with max `0.00390625`. These are reconstructed arithmetic controls, not an
actual replay of the original BF16 Conv3D. They do not isolate input rounding,
output rounding and backend arithmetic, or rule out a native defect. Both native
arms fail the strict gate against the reference. This does not establish full vision,
decoder, text, quality, or speed parity, and no production CUDA path exists:
the probe is a separate test target, and `project_cpu` and `qwen35_ocr_init`
remain CPU-only.

A later hosted T4 replay ran the pinned original BF16 patch module with the
frozen processor pixels and the authenticated checkpoint. The retained
[sanitized report](../tests/results/xiaomi-ocr-0/precision-replay/report.json)
and [comparisons](../tests/results/xiaomi-ocr-0/precision-replay/comparisons.json)
record bit-exact reproduction of all 1,523,712 original activation values,
including a repeated run; the source projection weight and bias also match the
exported F32 payloads bit-for-bit after BF16 widening. The native CPU F32 output
versus the F32 Conv3D control passes the strict gate (max `0.000004054`), but
the native output versus the original still fails (max `0.011677027`). Casting
the F32 control's input and output to BF16 leaves 269 differing values against
the replayed original, max `0.00390625`; backend/operation rounding therefore
remains observable. These controls identify the original activation and explain
the dominant precision difference for this frozen stage; they do not prove all
native paths free of defects or establish decoded quality. The report's broad
attribution statement should be read subject to that limitation. The executed
runner hash is retained in its source closure. That replay used an existing
Hugging Face token without prior authorization; the current runner has since
been changed to request the public checkpoint anonymously and has not been
rerun. The correct element count is `1984 * 768 = 1,523,712`; an earlier
informal count of 1,524,736 was an arithmetic error.

A [local 21-stage CPU audit](../tests/results/xiaomi-ocr-0/vision-prefill-local-v1/summary.json)
compared the complete qianziwen vision tower. The rotary cosine and sine stages
passed; the other 19 stages failed the unchanged strict gate. The lowest
recorded per-row cosine was `0.963822126` at layer 10; layer 11's maximum
absolute difference was `394.847168`. A maximum element error divided by a
whole-tensor L2 norm is not a BF16 ULP measurement and cannot establish that
these differences are harmless. The local run linked an existing, unpinned
ggml build, so it is not a pinned-toolchain result. A subsequent
[hosted CPU rerun](../tests/results/xiaomi-ocr-0/vision-prefill-hosted-v3/report.json)
used pinned ggml `a06fc0bb46b73cc10011fc90ae4054aee8438aeb` and verified the
model and reference hashes. It also passed only the rotary tables and failed the
other 19 stages. It rejected the negative control, returned native exit `1`, and
recorded hashes and sizes for 21 native trace files in the hosted outputs. The
first failure is patch embed (max `0.0116770267`); layer 10 has cos_min
`0.963821948`, and layer 11 has max `394.849609`. The source closure and
[sanitized log](../tests/results/xiaomi-ocr-0/vision-prefill-hosted-v3/vision-prefill.log)
are retained alongside the report. This is not OCR parity: decoder, image
frontend, decoded quality, quantization and speed remain untested.

For position, a separate [hosted CPU replay](../tests/results/xiaomi-ocr-0/position-replay-v1/report.json)
uses the pinned Transformers 5.8.1 `fast_pos_embed_interpolate` method and the
checkpoint's verified BF16 position table. Its BF16 output matches all 1,523,712
values of the frozen GPU reference bit-for-bit; a repeat also matches. With the
same table widened to F32, the original method and the retained native F32 trace
pass the unchanged strict comparison (max absolute error `0.0000010673`,
cos_min `0.9999999999992472`), although 23,085 values differ. The native-versus-
reference failure is reproduced and a permuted-row negative control is rejected.
This is consistent with a precision effect for this one case, not proof that
precision is the sole cause, and is not evidence of later-layer or OCR parity.
The checkpoint download used anonymous access. Raw native/output traces remain
in hosted outputs; only the reports were retained here.

The anonymous [hosted layer-0 replay](../tests/results/xiaomi-ocr-0/layer0-replay-hosted-v3-summary.json)
reproduced the original T4 archive exactly for all 1,523,712 values and verified
the 12 block tensors. On the same native input, native versus original CPU F32
passes the strict gate (max abs `0.0000114441`, cos_min `0.999999999994`), while
native versus the archived original fails (max abs `0.0336941481`, cos_min
`0.99994755`). The measured decomposition closes exactly; it does not prove
all error is inherited, eliminate a block defect, or establish a unique cause.
This is one layer and one frozen input, not decoded OCR or speed parity.

An earlier proposed bounds change was withdrawn after review. An uncommitted
replacement carries a profile-specific bound into preprocessing and vision;
its standalone preprocessing test accepts the three captured geometries, but
the real tokenizer and full vision path remain unverified. The Ovis bounds test
was corrected as described below, but no native page support is established. The native patch
buffer stores 1,536 floats per patch; the division by 1,536 and then four
remains unchanged.

The replacement is not ready to promote: the image prompt still rejects more
than 1,024 merged tokens, which excludes the receipt and dense captures, and
the larger CPU vision path needs explicit work and memory bounds. The changed
default broke an existing two-field Ovis bounds test (corrected below) and changes
the low-level multi-frame total bound. Xiaomi's validated Unicode-mark-aware
pretokenizer option is not yet passed for the Xiaomi profile. These are
source-review findings, not measured full-pipeline behavior; keep the null
profile pins and experimental gates in place pending real-call-site tests.

The Ovis regression is confirmed and fixed. `tests/test_ovisocr2_image_bounds.cpp`
built a two-field `image_config`, so `max_patches` silently took the ordinary
1,024 default while the test still expected the gated 2,048 bound: the
pre-change file printed `FAIL` and exited 1. The test now derives the bound from
`ovisocr2_patch_limit()` at every gate change and re-asserts it, and prints
`PASS` with exit 0. `tests/test_xiaomi_bounds.cpp` is now a registered target
(`test-xiaomi-bounds`) and passes with `1984`, `5916` and `9184` patches
admitted under both Xiaomi gates and the ordinary default rejecting them. All
touched sources pass an isolated `clang++ -fsyntax-only` check; neither the
repository build nor the hosted build has been run.

## Inference and frontend boundaries

The [checkpoint config](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0/blob/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30/config.json)
declares the dense Qwen3.5 conditional-generation architecture: 24 text
layers, 18 gated-delta and six full-attention layers, width 1024, FFN 3584,
full-attention 8 Q/2 KV heads of dimension 256, and a 12-layer width-768
vision tower. The [existing native blueprint](qwen35-ocr.md) is a candidate
for reuse, not evidence of Xiaomi inference parity. Compare the actual pinned
Transformers implementation and checkpoint schema before reusing any operator.

The [processor config](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0/blob/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30/processor_config.json)
declares RGB, bicubic resizing, normalization mean/std 0.5, patch size 16,
merge size 2 and image pixel bounds 65,536–16,777,216. The upstream
[evaluation entry](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/b1a9bb4c07060c453785e9151715ccda6b96f0b1/evaluation_pipeline/e2e_img2md.py)
additionally calls `fit_image` before sending an RGB PNG. It places task text
before the image and disables thinking. Its document prompt requests body text
in reading order, Markdown, OTSL tables and LaTeX formulas while ignoring
headers and footers; text, table and formula tasks use distinct prompts. Do not
substitute an empty prompt or swap image/text order without a distinct protocol.
The pinned [fit_image implementation](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/b1a9bb4c07060c453785e9151715ccda6b96f0b1/evaluation_pipeline/cropping.py)
downscales only if either edge exceeds 2048 pixels, using Pillow thumbnail with
LANCZOS while preserving aspect ratio; the model processor then has its own
bicubic resize. These are two distinct preprocessing steps.
The upstream [Ollama alignment notes](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/b1a9bb4c07060c453785e9151715ccda6b96f0b1/OLLAMA.md)
report matching input token counts on seven samples but only five exact raw
responses versus HF BF16/MPS; two formula differences persisted with their F32
GGUF control. These are upstream-reported results, not a CrispEmbed measurement.
Their conversion used an earlier model revision and a pinned llama.cpp converter,
which retains norms and selected vision tensors in F32. Neither that retention
policy nor its small smoke establishes this checkpoint's required precision.
The upstream [postprocessor](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/b1a9bb4c07060c453785e9151715ccda6b96f0b1/postprocess/README.md)
distinguishes raw output from repeated-tail cleanup, math normalization and
OTSL-to-HTML conversion. Two-stage results additionally depend on a layout model
and assembly; they are not evidence for a whole-page-only implementation.

The config's top-level EOS is 248046 and text-config EOS is 248044. The source
repository has no `generation_config.json`; determine actual loaded generation,
tokenizer and request stop behavior with the pinned original before deciding a
native stop policy. Preserve both source values separately. The exact resized
pixels, packed patches/grid, template, token IDs, masks and generated IDs must
be observed, not inferred from model-family similarity.

## Known exporter barrier and next gates

An initial metadata-only call to the baseline exporter rejected this config
with `MTP checkpoints need a separately audited export policy`. The experimental
[exporter](../models/convert-qwen35-ocr-to-gguf.py) now has an exclusion path,
but its full conversion and tied-head behavior are not yet verified.
The header independently shows 15 MTP tensors; do not assume there are none,
silently drop them, or claim speculative decoding support. Compare the loaded
original main-model state dict and real ordinary generation with the checkpoint;
then specify a complete, explicit policy for all 488 source tensors, including
any retained non-executed auxiliary weights and tied output head. The exporter
also requires missing metadata handling to be source-specific and audited.

1. On hosted CI, freeze source/license bytes and full checkpoint hash, inspect
   all source tensor shapes/dtypes against the actual pinned Transformers meta
   model, and capture processor/template/stops on owned asymmetric images and
   text, document, table and formula requests. Keep original stages and raw
   output separate from postprocessing. Preserve negative controls.
2. Add a default-off Xiaomi-specific profile and frontend only after this
   evidence; export F32 and F16 with complete source/export type and payload
   manifests and exact roundtrip. Compare preprocessed inputs, vision layers,
   merger, decoder projections, recurrent/KV state, logits, argmax, IDs,
   EOS and decoded/postprocessed text using actual `crispembed-diff`, absolute
   and RMS errors, magnitudes and cosine. Reference probes remain test-only.
3. Establish an original-first independent OCR quality panel before changing
   precision. Run actual `crispembed-quantize` Q8_0 and Q4_K recipes with
   per-tensor types and hashes; measure critical-tensor ablations, task quality,
   truncation and output differences separately. Do not choose precision or
   claim optimized performance from cosine alone. Keep rejected recipes.
4. Only after numerical and decoded gates pass, wire all required public
   surfaces, measure real device placement/paired latency and memory, and
   qualify immutable artifacts and consumers. No alias, publication, release
   or default flip follows from this audit.
