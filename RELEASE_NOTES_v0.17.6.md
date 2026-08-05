# CrispEmbed v0.17.6

A reranker-quality and reliability release. If you serve rerankers through the
HTTP server or use the mxbai / ms-marco / jina / bge cross-encoders, upgrade:
this fixes a server process-abort, two classes of silently mis-converted
reranker artifacts, and a family of environment gates where spelling a feature
*off* turned it *on* — in one engine, with a segfault.

## Fixed: `POST /rerank` could abort the whole server process

`crispembed_rerank_batch` (the server's `/rerank` endpoint) read 2-D
`classifier.dense` / `pooler.weight` tensors with a raw
`ggml_backend_tensor_get` sized as `H*H*sizeof(float)` — an out-of-bounds read
for any quantized weight, and the quantizer does quantize both. One `/rerank`
request against a quantized two-layer-head reranker (jina-reranker-v2, and any
pooler-bearing artifact) killed the entire process with
`"tensor read out of bounds"`.

The single-document CLI path already had the dequant-safe read; the batch path
had a *duplicated copy* of the cache population that never received the fix.
The duplicate is gone — both surfaces now share one lazy, dequant-safe cache.

## Fixed: mxbai rerankers re-shipped — the old artifacts had no scoring head

The published `mxbai-rerank-{xsmall,base}-v1` GGUFs were converted without the
DeBERTa ContextPooler stage: they scored `dot(CLS, w) + b` where HF computes
`classifier(gelu(pooler(CLS)))`. Calibration was destroyed (scores ±0.3
against a ±6 reference) and xsmall's ranking was near-inverted, with the wrong
top document on both test queries.

Both models were regenerated from fresh upstream checkpoints, gated on decoded
scores against the official ONNX exports before upload (f16 max |Δ| ≤ 9e-6,
orderings identical), and shipped as new `*-g7c.gguf` files — the old files
remain so released binaries' SHA pins keep working. The ContextPooler also now
uses erf-exact GELU (matching HF) instead of the tanh approximation;
`CRISPEMBED_RERANK_POOLER_GELU_ERF=0` restores the old curve.

The ms-marco MiniLM rerankers had the same class of defect (missing
BertPooler: ±0.2 instead of ±11) fixed the same way — `-g7c` artifacts,
re-pointed pins, f16 within 0.0009 of the ONNX reference.

## Fixed: environment gates where `=0` meant ON

Dozens of boolean gates were resolved with `getenv(X) != nullptr`, so
`FOO=0` — the spelling an operator uses to turn something off — enabled it.
Three sweeps, all value-parsed now (set, non-empty, and not `"0"` ⇒ on), one
shared helper (`core_env::on`), hermetically tested:

- **`deepseek_ocr2`**: 11 `DS_*`/`DS2_*` gates, including `DS2_FORCE_CPU=0`
  force-selecting the CPU backend.
- **Every `CRISPEMBED_*_BENCH` gate**: 68 sites across 60 files (all
  diagnostic-only; `=0` used to print the benchmark lines it should silence).
- **`unlimited_ocr`**: 40 sites over 17 `UOCR_*` variables — and here the
  inversion was crash-severity: **`UOCR_PD=0` turned the experimental
  persistent-decode path ON and segfaulted the process with empty OCR
  output.** With the fix, `UOCR_PD=0` and unset are both genuinely off.
  (`UOCR_DBG=1` now also prints a gate-resolution line, like deepseek's.)

## Better 4-bit rerankers: imatrix re-collection and re-pins

Six of the seven published reranker importance matrices carried a defect: the
runtime pre-merges BERT attention q/k/v, so their statistics were filed under
an unnamed tensor and the q/k/v weights quantized with *no* importance. All
seven were re-collected on corrected bases with full coverage:

- `jina-reranker-v2-base-multilingual-q4k` now points at the `-f7` quant —
  mean |Δscore| vs f16 down ~25% on both CPU and Metal.
- **New alias** `bge-reranker-v2-m3-q4k` (first sub-Q8 option for this
  family): Kendall-τ vs f16 0.920 → 0.942 (CPU) / 0.947 (Metal), |Δscore|
  −29/−33%, 25% smaller than q8_0.
- q8_0 stays the default tier everywhere (τ 0.98–1.00).

All re-pins were cross-checked locally on Metal *and* CPU before promotion;
the quantize pipeline now fails instead of silently shipping no-importance
quants when an imatrix matches zero tensors.

## Quantizer: attn q/k/v importance provenance is now selectable

Investigating a suspected mxbai regression surfaced a real (if ultimately
harmless) subtlety: DeBERTa-v2 applies the q/k weights a second time to
relative-position embeddings, so the imatrix collector files direct per-weight
entries with rel-position statistics that shadow the token-stream statistics.
Measured across q4_k/iq4_xs/q3_k on both mxbai models against the official
ONNX references, the shipped behavior is not a quality defect — the default is
unchanged and bit-identical — but the resolution is now explicit and
A/B-able: `CRISPEMBED_QUANT_IMATRIX_QKV=direct|merged|sum`.

## Also

- Windows CI: the env-gate test used POSIX `setenv`/`unsetenv`, which MSVC
  lacks; fixed with `_putenv_s` (the Windows CRT cannot represent a
  set-but-empty variable, so that check is POSIX-only).
- Known issue, pre-existing and opt-in only: the `unlimited-ocr` experimental
  persistent-decode path (`UOCR_PD=1`) segfaults at the second generated
  token. The default decode path is unaffected.
