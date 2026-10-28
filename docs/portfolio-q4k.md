# Portfolio Q4_K closure — active28 October

The owner authorized completing the ordered28 October roadmap and qualifying
Q4_K across Josev, MoEdern1.5M/27M/115M, UltraBERT, Unlimited-OCR-2 and MiniCPM-OCR.
No Q4_K result or release qualification is claimed by this preparation.

## Acceptance boundaries

Every quantized run must first execute its original-precision control against
frozen original inputs, intermediate observations and final decoded/retrieval
outputs. All historical F32/F16 numerical and source guards remain unchanged.
Actual `crispembed_diff::Ref` and independent errors/magnitudes must agree;
structural rejection is distinct from numerical drift. Preserve every failed
recipe, decoded divergence and original hash.

Quantized storage is a distinct profile. Report its full vector/stage drift
against the same strict numerical budgets even where those budgets reject it;
do not call that an F32 pass. Quantized retrieval additionally requires the
existing complete ranking criteria (NDCG/MRR/Recall degradation≤0.001 and
≥0.99 top1 agreement). Exact tokenizer IDs and output coverage remain mandatory.
OCR acceptance checks original generated IDs, actual raw decoded text, stop/cap
behavior and real-page quality; keep protected F16/F32 critical tensors where
necessary. Josev must retain question/option order, typed output/usage and report
logit/probability/ordinal drift; independent decision quality remains a separate
claim from the four authored cases. A conversion alone is not acceptance.

A file labelled Q4_K must contain actual Q4_K tensor payloads. Publish precise
stored-type counts, bytes, critical-tensor policy and any padding. Narrow MoEdern
row widths need explicitly bounded row padding plus zero-padded graph inputs,
or must remain at a supported higher precision. Never describe Q4_0 fallback as
Q4_K. Do not concatenate rows or change semantic shapes to hide incompatibility.
Use the production C quantizer where its shape policy applies, preserving its
critical guards; any new precision override needs authored type/byte controls.

## Current branches and executions

- Coordinator: `work/portfolio-closure-q4k-20261028`. It prepares a default-off
  `CRISPEMBED_ULTRABERT_QUANT` mixed Q4_K profile with F32 vocabulary/norms, actual
  Q4_K presence checks and unchanged legacy F32 validation. Exact tokenizer
  sources from proven02402a0d are integrated, preserving current truncation and
  the default-off Unicode context gate. This fixes a main-versus-audited-runtime
  integration omission before new retrieval measurements.
- Hosted model-free precision controls:40722418008 at4b095f0f9. It compiles the
  actual profile and production quantizer and runs authored override fixtures.
- Full UltraBERT frozen-original retrieval:40724453759 at87eed2035. It authenticates
  existing5483-row inputs and all eight original vector/ID shards, executes real
  Ref positive/one-bad-row/structural controls, then current-root F32 plus three
  declared Q4_K recipes. Every row's actual C API IDs are checked; raw/normalized
  CLS, strict Ref drift and complete300-query ranking metrics are retained.
  Qualified assets remain remote; no original forward is repeated.
- MiniCPM candidate: `perf/minicpm-ocr-graph-q4k-20261028`. New context-owned
  stored-type projection graphs retain the previous scalar norm/attention,
  recurrent state and KV layout. It is separately gated and has no silent
  fallback within a request. The coordinator links its new source with matching
  contraction flags; actual same-input Ref, composed stages and five decoded
  cases are required before a speed/precision claim. Q8 publication also protects
  the four-dimensional patch embedding and waits for the actual benchmark.
- Unlimited candidate: `fix/unlimited-ocr2-trial-findings-20261028`. Existing
  verificationv2 is running, not duplicated. Converter complete added-token
  metadata and card-only refresh are prepared, with a fail-closed original-text
  receipt gate. The published-weight fallback and output alternatives remain.
- Real-scan evaluation is already running from
  `eval/ocr-real-scans-20261028`; review only small results after terminal status,
  checking the declared image and ground-truth licenses independently.

GitHub rejected adding another reusable branch under the metadata dispatcher:
its transitive workflow count reached51 (limit50). The new precision audits have
standalone dispatch entries; this changes no model math or gate. Do not grow the
metadata dispatch graph beyond that limit. The superseded initial model-free
source40721988064 was cancelled while queued; its task-owned receipt is retained.

## Remaining closure

Finish current CI and initial model results before promoting code. Then close
Unlimited text/tiling/card fixes, MiniCPM performance and Q8/Q4 verdicts, real-scan
quality/divergences, unchanged-gate WASM parity attribution, all larger MoEdern
Rust/Dart/HTTP consumers and original-referenced retrieval, and remaining Josev/
MoEdern Q4_K profiles. Final integration requires one declared source, all-platform
builds, Ovis replays, immutable assets and actual consumers. Keep failed WASM,
UltraBERT historical stage errors and all quality limits explicit. No release or
application pin changes follow from source preparation or component cosine.

### CPU fallback for the frozen UltraBERT panel

The [Kaggle worker](../tools/kaggle/ultrabert-q4k/kernel.py) runs the same full
5,483-row F32/control and three declared Q4_K recipes when hosted Actions remain
queued. It records `Kaggle CPU` as its provider and authenticates the exact
submitted source; it does not impersonate a GitHub service run. It reuses the
original preparation/shards and exact successful producer-job identities, with
all archive and member hashes checked before conversion. Redirected artifact
storage requests receive no API credential. Only an admitted CPU session may
run it; preserve the queued-run identity and cancel that run before dispatching
a replacement, to avoid duplicate full panels.

The worker also verifies all 218 tensor names/shapes and the byte identity of
every retained F32 tensor against its lossless base. Vocabulary and norms must
stay F32; each candidate must contain actual Q4_K storage. These source controls
and existing model-free results do not replace the pending actual full retrieval
comparison or authorize a published model.
