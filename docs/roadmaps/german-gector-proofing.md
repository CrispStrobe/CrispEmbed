# German GECToR: conditional Q4_K backend

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: selected for the next native implementation/qualification slice; native backend not yet qualified.
Base: `b34a132c6c204b71f5c58ba8bc1705b8769edb9b`.

The [canonical Quds roadmap](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/roadmaps/german-gector-proofing.md) owns the frozen model,
backend decision, acceptance criteria and application integration.
Coordinate this bounded lane through PLAN.md; follow AGENTS.md/CLAUDE.md.

## Activation and scope

First benchmark onnx_runtime_dart and its separately named ONNX INT4 candidate.
Activate this native lane when measured latency/memory or model-distribution
requirements justify it. Reuse the shared Dart word preparation, alignment,
edit decoder and guards. Implement only inference and exact conversion here.

- Convert the pinned Proofbird v2.1 fine-tuned BERT encoder and BOTH trained
  heads: logits_labels[1,T,5000] and logits_d[1,T,2].
- Start from exact source tensors, preserving transpose/layout, tokenizer,
  special positions, layer normalization and GELU behavior.
- Expose prepared IDs with explicit sequence/mask semantics. Existing single
  sequence BERT raw execution assumes unpadded input; validate all-ones masks
  unless proper masking is implemented.
- Reuse run_encoder_raw and ggml matrix multiplication for both classifiers.
  Preserve raw/debug outputs for parity; return compact predictions normally.
- Add a default-off value-parsed core_env gate and document it if runtime code
  is introduced. Keep existing inference paths unchanged.

## Q4_K recipe and gates

Q4_K is the preferred candidate for eligible encoder matrices. The existing
native quantizer supports it and tensor-type overrides. Preserve sensitive
embeddings, normalization/bias tensors and the two output heads at F16 or Q8_0
when required; record every tensor's actual precision. ONNX MatMulNBits is a
different format and is not a Q4_K import path.

First prove an exact F32/F16 conversion against frozen original ONNX inputs,
both heads and final decoded corrections. Then compare Q4_K/mixed precision
against that control with no new clean-control edits or lost baseline fixes.
Do not qualify from cosine, file size or tokenizer identity alone.

Run builds/model batches on hosted CI in accordance with repository guidance.
Keep large GGUF/raw reference artifacts remote and retain small hashed receipts,
all failures, earliest differing stages, corrected text, latency and RSS.
No release, model publication, registry default, consumer pin or main merge is
part of roadmap publication. PLAN records open work; HISTORY/PERFORMANCE receive
completed evidence. Existing source guards and unrelated lanes stay intact.

## Verified native follow-up — 17 November 2026

The [Quds two-case baseline](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/validation/german-gector/2026-11-17-local-smoke/README.md)
records evaluation source `6bd03365ba3b146586148658b0e03998b36bdc9d` and
Dart source `0b49ca47fa056e6fc122ffe0192a9abb05627622`. Its native ORT
controls are promising; no CrispEmbed or Q4_K model was executed. Keep the
Dart optimization result and the native activation decision separate.
The full [hosted qualification run](https://github.com/CrispStrobe/quds_office_suite/actions/runs/44053903485)
completed successfully; the later optimized result and backend decision are
recorded below. The immutable local baseline retains its earlier
queued-at-capture status.

### Observed model and mapping

Inspection of the frozen ONNX model
`8e3b50beef7d775cedb7b876009e88bcf2e8d6af`, SHA-256
`6d7b43b4bd95c7a00c02c8babc7569d28558df894afe4874d96c202e6ab9a464`,
confirms 12 BERT layers, hidden width 768, FFN width 3072, 12 attention heads
of width 64, vocabulary 31,102 and 512 absolute positions. Its 25 LayerNorm
nodes use axis -1 and epsilon approximately 1e-12; GELU uses Erf. The graph
has 72 encoder weight MatMuls, 24 attention products and two output heads.

These structures match the available
[BERT graph and loader](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/src/crispembed.cpp),
not an existing ONNX import capability. The
[current converter](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/models/convert-bert-to-gguf.py)
loads HF checkpoints and does not implement this dual-head export.

| Frozen source role | Existing/proposed GGUF tensor names | Conversion |
| --- | --- | --- |
| Word / position / type embeddings | `token_embd.weight`, `position_embd.weight`, `token_type_embd.weight` | Retain logical table rows; shapes [31102,768], [512,768], [2,768] |
| Embedding LayerNorm | `embd_ln.weight/bias` | Copy |
| Layer i query / key / value / attention output | `enc.i.attn.q/k/v/o.weight` and matching bias | Transpose each MatMul RHS; copy bias |
| Layer i FFN up / down | `enc.i.ffn.fc1/fc2.weight` and matching bias | Transpose [768,3072] / [3072,768]; copy bias |
| Layer i attention / output LayerNorm | `enc.i.ln1/ln2.weight/bias` | Copy |
| Label head `onnx::MatMul_1682`, [768,5000] | Proposed `gector.labels.weight/bias` | Transpose weight; copy `m.labels_head.bias` |
| Detection head `onnx::MatMul_1683`, [768,2] | Proposed `gector.detect.weight/bias` | Transpose weight; copy `m.detect_head.bias` |

Resolve source matrices through consuming graph nodes, not numbered initializer
order. ONNX MatMul weights are [input,output]; GGUF writer arrays must be
[output,input]. Square attention matrices need this transpose too. Preserve
the fine-tuned encoder; do not substitute the original German BERT backbone.

Structural compatibility is not numerical parity. The ONNX export scales Q
and K separately by 0.3535533845 before their product; the current native graph
passes scale 0.125 to `ggml_flash_attn_ext`. Rounding order differs despite
mathematical equivalence. The source also replaces post-softmax NaNs with zero.
Attribute any mismatch using saved intermediate values before altering a
general BERT path.

### Conditional implementation slice

1. Add an authenticated ONNX-to-GGUF converter with a complete tensor/layout
   census. Preserve exact model/tokenizer/label identity and both heads.
2. Extend the existing loader/graph with validated optional head tensors and
   a default-off prepared-ID dual-output mode. A proposed C API accepts checked
   int64 IDs/masks, batch one, 1..512 unpadded tokens and an all-ones mask; reject
   other masks or overflow. Use zero type IDs and positions 0..T-1. Return both
   full raw logits for qualification; preserve existing embedding modes.
3. Add a bounded native driver and CMake target consuming the existing saved
   IDs and reference heads. Keep tokenizer, first-subword selection, iterative
   edits, guards and UTF-16 mapping in shared Dart. Add consumer bindings only
   after the native model is qualified.

The first deployment candidate is a declared mixed recipe: Q4_K for the 72
encoder matrices, Q8_0 token embeddings, F16 position embeddings and both
classifier matrices, with norms/biases/tiny type embeddings at source precision.
All encoder contraction widths are divisible by 256. Use the existing
[quantizer](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/tools/quantize.cpp)
and inspect actual per-tensor types; a filename does not prove Q4_K coverage
or exclude its Q4_0 fallback. This is a mixed Q4_K recipe, not Q4_K_M.

Use an unquantized control to attribute conversion/runtime error, then test the
candidate on identical inputs, both logits, decisions and complete corrections.
Report the [native precision policy](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/docs/precision-acceptance.md)
separately from Quds gates: no lost baseline fixes or new clean-control false
edits. Preserve the known Swiss false edit. Builds and model runs remain on
hosted CI with portable CPU flags, explicit threads, immutable artifacts and
retained failures. No GGUF conversion, native numerical parity, Q4_K quality,
native latency, release or app integration is qualified by this source audit.

### Dependency and packaging boundary

The source-only review at CrispEmbed
`b34a132c6c204b71f5c58ba8bc1705b8769edb9b` and ggml
`3a27b6c665b191cb230be367db5c52d81dd3e2ac` identified no mandatory
copyleft or non-commercial application dependency in the inspected CPU BERT
sources. This does not certify a distributed binary or every optional feature.

The [current CMake targets](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/CMakeLists.txt)
remain monolithic: static and shared libraries compile the same broad
OCR/document source list, with no native BERT-only switch. Runtime feature
gates do not remove those sources. The
[C2PA helper](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/cmake/CrispEmbedC2pa.cmake)
auto-detects local/system libraries even when `CRISPEMBED_C2PA_FETCH=OFF`;
CrispASR sibling audio, punctuation, language-ID and truecasing directories
also auto-enable when present. Control these inputs explicitly. PDFium is an
optional, explicitly selected runtime library; the
[document license audit](https://github.com/CrispStrobe/CrispEmbed/blob/b34a132c6c204b71f5c58ba8bc1705b8769edb9b/docs/marker-license.md)
separates permissive native code from restricted weights and build-specific
PDFium notices. Model-catalog entries are not evidence that their assets are
included in a grammar package.

The proposed prepared-input slice should therefore include a small native
target with a closed BERT/dual-head and GGUF-loader source set, controlled
[MIT ggml CPU dependencies](https://github.com/CrispStrobe/ggml/blob/3a27b6c665b191cb230be367db5c52d81dd3e2ac/LICENSE),
optional [MIT SGEMM](https://github.com/CrispStrobe/ggml/blob/3a27b6c665b191cb230be367db5c52d81dd3e2ac/src/ggml-cpu/llamafile/sgemm.cpp),
and only the pinned Apache-2.0 Proofbird assets with their retained notices.
Factoring the shared BERT implementation and closing its references is new
source work; an existing flag does not provide this boundary. Keep model
catalog/download code, PDF/OCR and C2PA outside the proposed target.

Hosted qualification must record actual compiled sources, linked libraries,
packaged assets and their applicable license/notice inventory alongside the
numerical and performance receipts. Review the chosen platform/toolchain and
any prebuilt components separately. Preparing this source boundary establishes
neither a permissive binary inventory nor successful native runtime
qualification.

## Backend decision after the bounded Dart round

The [optimized hosted run 44062093617](https://github.com/CrispStrobe/quds_office_suite/actions/runs/44062093617)
used Quds `69d153c0f81596af894e48e21880f0957b963994` and
Dart `b10258682d459847b63e012eaaba58a528e93956`. Both original and
INT4 models passed all 30 cases / 40 saved passes, with declared experiment
metadata checked. INT4 retained the same nine fixes and the same Swiss
clean-control false edit.

On that run's Intel CPU, optimized Dart INT4 first-forward median/p95 was
463/546 ms and peak RSS was 487,698,432 bytes. The native ONNX INT4 control
was 15.8/17.1 ms per forward and 33.2 ms p95 for the unguarded complete
reference correction. The original hosted baseline ran on an AMD CPU;
do not attribute differences between those jobs solely to the patch.

The optimized forward alone misses the provisional 500 ms complete-check
budget on these 5..12-token inputs. Longer inputs, guards and the full Dart
loop remain unqualified. The initial two runtime hypotheses are complete.

Activate this Q4_K lane as the next implementation/qualification target:
authenticated fine-tuned ONNX-to-GGUF conversion, both trained heads,
a small prepared-ID native target with controlled dependencies, F32 control
parity, then the declared mixed Q4_K candidate and downstream correction gates.
Keep shared word preparation, decoding and source offsets in Dart. Native
ONNX through Rust remains an engineering fallback only if this Q4_K path fails
its gates and the native binary can meet the strict permissive-license boundary.

This is an engineering direction backed by measured Dart limits and a fast
native control. It is not evidence that GGUF conversion, native GECToR
execution, Q4_K quality or application integration has already been implemented.

### Native ORT fallback license qualification

The measured ORT 1.23.2 control has an MIT top-level license, but its
[pinned third-party notices](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/ThirdPartyNotices.txt)
include Eigen under MPL-2.0. Treat those timings as a development oracle, not
approval to distribute that runtime under the feature's strict permissive
allowlist. The Rust fallback needs an exact API-24-compatible runtime and its
own complete dependency/license qualification. No such shipping binary has
been qualified here. This reinforces the controlled ggml source boundary above;
it does not remove that target's own compiled-source, linked-library and
packaged-asset audit.
