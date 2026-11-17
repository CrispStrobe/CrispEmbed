# German GECToR: prepared-input native backend

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: exact converter and native F32/mixed Q4_K bounded qualification complete;
guarded application, broader quality, device and distribution gates remain separate.
Planning base: `b34a132c6c204b71f5c58ba8bc1705b8769edb9b`.

The [canonical Quds roadmap](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/roadmaps/german-gector-proofing.md)
owns model identity, backend selection and application integration. PLAN retains
open work; HISTORY and PERFORMANCE record the completed native slice.

## Implemented scope and activation

The native lane was activated after the bounded onnx_runtime_dart optimization
round below. The [standalone CPU target](../../tools/gector/README.md) now loads
the exact fine-tuned Proofbird encoder and both trained classifiers. Its small
source closure uses `src/gector.cpp`, the existing GGUF loader and pinned ggml;
it does not configure the monolithic parent target or change general BERT
production behavior.

`CRISPEMBED_GECTOR` is a default-off value-parsed gate checked at creation and
forward execution. The prepared C API accepts one int64 ID sequence of 1–512
tokens and a same-length all-ones mask, with explicit ID/capacity validation,
zero type IDs and positions `0..T-1`. It returns caller-owned full label logits
`[T,5000]` and detection logits `[T,2]`; optional hidden output and 15 trace stages
support diagnostics. Tokenization, first-subword selection, bounded iterative
edits, guards, language policy and UTF-16 offsets remain in Quds. Immutable model
files are required while a mapped context is live.

The [ONNX converter](../../models/convert-gector-onnx-to-gguf.py) authenticates the
pinned source bundle, maps all 201 trained tensors by their graph consumers and
verifies exact GGUF readback. Each trained MatMul matrix is transposed from
source `[input,output]` to writer `[output,input]`, including square attention
matrices; embedding tables retain their source row layout.
The native graph reproduces word-plus-type then position addition, source
LayerNorm epsilon, Erf GELU, separate Q/K scaling and post-softmax NaN-to-zero
behavior. Both trained heads directly consume final encoder states.

The owner explicitly requested bounded local execution and incremental remote
checkpoints because GitHub Actions was starved. This slice used local portable
CPU builds and model probes; no Actions job was dispatched, polled or awaited.
The earlier hosted-execution planning below is historical, not a prerequisite
for continuing this authorized work.

## Completed native qualification

The [retained protocol-v2 evidence](https://github.com/CrispStrobe/quds_office_suite/tree/feat/german-gector-proofing/docs/validation/german-gector/2026-11-17-native) binds model bytes, actual tensor
precisions, native source dependencies and library SHA-256
`9b0b7bcc0a5f19288d5ec3772594c6e64339becb56204276530d48e45c512604`.
The measured source is present at CrispEmbed
`853a1ce6089e5d5486e37307f37051ad6a928745`, with ggml
`3a27b6c665b191cb230be367db5c52d81dd3e2ac`. Both native-build mode flags are off.

| Artifact | Actual tensor census | GGUF bytes |
| --- | --- | ---: |
| Exact F32 control | 201 F32 | 453,355,328 |
| Declared mixed Q4_K candidate | 72 Q4_K encoder matrices, 1 Q8_0 token table, 3 F16 position/head matrices, 125 F32 norms/biases/type-table tensors | 82,756,896 |

Each arm passes all 30 frozen cases and 40 saved passes, with exact input
alignment, both raw head argmaxes, selected labels, corrected text and UTF-16
suggestions against the original runner. Both retain nine baseline fixes and
the same known Swiss clean-control false edit. F32 maximum absolute errors on
the saved heads are 1.90735e-5 for labels and 5.72205e-6 for detection. The mixed
head minimum token cosines are 0.99894685 and 0.99399707. Required synthetic
64/128/256-token probes also pass both head shape and precision gates.

The [precision policy](../precision-acceptance.md) and frozen correction gates
are reported separately. Mixed absolute-error diagnostics still fail the
0.002 budget: the first saved strict difference occurs at the embedding, while
the worst intermediate token cosine is 0.98112354 at layer 11. Those comparisons
use the same-binary native F32 control, not original ONNX hidden-state dumps.
Exact final corrections do not establish broader German quality.

Protocol v2 adds explicit required-long-probe gates, exact mixed-parent/F32
control binding, and before/after hashes for the qualifier and its imported
policy helpers. Both arms were rerun with unchanged native binary/model bytes.
The archive retains the earlier successful protocol-v1 reports, gate-lifetime
and source-drift failures, manifests, source censuses and all timing samples.
Local checks passed 46 converter tests, 50 native contract checks and 38
qualifier tests with 136 subtests. An independent read-only audit recomputed
both final acceptance results and verified the source/binary/parent linkage.

On the observed AMD EPYC 9V74 host with two native threads, one warmup and three
measurements, the mixed short first-forward median/p95 was 19.477/35.037 ms;
complete unguarded reference correction was 25.255/54.892 ms. The latter includes
Python tokenization/iteration/decoding and excludes Quds guards, dictionaries,
worker transport and editor work. [PERFORMANCE](../../PERFORMANCE.md) retains
F32, longer-input and process-memory measurements. The mixed 256-token forward
already exceeded 500 ms; neither short timing nor GGUF size qualifies the
complete guarded checker on an intended device.

## Remaining application and distribution gates

Quds owns the optional provider, manual/F7 integration and its separate
application receipts. Preserve explicit de-DE support and unavailable de-AT /
de-CH until locale policy is qualified. Broader held-out quality, benefit over
existing deterministic rules, target-device latency/memory, cancellation and
editor responsiveness remain deployment work.

The actual build receipt inventories 31 translation units, installed license
files and five resolved system libraries. Native source components are MIT;
model assets retain Apache-2.0 LICENSE/NOTICE. The measured GNU/Linux binary
also resolves libstdc++/libgcc under the GCC Runtime Library Exception and
libc/libm/loader under their LGPL/component notices. These unbundled system
runtimes are not MIT components, and the strict shipping-license policy remains
a separate platform/toolchain decision. The source closure does not approve
an arbitrary binary, optional backend or distributable model pack. No default
promotion, release, registry alias, consumer pin or main merge follows.

## Historical source audit — before implementation

The following planning and source audit records why the native lane was chosen.
Its statements about unimplemented conversion or hosted execution describe
that earlier checkpoint; the completed local results above supersede them.

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

## Historical backend decision after the bounded Dart round

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
