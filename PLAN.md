> Current execution: [29 October portfolio handoff](docs/handoff-2026-10-29.md).
> Reviewed results are in HISTORY. Experimental runtime branches and running
> workflows are not promoted by this documentation-only main update.

# CrispEmbed — Active roadmap

Updated29 October2026. Completed work, measurements and dated handoffs belong in
[HISTORY.md](HISTORY.md). Detailed current model contracts remain in their model
documents; this file contains pending work, acceptance conditions and ownership.
Read [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md), [contributing](docs/contributing.md)
and [the consumer handoff](docs/current-state-and-next-steps.md) before claiming work.
Start with the [executable next-agent handoff](docs/next-agent-handoff.md), which
records exact sources, remaining work, acceptance checks and artifact expiry.

## 28 October authorized closure and Q4_K

The user requested completion of every ordered28 October lane and Q4_K for all
five fronts: Josev, MoEdern1.5M/27M/115M, UltraBERT, Unlimited-OCR-2 and MiniCPM-OCR.
A converted file is not qualification. Preserve original references, initial failed
recipes and strict existing gates; use separately declared quantized profiles and
full decoded outputs/retrieval metrics. Protect critical head/norm/embedding/vision
weights where measured necessary. Q4_K block-incompatible widths require explicit
storage/runtime support or an honestly recorded limitation, never a mislabeled file.
All heavy work stays on hosted resources. Do not close this effort while required
items remain pending; no new release/app pin until actual final consumers qualify.

| Owner | Lane | Immediate acceptance gate |
| --- | --- | --- |
| `audit/ultrabert-q4k-sharded-20261029` | Original-bound continuation after complete F32 / timed-out first Q4 arm | Freeze exact v2 service, all eight F32 stream/adapter shards and independent metrics without forwards; preserve5483-row strict F32 positive and nine-hour resource failure. First activation only the existing last-four-FFN recipe/shard0 after review, with per-row durable checkpoints and exact018 runtime. All eight shards/full300 rankings are required; no original/F32 replay or automatic24-job batch. |
| `audit/wasm-q8k-cross-admission-20261028` | Test-only retained natural-input IQ4_XS/Q8_K workspace causal A/B | Freeze saved GELU/workspace/scales/output hashes before intervention; same-backend and browser anchors, actual Ref and independent magnitudes precede any production/F32-branch proposal. No model/original rerun or gate widening. |
| `feat/moedern-q4k-padded-20261028` | Explicit padded-Q4_K storage/fast graph for all three MoEdern sizes | Preserve semantic shapes, original F32 loaders/guards and critical tensors; bound physical row padding before allocation, zero-pad only matching graph inputs, require actual Q4_K bytes, original-control/Ref/text/role/score and independent ranking qualification. Source preparation is not model acceptance. |
| `audit/wasm-f32-operator-attribution-20261028` | New same-input norm/flash/GELU/residual/affine replay from retained layer3 arrays | Preserve frozen580/a06, all old failures and unchanged0.002/0.99999; actual native/browser own-input outputs must reproduce captured bytes before operator attribution. No repeated quantized-projection panel or production/default change. |
| `audit/unlimited-ocr2-mixed-q4k-20261028` | Published-F16 protected2/8/22-matrix Q4_K experiments with isolated intermediate probes | Strict four-page original F16 IDs/raw text first; actual Q4_K types and byte-identical exclusions, no fallback; original four-page gold contains IDs/text only, so paired intermediate attribution is labeled separately. |
| Delegated MoEdern full retrieval continuation | Freeze all three original SciFact baselines, then original-bound full native/Q4 replay and all public clients | Authenticate all5,483 rows/vectors/masks remotely before models; unchanged F32 numerical gates and predeclared0.001 retrieval drops/0.99 top-one agreement; no original recapture or publication from component-only proof. |
| Coordinator, `work/portfolio-closure-q4k-20261028` | CI/current-source closure, real-scan review, shared quantizer, Josev/UltraBERT/MoEdern Q4_K, larger MoEdern consumers/retrieval, WASM attribution and final integration | Reuse existing jobs, compare earliest differing stages with actual Ref and final outputs, qualify assets/consumers under the final source; update HISTORY/PERFORMANCE as each lane completes. |
| `fix/unlimited-ocr2-trial-findings-20261028` continuation | Original added-token output/default tiling, converter correction, verified card refresh | Existing8ca9d3de hosted verification must reproduce all four original full texts and IDs; retain output alternatives and frozen references. |
| `perf/minicpm-ocr-bench-publish-20261028` continuation | Collected benchmark, stored-type ggml decoder/prefill/KV cache, protected Q8/Q4_K | Preserve old scalar path; actual five-case original IDs/text plus earliest-stage comparisons before performance/publishing claims. |

## 28 October wrap-up — open lanes

The scoped, ordered task list is in [the 28 October handoff](docs/handoff-2026-10-28.md#open-tasks-in-priority-order):
(1) main CI green at head; (2) merge Unlimited-OCR-2 trial fixes after Kaggle verify +
card refresh; (3) MiniCPM-OCR stage benchmark → optimize, Q8_0 verdict; (4) real-scan
OCR evaluation; (5) successor-assets v2 wasm parity; (6) MoEdern consumers/quality;
(7) OCR coverage beyond panels.

## Execution order

### Next bounded lanes after the completed Ovis audits

The25 October bounded audits and their initial failures are archived in HISTORY;
measurements are in PERFORMANCE. Start one lane at a time, using its exact
sources and acceptance gates below:

The bounded converted-weight cache experiment is complete in
[HISTORY](HISTORY.md) and [PERFORMANCE](PERFORMANCE.md). It remains default-off:
the three-pair warm median gain is only0.797%, with one slower pair and increased
RSS. Do not repeat its successful model runs for the corrected manifest adapter.

1. **Numerical attribution:** use the completed four-grid interpolation A/B and
   eleven-case first-F16 quality replay to freeze final integration source checks.
   Any later precision selection needs new
   development/evaluation inputs after the completed same-weight FFN isolation.
   Preserve all failed recipes, raw strict flags and the unqualified2048 path.
2. **Next permissive implementation:** the MI-GAN fixed256 original/native
   prototype is complete through PR85. Choose its separately scoped product
   integration or the next declared OCR source audit after this closure.
   Code/checkpoint/base licensing and original blueprint come first. LaMa's
   checkpoint terms/FFT and Indra's restricted weight scope remain unresolved.
3. **Delivery/portfolio:** UltraBERT attribution and representative/pair quality,
   immutable successor runtime assets/consumers and the other scoped lanes remain
   below. Ovis3's14-arm/eleven-case/six-interface diagnostics are complete; its
   capped original plain-OCR page still blocks model qualification/alias/upload.

No queued workflow or prepared harness counts as qualification. Large archives
remain hosted; independent review downloads small reports only. Preserve old
Ovis2 first-F16 receipts, calibrated vision budgets and failed quant recipes.
No app/runtime release/default change follows from these experiments.


### LightOnOCR-3-0.8B carry-forward

The following independent v1 panel describes LightOnOCR, separately from
OvisOCR2's qualified published F16. Preserve both reference sets and their limits.

Profiling, bounded parallel-F16 conversion and fixed-receipt completion/prompt
checks are complete; evidence is in [HISTORY](HISTORY.md) and
[PERFORMANCE](PERFORMANCE.md). Image-only/literal runs reach EOS at455/434 tokens
and match original Python exactly; both still emit HTML tables. The conversion
gate remains default-off. Independent-document inputs and scoring are frozen in
[the v1 quality contract](docs/ocr-quality.md). Its completed original/F16 audit
reaches EOS with exact1,404 tokens/stops and15/15 owned table cells, but retains
six prefill and13 book-logit failures. [Results](tests/results/qwen35-ocr/quality.json)
and [quality/timing](PERFORMANCE.md) keep recognition errors visible. The completed [book isolation](tests/results/qwen35-ocr/book-isolation.json) removes
the book failures with original prefill embeddings; the ordinary composed path
retains them. Audit the upstream vision boundary before quantized promotion. Further performance work remains below.

For every Qwen lane, preserve all 924 prior prefill records and 56 existing failures
(20/11/11/14), the 0.002 cache gate and qualified vision/norm budgets. Use the
faithful attention trace and original-embedding isolation for new divergence;
probe softmax exp/sum boundaries only where evidence warrants a correction.
Original arrays stay test-only. Exact Python output on this panel does not waive
numerical debt or establish transcription accuracy.

The bounded book vision trace and all nine independent-document quantization
measurements are complete; [HISTORY](HISTORY.md) archives their sources, raw
reports and matrix, and [PERFORMANCE](PERFORMANCE.md) records decoded/numerical/
confidence/timing limits. Q4 receipt truncates at1024 with unclosed HTML; all
quantized recipes change receipt/book text and retain numerical failures. No
candidate is selected for publication or default promotion.

1. [ ] **Finish UltraBERT attribution/integration:** use the completed
   same-input operator audit39009703362 and completed M1 evidence to scope
   completed Rust/Dart model-backed surfaces, representative retrieval and
   attribution follow-ups in U3/U4; gated F32 API/owned smoke are merged.
   Keep original183/1294 failures and attribution lanes; do not repeat backend arms.
   Default-off CPU/F32 text support is merged; general model/release qualification
   remains open.
2. [ ] **Resolve OCR correctness/independent quality before publication.** Keep
   existing failed F16/quantized records and held-out gold/caps. HF publication,
   pinned aliases and consumer catalogs require a validated candidate; none is
   selected from the completed quantization panel.
3. [ ] **Continue the permissive OCR queue below**, smallest applicable backend
   first after the preceding work; source/base-license and blueprint audit first.

### OvisOCR3 source-quality decision

Original plain OCR11 capture40178328864 is complete and reviewed; eight authored
cases have zero raw CER/WER, but the two-column page caps1024 without EOS with
CER0.52365/WER0.83993. Preserve that negative result in HISTORY; no cap/prompt/
gold/scorer adjustment. Native14, faithful F16 eleven-case capture and the
six-interface diagnostic are independently reviewed. Preserve all initial
harness/dispatch failures and the source-quality block. Model qualification requires all old screens.
An explicit default-off public diagnostic may verify interfaces only when all11
native outputs faithfully match the original, including its unchanged capped
case. It must emit model-qualified=false and a separate diagnostic review key.
Strict publication and a model alias stay blocked by the original completion
failure. Other upstream tasks require separate development/evaluation protocols.

## Quantization and publication follow-up

Preserve the completed panel's fixed pixels/gold/caps, original/F16 replay,
raw outputs/confidences and all failed precision/calibration alternatives.

- [ ] Further recipe work needs declared development inputs and independent
  evaluation before candidate selection; do not fit weights/prompts/tolerances
  or extend the cap to make this held-out panel pass.
- [ ] Resolve the numerical attribution below and establish an acceptable full-
  document quality/format/completion candidate before HF publication or aliases.
  The current panel qualifies none of the tested quantized recipes for promotion.
  Corpus/upstream benchmark reproduction and larger-page validation remain open.

Publication is conditional on validation. UltraBERT is the next independent
implementation lane while the OCR correctness/publication follow-ups remain open.

## Remaining Qwen numerical attribution

- [ ] **Resolve projection/reduction accumulation before numerical promotion.**
  The completed original-embedding isolation removes the book's5 prefill/13 score
  failures, but the vision trace does not establish a single defective operator.
  First captured composed raw exceedance is block7 norm1 (0.00275558233 versus
  original CPU-backend0.00267150998); original-input block11 down projection is
  0.00268554688 in every arm, norms53287.9883/53287.9922. All280 conditional
  attention vectors pass raw0.002. Next isolate native/Python linear reduction
  order and rounding using original inputs, earliest-stage magnitudes and exact
  native replay controls; do not infer a norm/softmax fix or use the held-out book
  to select/calibrate quantized weights. Any runtime candidate stays gated and
  requires full decoded IDs/EOS/text/confidence/quality A/B. Retain0.99999/0.002,
  all924/56 historical debt, frozen pixels/gold/scorer and all failed alternatives.
  Quantized evaluation may proceed independently; publication/default promotion
  still requires its own complete quality and numerical evidence.

## Permissive OCR queue

Scope: MIT, Apache-2.0, BSD or similarly permissive code **and checkpoint/base**
terms. Inspect actual source/license/dependencies before weight use. Indra,
Chandra-OCR-2, MinerU2.5-Pro, Hunyuan-OCR and dots.ocr remain deferred from this
lane. Existing restricted registry entries are not enabled by this roadmap.

| Order | Checkpoint | License audit | Approximate Q4 storage | Pending work |
|---|---|---|---|---|
| 1 | LightOnOCR-3-0.8B | Apache-2.0 source/base pinned | 0.964GB Q4;1.213GB Q8;1.150GB Q8/Q6;1.918GB F16-text/F32-vision | Numerical drift, latency, independent/large documents, quantized quality and publication; priorities above |
| 2 | [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | Pinned checkpoint/base Apache-2.0 audited | Q4/calibrated Q4 1.119GB rejected; Q8 1.251GB rejected; selected F16 1.717GB | Conditional cache adoption with representative resource evidence; quantized operator attribution;2048 numerical debt and broader licensed-page benchmarks. Preserve qualified F16/alias and all completed evidence in HISTORY. |
| 2b | [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd) | Pinned LICENSE/checkpoint/base Apache-2.0 audited | ~1.12–1.34GB extrapolated from Ovis2 recipes, unmeasured; actual F16 1.717GB | Native14/F16 eleven-case/six-interface diagnostics are complete; [source plain-OCR quality block](docs/ovisocr3.md#original-plain-ocr-quality-decision) remains. Declare a separate original-first task/evaluation protocol before broader tasks, quantization, alias or publication. |
| 3 | [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) | Apache-2.0; Qwen3.5-0.8B-Base | ~0.74 GB | Queued after shared backend; upstream has not validated community GGUF quality |
| 4 | [LightOnOCR-3-1B](https://huggingface.co/lightonai/LightOnOCR-3-1B) | Apache-2.0; retained LightOnOCR-2 architecture | ~0.85–1.23 GB | Queued: reuse `lightonocr`; validate official preprocessing/tokenizer, transcription and grounding; estimated 3–7 engineering days |
| 5 | [TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) | Apache-2.0; base terms still to audit | ~1.3–1.8 GB | Queued: reuse Qwen2.5-VL where exact config permits; verify renamed NaviDC artifacts/provenance and structured output; estimated 1–2 weeks |
| 6 | [LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B) | Apache-2.0; Qwen3.5 family | ~3.08–3.38 GB | Queued after small-model parity; larger memory/long-page and grounding checks |
| 7 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache-2.0; base license text to audit before weights | ~6–7 GB, estimate | Queued after small dense Qwen3.5; general VLM/OCRBench evaluation, not an OCR-specific checkpoint |
| 8 | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Apache-2.0; full dependency audit pending | ~17–20 GB, estimate | Queued: config is dense `qwen3_5`, 64 layers; long-context memory/GPU reference resources required |
| 9 | [Infinity-Parser2-Pro](https://huggingface.co/infly/Infinity-Parser2-Pro) | Apache-2.0; base/dependency audit pending | ~22–25 GB, estimate | Queued last: `qwen3_5_moe`, 40 layers/256 experts, 35.1B total parameters; requires router/expert implementation beyond the dense backend; active parameters do not determine storage |

These are storage estimates, not RAM or verified CrispEmbed exports for queued
models. Critical precision can increase sizes. Keep converters, original Python
references, complete tensor audits, every-layer/cache errors/norms, IDs/EOS/text,
public interfaces and fresh registry-download checks in each checkpoint's gates.
Existing PP-OCRv6, PaddleOCR-VL-1.6, GLM-OCR, LightOnOCR-2 and olmOCR-2 coverage
does not establish reproduction of upstream OCRBench/OmniDocBench/olmOCRBench
scores. tau-squared-bench is an agent benchmark and adds no OCR model task.

## Text encoders and typed decisions

- [ ] **Full SciFact original-first task audit:** execute the hosted `prepare` phase of
  [the versioned contract](docs/ultrabert-scifact.md); independently review exact
  author/BEIR mapping, data/input/gold freeze, all lengths and original-only timing
  before the8-original→freeze→8-native panel. The explicit right-truncation gate
  stays default-off; fixed vector/ranking budgets, all rows and old failures remain.
- [ ] **UltraBERT native port:** follow [U1–U4](docs/next-agent-handoff.md) and
  [the architecture contract](docs/ultrabert.md). Original reference/conversions
  and8-row native tokenizer/template parity pass; the test-only native graph
  retains183 F32/1,294 F16 intermediate failures. Both strict replay failures
  remain archived. Completed local audit39009703362 passes all21 F32 local probes,
  with six F16 down failures; scoped local targets cannot replace old full-model references.
  M1 original replay is complete; retain its RoPE/hidden differences and small
  CLS errors. Gated F32 public API and owned retrieval smoke pass; F16 raw CLS
  fails all seven cases. Public API39943940607 passes the gated F32 text/CLS/mask,
  C ABI/Python/CLI/HTTP checks at4be9f864. PR74 merged3f3d023a after complete
  Build/Flutter/WASM/style/security CI. Original/native retrieval39950831201
  passes all32 ID/64 vector checks and8/8 top1, all three task metrics1.0 on
  the owned smoke; full evidence is in HISTORY and retrieval-smoke.json.
  PR75 is merged at2f078406; actual Rust/Dart39979622930 passes all42 pooled checks and14 binding ID rows. Pending: numerical attribution and public pair
  support; a separately frozen, licensed representative retrieval benchmark;
  long-context/GPU/performance and validated quantization/publication. Base150M/
  380M masked-LM heads remain later separate ports. Original arrays are preserved through21 January2027; keep their
  original identities and use the recorded preservation run; exact acceptance is in U1–U4.
  [Frozen-input RoPE attribution](tests/results/ultrabert/rope-attribution.json)
  is complete and merged through PR84:1,344 comparisons/336 bit controls pass.
  Empty position0 cannot explain the earlier pre-RoPE Q difference; no
  production RoPE change is justified. Further attribution needs a distinct
  hypothesis using preserved inputs and actual backend fingerprints.
- [ ] **LFM2.5 Encoder:** CUDA F16/T4 reference audit and one-session paired
  CPU/CUDA performance review are complete (HISTORY): all15 cases run faster
  on CUDA, with9/20/649-token pooled medians27.93x/27.64x/120.11x. Next
  validate other GPU devices/precisions, hardware-qualified quantized/MMQ paths,
  paired throughput/RSS and actual GPU scheduling on broader workloads, long-context behavior beyond
  the measured suite and independent task quality. F16 remains default; published
  mixed profiles change low-margin predictions. [Usage/evidence](docs/lfm2-encoder.md).
- [ ] **EmbeddingGemma typed decisions:** independent task-accuracy, GPU and
  performance evaluation. Preserve schema-conditioned rendering/head/calibration;
  the published FP32/Q8 reference checks are not a task benchmark.
  [Contract/evidence](docs/embeddinggemma2-typed-decision.md).
- [ ] **E4 language-quality ranking:** MTEB-JA/JMTEB and German embedding-quality
  evaluation on suitable hosted resources, distinct from encoder numerical parity.
- [ ] **Encoder ecosystem backlog:** reconcile the community architecture matrix,
  fastembed-rs feature gaps and the gated Nomic MoE broadcast path. Adopt the latter
  only after reliable latency evidence; do not infer speed from byte equality.

Indra public-code analysis is complete and archived. Native ONNX-operator/network
inspection, conversion and inference remain outside the selected scope; its
[port contract](docs/indra-ocr/PLAN.md) is retained for a future eligible lane.

## Packaging, releases and consumers

Completed wheel validation and PR/source review are in HISTORY. Remaining work:

[M1 handover](docs/handover-m1-validation.md) is completed: original CPU/BLAS
replay retains intermediate differences while raw/normalized CLS remain close.
The F32 public API/owned smoke lane is merged; continue the remaining binding/
representative-quality gates, retaining the unchanged full-model audit; no repeat of the completed four arms. Kaggle remains for actual GPU experiments per the private environment
guide; current CPU/Apple reference jobs are not interchangeable with GPU runs.

Merged-source CI and live Pages verification are complete (HISTORY23 October).
Require affected exact-source checks again for each new implementation change.

- [ ] **Successor artifact/consumer qualification:** PosFormer default restoration
  is merged viaPR73 at75e87ea6 after all source790f9a6b checks, including Flutter
  aggregation and actual Linux released/default/reference validation. Before any
  tag, qualify actual portable native/WASM release assets, their hashes and
  consumers on affected platforms, including an actual Apple released/default
  comparison. Source success is not an artifact release. Keep legacy `=0` and
  unchanged7/50 quality explicit; preserve per-case native logs on future runs.
- [ ] **Development artifact portability:** PR76 merged1f8ff4d5 and actual
  desktop [39979940916](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39979940916)
  passes Linux, macOS and Windows configure/build/smoke/install jobs with both
  native flags off and the release CPU-baseline checker before compilation.
  [Archived report](tests/results/packaging/development-cpu-baseline.json).
  Cross-host installed bundle reuse remains a separate consumer check; keep
  performance experiments separate.
- [ ] **Release metadata:** keep development VERSION0.17.13 distinct from published0.17.12
  before a new release; do not change the app's immutable release pin as cleanup.
- [ ] **R1:** promote any successor runtime atomically through native/WASM build,
  artifact/hash, regression, model and actual consumer validation.
- [ ] **O1/O2 handwriting:** diagnose full decoded original-checkpoint behavior
  before choosing/training new licensed weights. Current recorded50-image results
  remain7/50 original and0/50 candidate; access/resources are dependencies.
- [ ] **App follow-ups:** physical-device checks, accessibility/localization/
  narrow-layout coverage, fresh math problems and App Review/store presentation.
  Cloud deployment and collaboration remain separate optional product decisions.
  Exact scopes and dependencies: [consumer handoff](docs/current-state-and-next-steps.md).

- [ ] **Issue71 MI-GAN product integration:** the reviewed default-off
  [bounded CPU/F32 prototype](tests/results/migan/native-parity.json) is merged
  through PR85 after platform checks. Original40330674649 and corrected
  native40344396533 complete the frozen
  eight-fixture/494-comparison panel; evidence and the initial stale-observer
  rejection belong in HISTORY. Keep the prototype default-off and excluded from
  ordinary runtime artifacts. Public C ABI/orchestrator/CLI/server/bindings and
  registry/publication require a separate coherent integration and actual
  consumer/error-recovery checks. First freeze licensed real subtitle/scene
  images and masks plus perceptual/seam acceptance before quality claims.
  Other resolutions, F16/quantization, backend claims and batching need their own
  original/native gates. Text-derived masks and static-span/video temporal
  propagation require separately frozen geometry/motion/flicker tests.
  An optional [M1 build-only handoff](docs/handover-m1-migan-build.md) complements,
  rather than replaces, Linux evidence. LaMa remains held on checkpoint terms
  and FFT semantics; issue71 stays open.

## OCR portfolio and performance carry-forward

These tasks survive the historical handoffs. First reconcile each brief with
later code/history; an archived failure or timing is not automatically current.
The detailed original briefs are indexed below, with all evidence preserved.

- [ ] **Larger Qwen pages:** extend the1024-patch bound only after memory,
  geometry and original-reference coverage; never silently downsize production
  inputs. The completed receipt used the existing explicit448×576 derivative.
- [ ] **Further Qwen latency and default adoption:** profile the accelerated
  path before selecting another optimization. Evaluate bounded F32 reuse/head
  caching with actual memory limits (full head 1,017,118,720 bytes), plus x86/macOS,
  thread-count, repeated-context and failure-fallback coverage before any default
  change. Retain the serial and parallel paths. Graph setup/creation were each
  below 0.05% in the original attribution; do not prioritize reuse without new
  evidence. Preserve exact values, finite rejection, all prior numerical records,
  full decoded output/confidence and unprofiled paired timing/RSS gates.
- [ ] **Tesseract/Fraktur:** align page segmentation, line counts/crops, spacing,
  recoder/decoder semantics and dictionary candidates with official output;
  validate beam/sequence confidence against official certainty independently.
- [ ] **Tesseract recognition performance:** controlled crop batching and
  weight/scratch reuse; retain default DBNet routing and gated page segmentation.
- [ ] **PP-OCRv6:** detector geometry acceptance; reproducible artifact-backed
  tiny/small/medium tests; profile recognizer backbone/SVTR/head/readback and crop
  batching. Recheck historical CUDA empty-output/readback reports before fixing.
- [ ] **Orientation:** isolate the recorded Metal SE/depthwise outlier; retain
  safe CPU fallback until measured numerical and decoded behavior passes.
- [ ] **EasyOCR/structured handoff:** complete licensed detector/recognizer
  coverage, validate structured word boxes/order against Tesseract TSV and actual
  LayoutLMv2/v3 `apply_ocr=False` inputs. Track detector/order/recognizer provenance.
- [ ] **T12/O8/O9:** real public-domain corpus/annotation coverage and consistent
  external engine comparisons; correct timing spans and distinguish model
  reference agreement from recognition accuracy and reading-order quality.
- [ ] **T2/O10 preprocessing:** evaluate cleanup, orientation, dewarp, restoration,
  super-resolution and binarization by real OCR outcomes; per-page selection
  depends on trustworthy annotations, not synthetic-reference agreement alone.
- [ ] **T3/T4/T5/T6:** depthwise-kernel work, crop batching, PP-OCR invocation
  overhead and text-height-aware detector resizing, each with decoded A/B gates.
- [ ] **Layout/TableFormer:** GPU value projection and remaining measured layout
  hot spots; T16 TableFormer plus learned reading order and document-structure gold.
- [ ] **R3/O3:** one formula encoder/backend scheduling experiment at a time;
  keep CPU fallback and actual decoded-output/timing evidence. No family-wide flip.
- [ ] **R6/O7:** register-blocked CPU GEMM experiments and per-engine thread
  propagation for im2col paths; accumulation changes require quality checks.
- [ ] **UOCR/ggml replay:** reproduce and isolate persistent-decode/replay lifetime
  faults; retain functioning default paths. Submodule work remains separately scoped.
- [ ] **F2/F3:** Metal pipeline-cache-cap adoption and an embedding CLI thread
  default only after current measurements and consumer compatibility checks.
- [ ] **F4/F5/F6:** SmolDocling vision scheduling, DeepSeek-OCR2 crop contracts
  and quantitative F16 KV effects. Do not port graph caches solely by analogy.
- [ ] **Backend bring-up:** reconcile CUDA conv experiments, probe Vulkan compute,
  then CUDA/Metal device-resident AR decode and residency changes where profiling
  warrants them; allocation backend is not proof of compute placement.
- [ ] **VLM frontend gaps:** Qwen3-VL sharp-edge resize fidelity and jina OCR tiled
  Gundam preprocessing, subject to current permissive checkpoint/base terms.
- [ ] **Restoration quantization:** validate Q8 ESRGAN/SCUNet and portfolio
  quantize-and-run candidates; do not publish the known bad ESRGAN Q4 recipe.
- [ ] **OMR:** reconcile remaining Transcoda decode/vocabulary work and music
  model candidates with current notation/sequence accuracy and checkpoint terms.
  Transcoda remains outside the permissive queue (AGPL code / CC BY weights);
  oemer is reference-only pending a suitable reconstruction design. Completed
  TrOMR and Flova integrations are preserved in HISTORY.
- [ ] **Other VLM candidates:** audit MiniMonkey, InternVL2-4B and
  H2OVL-Mississippi-0.8B against current licensing, architecture and resource
  requirements before scheduling them; historical candidate tables are archived.
- [ ] **T9:** audit control-test coverage and native output paths, not merely
  whether tests pass. Deferred SAM flash-attention work stays low priority.
- [ ] **F10:** coordinate any shared CrispASR changes in that repository; do not
  silently alter sibling code, releases or independent workstreams.

## Original unfinished brief index

Each link points into the frozen source context in HISTORY, not a second active
todo list. Historical priorities, delegation labels and machine commands are
superseded by the execution order and repository rules above.

- [OPEN 2026-09-30 — follow-ups from the Qwen3-VL / jina / Nomic round](HISTORY.md#plan-20261015-L292)
- [Task queue](HISTORY.md#plan-20261015-L353)
- [Task queue — FABLE-tier (graph/decoder semantics, hand-kernel math, subtle numerics; per the standing rule, never delegate the math)](HISTORY.md#plan-20261015-L456)
- [Task queue — OPUS-tier (well-scoped, established patterns, protocol does the heavy lifting)](HISTORY.md#plan-20261015-L495)
- [OPEN — embedding-lane language coverage: E4 only (E1-E3, E5-E7 DONE, see HISTORY)](HISTORY.md#plan-20261015-L526)
- [Remaining work, in value order (orchestrator Fable; per-task tiers noted)](HISTORY.md#plan-20261015-L574)
- [F2 — Metal pipeline-cache cap adoption across the other Metal lanes (delegable)](HISTORY.md#plan-20261015-L675)
- [F3 — Embed-CLI `-t 1` default (coordinator decision, small)](HISTORY.md#plan-20261015-L690)
- [F4 — SmolDocling vision backend port (OWN WORK — graph/residency, do not delegate)](HISTORY.md#plan-20261015-L701)
- [F5 — DeepSeek-OCR2 dynamic-crop contract gap (session-sized)](HISTORY.md#plan-20261015-L715)
- [F6 — Quantify DS2_KV_F16 (delegable, small)](HISTORY.md#plan-20261015-L725)
- [F10 — CrispASR twins (other repo, coordinate before touching)](HISTORY.md#plan-20261015-L732)
- [OCR external head-to-head — CrispEmbed vs Tesseract / EasyOCR / PaddleOCR](HISTORY.md#plan-20261015-L1110)
- [T12 — External-parity arms + bench-span audit for all six families](HISTORY.md#plan-20261015-L1972)
- [T19 — German embedder quality snapshot (official MTEB data, 2026-08-04) + port candidates](HISTORY.md#plan-20261015-L2159)
- [T16 — TableFormer port (MIT) + reading order — the missing Docling half](HISTORY.md#plan-20261015-L2199)
- [T17 — Tesseract quality round: Fraktur decode, WER, crop batching](HISTORY.md#plan-20261015-L2210)
- [T2 — H9 cleanup axis: pick cleanup per page [BLOCKED ON T1]](HISTORY.md#plan-20261015-L2226)
- [T3 — Depthwise convolution: unroll across taps [detector, ~20% of conv time]](HISTORY.md#plan-20261015-L2278)
- [T4 — Batched crop recognition (was H4)](HISTORY.md#plan-20261015-L2303)
- [T5 — Unblock the ppocrv6 lane's per-invocation Metal cost](HISTORY.md#plan-20261015-L2338)
- [T6 — Detector resize rule keyed on text height (was H6)](HISTORY.md#plan-20261015-L2348)
- [T9 — Re-read the control tests for coverage, not correctness (method, ~1h)](HISTORY.md#plan-20261015-L2359)
- [Lane A — local (M1), immediately actionable](HISTORY.md#plan-20261015-L2758)
- [Lane B — offloaded (Kaggle), unblocks the discrete-GPU story](HISTORY.md#plan-20261015-L2808)
- [Lane C — architecture, once Lane B lands](HISTORY.md#plan-20261015-L2830)
- [EasyOCR / LayoutLM compatibility — IN PROGRESS](HISTORY.md#plan-20261015-L2865)
- [O8 — Make corpus provenance and real-world coverage explicit [IN PROGRESS]](HISTORY.md#plan-20261015-L3386)
- [O9 — Benchmark every available engine on shared inputs [IN PROGRESS]](HISTORY.md#plan-20261015-L3406)
- [O10 — Preprocessing inventory, parity, and live outcome gates [IN PROGRESS]](HISTORY.md#plan-20261015-L3472)
- [Community-matrix coverage roadmap — candidate archs to add](HISTORY.md#plan-20261015-L4035)
- [Transcoda OMR decode enhancements (deferred, 2026-07-13)](HISTORY.md#plan-20261015-L4071)
- [Optical Music Recognition (OMR) — models to port (2026-07-12)](HISTORY.md#plan-20261015-L4112)
- [OCR — next-gen models to port](HISTORY.md#plan-20261015-L4352)
- [Feature gaps vs fastembed-rs](HISTORY.md#plan-20261015-L4456)
- [DeepSeek-OCR-2 performance (remaining levers)](HISTORY.md#plan-20261015-L4462)
- [Open performance levers](HISTORY.md#plan-20261015-L4512)
- [Open correctness / infrastructure](HISTORY.md#plan-20261015-L4690)

## Active work in flight

Claim a row on main before implementation and push it at every coherent
checkpoint. Work in an isolated worktree; fetch/merge main before edits. Keep
builds, inference and batches on hosted CI. New runtime alternatives remain
value-parsed opt-ins until quality and performance justify promotion. Retain
failed alternatives, independent references and public evidence; keep credentials
and host locations out of public Markdown. Remove a row when its lane lands.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-10-29 | `audit/gemma-decision-first-q4k-20261029` | Current-source replay and first protected typed-Gemma Q4_K development source | Read encoder audit first. New source must reuse immutable five-probe/145-stage and8 typed references, no original recapture; strict current-source FP32 baseline before actual C quantizer Q4 recipes. Preserve head/projection/temperatures/norms/scalars F32; predeclare broad-backbone and input/PLE-protected Q4 alternatives before outputs, validate actual role/block widths/header/payloads, retain every old/Q8 approximation failure. Existing model and F32-dot gates remain opt-in; no new runtime arithmetic, alias/default/HF publication. Q4 numeric flags and typed probability budget remain fixed, with current Q8/control/client gaps explicit. Source/profile/review before any hosted model run. |
| 2026-10-29 | `fix/moedern-first-row-packed-tokenizer-20261029` | Repair one-case probe tokenizer argument after actual pre-inference failure | Actual Kaggle CPUv2/024 passes setup/auth/build and1 original ordinary forward/strict Ref, then native init rejects empty tokenizer component path;0 native encodes/computations,3 intentional init rejections,no observers. Correct both native calls to pass the authenticated packed GGUF as model AND tokenizer argument and explicitly select existing `init_packed` in a test-only probe adapter. Source tracing shows component-only `init` requires a zero-tensor tokenizer and cannot consume the packed GGUF. Use the adapter for positive/second-owner and OFF init checks; preserve old component path, production arithmetic/gates/maps/counts. Preserve v2 and its one original call. A fresh-host ordinary control remains required before observers on retry; per-job5 original cap unchanged, previous1+retry max5 explicitly6 aggregate targeted attempts, never full corpus recapture/new gold. New source/review/fresh capacity before same-kernelv3. |
| 2026-10-29 | `audit/wasm-live-norm-operands-20261029` | Attribute actual full norm candidate rejection on one frozen sentence | Actual40852835039@64eef completes: component12 norm recoveries pass, but16/20 final and6/16 stage comparisons fail; ON/OFF9/10 output bytes differ, so gate activation is supported. New test-only first-sentence capture must prove13 connected norms/actual geometry, retain live norm input/output pairs and layer3 23-node operands with ordinary/observer/repeat neutrality, replay exact inputs through native a06 norm/candidate and actual Ref before any arithmetic change/full10 rerun. Keep original .002/.99999 and64 negative/defaultOFF; no tensor/precision guesses from cosine alone, no runtime reference injection. |
| 2026-10-29 | `audit/encoder-precision-evidence-20261029` | Read-only audit of existing LFM2.5/typed-Gemma Q8/Q4 evidence | Before scheduling missing work, map existing exact source/runtime/model/recipe/intermediate/Ref/task/published-HF-readback receipts for both precisions. Inspect only public metadata/small reports, no weights/models/builds or old-reference recap. Record which critical-tensor/calibrated recipe is actually qualified versus merely stored, and executable remaining gates/owners. Preserve all failures and existing aliases; no republish/default change from document audit. |
| 2026-10-29 | `audit/moedern-first-row-cpu-disk-profile-20261029` | Declare one-case disk admission from actual Kaggle20GiB boundary | Actual honest Kagglev1/e7ff first-row bootstrap rejects before any command/model at24GiB disk gate:32.67GB RAM,20.94GB disk,4CPUs. Preserve initial failure. Predeclare a defensible smaller disk requirement for one1.5M/307-token diagnostic with explicit source/retained/model/package/build extents and output/reserve caps; retain RAM14GiB/twoCPUs and every numerical/reference/call-count gate. No bootstrap-only bypass, fake provider, original corpus recap or automatic resubmission. New source/profile needs review and fresh whole-account capacity. |
| 2026-10-29 | `audit/unlimited-reused-q4-first-panel-20261029` | Reuse authenticated first Q4 candidate for a bounded numerical/decoded panel | Reviewed40840732942 confirms exact540 files and first model6,481,651,840B/SHA8b9d8f28...,2 Q4/737 F16. New closed caller must preserve old sources, authenticate retained payloads before arrays, match current-host F16 to frozen controls before first Q4 last1 case panel with actual Ref/intermediates and IDs/text/EOS. No original/first-quantizer repeat; last4/all11 and Q8 remain subsequent declared work, not silently omitted or qualified. Prepare/review before dispatch. |
| 2026-10-29 | `fix/minicpm-checkpoint-blob-fetch-20261029` | Repair exact checkpoint manifest identity at seven-only fetch boundary | Actual40838280839 traceback locates frozen fetch.py96/26: checkpoint `.gitattributes` has bytes+Git blob OID, no SHA256. Prior134-member authentication/build passed; no model forward. New closed caller preserves frozen fetch/guards, authenticates retained arrays then checkpoint SHA256/Git blob identities explicitly, checks tiny corruption/unsupported schema before transfer, retains initial failure. Same seven absent arms/four model hashes/42 added stages; no original recapture or threshold/runtime change. |
| 2026-10-29 | `audit/ultrabert-reviewed-q4-shard0-20261029` | Independently review converted conservative model, activate only shard0 | Actual40843704011 conversion-only candidate576997440B/SHA087c9be0...,12 last-four FFN Q4 plus206 F32. Recheck all218 roles and protected payload records/build/source; bind immutable model artifact/report and activate only686 strided rows. Reuse authenticated F32/original freeze; no original or F32 forwards, no other shard/Q8/full qualification. Preserve legacy header/publication-disabled status. |
| 2026-10-29 | `audit/josev-last-projection-precision-20261029` | Isolate final gate/up precision after four completed protected recipes | Actual40838785499: Q8 all/down-protected still fail; final-pair Q4 with F32/F16 down passes both ordinal gates but fails two probability gates (~0.000145/0.000179 vs0.0001). F16-down changes probabilities only~1e-7. Predeclare only finalgate Q4, only finalup Q4 and finalpair Q8, all other projections F16/292 critical F32, same oldfour227/typed/Ref controls. Preserve all negatives/no NEW20 access or threshold change. Native activation/imatrix capture remains a separate later calibration lane if needed. New closed source `3cc96dbab` on this branch passes370 protected/6 owned closure, seven storage controls, five process controls and actionlint; actual hosted execution pending peer review. |
| 2026-10-29 | `audit/moedern-first-row-stages-20261029` | Actual one-document intermediate attribution after retained18 diagnosis | Source24726600 alone is stable against frozen original but differs at input58/token29084 (selectedrow57); two other fresh originals drift and are not replacement gold. New closed caller must pass original ordinary old-strict and native ordinary exact93 all/selected anchors BEFORE observations, capture actual aten softmax/topk/div under FFN context and byte-exact observer/repeat neutrality, masks/IDs/SDPA. No derived-router expected tensors, oldguard bypass, full-corpus/full-original recapture, production change or budget waiver. Explicitly count the new targeted one-case original/native forwards; frozen references stay unchanged. Hosted-only1.5M case. |
| 2026-10-29 | `audit/ultrabert-first-q4-shard-20261029` | Review completed retained F32 freeze; activate only first conservative Q4 model/shard | Actual40834743025 at08b0 succeeds with zero original/F32 forwards. Authenticate its13 small members, durable132MB payload and all old vectors/eight Ref/full300 rankings before source activation. Preserve018 runtime/old37/08 guards and declare only existing last-four-FFN recipe, model conversion then shard0 with per-row checkpoints; no unreviewed multi-shard/recipe batch, repeated original/fullF32 control or Q8 mislabel. |
| 2026-10-29 | `fix/wasm-native-order-norm-candidate-20261029` | Default-off live-input native-order MiniLM norm candidate after independently positive operator risk | Actual40834745535 passes30 anchors/12 byte recoveries/56 Ref comparisons; natural-input FC1 error0.000573, same-input1.43e-6. Base integration8da6636b3; add only a per-context value-parsed WASM384/six-layer/post-LN/epsilon1e-12 candidate, keep native/off/unsupported old ggml_norm path exact. Connected live-input custom norm only, existing affine/quantizers unchanged, no oracle injection. Model-free controls then separately closed full10 oldnative/eightstage A/B with unset/empty/0 and0/1/0 controls; no default flip or original recapture. |
| 2026-10-29 | `fix/josev-protected-source-header-20261029` | Correct actual published F32 source metadata assumption | Actual40834737477 passes strict F32, builds and the C Q8 conversion, then rejects a nonexistent source general.file_type before Q8/native comparisons. The source intentionally omits file_type/quantization_version; immutable quantized outputs still require typed7/14. Retain the initial raw model/reports, fix authored source fixture to match real header, preserve all four recipes/gates and original reference arrays. |
| 2026-10-29 | `audit/moedern-router-topk-operator-20261029` | Source-faithful model-free router selection audit before any tie behavior change | Original27M/115M use Torch2.8.0+cpu softmax→topk(2)→renorm on eight experts; official CPU implementation uses value-only nth_element, with no stable-index guarantee. Predeclare finite tie/ULP/stride/thread controls and a newly hash-locked Torch binary; compare same probabilities and same logits against source-derived C++ and actual a06 ggml. No model/original/full-corpus forwards, replacement gold, production guard change or old-wheel identity claim; later exact three-row capture remains separate. |
| 2026-10-29 | `fix/minicpm-seven-retained-v2-20261029` | Review actual138 retained comparisons; bind only seven missing MiniCPM arms | Actual40831080193 reports completed138=36 carried+102 new, pixels exact, numeric negatives preserved. Independently authenticate/review small proof before a separately versioned seven-only continuation; preserve frozen19c/cec guards and all23 prior arms, no original/native recapture or gate widening. |
| 2026-10-29 | `ci/moedern-q8-controls-20261029` | Hosted model-free compile/quantizer proof for protected all-size MoEdern Q8 profile | Preserve source1f3573137 and actual full-F32 corpus blockers. Build its existing tiny Q8 controls and C quantizer on hosted CPU only; verify actual20/20/52 selected types, critical-role payloads, gates and authored graphs. No model/corpus run or publication claim. |
| 2026-10-29 | `audit/unlimited-retained-gha-fetch-20261029` | Exact-version retained Unlimited v3 delivery and read-only structural audit | Preserve failed v1 bootstrap/v2 missing-mount attempts. Direct exact-v3 GET is accessible (HEAD is unsupported); authenticate Get/Status/every List page, original script and all540 panel files on hosted CPU before array imports. Reuse the retained6.481GB candidate, no original/native forwards or quantizer run; numerical/decoded continuation remains separate. |
| 2026-10-29 | `feat/josev-protected-precision-20261029` | Evidence-led Josev Q8 and two-matrix Q4_K development recipes | Preserve actual A/B/C rejection40787077302, all292 critical F32 tensors and fixed original budgets. Predeclare Q8-all, Q8-with-F32-down, and last-layer-only Q4 gate/up with F32/F16 down alternatives; actual C quantizer/variable-length profile stamp, exact payload inventory and Python intermediate/Ref/final typed comparisons. Old four only, no NEW20 tuning, no local builds/inference or publication before qualification. |
| 2026-10-27 | `perf/moedern-family-fast-20261027` | Default-off MoEdern family fast path `CRISPEMBED_MOEDERN_FAST` | Default family path is ~4.5x slower than the author engine on 27M (per-op graphs, all8 experts dense, scalar attention/RoPE/conv, 1-2 thread cap). Same math as one ggml graph per encode on a persistent allocator: NEOX RoPE theta160000, GQA masked softmax, top-2-only `ggml_mul_mat_id` with renorm denominator+1e-9 and exact boundary-tie rejection, erf GELU, gated dilated conv, threads up to64. Syntax/format only; no run yet. Summation order differs, so acceptance is the unchanged0.002/0.99999 vector and0.002 MaxSim gates vs frozen originals with identical IDs/masks (probe `compare_paths.py`; vago-compare `ours-fast` arms at2/4 threads). Default path and gates unchanged; the Python `_moedern_text.py` owned-hash pin needs re-declaration if promoted. |
| 2026-10-27 | `perf/moedern-quant-20261027` | Default-off quantized weights `CRISPEMBED_MOEDERN_QUANT` (requires `CRISPEMBED_MOEDERN_FAST`) | Author's Q4_0 is ~1.5-2.7x faster than F32 on 27M/115M. `tools/moedern-colbert/quantize_gguf.py` writes Q8_0/Q4_0 GGUFs quantizing only blk.* attention q/k/v/output, routed/shared experts and conv in/out (32-aligned rows); hash embedding+projection, router, dense head, norms, biases and depthwise conv stay F32. Without the gate any non-F32 tensor is rejected as before; F32 default/fast paths unchanged. Synthetic quantizer test only; no run yet. Kernel `tools/kaggle/moedern-quant` reports cosine/max-abs vs frozen originals and docs/s at2/4 threads for F32-fast/Q8_0/Q4_0 on27M and115M; F32-fast keeps the0.002/0.99999 gate, quantized arms are diagnostic. |
| 2026-10-27 | `feat/moedern-public-text-20261027` | MoEdern1.5M public roles and metadata package | Reviewed raw-text1a053116/40455093591 passes ten cases/twenty original vector arrays/three MaxSim pairs. Unfinished draft a925f7e5 prepares role-aware C/Python/Rust/Dart modules and streaming32-tensor/tokenizer repack; seven stdlib/source controls pass, no compile/inference. Shared header/CMake/exports/CLI/server/registry, actual hosted packed byte proof and ten-case/twenty-vector/three-score public recovery/ownership/offline controls remain. Keep generic dense/token interfaces unchanged; default off, no publication before consumer proof; larger variants remain separate. |
| 2026-10-27 | `feat/moedern-text-composed-20261027` | Smallest MoEdern raw-text native composition | Prepare separately versioned CPU/F32 text+role frontend→native model→selected128-vectors/MaxSim; require reviewed actual frontend before execution, preserve old250 primary+21 supplemental lineage. Compile reference observers only in test variant before public wiring; old engine algebra/gold/guards unchanged on frozen sources. Root owns shared CAPI/registry/build integration. Hosted builds/payloads only. |
| 2026-10-27 | `feat/josev-systemone-api-20261027` | Dedicated SystemOne API and bindings | Prepare distinct opaque SystemOne C ABI/context plus Python/Rust/Dart and CLI/HTTP surfaces over packed CPU/F32 owner; keep Gemma flat JSON/768 scoring and generic embedding APIs unchanged. No model alias/publication or public parity claim before reviewed composition/packing/packed runtime; actual original four-case surfaces and failure/recovery/offline tests required. All builds/models hosted. |
| 2026-10-27 | `feat/josev-packed-runtime-20261027` | Josev packed CPU/F32 runtime and dedicated SystemOne capability | Root: prepare one owned442-tensor context with strict borrowed320-backbone/122-head views and embedded tokenizer, preserving old component loaders/gates. Require reviewed native composition and byte-preserving packing before actual packed/public surface audit. New dedicated SystemOne C/Python/CLI/HTTP/Rust/Dart; Gemma768/scoring/flat JSON unchanged, no injected gold/promotion. Hosted builds/models only. |
| 2026-10-27 | `feat/josev-packed-converter-20261027` | Josev self-contained442-tensor GGUF converter | Prepare bounded byte-preserving streaming repack of reviewed320F32 backbone+122F32 head+zero-tensor tokenizer metadata; validate source/shape/byte hashes and exact readback. No new checkpoint conversion, quantization or local models. Execution requires reviewed composed native proof; root owns borrowed packed runtime/public integration separately. |
| 2026-10-27 | `audit/ultrabert-scifact-native-v3-20261027` | Frozen-original representative retrieval after verified Unicode fix | Reviewed token-only40434717493 matches all5483 IDs/masks/full hashes with gate1 while preserving old24/25 gate-off mismatches. Build an explicit new source closure using identical candidate BPE code plus existing integration truncation gate; preserve old e35 workers/guards/8 successful originals, gold and183/1294 strict failures. New actual CAPI token preflight then8 native shards(max2) and aggregate against frozen vectors/ranking gates. No original rerun, default/pair/alias/quantization promotion; all model/corpus/build work hosted. |
| 2026-10-27 | `feat/josev-answer-serializer-20261027` | Native original SystemOne output formatting | Separate internal JSON serializer for native F32 option probabilities and ordered encoded schema. Match exact publisher model/answers/usage, original-choice-order tie selection, noul/score/legend and Python four-decimal ties-even using exact binary64 rounding. Original pure function freezes model-free controls before C++ comparison; preserve Gemma interfaces and frozen frontend producer. No tokenizer/model weights, API/registry/default/quality claim. |
| 2026-10-27 | `audit/moedern-native-dual-reference-v2-20261027` | Primary frozen-native comparison plus supplemental missing stages | After actual SDPA40434080299 reproduces corrected CPU-flash output with1.1921e-7 deviation from old gold, retain the old250 arrays/final vectors/MaxSim as primary targets and independently pinned21 missing pipeline observations as supplemental targets. New explicit CPU/a06 protocol; old v1/strict byte-continuity rejection stays unchanged and unaccepted. Preserve32F32 native model math from79, fixed numeric/mask/vector/MaxSim gates, real Ref and earliest errors; no new gold, tolerance relaxation, public model or quality claim. |
| 2026-10-27 | `audit/ovis-final-integrated-v2-20261027` | Explicit successor final-source validation | Separate v2 protocol/helper/workflow preserves old v1 and all historical guards. Pin reviewed Ovis8 grid/gates and reviewed negative WASM dependency, both defaults off. Final runtime/base/closure remains an explicit closed pending declaration until code decisions finish; then actual same-source frozen14/first-F16 eleven-case replay and archive/real consumers. No network/model access while declaration pending; no dynamic latest-source exception. |
| 2026-10-27 | `fix/ultrabert-unicode-pretoken-20261027` | Gated Unicode-aware UltraBERT pretokenization | Completed tokenizer40424357191 retains24 clipped/25 full mismatches; both two-row details show curly punctuation wrongly grouped as letters. Implement context-scoped default-off Unicode GPT2 byte-level pretokenization using actual original boundaries; preserve old generic/CLIP behavior. Hosted tokenizer-only A/B must retain old mismatch set and match all5483 original clipped IDs/masks/full hashes before any embeddings. Frozen original vectors and ranking/numerical budgets unchanged. |
| 2026-10-27 | `audit/moedern-sdpa-reproducibility-20261027` | Attribute real cross-run Python SDPA drift | Corrected observer40423259251 rejects156/250 old array hashes despite exact inputs/weights and within-run neutrality; first captured divergence is SDPA0. Preserve both immutable originals and strict rejection. Hosted operator-only replay of two frozen attention inputs, actual default/math/CPU-flash kernels, layout/hardware evidence and Ref magnitudes versus both originals. No replacement gold/full-model/native qualification or tolerance changes. |
| 2026-10-26 | `integration/bounded-closure-20261026` | Integrate reviewed bounded lanes and immutable consumers | DraftPR86 records completed Ovis8, WASM attribution, original preservation, SDPA and Josev head evidence. Backbone/native MoEdern/tokenizer component proofs are reviewed; frontend composition, retrieval and public surfaces remain gated; preserve frozen experiment branches. Declare final explicit source closure after code choices, then actual Ovis14/11, native/WASM assets/real consumers, main checks and app documentation mirror. App stays0.17.12. |
| 2026-10-26 | `feat/moedern-colbert-20261026` | SauerkrautLM-MoEdernColBERT family support | All6 original/author-GGUF licenses/sources pinned; smallest original/conversion and SDPA attribution reviewed. Native-v2 actual40441341206 passes10 cases/177 checks/three MaxSim pairs. Finish exact Unigram/roles/public surfaces. Extend27M then115M including6 dilated-conv blocks, independent retrieval/quantization and qualified aliases. |
| 2026-10-26 | `audit/ultrabert-scifact-canonical-integration-20261026` | Complete frozen SciFact retrieval protocol and original-first hosted audit | Canonical v2 preparation is reviewed after strict-v1 rejection and complete legacy/author provenance. Full40375626336 at frozene35 runs8 originals→freeze→8 native→aggregate; exact5483 rows and unchanged ranking/vector gates. No pair API or model publication. |
| 2026-10-26 | `chore/successor-assets-20261026` | Immutable successor archive manifest and fresh consumer qualification | Initial seven-native/two-WASM producers complete; first consumers fail Windows/Apple/browser. Corrected corpus, UTF8 audit, strict cross-desktop aggregation and actual rounding A/B precede fresh final-source producers/all consumers. No tag/app pin change; device limits explicit. |
| 2026-10-26 | `feat/wasm-q8-rounding-candidate-20261026` | Conditional WASM Q8_0 rounding candidate | Isolated ggml890-based fork a06, encoder-only per-backend opt-in and preserved legacy/Q8_1 kernels. Actual40391544739 requires numeric operator controls before ten-case model A/B; candidate stays off before reviewed strict parity/default decision. |
| 2026-10-26 | `test/wasm-q8-rounding-ab-20261026` | Versioned conditional WASM model A/B | [Prepared v1 consumer](docs/wasm-q8-rounding-ab-v1.md) pins reviewed first trace40379523009, freezes ten owned old-native/browser controls before candidate execution, and preserves strict legacy/final/full-stage gates. Actual model A/B is numerically rejected (8/10 final cases; first remaining stage layer3). Preserve actual receipt and attribute same-input operators before a new candidate; no default promotion. |
| 2026-10-06 | `main` (Crisp3DS SAM work) | SAM 2.1 Hiera-tiny image segmentation (`docs/sam2.md`): engine, converter, reference dump, `test-sam2-diff`, C API, CLI `--sam2`, server `POST /sam2/segment`, Python/Rust/Dart `CrispSam2`, registry `sam2.1-hiera-tiny` (F16, default) and `sam2.1-hiera-tiny-f32`, both pinned by hand | Parity with PyTorch CPU: F32 and F16 cos >= 0.999999 at every stage, masks identical; Q8_0 below the 0.999 gate (mask IoU 0.9895 worst, not published), Q4_K not usable. Registry fetch of the F16 verified. **Note for the owner:** `tools/fetch_model_hashes.py --check` reports `model_hashes.h` stale for reasons unrelated to SAM 2 — no pin for `nomic-embed-vision-v1.5-f16` and for `jina-ocr-v1-q4_k` / `-q8_0`, and three pinned files whose upstream bytes changed. Deliberately not re-pinned here. Not yet: 73-photo end-to-end check, Metal timing |

The registry warning in the SAM ownership row describes the pre-repair state;
[PR61](https://github.com/CrispStrobe/CrispEmbed/pull/61) resolves those six artifacts
with preserved digests and complete freshness evidence in HISTORY. SAM quality
and performance follow-ups remain with their owner.

## Active Ovis completion sequence

**Completed:** PR78 frontend, pinned original GPU, native14/14 against943 frozen
fields,27 actual public-interface arms, all11 first-F16 Python ID/text matches,
four rejected quantization attempts, qualified immutable F16 publication and
actual published CLI/HTTP/offline consumer. [PR79](https://github.com/CrispStrobe/CrispEmbed/pull/79)
is merged after final-source platform/Flutter/WASM/style/security checks;
merged-source checks also pass. Exact commits/runs, all initial failures and
measured limits are in [HISTORY](HISTORY.md) and [PERFORMANCE](PERFORMANCE.md).
[Consumer proof](tests/results/qwen35-ocr/ovisocr2-published-consumer.json) and
[publication proof](tests/results/qwen35-ocr/ovisocr2-publication.json) are frozen.
Do not repeat completed captures/panels. No runtime tag/default/app pin change;
development0.17.13, app immutable0.17.12. The optional
[M1 audit](docs/handover-m1-ovisocr2.md) remains complementary Apple coverage.

Continue the pending bounded lanes below. Profiling and the fourth selective
precision attempt are complete; their measurements and rejection are in HISTORY.
The bounded default-off weight cache and first-FFN same-weight attribution are
complete, with limits in their receipts. PR82 and all11 fresh-main workflows
pass at`160520e8`; norm/position attribution and the MI-GAN prototype are now
complete. Finish the corrected composed interpolation panel and delivery gates. Any later precision selection needs a fresh development/evaluation
protocol. Consumer catalogs/providers still require qualified immutable successor
binaries. Preserve all initial failures, old recipes and
raw strict flags. The explicit three entry gates and reviewed `both` math preset
remain required; neither delivery nor speed permits a default flip.

- [ ] **Remote artifact preservation:** the November Ovis3 raw vectors are
  [preserved by actual40287253703](tests/results/qwen35-ocr/ovisocr3-vector-preservation.json):
  original ZIP unchanged, all1,662 source-manifest payloads verified before copy
  and after empty-destination remote readback. Copy12635202371 expires24 January2027;
  original source identity remains40183489780. Other January retention work
  remains: before21 January2027 preserve the
  original871-field archive and before the respective22 January expiries preserve
  full943 fields/native vectors/quality and quantization artifacts. Inventory
  run/artifact IDs and ZIP/member hashes in the existing receipts and HISTORY;
  copy remotely without recapturing/reconverting. Verify source identity and
  every copied member, retain original producer attribution and expiry. Download
  only the small review; never bring multi-GB archives to the shared host.

## Ovis cache adoption — conditional after the completed bounded experiment

Completed source, component, frozen-array, eleven-case and paired resource
evidence is in [HISTORY](HISTORY.md), [PERFORMANCE](PERFORMANCE.md),
[the raw reanalysis](tests/results/qwen35-ocr/ovisocr2-weight-cache.json) and
[the quality receipt](tests/results/qwen35-ocr/ovisocr2-weight-cache-quality.json).
The256MiB decoder cache remains default-off; its current small warm gain does
not justify default adoption. The970MiB full head uses the original fallback.

- [ ] Before any adoption proposal, freeze a representative same-host document
  latency/RSS protocol with identical inputs/caps/outputs, cold versus repeated
  recognition, initialization separate, and one/two contexts. Include a
  meaningful resource/performance acceptance rule before candidate timings.
  Other budgets/admission policies and Apple model execution need their own
  scoped evidence. Keep existing serial/parallel paths and literal0 controls.
- [ ] Any new runtime variant must retain the complete14-arm original/diff
  replay, all eleven first-F16 IDs/raw/processed text/EOS/caps/confidences and
  existing raw/calibrated flags. Do not widen older historical arithmetic
  allowlists or treat the analysis-only recovery as newly generated references.
  No runtime/app release or default change follows from this experiment.

## Ovis quantized-kernel attribution — before another precision selection

All four current attempts are rejected in
[selection v2](tests/results/qwen35-ocr/ovisocr2-quant-attempts-v2.json).
Keep the published F16 bytes/pin and all previous wrappers unchanged.

[Actual first-FFN40231922668](tests/results/qwen35-ocr/ovisocr2-first-ffn-attribution.json)
completes the [frozen five-arm protocol](docs/ovisocr2-first-ffn-attribution.md).
All11,010,048 independent Q4 dequantized values, source input bits, original
reconstructed layer0 and retained rejected native layer0 match exactly. Native
same-weight gate/up/product/down maxima are1.669e-6/1.311e-6/1.192e-6/2.384e-7;
the composed source-input layer control is1.431e-6. The independent original
MLP with those Q4 weights reproduces layer error0.03288919 versus retained
native0.03288949, isolating intended precision loss at this first FFN rather
than a local implementation mismatch. Original/F16 matrix payloads and first
F16 outputs are unchanged. The small ZIP/source/log review covers all20 metrics;
large raw arrays remain remote and were not separately re-downloaded for review.
Two earlier dispatches were cancelled while queued, before model execution.
PR82 is merged at`160520e8` after all22 final PR checks; all11 fresh-main
workflows pass. Tested observer/dumper/driver/build/workflow hashes and native
source tree remain identical to the actual inference run.


- [x] Audit the first prefill FFN with the same candidate weights/activation semantics against
  an independent Python/ONNX blueprint. Start at the first FFN boundary after
  the passing source-fed linear block for the attention-F16/FFN-Q4 file. Use
  `tools/dump_ovisocr2_reference.py`, `tools/ci-heavy/ovisocr2_selective_precision.py`,
  `tests/test_qwen35_decoder.cpp` and pinned ggml quant/dequant/vector-dot code.
  Verify Q4_K scales/values, input bits and any activation quantization before
  comparing gate/up/product/down and full layer output. Keep this quantized
  reference distinct from the immutable original full-precision reference.
- [x] Separate implementation mismatch from intended precision loss at this
  first FFN using original-module gate/up/product/down, absolute errors/norms,
  exact observer/cache controls and the unchanged decoded-quality rejection.
  No correction is justified at the audited boundary. This does not attribute
  later layers/cache phases/full scores or waive document-quality failures.
- [ ] Any later precision/calibration selection needs a separately declared
  development/evaluation protocol. Do not tune prompts, caps, weights or
  tolerances against the already-observed held-out documents. A different
  quantization methodology is a new scoped task, not a rewrite of old gates.
  Publication still requires its own positive quality and integrity receipts.

## Ovis 2048-patch numerical attribution — before broader input claims

[The four-case audit](tests/results/qwen35-ocr/ovisocr2-larger-inputs.json) is
complete. Frozen control and authored1024/1200 inputs pass14/14;2048 passes8/14.
Six composed native-vision/positions/frontend arms first fail prefill.decoder.norm
(F32/F16 max0.00394558907/0.00438690186; calibrated ratios1.073/1.193), then
full-prefix norm. Norms round3187.82251 on both sides; cosine rounds1.0. Replay
with original prepared embeddings passes at0.000289916992/0.000239372253.
All nine IDs/raw text/EOS match, but the overall larger-range qualification is
negative. Default1024 and the existing opt-in2048 alternative remain unchanged.
The F32 native-vision input-embedding difference is3.92198563e-5 (norms
59.2414284/59.2414513); layer23 grows to0.000309944153 (norms
209.240433/209.240448), before the final norm's0.00394558907. Preserve these
magnitudes rather than inferring a root cause from the first threshold failure.


[Retained norm isolation40286291355](tests/results/qwen35-ocr/ovisocr2-2048-norm-attribution.json)
now passes all12 original anchor/identical-input controls: original anchors are
bit exact; same-input native/original norm max3.05176e-5. Original RMSNorm on
native hidden states reproduces composed max0.00394583/0.00438643, retaining
calibrated ratios1.07324/1.19308. Final norm is propagating upstream error.
The trace begins at patch3.81470e-6, position4.52995e-6 and rotary5.96046e-8;
position helper source uses multiply-then-divide while native retains legacy
linspace. The completed position operator audit matches original coordinates/weights/output
exactly. Its source-order alternative remains default-off pending the composed
four-grid replay; neither operator receipt qualifies2048 or waives budgets.

- [ ] Use the byte-exact [producer reports](tests/results/qwen35-ocr/ovisocr2-larger-input-producers/)
  and retained original/native arrays; boundary raw artifact12598011436 is in
  the combined receipt. Preserve original source ac1bb9cb, all initial failures,
  calibrated budgets, pixels, gold, caps and raw strict flags. Download large
  arrays only inside hosted CI; keep shared-host work to small reports.
The composed four-grid A/B and first-F16 replay are completed and reviewed in
HISTORY. Gate0 preserves all old failures; gate1 passes existing calibrated gates
with strict exceedances retained. They do not qualify2048 or a global default.
- [ ] Before any runtime/bound/preset promotion, keep an experimental gate and
  require all original14 numerical/decoded checks at each grid, owning OCR,
  rejection/recovery and final-source frozen14/eleven-case first-F16 controls.
  Measure scoped latency/RSS; original capture uses four threads and native
  owning two, so current timings establish no paired speedup. Do not raise
  tolerances or silently downscale to make the2048 case pass.
- [ ] Upstream2880-squared resolution, dense licensed pages, multi-image/video
  and OCRBench/OmniDocBench reproduction are separate protocols. Owned single-
  text geometry samples do not establish representative large-page quality.

## Pipeline cache identity follow-up

The Ovis-only path-key correction is merged in PR79: ordered Ovis stages
must use their own model_a, and a hosted two-stage negative probe distinguishes
second-file loading from silently reusing the first context. Audit older engine
caches separately, beginning with legacy engine20: ordered stages permit distinct
model paths, while several existing caches are keyed only by engine. Preserve
working paths, add a forced-rejection/second-model regression, and scope any
correction to the demonstrated engine rather than changing all backends at once.

## Downstream OCR provider/catalog follow-up — after qualified runtime assets

CrispEmbed source/model publication does not upgrade a consumer's immutable
runtime. Keep CrispMath pinned0.17.12 until the separate successor-asset and
consumer lane qualifies its actual library hashes. Do not expose engine21 as a
working option against an older runtime.

For CrispSorter, the public source inspected at
[a142496e](https://github.com/CrispStrobe/CrispSorter/tree/a142496ead324b47c83faaa31a5bb7b12f91a04f)
uses Rust/Tauri/Svelte, not the old Dart paths in the contribution guide.
After native/model qualification and a compatible immutable runtime are ready:
update `src-tauri/src/extractors/mod.rs::engine_id`, the qualified default-model
mapping in `src-tauri/src/extractors/ocr_crispembed.rs`, and the actual engine
configuration/selector. Use orchestrator stage21 for qualified OvisOCR2. Stage22 is OvisOCR3 and stays
unqualified until its separate source-quality protocol passes; do not register
it from a positive interface diagnostic. The older `vlm_engine_id` escalation
API uses a different number space. Preserve existing
default engines and the three experimental gates. Acceptance requires actual
selected-library identity, empty-cache model hash resolution, output/confidence,
cap/custom-prompt rejection and fallback-stage behavior. Record the consumer
source/artifact proof separately; do not claim it from vendor component parity.


## Active four-lane closure — actual results and remaining gates

MI-GAN256 CPU/F32 prototype and the OvisOCR3 preservation copy are complete and
merged through PR85/PR83. UltraBERT RoPE attribution is complete through PR84.
Their original failures, qualified scopes and later product/quality tasks stay in
HISTORY and the model sections above. These are not jobs to rerun.

The integration branch `integration/bounded-closure-20261026` assembles the
reviewed source changes without changing any frozen producer branch. It is
unqualified until the actual remaining results and final-source checks below.
Keep unrelated SAM work and the app's immutable0.17.12 pin intact.

- [ ] **Close exact final Ovis integration.** Completed four-grid and eleven-case
  evidence is archived in HISTORY and [the combined report](tests/results/qwen35-ocr/ovisocr2-four-grid-interpolation.json).
  Keep both new global defaults off and2048 unqualified. Define an explicit
  successor source closure accounting for the newly added Josev regression
  fixture, preserving all old protected bytes. The historical
  [v1 guard](docs/ovisocr2-final-integrated-v1.md) must continue rejecting the
  changed tree; do not grant a dynamic exception. Pin reviewed Ovis and WASM
  outcomes, then execute actual final frozen14/first-F16 eleven-case replay.
  Any later profile/default choice needs its own original-first proof.
- [ ] **Finish canonical SciFact retrieval.** The original-first full
  [40375626336](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336)
  uses`e35cd1a0ba3c2dcac1fdfbded739f5663d0b8902`; do not modify its branch
  during the panel. Preparation40372423343 is independently reviewed in
  [the receipt](tests/results/ultrabert/scifact-canonical-preparation.json).
  All5,483 wrapper inputs/masks are frozen, including460 right-truncated
  documents. All eight original shards completed and their exact disjoint5,483-ID union
  was independently checked. The [actual original barrier](tests/results/ultrabert/scifact-original-freeze.json)
  completes22:37:43UTC. Native shards0/1 reject two under-512 document ID
  streams before vector acceptance; [the negative receipt](tests/results/ultrabert/scifact-initial-native-tokenizer-rejection.json)
  retains both exact streams. Six remaining shards are cancelled while queued
  with zero steps. Token diagnostic40424357191 and Unicode A/B40434717493
  are complete (HISTORY); all5,483 IDs/masks/full hashes now match with1 and
  old unset/empty/0 mismatches remain. Reviewed explicit v3ee51e10e is dispatched
  as40444873884; reuse all frozen original vectors unchanged. Finish its native
  shards and aggregate only after per-shard actual ID preflight. Require all rows/IDs,0.002/0.99999 vector gates,
  NDCG@10/MRR@10/Recall@100 degradation≤0.001 and top1 agreement≥99%.
  V1 author-column rejection and1,055 author-abstract differences remain
  preserved; canonical v2 uses exact legacy JSON/Parquet literal columns,
  selected before model outputs. No gold/text normalization, row drops, public
  pair API or model publication follows. Keep strict183/1,294 numerical debts.
- [ ] **Repeat verified Mac compatibility on final integrated archives.**
  [Mac-only40399358753](tests/results/packaging/successor-mac-only.json)
  at immutable7d561149 is complete and positive for initial40 archives:
  all50 released17.12/default/1 IDs, text, EOS and confidences are exact;
  all200 logs verify and legacy0 retains50 original-zero outputs. Corpus
  artifact/RGB checks and23 runtime-source objects pass. The historical7/50
  recognition quality is unchanged. This diagnostic completion belongs in
  HISTORY; no repeat is needed against the same40 archive. The final changed
  runtime still needs fresh seven-native/two-WASM producers and all consumers,
  including the same Mac50 panel. Initial failures40376735641/40354251257 stay
  retained. Audit-only RSS unit normalization is repaired separately without
  changing original producer JSON, image/gold/score or comparison budgets.
- [ ] **Choose a measured WASM correction after completed layer3 attribution.**
  [Corrected attribution40424064413](tests/results/packaging/wasm-layer3-attribution-v1.json)
  reproduces all old anchors and twelve same-input projection replays. Input
  differences amplify through FFN quantization/nonlinear boundaries; no
  same-input kernel defect is established. Retain initial allocation failure,
  full-model8/10 rejection and rounding default-off. Propose a separately scoped
  arithmetic correction only from measured operator evidence, then rerun the
  complete unchanged model A/B before packaged qualification/default changes.
- [ ] **Qualify final immutable assets.** After reviewed runtime changes and
  any justified default decision, freeze one final integrated source. Build
  fresh seven CPU/mobile archives and both WASM bundles from that same source;
  initial producers40339796226/40339800364 at40a82158 do not qualify changed
  runtime objects. Run all desktop C/Python/CLI/cache, Rust/Dart, real browser,
  released/candidate Mac50 and mobile cross-link consumers, then independent
  aggregation. Windows UTF-8 repair still requires its actual hosted retry.
  Pass the reviewed corpus proof through metadata`candidate_run_id`.
  Candidate development manifests are0.17.14; main is0.17.13 before integration.
  Device inference, optional GPU assets, broad model quality, tag publication
  and app upgrade stay separate.
- [ ] **Close documentation and remote integration.** Commit reviewed results,
  move completed work to HISTORY and measurements to PERFORMANCE, refresh
  CLAUDE/executable/shared handoffs and mirror the shared public document in
  CrispMath. Scope every remaining task with source, acceptance and dependency.
  Preserve exact remote January archives before their recorded expiry; the
  urgent November Ovis3 copy is already closed. Private continuity stays ignored.

Queued jobs and successful diagnostic execution are not model qualification.
Retain every failed numerical/quality screen; never replace gold, widen budgets
or regenerate original anchors to turn a panel green.

## Newly requested model families — original-first support

The user also requested the complete
[SauerkrautLM-MoEdernColBERT collection](https://huggingface.co/collections/VAGOsolutions/sauerkrautlm-moederncolbert)
and [Josev-2B](https://huggingface.co/mayflowergmbh/Josev-2B).
These extend the roadmap while the active Ovis/SciFact/immutable-consumer audits
continue on their frozen branches. Do not alter those producer sources.

Completed source/license audits, original captures, exact head conversion,
package provenance, CPU/CUDA head parity and initial failures are archived in
HISTORY with immutable receipts. Do not repeat those jobs or treat their old
preparation commands as current actions.

- [ ] **MoEdern public text and variants:** native-v2 and exact Unigram frontend
  are completed in HISTORY. Raw-text composition1a053116/40455093591 is independently reviewed in
  HISTORY: ten native texts, twenty original arrays and three MaxSim pairs pass.
  Integrate the explicitly gated production engine/frontend under a new source
  profile, preserving ordinary observer exclusion and all recovery controls.
  Author GGUF lacks tokenizer metadata: require a pinned sidecar or byte-preserving
  repack. Add explicit query/document roles through C/Python/CLI/HTTP/Rust/Dart;
  never substitute pooled dense vectors or generic CLS/SEP dropping. Extend27M
  then115M with separate originals and six dilated-conv blocks for115M. Original
  inventories are32/32/94 tensors; narrow1.5M/27M tensors are not automatically
  Q4_K compatible. Keep old byte-continuity rejection and every original hash.
- [ ] **Josev packed model and interfaces:** raw-request composition40449265616
  ated23f1ae passes all227 original stages/four typed responses (HISTORY).
  Streaming converter9772c896 now pins that reviewed receipt; actual
  packing/packed-owner40456235976 is running. Require the exact442 F32 tensors
  and embedded tokenizer metadata, immutable small report and unchanged original gates. Packed runtime
  0f0b1e59 passes model-free build40451936611; require actual packed-owner227
  comparisons and four complete responses before publication. Preserve old
  strict component loaders and Ovis1024/3584 shape guards.
  Public SystemOne candidate5f639e14 implements separate C/Python/CLI/HTTP/
  Rust/Dart paths. Initial40455424346 builds and passes native ABI controls but
  fails a compiler-row selector; corrected40456478353 public-build job passes
  actual root/ABI/Rust/Dart checks (parent metadata controls still queued). Hosted
  Dart output is applied in061d9ba4. Actual six-surface
  hosted orchestrator/workflow/protocol is not written; test consumers are prepared. Review
  compiler arithmetic, C exports/gates/errors, wrappers and CLI/server first,
  then run all four original requests through each actual public surface with
  missing-model/schema/order/tie/rounding/media/error/recovery/offline controls.
  Root Qwen compiler contraction changes require final same-source OCR audit.
  [API boundaries](docs/josev2b.md#public-api-integration-boundaries-after-numerical-parity)
  retain Gemma flat JSON/768-vector semantics. Registry/catalog aliases and
  publication wait for immutable self-contained model and actual consumer proof.
- [ ] **Quality and publication:** only after the above original-first gates,
  freeze independent licensed retrieval/decision panels, assess F16/quantization
  while retaining critical head/norm precision and every rejected recipe, and
  publish checksummed assets/qualified aliases per contributing. Preserve media,
  batching/padding, long context, GPU backbone and task-quality limits explicitly.

All runtime alternatives stay value-parsed opt-ins until their own original,
intermediate and final-output gates pass. Large artifacts remain hosted.

### Independent M1 build-only check

- [ ] The [smallest MoEdern handoff](docs/handover-m1-moedern-build.md) can check
  immutable79e350a0 compilation, weight-free controls and actual Ref synthetic
  positive/one-bad-row rejection while hosted original work queues. It downloads
  no model/corpus/framework and authorizes no model outputs. Return the bounded
  build/control evidence; keep machine paths and local setup notes outside Git.
  Native original-reference guards remain closed until corrected capture review.

## Executable audit continuation — 27 October 2026

All initial failures and original byte hashes are retained. These are execution
gates, not completed model support or release qualification. The app remains
on immutable0.17.12.

- [ ] Integrate MoEdern public query/document roles and self-contained metadata
  after reviewed raw-text40455093591. Then27M,115M and separately frozen
  retrieval/quantization. See the detailed model lane above.
- [ ] Execute/review UltraBERT native SciFact v3 after completed token A/B
  40434717493: all5,483 IDs/masks/full lengths/hashes and14 authored cases exact.
  Preserve unset/empty/0 historical24/25 mismatches, generic/CLIP bodies and
  existing truncation. Assembled runtime02402a0d retains integration truncation.
  Initial40444873884 at ee51e10e failed before all eight native forwards
  because the long-lived Python process could not resolve the newly materialized
  GGML dependencies. Corrected d993038e is dispatched as40452077037: mandatory
  actual fresh-process CDLL smoke, authenticated loader environment and all
  five exports before releasing eight matrix shards (max2 parallel); require every actual CAPI ID stream preflight before
  embeddings; label all-attended masks as source-derived, not a CAPI getter.
  Reuse frozen originals and fixed raw-CLS/dense/ranking gates; retain all
  historical183/1294/M1 numerical debt. See token review in HISTORY.
- [ ] Decide the next bounded WASM attribution from completed
  [layer3 evidence](tests/results/packaging/wasm-layer3-attribution-v1.json).
  All old anchors and twelve same-input kernel replays pass; composed input
  differences amplify at FFN quantization/nonlinear boundaries. Keep the
  initial allocation failure, rejected full panel and rounding default off.
  A proposed correction needs measured same-input evidence or a separately
  scoped arithmetic profile, then actual full-panel A/B; no component waiver.
- [ ] Finish Josev packed/public qualification after completed227-stage
  raw-request parity. Exact active branches, model-free builds and actual public
  acceptance gates are listed above. Independent decision quality, GPU backbone,
  media/batching/long context and quantization remain separate. No original rerun.
- [ ] Finish final same-source Ovis, native/WASM archive producers and actual
  consumers before main/release closure. Both experimental defaults remain off.
  Preserve app0.17.12, no model alias/publication or application pin change.

### Later on 27 October — status of the gates above

Completed (see HISTORY): UltraBERT SciFact native-v3 40452077037 passes with zero
degradation; SystemOne six public consumers 40463407332 pass and
`cstr/Josev-2B-SystemOne-GGUF@bffe6252` is published; MoEdern 1.5M seven public
consumers 40470702013 pass and its GGUF is published at `@6f3b3514`; MoEdern 27M
and 115M frozen-ID native parity pass. Remaining owned gates:

- [ ] Re-run the native Unlimited-OCR-2 decoder after the ODR fix (internal linkage
  for same-named structs in `src/unlimited_ocr.cpp`, `src/deepseek_ocr2.cpp`,
  `src/lfm2_vl_ocr.cpp`) against the frozen three-case
  [original reference](tests/results/unlimited-ocr2/original-reference.json):
  exact token IDs/text/stops before any quality or publication claim. The first
  native runs crashed; nothing native is validated yet.
- [ ] Run and review the hosted Ovis final-v2 validator at declared base
  1a3315f03c34a7ff6b24e59f68763947fb58c8dc (ggml a06fc0bb) on
  `integration/final-candidate-20261027` (model-free build 40475963161 passed).
  No result is claimed until it completes.
- [ ] MoEdern 27M/115M: tokenizer composition, public query/document roles,
  packing, retrieval quality and publication. Only frozen-ID native parity exists.
- [ ] SystemOne/MoEdern 1.5M: registry aliases are on the candidate; independent
  task/retrieval quality, quantization, GPU and release qualification remain.
  Both stay default-off.
- [ ] UltraBERT: public pair API, historical 183 F32/1,294 F16 stage failures and
  strict replay failures remain open despite the SciFact pass.
- [ ] Keep five removed one-off dispatcher phases (scifact-tokenizer-diagnostic,
  wasm-q8-rounding-ab-v1, wasm-layer3-attribution-v1, weight-cache,
  ovisocr3-public-diagnostic) out of the dispatcher unless a reference slot is
  freed (GitHub's 50 reusable-workflow limit); their workflow files are kept.

### Late 27 October — status update

Completed (see HISTORY): MoEdern 27M/115M public consumers (C/Python/CLI) pass and
both GGUFs are published (`@351adfad`, `@9c5bcf77`); MoEdern fast path merged
(default off); MoEdern 1.5M SciFact recorded (no gate); Unlimited-OCR-2 untiled
native IDs exact, tiled text exact with boxes within 3/999, fast decoder IDs
identical to the exact decoder. Remaining (see [handoff](docs/handoff-2026-10-28.md)):

- [ ] Unlimited-OCR-2: validate the merged reference-exact tiling preprocessing with
  the vision dump/diff kernel, then require exact tiled IDs.
- [ ] Unlimited-OCR-2: run the precision sweep (F32/F16/Q8_0/Q4_K/mixed vs frozen
  reference), choose the published precision, then HF publication.
- [ ] MoEdern quantized Q8_0/Q4_0 (in progress) under the unchanged gates.
- [ ] Rerun the MoEdern 115M author-runtime compare with fast arms.
- [ ] MoEdern 27M/115M: Rust/Dart/HTTP consumers and retrieval quality.
- [ ] Run Ovis final-integrated v2 and successor-assets v2 on hosted CI at the
  re-declared base (6a61759d) and review before any closure claim.

### Frozen full-SciFact hosted provider continuation

The standalone `moedern-scifact-gha-adapter-v1.yml` workflow accepts the exact
reviewed source and27M only. The current source is
`fc4d84293fb0ee4115636d4f241d02854b41405d`. It authenticates the full original v1
and older authored v3 before model access; every request/page binds its version.
Inference, numerical and retrieval gates remain the corrected3c worker's.
The standard runner must pass the unchanged14GiB available RAM and24GiB disk
preflight; otherwise retain an admission rejection and use another authorized
resource. Healthy1.5M Kaggle work is not repeated by this workflow.
Registration is metadata only and establishes no new runtime/model result.

### Frozen real-scan encoded-image audit

A model-free hosted audit at source813e1ec9b813ebf0dc72c8faf5841659a024fbce
authenticates all20 unchanged encoded inputs before comparing the exact production
STB prefix with the original Pillow RGB decode policy. It retains pixel hashes,
geometry, per-channel errors and codec/compiler identities. The standalone
`ocr-rgb-input-audit-v1.yml` registration is metadata only; no pixel difference
establishes model-error causality or quality qualification. No original model
is rerun. Numerical attribution follows the earliest measured input boundary.

### Current portfolio execution snapshot

The refreshed28 October handoff records exact source-qualified original evidence,
active corrected jobs, partial MiniCPM disk-failure scope, successful bounded
WASM operators and remaining whole-model/quantized/final-source gates. See
`docs/handoff-2026-10-28.md`. This main change contains documentation only;
candidate runtime remains on `work/portfolio-closure-q4k-20261028`.

### WASM natural-input workspace causal audit: source ready

Exact source9991426b9e3296def3457340a989e841a46a6383 freezes two natural
GELU inputs and their actual IQ4_XS/Q8_K workspaces from the retained trace.
All18 normal/direct byte anchors must pass across native1/native2/browser
before two consistent scale/integer hybrids execute.44 actualRef/independent
comparisons retain numeric rejection separately from structural failure.
No model/original forward, F32 production branch or new quality budget is used.
The standalone `wasm-q8k-cross-v1.yml` registration is source preparation only.
Protocol SHA649eaad8e6b3d0e20f03ad43a9a6c51051bb5728b37aae3d86367fdc7b4443d1;
15 small controls and34 protected/eight owned source checks pass independently.

### Frozen form input continuation — claimed

Owner `audit/ocr-form-retained-input-20261028`: preserve initial RGB source813
and its20 visible rows/19 admitted comparisons. The live form URL again serves
different bytes; the exact709,225-byte frozen form already survives in the
reviewed original form output. Prepare separately scoped form-only decode using
that retained input, exact-version service/source/member admission, and the
unchanged production/Pillow codec policies. No19-image or original-model repeat.
Add a reusable mounted-input admission helper for native form follow-ups: reject
wrong/ambiguous candidates before decoding/inference and preserve the default
URL path when no explicit retained input is supplied. Keep prior invalid rows,
source-qualified partial scopes and all manifest/gold/prompt/cap bytes unchanged.

### Retained MiniCPM analysis and form continuation registered

Mini source19c3f7bdf6d452614721946adefb95f6d3cf0180 passes11 independent
small controls and429 protected/11 owned source checks. First dispatch only
`minicpm-graph-retained-analysis-v1.yml` with that explicitsource: retained
vision/merger actualRef, zero model weights/forwards. Review completion before
activating the separately registered seven-arm workflow. No23-arm recapture,
original forward or numerical waiver. Canonical cap12GiB/reserve4GiB, atomic
receipts and whole-file reconstruction precede adapter cleanup.

Form source535c10fceac28ec04401b7f6a57a027e1be9bbc9 passes14 controls and
25 protected/nine owned source checks. `ocr-rgb-form-v1.yml` retrieves the exact
reviewed original form image by explicit-version service/source/member proofs,
then decodes only that form. Carry initial19 comparisons separately; no19-image
or original-model recapture. The native retained-input helper changes delivery
only, never manifest, pixels, prompt, caps, scorer or inference.

### Josev role-based development recipes and held-out owner

Source54c2b5767ae82c853d8649870abd234a2fbe7ee5 freezes A/B/C before any
NEW20 native outputs. Protocolf6f7b694f5080e1cf59fefc84400abe93b76b92664187f7b251ed0b354075e12
keeps all292 critical F32 tensors. A stores78 attention matrices F16 and72FFN
Q4_K; B additionally protects24down matrices F32; C protects first20 gate/up
matrices F16 and leaves onlylast4layers' eight gate/up Q4_K. Actual Cquantizer,
independent IEEE F16 verification and full shape/hash/type readback are required.
Run strict oldfour F32227-stage control before all three development candidates,
retaining every old budget/negative and selecting no publication automatically.

Owner `audit/josev-new20-native-owner-20261028` prepares a separately versioned
held-out runner. Preserve original110 files byte-for-byte. Authenticate all1480
original NPYs before NumPy/models; freeze exact runtime9c Gitobjects and a later
reviewed development-selected model/service/report. Keep candidate pins closed
until actual A/B/C review; do not run NEW20 for known-negative full150Q4 merely
to fill coverage. New owner corrects only nonexistent build-support path to
actual `cmake`, and predeclares14,400-second child/6hworkflow resource bounds.
The unchanged old worker remains historical. No new gold/cap/quality tolerance,
original-model forward, alias or publication follows from source preparation.

### GGUF standard file-type metadata — claimed correctness repair

Owner `audit/gguf-standard-file-type-20261028`. The current shared writer uses
GGML's internal file-type enum in `general.file_type`; the GGUF standard assigns
different values for Q4_K/Q5_K/Q6_K and IQ4_NL/XS. Tensor Q4_K storage code12
is correct and remains unchanged; standard Q4_K_S header code is14.
Implement explicit mappings for all exposed CLI methods, retain explicit legacy
header output only for reproduction, and admit canonical14 alongside legacy12
only within the existing strict gated Q4 profiles. No inference arithmetic,
weights, internal GGML enum, historical producer or default model changes.

Before publication, authenticate a selected legacy model and create a fresh
canonical copy with only the verified four-byte header change. Prove identical
size/offsets/prefix/suffix/tensor payloads and exact old/canonical decoded/stage
continuity on the explicitly reviewed metadata-aware runtime. Retain both whole
SHAs and source/report lineage; never relabel an existing public immutable file.
NEW20 activation keeps runtime/canonical model/repack proof pins closed until
this bridge and development selection are reviewed. Actual runtime release
versioning remains a final integration task; app0.17.12 stays pinned.

[Authoritative GGUF field definition](https://github.com/ggml-org/ggml/blob/6af560d55df03ad92116e3c0a697779584477e85/docs/gguf.md#general-metadata).

### Required Q8_0 and Q4_K coverage for new permissive models (29 October)

The owner explicitly requires both precisions, produced with the actual
`crispembed-quantize` and tested with `crispembed-diff` plus final decoded/task
outputs. A stored file is not qualification. Preserve frozen original references,
existing rejects, per-tensor shapes/types/payload hashes, critical F16/F32 bytes,
actual compiler/runtime provenance and full input/output gates.

| Model/front | Q8_0 work | Q4_K work |
| --- | --- | --- |
| Josev-2B |Correct standard7/mixed-v2 admission exists on678 runtime branch. Actual protected40838785499 tests Q8-all150 and Q8-with-F32-down126, both rejected; actual40853720400 tests only final pair Q8/rest F16. No Q8 qualification or main runtime merge yet. Same strict F32/old-four screen then independent NEW20 and public consumers. |Full150 and A/B/C remain rejected. New protected final-pair Q4 passes ordinal but misses two probability gates; actual40853720400 isolates gate-only/up-only Q4 with292 critical F32 and other projections F16. These are conservative mixed recipes, with sizes/types explicit, not broad Q4-sized models. |
| UltraBERT | Add explicit Q8 file-type admission with actual Q8 tensor inventory and protected vocabulary/norms; freeze/checkpointed original-bound sharding first. Do not mislabel Q8 as a Q4 header to pass old018 guards. | Reuse completed5483-row F32 control, durable per-row shards, all8 Ref shards and full300 rankings. First conservative existing last-four-FFN recipe/shard0 only; no unreviewed batch activation. |
| MoEdern1.5M/27M/115M | Reuse the older protected Q8 diagnostic architecture but produce current candidates through the C quantizer. Keep hash embedding/projection, router, dense head, norms, biases and depthwise conv F32. Resolve full-corpus F32 routing/numerical blockers first; old ten-case Q8 cosine alone is insufficient. | Genuine padded Q4_K20/20/52 matrices; preserve critical bytes/semantic widths. Diagnose1.5M three numerical rows and27M/115M exact-router-tie rows before full F32→Q4 retrieval/client qualification. |
| Unlimited-OCR-2 |Published Q8 exists with retained coordinate/ID differences. Revalidate protected higher-precision alternatives against the same four original pages, actual intermediate dumps and repaired output processing; retain earlier rejects. |Retained540 structural40840732942 reviewed positive, but no Q4 forward. Claimed first-candidate reuse caller must match same-host F16 before actual last1 numerical/decoded test; last4/all11 and original2/8/22 recipes stay intact and pending. |
| MiniCPM-OCR | Existing30-arm protocol includes Q8. Finish source-proven merger-span alignment, retained analysis and only seven missing arms; then scope higher-precision repairs from earliest vision/prefill differences. | Same source-bound30-arm protocol includes Q4_K; preserve23 decoded-exact arms/all numerical failures. Do not use tiny component cosines or observed timing as production quality/speed proof. |
| LFM2.5 Encoder / typed EmbeddingGemma | Audit already published precision evidence before scheduling any missing Q8 work; representative task/GPU gates remain separate. | Preserve existing calibrated/critical-tensor evidence and qualify missing precision/task surfaces without recapturing originals. |
| OvisOCR2 / OvisOCR3 | Keep Ovis2's measured Q8 rejection; scope targeted protected repairs, not a new default. Ovis3 needs its separate original-first task protocol before qualification. | Keep all Ovis2 plain/calibrated/mixed rejects and existing operator attribution. Ovis3 quantization remains contingent on its task/quality protocol. |

For each repair, align the earliest differing actual intermediate with its tensor
producers. Record minimum row/global cosine, maximum absolute/RMS errors and
original/native magnitudes before deciding which precise tensors stay F16/F32.
Embeddings, normalization, routing, output heads and vision/projector boundaries
are initial protection candidates, not an excuse to skip measurement. Run the
same decoded outputs/retrieval/scoring and failure/recovery probes after every
selected precision recipe. Calibration uses development cases only; never tune
on NEW20 or silently relax frozen gates. Architecture-dependent CNN/restoration
quantization must have genuine supported stored tensor types; retaining every
weight floating point cannot be advertised as a Q8/Q4 model.

Q8 admission source is being prepared on `feat/portfolio-q8-profiles-20261029`:
Josev gets an explicitly named `crispembed-josev-packed-mixed-v2` profile;
header7 requires actual Q8_0 and no Q4_K, while legacy Q4 requires real Q4_K.
UltraBERT uses the same storage/header consistency. The existing quantization
gates remain mandatory. A bounded `--metadata-string KEY=VALUE` C-quantizer
option replaces existing STRING keys before header layout, enabling genuine
profile stamps without ad hoc byte-length patches. Numeric or unknown metadata
keys reject before output creation. Actual compiler/metadata fixtures and the
original-bound Q8 tensor/intermediate/decoded panels still gate qualification.
### Protected tensor experiment checkpoint (29 October)

Actual A/B/C40787077302 is reviewed and all candidates remain rejected;
completed evidence is in HISTORY/PERFORMANCE. Actual Q8 profile fixture
40829558531 passes; its13 quantizer/header pairs are independently parsed.
The [new protected recipe protocol](docs/josev2b-protected-recipes-v1.md) at
`feat/josev-protected-precision-20261029@acb248451` predeclares two Q8 arms and
two last-layer-only Q4 arms from unchanged four-case development references.
Source peer review and hosted execution remain pending; no candidate selection
or NEW20 native access is authorized by the model-free controls.
The executable feature source owns its contract and callers; the workflow
registration here does not merge that runtime or qualify a model.

### Reviewed hosted continuations (29 October)

- Josev protected precision source`d339be99c` fixes the independently spotted
  standalone numeric-report flag to require actual Ref success; the unchanged
  combined F32 barrier was already strict. Six corruption controls/source closure
  and independent source peer review pass. Run only the fixed four development
  cases before any selection/NEW20 native access.
- Unlimited read-only exact-v3 delivery source`fbd8f68f6` uses existing hosted
  service configuration, blocks all array/framework imports before full540-file
  authentication and resolves the SDK package shadow explicitly. Seven mocked
  delivery/import/pagination controls pass; hosted SDK access and structure are
  pending. No model/quantizer/original rerun.
- Ultra freeze correction`08b0a5727` preserves initial40831085595: the old
  global symbol grep rejected27 pre-existing unrelated comparator symbols after
  compilation, before any payload. Audit the actual compiled Ultra object plus
  whole-library inventory, retain exact unrelated names, then retry freeze only.
- Norm-risk format successor`e7f3ab64d` preserves frozenf781 token identity,
 30 old byte anchors and12 native norm recoveries before56 candidate comparisons.
  This is a test-only replay, with no production/model forward/default change.
- MoEdern compare-only`92f75550a` retains failed40825166813 and uses bijective
  safe adapter names to finish18 saved comparisons. A structurally completed run
  must still exit nonzero when numerical failures remain; fresh Python drift
  cannot replace frozen gold or prove native causality.

### Actual hosted submissions,29 October07:20 UTC

All five are newly declared bounded continuations; check actual service state
before acting and never duplicate a running job. At submission they are queued:

| Job | Exact source | Gate |
|---|---|---|
| [Josev40834737477](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834737477) |`d339be99cfce99c1801fa3d1268deec122ae63e0` |Strict original F32, then four protected Q8/Q4 development recipes only. |
| [Unlimited40834740153](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834740153) |`fbd8f68f6753a64751a02f42771fc475670f64e4` |Exact-v3 GET/540-file authentication, then read-only structure; zero forwards. |
| [Ultra freeze40834743025](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834743025) |`08b0a5727aa25412a1181db4695efb01be578b24` |Actual Ultra-object symbol proof, retained F32 adapters/eight Ref/full rankings; no new F32/Q4 forward. |
| [Norm risk40834745535](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834745535) |`e7f3ab64dda2fdb42c8305637a258ce4b10bdef4` |30 byte anchors,12 native norm recoveries before56 actual Ref comparisons. |
| [MoEdern40834748271](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834748271) |`92f75550a0ce0f1486f6d7603164a6628aeb24a0` |18 retained three-document comparisons; numerical negatives must remain nonzero. |

Mini-v2 [40831080193](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40831080193)
atcec1faeb completes structurally:138 comparisons,36 carried+102 newly measured,
all23 input pixels exact and all historical numerical negatives preserved. It
remains unqualified. Independent small-proof review and the separately claimed
seven-only continuation follow; no23-arm/original rerun.

### Integration source qualification boundary

The default-off MoEdern Q8 runtime and source-bound component caller are now
integrated in the portfolio feature branch. All eight MoEdern runtime/profile
stdlib controls pass there. The caller's separate frozen whole-SRC assertion
correctly rejects this combined tree: its immutable1f source differs only in
UltraBERT's later UINT32 file-type admission check. The reviewed actual fixture
runs on exactdd592, not the integrated tree; no assertion or source gate was
weakened to claim combined-source qualification. Final source declarations and
actual composite/runtime/platform gates remain required.

### Protected MoEdern Q8 component execution

Source`dd592b15f8191be0b441eda3932c98364324e776` is independently reviewed for
hosted model-free execution. It preserves all1f runtime/CMake/fixture/converter
objects, uses two build workers and distinguishes20/20/52 typed profile
declarations from the tiny C fixture's two actual Q8 tensors/seven F32 roles.
Four new/eight existing stdlib controls, actionlint and both source guards pass;
actual compiler/graph/C-quantizer execution remains pending. The standalone
workflow registration does not qualify model weights or remove full-F32 corpus
blockers.

### Mini seven-only source review

Source`db4c1bb25a342c3ee70a1610e8c3476600d0bccc` is reviewed for the next
seven-arm continuation. It authenticates actual completedcec all134 small
members before optional packages/models, preserves every frozen19c script and
guard, and requires the same source in workflow input/service checkout. Only
seven missing ordinary native arms and two existing tiny projection probes run.
Postanalysis carries138 records unchanged and adds42 six-stage comparisons
from the new saved traces, for180 vision records; no second forward. All strict
failures and original budgets remain; coverage never sets qualification true.
13 new/11 inherited stdlib controls and actionlint pass.

### Ultra freeze completed; first Q4 conversion source reviewed

Actual40834743025 independently passes its retained freeze without original or
F32 forwards: all5483 vectors/eight16-stage Ref/full300 rankings remain positive.
Worst raw error0.00109804/cosine0.999999996825; reported ranking metrics exact.
Source`c3d2fc051f6632e06d5e7ba69ac700a03a703e22` is reviewed for model conversion
only: reproduce exact609437696-byte F32 hash, create existing last-four-FFN
12 Q4_K/206 F32 model with the unchanged018 quantizer/body and record its SHA.
The model descriptor remains null and shard0 activation false until actual
conversion/artifact/report review. No forward, Q8 route, automatic shard batch
or publication claim is part of this first phase.

Norm candidate source checkpoint29 October: `fix/wasm-native-order-norm-candidate-20261029@64eef68493deb240e81287e249f55def08d5753c`
retains the live-input default-off norm math from812c4e4d and fixes the byte
audit to preserve actual returned WASM bytes before free, including signed zero.
Fourteen bounded controls pass. The registered `wasm-native-order-norm-candidate-v1.yml`
requires component recoveries, native anchors and old OFF byte controls before
ON ten-case/eight-stage actual Ref comparisons. Heavy execution remains hosted;
no default or whole-model parity claim follows from source review.

MoEdern Q8[component40837105606](docs/moedern-q8-component-controls-result.md)
is independently reviewed positive for authored graphs/storage only. Next real
Q8/Q4 model qualification still requires resolving frozen F32 corpus/router
failures; no model/corpus forward was performed by this fixture job.
Norm source64eef684 full A/B actual40852835039 is dispatched and pending review.

Josev final-projection source3cc96dbab independently reviewed GO; actual
[40853720400](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40853720400)
is queued for all three old-four recipes. No numerical outcome yet.

Mini checkpoint repair source`2fde0c64b` on`fix/minicpm-checkpoint-blob-fetch-20261029`
passes16 stdlib controls including actual pinned public2213-byte Git blob and
corruption/unsupported-identity rejection. Historical db4 failure is preserved
in HISTORY. New `minicpm-graph-seven-arm-v3.yml` registered; peer review and exact
source-bound hosted execution remain pending. No model/quality result yet.

Unlimited[retained540 structural result](docs/unlimited-retained-v3-structural-result.md)
is reviewed. Next claim a new versioned candidate-reuse inference caller: authenticate
exact-v3 producer and first6,481,651,840-byte model, perform same-host F16 control
against carried traces, then first Q4 stage/decoded panel. Preserve last1/last4/all11
roles and all previous negatives; no original or first-quantizer repeat. Full Q8,
remaining Q4 recipes and publication/consumer qualification remain separate.

Mini v3source2fde0c64b independently reviewed GO; exact-source
[40855153707](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40855153707)
is queued for unchanged seven-only continuation. Kaggle first-row bootstrap
is prepared privately; fresh whole-account status visibility is rate-limited
(HTTP429), so no new submission was made without capacity admission.

Ultra conversion40843704011[small review](docs/ultrabert-first-conservative-q4-model-result.md)
is positive for storage/source only, with12 Q4_K/206 F32 and no model forward.
New closed shard0 source`a5feac83e177d7e08cfa23384089f7584f0736a9` on
`audit/ultrabert-reviewed-q4-shard0-20261029` passes nine controls,14 protected/7
owned plus11 runtime Git objects; parent source review GO. Workflow registered,
actual only686-row execution pending. Legacy header and unqualified status stay
explicit; all other shards/Q8 require later independent activation.

Actual queued jobs: Josev40853720400@3cc96dbab, norm40852835039@64eef684,
Mini40855153707@2fde0c64b, Ultraonlyshard040855927708@a5feac83e.
MoE one-document actual-stage job submitted on Kaggle CPUv1 at e7ff72f1 after
fresh whole-account CPU2/GPU0 admission; external exact-version proof pending.
Service acceptance does not establish completed numerical or final task quality.

MoE first-row Kagglev1 exact sourcee7ff/submitted8835-byte script was externally
authenticated with explicit versionLabel on Get/Status/all List pages. Terminal
ERROR occurred at bootstrap disk admission before any command/package/reference
transfer/model access, not a numerical failure. Source resource-profile repair
is separately claimed above; initial24GiB rejection remains preserved.

Unlimited candidate-reuse source`299eedb67ecdfda337d7b37e360c599dcd30ead7` on
`audit/unlimited-reused-q4-first-panel-20261029` passes17 stdlib controls,353
protected/8 owned closure and parent review. Workflow registered. All544 inputs
(540 panel+4 exact frozen PNGs) authenticate before imports;16 F16 controls
compare47 retained float stages plus one separately labeled assembled-vision
stage per case under unchanged0.002/0.99999,11 exact integer stages and current
repeat-byte neutrality before12 Q4 calls/16 local probes. Cross-host float SHA
is diagnostic, not an invented stricter gate. Initial unexecuted d716 checkpoint
is retained. Actual numerical/decoded execution pending.

MoE resource-v2 source`024f3e7a58cc4ed8e219689d1bea387342dd9556` is independently
reviewed with23 controls. An explicit10GiB disk/14GiB RAM/twoCPU profile with2GiB
output cap/live reserve replaces only the overbroad one-case disk requirement;
all numerical code, maps, references and forward limits remain unchanged. Kaggle
CPUv2 submission accepted after fresh whole-account CPU2/GPU0 admission; external
exact-version proof and actual results remain pending. Initial v1 stays negative.

### One-sentence connected norm attribution — source frozen 29 October

Register `wasm-live-norm-operands-v1.yml` and dispatch exact
`fix/wasm-live-norm-result-scope-20261029` at
`ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265`. The sixteen bounded source tests
pass. This diagnostic captures thirteen actual live norm input/output pairs and
layer-three boundaries for one unchanged twelve-token sentence. Preserve all old
native and WASM stage hashes before replay. Report 208 same-input and 196
natural comparisons separately; natural numerical failures make the completed
report negative and exit nonzero only after preserving every outcome. No
production arithmetic, precision, default, original recapture or full ten-case
rerun is authorized by this source registration. Review earliest differing
operator, absolute/RMS errors and magnitudes before proposing a correction.

### Actual seven-arm completion and two pre-precision harness repairs

MiniCPM40855153707 completes seven missing forwards: all30 decoded cases exact,
130/180 vision stages pass,50 fail;96 later merger crop parts remain unmapped.
Use [the result and bounded attribution scope](docs/minicpm-seven-arm-v3-result.md).

`fix/unlimited-ref-control-import-20261029` owns only a test-wrapper repair after
40859258832@299eedb fails isolated Python import `adapter` during Ref controls,
before F16/model access/any forward. Preserve frozen controls/runtime/candidate;
new wrapper must bootstrap exact frozen helper imports under `-I`, test actual
subprocess imports, preserve16 F16/12 Q4/16 projection counts and initial receipt.
Declare new source closure and review before hosted retry; never rerun quantizer.

`fix/moedern-actual-softmax-dispatch-20261029` owns a dumper-only repair after
exact Kagglev3@46bac passes both ordinary barriers (six native full computations,
byte-exact93 anchors; one original strict Ref control), then observed original
forward remains feature-byte-neutral but only six of eight required operations
are recorded: two SDPA, two topk and two div, no softmax. Preserve
[initial capture failure](tests/results/moedern-colbert/first-row-softmax-initial-review.json).
Capture real `aten.softmax.int` or `aten._softmax.default` inputs/outputs using
their actual names/schemas, with bounded FFN dispatch metadata to identify actual
behavior rather than reconstructing probabilities. Keep exact eight operations,
all44 floating/two integer stages and all ordinary/observer/repeat checks.
Previous targeted original calls are now3 acrossv2/v3; a separately reviewed
retry admits at most5, aggregate8; native previous6, next maximum14. No full
original recapture, new gold, production math change or threshold waiver.

### Same-candidate Ultra completion — source preparation claimed

`audit/ultrabert-q4-remaining-shards-20261029` owns a new standalone six-hour
hosted controller: build exact018/a06 once, authenticate immutable freeze/model
and actual40855927708 shard0 service/whole ZIP/report/controller/journals, then
run shards1–7 sequentially with unchanged forward and actual Ref bodies and
durable row commits. New9594 public raw/normalized calls plus imported1372
means10966 total over5483 rows. Aggregate all eight shards/300 queries through
frozen ranking code and unchanged0.001 degradation/0.99 top-one gates. Preserve
strict vector negatives separately, keep publication false. No conversion,
original/F32 replay, new recipe, silent historical source retarget or incomplete
shard substitution. Source/protocol/closure review before any hosted dispatch.

### Josev final-pair activation calibration — source preparation claimed

`audit/josev-final-pair-imatrix-20261029` owns a development-only calibrated
Q4 experiment after actual single-projection ablations. First trace the real
CPU route: `josev2b_backbone::run_impl` calls `decoder_hidden_cpu`, whose
`qwen35_decoder.cpp` post-attention RMS output feeds both final-layer gate/up
projections. The generic scheduler imatrix collector must not be assumed
connected. Existing251 original arrays contain layer23 output, no pre-FFN input.
A test-only disposable overlay may capture actual native F32 normalized input
for the four old requests, after unchanged strict227-stage/four-typed F32 and
ordinary/observer/repeat controls. Label native activation calibration honestly;
no original tensor invention or production callback. Predeclare uncalibrated
versus calibrated final-pair Q4:292 critical F32,148 F16,2 actual Q4_K. Write
exact per-column sum-of-squares/count for both2048-wide names, validate finite
nonzero coverage and actual C quantizer use of both vectors; the loader's silent
fallback must reject at this caller. Keep original budgets/all outcomes and
all earlier recipes. No Q8 imatrix, NEW20/holdout access, recipe selection or
publication. Explicit source/forward/quantizer counts and review precede runs.

### Typed Gemma first Q4 source reviewed and registered

`audit/gemma-decision-first-q4k-20261029` at
`6e327e902ebdd0feb531c3beae4b52902b7e781b` adds eleven test/driver/protocol files
only. Parent source review and11 stdlib controls pass; two NumPy controls stay
explicitly hosted. Minimal engine/core linking uses unchanged source and root
ISA helper/a06/LLAMAFILE, actual C quantizer and original Ref. Registered
`gemma-decision-first-q4-v1.yml` must execute this exact branch, with strict
F32145 stages/eight typed/30 bank gates before either predeclared recipe.
Maximum264 encoder forwards/zero original. No public CAPI or publication claim.

### Critical-tensor precision decisions and MiniCPM Q8 follow-up

Use [the measured per-model precision audit](docs/critical-tensor-precision-20261029.md).
MiniCPM's positive vision similarity for Q4 comes from retaining vision source
precision:186 decoder matrices selected,593 excluded, while Q8 quantizes vision
linears. The next separately closed Q8 recipe should use the SAME186 decoder
names as the frozen Q4 selection and actual Q8 storage, preserving all593
source-F16-model payloads/types/shapes (including F32 norms) byte-exact. No new
generic quantizer default. Verify full779-role census/header and actual C
quantizer use before all five decoded/intermediate cases with a same-source
F32/F16 control. Reuse immutable original/frozen inputs; no23-case original
recapture. Report unchanged F32 layer26 failures and96 unmapped crop parts
separately; this ablation cannot waive either. Source/model/forward counts,
reference receipts and review precede hosted execution/publication.

### MoEdern actual router-input replay — parent source preparation claimed

`audit/moedern-router-real-input-replay-20261029` owns a component-only
continuation of [completed actual CPUv4/f336](docs/moedern-first-row-actual-v4-result.md).
Freeze the exact-v4 mounted kernel outputs (explicit version4, never latest),
producer script/source, original/native normalized router inputs and captured
logits/probabilities/IDs/weights; full307-row shapes for both layers. Authenticate
selected payloads/actual published F32 weight bytes before framework imports.
Replay both original/native inputs through actual native linear/softmax and
original Torch linear/softmax/top-k, with own-byte anchors and interleaved
repeat/thread controls. Record all near-boundary expert identities; selected
weights alone hide this case. First strict failure is layer1 row58 [2,4] versus
[2,6], unequal probability margins1.3411e-7 versus1.0431e-7. Source/numeric
protocol, exact component call counts and peer review precede hosted execution.
No original/full-model forward, new gold, production arithmetic/tie rule,
quantizer run or budget waiver. CPU-only Kaggle may use exact-version attached
real capture inputs while GHA has no running jobs; fresh whole-account capacity
is required. Ordinary whole-model8 aggregate original calls stay frozen.

### Ultra seven-shard source reviewed; honest Gemma CPU fallback claimed

Ultra source`ef7153156aba352bffdef60bd44d4155a51f2ddc` on
`audit/ultrabert-q4-remaining-shards-20261029` passes parent source review,
12 new plus nine frozen shard controls and24-file/11-Git-object source closure.
Register `ultrabert-remaining-q4-v1.yml`, execute exact ref/source_ref together.
Import old shard0 only; strict negatives retained separately from full task
gates. No original/F32/quantizer repeats or publication/header qualification.

`audit/gemma-q4-kaggle-cpu-20261029` is claimed as an honest provider successor
while GHA has queued jobs and no in-progress workers. Preserve6e327e902's
original driver/recipe/protocol/build/probe bytes. A new closed caller may
reuse unchanged low-level numerical/typed/quantization functions in a separately
named implementation, with AST/source equivalence for the numerical body and
genuine Kaggle/source identity, never fake GHA variables or historical-guard
mutation. Retain8GiB RAM/16GiB disk/two CPUs and original264 maximum native
encoder calls/zero original; no new numeric/typed budget. Parent must verify
exact source/script plus Get/Status/every List page, and fresh whole-account
capacity before submission. Avoid duplicate model runs: if moving to Kaggle,
cancel only our still-queued/never-started Gemma40868968991 after checking
service/job steps and preserve its cancellation/source receipt. Do not cancel
other work. New protocol/source review is required before hosted execution.
