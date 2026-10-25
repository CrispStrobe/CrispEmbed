# CrispEmbed — Active roadmap

Updated 25 October 2026. Completed work, measurements and dated handoffs belong in
[HISTORY.md](HISTORY.md). Detailed current model contracts remain in their model
documents; this file contains pending work, acceptance conditions and ownership.
Read [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md), [contributing](docs/contributing.md)
and [the consumer handoff](docs/current-state-and-next-steps.md) before claiming work.
Start with the [executable next-agent handoff](docs/next-agent-handoff.md), which
records exact sources, remaining work, acceptance checks and artifact expiry.

## Execution order

### Active approved Ovis follow-up loop

The current owner is finishing the four approved items before closing this loop:

1. Profiling and issue71 feasibility are complete; exact evidence is in HISTORY.
2. The full attention-F16/FFN-Q4 attempt is complete and rejected; move its
   measured outcome to HISTORY/PERFORMANCE. The immutable four-attempt selection
   keeps published F16. Future precision work needs a separately declared
   development/evaluation protocol; do not tune this held-out panel.
3. Complete [larger-input40177774574](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40177774574):
   old frozen control plus newly authored1024/1200/2048-patch originals, all14
   F32/F16 arms, dynamic per-score raw manifests, owning OCR IDs/text/EOS, timing
   and RSS. New2048 bound needs four value-parsed gates; default1024 remains.
   Actual compiled bound/recovery tests include2176 rejection and legacy controls.
   After final core edits, rerun old11 F16 quality against frozen originals.
4. Complete OvisOCR3 original-first qualification:
   [reference40177416485](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40177416485)
   completes the three anchors and943-stage original capture. Native14
   [40178327173](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40178327173),
   original11 [40178328864](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40178328864)
   and current old Ovis2 F16 quality
   [40178330437](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40178330437)
   are dispatched; independently review terminal results before later gates.
   The original reference replays three reviewed processor anchors before loading weights; capture every
   layer/cache/logit/generated score and actual greedy output. Independently
   review the small report and bind immutable service/artifact/source hashes
   before original11, native14, F16 quality11 and public engine22 audits.
   Separate plain OCR scoring preserves raw CER/WER/EOS/caps, without the Ovis2
   HTML presentation scorer. Source-only format3 remains rejected by inference.
   The controlled original T4 run is reviewed in HISTORY and its GPU receipt;
   preserve its initial hardware preflight failure. Its measured installed
   vLLM dual-EOS policy remains distinct from explicit HF runtime qualification.

No queued workflow or prepared harness counts as qualification. Large archives
remain hosted; independent review downloads small reports only. Preserve old
Ovis2 first-F16 receipts, calibrated vision budgets and failed quant recipes.
No app/runtime release/default change follows from these experiments.


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
gold/scorer adjustment. Finish native14 retry40179894600 and the paired native
F16 diagnostic after its proof. Model qualification requires all old screens.
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
| 2 | [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | Pinned checkpoint/base Apache-2.0 audited | Q4/calibrated Q4 1.119GB rejected; Q8 1.251GB rejected; selected F16 1.717GB | [Metadata/schema/processor qualification](tests/results/qwen35-ocr/ovisocr2-metadata.json) passes; experimental proof-bound F32/F16 export and all473-tensor roundtrip pass [39992662147](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39992662147); PR77 merged2f8b5354. Frontend PR78 and controlled original T4/TRITON_ATTN GPU execution pass. PR79 initial943-field reference/native12/14 and vision failures are retained. Reviewed preset=both passes the full fourteen-arm frozen replay40049244654; six public interfaces40053283383 and original HF11 reference40054043629 pass. First F16 quality40056641800 matches all eleven Python ID/text outputs; protected Q8 40061433357 is rejected for numerical and two-column regression; all three quant recipes rejected; F16 published/alias consumer40080458492 passes; PR79 integrated; next latency/RSS lane below |
| 2b | [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd) | Pinned LICENSE/checkpoint/base Apache-2.0 audited | Similar small-family estimate; unmeasured | Source/header audit and actual hosted processor metadata pass; original per-module reference40177416485 dispatched; plain OCR11 frozen before outputs; native14/F16 quality/public22 remain unqualified |
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

- [ ] **Issue71 inpainting feasibility:** audit exact code/checkpoint licenses for
  MI-GAN/LaMa; trace image/mask contracts and existing primitive coverage. Keep
  still-image parity/public interfaces separate from batching/temporal propagation.
  Do not treat issue descriptions as verified license or quality evidence.

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
| 2026-10-25 | `feat/ovis-followups-20261025` | Ordered follow-up loop: Ovis latency/RSS; issue71 license/blueprint feasibility; attention-F16/FFN-Q4; bounded larger-input and next permissive checkpoint qualification | Profile40171223309 complete and reviewed; issue71 feasibility/source terms documented. Attention-F16/FFN-Q4 prepared for actual eleven-case/diff evaluation; OvisOCR3 original metadata40172054543 queued. Preserve original/first-F16/publication receipts; no local builds/inference. Each later runtime change requires original-module/crispembed-diff and decoded-quality evidence before promotion. |
| 2026-10-06 | `main` (Crisp3DS SAM work) | SAM 2.1 Hiera-tiny image segmentation (`docs/sam2.md`): engine, converter, reference dump, `test-sam2-diff`, C API, CLI `--sam2`, server `POST /sam2/segment`, Python/Rust/Dart `CrispSam2`, registry `sam2.1-hiera-tiny` (F16, default) and `sam2.1-hiera-tiny-f32`, both pinned by hand | Parity with PyTorch CPU: F32 and F16 cos >= 0.999999 at every stage, masks identical; Q8_0 below the 0.999 gate (mask IoU 0.9895 worst, not published), Q4_K not usable. Registry fetch of the F16 verified. **Note for the owner:** `tools/fetch_model_hashes.py --check` reports `model_hashes.h` stale for reasons unrelated to SAM 2 — no pin for `nomic-embed-vision-v1.5-f16` and for `jina-ocr-v1-q4_k` / `-q8_0`, and three pinned files whose upstream bytes changed. Deliberately not re-pinned here. Not yet: 73-photo end-to-end check, Metal timing |

The registry warning in the SAM ownership row describes the pre-repair state;
[PR61](https://github.com/CrispStrobe/CrispEmbed/pull/61) resolves those six artifacts
with preserved digests and complete freshness evidence in HISTORY. SAM quality
and performance follow-ups remain with their owner.

## Active Ovis completion sequence

**Completed:** PR78 frontend, pinned original GPU, native14/14 against943 frozen
fields,27 actual public-interface arms, all11 first-F16 Python ID/text matches,
three rejected quantization attempts, qualified immutable F16 publication and
actual published CLI/HTTP/offline consumer. [PR79](https://github.com/CrispStrobe/CrispEmbed/pull/79)
is merged after final-source platform/Flutter/WASM/style/security checks;
merged-source checks also pass. Exact commits/runs, all initial failures and
measured limits are in [HISTORY](HISTORY.md) and [PERFORMANCE](PERFORMANCE.md).
[Consumer proof](tests/results/qwen35-ocr/ovisocr2-published-consumer.json) and
[publication proof](tests/results/qwen35-ocr/ovisocr2-publication.json) are frozen.
Do not repeat completed captures/panels. No runtime tag/default/app pin change;
development0.17.13, app immutable0.17.12. The optional
[M1 audit](docs/handover-m1-ovisocr2.md) remains complementary Apple coverage.

Continue with one bounded lane below, starting with latency/RSS attribution.
Selectively quantized attention-F16/FFN-Q4 is separate subsequent work; larger
inputs follow resource attribution. Consumer catalogs/providers require qualified
immutable successor binaries. Preserve all initial failures, old recipes and
raw strict flags. The explicit three entry gates and reviewed `both` math preset
remain required; neither delivery nor speed permits a default flip.

- [ ] **Remote artifact preservation:** before21 January2027 preserve the
  original871-field archive and before the respective22 January expiries preserve
  full943 fields/native vectors/quality and quantization artifacts. Inventory
  run/artifact IDs and ZIP/member hashes in the existing receipts and HISTORY;
  copy remotely without recapturing/reconverting. Verify source identity and
  every copied member, retain original producer attribution and expiry. Download
  only the small review; never bring multi-GB archives to the shared host.

## Ovis latency/RSS follow-up — after the current qualification/publication loop

The first qualified [F16 panel](tests/results/qwen35-ocr/ovisocr2-f16-quality.json)
has exact Python IDs/text on all eleven cases but native recognition walls of
51.14 seconds for Hello OCR,1375.88 seconds for the receipt and1110.88 seconds
for the two-column page. Native/HF CPU models and timing boundaries differ;
recognition parity does not settle latency. This is a later bounded lane, not
permission to recapture or weaken the original/F16 quality anchors.

1. On hosted CPU, adapt `tests/test_qwen35_profile.cpp` to the explicit Ovis
   format-2 context and reviewed math/gates. Use `src/qwen35_profile.{h,cpp}`
   counters to separate setup/materialization/read/conversion/validation/matmul,
   vision/prefill/decode and initialization. Start with the frozen owned image;
   a capped document prefix is a profile only, not EOS/decoded-quality proof.
   Check profile on/off neutrality against the frozen first outputs and record
   actual source, CPU/compiler/library/thread, RSS and timing boundaries.
2. Only after measured attribution, evaluate one gated optimization at a time
   in `src/qwen35_linear.cpp::project_cpu`/`tensor_floats`, such as per-context
   converted-weight reuse or backend/context reuse. Keep the existing path and
   bound any extra RAM. Cover context separation, reset, failed load, free and
   survivor behavior; do not introduce a shared stale tensor-pointer cache.
3. Require the full frozen original-module/crispembed-diff stage/cache/logit
   gates and all eleven actual first-F16 IDs/text/EOS/caps/quality/confidence
   comparisons for a changed runtime. Measure paired same-host baseline/new
   latency and RSS; report any memory tradeoff. Preserve every failure and raw
   strict flag. No default or app-runtime flip without separate qualification.

## Ovis selective precision follow-up — separate from the current three attempts

The [Q8 raw replay](tests/results/qwen35-ocr/ovisocr2-q8-raw-diff.json) locates
source-fed divergence at `l.layers.0.linear_attn.in_proj_qkv.weight`: input max
absolute0, QKV max0.0564889908 despite cosine0.999990225. The current two Q4
recipes retain that projection in Q8; calibrating only FFN inputs cannot change
its quantized bytes. The [cross-arm record review](tests/results/qwen35-ocr/ovisocr2-quant-cross-arm.json)
verifies all72 calibrated FFN payloads change while all78 attention/323 protected
records stay identical, including the exact first-QKV SHA across all three arms.
Do not infer that this alone explains all decoded errors,
or that restoring it will qualify a smaller model. The completed three-attempt/
publication/consumer loop is archived in HISTORY; this is a new bounded lane.

Use `tools/ci-heavy/ovisocr2_quality_contract.py::RECIPES`, the existing C-side
quantizer overrides and all473 before/after tensor records. First compare a
separate attention-F16/FFN-Q4 recipe with the existing attention-Q8 recipe;
protect vision/embedding/norms/recurrent A/B and retain both old alternatives.
Freeze the recipe on development evidence before heldout/document output.
Run the same-source F16 local control, source-fed first-linear projections and
full original-module/crispembed-diff caches/logits/decoded comparison, retaining
actual raw vectors and every unchanged numerical verdict. Isolate additional
sensitive matrices only from the earliest differing operation. Reuse the frozen
first F16 and original11-case anchors; all eleven candidate IDs/text/EOS/caps,
confidence, raw/parsed CER/WER and15-cell evidence remain required. Record actual
file bytes/tensor types, paired host latency/RSS and calibration separation;
no size estimate or component cosine is a quality result. Use hosted CPU only.
Failed numerical or document screens still block the candidate; any proposal
for different quantization acceptance criteria is a separately documented
methodology task and must not rewrite historical float comparisons. Keep the
current F16 selection/publication receipts and model pin immutable. A future
qualified precision needs versioned decision/publication evidence and a reviewed
destination/allowlist: the present publisher intentionally permits one GGUF plus
four provenance files and refuses additional unreviewed model files.

## Ovis larger-input qualification — after latency/resource attribution

The qualified reader caps the original-resized image at1024 visual patches;
upstream's2880-squared-pixel maximum remains unqualified. Keep that bound in
`src/qwen35_vision.cpp::geometry` and the owning-context validation until this
separate lane passes. Source parity on eleven bounded pages is not full upstream
input-range or OCRBench/OmniDocBench qualification.

Freeze separately licensed additional pages, original resize/grid/masks/positions,
expected greedy IDs/EOS/text and quality policy before a larger native candidate.
Use the original-module dumper and crispembed-diff on a bounded incremental grid
panel, including non-square images and boundary rejection; preserve the current
943-field/eleven-case anchors. Record visual attention/prefill/cache/logit errors
and actual decoded output, wall/RSS and backend/precision. Identify the necessary
native allocation/attention/reduction changes from the real blueprint before
raising dimensions; the current scalar attention cost makes a cap-only change
insufficient. Run heavy captures/model batches only on hosted/authorized resources
with an explicit RAM/time budget. Keep larger inputs gated until the same
original numerical and decoded-quality gates pass, with safe failure/recovery.
Broad benchmark claims require their own fixed revision/split/input/metric and
license plan; do not infer them from this owned panel or published weight size.

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
configuration/selector. Use orchestrator stage21 for Ovis; the older
`vlm_engine_id` escalation API uses a different number space. Preserve existing
default engines and the three experimental gates. Acceptance requires actual
selected-library identity, empty-cache model hash resolution, output/confidence,
cap/custom-prompt rejection and fallback-stage behavior. Record the consumer
source/artifact proof separately; do not claim it from vendor component parity.
