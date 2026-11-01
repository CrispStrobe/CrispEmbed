# CrispEmbed active plan —29 October2026 wrap-up

## Issue 71 latent / MI-GAN512 qualification — 1 November 2026

Branch `feat/issue71-latent-qual` (five-engines c4bb1f93 + graphs delivery 4
06d75fb8). Not merged into feat/teleocr or main; nothing published or
registered. Status and numbers: [precision guide](docs/inpainting-precision.md),
HISTORY 1 November.

- MI-GAN512: F32 eager and graph pass the frozen 8-fixture panel; F16/Q8_0/Q4_K
  pass the owner cosine gate on all four execution arms. Live real-photo
  evaluation recorded. Open: representative subtitle/video quality, a reviewed
  author digest (only first-observed), publication decision.
- Moebius / PixelHacker: original CUDA captures exist (Kaggle T4; PixelHacker
  with the recorded torch 2.6.0 deviation). Eager F32 Moebius reproduces the
  original image (RGB max diff 1) but exceeds the strict 0.002 component budget
  on VAE moments and late UNet steps; keep that failure recorded and unchanged.
  Pending in hosted lanes: PixelHacker native, all graph arms after the pin
  fix, precision arms and live photos.
- Graph core (delivery 4): initializers and parameters are now pinned against
  gallocr reuse; graph controls pass on CPU (41302790327). CUDA/Metal graph
  qualification needs provisioned runners.
- Speed: the eager latent path takes 6–9 min per 512 image on 4 vCPU; graph is
  about 25% faster. Profile before any default change.
- HF private storage for the account is full; further private preservation
  needs the owner to free space or upgrade.

## Native Marker-style document pipeline — 31 October 2026

The requested implementation and bounded validation are complete on
`feat/marker-native`, preserving the completed TeleOCR
branch and merges `origin/main@b8cd249` before code changes. Marker source is
pinned to `71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4` (2.1.0, Apache-2.0).
The requested GGUF/GGML route uses the existing native TeleOCR backend and a
C++ document pipeline. Current Surya checkpoint inference/parity is outside
this alternative backend: its separately restricted models are not downloaded.

- Native C ABI/orchestration, optional PDFium, layout/crop OCR, document
  processors/renderers, CLI, Python/Rust/Dart and HTTP are implemented. See
  [usage](docs/marker.md), [coverage](docs/marker-feature-coverage.md),
  [licenses](docs/marker-license.md) and [validation](docs/marker-validation.md).
- Final document/backend/C API/save panels pass 86/39/38/38 checks; public
  integration 112, compiled Rust/Dart 13/14, final ASAN/UBSAN 86 and actual
  Python packaging 26. The independent native/reference panel passes 86
  checks across eight PDFs/ten pages. Initial failures and corrections remain.
- Real model runs use a declared, tiny owned panel. Final-source F16/Q8_0/Q4_K
  recover the three-line image exactly; the initial F16 full panel also covers
  neural PDF layout with text reuse and a 4×3 table. Keep raw label errors and
  table-header semantics limitations visible. The separately declared final
  force-OCR PDF also matches authored text with embedded extraction bypassed.
  Seven mode/build/precision trials reuse three distinct inputs; complete
  per-build receipts remain in [the aggregate](tests/results/marker/summary.json).
- Keep `CRISPEMBED_MARKER` and `CRISPEMBED_TELEOCR` default-off. No model
  publication, registry alias, runtime release or numerical qualification change.
- Remaining qualification is representative multilingual/layout/reading-order
  quality, more than two columns, complex tables, large-document resource
  boundaries and non-Linux backends/consumers. This implementation does not
  claim the original Surya model behavior, Office/EPUB providers or hosted LLM
  processors. Upstream models retain their separate restrictions.
- Resource check before bounded work: idle host, CPU-only inference sequentially,
  no simultaneous model forwards. Reuse the frozen TeleOCR numerical evidence;
  this pipeline does not waive its failed strict intermediate gates.

## TeleOCR GGUF/GGML integration — 31 October 2026

The requested native implementation and bounded CPU validation are complete
on `feat/teleocr-ggml`, based on `15adb672f71f8470a23db1284dafa102515adae9`;
ggml remains pinned to `a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. The complete
source delivery includes ggml, an applicable patch and retained evidence.
See [usage](docs/teleocr.md), [license audit](docs/teleocr-license.md),
[validation](docs/teleocr-validation.md) and the [matrix](tests/results/teleocr/summary.json).

- Sources: `XingChen-AGI/TeleOCR@b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d`
  and Apache-2.0 `rzafiamy/teleocr-rs@957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe`.
  The pinned HF model module has an explicit Apache source header. Official
  unlicensed document-client/postprocessor files were not directly translated.
- Converter checks 701 source tensors and writes 700; F32/F16 and protected
  Q8_0/Q4_K artifacts were actually converted and tested. C/Python/CLI/HTTP,
  stage 24, compiled Rust/Dart and the native-backed document parser are wired.
- All 24 CPU cases reach EOS with complete 75-stage coverage and valid
  confidences. F32/F16/Q8_0 each match six complete original sequences; Q4_K
  matches five and adds a trailing newline on code. Raw and normalized scores
  remain separate. Final loader guards pass 43 metadata/header cases and the
  rebuilt production CLI repeats the full 66-token text result exactly.
- **Strict numerical qualification remains false for every precision** under
  the unchanged 0.002 absolute / 0.99999 row-cosine gates. Preserve the initial
  failures, original references and current decoded results. The limited
  F32 ULP/propagation diagnosis does not waive the gates or explain all errors.
- Keep `CRISPEMBED_TELEOCR=1` default-off and CPU placement default. Next work is
  numerical qualification against the same frozen source, representative
  independent OCR/layout gold, then actual CUDA/Metal/Vulkan/mobile/WASM and
  resource-boundary qualification. Quantized raw differences must remain visible.
- Model publication, immutable registry aliases, release tags and downstream
  app changes remain separate. No runtime version or default is promoted here.

Unrelated portfolio work and downstream application pins remain with their
existing owners. Completed details are recorded in HISTORY and PERFORMANCE.

Start with [the executable handoff](docs/handoff-2026-10-29.md). It gives exact
sources, job states, inputs, commands, limits and acceptance gates. Completed
results and superseded task claims are in [HISTORY](HISTORY.md); measurements
are in [PERFORMANCE](PERFORMANCE.md). The full previous plan is preserved in
[the dated archive](HISTORY.md#archived-roadmap-snapshot-29-october2026).

This is an open roadmap. No new portfolio release, default flip or quantized
qualification is claimed. Development VERSION is0.17.14; the application stays
on immutable0.17.12. Do not bump versions for this documentation handoff.

## pub.dev publication handoff (31 October 2026)

The published Flutter package is 0.16.1; the development package is 0.17.14.
The existing native release workflow uses `v<version>` tags, while the current
pub.dev trusted publisher uses `crispembed-v{{version}}` (administrator
confirmed). A normal native release tag therefore does not run that publisher.
Do not change the pub.dev setting until a replacement is qualified and merged.

[Draft PR #91](https://github.com/CrispStrobe/CrispEmbed/pull/91) proposes a
release preflight, but leaves the new publisher disabled and retains the old
publisher. It is not a completed publishing solution: the preflight checks
asset identity, not source qualification or consumer functionality, and GitHub
release assets may become public before it runs. The package changelog also lacks
the current development version. The draft Android version change needs an
actual Gradle consumer test, and existing cached libraries are not verified
against the requested version. Preserve the current publisher until replacement.

The draft's model-free preflight has seven hosted contract tests passing. A
[hosted Apple baseline](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41213168996)
using the already-published 0.16.1 package built a clean Flutter app for macOS
and iOS; the macOS build warned about arm64 libraries in its x86_64 link, and
the app did not call native symbols. This does not qualify the development
package, Apple FFI runtime, Android, Linux, Windows, OIDC, or publication.
Before a release, verify source-matched assets and fresh consumers, coordinate
the trusted tag pattern, update the changelog, and confirm the actual pub.dev
publication by registry readback. No version bump or publication is made by
this handoff.

## Immediate executable order

### Model weight licenses (31 October 2026)

[The registry audit](tests/results/licenses/registry-qwen-audit.json) found
five families labelled apache-2.0 whose weights inherit stricter Qwen terms:
`qwen2vl-3b`, `nanonets-ocr-s`, `uni-mumer-qwen2.5-vl-3b` and `teleocr-*`
(Qwen2.5-VL-3B: Qwen RESEARCH LICENSE, non-commercial) and `got-ocr2`
(Qwen1.5-0.5B decoder: Tongyi Qianwen RESEARCH LICENSE, non-commercial and no
use to improve other LLMs). Registry licenses, download acceptance, engine gates
(`CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE`, `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE`)
and the published cstr/* cards are corrected. Open: CrispSorter/CrispCalc must
not offer these models without the acknowledgement; Apache-2.0 replacements for
GOT's tasks are MinerU2.5-Pro, FireRed-OCR, Uni-MuMER-Qwen3-VL-2B and PaddleOCR-VL.


### OXR-1.0 native integration (31 October 2026)

The isolated `feat/oxr-1.0` worktree retains the earlier MinerU integration and
merges upstream `b8cd2497765f2fd8db0b9c713d952898bdd9b996`. The OXR model pin is
`c708407f9d2b4df658f2543d9132b62bb7221df4`; the document helper pin is
`c1a1dcb312ee9f691404f127785510453c0d9350`. A dedicated native graph, strict
converter, document adapter, public bindings, and original-reference qualification
tools are implemented behind `CRISPEMBED_OXR=1`. This is a separate source lane
under OXR's additional license terms, not an addition to the permissive-model
publication queue. [The guide](docs/oxr.md) records exact scope and commands.

Completed component evidence is in
[the implementation receipt](tests/results/oxr/implementation-validation.json).
Full library linking, the linked GGML contracts, checkpoint conversion/inference,
all 66 original/native stages, representative document metrics, same-host F32
speed, reusable-graph A/B, GPU/backends and Rust/Dart runtime checks remain open.
The supplied [hosted workflow](.github/workflows/oxr.yml) has not been dispatched.
Do not publish aliases, promote defaults or infer vLLM throughput from component
checks. The original D-FINE layout service remains separate; supplied regions
allow the adapter to avoid that service. The native D-FINE layout detector now
lives on `feat/oxr-dfine` (see below).

### OXR-1.0 D-FINE layout detector (feat/oxr-dfine, 1 November 2026)

Native `dfine` architecture of `layout_detect` ([docs/oxr-dfine.md](docs/oxr-dfine.md)),
gated by `CRISPEMBED_OXR=1` + `CRISPEMBED_ACCEPT_OXR_LICENSE=1`. Hosted run
41250159815: F32 OXR-selected regions match the original service on all five
pages (boxes <= 4.4e-4 px, scores <= 1.8e-4), preprocessing bit-exact, heron
byte-identical; stage budget 5e-3 fails in decoder layers 2-3 on receipt/german
(native closer to float64 than PyTorch FP32 on 55-56/56 stages); F16 rejected.
Open, in order:

1. Speed: native 1.44 s vs PyTorch 0.52 s per page (run 41251744740, 4 threads;
   2.6-2.7x). Opt-in dw-direct is 18% faster, same regions, one extra stage-budget
   failure; conv-direct is slower. Next: per-op profile of the encoder graph
   (im2col 3x3 convs, concat chains) before any default change; keep paths gated.
2. End-to-end `CrispOXR(layout_model=)` document output vs the service-backed
   pipeline on the same pages (JSON/Markdown), once the document-assembly lane lands.
3. Optional: `class_confidences`, GPU backend qualification, Rust/Dart runtime tests.
No publication, alias, merge into feat/teleocr, tag or default flip follows.

The shared-operator audit additionally replaces MinerU-only QuickGELU's F16 LUT
with explicit F32 math. Its linked operator check and full-model replay remain
pending; the earlier MinerU component receipt retains its original source hashes.

### MinerU2.5-Pro-2604 integration (31 October 2026)

The isolated `feat/mineru25-pro-2604` source delivery adds the checkpoint
`opendatalab/MinerU2.5-Pro-2604-1.2B` at revision
`d3f5e08d073c21466bbabe21c71bb1e9c2e595da`. The implementation reuses the
Qwen2-VL GGML graph behind `CRISPEMBED_MINERU25_PRO=1`. Its converter,
preprocessing/token contracts, document adapter and qualification tools are
tracked in [the integration guide](docs/mineru25-pro.md). Model-backed
intermediate, decoded-quality and same-host speed gates must pass before this
path is qualified or receives a published model alias. No existing model default,
release version, application pin or remote model artifact is changed here.

Full native document route (`feat/mineru-native-doc`): C++ port of every
mineru-vl-utils branch behind `CRISPEMBED_MINERU25_PRO_FULL_DOC` (default off),
composed from the layout/router/Marker-document components with an exact
upstream output profile. Weight-free parity is identical (see the guide).
Model-backed parity on hosted run 41251475795: 8/8 protocol cases identical.
Next: review and decide default-on (keep basic/Python routes); add an
image-bearing-table fixture to exercise masking with the real model; token-glyph pixels in image-bearing tables and
cross-page table merge remain documented exceptions.


Choose one bounded lane. The source table and current remote states in the
handoff take precedence over historical commands.

1. Inspect the two already queued jobs: Unlimited40868634374 and
   Ultra40871150002. Verify exact source/attempt and collect small reports after
   completion; do not duplicate them. Keep models/raw arrays remote.
2. Repair the weight-free WASM component-profile guard after40867038915 rejects
   `Actual component rounding flags differ`. Save/review the actual compile
   inventory before asserting exact target counts/flags. Preserve ff3 and the
   full norm negative. No model run until component/browser controls pass.
3. Independently review MoE real-router source81fa4e26, then execute its
   component-only replay using exact-v4 attached captures. No whole-model or
   original forward. Require own-byte controls before cross-input conclusions.
4. Implement Gemma's frozen live-snapshot/768-wide-adapter design0ee3fb674.
   It is **not executable**. Review the implementation, then activate only one
   18-ID probe/three native calls. Review actual live/late/pooled evidence before
   separately declaring any full145-stage/typed/Q4 retry.
5. Independently review Josev imatrix source208fbefe, register its workflow, then
   execute its fixed40-prefill development panel on a host satisfying14GiB
   RAM/64GiB disk. Do not silently lower its resource/source guards for Kaggle.
6. Complete the remaining precision/task/client gates below, then integrate
   reviewed opt-ins and qualify final runtime/assets/consumers. Each new model
   or arithmetic source needs its own reference/quality/consumer evidence.

## Required Q8_0 and Q4_K coverage for new permissive models (29 October)

Use actual `crispembed-quantize`, original Python/ONNX dumps and actual
`crispembed-diff`. Keep critical tensor types/payload hashes, axes, compiler/
backend provenance, absolute/RMS errors, magnitudes, cosine and final task
outputs. Numerical and task verdicts are distinct; preserve negative flags.
[Measured critical-tensor evidence](docs/critical-tensor-precision-20261029.md)
explains why a generic exemption list is insufficient.

| Front | Current boundary | Remaining work and completion gate |
|---|---|---|
| Josev-2B |F32 composition227 stages/four typed requests is established; protected/single-projection quantization negatives are retained. Final-pair Q8 passes a limited decision screen but75 strict stages/full serialization remain negative. |Execute reviewed calibrated/uncalibrated final-pair Q4 experiment; keep292 critical F32/148 F16/2 Q4 and exact uncalibrated model continuity. Do not select/tune on NEW20. A separately declared independent task panel, actual six-client/public/error/recovery checks and immutable HF readback remain before publication. Q8 and Q4 qualification are both open. |
| MoEdern1.5M |Actual F32 capture localizes one near-boundary expert change; router weights already F32. Component-only replay source is ready, execution unreviewed. |Separate same-input operator rounding from upstream input effects; preserve exact-tie guards and all original matrices. Only then declare a gated correction, original-bound full5483-vector/300-query replay and Q8/Q4 retrieval/client panels. Keep explicit query/document roles and late-interaction semantics. |
| MoEdern27M/115M |Original token checks complete but exact-router-tie failures prevent full reference completion. Authored Q8/top-k tests are component evidence only. |Use source-backed routing semantics and bounded failed-row attribution before separately frozen original/full native controls. Preserve115M dilated-conv behavior. Validate actual20/20/52 padded Q4 and protected Q8 tensors, full retrieval and all public consumers; no component-to-model promotion. |
| UltraBERT |Frozen5483-row F32 control/300 rankings pass. First conservative Q4 shard686 rows completes with strict vector negatives. |Review queued seven-shard/full300-query aggregate on the SAME candidate; no original/F32/quantizer repeat. Keep .001 task-degradation/.99 top-one gates and strict vector flags separately. Q8 needs explicit metadata/type admission; standard-header normalization, public clients, independent activation and publication/readback remain open. Public pairs/F16/long-context/GPU/performance are distinct follow-ups. |
| Unlimited-OCR-2 |Exact retained540-file/model structure passes; first numerical attempt stopped at an isolated import before forwards. |Review queued repaired last1 reuse panel:16 F16 controls before12 Q4 calls/16 projections. Preserve decoded body/box gates and exact source. Last4/all11 and protected Q8 need separately declared panels. Quantized task/public clients/publication remain unqualified. |
| MiniCPM-OCR |All30 decoded cases exact; vision130/180 pass,50 fail;96 crop parts unmapped. Q4 protects vision while generic Q8 quantizes it. |Declare decoder-only Q8 on the SAME186 decoder matrices; keep593 source-model exclusions byte-exact. Independently attribute earliest F32 vision differences with same inputs/weights and document unmapped crop limits. Preserve original23 plus added7 outputs; qualify all five cases/intermediates/clients before publication. No original corpus recapture or generic quantizer default change. |
| LFM2.5 masked230M |Published Q8/Q4/calibrated/mixed profiles exist, but mixed330 still changes one masked prediction. |Reuse immutable original/same-weight references; isolate remaining prediction-sensitive groups. Preserve attention/ShortConv/FFN-down high precision supported by actual ablations. Freeze independent task inputs before selection; extend quantized/GPU/client evidence separately. Do not confuse this with the350M retrieval model. |
| Typed EmbeddingGemma |Q8 approximation published; first new Q4 worker stops at F32 saved-stage/adapter failures. All five pooled F32 embeddings pass. |Implement/review design-only repair, three-call first probe, then separately activate full145/eight-typed/30-bank F32 barrier before unchanged broad/protected-input Q4 recipes. Q8 current-source replay and Rust/Dart model-backed paths remain open; old wrapper helpers hardcode F32. Independent task/GPU/performance/publication gates remain separate. |
| OvisOCR2/OvisOCR3 |Ovis2 F16 is published/consumer-verified. All existing Q8/plain/calibrated/mixed Q4 candidates are rejected. Ovis3 faithfully matches an original capped/poor-quality page. |Keep Ovis2 F16/alias; new precision selection needs fresh development protocol after completed same-weight attribution. Ovis3 needs a separate original-first task/evaluation protocol before quantization, alias or publication. Never raise caps/change prompt/gold to pass old quality gates. |

For each candidate: authenticate all declared inputs before model use, preserve
ordinary/observer/repeat and rejection/recovery controls, compare earliest actual
intermediates, and evaluate complete decoded/typed outputs or retrieval rankings.
Only a separately qualified candidate proceeds to all public surfaces, immutable
HF upload/empty-cache readback, registry pins and consumers. Stored files alone
are not qualification. Calibration is development-only; never invent reference
activations or waive historical budgets.

## Active Ovis completion sequence

Draft [PR89](https://github.com/CrispStrobe/CrispEmbed/pull/89) holds integration
`8da6636b35d33623e143ae34120c09fb9878a460`. Its frozen test-source formatting
failure must be repaired **only in a new integration closure**, preserving every
producer's source/receipts. Merge carefully: old branches can delete unrelated
registered workflows/evidence; review all deletions against current main.

After final code decisions, declare a new complete source closure. Require
original-bound Ovis fourteen-arm plus first-F16 eleven-case replay on BOTH gates,
all frozen IDs/text/EOS/caps/confidences, appropriate public interfaces and the
full required CI. Then create fresh native/WASM assets and verify actual Linux,
Apple, Windows, Android/browser consumers, hashes, baseline CPU portability,
installed loads and aggregation. Only then use the version/release procedure,
immutable model aliases and a separate downstream app-pin upgrade.

The old frozen full-source guards intentionally reject a changed runtime. Do
not weaken them, relabel another source or infer asset qualification from CI.

## Ovis 2048-patch numerical attribution — before broader input claims

Default1024 stays. Frozen/1024/1200 pass14 arms;2048 has six composed final-norm
failures despite exact text/EOS. Reuse original embeddings and retained inputs;
isolate the first divergent position/interpolation/norm boundary with actual
source-fed operators before any broader-page claim. Preserve the four opt-in
large-image gates and existing strict/calibrated budgets.

## Other open numerical, performance and consumer lanes

Each row requires a new bounded protocol, frozen inputs, original/reference
comparison, final-output checks and actual hosted evidence before adoption.
Detailed prior acceptance clauses remain in the archived roadmap/model docs.

| Lane | First bounded action | Gate/dependency |
|---|---|---|
| WASM layer3 |Complete corrected one-sentence live operand capture; use actual13 norms/23 layer3 boundaries and same-input replays. |Preserve full norm/stored-F32/rounding negatives and default-off gates.208 local/196 natural comparisons and native/old-WASM byte anchors stay distinct. No full ten-case retry or arithmetic change before attribution. |
| LightOnOCR-3-0.8B |Use retained book isolation to investigate upstream vision, with one fixed input/source-fed replay. |Exact independent-document text retains numerical debt; all quantized receipt/book recipes remain rejected. No publication/default from literal EOS or table-cell equality. |
| OCR real pages/benchmarks |Freeze licensed annotations/reading order and consistent timing spans for common pages across supported engines. |Report CER/WER, boxes, order, EOS/caps, confidence and paired latency/RSS; upstream OCRBench/OmniDocBench/olmOCRBench scores are not reproduced merely by integration. Tau-squared is an agent benchmark. Preserve Mini's one and Unlimited's19 real-page token mismatches. |
| Tesseract/Fraktur |Freeze exact line/crop geometry and compare official recoder/decoder/dictionary/spacing outputs. |Keep multi-code recoding/default DBNet and beam alternatives. Certainty, line counts, CER/WER and crop batching must be measured independently. |
| PP-OCRv6/orientation/EasyOCR |Reproduce one detector geometry/readback or SE/depthwise outlier; freeze word boxes/order for structured LayoutLM inputs. |Artifact-backed sizes, original/native crops/recognition and actual device evidence; retain CPU fallback and provenance. |
| PreprocessingT2/O10, batchingT3–T6 |Choose one cleanup/dewarp/binarization/resize/depthwise/crop-batch change against annotated pages. |Recognition/geometry quality first; identical input/output timing spans. Do not select preprocessing from synthetic-reference similarity alone. |
| Layout/TableFormerT16 |Freeze document structure/reading-order gold, then profile one GPU value projection/hot spot. |Actual decoded structure/boxes and device execution, not allocation backend alone. |
| Qwen/formula/backendR3/R6/O3/O7 |Profile one measured decoder/encoder/GEMM/thread-propagation or residency hot spot. |Preserve CPU/serial/parallel alternatives; actual F32/cache/decoded and paired memory/timing controls. Graph setup was negligible in old profiling; do not optimize it by analogy. |
| Ovis decoder cache/F2 |Freeze representative same-host cold/warm initialization, repeated recognition and one/two-context latency/RSS. |256MiB cache remains off after only0.797% warm median gain/increased RSS. Full970MiB head remains fallback. Metal cap adoption needs actual device evidence. |
| Pipeline cache identity |Start with legacy engine20 and two distinct model paths. |Force second-model/rejection/recovery proof before an engine-specific key fix; preserve completed Ovis stage-owned path identity. |
| F3–F6/VLM frontends |Separately scope CLI thread default, SmolDocling vision scheduling, DeepSeek-OCR2 crop/F16-KV or Qwen3-VL/jina resize/tiling contracts. |Current source/licensing, original input geometry and final outputs/performance; do not port caches or defaults family-wide. |
| Backend bring-up |Use actual CUDA/Metal/Vulkan hardware for a single proven compute/scheduling gap. |Verify numerical/decoded quality, actual compute placement, residency and memory; host allocation is insufficient. |
| E4/encoder ecosystem |Freeze German/Japanese representative quality inputs; reconcile feature matrix/Nomic MoE opt-in independently. |MTEB/JMTEB task quality, actual representative latency and consumer compatibility; byte equality alone is not a speed claim. |
| Restoration/OMR |Validate Q8 ESRGAN/SCUNet or one eligible notation/vocabulary candidate under its own original blueprint. |Do not publish known bad ESRGAN Q4. Transcoda remains outside permissive queue; oemer remains reference-only. Preserve completed TrOMR/Flova evidence. |
| Coverage/T9/F10/SAM |Audit one native output/control gap; coordinate sibling changes explicitly. |No silent CrispASR changes. SAM stays with its existing owner; leave unrelated work intact. |

## Permissive OCR queue

Audit actual code, checkpoint, base and dependencies first: MIT/Apache/BSD or
similarly permissive terms only. For each model use the contribution checklist:
converter/full tensor census; actual Python/ONNX blueprint and intermediate
reference; gated native math with earliest-stage/final-output comparison; C ABI,
orchestrator, CLI/server, Python/Rust/Dart; actual Q8/Q4_K task qualification;
immutable upload/readback, registry hashes and real consumers.

Proceed smallest applicable backend first after the current closure:

1. Xiaomi-OCR-0: source/base audit and exact Qwen3.5-0.8B frontend/native reuse;
   community GGUF quality is unverified.
2. LightOnOCR-3-1B: verify exact official preprocessing, tokenizer, transcription
   and grounding against the existing LightOn backend.
3. TeleOCR: the gated CPU implementation and bounded panel are complete;
   continue its numerical/representative-quality and provenance limits in
   the TeleOCR lane above before publication or platform promotion.
4. LightOnOCR-3-4B and Qwen3.5-9B: separate long-page/memory/GPU reference and
   task protocols after small dense-model parity.
5. Qwen3.6-27B: dense64-layer config and resource-qualified long-context port.
6. Infinity-Parser2-Pro: MoE40-layer/256-expert implementation and total-parameter
   storage; active parameters do not determine GGUF size.

Indra is public-code-analysis-only under the selected scope. Chandra-OCR-2,
MinerU2.5-Pro, Hunyuan-OCR, dots.ocr and other restricted candidates remain
outside this lane. Do not infer weight permission from a public repository.

## Issue71, packaging and downstream delivery

- CPU inpainting execution, 1 November 2026: native builds, unit checks and
  live model tests were executed for this authorized run. Persistent-input
  allocation and long GGUF tensor-name loading defects are corrected; all
  2,080 latent tensor payloads remain unchanged. The source package records
  executed checks separately from the remaining qualification gaps. See the
  [completed-results report](docs/inpainting-optimization-results.md).
- MI-GAN256/512 baseline and optimized paths each pass 20 original-model cases
  and 1,070 stage comparisons; corresponding native stage hashes are identical.
  Ordinary image API medians improve 2.61–3.27 times over eager on the admitted
  256/512, four/eight-thread arms. The 512/eight-thread quiet confirmation
  improves 3.24 times. Initial contended samples remain excluded and retained.
- LaMa completes 24 synthetic arms and 5,448 original-stage comparisons.
  Seven of eight fixtures pass all stages in every implementation. Full-mask
  intermediate failures remain 37/34/34, despite passing final-image gates.
  All twelve photo arms pass original RGB; optimized warm calls improve
  2.37–2.49 times over the portable graph. Strict full-model parity is negative.
- Moebius completes the four-case original replay with local/relative lambda
  reassociation. Its integrated optional CPU depthwise path preserves all eight
  prior generated/composited RGB outputs and all 608 reference metric records;
  the same 35 strict absolute-stage failures remain. A completed ordinary API
  pair also has exact RGB and known pixels, but the warm median changes from
  142.796 to 181.397 seconds under contention. No complete-image speedup is
  qualified; keep depthwise outside the recommended throughput preset.
- The resumed PixelHacker eager/candidate pair completes one cold and three
  warm calls per arm on the same executable. Native RGB agreement passes:
  filled-channel maximum difference 1, mean 0.0000949436 and known errors 0.
  Warm medians are 310.958/144.243 seconds; the observed 2.156 ratio remains a
  diagnostic result from one fixture and fixed run order. Preserve the earlier
  cancelled eager arm and timing-only revalidation unchanged. Original CUDA/FLA
  parity, portable-only graph timing and representative quality remain open.
- Optional SHA-NI model verification hashes every byte against the same digest,
  with observed hardware/block counts and scalar fallback. Its bounded memory
  hashing benchmark measures 4.6835 times the scalar throughput, separate from
  image inference. Production CPU convolution stays at 32 MiB; the private
  128 MiB workspace screen was mixed and is not integrated. The resumed control
  suites pass 244 Python/CLI tests, 20 native configurations and 21 additional
  private API-controller controls; exact reports are linked from the results.
- Actual production CLI checks cover both MI-GAN sizes and LaMa at 256/512,
  with empty masks, exact API outputs and batch-context reuse. GPU execution,
  original PixelHacker parity, representative perceptual/video quality,
  quantization, public bindings and downstream release qualification remain
  separate. The current SHA-enabled CLI panel passes for all five profiles:
  12 invocations/16 PNGs on one binary, with exact ordinary-API outputs, empty
  masks and both latent filled/empty/filled batches. The earlier CLI panels
  remain separate. Keep issue71 open, all candidates default off and the
  model/source pins intact. Follow-up work must preserve prior receipts and
  fixed numerical budgets, including every original intermediate failure.
- Source/wheel success is not release qualification. Require fresh portable
  native/WASM producers, installed bundle/ABI/platform consumers and full
  aggregation for the final source, then update release metadata coherently.
- PosFormer/handwriting: retain7/50 original versus0/50 candidate, old legacy=0
  and full-checkpoint/sequence parity debt. No new training or quality claim
  without licensed inputs and an original blueprint.
- CrispMath stays immutable0.17.12. After compatible qualified assets exist,
  separately update its pin/actual consumers. Physical-device/accessibility/
  localization/narrow-layout and App Review/store work remain consumer tasks;
  cloud/collaboration are optional separate decisions.
- CrispSorter: use the actual Rust/Tauri/Svelte extractor/config/selector paths,
  orchestrator stage21 for qualified Ovis2; stage22/Ovis3 remains unqualified.
  Require selected-library identity, fresh model hashes, outputs/confidence,
  prompt/cap rejection and fallback behavior. Do not reuse legacy VLM number space.

## Executable audit continuation — 27 October 2026

This anchor is retained for older links. Its dated commands/status snapshots
are archived in HISTORY. Use [the29 October handoff](docs/handoff-2026-10-29.md)
for current sources, jobs and exact next actions; do not repeat completed runs.
