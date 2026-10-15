# CrispEmbed — Active roadmap

Updated15 October2026. Completed work, measurements and dated handoffs belong in
[HISTORY.md](HISTORY.md). Detailed current model contracts remain in their model
documents; this file contains pending work, acceptance conditions and ownership.
Read [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md), [contributing](docs/contributing.md)
and [the consumer handoff](docs/current-state-and-next-steps.md) before claiming work.

## Execution order

1. [ ] **Verify the complete main platform build after the iOS repair.** The
   scoped library-only cross-build and18-case native probe pass
   [38696900574](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38696900574).
   Completed repair and initial failures are recorded in HISTORY. Require the
   full matrix and Flutter bundling before treating current main as releasable.
2. [ ] **Diagnose Qwen3.5 larger-input numerical drift.** Compare unrounded text
   weights against the current F16-text/F32-vision source and capture original
   CPU-backend controls. Locate the first differing projection/cache stage on
   form, receipt and German inputs. Retain the0.002 cache budget and all failures;
   exact decoded text alone is insufficient. [Current evidence](docs/qwen35-ocr.md).
3. [ ] **Profile and optimize native Qwen inference.** Attribute preprocessing,
   vision, projection/dequantization, graph creation/allocation and decode cost.
   Test one gated F32 materialization/graph optimization at a time with the same
   weights, logits, IDs, EOS and text. Measure peak memory and paired timing;
   compressed files currently still materialize F32 projection matrices.
4. [ ] **Complete long-document decoding.** Finish the receipt under an adequate
   explicit cap; test literal transcription prompts separately from upstream
   figure-description behavior. Extend the1024-patch bound only after memory,
   geometry and original-reference coverage; never silently downsize production inputs.
5. [ ] **Revisit quantization on independent documents.** Q8 differs on the form;
   Q8/Q6 differs on form/German. Retain all calibration alternatives, separate
   development/calibration data from evaluation and compare full output plus
   confidence/CER/WER. High cosine and synthetic matches are not promotion gates.
6. [ ] **Publish only a validated OCR candidate.** Include source/base license
   provenance, measured precision/size/quality limits, HF card, remote byte/hash
   verification, pinned registry aliases and empty-cache download checks. Complete
   CrispSorter/CrispCalc provider/catalog work when a validated artifact exists.
   No LightOnOCR-3 HF artifact, production alias or runtime release is selected yet.
7. [ ] **Continue the permissive OCR queue below**, starting with shared small
   backends after the preceding correctness/performance work.

## Permissive OCR queue

Scope: MIT, Apache-2.0, BSD or similarly permissive code **and checkpoint/base**
terms. Inspect actual source/license/dependencies before weight use. Indra,
Chandra-OCR-2, MinerU2.5-Pro, Hunyuan-OCR and dots.ocr remain deferred from this
lane. Existing restricted registry entries are not enabled by this roadmap.

| Order | Checkpoint | License audit | Approximate Q4 storage | Pending work |
|---|---|---|---|---|
| 1 | LightOnOCR-3-0.8B | Apache-2.0 source/base pinned | 0.964GB Q4;1.213GB Q8;1.150GB Q8/Q6;1.918GB F16-text/F32-vision | Numerical drift, latency, full documents, quantized output parity and publication; priorities above |
| 2 | [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | Apache-2.0; Qwen3.5-0.8B | ~0.74 GB | Queued after shared backend; verify prompt, preprocessing, EOS, any MTP-only tensors and full decoded output |
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

- [ ] **UltraBERT native port:** use the completed pinned-source audit in
  [docs/ultrabert.md](docs/ultrabert.md). Prefer UltraBERT-Embed-150M first for
  retrieval, then base150M masked-LM and380M. Verify exact local/global masks,
  per-layer RoPE, BOS/padding IDs, CLS pooling/query prefixes and the untied MLM
  head. Add a gated native architecture/converter/reference/surface pipeline;
  no implementation, 32K validation or quantized publication is claimed yet.
- [ ] **LFM2.5 Encoder:** validate GPU scheduling, long-context behavior beyond
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

- [ ] **W1:** repair macOS arm64 Python-wheel delocate/deployment-target failure;
  validate installed wheel load, then the full matrix and aggregation.
- [ ] **V1:** reconcile the released PosFormer normalization/ceil-pooling fixes
  with current main and draftPR60 using native/reference/decoded evidence.
- [ ] **Release metadata:** reconcile source VERSION0.17.11 and published0.17.12
  before a new release; do not change the app's immutable release pin as cleanup.
- [ ] **Registry audit:** investigate unrelated missing/drifted model hashes;
  preserve existing pins until changed upstream bytes are independently validated.
- [ ] **R1:** promote any successor runtime atomically through native/WASM build,
  artifact/hash, regression, model and actual consumer validation.
- [ ] **O1/O2 handwriting:** diagnose full decoded original-checkpoint behavior
  before choosing/training new licensed weights. Current recorded50-image results
  remain7/50 original and0/50 candidate; access/resources are dependencies.
- [ ] **App follow-ups:** physical-device checks, accessibility/localization/
  narrow-layout coverage, fresh math problems and App Review/store presentation.
  Cloud deployment and collaboration remain separate optional product decisions.
  Exact scopes and dependencies: [consumer handoff](docs/current-state-and-next-steps.md).

## OCR portfolio and performance carry-forward

These tasks survive the historical handoffs. First reconcile each brief with
later code/history; an archived failure or timing is not automatically current.
The detailed original briefs are indexed below, with all evidence preserved.

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
| 2026-10-15 | `ocr-qwen35-drift` | Larger-input F16/F32 and original CPU-backend numerical controls | Source263d6abe pushed; [hosted38698415749](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38698415749) queued; adds the unchanged per-element strict comparator/budget records (supersedes earlier queued dispatch). Identical derivative hashes, F16/F32/F32-precise-RMS, original backend-off/patch-convolution controls,77 prefill diagnostics per case; preserve0.002 cache budget and full-output panel evidence. AST/format/167 clean-exit guards pass; no runtime math or default changes |
| 2026-10-06 | `main` (Crisp3DS SAM work) | SAM 2.1 Hiera-tiny image segmentation (`docs/sam2.md`): engine, converter, reference dump, `test-sam2-diff`, C API, CLI `--sam2`, server `POST /sam2/segment`, Python/Rust/Dart `CrispSam2`, registry `sam2.1-hiera-tiny` (F16, default) and `sam2.1-hiera-tiny-f32`, both pinned by hand | Parity with PyTorch CPU: F32 and F16 cos >= 0.999999 at every stage, masks identical; Q8_0 below the 0.999 gate (mask IoU 0.9895 worst, not published), Q4_K not usable. Registry fetch of the F16 verified. **Note for the owner:** `tools/fetch_model_hashes.py --check` reports `model_hashes.h` stale for reasons unrelated to SAM 2 — no pin for `nomic-embed-vision-v1.5-f16` and for `jina-ocr-v1-q4_k` / `-q8_0`, and three pinned files whose upstream bytes changed. Deliberately not re-pinned here. Not yet: 73-photo end-to-end check, Metal timing |
