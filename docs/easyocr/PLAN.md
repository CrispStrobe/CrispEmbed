# EasyOCR → ggml port

## NOW — active work

- Branch: `feat/easyocr-ggml`
- Worktree: `.codex/worktrees/feat-easyocr-ggml`
- Selected next item: unify detector geometry with explicit EasyOCR line mode
  and Tesseract/LayoutLM word mode, then promote the DBNet smoke path into a
  production handoff. CRAFT decoded parity now passes with the runtime-BN
  GGUF: 106 native boxes versus 106 Python boxes.
- CRAFT cause/fix: the older folded-weight F16 artifact accumulated enough
  convolution/BN error to produce 107 boxes. Re-converting with raw
  convolution weights plus explicit BN scale/shift tensors makes F32 match to
  floating-point noise; runtime-BN F16 reaches score-map global cosine
  `0.9999999` and the same 106 decoded boxes. CPU-forced and Metal outputs are
  byte-identical.
- Status: English Gen-2 and Latin Gen-1 ResNet graphs pass the agreed 0.99
  cosine gate with mine/ref magnitude reports; decoded outputs are `5a` and
  `=#4#4#` respectively
- Latin Gen-2 conversion/reference generation and dynamic-width recognition
  now pass real `formula_quadratic.png` and `scan_strip.png` references. The
  scan uses EasyOCR's actual width 128 (`ceil(64 * 520 / 260)`), not the fixed
  200-column standalone shape; all six captured stages pass and logits have
  `0/31` argmax mismatches, with decoded output `82` matching Python.
- [x] Re-generated and reran the Latin family against fresh references from
      the official checkpoints, current EasyOCR checkout, current dumper, and
      explicit widths. Latin Gen2 formula (width 200) passes all six stages
      with minimum cosine `0.9999106` and decodes `x=0442`; Latin Gen2 scan
      (width 128) passes with minimum cosine `0.9996817`, logits cosine
      `0.9999978`, and decodes `82`; Latin Gen1 ResNet scan (width 128) passes
      with minimum cosine `0.9996817`, logits cosine `0.9999934`, and decodes
      `==#`. The older similarly named local references failed already at
      `input_image` and had conflicting decoded metadata, so they are stale
      fixtures and are not parity evidence.
- [x] Re-generated English Gen2 references after the native sampler update.
      The fixed 200-column scan passes with exit code 0, decoded `032`, and
      logits cosine `0.9979853`. The actual EasyOCR width-128 scan decodes
      `@32` with `0/31` argmax mismatches; all stages pass the global gate, but
      the strict row-wise logits gate remains open at cosine `0.973824` on
      timestep 11. This reproduces the earlier result with fresh official
      inputs and isolates the remaining exception to dynamic-width numerical
      sensitivity.
- [x] Add repeated native/reference timing and output checks. On this M1
      Metal build, 10 repeated recognitions were compared with 10 Miniconda
      PyTorch CPU recognitions using identical images and widths: Latin Gen2
      formula 200 native/reference total `16.523/12.460 ms` (`1.33x`),
      decoded `x=0442`/`x=0442`; Latin Gen2 scan 128 `10.885/7.137 ms`
      (`1.53x`), decoded `82`/`82`; Latin Gen1 ResNet scan 128
      `154.082/78.648 ms` (`1.96x`), decoded `==#`/`==#`; English Gen2 scan
      200 `16.536/10.035 ms` (`1.65x`), decoded `032`/`032`; and English
      Gen2 scan 128 `10.697/7.287 ms` (`1.47x`), decoded `@32`/`@32`.
      Native is currently slower in every measured total/graph path despite
      using Metal; graph/kernel and recognizer-width optimization are explicit
      performance TODOs. These are cross-runtime numbers (PyTorch CPU versus
      native Metal), so they are directional rather than final apples-to-
      apples measurements.
- [x] Fix the benchmark-only repeated-recognition quality regression: the
      four host-reset LSTM initial-state tensors could alias intermediate
      graph storage under the allocator after the first run. Marking them as
      graph outputs keeps their storage live; repeated Latin Gen2 now remains
      `82` and Latin Gen1 remains `==#`, matching Python.
- [ ] Add equivalent repeated per-stage timing and output manifests for CRAFT
      detector, DBNet detector→EasyOCR page modes, and Tesseract line/page
      paths. Their current checks establish geometry/text or tensor evidence
      but do not yet provide native/reference timing ratios. Any slower native
      stage and any worse text/box/ordering output must be recorded as a
      separate optimization or quality TODO before those lanes are accepted.
- [x] Add the repeated CRAFT inference benchmark. On the fresh
      `scan_strip.png` reference input, 10 warm runs produced 106 boxes in
      both implementations: Miniconda PyTorch CPU averaged `396.027 ms`,
      while native runtime-BN F16 Metal averaged `850.018 ms` graph time
      (`2.15x` directional slowdown). This is not an apples-to-apples device
      comparison, but the native graph/kernel path is a clear optimization
      TODO; CRAFT output quality is currently on par for this fixture.
- [x] Recheck CRAFT against a fresh official reference after the benchmark
      audit. The reference for `scan_strip.png` uses a 288x544 canvas and
      decodes 106 boxes. The old folded F16 GGUF decoded 107, while a freshly
      converted runtime-BN F32 GGUF matches every captured tensor to floating-
      point noise and runtime-BN F16 decodes 106 with score-map global cosine
      `0.9999999`. CPU and Metal runs are byte-identical. The older local
      reference produced 104 versus 106 and remains stale. Inference-only
      repeated CRAFT timing and output manifests remain performance TODOs;
      the prior 2.34 s native diff versus 9.13 s Python dump included unequal
      model-load/serialization work and is not a benchmark ratio.
- [x] Extend the CRAFT diff report with max/mean/RMS error and magnitudes. The
      fresh mismatch's earliest divergent captured stage is `basenet_0`
      (`max_abs=1.52823`, `rms=0.195515`, global cosine `0.995623`); the score
      map reaches `max_abs=0.06910`, `rms=0.008026`, global cosine `0.999716`.
      A native component at text max `0.701290` crosses EasyOCR's `0.7`
      threshold, while the Python component structure yields 106 boxes and
      native yields 107. Do not tune the postprocessing threshold; fix the
      first CNN/layout or numerical divergence before claiming box parity.
- [x] Add early CRAFT VGG taps to the diff harness. The `slice1` tap global
      cosines are `0.9999831`, `0.9998919`, `0.9999237`, `0.9997602`, and
      `0.9992513` across the first block; later source taps decline to
      `0.9986382`, `0.9978006`, `0.9975555`, and `0.9956226` at `basenet_0`.
      The mismatch accumulates through the CNN rather than appearing solely in
      CRAFT connected components. The next quality task is convolution/BN or
      layout numerical parity, with the captured taps retained for debugging.
- [x] Restore the official MMOCR DBNet ResNet-18 IC15 checkpoint and add a
      fresh `scan_strip.png` reference plus `test-dbnet-diff`. The native F16
      backup passes the probability-map boundary with max error `0.00154233`,
      RMS `0.00008044`, cosine `0.9999974`, global cosine `1.0000000`, and
      96 decoded regions. Q4_K produces the same 96 regions but fails tensor
      parity at cosine `0.9311001` / global `0.9986384`; do not repeat the
      README's old blanket Q4 parity claim. The diff now retains and compares
      all backbone, neck, head, and final-map taps; F16 passes them all under
      the global/magnitude gate. Q4_K's earliest real divergence is already
      `backbone_stage_0` (global cosine `0.9960006`, RMS `0.07697`), worsening
      through the neck and ending at final-map cosine `0.9311001`; this is a
      quantization-quality TODO, not postprocessing. Inference-only timing
      remains open. A fresh native CPU-forced page run reports detector graph
      `4178.6 ms`, postprocess `8.3 ms`, total `4186.9 ms`, and 12 line units;
      Miniconda PyTorch CPU inference-only timing is now isolated at
      `1213.450 ms` for the same 736x1472 input, so native CPU is `3.45x`
      slower and Metal is `3.90x` slower (`4732.1 ms` graph). Both native
      backends retain the F16 reference taps and 96 decoded regions. Graph and
      kernel optimization is a mandatory performance TODO for both backends.
      The detector now uses a shape-keyed persistent graph, and tap retention
      is opt-in via `OCR_DETECT_CAPTURE_TAPS=1`. Corrected rapid-mode repeated
      benchmarking gives `5661.1 ms` warm CPU with 4 threads and `2907.2 ms`
      with 8 threads, versus `1213.450 ms` Python CPU; Metal is `3499.4 ms`.
      Python uses 4 compute threads, so the native 8-thread result is still
      `2.40x` slower despite twice the compute threads. The same Python
      blueprint on MPS averages `577.342 ms`, making native Metal `6.06x`
      slower on the same device; native convolution/deconvolution kernel
      optimization remains mandatory.
      With `OCR_DETECT_THREADS=8`, native CPU improves to `2727.3 ms` graph,
      `9.3 ms` postprocess, `2736.7 ms` total, still `2.25x` slower than the
      reference; output remains 12 readable line units with `Brighton`.
- Next: validate remaining VGG/ResNet
  recognizers, and promote the two OCR ordering policies into production
  adapters before broad detector/model expansion.
- The local F16 family check found and repaired a stale English Gen2 artifact:
  the old file had flattened convolution weights and unfused BatchNorm tensors
  and aborted graph construction. Re-conversion from the official checkpoint
  produces the current 36-tensor GGUF and runs normally. Latin Gen1 ResNet F16
  passes all six stages at dynamic width 128 (minimum cosine `0.999860`,
  logits `0.999993`) and decodes `==#`. English Gen2 F32 and regenerated F16
  both decode `@32` with `0/31` argmax mismatches; both retain the same
  shape-specific row-11 logits minimum cosine (`0.973824` F32,
  `0.975300` F16) despite global cosine around `0.9998`. This is not an F16
  artifact regression, but that dynamic-width logits gate remains open and is
  not claimed green.
- [x] Localize the English width-128 exception with CPU/Metal A-B and
      class-level logits diagnostics. `EASYOCR_FORCE_CPU=1` and the default
      Metal backend produce identical reports; the first row-specific drift is
      BiLSTM-1 timestep 11 (`cos 0.998517`, norms `27.04/27.76`) and the final
      projection amplifies it at logits timestep 11 (`cos 0.973824`, norms
      `28.81/30.84`). The largest single difference is blank class 0
      (`15.94` native vs `17.69` Python); all 31 timestep argmaxes still match.
      This is a backend-independent numerical-sensitivity case, not an F16
      layout or tokenizer error. The strict per-row tensor gate remains open;
      decoded parity is recorded separately.
- [x] Quantify the input-boundary residual behind the open row gate: the
      native linear sampler differs from OpenCV by at most one uint8 level on
      the scan fixture, across roughly 1.1k resized pixels. The new
      `EASYOCR_FORCE_CPU=1` diagnostic confirms CPU and Metal are identical,
      and class-level tracing shows the drift is amplified in blank-class
      logits rather than caused by a token-table or backend mismatch.
- [x] Prove causality with an identical-raster control: feeding both Python
      and native the exact 128x64 OpenCV-resized grayscale raster makes input,
      CNN, sequence, both BiLSTM layers, and logits cosine `1.000000` (only
      ~1e-6 float noise). The graph and recurrent math are therefore correct;
      the remaining strict-gate failure is solely the native sampler's
      one-level uint8 difference from OpenCV. Keep this sampler boundary
      isolated and do not weaken tensor parity or add OpenCV as a production
      dependency.
- [x] Port the generic OpenCV CV_8U `INTER_LINEAR` sampler structure into the
      native path: float coordinate tables are quantized to 11-bit coefficients,
      horizontal interpolation is accumulated in 32-bit integers, and the
      vertical result uses the 22-bit rounded cast. This is cleaner and more
      faithful than the former float-only sampler, but the official macOS ARM
      OpenCV build still emits a one-level residual on roughly 1.1k pixels;
      the English width-128 logits row gate therefore remains open. The exact
      raster control still passes, so no graph or LSTM change is justified.
- DBNet→EasyOCR page smoke is now wired in `test-easyocr-dbnet`: the
  existing `cstr/dbnet-ic15-GGUF` F16 detector finds 98 regions on
  `scan_strip.png`, crops them before CRNN inference, and recognizes the
  `Brighton` region. The harness now has explicit `lines` mode (EasyOCR
  grouping plus dynamic-width CRNN graphs) and `words` mode (LayoutLM/Tesseract
  handoff style). This is a pipeline smoke gate only; Python box/text parity
  and production orchestration remain open. The detector-independent
  production handoff is now explicit: `easyocr_pipeline::run_regions` accepts
  caller-supplied detector boxes and applies the configured lines/words
  ordering, crop, recognizer, and LayoutLM normalization path. The model-backed
  pipeline test replays the DBNet boxes through this API and matches the normal
  98-record run; external Tesseract TSV parity remains open.
  The public signature now accepts only `easyocr_layout::region`, keeping
  DBNet-specific types inside the implementation. Compile/link proof passes;
  the post-merge model replay is currently blocked by the unrelated shared
  `ggml` submodule checkout difference and is not claimed green.
  A real page comparison confirms why TSV parity cannot be asserted by zipping
  records: native DBNet/CRNN `words` mode emits 98 records, while Tesseract
  5.5.2 `--psm 6` emits 106 TSV words. The first geometry already differs
  before recognition (`[46.97,0,62.56,20.88]` native versus `[50,0,58,19]`
  TSV), and full-page Tesseract reads `Drighton;` while the instrumented
  internal PSM7 crop reads `Brighton`; page segmentation/crop selection remains
  the active parity gate.
- [x] Verify the Miniconda reference environment and restore the exact official
      `english_g2.pth` checkpoint (MD5 `5864788e1821be9e454ec108d61b887d`).
      Torch 2.7.1, EasyOCR 1.7.2, OpenCV, and Transformers import cleanly, and
      EasyOCR's own recognizer initializes on CPU. On the same 12 native DBNet
      line crops, Python and native recognition match exactly on 8/12 lines;
      the remaining differences are small CTC character/spacing choices, and
      both paths read `Brighton`. The previous Python-reference blocker was the
      missing checkpoint asset, not a Miniconda/Torch problem.
- [x] Correct the recognizer preprocessing to the actual EasyOCR page path:
      rounded ITU-R 601 grayscale plus `cv2.resize` interpolation value 1
      (`INTER_LINEAR`, because EasyOCR passes `Image.Resampling.LANCZOS` to
      OpenCV). The fresh official Python `-ref.gguf` and native graph now pass
      `crispembed-diff` at every captured stage: input `0.999084`, sequence
      `0.991019`, BiLSTM-0 `0.998646`, BiLSTM-1 `0.999340`, logits `0.997985`
      minimum cosine, with global cosines and mine/ref magnitudes printed. The
      decoded output also matches (`032`). The previous divergence was the
      native/dumper bicubic preprocessing mismatch, not recurrent math.
- [x] Add an optional width argument to `test-easyocr-diff` and verify the
      dynamic-width Latin Gen2 page shape. At width 128, the fresh Python and
      native references pass input, CNN, sequence, both BiLSTM layers, and
      logits (minimum cosines `0.999682`, `0.999883`, `0.999781`, `0.999823`,
      `0.999907`, `0.999998`) with identical decoded output `82`.
- Tesseract parity is explicitly **not proven**. The converter, Python
  reference dumper, and `test-tesseract-lstm-diff` exist, but there is no
  recorded completed reference run for the exact installed `eng.traineddata`,
  no verified source hash for the backup GGUF, and no full page-segmentation or
  word-spacing parity. Tesseract remains a separate recognizer/segmentation
  acceptance lane, not ground truth for the EasyOCR page smoke.
- The page path audit found a separate preprocessing boundary: EasyOCR's
  `get_image_list` uses OpenCV `resize(..., interpolation=1)` (bilinear), while
  the standalone recognizer reference fixture was generated with PIL bicubic.
  An experimental native bilinear substitution failed the existing diff at
  `sequence_input`, `bilstm_0`, and `logits` (`Ea` versus `5a`), so it is not
  retained in production until a matching bilinear `-ref.gguf` is regenerated.
- A strict port of EasyOCR's horizontal gap thresholds was rejected for the
  current DBNet artifact: it split 98 fragmented detector regions into 26
  recognition units instead of the existing 12 line units. DBNet therefore
needs a detector-specific line adapter before those thresholds can replace
the current y-band grouping.
- The first DBNet adapter is now explicit in `easyocr_layout`: it preserves
  fragment-tolerant y-band aggregation for DBNet line crops, while word mode
  remains left-to-right y-band ordering. Horizontal-gap splitting stays a
  later detector-specific refinement.
- CRAFT source/inference audit is complete; the Python `-ref.gguf` dumper is
  implemented and produces an 84-stage reference archive with score-map and
  decoded-box-count metadata on a 256x512 smoke input.
- The real CRAFT checkpoint now converts to a 54-tensor BN-folded F16 GGUF;
  the persistent GPU graph builds and passes input, all VGG taps, U-Net feature,
  and NHWC-reordered score-map parity at the agreed global cosine >= 0.99 gate
  on both diagnostic CPU and default Metal backends. Score reports retain the
  low sparse-tensor row cosine and mine/ref norms for review. Native decoded
  box postprocessing remains open.

## Scope

EasyOCR is a pipeline, not a single checkpoint:

1. Text detection: CRAFT by default, with DBNet18/50 alternatives.
2. Text recognition: one CRNN checkpoint per script/language family.
3. CPU-side preprocessing, CTC decoding, confidence calculation, dictionaries,
   grouping, and paragraph post-processing.
4. LayoutLMv2/v3 compatibility: accept externally produced words and normalized
   boxes, matching the Transformers processor contract. Transformers' default
   `apply_ocr=True` path uses PyTesseract; it is not an additional LayoutLM OCR
   model to convert.

## Interoperability design: detector, ordering, recognizer, handoff

Tesseract does not use DBNet. Its page-segmentation modes use image
preprocessing and connected-component/block/line analysis before its own LSTM
recognizer. EasyOCR uses CRAFT by default, supports DBNet18/50 as detector
alternatives, and normally turns detected boxes into ordered line crops for a
CRNN or Transformer recognizer. LayoutLMv2/v3 are downstream document models:
their Transformers processors normally invoke PyTesseract and pass words plus
normalized boxes; with `apply_ocr=False`, callers supply those fields. The
Transformers library has no single OCR detector, and TrOCR is a recognizer,
not a page detector.

We will port the compatible contracts, not pretend these are identical model
architectures:

```text
CRAFT | DBNet | Tesseract-compatible geometry
                    ↓
             boxes + scores
                    ↓
       ordering/grouping policy
          ↙                       ↘
  EasyOCR lines                 word records
  dynamic CRNN                 Tesseract/LayoutLM
          ↓                       ↓
       line text       text + pixel/normalized boxes
                    ↘       ↙
             structured OCR handoff
```

The stable handoff record is `text`, pixel coordinates, confidence, block,
line, reading-order index, and optional LayoutLM `[0,1000]` coordinates. The
two supported policies are deliberately separate and independently tested:

1. `lines`: EasyOCR-style y clustering plus x sorting, crop each line, and
   run the native dynamic-width EasyOCR recognizer.
2. `words`: Tesseract/LayoutLM-style y-band and x ordering, retain each
   detector box, recognize each crop, and export words for downstream models.

DBNet remains a native detector adapter. Tesseract-style ordering is a
postprocessing adapter, not a replacement DBNet checkpoint. This lets us
compare CRAFT, DBNet, and external/Tesseract geometry against the same
recognizer and LayoutLM consumer.

### Selected implementation sequence

- [x] Extract detector boxes/scores into a reusable layout-region API with
      explicit line and word ordering policies.
- [x] Move `lines` and `words` policy selection into the reusable
      `easyocr_pipeline` configuration API; retain the smoke test as a
      model-backed regression caller.
- [x] Add a manifest-driven Python reference for boxes, order, crops, text,
      confidence, and normalized boxes. `tools/easyocr_postprocess_reference.py`
      consumes EasyOCR Python `readtext(detail=1)` output and
      `tests/test_easyocr_postprocess_reference.py` covers both modes.
- [x] Emit the same versioned manifest from `test-easyocr-pipeline` and add
      `tools/compare_easyocr_manifests.py`; native serialization and comparator
      self-check pass on the 98-word DBNet page run, with explicit mismatch
      coverage in `tests/test_easyocr_manifest_compare.py`.
- [x] Add native handoff invariants for word-mode line/x ordering and
      normalized-box bounds; external Tesseract TSV geometry/text parity is
      still pending.
- [x] Add a standard-library Tesseract TSV geometry/order comparator and
      self-test; a real page comparison remains an evidence gate, not a claim
      of Tesseract text parity.
- [x] Add independent postprocessing tests: grouping, reading order, CTC
      collapse, dictionary/vocabulary validation, EasyOCR custom-mean
      confidence, and box normalization. The production recognizer now uses
      the same nonblank confidence convention.
- [x] Exercise the structured handoff contract for LayoutLMv2/v3 using
      `apply_ocr=False`; `tools/validate_layoutlm_handoff.py` emits the
      processor's `words`/`boxes` payload and preserves confidence/pixel boxes
      in sidecar metadata. No LayoutLM weights are needed for this gate.
- [x] Keep Tesseract LSTM as a separately measured recognizer lane. The
      `test-ocr-identical-crops` harness feeds exact shared RGB crops to the
      dynamic-width EasyOCR CRNN and grayscale Tesseract LSTM; three official
      TSV boxes show structurally matching text with recognizer-specific
      ambiguous-character and punctuation differences.
- [ ] Validate Tesseract beam confidence against official certainty
      aggregation. The new confidence comparator shows greedy text matching on
      two of three direct English lines; beam confidence remains a sequence
      probability and is not treated as per-word certainty. Native greedy
      `word_confidence` now follows Tesseract's source rule (minimum selected
      path log-probability, then `100 + 5*certainty`); the direct second-line
      result is `0.965889` versus official `0.959698`. Page-level aggregation
      and beam certainty remain open.
- [x] Prove the controlled line-recognizer boundary separately: exact hashed
      Homebrew `eng.traineddata`, Python `-ref.gguf`, native captures, decoded
      text, and the official instrumented PSM7 internal crop all match.
- [ ] Compare page segmentation, spacing, and CLI crop geometry independently;
      direct line fixtures are not the same internal crops selected by PSM7.
      The native classical page-segmentation adapter is now wired behind the
      explicit `--tesseract-pageseg`/stage option and has a model-free synthetic
      geometry regression. It now also bypasses generic scan cleanup so page
      segmentation sees original-image coordinates, and its row threshold
      rejects sparse antialiasing bridges on gray paper; this does not close
      real CLI parity.
      On `scan_strip.png`, the tuned native CLI path improved from 3 to 7
      decoded regions, while official Tesseract `--psm 3/6` emits 12 lines;
      exact RGB-to-gray conversion is now shared with the proven reference.
      Height-based splitting now recovers 12 candidates and 12 decoded regions
      on `scan_strip.png`; crop widths are tightened per split band. Text still
      differs on `Meryton` and punctuation/quotes, so decoded page parity and
      official crop equivalence remain open.
      Review of Tesseract `textord/makerow.cpp` confirms its authoritative
      boundary is connected blobs assigned by vertical overlap, line size,
      spacing, and fitted baselines; our projection splitter is only an
      interim adapter. An opt-in component prototype is available behind
      `CRISPEMBED_TESSERACT_COMPONENT_PAGESEG`. After a Tesseract-style
      reassociation pass for short/detached blobs it produces the expected 12
      rows on `scan_strip.png`, but its enlarged first-line crop currently
      worsens recognizer output, so it remains experimental and is not enabled
      in production.
      `tools/compare_tesseract_page_geometry.py` now measures the independent
      geometry boundary from official TSV level-4 rows. On `scan_strip.png`
      The legacy component path remains the default because the German
      official-print gate regressed under the newer baseline matcher. With
      `CRISPEMBED_TESSERACT_COMPONENT_BASELINE=1`, the baseline experiment now
      has 12/12 indexed rows with mean IoU `0.813562` after vertical crop
      tightening; the projection fallback has 12/12 with mean IoU `0.865993`.
      Its first-line crop is `[48,0,434,20]` and short final-row crop
      `[27,237,72,22]`; both page ends decode coherently, although the
      baseline variant drops the final exclamation mark. Character choices and
      quote/spacing differences remain a
      decoded-text parity gate. A page-level beam A/B at widths 1, 5, 10, and
      25 keeps the same first-line choices; generic CTC beam search remains
      opt-in and is not the cause of the remaining CLI discrepancy.
- [x] Record the exact `.traineddata` SHA-256 in converted and reference GGUF
      metadata; the controlled-line reference and stage/output parity are
      complete, while page parity remains open.
- [x] Align the diagnostic reference dumper and native recognizer with
      Tesseract's actual Leptonica `pixScaleGrayLI` fixed-16 bilinear
      contract (top-left sampling and edge replication); the earlier
      half-pixel resize was wrong. Full Tesseract page preprocessing parity
      remains a separate acceptance gate.
- [x] Expand the Tesseract lane with CC0 Commons OCR-document and receipt
      fixtures, including source URLs, licenses, and SHA-256 metadata. The
      receipt crop that previously failed at `after_conv_fc` now passes all
      9 diff stages at cosine `1.000000`; native/Python raw greedy output is
      still distinct from the CLI's normal choice-search output.
- [x] Trace the remaining CLI discrepancy: the exact Homebrew model carries
      `training_flags=65` (`TF_INT_MODE` enabled). Tesseract CLI therefore
      quantizes activations and per-output weight rows to int8 before its
      recode beam; our current F32 graph is not expected to reproduce those
      logits. Metadata now preserves the flag in both GGUF and references.
- [x] Implement the first int8-equivalent activation path and a matching
      quantized Python reference; all 9 captured stages pass the 0.99 gate
      (int-mode logits cosine `0.997227`), but native still decodes
      `Lhey ... Drighton` versus the CLI's `ihey ... Brighton`.
- [x] Add and test an opt-in generic CTC prefix beam through
      `CRISPEMBED_TESSERACT_BEAM_WIDTH` at widths 2, 3, 5, 10, 16, 25, and 50;
      it does not change the native result, so it is not the CLI explanation.
- [ ] Match Tesseract's exact int-mode logits (lookup-table nonlinearities and
      quantized matrix arithmetic), then port its recode beam/dictionary
      scoring and validate decoded output.
- [x] Added the 1/256 Tesseract nonlinear LUT contract and reconstructed
      per-row int8 dot products. Native/Python int-mode logits now reach
      cosine `0.998405` with identical decoded output; generic CTC and
      Viterbi/recode-style diagnostic beams at widths 2-50 still do not select
      CLI `Brighton`.
- [x] Obtain an official Tesseract internal activation/raw-row comparison;
      the remaining discrepancy was traced to seeded Convolve padding and is
      now resolved below.
- [x] Resolve the official Tesseract discrepancy: PSM7 creates a 601x36
      internal crop, and its first divergent tensor was `Convolve` boundary
      padding. Tesseract uses a seeded `TRand` based on serialized
      `sample_iteration` rather than zero padding. GGUF metadata, the Python
      reference, and native path now preserve/replay that state. Fresh
      official-vs-native comparison passes all 9 captured tensors at cosine
      1.000000 (logits max error 6.6e-7) and decodes `Brighton` identically.
- [ ] Port and validate Tesseract recode/dictionary scoring separately from
      the now-proven network arithmetic; do not enable a production beam from
      the diagnostic implementation.
- [x] Preserve `recoder_map`/`recoder_offsets` and enforce legal recoder-code
      prefixes in the opt-in diagnostic beam. Official PSM7 width-25 testing
      remains `Brighton` with 9/9 tensor stages passing. Certainty aggregation,
      top-N transition rules, and DAWG dictionary scoring are still not ported
      or production-enabled.
- [x] Run the six available English line fixtures with official CLI PSM7 and
      compare `language_model_ngram_on=0` versus `=1`: outputs were identical
      for every line. This keeps DAWG scoring as an unproven pending feature,
      while documenting that the observed line differences are currently
      caused by CLI crop/spacing/case behavior.
- [x] Harden `test-tesseract-lstm-diff` so decoded metadata mismatches fail the
      test. The official 601x36 PSM7 crop remains 9/9 plus decoded-pass; the
      direct six-line sweep initially exposed line 4 as an input mismatch.
- [x] Match the reference dumper's RGB-to-gray conversion to native
      `stb_image` (`(77R+150G+29B)>>8`), eliminating the line-4 input error.
      All six direct line fixtures now pass decoded parity; input tensors are
      exact and stage cosines remain at or above `0.998821`.
- [x] Run exact hashed Homebrew English references after the Leptonica fix:
      the controlled line fixture decodes identically in native/Python as
      `_ “ ihey are going to be encamped near Drighton ;`, with all 9 stages
      passing the 0.99 gate and logits cosine `0.999863`. The full-page image
      is not a valid single-line recognizer fixture; page segmentation remains
      a separate acceptance gate.
- [ ] Only after these gates pass, generalize DBNet18/50 and all recognizer
      languages, then run GPU performance A/B.

The first implementation slice targets the Generation-2 VGG recognizer
(`english_g2`), then expands to every shipped recognizer and both detector
families:

`[1, 64, W] → VGG CNN → width sequence → 2× BiLSTM → linear CTC`

The model is grayscale, normalizes pixels to `[-1, 1]`, resizes to height 64,
right-pads to the configured width, and uses greedy CTC decoding. The model
weights are PyTorch `.pth` state dictionaries; the language character list is
metadata, not learned parameters.

## License policy

- EasyOCR source and its released model files are treated as Apache-2.0, with
  the upstream copyright/license notices preserved in GGUF metadata and the
  model catalog.
- CRAFT is separately attributed under its upstream BSD-2-Clause license.
- DBNet code/checkpoint provenance is recorded separately; the EasyOCR release
  asset is not silently relabeled based only on the repository's source license.
- PyTorch/torchvision are build-time conversion dependencies only and are not
  redistributed with the runtime.
- Every converted artifact gets a machine-readable `general.source` and
  `general.license`, plus a checked-in license manifest before publication.

## Gates

- [x] Read and freeze the exact Python recognition path and tensor shapes.
- [x] Add a PyTorch/safetensors-to-GGUF converter with explicit charset and
      model metadata.
- [x] Add a per-stage reference dumper and license manifest skeleton.
- [x] Add the CRAFT Python reference dumper with EasyOCR preprocessing,
      score-map captures, and box-count metadata.
- [x] Add the CRAFT converter with explicit BN folding and source/license
      metadata; validate it against the released checkpoint.
- [x] Fix the CRAFT VGG slice/pool/ReLU schedule, validate U-Net feature and
      score/link logits with explicit NHWC layout conversion on CPU and Metal.
- [x] Validate CRAFT decoded boxes and postprocessing against the Python box
      count; both F32/F16 artifacts produce 106 boxes on CPU and Metal.
- [x] Run the dumper against a real English Gen-2 checkpoint and inspect the
      generated `-ref.gguf` tensors.
- [x] Add a C++ diff fixture using `crispembed_diff::Ref`, including explicit
      layout conversion and cosine/magnitude reports for every captured stage.
- [x] Fix the current graph/output divergence before model-family
      generalization; accept per-stage cosine >= 0.99 and inspect global
      cosine plus mine/ref magnitudes for sparse CNN captures.
- [x] Remove VGG graph hardcodes for feature channels and sequence width;
      derive them from the built CNN tensor and GGUF metadata.
- [x] Add the Gen1 ResNet converter BatchNorm folding and residual graph path;
      convert the real Latin Gen-1 release checkpoint, generate its Python
      `-ref.gguf`, and pass all captured stages at the 0.99 gate.
- [ ] Implement the recognizer as reusable ggml graphs on CPU/Metal/CUDA/
      Vulkan, including GPU-resident CNN + BiLSTM and CTC logits. A CPU-scalar
      forward is permitted only as a temporary diagnostic oracle, not as the
      shipped implementation.
- [ ] Decide whether the BiLSTM should use a graph-unrolled cell or a fused
      ggml op; the current ggml fork exposes no LSTM primitive, so the initial
      implementation will use a static graph-unrolled cell with shared GPU
      weights across all time steps. Benchmark any later fused op against this
      graph while preserving the same captures.
- [ ] Store convolution weights in the ggml `conv_2d` layout rather than the
      converter's temporary flattened CPU layout; validate Metal/CUDA/Vulkan
      kernel support before quantizing them.
- [x] Verify decoded strings against the Python reference metadata on real
      crops; the diff test separately covers the harness-blind CTC token table
      and greedy collapse by requiring `easyocr.decoded` parity.
- [ ] Add confidence, dictionary, grouping, and paragraph-postprocessing tests
      once detector/orchestration outputs are wired.
- [ ] Add quantization rules and CPU/Metal parity before orchestration wiring.
- [x] Port CRAFT detector, preserve its upstream license notice, and validate
      its decoded box-count boundary.
- [ ] Audit and port DBNet-18 and DBNet-50 detector checkpoints.
- [ ] Port every Gen-1 and Gen-2 recognition checkpoint, sharing one runtime
      across language-specific charsets.
- [x] Add a weight-free LayoutLMv2/v3 OCR handoff contract test: ordered words,
      confidence, pixel boxes, and normalized `[0,1000]` boxes are preserved.
