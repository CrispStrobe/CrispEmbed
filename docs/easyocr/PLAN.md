# EasyOCR → ggml port

## NOW — active work

- Branch: `feat/easyocr-ggml`
- Worktree: `.codex/worktrees/feat-easyocr-ggml`
- Selected next item: unify detector geometry with explicit EasyOCR line mode
  and Tesseract/LayoutLM word mode, then promote the DBNet smoke path into a
  production handoff. CRAFT now passes decoded box parity: 106 native boxes
  versus 106 Python boxes on CPU and Metal.
- CRAFT cause/fix: BN folding changed Conv→BatchNorm evaluation order enough
  to attenuate low-confidence score regions. The GGUF now retains raw
  convolution weights plus BN scale/shift tensors, and the persistent graph
  executes BN explicitly. F32 reaches exact parity; F16 reaches global cosine
  >= 0.999999 and the same decoded box count.
- Status: English Gen-2 and Latin Gen-1 ResNet graphs pass the agreed 0.99
  cosine gate with mine/ref magnitude reports; decoded outputs are `5a` and
  `=#4#4#` respectively
- Latin Gen-2 conversion/reference generation works and passes a real
  `formula_quadratic.png` crop end to end (`x =644 ~`). A separate
  `scan_strip.png` crop has one explicitly diagnosed CTC near-tie at timestep
  12 (`(2a` native vs `(a` Python) and remains unaccepted.
- Next: resolve Latin Gen-2 decoded parity, validate remaining VGG/ResNet
  recognizers, and promote the two OCR ordering policies into production
  adapters before broad detector/model expansion.
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
- [ ] Exercise the structured handoff with LayoutLMv2/v3 using
      `apply_ocr=False`; no LayoutLM weights are needed for the contract test.
- [ ] Keep Tesseract LSTM as a separately measured recognizer lane and compare
      it with EasyOCR CRNN on identical detector crops.
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
