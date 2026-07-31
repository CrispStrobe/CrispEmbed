# EasyOCR → ggml port

## NOW — active work

- Branch: `feat/easyocr-ggml`
- Worktree: `.codex/worktrees/feat-easyocr-ggml`
- Status: English Gen-2 and Latin Gen-1 ResNet graphs pass the agreed 0.99
  cosine gate with mine/ref magnitude reports; decoded outputs are `5a` and
  `=#4#4#` respectively
- Latin Gen-2 conversion/reference generation works and passes a real
  `formula_quadratic.png` crop end to end (`x =644 ~`). A separate
  `scan_strip.png` crop has one explicitly diagnosed CTC near-tie at timestep
  12 (`(2a` native vs `(a` Python) and remains unaccepted.
- Next: resolve Latin Gen-2 decoded parity, then validate remaining
  VGG/ResNet recognizers and port detectors
- CRAFT source/inference audit is complete; the Python `-ref.gguf` dumper is
  implemented and produces an 80-stage reference archive with score-map and
  decoded-box-count metadata on a 256x512 smoke input.

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
- [ ] Port CRAFT detector and preserve its upstream license notice.
- [ ] Audit and port DBNet-18 and DBNet-50 detector checkpoints.
- [ ] Port every Gen-1 and Gen-2 recognition checkpoint, sharing one runtime
      across language-specific charsets.
- [ ] Add a weight-free LayoutLMv2/v3 OCR handoff test: words + normalized
      boxes in, tokenization-compatible structured output out.
