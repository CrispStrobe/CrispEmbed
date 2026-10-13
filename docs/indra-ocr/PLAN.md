# Indra OCR: public-code feasibility

## Status — analysis complete, 13 October 2026

The user selected public-code analysis pending clarification of weight-use eligibility.
The public client pipeline has been traced. Native runtime support remains a future
lane: architecture/operator coverage and numerical parity require inspection and
execution of the licensed exported graphs. This audit downloaded public source
files and read repository metadata; it did not download model binaries or run OCR.
No registry entry, converted weights, benchmark or publication is claimed.

## Frozen sources

- [Inference code](https://github.com/datalab-to/indra/tree/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b), package version 0.3.1.
- [Model repository snapshot](https://huggingface.co/datalab-to/indra-ocr/tree/5993a956d7ea1d43c0d40d409962bb0b033ac1e8).
- [Model license](https://huggingface.co/datalab-to/indra-ocr/blob/5993a956d7ea1d43c0d40d409962bb0b033ac1e8/LICENSE).
- [Code license](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/LICENSE): Apache-2.0.

[models.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/models.py)
defaults to HF tag `v3.1`. That tag's immutable revision was not resolved in this
audit: metadata requests returned HTTP 429. Resolve and record the intended
code/model pair before reference tests; the observed HF `main` snapshot above
must not silently substitute for the package's declared tag.

The model card's roughly 18 MB footprint and 1M recognizer parameter count are
publisher descriptions. They are not measurements of a CrispEmbed implementation.
The small recognizer is one component of a document pipeline.

## Actual inference contract

The [pipeline](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/pipeline.py)
uses ONNX Runtime on CPU, with intra-op threads configurable, inter-op threads
set to one and the CPU allocation arena disabled. Mandatory local inputs are
`det.onnx`, `rec.onnx` and `meta.json`; reading order and the additional classifiers
are discovered by file presence. The published snapshot has ten ONNX files.

| Stage | Publicly visible contract | Native port boundary |
|---|---|---|
| Orientation and deskew | Optional `orient.onnx`; 768-square padded ink thumbnail; turn only above 0.95 probability. Tilt search uses OpenCV affine interpolation. | Match threshold, rotation direction, pixel rounding and border modes. |
| Line detection | `det.onnx` consumes grayscale ink maps padded to multiples of 32; output is three channels: core logits and top/bottom distances. Decode at stride 2 with four-connected components, then restore page scale. Adaptive rerun for sufficiently many small lines. | This is not DBNet's polygon decoder. Preserve component ordering, distance reductions, clipping and the adaptive pass. |
| Line recognition | `rec.onnx` consumes `x` and int64 `widths`, returning best token IDs, peak probabilities and valid lengths. Crop height is metadata-driven, with a default helper height of 32. | Inspect the exported graph before selecting a CNN/sequence implementation. The public client does not define its layers. |
| Script routing | `script.onnx` routes probability above 0.5 to `rec_cjk.onnx` and its separate vocabulary. | Preserve separate vocabularies and CJK character token positions. |
| Reading order/layout | `order.onnx` receives 16 geometry/text features and a boolean line mask. Output arity controls optional class, same-block and table heads. | Reproduce clustering, block ordering, geometry fallbacks and optional table-cell reruns. Sorting boxes top-to-bottom is insufficient. |
| Optional math | `mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`, `math_dec_step.onnx`, plus `math_meta.json`. | Separate autoregressive recognizer with self/cross KV caches, masks, EOS and confidence-based stopping. |
| Assembly | Line/token gaps split columns and cells; block classes drive markdown, lists and HTML tables; furniture is excluded by default. | Retain token positions through recognition and expose structured results, not only a concatenated string. |

### Preprocessing and decoding details that affect parity

[rec_preproc.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/rec_preproc.py)
expands each line by 0.2 line heights vertically and 0.3 horizontally. Bounds use
floor/ceil, with white outside the page. Aspect-preserving resizing switches
between OpenCV area and linear interpolation; widths are clamped to 8–4096 and
batch padding rounds upward to a multiple of 8. Width grouping also limits batch
area. Matching output dimensions alone does not establish matching pixels.

CTC uses blank ID 0, removes adjacent repeated IDs before dropping blank entries,
respects exported valid lengths and averages retained peak probabilities. Below
0.5 confidence, the line remains geometrically present with empty text. Token
centres are mapped back to page x-coordinates and drive later cell/column splits.
[vocab.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/vocab.py)
also inserts separators after LaTeX command tokens where concatenation would
change their meaning. Generic character concatenation loses this behavior.

[order_post.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/order_post.py)
uses ten geometry and six text features. Same-block clustering uses average
linkage, followed by within-block geometry and learned between-block ordering.
Preserve Unicode digit/case behavior, reduction precision and deterministic ties.
`table_heads` defaults to false; geometry-based tables remain part of the default
layout assembly. A local `mask.onnx` optionally changes line crops through
[line_mask.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/line_mask.py),
but that file is absent from the observed published snapshot.

[math.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/math.py)
uses BOS 1, EOS 2 and four reserved IDs. It introspects actual graph inputs because
exports prune unused inputs. Self KV starts with a zero-length time dimension;
position is int64. EOS contributes to mean token confidence. Low-confidence
sequences can stop after eight predictions; a reading replaces the CTC line only
when it contains a dollar marker and mean confidence is at least 0.9. Accepted
math replacements are reported by the caller with confidence 1.0. Preserve these
semantics rather than presenting that value as measured model certainty.

Final line boxes are rounded to one decimal and confidence to four decimals.
Boxes refer to the quarter-turn-corrected page, before deskew; a native API must
make the coordinate frame explicit.

[document.py](https://github.com/datalab-to/indra/blob/661fb47ac332dce07f1ebc0d7b37ffe2361bd85b/indra_ocr/document.py)
adds another layer: PDFium rendering/text extraction, text-layer validity checks,
PDF content-order checks, hybrid image-region OCR and right-to-left handling.
Image OCR parity would establish only the OCR-mode lane, not equivalence to the
publisher's PDF auto-mode benchmark.

## CrispEmbed integration plan

No native ONNX executor or ONNX Runtime integration was found in the current
source/CMake files. A native GGUF implementation needs a graph/tensor inventory
first. The public client references LineNet, OrderNet and MathRec but does not
ship their network definitions or training/export modules. Do not infer exact
layers, quantization or dimensions from the parameter-count claim.

Existing starting points are [core CPU operations](../../src/core/cpu_ops.h),
[GGUF loading](../../src/core/gguf_loader.h),
[PP-OCRv6 recognition](../../src/ppocrv6_ocr.cpp),
[Surya detection](../../src/surya_det.cpp), and the
[OCR orchestrator](../../src/ocr_orchestrator.h). These provide infrastructure;
they do not establish architectural compatibility. Existing CTC handling needs
an audit for retained peak positions, lengths, confidence and command tokens.

Once eligible weight use is established, the bounded implementation sequence is:

1. Pin the package's intended model revision; inventory all graph operators,
   input/output names, shapes, dtypes, constants and any original quantization.
   Establish whether all required operators have faithful native counterparts.
2. Implement converter and CPU runtime for detection and both CTC recognizers;
   preserve original precision for the first reference comparison. Proposed gate:
   `CRISPEMBED_INDRA_OCR=1`, parsed with `core_env::on()`.
3. Match image preparation, component decoding, orientation, script routing and
   token-position-aware splitting. Then add the order/layout model and assembly.
4. Add math as a separate opt-in capability after step-by-step cache/logit/token
   parity. Define PDF/text-layer support as a separate lane with explicit dependencies.
5. Wire C ABI, CLI/HTTP, Python/Rust/Dart, orchestrator enum and `map_engine`,
   dispatch, and CrispSorter according to [contributing](../contributing.md).
   Preserve ABI layouts; use a versioned structured-result interface for blocks,
   rotation, coordinate frame and markdown if existing interfaces cannot carry them.
6. Consider quantization only after decoded reference parity. Registry aliases,
   generated download pins, catalogs and redistribution follow verified artifacts
   and license eligibility; no placeholder URLs or permissive-license labels.

An optional ONNX Runtime adapter would add a dependency and a separate backend
contract. It would not establish a dependency-free native GGUF port or its speed.

## Independent validation gates

All model execution belongs on hosted runners. Freeze references from the pinned
original ONNX Runtime client before invoking the native implementation.

- Compare prepared detector inputs and line crops pixel-by-pixel, including edge
  boxes, rounding ties, padding, shrinking/enlarging and adaptive detection.
- Instrument the licensed ONNX graphs for intermediate outputs. Compare earliest
  divergence using absolute errors, magnitudes and relative errors. CTC exported
  argmax IDs alone cannot reveal pre-argmax score errors.
- Compare detector maps, connected components, boxes, recognizer IDs/lengths/peak
  probabilities, token x-coordinates, script/orientation decisions and line confidence.
- Compare layout features/logits, block membership, reading-order permutations,
  cell splitting, complete markdown/JSON and coordinate frames.
- For math compare encoder memory/masks, initialized cross KV, every decoding
  step's logits and self KV, selected tokens, EOS, stopping and final LaTeX.
- Use frozen transcribed pages covering columns, grids, rotations, tiny print,
  blank pages, Chinese, Indic text and math. Report actual text, line/block counts,
  CER/WER, confidence, boxes and cold/warm timings with matching thread counts.
- Keep OCR-only and PDF auto/hybrid results separate. Publisher throughput is
  not a native runtime speed result. Respect output/activation license terms
  before retaining or publishing reference fixtures.

## Weight license boundary

The code's Apache-2.0 declaration does not license the weights. The published
modified OpenRAIL-M text has revenue and funding conditions in Attachment A
2(a)/(b), and a separate restriction on entities providing competing products
or services in 2(c). The personal/research exceptions in 2(a)/(b) are not repeated
in 2(c). Section 8 also expressly applies its share-alike terms to outputs, while
sections 4, 5 and 7 impose downstream conditions and attribution requirements.
This audit records the text, not a determination that a particular deployment
qualifies. Clarify the intended use or obtain covering terms before weight-based
porting, evaluation, conversion, redistribution or catalog promotion.

Analysis is complete. Implementation and parity remain unclaimed pending that
clarification and the graph inventory.
