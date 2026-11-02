# Native Marker-style document pipeline: coverage and boundaries

This is a C++ document-processing integration with native TeleOCR inference.
It is not the original Marker Python package, and it does not run the original
Surya 2.5, fast-layout, reading-order or OCR-error models. The source and model
distinction is documented in [the license audit](marker-license.md).

The implementation follows Marker-style document blocks, deterministic
processing and output formats. PDFium supplies native PDF text, font and
geometry data. TeleOCR supplies model-based layout and recognition when the
selected mode requests them. This changes model behavior and therefore
requires its own quality evidence.

## Comparison with the audited original

| Area | Original Marker 2.1.0 | Native integration scope |
| --- | --- | --- |
| Production runtime | Python, PyTorch and shared model servers | Native C++/ggml; no Python subprocess or interpreter |
| PDF input | pdftext through pypdfium2/PDFium | Explicit dynamically loaded PDFium C API |
| Digital text | PDF text layer with model/heuristic quality decisions | PDF text layer with explicit conservative quality heuristics |
| Learned layout | Surya VLM or Datalab RF-DETR checkpoint | Existing TeleOCR layout task; separate model vocabulary and errors |
| Text-only layout | Upstream reference is forced to one Text block per page | Geometry/font-based native grouping; not expected to reproduce forced layout |
| Reading order | VLM/learned order or PDF character stream | Native geometric order for at most two columns, quarter-turn flow checks, and TeleOCR block order; independently tested |
| OCR | Surya 2.5 full-page or crop recognition | Native TeleOCR crop recognition |
| Tables | PDF text reconstruction and Surya fallback | Automatic aligned-text tables require at least three rows and numeric evidence; explicit Table regions can contain only text. No inferred spanning PDF cells. Parsed TeleOCR OTSL supports validated spans |
| Equations | Surya equation recognition and optional LLM correction | TeleOCR-recognized content; no claim of Surya math accuracy |
| Document structure | Broad block schema and processor set | Relative-font-size headings, indented lists, monospace code, tables, repeated marginalia suppression, adjacent caption/image grouping and paragraph continuation |
| Outputs | Markdown, HTML, nested JSON and other converters | Markdown, HTML, nested JSON and chunks with page/block geometry and `crispembed.marker/1` provenance |
| External LLM services | Optional multiple hosted services | Outside this integration |
| EPUB/Office/HTML inputs | Optional provider dependencies | Outside this PDF/image integration |
| Model weights | Separately restricted Datalab checkpoints | Separately audited TeleOCR checkpoint; no Datalab weights |

Native results must identify their actual extraction method and backend.
The formats share document concepts, but complete byte-for-byte equivalence
with all original Marker output, configuration options or metadata is not
promised. Unknown or unsupported content must be reported through the native
API's documented errors/warnings rather than silently represented as tested.

Quarter-turn reading order uses ordered word-span progression. At least 75%
of usable observations must agree on a 90/270-degree flow before sorting in
the corresponding unrotated coordinate system; output geometry stays in page
coordinates. Ambiguous flow preserves provider order with a warning. This is
a bounded heuristic, not general document-orientation or script-direction
recognition.

PDFium can collapse several literal spaces into one even within a text object.
For fixed-pitch text only, the provider can recover a bounded, near-integral
space count from character origins and emits a diagnostic warning. The rule
does not reconstruct arbitrary proportional-font indentation, mixed text
objects or generated whitespace.

Malformed UTF-8 from capped model output is represented with replacement
characters in valid UTF-8 renderings, with conditional byte-hex fields in JSON
to preserve the original bytes. Raw model strings and warning/cap information
remain inspectable. These contracts are independent of whether a particular
OCR result is correct.
Code and equation Markdown also replace NUL and other disallowed ASCII
control bytes, preserving tabs and line breaks, so a model byte cannot
truncate the public C string. The original strings remain escaped in JSON.

## Reference and validation protocol

`tests/marker_reference.py` generates owned, deterministic digital PDFs and
captures the actual pinned original Marker implementation. The input gold is
authored before either implementation is run. It covers Unicode text, literal
HTML/Markdown punctuation, headings, two-column content and table cell text.
Separate explicit schema probes exercise upstream renderer behavior without
depending on layout inference.

The original run uses `disable_ocr=True`, `force_layout_block=Text`, no LLM,
one PDF extraction worker, an existing local font, offline model settings,
network-call rejection and throwing predictor objects. Any predictor or
network call makes the run fail. Those controls are part of the receipt.

The comparisons are deliberately scoped:

- Compare recovered text against the owned input gold and original text-layer
  extraction, retaining character/word differences and reading-order results.
- Compare escaping and renderer contracts using explicit expected content.
  Preserve actual original Markdown, HTML and JSON for inspection.
- Test native inferred headings, lists, code and tables against the authored
  logical structure. Forced single-block original layout is not their gold.
- Run TeleOCR inference separately and retain real outputs, caps, warnings,
  timing and model identity. Text-only reference success cannot certify OCR.
- Keep upstream Python and reference dependencies out of production artifacts.

The resulting receipts describe exactly which cases and properties were
executed. Passing a small deterministic panel does not establish representative
document quality, full upstream feature parity or cross-platform inference
qualification. Native error handling, lifetime/ownership, resource limits and
the optional PDFium dependency require their own implementation checks.

## Executed original reference

The [original reference receipt](../tests/results/marker/upstream-model-free-attempt-2/receipt.json)
records seven PDF cases, nine pages and ten explicit renderer checks. All
authored content fragments survived, all ten renderer checks passed, and no
model, checkpoint, network or process-launch attempt occurred. Four negative
controls first demonstrated that each rejection guard was active. The
encrypted PDF is an additional native fixture, excluded from the upstream
conversion because the pinned provider has no password argument.

| Original PDF case | Provider CER / WER | Rendered HTML text CER / WER | Authored order |
| --- | --- | --- | --- |
| Simple digital text | 0 / 0 | 0 / 0 | Exact after whitespace collapse |
| Unicode and escaping | 0 / 0 | 0 / 0 | Exact after whitespace collapse |
| Headings, lists and code text | 0 / 0 | 0 / 0 | Exact after whitespace collapse |
| Two columns | 0 / 0 | 0 / 0 | Exact after whitespace collapse |
| Table cell text | 0 / 0 | 0 / 0 | Exact after whitespace collapse |
| Three pages | 0 / 0 | 0 / 0 | Exact after whitespace collapse |
| Rotated page | 0.565217 / 0.666667 | 0.565217 / 0.666667 | Original puts sentence before heading |

CER and WER use only whitespace collapse; punctuation, Unicode and case are
retained. The rotated case retains all content but differs in order. These
numbers measure the actual original deterministic path against authored input
gold, not native-vs-original inference parity. The forced-text path renders
headings mostly as bold text and does not reconstruct the table's semantics.

The [renderer probe](../tests/results/marker/upstream-model-free-attempt-2/renderer-probe/gold.json)
uses explicit owned blocks for a heading, literal text, list, code, table and
equation. Original HTML safely escapes literal tags, while its Markdown emits
the literal tag spelling; its Markdown table renderer replaces a literal cell
pipe with spaces. Native Markdown deliberately escapes HTML-active plain text
and uses generated table HTML to preserve cell content and spanning structure.
Those differences are intentional, not byte-parity failures hidden by a scorer.

The retained [initial failed receipt](../tests/results/marker/upstream-model-free-attempt-1/receipt.json)
records a test-harness namespace-package location check that failed before PDF
processing. The correction authenticates the imported converter module rather
than a namespace package's absent `__file__`. It changes no source, input gold,
model, rendering rule or acceptance threshold.

### Reproducing the bounded reference

Use a separate test environment. Install CPU Torch/TorchVision appropriate for
the host first, then `tests/requirements-marker-reference.txt`. The receipt
records the executed Python 3.12.14 / CPU Torch 2.8.0 environment and all package
versions. Do not install the upstream `full` or development extras.

```sh
git clone https://github.com/datalab-to/marker.git marker-reference-source
git -C marker-reference-source checkout 71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4
python tests/marker_reference.py capture \
  --upstream marker-reference-source \
  --fixtures tests/fixtures/marker \
  --output marker-reference-results \
  --font /path/to/an/existing/DejaVuSans.ttf
```

The output directory must be empty. Existing captures are not overwritten.
The executable harness is preserved next to the successful receipt, together
with upstream source hashes and the [runtime fingerprint](../tests/results/marker/upstream-model-free-attempt-2/environment.json).
The reference used PDFium 151.0.7891.0 from pypdfium2 5.10.1; a different
PDFium build is a separate environment, even if the Python binding version
or native wrapper is otherwise unchanged.

## Executed native comparison

The [final independent native receipt](../tests/results/marker/native-reference-final/receipt.json)
passes **86/86 checks across eight PDFs and ten pages**, using the same PDFium
binary as the original reference. It captures all four native output formats
and metadata, checks page geometry and unique IDs, and evaluates headings,
list items, exact code bytes/indentation, table cells, column order, rotated
order, Unicode/escaping, marginalia and the encrypted input.

All eight cases have zero CER/WER against their explicit rendered-content
gold. The gold itself was authored before the runs and never changed. The
rendered profile removes only the running header/footer declared in the
three-page fixture and represents the two declared bullet markers as HTML
list structure. Literal-input metrics remain in the same receipt; seven of
eight cases have zero literal-body CER/WER. The structure case differs only
by the two bullet-marker text nodes, while the native Markdown retains the
list bullets and the separate list-item structure checks pass.

| Native case | Rendered gold CER / WER | Literal body gold CER / WER | Raw original HTML text CER / WER |
| --- | --- | --- | --- |
| Simple digital text | 0 / 0 | 0 / 0 | 0 / 0 |
| Unicode and escaping | 0 / 0 | 0 / 0 | 0 / 0 |
| Headings, lists and code | 0 / 0 | 0.021622 / 0.062500 | 0.021622 / 0.062500 |
| Two columns | 0 / 0 | 0 / 0 | 0 / 0 |
| Aligned table | 0 / 0 | 0 / 0 | 0 / 0 |
| Three pages | 0 / 0 | 0 / 0 | 0.323907 / 0.333333 |
| Rotated page | 0 / 0 | 0 / 0 | 0.565217 / 0.666667 |
| Encrypted input | 0 / 0 | 0 / 0 | Not run by original provider |

Raw comparisons with original Marker are intentionally not all zero. The
native list renderer represents bullets structurally, native marginalia
processing removes the repeated running text, and native rotation handling
restores the authored order where the original forced-text path reverses it.
The table stores the actual original-vs-native differences, without treating
these pipeline changes as original-model parity.

The [initial native receipt](../tests/results/marker/native-reference-attempt-1/receipt.json)
is retained with its real failures: two-column prose was classified as a
table, rotated word order was damaged, and four code-indent spaces collapsed
to one. The fixes were exercised through a rebuilt public native library.
Two other initial checks incorrectly compared literal PDF bullets to HTML
`li` text; the corrected harness makes the list representation explicit and
retains both literal and rendered metrics. No fixture, source text or runtime
failure was removed from the evidence.

The final code block retains exactly four spaces before `print`, with a
warning recording the three spaces reconstructed from fixed-pitch glyph
advances. The two-column and rotated cases also retain their heuristic-use
warnings. Original and native output files, initial rejections, executed
harness snapshots and binary/source hashes accompany the receipts.
The [earlier corrected comparison](../tests/results/marker/native-reference-attempt-2/receipt.json)
is retained separately. The final run repeats all 86 checks after source
formatting, the software-version metadata correction and the targeted
Markdown control-byte fix; its recorded library SHA-256 begins `f1177617`.

Fixture regeneration was independently repeated: all
[20 file hashes matched](../tests/results/marker/fixture-reproducibility.json),
including the encrypted PDF, ten rasterized pages and unchanged input gold.
This is a small, owned deterministic panel. It does not establish learned
OCR accuracy, original Surya equivalence or representative-document quality;
the native TeleOCR model runs and its
[existing numerical/precision limits](teleocr-validation.md) remain separate.

To repeat the independent native check after a build:

```sh
python tests/marker_compare.py \
  --library /path/to/libcrispembed.so \
  --pdfium /path/to/the/reference/libpdfium.so \
  --reference tests/results/marker/upstream-model-free-attempt-2 \
  --fixtures tests/fixtures/marker \
  --output marker-native-comparison
```

The harness verifies the input-gold hash, original control result and exact
PDFium binary before conversion. Run it in the test environment with the
native package's NumPy requirement available; it neither imports the upstream
Marker package nor initializes a neural model.

## Executed native model panel

The [independent model-output review](../tests/results/marker/native-review.json)
passes 156 saved-output/provenance checks over **seven mode/build/precision
conversions of three distinct owned inputs**. The five panel receipts retain
their declared model hashes, input gold, source fingerprints, outputs, caps,
warnings and single-run timings. All seven conversions have zero CER/WER after
whitespace collapse, preserving case, punctuation and Unicode.

| Build and precision | Input/mode | Layout tokens | Recognition tokens | Seconds | CER / WER |
| --- | --- | ---: | ---: | ---: | --- |
| Initial F16 | Synthetic text image / auto | 73 | 19 | 126.277 | 0 / 0 |
| Initial F16 | Digital PDF / auto | 73 | 0 | 117.855 | 0 / 0 |
| Initial F16 | Table image / auto | 76 | 76 | 165.626 | 0 / 0 |
| Final F16 | Synthetic text image / auto | 73 | 19 | 129.878 | 0 / 0 |
| Final protected Q8_0 | Synthetic text image / auto | 73 | 19 | 129.758 | 0 / 0 |
| Final protected Q4_K | Synthetic text image / auto | 73 | 19 | 124.335 | 0 / 0 |
| Final F16 | Digital PDF / forced OCR | 73 | 17 | 135.745 | 0 / 0 |

The [initial full F16 panel](../tests/results/marker/native-ocr-f16/summary.json)
uses library SHA-256 beginning `6453530a`. The final
[F16](../tests/results/marker/native-final-f16/summary.json),
[Q8_0](../tests/results/marker/native-final-q8/summary.json),
[Q4_K](../tests/results/marker/native-final-q4/summary.json) and
[forced-PDF](../tests/results/marker/native-final-force-pdf/summary.json)
panels use the final library beginning `f1177617`. The final F16 synthetic
repeat preserves the initial text, raw layout and boxes, while correcting the
software-version metadata from `unknown` to `0.17.14`. Historical receipts
and their executed runners remain unchanged.

Digital auto mode exercises neural layout plus embedded PDF text; its zero
recognition-token count makes that boundary explicit. Forced OCR separately
produces exact text with 17 recognition tokens and every block sourced from
`teleocr-native`, bypassing the text layer. The table review independently
checks four rows, three columns, cell order and unit spans, beyond the original
runner's flattened-cell comparison. All twelve cells are `td`; header
semantics and merged-cell recognition are not qualified by this case.

On the one shared synthetic image, both quantized files match final F16's
visible text, raw recognition strings, raw layout and boxes. The reported
aggregate block confidences differ by up to 0.000053 for Q8_0 and 0.000266 for
Q4_K. The synthetic lines are all labeled `SectionHeader`, and the table title
is labeled `Caption`; these content checks do not score learned label quality
or box IoU. Finite in-page geometry and confidence ranges are integrity checks.

All runs use CPU execution, two threads and unchanged 1024-token layout and
recognition caps. No cap is reported. This document harness does not export
token IDs or terminal tokens, so it does not establish EOS or token-ID parity.
The timings are single-run observations, not controlled performance claims.
Q8_0 and Q4_K were tested only on the synthetic image. This small English panel
does not establish representative, multilingual or cross-platform OCR quality.
The inherited TeleOCR strict numerical gates remain failed, and no Surya model
was executed. The model-free
[reviewer](../tests/results/marker/native-review.executed.py) rederives the
scores and table structure from saved outputs without loading any model.
