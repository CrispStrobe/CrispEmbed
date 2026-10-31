# Native Marker-style pipeline: executed validation

This record distinguishes native document processing, public API behavior,
learned OCR accuracy and numerical model qualification. The source implements
a native Marker-style pipeline using TeleOCR, not the original Surya models.
The [machine-readable summary](../tests/results/marker/summary.json) links the
executed receipts, source/binary hashes, initial failures and final results.
The [coverage document](marker-feature-coverage.md) explains the differences
from the audited original.

## Source and environment

| Component | Executed source or identity |
| --- | --- |
| Marker reference | 2.1.0, `71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4` |
| TeleOCR checkpoint | `XingChen-AGI/TeleOCR@b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d` |
| Apache Rust reference | `rzafiamy/teleocr-rs@957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe` |
| ggml | `a06fc0bb46b73cc10011fc90ae4054aee8438aeb` |
| Native backend | Linux x86_64 CPU; portable Release build, no CUDA/Metal/Vulkan |
| Paired PDFium | 151.0.7891.0, from pypdfium2 5.10.1 |
| PDFium SHA-256 | `cb59522bf0c48a6408fd2adc8c87120a1418bacbda193f898e8cdccda7d5ca27` |
| Final shared library SHA-256 | `f117761723bc909b4c6194e096e6bbfe18ba6bf5576d24108b6af0b780f5a5e9` |

The original Marker environment is an isolated validation overlay. No Surya
weights were downloaded or executed. The production C++ library has no Marker
Python dependency. Neural tests use the existing, previously converted and
hashed TeleOCR GGUF files. They run sequentially with two CPU threads and
fixed 1024-token recognition/layout caps; no cap, prompt or gold was adjusted
in response to output.

## Native processing and public interfaces

| Executed check | Result | What it establishes |
| --- | ---: | --- |
| Native document tests | 86/86 | Structure processing, OTSL/table spans, rendering/escaping, malformed data, control-byte safety |
| Native backend tests with PDFium | 39/39 | Crop/polygon/rotation/alpha behavior, PNG provenance, actual PDF text/fonts/links/geometry, limits and shared PDFium lifetime |
| C ABI tests with PDFium | 38/38 | Config layout/gates, actual conversion/formats, errors, stale-output clearing and recovery |
| C++ output writer | 38/38 | Asset collisions, byte-identical reuse, path aliases/symlinks and write contracts |
| Python client regressions | 17/17 | Actual PNG transparency roundtrips/native input capture, output preflight and preservation |
| Python/CLI/HTTP integration | 112/112 | Eight owned PDFs, four formats, native metadata, semantic content, page/password handling, errors and HTTP path confinement |
| Compiled Rust consumer | 13/13 | Actual native PDF calls, all formats, metadata, bad-input and recovery checks |
| Compiled Dart consumer | 14/14 | Actual native PDF calls, all formats, copied results, lifecycle and bad-input checks |
| Actual Python package archives | 26/26 | Wheel, sdist and isolated-sdist wheel retain exact notices/native bytes; missing/empty notices reject |
| CMake notice installation | 8/8 file comparisons | MIT/Apache licenses and both notices installed in shared and static-only configurations |

These are separate panels with overlapping assertions; their counts must not
be added together as unique independent test cases. Rust and Dart checks use
native digital-PDF conversion. They establish binding behavior, not OCR
quality or mobile/platform qualification. Python transparency tests capture
the real bytes passed to the native entry point; they do not rerun a neural
model for every pixel-format fixture.

The final document implementation also passed all 86 tests under both
AddressSanitizer and UndefinedBehaviorSanitizer, without diagnostics. The
earlier 82-check source-specific receipt is preserved alongside the
[final sanitizer receipt](../tests/results/marker/sanitizers-final/receipt.json).
LeakSanitizer was disabled
because the repository's one-shot test entry point uses `clean_exit/_Exit`.
No leak-freedom claim follows from these runs.

The model-free C++ tests are wired into the existing Unix build workflow.
That workflow edit is provided as source; no new hosted CI run or Windows,
Apple, Android, iOS or WASM execution is represented by the Linux receipts.

## Owned fixtures and the actual original Marker reference

The eight PDFs cover simple digital text, Unicode and escaping, headings/
lists/code, two columns, a 4×3 table, a three-page document, rotation and an
encrypted document. Together they contain ten pages. Gold content and
structure were authored before the comparison. Independent regeneration
matched all twenty PDF/PNG/font-notice/gold file hashes, and all ten rasterized
pages were visually checked.

The pinned original Marker implementation processed seven PDFs/nine pages
with OCR disabled and one forced Text block per page. Throwing predictors,
offline settings and network/process/checkpoint rejection guards excluded
model calls; all four negative guard controls and ten renderer probes passed.
The encrypted input is excluded because that upstream provider has no
password argument. This is an actual original processing run, not a simulated
reference and not learned-model parity.

The independent native comparison passes 86 checks across eight PDFs/ten
pages using the same PDFium binary. All eight match the declared rendered
content with zero character/word error. Seven match the literal body text;
the structure fixture represents its two bullet markers as HTML list
structure. Both metrics remain visible. Code indentation is checked as exact
bytes, separately from whitespace-normalized text metrics.

Original-versus-native raw HTML text matches four of the seven shared cases.
The other three retain explicit differences: structural list bullets,
suppressed repeated page marginalia, and native restoration of the authored
rotated-page order. The original forced-text path places the rotated sentence
before its heading. Native structure inference is scored against authored
structure, not against the original's forced single-block layout.

See [the complete comparison and output files](marker-feature-coverage.md#executed-native-comparison)
for case-by-case metrics and links to both initial and corrected runs.

## Real neural execution and its limits

The retained initial F16 full panel runs three complete document conversions:
an owned three-line image, a digital PDF using neural layout and embedded
text, and a rasterized table. All three have exact whitespace-normalized
content, finite confidence values and no reported layout/region cap. The
table has the correct four rows and three cells per row. Digital auto mode
generates layout tokens and uses zero recognition tokens; this validates its
hybrid path, not PDF OCR recognition.

Learned layout labels are not gold-qualified by text success. On the short
image, TeleOCR labels all three lines as section headings; on the table page,
it labels the title as a caption. These outcomes remain in the saved JSON.
No representative layout AP, reading-order score, OCR corpus accuracy or
probability calibration is claimed.

Final-source F16, Q8_0 and Q4_K synthetic panels and the separately declared
force-OCR PDF panel all pass their functional and exact-content checks. Together
with the retained initial F16 full panel, these are seven mode/build/precision
trials over three distinct owned inputs. They are listed in the aggregate
receipt. Each capture stores
the declared inputs/gold/caps before model loading, authenticates the actual
GGUF hash and saves Markdown/HTML/JSON/chunks, metadata, runtime and source
identity. The final force-OCR panel requires recognition tokens and verifies
that no leaf block uses embedded PDF text. Source changes between the initial
full F16 panel and final build are retained in review snapshots.

Final synthetic times are 129.878 seconds (F16), 129.758 seconds (Q8_0) and
124.335 seconds (Q4_K). The force-OCR PDF takes 135.745 seconds and generates
73 layout plus 17 recognition tokens. These exclude model loading and are
diagnostic observations. Final synthetic raw text/layout and block boxes match
across all three precisions; confidence differences remain visible. See
[PERFORMANCE](../PERFORMANCE.md) for all case timings and process RSS.
An independent [saved-output review](../tests/results/marker/native-review.json)
passes 156 checks across all seven trials, including exact table topology,
output/metadata consistency and all eighteen final source hashes. Its
[reproducible reviewer](../tests/results/marker/native-review.executed.py)
does not run models or rewrite the raw captures.

This document-level runner records token counts and cap flags, but does not
expose complete generated token IDs or independently prove EOS parity.
The earlier [TeleOCR validation](teleocr-validation.md) provides separate
original/native sequence and intermediate evidence. Its unchanged strict
numerical gates remain failed for every tested precision. The new pipeline
tests do not waive those failures, and a successful one-image Q8/Q4 check
does not establish full quantized-document qualification.

## Initial failures and final review

Retained initial captures include real two-column prose misclassification as
a table, rotated word order damage and collapsed code indentation. The fixes
require numeric evidence for automatic table discovery, bounded quarter-turn
flow handling and fixed-pitch glyph-advance spacing reconstruction. Each
heuristic is described in coverage documentation and emits diagnostics where
appropriate. The original fixtures/gold remain unchanged.

The first comparison harness also treated literal list bullets as if they
had to remain HTML text nodes. The corrected scorer retains both literal and
rendered-content profiles and separately verifies list structure. This test
correction is distinct from the runtime fixes. Earlier environment, build and
reference-harness setup failures remain in their logs rather than being
reported as successful executions.

Final review fixed transparent Pillow palette/RGB-tRNS input, output asset
collisions, C-string truncation from Markdown control bytes, shared-library
software-version provenance and static/Python package notice inclusion.
Snapshots authenticate the earlier source. The Python `convert(path)` method
remains byte-identical across its client fix; the new `convert_image` and
`save` behavior have targeted regressions. Neural arithmetic, checkpoint
bytes, frozen gold and token budgets are unchanged.

## Reproducing the panels

Build the CLI/shared library and the four test targets. Supply a native
PDFium library explicitly:

```sh
cmake --build build --target test-marker-document test-marker-backend \
  test-marker-api test-marker-save --parallel 2
./build/test-marker-document
./build/test-marker-backend /path/to/libpdfium.so
./build/test-marker-api /path/to/libpdfium.so tests/fixtures/marker/simpledigital.pdf
./build/test-marker-save

python tests/marker_integration.py \
  --library build/libcrispembed.so --pdfium /path/to/libpdfium.so \
  --cli build/crispembed-marker --server build/crispembed-server \
  --out marker-public-results

python tests/marker_native.py \
  --library build/libcrispembed.so --pdfium /path/to/libpdfium.so \
  --model teleocr-f16.gguf \
  --expected-model-sha256 6c52a2800e313964717cf4e5d06e0392c2b5936d8adeac423d563b8a026a2780 \
  --panel synthetic --threads 2 --out marker-native-results
```

`--panel full` uses the three declared auto-mode cases; `--panel force-pdf`
uses the owned digital PDF with embedded text bypassed. Outputs must use a
new directory so a previous run is not overwritten. The model hashes in the
receipts identify this exact converted checkpoint/recipe; independently
converted files with different bytes require their own recorded identity.

GPU and other operating-system runs, representative multilingual/layout
quality, larger documents/resource boundaries, original Surya inference,
external LLM correction and upstream Office/EPUB providers remain outside
this validation. Both feature gates remain default-off. No model alias,
publication, runtime release or downstream application version is promoted.
