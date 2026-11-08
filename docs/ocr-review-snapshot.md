# OCR review snapshot

The additive OCR review accessor makes the existing native orchestrator's page
structure and diagnostics available to document workspaces. It does not change
recognition, model selection, preprocessing defaults, acceptance thresholds or
the layout of any public C struct.

## Clients

After running a pipeline, call the following accessor on the same context:

    const char * crispembed_ocr_pipeline_review_json(void * ctx, int * out_len);

The result is a UTF-8 JSON object. The native context owns the buffer until its
next run, review call or free. Copy exactly the returned byte length; it excludes
the terminating NUL. A null context, no preceding run, invalid run or
serialization/allocation failure returns NULL and sets the length to zero.
The accessor performs no inference and never writes an image or model file.

The safe Rust pipeline exposes review_json(), which returns an owned
Option<String>. Use matching native and Rust sources; linking new bindings
against a library without the symbol is unsupported. Python
CrispOcrOrchestrator.review_snapshot() and Dart
CrispOcrOrchestrator.reviewSnapshot() return an owned dictionary/map and detect
older libraries by symbol availability. They return None/null if the accessor
is unavailable. They never load a second native library for an existing context.

The CLI accepts --ocr-pipeline PAGE --output-format review-json with the usual
engine/model arguments. This mode prints exactly one JSON object. The HTTP
POST /ocr/pipeline endpoint accepts response_format: "review-json" alongside the
image request. Inference and serialization share the same context lock. Ordinary
CLI and HTTP response shapes remain unchanged.

## Version 1 contract

The root has schema "crispembed.ocr.review" and schema_version 1. Consumers
should reject unsupported major schema versions and ignore unknown additive
fields. Result-local region IDs are stable within a snapshot. Applications
should assign their own durable page and region IDs when persisting edits.

| Field | Meaning |
|---|---|
| status | input_unreadable, no_text, recognized, or review_required |
| accepted | Whether the selected stage passed its configured acceptance gate |
| engine / selected_stage | Selected engine and one-based stage metric index; null/0 when no result was selected |
| source_type | auto, screenshot, scanned_document, or photo |
| text / recognition_text | Joined text after the optional punctuation restorer, and the orchestrator text before that restorer |
| text_postprocessed | Whether those two strings differ |
| markdown | Existing structured Markdown rendering, with no new inference |
| reading_order | Complete permutation of region indices, with invalid/duplicate indices discarded and unmentioned regions appended |
| regions | Text, xywh box, confidence source, sequence scores and orientation per region |
| stages | Engine, elapsed milliseconds, requested cleanup, acceptance, UTF-8 text byte count and engine confidence |
| detected_language / language_confidence | Language identifier and score, or null when unavailable |
| capabilities | Configured layout/table/formula backends and text fallback; configuration does not prove successful inference |
| layout / tables / formulas | Existing structured regions, table HTML and formula LaTeX |
| routing | Layout dispatch decisions, text-to-layout indices and the text-suppression mask |
| warnings | Explicit missing-result, coordinate and text-provenance limitations |

### Geometry

All boxes use top-left pixel coordinates in xywh order. The geometry object
records three dimensions: original input, effective input after optional
super-resolution, and the image actually used for recognition. Text regions
refer to the recognition image. Layout refers to the effective input. Internal
cleanup can rotate/crop the recognition image without remapping the layout.

coordinates_valid_for_input is true only when the original input has known
positive dimensions, the recognition dimensions match, and no internal cleanup
or super-resolution output was applied. It is deliberately conservative:
dimensions alone do not prove that deskewed or cropped coordinates map to the
original.

A review workspace should explicitly prepare its canonical page image, retain
that image with the page, disable internal cleanup and super-resolution, then
recognize that exact image. Draw overlays only after checking geometry validity.
The snapshot does not expose temporary preprocessing files or invent an inverse
transform. The original API's geometry remains unchanged.

### Confidence

No score is a calibrated probability that a document is correct. The root
confidence_calibrated field is false. Each region includes:

- confidence, a nullable recognition score when available, or a detector score
  only for a known detection stage;
- confidence_source, one of recognizer, detector, or unavailable;
- engine_score, the legacy engine value retained for diagnostics;
- recognition_confidence and sequence_confidences, with absent measurements
  represented by null/an empty array;
- sequence_unit, recognizer_character for the Tesseract paths and engine_token
  otherwise;
- character_alignment_verified, false: the legacy pipeline does not expose
  offsets that certify alignment between its sequence scores and Unicode text.

Zero is retained when a sequence establishes that recognition scores exist.
Some whole-page engines use a legacy engine score of 1 when no measurement is
available; the snapshot does not promote that placeholder to full certainty.
Invalid/out-of-range scores and nonfinite numbers serialize as null, not invalid
JSON. UI clients must not index Unicode characters with model-token positions.
The PP-OCRv6 orchestrator currently copies its detector score into the legacy
recognition field; the snapshot identifies that score as detector confidence.

Malformed UTF-8 decoder bytes are replacement-decoded for JSON transport. An
additional sibling field such as text_bytes_hex preserves the exact original
bytes whenever display text needed repair. Valid multilingual text and embedded
NUL/control characters round-trip unchanged through JSON escaping.

The stage metric's text_bytes is a UTF-8 byte count, despite the historical
native field name text_chars. cleanup_requested reports that existing metric's
requested recipe; the selected result separately records whether a cleanup
output was actually used.

### Missing text and structured content

The existing engine interface returns an empty region vector for both a blank
page and several model/load/inference failures. no_text explicitly preserves
that uncertainty and adds a warning. It is not a successful blank-page proof.
Applications should retain it for review or retry and surface their engine
initialization diagnostics. recognized means the configured acceptance gate
passed, not that human review or a quality benchmark passed.

Table HTML and Markdown are document data. Render them through the application's
sanitizer or show them as text; do not execute source content in a webview.
The text_postprocessed field covers the optional wrapper punctuation restorer.
Other existing engine/orchestrator postprocessing is already reflected in its
region and joined outputs; this contract does not claim raw decoder-token
provenance.

## Validation and rollout

test-ocr-review is a model-free C++ target. It exercises valid JSON, multilingual
and control/NUL round trips, locale independence, finite numeric output, nullable
and zero confidence, preservation of all reading-order regions, selected-stage
metadata and coordinate warnings. The existing test-ocr-orchestrator C API test
also covers accessor availability, null inputs, actual image dimensions, repeat
calls and invalid-run stale-result prevention. Both targets run in the hosted
native build's model-free tier.

These tests validate the result contract. They do not qualify an OCR model,
precision, backend or corpus. Consumers pinned to an older immutable release
continue to use their existing results; adopting this accessor requires a
matching native build and client binding. Publication and application release
qualification remain separate from this feature branch.
