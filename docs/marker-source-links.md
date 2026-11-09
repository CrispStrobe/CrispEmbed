# Source-linked native document processing

This feature adds immutable block-text snapshots, byte-accurate source ranges,
explicit paragraph continuation targets, assembled retrieval chunks, conservative
multilingual whitespace handling, and inspectable footnote/page-label links.
It builds on the existing native Marker document representation. These algorithms
and fixtures are independently authored; they do not use the MinerU-derived
paragraph merger excluded by the [PDF Craft source audit](pdf-craft-adoption.md).

The original page-local blocks, raw recognition output, geometry, and existing
renderers remain available. An assembled retrieval chunk can cross a page or a
footnote boundary while still identifying every contributing source block.

## Independent opt-in gates

All gates use `core_env::on()`. Unset, empty, `0`, `false`, `off`, and `no` disable
them; the word spellings are case insensitive. Fresh conversions with all gates
disabled retain the previous JSON and chunk output byte for byte.

| Gate | Enabled processing |
| --- | --- |
| `CRISPEMBED_MARKER_SOURCE_LINKS=1` | Capture immutable source records; identify explicit continuation targets; expose source-linked assembled chunks. |
| `CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE=1` | Capture source records and normalize eligible prose whitespace with a byte-range map for every change. |
| `CRISPEMBED_MARKER_REFERENCES=1` | Capture source records; assign footnote identities; match explicit bracket references; report physical indices and provider/observed page labels. |

The normalization and reference gates imply source capture, so neither feature
can produce changed or linked text without its underlying source evidence.
Continuation inference requires `SOURCE_LINKS`. The document pipeline's existing
`CRISPEMBED_MARKER` gate and any required model-license acknowledgement remain
independent prerequisites. These processors load no additional model or external
dependency.

For example, add these environment variables to an existing Marker conversion:

```sh
CRISPEMBED_MARKER=1 \
CRISPEMBED_MARKER_SOURCE_LINKS=1 \
CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE=1 \
CRISPEMBED_MARKER_REFERENCES=1 \
./build/crispembed-marker --help
```

The command shows the installed CLI's options; use its ordinary PDF/text or
model-backed conversion configuration. No new model download follows from these
three gates.

## What the source text means

`source_records` is an additive JSON/chunk catalog. Each record contains a stable
ID within the extraction, exact source text, physical page index, original text
provider, and the complete source block's bounding box and polygon.

The snapshot stage is **after the existing baseline structure processors and
before the new optional whitespace normalization**. A snapshot therefore records
the text of a split list item or an already grouped PDF paragraph. It is not a
claim that the snapshot equals the neural decoder's raw string. The independent
`raw` field continues to preserve recognition/provider evidence.

The metadata makes this stage explicit:

```json
{
  "source_links": "block-bytes-v1",
  "source_text_stage": "baseline-structure-before-optional-normalization",
  "source_range_units": "utf8-bytes"
}
```

Source text is never overwritten during repeated processing. Unchanged blocks
reuse their existing source IDs and ranges. If a caller edits a block's text,
the next enabled processing pass creates a new snapshot and retains the old
record. Older unreferenced records can consequently remain in the catalog.
Source records must refer to physical pages present in the document.

### Range contract

Every `source_ranges` entry has:

- `source_id`: the immutable source record.
- `source_start`, `source_end`: half-open UTF-8 **byte** offsets into that record.
- `text_start`, `text_end`: half-open byte offsets into the current block's text,
  or into the assembled chunk's text when the range belongs to a chunk.
- `transform`: the operation connecting those ranges.

Persisted block ranges cover the output text in order. `copy` means the two byte
slices are equal. `whitespace` maps an ASCII whitespace run to one space or an
empty output range. Empty output ranges retain evidence for removed whitespace.
An identity range is a `copy` of the entire original text.

For the original text `日\n本`, UTF-8 offsets are:

| Source bytes | Output bytes after normalization | Transform |
| --- | --- | --- |
| `[0, 3)` — `日` | `[0, 3)` — `日` | `copy` |
| `[3, 4)` — newline | `[3, 3)` — empty | `whitespace` |
| `[4, 7)` — `本` | `[3, 6)` — `本` | `copy` |

Geometry always belongs to the complete original block. A text slice does not
invent a smaller rectangle, and a chunk spanning pages does not invent a single
bounding box covering those pages. Invalid UTF-8 remains recoverable through the
existing conditional byte-hex fields; whitespace normalization skips such text.

## Assembled retrieval chunks

The `chunks` format retains its existing `blocks`, `page_info`, and `metadata`
fields. When source records exist, it also contains `source_records` and
`assembled_chunks`. Each assembled chunk carries its text, contributing
`block_ids`, physical `pages`, byte ranges, footnote references, and an
`assembly_status`.

Explicit `continuation_target` IDs link eligible text blocks. Candidate status
and the candidate block ID remain inspectable in `document_links`, including
when the relationship is unresolved. These explicit links drive assembled
chunks independently of the older `has_continuation` flag used by the legacy
Markdown renderer.

Continuation rules require consecutive physical pages or a geometric column
transition, compatible prose, and either a hyphenated word or sufficient line
geometry. A lowercase Latin start or a Han/Kana boundary supplies the language
signal. Sentence-ending punctuation, malformed UTF-8, and token-capped blocks
prevent resolution. Missing physical pages are not silently bridged. Cases
without sufficient evidence retain an unresolved candidate.

Footnotes and running marginalia do not interrupt a confirmed body-text chain.
Footnotes still receive their own chunks. When joining a hyphenated word, the
removed hyphen is represented by a zero-output `dehyphenation` range. An inserted
inter-block space has an `inserted` range with `source_id: null`; it explicitly
has no source bytes or source geometry. These two transforms exist only in
assembled output, not in persisted block ranges.

Other content stays in separate typed chunks. A table's plain `text` can still
be model notation; its original structured block/HTML remains the source for
table semantics. This feature does not claim a new table-to-prose conversion.

Assembly preserves page-local `text` and `raw`. Markdown and HTML retain their
existing page/block rendering behavior, apart from text changes explicitly
enabled by multilingual normalization. Cyclic, backward, missing, or invalid
continuation targets stop assembly with `unresolved-continuation`.

## Conservative multilingual whitespace

The normalizer collapses runs of ASCII space, tab, CR, and LF in eligible prose.
It removes a line-break run between Han/Kana characters. Plain spaces between
those characters remain spaces; it does not assume every visible space is an
OCR error.

The following are preserved:

- Join controls, variation selectors, combining sequences, and zero-width
  characters. No Unicode normalization form is applied.
- Nonbreaking and narrow nonbreaking spaces, ideographic spaces, and other
  non-ASCII whitespace.
- Meaningful word spacing in Hangul, Thai, and other scripts outside the narrow
  Han/Kana line-break rule.
- Code, equations, chemical notation, tables, and image subtrees.
- Prose containing backticks, dollar signs, `\(`, or `\[`, which conservatively
  bypasses normalization to preserve possible inline code or TeX.

All modifications retain source ranges. If a range budget is exhausted, the
affected block stays unchanged and receives a diagnostic. An already transformed
source map requiring an additional unsupported composition also leaves the block
unchanged. Reprocessing ordinary normalized output is idempotent.

## Footnotes and page labels

Every explicitly classified `Footnote` receives a `footnote_id`. Definitions
support bounded numeric labels and `*`, `**`, `***`, `†`, and `‡`, with ordinary
leading markers or bracketed `[...]` / `[^...]` forms. Unlabeled footnotes retain
an identity and an explicit `unlabeled` status.

References use explicit bracket syntax. A unique same-page definition resolves
the reference. Duplicate same-page labels produce `ambiguous`, with bounded
candidate IDs and no target. A definition found only on another page remains
an `unresolved` candidate; no cross-page identity is fabricated. `resolved`
describes this syntactic match, not proof of the author's intended semantics.
Bare numbers and isolated superscripts are not treated as references. Code and
structured subtrees, escaped brackets, and Markdown links are excluded.

Reference offsets are bytes in the current block/chunk text. Source ranges let
callers map them back to immutable text. The original text is not rewritten into
HTML anchors or new footnote markers.

Page output distinguishes:

- `physical_page_index`: the existing zero-based PDF/page index.
- `printed_page_label`: the selected label or `null` when unavailable/ambiguous.
- `page_label_source`, `page_label_status`, and `page_label_candidates`.

A PDFium page label is identified as `pdfium-page-label`; it is metadata and is
not proof of visible printed text. An explicitly empty provider label is retained
as `""` with `resolved` status, distinct from missing metadata. Provider labels
take precedence, while independently observed footer candidates remain visible.
Without a provider label, one distinct standalone decimal or canonical Roman
footer label resolves; multiple distinct candidates are ambiguous. No label is
invented from the physical index.

## Bounds and validation

| Resource | Bound |
| --- | ---: |
| Immutable source records | 100,000 per document |
| Source text | 4 MiB per record; 64 MiB total |
| Source ranges | 16,384 per block; 262,144 per document |
| Assembled chunk | 4 MiB text; 16,384 ranges before another join |
| Footnote references | 4,096 per block; 100,000 per document |
| Reference candidate IDs | 64 per reference |
| Tree depth | Existing 24-level limit |

Source-catalog limits fail explicitly. A normalization range limit leaves that
block unchanged. Chunk limits stop the current assembly with `assembly-limit`,
allowing the next block to become a separate chunk. Reference limits retain all
text and report that subsequent references were not indexed.

`tests/test-marker-document.cpp` contains owned, model-free regression cases for
disabled-path compatibility, immutable source reuse, byte offsets after multibyte
characters, removal and insertion provenance, cross-page/footnote continuation,
missing-page and capped-output controls, Unicode preservation, local/ambiguous/
unresolved footnotes, physical/provider/observed page labels, and explicit empty
provider labels. The owner-authorized Linux CPU execution passes all 185 document
checks at final native source `891ea3d3632c6a3c3155e25b34705da6a02ec807`; the actual
Rust/Dart clients also pass exact source-map and four-render reopen checks.
See [the execution evidence](../tests/results/marker/pdf-craft-local-2026-11-09/README.md).
The existing hosted Marker document target retains the same controls.

These deterministic fixtures establish the stated contracts, not representative
OCR quality. Default promotion, broader multilingual joins, cross-page footnote
resolution, and inference-dependent continuation changes require a separately
reviewed document corpus and retained negative cases.
