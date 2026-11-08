# Reviewed OCR exports

The native renderer creates text, hOCR, ALTO and searchable PDF from reviewed
page regions. It does not run recognition or reconstruct original typography.
Image and text coordinates must refer to the same canonical page image.

## Checked interfaces

`crispembed_ocr_render_capabilities()` returns a bit mask: searchable PDF `1`,
Unicode text `2`, decoded raster images `4`, archival conformance `8`. This
implementation returns `7`; PDF/A is unsupported. Requesting archival mode
produces an error and no file, rather than an unvalidated compliance label.

The C lifecycle remains `create`, `begin`, `add_page`, `end`, `output`, `free`.
Call `ocr_render_error()` before publishing the result: a non-null message means
the operation failed and output was cleared. Input C structs are unchanged.

For each page, callers can set physical dimensions with
`ocr_render_set_page_size_points(renderer, width, height)` after `begin` and
before `add_page`. The setting applies to the next page only. One inch is
72 points; images and text scale together from the supplied pixel coordinates.
The PDF 1.4 writer accepts physical dimensions from 0.01 to 14400 points.
Without a physical size, the old one-pixel-per-point convention is retained.
Applications should carry original PDF dimensions or explicitly choose a scan
resolution, rather than infer physical size from pixels alone.

Rust exposes `ocr_render_pages_checked_with_sizes(pages, format, pdfa, sizes)`
returning `Result<Vec<u8>, String>`, where `sizes` is an optional slice of
`(width_points, height_points)` pairs. `ocr_render_pages_checked` keeps the
three-argument convention. The original `ocr_render_pages` returns `None` on
failure. New methods require a matching native library with the new symbols.
NUL-containing Rust text or paths fail explicitly instead of becoming empty.

## Representation and limits

PDF images are decoded into RGB, composited onto white when they carry alpha,
and losslessly compressed. A PNG file is not treated as a raw PDF image stream.
The decoded dimensions must match the OCR page coordinate space. Missing or
undecodable images fail the complete document; a null image path intentionally
requests only a text layer. Each decoded RGBA image is limited to 256 MiB, and
the complete output must fit the existing signed-int byte-count ABI.

Text-only PDFs encode their small font/map streams as ASCIIHex so legacy
NUL-terminated string callers remain usable. PDFs with page images still require
the binary-safe APIs. The one-shot interface returns null on rendering failure.

An original, embedded metric font and per-document Unicode mappings avoid
system-font dependencies. Supplementary characters map through complete UTF-16
surrogate pairs. Page objects and cross-reference offsets are constructed from
complete object bodies, without a fixed page-list placeholder. These mechanisms
follow the Unicode-font and object rules in
[ISO 32000-1](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf).

Text is invisible: the page image determines visual appearance. Selection
geometry is estimated within each supplied region, not character alignment
recovered from OCR. Explicit newlines remain line boundaries. Extremely overfull
paragraph regions wrap conservatively inside their box to avoid reader
deduplication of cramped repeated letters. Exact line layout is not promised.
Searchable PDF currently rejects right-to-left script blocks and directional
formatting controls with an actionable error. Independent readers returned
reversed words from a logical-order hidden layer, and simple string reversal
cannot correctly handle mixed scripts, numbers and combining marks. A verified
full bidirectional layout engine is required before enabling these cases.
hOCR, ALTO and plain text retain their logical Unicode text without this PDF
restriction. The conservative guard covers the reserved right-to-left ranges
in [Unicode 17](https://www.unicode.org/Public/17.0.0/ucd/extracted/DerivedBidiClass.txt).

hOCR and ALTO preserve long UTF-8 corrections without a fixed text buffer.
Attribute newlines, tabs and carriage returns are escaped so XML parsing retains
them. Invalid UTF-8 and forbidden XML control characters become U+FFFD. A
negative or nonfinite confidence omits the optional XML confidence attribute;
a genuine score of zero remains zero. Callers should use unknown confidence
for detector-only scores and after manual edits invalidate recognition scores.

## Evidence

`tests/test_ocr_render.cpp` checks native lifecycle, error behavior, long text,
Unicode structures and confidence absence. `tests/test_ocr_render_artifacts.py`
creates real outputs and independently opens them with pypdf and PyMuPDF,
extracts text with three readers, rasterizes with two readers, and checks
physical size, page counts, complete corrections, search and XML parsing.
A 65542-scalar case crosses multiple font subsets and checks all three readers;
CID numeric slots in the surrogate range are avoided for reader interoperability.
The tests also require explicit PDF rejection and lossless alternate exports
for Arabic, Hebrew and directional-control inputs.

The `OCR export artifacts` workflow builds just the renderer and existing image
codecs, with no recognition models or GPU work. Its artifact retains original
images, PDFs, XML, previews, source SHA and a validation receipt. Passing these
bounded tests is not an archival conformance or complete typography claim.
