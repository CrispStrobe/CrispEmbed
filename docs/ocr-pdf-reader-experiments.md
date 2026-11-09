# Reviewed PDF reader representation experiments

These hosted fixtures are diagnostic code, not a production export path or an
application dependency. Every mapping contains its actual Unicode scalar or
logical combining cluster. No reversed replacement text or invalid Unicode is
introduced to satisfy a reader. Original source, PDFs, copy/search output,
font resources, pixel hashes and veraPDF reports remain in immutable artifacts.

## Completed visual-emission matrix

Source `2a0da548ac28380d6d0d1905471b2b12c7938939`,
[run 42624231301](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42624231301),
fixes the earlier fontTools fixture construction errors and actually exercises
all 84 files. All 84 retain the original raster and pass veraPDF PDF/A-2b.
Exact normalized logical copy matches across twelve cases are:

| Representation | MuPDF | pypdf | Poppler CLI | PDFium |
| --- | ---: | ---: | ---: | ---: |
| Native logical run | 11 | 12 | 3 | 1 |
| Native run without ActualText | 11 | 12 | 3 | 2 |
| Logical scalar emission | 9 | 11 | 1 | 3 |
| Visual scalar emission | 4 | 3 | 3 | 3 |
| Visual cluster emission | 4 | 3 | 3 | 3 |
| Visual scalars plus logical ActualText | 3 | 3 | 3 | 5 |
| Logical scalars plus logical ActualText | 9 | 11 | 3 | 5 |

Pure Arabic and Hebrew succeed in all four readers with visual scalar emission.
Mixed-direction run order, combining order, and supplementary RTL still fail.
No representation passes the complete twelve-case copy/search matrix. The
ordinary LTR supplementary case copies correctly without ActualText but still
fails PDFium's whole-scalar emoji search. The native renderer in this diagnostic
is the preserved earlier source, not the later production writer candidate.
See the [source-bound receipt](../tests/results/ocr-render/strategies-2a0da54.json).

## Reflected-axis and paragraph-direction controls

The next isolated matrix keeps the complete ground truth and tests reflected
horizontal text matrices for resolved RTL runs. An
[upstream OCR PDF renderer](https://github.com/tesseract-ocr/tesseract/blob/main/src/api/pdfrenderer.cpp)
also uses a reflected horizontal axis for right-to-left text. This experiment
uses the existing original metric fonts and independently resolved Unicode 17
positions; it does not copy that implementation. A `TJ` array applies exact
spacing corrections within each run so changing the text axis cannot move the
selection cells or alter the visible image.

Controls independently vary scalar versus combining-cluster mappings, logical
emission with a reflected axis, one show array per run, and visual emission
inside logically ordered runs. Document viewer direction is a separate control.
The audited [PDFium text API](https://pdfium.googlesource.com/pdfium/+/refs/heads/main/fpdfsdk/fpdf_text.cpp)
passes document `ViewerPreferences` direction into the text-page constructor;
its file blob was `960b2aefce19f586cc3e70d6a54948e67553d6a4` on 9 November.
The [direction heuristic](https://pdfium.googlesource.com/pdfium/+/refs/heads/main/core/fxcrt/fx_bidi.cpp)
counts strong-direction segments and chooses RTL only when they outnumber LTR
segments (blob `17e43c7f1e20b81951406fe0657217fe2da1c10a`). A document preference
may resolve a tie, but cannot establish different base directions for every
paragraph. The multiline case deliberately mixes paragraph directions to
expose that limitation. No directional characters are inserted into reviewed
text. Every positive requires independent bounds within the supplied region,
exact pixels, exact logical copy, successful requested searches and veraPDF.

The reflected-axis source has no result until its own hosted reports return.
Its findings must not be used to enable the currently withheld bidi capability.
