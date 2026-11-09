# Learning from pdf-craft

Review date: 9 November 2026. This review uses pdf-craft
[`cf003c711b6a63cd0a0a64bfc490bd319ab20d92`](https://github.com/oomol-lab/pdf-craft/tree/cf003c711b6a63cd0a0a64bfc490bd319ab20d92),
whose [package metadata](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pyproject.toml)
declares version 2.6.7. The CrispEmbed baseline inspected was
[`e4ee71937a68f1ff9d28f4d7cdbd66bda68ba05a`](https://github.com/CrispStrobe/CrispEmbed/tree/e4ee71937a68f1ff9d28f4d7cdbd66bda68ba05a).

The useful near-term addition is a bounded, opt-in diagnostic for repeated
multi-character OCR output. Longer-term lessons concern durable extraction,
source identity and bounded incremental processing. Adopting the entire Python
package would introduce dependencies outside this project's permissive-code
policy; a root MIT declaration does not license every dependency or every
referenced source.

## Code and model license boundaries

This is a targeted review of the selected source and visible integration
hazards, not a complete bill of materials for every pdf-craft extra or model.

| Component or source | Inspected terms and provenance | Decision for this integration |
| --- | --- | --- |
| pdf-craft's own source | [MIT](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/LICENSE), copyright 2025 Tao Zeyu | Selected code may be adapted with its copyright and permission notice. Check file-level provenance as well as the root license. |
| `pdf_craft/pdf/ngrams.py` | Standalone character-level helper under the root MIT license; introduced in [`d8e0f803a56a7cf33d089566a8cbd946abe69bd4`](https://github.com/oomol-lab/pdf-craft/commit/d8e0f803a56a7cf33d089566a8cbd946abe69bd4), then formatted in `f1009807be98e724d176fb6058f30a581f93d116` | Selected conceptual/algorithmic source for the native repetition diagnostic. No Python runtime dependency is needed. |
| pdf-craft `extractor/chapter/mergeable.py` | [The inspected file](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pdf_craft/extractor/chapter/mergeable.py) cites MinerU `fa1149cd4abf9db5e0f13e4e074cdb568be189f4` for paragraph-merging logic and stop flags. [MinerU's license at that exact revision](https://github.com/opendatalab/MinerU/blob/fa1149cd4abf9db5e0f13e4e074cdb568be189f4/LICENSE.md) is AGPL-3.0. | Excluded from the port pending clarification of the referenced material's lineage. This is a file-level provenance concern, not a claim that all pdf-craft source is AGPL. |
| `pdf2image` and Poppler | `pdf2image` is an MIT wrapper; [its documentation](https://github.com/Belval/pdf2image) requires Poppler. Poppler's [upstream README](https://gitlab.freedesktop.org/poppler/poppler/-/blob/master/README.md) identifies its GPL license. pdf-craft also [loads Poppler's GLib API](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pdf_craft/pdf/page_labels.py) to read page labels. | Do not import or bundle this rendering stack. Continue using CrispEmbed's existing PDFium integration and its build-specific notices. |
| PySide6 / Qt | pdf-craft lists `pyside6` as a direct dependency. [Qt's licensing documentation](https://doc.qt.io/qt-6/licensing.html) distinguishes commercial, LGPL and GPL components. | Not approved as a permissive dependency for this work; no Qt dependency is introduced. LGPL is not being treated as equivalent to MIT/Apache/BSD. |
| `doc-page-extractor` | [MIT source](https://github.com/Moskize91/doc-page-extractor/blob/603a6fad32d034182abdd3704bc4cceec996ff7d/LICENSE), inspected at `603a6fad32d034182abdd3704bc4cceec996ff7d`, version 1.2.2 | Its structured OCR adapter design is useful, but the package is not added. |
| `doc-page-extractor[local]` | [The pinned extra](https://github.com/Moskize91/doc-page-extractor/blob/603a6fad32d034182abdd3704bc4cceec996ff7d/pyproject.toml) explicitly includes `easydict` and identifies it as LGPLv3; local execution also loads upstream model code. [EasyDict's own repository](https://github.com/makinacorpus/easydict) confirms LGPL-3.0. | Do not infer that an MIT wrapper or permissive weights make this runtime stack permissive. No local Python OCR runtime is imported. |
| DeepSeek-OCR weights | [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR) and [license](https://huggingface.co/deepseek-ai/DeepSeek-OCR/raw/main/LICENSE) declare MIT | No new model is added. Any future selected artifact still needs exact revision, hashes, model lineage and runtime-source review. |
| DeepSeek-OCR-2 weights | [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) declares Apache-2.0 | No new model is added; existing native support and qualification remain separate from pdf-craft adoption. |
| Unlimited-OCR weights | [Official model card](https://huggingface.co/baidu/Unlimited-OCR) declares MIT | No new model is added; do not conflate this first-generation card with a separate Unlimited-OCR-2 artifact. |
| Existing TeleOCR path | [CrispEmbed's lineage audit](teleocr-license.md) and [Marker license review](marker-license.md) identify Qwen RESEARCH LICENSE restrictions on the vision tower | Keep `CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1` mandatory for every TeleOCR mode. Model-free PDFium text mode requires no model acknowledgement. |

The three current pdf-craft model cards above are not labeled noncommercial.
That observation does not remove restrictions from other models, fine-tuned
base models or executable model code. Restricted models must retain their
applicable use restrictions and an appropriate acknowledgement before access;
acknowledgement is not a grant of commercial rights. New model variants must
keep license gates independent of experimental feature gates.

This review verifies TeleOCR's direct-runtime and Marker acknowledgement
checks; it does not establish universal runtime enforcement. The registry's
download consent does not cover every local-file or cached-model load: those
paths return before the download check. The existing `jina-ocr-v1` profile's
reuse of the Unlimited runtime had a model-specific noncommercial access-gate
gap. The authorized follow-up now adds [native admission](jina-ocr-license-gate.md)
for known Jina identity or declared CC-BY-NC terms. All 106 metadata-only controls
pass in the owner-authorized Linux CPU execution. This is not a claim of universal model enforcement;
editable metadata cannot authenticate wholly relabeled artifacts. Backend
selection remains unchanged. See the current
[registry resolver](../examples/cli/model_mgr.cpp) and
[Unlimited loader](../src/unlimited_ocr.cpp).

The selected change imports no GPL/AGPL code, model weights, network service,
Python dependency, Poppler, Qt or EasyDict. Native adaptation must retain the
pdf-craft MIT notice alongside the existing Marker Apache-2.0 attribution.

## What CrispEmbed already provides

The [native Marker pipeline](marker.md) already performs substantial document
processing. The comparison should start from that implementation rather than
duplicate it or replace it with pdf-craft's Python orchestration.

| Area | Existing CrispEmbed behavior | Useful pdf-craft lesson or remaining gap |
| --- | --- | --- |
| OCR engines | Native DeepSeek-OCR-2 and Unlimited-OCR engine paths already exist; their release and quality gates are separate. The Marker integration specifically uses PDFium and TeleOCR. | Adding another wrapper around the same models is not the main opportunity. A common page/layout adapter could later let already-qualified engines serve the document pipeline. |
| Document structure | [Marker's IR](../src/marker_document.h) retains pages, blocks, geometry, raw text, raw layout, confidence, cap status, source strings and section hierarchy. | pdf-craft persists a versioned extraction artifact that downstream rendering and translation can reopen. Marker currently renders the in-memory result; it has no equivalent persistent-extraction input contract. |
| Processing | Native heading hierarchy, lists, captions, two-column heuristics, repeated marginalia suppression and paragraph continuation are already implemented. | Richer cross-page identities and OCR-tolerant margin matching require separate owned fixtures and provenance work. They are not obtained by copying a whole paragraph-merging module. |
| Text quality | `text_is_usable()` checks malformed/control/private-use characters and identical-codepoint runs longer than 48 when selecting the PDF text layer. | It does not diagnose multi-character repetition in already-recognized OCR output. This is the bounded gap selected here. |
| Outputs | Markdown, HTML, nested JSON and chunks expose document structure and provenance; malformed UTF-8 remains recoverable through JSON byte fields. | Preserve source evidence while adding diagnostics. A diagnostic must not silently delete text, alter coordinates or claim corrected OCR. |
| Footnotes | Footnote blocks are classified, retained and rendered. | Cross-page note identity and explicit reference links are not implemented. A source-mapped reference graph would be a distinct feature. |
| Paragraph continuation | Native continuation is represented by a boolean and is inferred from PDF line geometry and conservative textual cues. | Explicit continuation targets, multi-page source ranges and retrieval chunks that preserve those relationships need an additive contract. OCR blocks generally lack the PDF line geometry used by the current heuristic. |
| Execution | Conversion finalizes the document as a whole; failures clear the result. | pdf-craft's bounded page scheduling, progress events, cancellation and persistent completed work suggest a larger resumable-processing design. No such lifecycle change is made by this diagnostic. |

See [feature coverage](marker-feature-coverage.md) and
[validation boundaries](marker-validation.md) for what existing fixtures prove.
Native implementation availability is not evidence of representative OCR
accuracy or parity with an upstream model.

## Selected adaptation: repeated OCR text diagnostics

The source is
[`has_repetitive_ngrams`](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pdf_craft/pdf/ngrams.py).
It searches for consecutive copies of a character sequence, so it works with
scripts that do not separate words with spaces. Its
[page-extraction caller](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pdf_craft/pdf/page_extractor.py)
uses these two threshold bands:

| Repeated unit length, in Unicode codepoints | Minimum consecutive copies |
| --- | --- |
| 2 through 5 | 16 |
| 6 through 20 | 8 |

A match is evidence of a repetitive span, not proof of a recognition mistake.
Repeated table cells, program text, equations and intentional prose repetition
can be legitimate. Upstream already exempts its asset tags and drops matching
non-asset layouts. CrispEmbed's selected adaptation is more conservative:

- Gate the new diagnostic with `CRISPEMBED_MARKER_REPETITION_CHECK=1`, default
  off, using the shared value-parsed environment helper. Values `0`, `false`,
  `off` and `no` keep it disabled.
- Inspect Unicode codepoints rather than UTF-8 bytes; a multi-byte character
  must not be counted as several characters. Keep the scan bounded by the
  existing document text limit and fixed maximum repeated-unit length.
- Diagnose ordinary prose-like blocks after list splitting, including each
  list item's own text when it contains a nested list. Skip aggregate container
  text to avoid counting the same content twice. Exclude
  tables, images, figures, equations, chemical content and code, including their
  structural children. Preserve the current handling of model cap warnings and
  invalid text independently.
- Expose the detector as `configuration.repetition_check=unicode-tandem-v1`
  and the decimal string `configuration.repetitive_blocks`; block warnings
  appear in JSON and chunks. Reprocessing clears only owned repetition
  diagnostics and recomputes counts, including when the gate is turned off.
- Retain the original text, raw text, geometry, source attribution and rendered
  content. Add inspectable warnings and diagnostic metadata only. Do not retry
  OCR, change decoding, reject a document or remove a block because of a match.
- Do not change `text_is_usable()` selection behavior in this first increment.
  The post-extraction diagnostic and the existing PDF-layer heuristic answer
  different questions.

This also differs from the native engines' existing token-level no-repeat
generation controls. Those controls affect decoding; the selected diagnostic
observes completed document text without modifying the generated tokens.

The adoption does not claim an OCR accuracy improvement. It gives callers and
review tooling a precise warning to inspect while retaining all available
evidence. Promotion to a default, automatic rerun policy or text suppression
requires a separately reviewed corpus study including intentional repetition
and non-Latin documents.

## Authorized implementation roadmap

The user authorized these follow-ups on 9 November 2026. Their implementation
and acceptance are tracked in [the task checklist](pdf-craft-tasks.md). The
repetition diagnostic alone does not deliver or validate these features.

1. **Persist and reopen extraction.** pdf-craft's
   [PCEX contract](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/docs/en/PCEX_FORMAT.md)
   separates source extraction from rendering and translation, with a versioned
   manifest, source geometry and assets. For CrispEmbed, start with a bounded,
   versioned serialization of the existing native IR. Define migration,
   untrusted-input limits, artifact hashes and model/preprocessing provenance
   before adding a reader. Reuse must invalidate stale extraction when those
   identities change.
2. **Make retrieval chunks traceable across pages.** pdf-craft's
   [source-range compiler](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pdf_craft/extractor/chapter/source_ranges.py)
   keeps offsets against immutable source text and does not invent smaller
   rectangles for text slices. Extend Marker with explicit source ranges and
   continuation targets so an assembled chunk can still identify every
   contributing page/block. Keep page-local raw records independently
   inspectable.
3. **Bound work and preserve completed units.** The upstream
   [streaming design](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/docs/design/pdf-extraction-streaming.md)
   uses bounded admission, backpressure and disk-backed results while retaining
   semantic paragraph boundaries. A native design needs an explicit progress,
   cancellation, partial-result and resume contract across C/Python/Rust/Dart
   surfaces. It should not change existing all-or-nothing behavior implicitly.
4. **Improve multilingual normalization conservatively.** The independent
   [whitespace helper](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/pdf_craft/common/whitespace.py)
   distinguishes removable padding from join controls and variation selectors,
   and preserves inline boundary spaces. That principle can improve native
   text normalization, provided code, TeX, source offsets and intentional
   spacing remain separate. Use owned Unicode fixtures before any default
   change.
5. **Add source-backed footnote and page-label identities.** The upstream
   [document contract](https://github.com/oomol-lab/pdf-craft/blob/cf003c711b6a63cd0a0a64bfc490bd319ab20d92/references/architecture.md)
   separates physical page identity, reading flow and source fragments. The
   native implementation can adopt that distinction through existing
   permissive PDF APIs. Do not import the Poppler implementation or the
   MinerU-referenced merge rules to obtain it.

Translation, EPUB export and layout-preserving PDF writeback are larger product
choices. They are not prerequisites for better OCR diagnostics or better
retrieval provenance, and their upstream dependencies require their own review.

## Executed validation and remaining scope

The owner-authorized Linux CPU execution at native source
`891ea3d3632c6a3c3155e25b34705da6a02ec807` passes all 185 native document checks,
including the repetition diagnostic and the following controls. The
[execution evidence](../tests/results/marker/pdf-craft-local-2026-11-09/README.md)
also records native artifacts/lifecycle, public clients, model admission and
package notices. Windows/macOS execution remains separate.

- Threshold boundaries for both bands, including a just-below-threshold control,
  a qualifying span within otherwise ordinary prose, and interrupted repeats.
- UTF-8 fixtures covering CJK and supplementary-plane characters; invalid
  encoding must not create a misleading repetition diagnosis.
- Asset and code exclusions, including nested structural content, and evidence
  that text/raw/geometry/rendered content remain unchanged.
- Gate-unset and explicit-false controls, document/block warning exposure and
  repeated processing without duplicate warnings.
- The existing bounded Marker tests, environment-variable documentation checks
  and attribution/package checks appropriate to the changed files.

Model inference, model downloads and broad document-quality claims are outside
this validation scope. A model-free diagnostic test demonstrates its contract,
not a change in OCR accuracy, retrieval relevance or production defaults.
