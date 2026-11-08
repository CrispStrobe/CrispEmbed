# OCR gold corpus v1 (layout, reading order, tables)

[`ocr-gold-corpus-v1.json`](ocr-gold-corpus-v1.json) is the licensed real-page gold set for
layout-aware OCR benchmarks (Marker-style pipeline, TeleOCR, MinerU, the real-page lanes).
It grows the [real-scan v2 manifest](ocr-real-scans-v2.json), which stays unchanged: all
20 v2 pages are reused (`v2_id`) with their image pins byte-for-byte, and 37 pages are new (57 total).

| | pages |
|---|---|
| English / German / French Wikisource (Validated) | 17 / 13 / 12 |
| CORD-v2 receipts (CC BY 4.0) | 15 |
| Wikisource pages with Fraktur / multi-column / table / footnote / list / figure blocks | 11 / 9 / 12 / 10 / 2 / 4 (+15 receipt tables) |

Per page: image pin, licences, provenance, and ordered `blocks` with `type` (text, title,
table, list, figure, footnote, header, footer), `text`, `bbox` (`[x0, y0, x1, y1]` pixels of
the pinned image, or null), reading `order`, and table structure (`table.cells` with row,
col, rowspan, colspan, text). Every block says how it was made: `type_provenance` and
`bbox_provenance` start with `derived:` (rule-based, unreviewed) or `manual:` (dataset
annotators or inspection of the pinned scan).

## Licence policy

- Images: public domain, CC0, CC BY or CC BY-SA only (Commons `LicenseShortName` / dataset
  card). Non-commercial, no-derivatives, research-only and unstated licences are rejected;
  the rejections and reasons are listed in `rejected_sources` (FUNSD, SROIE, DocVQA,
  DocLayNet and PubLayNet image rights, Fox/OmniDocBench, IAM, olmOCR-bench, plus excluded
  Wikisource groups whose authors or translators may still be in copyright).
- Ground truth: Wikisource transcriptions CC BY-SA 4.0 (attribution via the pinned revision
  URL); CORD annotations CC BY 4.0.
- Wikisource pages must be ProofreadPage quality 4 (Validated by two contributors) at the
  pinned `revision_id`.

## Image pinning and drift

Every image is pinned by sha256; the sha256 is the gate. Wikimedia images additionally pin
the Commons file revision (timestamp + file sha1).

- Single-image files (`.jpg`/`.png` uploads) use the original upload URL at that revision:
  immutable bytes.
- Pages of multi-page DjVu/PDF files only exist as Wikimedia renders. Those bytes have been
  observed to drift (v2 history). The revision-qualified URL is tried first, then the bare
  URL; on mismatch the page is reported as drifted and not scored. The recovery is to
  re-render that page of the pinned file revision at the recorded width and publish the new
  sha256 as a new corpus version, never to edit v1 in place.
- All accepted licences permit redistribution, so an exact-byte mirror is allowed if
  rendering drift becomes a problem. Bytes are not committed here (repository size policy).
- Hosted CI run 42372548797 verified 40/42 Wikimedia pins. Two DjVu renders were served
  with different bytes from the hosted runner than at pin time (`en-form-economic-journal-67`,
  the known v2 drifter, and `en-register-1758-16`); both observed hashes are recorded in
  `image.render_variants`, are not accepted by the gate, and those pages carry no derived
  boxes. The lane exits non-zero while any drift exists, by design.
- Image `width`/`height` are measured from the verified bytes. Wikimedia snaps paged
  renders to standard widths (a 1600px request is served at 1920px), so the API-reported
  size, kept as `api_reported_size`, was wrong for 36 pages. The same API sizes are in the
  unchanged v2 manifest; consumers of v2 should measure images rather than trust them.
- CORD images live in the pinned Hub parquet (revision + file sha256 + row).

## How annotations are made

- **Wikisource text, types, order, tables** — derived from the MediaWiki rendering of the
  pinned revision (`tools/ocr-gold-corpus/wikisource_layout.py`): header section → header,
  `<references/>` items → footnote, headings and short centred lines → title, grid tables
  → table with cells, lists → list, file embeds → figure, a long note in the footer section
  → footnote, a lone lower-case word ending the body → footer (catchword). Reading order
  is the transcription order (columns in reading sequence). The concatenated block text of
  every reused v2 page reproduces its v2 ground truth exactly (CER 0 under the v2 scorer).
- **Manual edits** — the v2 scan-inspection edits are carried as explicit ops
  (`provenance.manual_edits`), e.g. running heads that de.wikisource does not transcribe and
  a table header row missing from a transcription.
- **Wikisource boxes** — derived on hosted CI by aligning each block's transcription to
  official Tesseract word boxes (`align_boxes.py`; the Tesseract text never enters the gold).
  Run 42372548797 (Tesseract 5.3.4) boxed
  307 of 381 blocks. A block gets a box only when at least 30% of its tokens (and two words) align; the match
  ratio is recorded. Unaligned blocks keep `bbox: null` and are scored by text matching.
- **CORD** — boxes are the annotators' word quads (manual); menu lines become one table
  block (rows by group/sub-group, columns by field), other groups become text blocks.
  CORD annotations are partial (no store header/address), as in v2.

Known derivation limits are listed in `policy.known_derivation_limits` (small-print notes
typed only by templates stay text, footnote marker glyphs are not recoverable from
wikitext, de.wikisource running heads exist only where a manual op adds them).

## Scoring

`tools/ocr-gold-corpus/layout_score.py GOLD PRED [OUT]` scores predicted blocks per page:
block precision/recall/F1 (one-to-one matching: IoU ≥ 0.5 when both sides have boxes,
otherwise normalised text similarity ≥ 0.5), mean IoU, type accuracy (common label aliases
folded), reading-order Kendall tau-b over matched blocks, and table cell F1 over
(row, col, text) plus content F1 over the cell texts (cells, HTML or Markdown tables
accepted). Text CER/WER stays with `tools/kaggle/ocr-real-scans/score.py`, whose
normalisation the layout scorer reuses. Tests: `tests/test_ocr_gold_corpus.py`.

## Rebuilding

```bash
python3 tools/ocr-gold-corpus/build_corpus.py fetch docs/ocr-gold-corpus-v1.json   # network only
gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/ocr_gold_corpus.sh \
    -f isolated_lane=ocr-gold-corpus   # pin verification, CORD bytes, Tesseract boxes
```

The CI artifact `ocr-gold-corpus-v1.json` replaces the committed file only after review.

### Intermittent thumbnail bytes (7 November 2026)

In the GOT-OCR2 gold run, `en-form-economic-journal-67` and `en-register-1758-16` failed their sha256 pin in
two of three parallel Kaggle sessions, while the third session fetched the pinned bytes from the same
revision-qualified URL. The pins are correct; Wikimedia's thumbnail CDN intermittently serves other bytes.
`tools/kaggle/ocr-real-scans/kernel.py` now retries all pinned URLs for up to four rounds and records every
attempt's sha256; pins are never changed by a fetch.
