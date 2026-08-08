# Language support matrix (OCR lanes)

Evidence-based per-lane language coverage. Three evidence tiers, stated per
row — never conflate them:

- **verified** — decoded-output test on a fixture in this repo, result stated;
- **dict-scan** — the shipped GGUF's recognition dictionary provably contains
  the script's characters (method below); coverage is necessary but NOT
  sufficient for quality;
- **upstream claim** — the base model's own documentation; unbenchmarked here.

## PP-OCRv6 pipeline (`--ocr-engine ppocrv6`)

⚠ **The tiny recognizer has NO kana — it is not Japanese-capable.** Dict scans
of the shipped GGUFs (2026-08-08):

| rec model | dict classes | CJK ideographs | kana (hira+kata) | Latin | Hangul | Cyrillic |
|---|--:|--:|--:|--:|--:|--:|
| `ppocrv6-tiny-rec` | 6 904 | 6 174 | **0** | yes | 0 | 0 |
| `ppocrv6-small-rec` | 18 708 | 15 565 | 180 | yes | 0 | 0 |
| `ppocrv6-medium-rec` | 18 708 | 15 565 | 180 | yes | 0 | 0 |

- **Chinese (Simplified/Traditional), English, Japanese**: dict-scan for all
  three on small/medium; **Japanese verified 2026-08-08** — the synthetic
  printed fixture `tests/regression/images/japanese_print.png` (ground truth
  beside it) decodes **exactly** (3/3 lines, conf 0.97–0.98, boxes within 1 px
  of official PaddleOCR running the same models):

  ```
  crispembed -m <any> --ocr japanese_print.png --ocr-engine ppocrv6 \
      --ocr-det ppocrv6-medium-det --ocr-rec ppocrv6-medium-rec
  ```

- **Korean, Cyrillic, Arabic, Devanagari…**: NOT in these dicts. Upstream
  PaddleOCR serves separate per-language rec models
  (github.com/PaddlePaddle/PaddleOCR, `korean_…`, `cyrillic_…`, etc.);
  converting one through `models/convert-ppocrv6-to-gguf.py` picks up its
  dictionary automatically (the converter reads the YAML `character_dict`) —
  unconverted today.

⚠ **Invocation trap (costs you the correct recognizer silently):** the
pipeline reads `--ocr-rec`/`--ocr-det`. Passing the recognizer only as
`-m rec.gguf --ocr img` dispatches by GGUF arch into **single-model rec-only
mode** — the whole page is squashed to one 48-px line strip and the output is
one garbled line. Always pass `--ocr-rec` for pipeline use. Since 2026-08-08
the CLI prints a warning naming the correct command when a line recognizer is
handed a page-shaped image (a single cropped line stays silent — that flow is
legitimate).

## Tesseract-LSTM lane (`-m tesseract-<lang>`)

Registry ships 12 converted models: `eng deu fra spa ita por nld rus ara jpn
kor chi_sim` (HF `cstr/tesseract-lstm-GGUF`). Latin-script + Fraktur languages
are the tuned, benchmarked path (CER work in `PERFORMANCE.md`).

All 12 dict-scanned 2026-08-08 (`tools/scan_model_languages.py`, unicharset
classes; the registry `languages` field and `--list-models` carry these):

| model | classes | scripts | note |
|---|--:|---|---|
| `tesseract-{eng,deu,fra,spa,ita,por,nld}` | 109–151 | latin | |
| `tesseract-rus` | 125 | latin+cyrillic | |
| `tesseract-ara` | 85 | latin+arabic | |
| `tesseract-chi-sim` | 4 022 | latin+cjk | 3 899 ideographs, **no kana** |
| `tesseract-jpn` | 2 693 | latin+cjk+kana | 2 380 ideographs + 171 kana |
| `tesseract-kor` | 1 158 | latin+hangul | 1 089 hangul, **zero CJK ideographs** — mixed hanja Korean is out of dict |
**`tesseract-jpn` root-caused 2026-08-08 — two stacked defects:**
(1) the production default decode is the single-code greedy path, but CJK
traineddata encodes kanji as MULTI-CODE (radical-stroke) sequences — every
kanji decodes as `<class>` while kana pass through. The opt-in
`CRISPEMBED_TESSERACT_RECODE_COMPOSE=1` fixes recognition COMPLETELY on
clean line crops (`日本語のテキスト認識テスト` exact). (2) the lane's
page segmentation fails on the Japanese page independently (garbage crops;
output unchanged by compose).

**Both are FIXED as of 2026-08-08.** (1) Composed recoding auto-enables when
the loaded recoder is multi-code (`b61f22ae`); Latin models keep the greedy
default byte-identically. (2) The tesseract stage now accepts a **PP-OCRv6
detector** in the `--ocr-det` slot, which supplies line-level CJK boxes where
DBNet-IC15 fragmented the page into word boxes. Full-page Japanese decodes
**byte-exactly** (3/3 lines, page CER 0.0000) against
`tests/regression/images/japanese_print.gt.txt`:

```
crispembed -m tesseract-jpn --ocr japanese_print.png \
    --ocr-det ppocrv6-medium-det --ocr-rec tesseract-jpn
```

The Latin lane is unaffected — with a DBNet detector the historical path runs
unchanged (byte-identical on fox, scan_strip, simple_form, receipt_example,
german_official_print).

## VLM document engines (upstream claims unless noted)

| engine | languages | tier |
|---|---|---|
| PaddleOCR-VL 0.9B / 1.6 (`paddleocr-vl`) | 109 languages incl. ja/ko/ru/ar | upstream claim |
| Qwen2.5-VL / Qwen3-VL lanes | broad multilingual | upstream claim |
| GLM-OCR | zh/en-centric | upstream claim |
| GOT-OCR2 | zh/en | upstream claim |
| Qari-OCR | Arabic only | upstream claim |
| german-ocr-3.1 | German | upstream claim |
| deepseek-ocr2, olmocr, … | see model cards | upstream claim |

For Japanese documents beyond clean print (photos, layouts, handwriting),
`paddleocr-vl` is the first VLM to try; for clean printed Japanese the
PP-OCRv6 pipeline above is verified and far cheaper.

## Detection / auxiliary

- DBNet, PP-OCRv6 det, Surya det (91-language training set): script-agnostic
  region detectors — language support is decided by the recognizer.
- LID: CLD3 (109 langs) / GlotLID (2102 ISO 639-3) for text language ID.

## How to verify a dict yourself

The recipe below now ships as a tool, and its output IS the registry field:

```bash
python tools/scan_model_languages.py model.gguf     # per-script class counts
crispembed --list-models                            # the "Scripts" column
```

Both PP-OCRv6 and Tesseract GGUFs expose `tokenizer.tokens` (for Tesseract
that is the unicharset), so one scanner covers both. The bare recipe:

```python
from gguf import GGUFReader
r = GGUFReader("model.gguf")
f = r.fields["tokenizer.tokens"]
toks = [bytes(f.parts[i].tobytes()).decode("utf-8","replace") for i in f.data]
kana = sum(1 for t in toks if len(t)==1 and 0x3040 <= ord(t) <= 0x30FF)
```

Scan the Unicode block of the script you need; zero coverage = the model
cannot emit that script, whatever the marketing says.

⚠ **Read the direction of the implication.** Coverage is NECESSARY but NOT
SUFFICIENT: kana in the dict says the model *can* emit kana, never that it
reads Japanese well. Only the zero direction is conclusive. A blank `Scripts`
column means *not scanned* — not "no coverage".

`tests/test_registry_languages.py` guards the field against typo'd labels and
against drift on the measured facts (tiny-rec has no kana; `tesseract-kor` has
no CJK ideographs).
