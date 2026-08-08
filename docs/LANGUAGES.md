# Language support matrix (OCR lanes + embedding models)

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

## Embedding / retrieval models — Japanese (verified 2026-08-08)

Prior to this the whole matrix was OCR-only, and the registry's language
strings for embedders (`"XLM-R 768d 100+ languages"`) were **upstream claim**
inherited from the model card — no Japanese had ever been run through an
embedder here. It has now, on this fixture set:

| Model (quant tested) | Tokenizer | JA paraphrase | JA-EN cross-lingual | Verdict |
|---|---|--:|--:|---|
| `granite-embedding-107m-multilingual` (q4_k_m) | SentencePiece 250k | 0.966 vs 0.423 | 0.943 vs 0.458 | **best of the set** |
| `bge-m3` (iq4_xs) | SentencePiece 250k | 0.945 vs 0.406 | 0.892 vs 0.440 | **strong** |
| `jina-v5-small` (q4_k) | SentencePiece | 0.947 vs 0.065 | 0.919 vs 0.081 | **strong, sharpest separation** |
| `LFM2.5-Embedding-350M` (q8_0) | - | 0.882 vs 0.066 | 0.842 vs 0.062 | strong |
| `Qwen3-Embedding-0.6B` (q8_0) | - | 0.885 vs 0.196 | 0.801 vs 0.247 | strong |
| `nomic-embed-text-v2-moe` (q4_k_m) | - | 0.885 vs 0.161 | 0.789 vs 0.183 | works |
| `arctic-embed-m-v2` (q4_k-imatrix) | - | 0.711 vs 0.169 | 0.701 vs 0.173 | works, weaker |
| `jina-v5-nano` (q4_k) | SentencePiece | 0.671 vs 0.217 | 0.600 vs 0.279 | works, weakest |
| `all-MiniLM-L6-v2` (q4_k_m) | **WordPiece 30k EN** | 1.0000 vs 0.286 | -0.007 vs 0.140 | **DO NOT USE for JA** |
| `all-mpnet-base-v2` (q8_0) | **WordPiece 30k EN** | 1.0000 vs 0.129 | -0.085 vs 0.078 | **DO NOT USE for JA** |

Columns are cosine similarities: *paraphrase* = two Japanese sentences meaning
the same thing vs. an unrelated Japanese sentence; *cross-lingual* = a Japanese
sentence vs. its English translation vs. an unrelated English sentence. Higher
first number and a bigger gap is better. Every multilingual model separates
both pairs cleanly - **Japanese works**.

WARNING: **the English-only models fail in a way that LOOKS like success.**
Read the `1.0000`: `all-MiniLM-L6-v2` and `all-mpnet-base-v2` return
**bit-identical embedding vectors for two different Japanese sentences**
(verified by direct vector comparison, not just rounding). A naive paraphrase
test therefore "passes" them with a huge margin while the model is emitting a
constant. The tells are the cross-lingual column going *negative* and the
tokenizer column: a 30k uncased WordPiece vocab has no meaningful Japanese
coverage, so distinct inputs collapse onto the same tokens. Consequence for
users: picking an English-only embedder for Japanese does not give you
"somewhat worse" retrieval - it gives you **silently arbitrary** retrieval at
high confidence.

This collapse is out-of-domain behaviour of an English model, not a regression
in the multilingual lanes: every multilingual embedder in the registry uses a
SentencePiece/XLM-R-class vocabulary (250k), a different tokenizer path from
the 30k WordPiece one. Reference HF tokenization of the same two sentences
*does* differ, so our WordPiece CJK path is not byte-faithful to HF on
Japanese input - recorded as a follow-up in `PLAN.md`; it changes nothing for
the models anyone should use on Japanese.

**Recommendation for Japanese retrieval:** `granite-embedding-107m-multilingual`
(best measured, 107M) or `bge-m3` (strong, 8k context, also does sparse and
ColBERT retrieval).

### Re-running this

```bash
python tests/embed_language_eval.py ./build/crispembed ~/models out.json
```

The harness is deliberately built around three checks, because the first alone
is not a test: (1) monolingual paraphrase > unrelated, (2) cross-lingual
alignment, (3) **non-degeneracy** - distinct inputs must not produce identical
vectors. English-only models stay in the model list on purpose as a negative
control: a language test that every model passes is measuring nothing.
Extending it to another language is a five-line edit of `TEXTS`.

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
