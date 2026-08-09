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
| `granite-embedding-107m-multilingual` (q4_k) | SentencePiece 250k | 0.967 vs 0.436 | 0.940 vs 0.470 | **best of the set** |
| `paraphrase-multilingual-MiniLM-L12-v2` (q8_0) | SentencePiece 250k | 0.981 vs -0.055 | 0.987 vs -0.054 | **strongest separation** |
| `multilingual-e5-large` (q8_0, no prefix) | SentencePiece 250k | 0.986 vs 0.805 | 0.913 vs 0.747 | strong (narrow margin) |
| `multilingual-e5-base` (q8_0, no prefix) | SentencePiece 250k | 0.977 vs 0.815 | 0.908 vs 0.730 | strong (narrow margin) |
| `multilingual-e5-small` (q8_0, no prefix) | SentencePiece 250k | 0.968 vs 0.791 | 0.903 vs 0.735 | pass (narrowest margin) |
| `granite-embedding-278m-multilingual` (q8_0) | SentencePiece 250k | 0.957 vs 0.392 | 0.938 vs 0.424 | **strong** |
| `bge-m3` (iq4_xs) | SentencePiece 250k | 0.945 vs 0.406 | 0.892 vs 0.440 | **strong** |
| `jina-v5-small` (q4_k) | SentencePiece | 0.947 vs 0.069 | 0.920 vs 0.082 | **strong, sharpest separation** |
| `LFM2.5-Embedding-350M` (q8_0) | - | 0.882 vs 0.066 | 0.842 vs 0.062 | strong |
| `Qwen3-Embedding-0.6B` (q8_0) | - | 0.885 vs 0.196 | 0.801 vs 0.247 | strong |
| `nomic-embed-text-v2-moe` (q4_k_m) | - | 0.885 vs 0.161 | 0.789 vs 0.183 | works |
| `arctic-embed-m-v2` (q4_k-imatrix) | - | 0.711 vs 0.169 | 0.701 vs 0.173 | works, weaker |
| `jina-v5-nano` (q4_k) | SentencePiece | 0.954 vs 0.075 | 0.934 vs 0.096 | works |
| `granite-embedding-278m` (q8_0, non-multi) | SentencePiece 250k | 0.957 vs 0.392 | 0.938 vs 0.424 | = multilingual variant |
| `all-MiniLM-L6-v2` (q4_k) | **WordPiece 30k EN** | 1.0000 vs 0.330 | 0.019 vs 0.170 | **DO NOT USE for JA** |
| `all-mpnet-base-v2` (q8_0) | **WordPiece 30k EN** | 1.0000 vs 0.355 | -0.076 vs 0.093 | **DO NOT USE for JA** |

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

**E1 additions (2026-08-08, VPS run):** 5 new models verified. Notable
findings:

- **`paraphrase-multilingual-MiniLM-L12-v2`** has the strongest JA separation
  of ANY model tested (margin >1.0 on both checks, unrelated cosine goes
  NEGATIVE), despite being only 12 layers / 384d. Only 125 MB at q8_0.
- The **e5 family** (`multilingual-e5-{small,base,large}`) all pass but with
  notably narrower margins (0.16–0.18) than the granite/bge/jina models. This
  may be because e5 expects `query: ` / `passage: ` prefixes, which were NOT
  used here — these are prefix-free runs, fair for relative comparison but
  potentially undervaluing e5. The `--prefix` flag exists for users who want to
  test with the contract.
- **`granite-embedding-278m`** (non-multilingual) produces identical scores to
  `granite-embedding-278m-multilingual`. They may share the same weights /
  differ only in metadata.
- `granite-embedding-97m-r2` and `granite-embedding-311m-r2` were not cached on
  the VPS; recorded as SKIP (not FAIL). They are BPE/o200k models, a different
  tokenizer family from the XLM-R ones tested here.

**Runtime warning:** since `1b5870da`, CrispEmbed emits a one-shot stderr
warning when ≥50% of content tokens are `[UNK]`, signalling a vocabulary
mismatch. This catches the English-model-on-Japanese case automatically.
Silenced by `CRISPEMBED_WARN_UNK=0`.

### Accented-Latin tokenizer divergence on uncased WordPiece models — FIXED

**Status: fixed.** The runtime now applies HuggingFace's `BertNormalizer`
strip-accents + lowercase stage before WordPiece lookup for uncased WordPiece
models. Historical behavior is still reachable with
`CRISPEMBED_WORDPIECE_HF_NORM=0`.

The bug: HF's normalizer, for a model whose `tokenizer.json` says
`lowercase: true, strip_accents: null`, runs NFD and drops combining marks
before the vocabulary lookup, so `café` is the single in-vocab token `cafe`.
CrispEmbed lowercased with a per-**byte** `std::tolower`, which does nothing to
a multi-byte UTF-8 sequence and strips no accents at all. Measured on
`all-MiniLM-L6-v2`:

| Input | HF tokens | CrispEmbed before | CrispEmbed now |
|---|---|---|---|
| café | `cafe` (1 token, in-vocab) | `caf` + `[UNK]` | `cafe` |
| Müller | `muller` (1 token, in-vocab) | `m` + `[UNK]` | `muller` |
| naïve | `naive` (1 token, in-vocab) | `na` + `[UNK]` | `naive` |
| résumé | `resume` (1 token, in-vocab) | `r` + `[UNK]` | `resume` |
| über | `uber` (1 token, in-vocab) | `[UNK]` | `uber` |
| Ángel | `angel` (1 token, in-vocab) | `[UNK]` | `angel` |
| François | `francois` (1 token, in-vocab) | `fran` + `[UNK]` | `francois` |

**Token-id parity vs HF**, 24 sentences across 6 sections
(`tests/wordpiece_hf_parity.py`, driving the shipping C++ tokenizer against
HF's fast tokenizer on the real vocab):

| model | exact-match before | exact-match now | `[UNK]` before → now |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 4/24 | **24/24** | 80 → **0** |
| `all-mpnet-base-v2` | 4/24 | **24/24** | 80 → **0** |

The 4 that already matched are the ASCII section — which is the point below.

**End-to-end embeddings** (`tests/embed_accent_parity.py`, real CLI + real
`all-MiniLM-L6-v2` f32 GGUF, reference = the model's own ONNX export under ONNX
Runtime, mean-pooled and L2-normalized). Token-id parity is not the acceptance
test; this is:

| section | cos vs reference before | after |
|---|---|---|
| ASCII (3 sentences) | 1.000000 | 1.000000, **bit-identical to before** |
| accented (5 sentences) | 0.646574 mean | **1.000000** |

Per sentence the old path ran 0.487 (French), 0.566 (Spanish), 0.574
(Portuguese), 0.731 (German), 0.875 (Norwegian) — a French sentence embedded at
cosine 0.487 to its own reference is, for retrieval purposes, a different
vector. All five are now exact.

**Why this was safe to turn on by default.** The normalization is exactly
`std::tolower` over the whole of ASCII, so no pure-ASCII input can tokenize
differently than it did before and no shipped English embedding moves. That is
not a claim, it is asserted three times: at table-generation time
(`tools/gen_unicode_bert_norm.py` refuses to emit a table where ASCII maps to
anything but plain `A-Z` lowering), over all 128 ASCII codepoints in
`tests/test_bert_norm.cpp`, and as a gate in the parity harness above.

**Why the table is generated, not hand-written.** Two traps, both silent:

- `Ø`, `Ł`, `Đ`, `ß`, `ı`, `ﬁ` have **no canonical decomposition**, so HF
  keeps them exactly as they are. `Łódź` → `łodz`, *not* `lodz`. A
  "strip the diacritic" table gets every one of these wrong.
- The **Rust** normalizer is the authority — every affected model ships a
  `tokenizer.json`, so `BertTokenizerFast` runs it — and it disagrees with
  Python's `unicodedata` on **441** late-Unicode combining marks (NKO, Arabic,
  Bengali, Gujarati) which Python drops and Rust keeps. Generating the table
  from `unicodedata`, the obvious approach, would have shipped 441 divergences
  from the thing users actually run. It also does not apply `Final_Sigma`:
  `ΟΔΟΣ` → `οδοσ` under the fast tokenizer, `οδος` under the slow one.

**Who was affected:** only the 30k uncased WordPiece models. Every
multilingual model in this table uses a SentencePiece/XLM-R tokenizer (250k
vocab) which handles accented text natively — those were never affected.
`LaBSE` is a **cased** WordPiece model whose `tokenizer.json` declares
`lowercase: false`, so it correctly does not strip accents; the fix is
conditioned on the same `do_lower_case` the runtime already infers from the
vocabulary, so LaBSE is untouched.

An earlier note here said a fix was blocked on the converter recording
`strip_accents`. That was wrong: HF resolves it as
`strip_accents.unwrap_or(lowercase)`, so `do_lower_case` alone determines it
and no metadata change or GGUF re-conversion is needed.

Full comparison: `tests/wordpiece_hf_parity.py` (Latin/parity),
`tests/wordpiece_cjk_parity.py` (CJK).

**Recommendation for Japanese retrieval:** `granite-embedding-107m`
(best measured, 107M) or `bge-m3` (strong, 8k context, also does sparse and
ColBERT retrieval). Both are registry aliases — `crispembed -m
granite-embedding-107m --json "テキスト"` auto-downloads and runs. The alias
default quant (`iq4_xs`) was re-verified end-to-end and reproduces the
`q4_k_m` row above to 3 decimals (0.966/0.430 paraphrase, 0.940 cross-lingual).

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

## Reranker models — Japanese (verified 2026-08-08)

Rerankers had never been tested on non-English text. All three shipped
rerankers now verified on Japanese (2 fixture queries, scored against a
relevant vs irrelevant JA document):

| Model (quant) | JA cats query | JA cooking query | EN control | Tokenizer |
|---|--:|--:|--:|---|
| `bge-reranker-v2-m3` (q4_k) | +17.13 | +10.07 | +10.62 | SentencePiece 250k |
| `jina-reranker-v2-base-multilingual` (q4_k) | +4.57 | +2.14 | +4.09 | SentencePiece 250k |
| `bge-reranker-base` (q4_k) | +14.75 | +14.84 | +13.65 | SentencePiece 250k |

Columns show the score gap (relevant minus irrelevant document score); positive
= correct ranking. All three pass.

**Evidence caveat:** unlike the embedder table above, these results have **no
negative control**. The embedder table includes English-only models that
demonstrably fail on Japanese (bit-identical vectors, cross-lingual FAIL),
proving the test can distinguish working from broken. Here, all three
rerankers use a 250k SentencePiece/XLM-R vocabulary — there is no
English-only 30k-WordPiece reranker in the registry to serve as a control.
`bge-reranker-base` was intended as one, but turned out to be multilingual at
the tokenizer level. This means we know Japanese reranking produces correct
orderings, but we cannot demonstrate what a failure looks like on the same
harness. The embedder table carries stronger evidence.

The absence of an English-only reranker also means the "wrong model for the
language" trap that affects English-only embedders does NOT apply to any
shipped reranker — all have multilingual tokenization.

The `CRISPEMBED_WARN_UNK` warning would not fire on any reranker fed Japanese,
because the SentencePiece tokenizer does not produce `[UNK]` tokens for Japanese
(it has byte-fallback). This is correct behaviour — the warning is only useful
when the tokenizer is genuinely out-of-vocabulary.

Fixture: `tests/reranker_language_eval.py`. Reproduce:
```bash
python tests/reranker_language_eval.py ./build/crispembed <models-dir> out.json
```

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
