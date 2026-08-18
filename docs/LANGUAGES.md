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

### WordPiece tokenizer divergence from HuggingFace — FIXED

**Status: fixed, and it was three separate defects, not one.** Chasing the
accented-Latin report surfaced two more, each independently capable of
producing a different token sequence than HF:

| # | defect | gate (default on) | what it broke |
|---|---|---|---|
| 1 | per-**byte** `tolower`, so `BertNormalizer`'s NFD accent strip never ran | `CRISPEMBED_WORDPIECE_HF_NORM` | `café` → `caf`+`[UNK]` |
| 2 | per-**byte** `isspace`/`ispunct` split instead of HF's `BertPreTokenizer` | `CRISPEMBED_WORDPIECE_HF_PRETOK` | CJK glued into one `[UNK]`; `“hello”` → `“`+`##hell`+`##o`+`##”` |
| 3 | unsegmentable words kept their matched prefix instead of collapsing to one `[UNK]` | `CRISPEMBED_WORDPIECE_HF_UNK` | `catソファ` → `cat`+`[UNK]` where HF gives `[UNK]` |

Setting any gate to `0` restores that stage's historical behavior; all three
to `0` is bit-exact with what shipped before.

**This was never only a European-language problem.** Defect 2 mangles ordinary
English with typographic punctuation — `He said “hello” — then left…` embedded
at cosine **0.430** to its own reference.

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

**Token-id parity vs HF**, 35 sentences across 9 sections
(`tests/wordpiece_hf_parity.py`, driving the shipping C++ tokenizer against
HF's fast tokenizer on the real vocab). One fix per arm, so every column is
attributable to a single change:

| model | historical | +accents | +split | +wholeUNK |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 4/35 | 25/35 | 34/35 | **35/35** |
| `all-mpnet-base-v2` | 4/35 | 25/35 | 34/35 | **35/35** |
| `LaBSE` (cased control) | 25/35 | 25/35 | 35/35 | **35/35** |

`[UNK]` count on the two uncased models went 80 → 0 on the Latin sections.
LaBSE is the control that matters: it declares `lowercase: false`, so the
accent arm leaves it **completely unchanged** (25 → 25) — the exact breakage
the old note in this file predicted a fix would cause.

**End-to-end embeddings** (`tests/embed_accent_parity.py`, real CLI + real
`all-MiniLM-L6-v2` f32 GGUF, reference = the model's own ONNX export under ONNX
Runtime, mean-pooled and L2-normalized). Token-id parity is not the acceptance
test; this is:

| section | cos vs reference before | after |
|---|---|---|
| ASCII (3 sentences) | 1.000000 | 1.000000, **bit-identical to before** |
| accented (5 sentences) | 0.646574 mean | **1.000000** |
| CJK + Unicode punctuation (4) | 0.590807 mean | **1.000000** |

Per sentence the old path ran 0.487 (French), 0.566 (Spanish), 0.574
(Portuguese), 0.731 (German), 0.875 (Norwegian), 0.461 (Japanese) and 0.430
(English with typographic quotes and an em-dash). A sentence embedded at cosine
0.43–0.49 to its own reference is, for retrieval purposes, a different vector.
All are now exact.

**Why this was safe to turn on by default.** All three fixes are no-ops on
printable ASCII, so no shipped English-on-ASCII embedding moves. The strength
of that guarantee differs per fix, and it is worth being exact about which:

- **Fix 1 is a hard invariant.** The normalization is *exactly* `std::tolower`
  over the whole of ASCII, asserted at table-generation time
  (`tools/gen_unicode_bert_norm.py` refuses to emit a table where ASCII maps to
  anything but plain `A-Z` lowering) and over all 128 ASCII codepoints in
  `tests/test_bert_norm.cpp`.
- **Fix 2 is a hard invariant with one deliberate exception.** `isspace`
  matches `CAT_WS` and `ispunct` matches `CAT_P` across `0x20`–`0x7E`, asserted
  per codepoint, plus a direct comparison of the two splitters on ASCII strings
  built to hit punctuation runs, leading/trailing separators and empty input.
  The exception: raw **C0 control bytes and DEL**, which HF's `clean_text`
  drops and the historical loop glued into the surrounding word (making that
  word `[UNK]`). That divergence is itself a fix and has its own test.
- **Fix 3 is empirical, not an invariant.** A word only changes if it is
  unsegmentable, and a 30k WordPiece vocab contains every printable ASCII
  character as its own token, so no ASCII word ever is. That holds for every
  vocab tested rather than being guaranteed by construction, which is why the
  ASCII gate in the parity harness checks it against real vocabularies instead
  of assuming it.

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

### The multilingual SentencePiece models had their own divergence — FIXED

**Status: fixed.** This section used to say the multilingual models were "not
affected". That was an argument about *accent stripping* and it is still true
for accents — but their token ids had never been compared against HF. They
were, and they diverged.

Every multilingual embedder here is an XLM-R-family **Unigram** model whose
`tokenizer.json` declares a `Precompiled` normalizer (an `nmt_nfkc`
precompiled charsmap). CrispEmbed's SentencePiece path implements **no
normalizer at all** — `grep precompiled_charsmap` finds nothing in the
runtime, the converter, or any GGUF. Measured on `multilingual-e5-small`
(`tests/embed_tokenizer_parity.py`, real GGUF through the public C API):

| section | exact vs HF |
|---|---|
| ascii | 2/2 |
| accented | 4/4 |
| cjk | 3/3 |
| unicode punctuation | 1/2 |

The failure: `…` (U+2026) must normalize to `...`, one in-vocab token. We emit
three `<unk>`.

**Scope — 4837 codepoints**, and they are not exotic:

| input | HF | CrispEmbed |
|---|---|---|
| `…` `‥` | `...` `..` | `<unk>` ×3 |
| `Ａ` `ａ` `１` (fullwidth) | `A` `a` `1` | unnormalized |
| U+3000 ideographic space | plain space | unnormalized |
| `ﬁ` `ﬂ` | `fi` `fl` | unnormalized |
| `①` `Ⅳ` `㎏` `㈱` | `1` `IV` `kg` `(株)` | unnormalized |

Typographic quotes and dashes are *not* touched by this charsmap, which is why
Latin text scores 4/4 above. **Fullwidth forms and U+3000 are routine in
Japanese and Chinese text**, so this matters most for exactly the retrieval
case this document recommends these models for.

**The fix.** `core/spm_norm.h` + a table generated from HF's own `Precompiled`
component (`tools/gen_unicode_spm_norm.py`, 4837 rows), applied before
segmentation — HF's order is `Precompiled` *then* `Replace(" " → "▁")`, and it
matters, because the charsmap turns U+3000 into a plain space that must then
become a word boundary like any other. Gated `CRISPEMBED_SPM_HF_NORM`
(default on for the embedding path; `=0` restores the historical path).

Token-id parity, two arms × two models
(`tests/embed_tokenizer_parity.py`, 16 sentences):

| model | historical | hf-norm |
|---|---|---|
| `multilingual-e5-small` | 10/16 | **16/16** |
| `arctic-embed-m-v2` | 10/16 | **16/16** |

with the `charsmap` section going **0/5 → 5/5** and ASCII byte-identical
between arms.

End-to-end embeddings vs e5's own ONNX export (`tests/embed_accent_parity.py`
with `PARITY_GATES=CRISPEMBED_SPM_HF_NORM`):

| section | before | after |
|---|---|---|
| ASCII | 0.975390 | 0.975390, **bit-identical** |
| accented | 0.988305 | 0.988305 (unchanged — accents are not in this charsmap) |
| CJK + unicode punctuation | 0.975832 | **0.982032** |
| charsmap material | 0.907015 | **0.987556** |

These cosines top out around 0.98 rather than 1.0 because the GGUF is `q8_0`
and the reference is `f32` — that is the quantization floor, not a tokenizer
gap. What matters is that the charsmap section sat at **0.907, far below the
floor**, and now sits at it. Token-id parity is the exact gate; this is the
magnitude.

**Scoped deliberately.** `set_hf_normalize()` is enabled only on the embedding
path, which is the one measured. `clip_text_embed` (SigLIP) carries the *same*
charsmap but wraps it in `Lowercase + Strip` steps we do not implement, so
flipping it there needs its own A/B first; `gliner`'s shipped GGUF is an LFM2
**BPE** model with no normalizer and never reaches this code. An earlier note
here claimed those two "declare different normalizers" and that the converter
would have to record each model's choice — that was an inference, and checking
it showed the charsmap is in fact shared.

Harness: `tests/embed_tokenizer_parity.py` + `tests/dump_token_ids.cpp` (dumps
the ids the shipping runtime produces for any tokenizer family; SPM/BPE parity
cannot be checked from a vocab file the way WordPiece can, because merges,
charsmap and pre-tokenizer selection live in the GGUF).

**Who was affected by the WordPiece defects:** defect 1 hit only the 30k
**uncased** WordPiece models.
Defects 2 and 3 hit **every** WordPiece model, cased included — LaBSE was at
25/35 against HF before this work purely from the split stage. The
multilingual models use a SentencePiece/XLM-R tokenizer (250k vocab), a
different code path, so none of the three WordPiece defects reached them —
but see the open SentencePiece divergence above, which does.

`LaBSE` declares `lowercase: false`, so it correctly does not strip accents;
the accent fix is conditioned on the same `do_lower_case` the runtime already
infers from the vocabulary, and the parity table above confirms that arm leaves
LaBSE byte-identical.

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

## Embedding models — Arabic + Korean (E3, verified 2026-08-17)

Extended the JA matrix to Arabic (RTL script, different normalization) and
Korean (Hangul, agglutinative morphology). Same three-check structure, same
negative controls.

### Arabic

| Model (quant tested) | AR paraphrase | AR cross-lingual | Verdict |
|---|--:|--:|---|
| `granite-embedding-107m-multilingual` (q4_k) | 0.812 vs 0.690 | 0.676 vs 0.621 | pass, narrow margins |
| `paraphrase-multilingual-MiniLM-L12-v2` (q8_0) | 0.730 vs -0.026 | 0.984 vs -0.060 | **best cross-lingual** |
| `multilingual-e5-large` (q8_0, no prefix) | 0.976 vs 0.861 | 0.875 vs 0.759 | strong (narrow margin) |
| `multilingual-e5-base` (q8_0, no prefix) | 0.961 vs 0.807 | 0.894 vs 0.737 | strong (narrow margin) |
| `multilingual-e5-small` (q8_0, no prefix) | 0.953 vs 0.862 | 0.880 vs 0.790 | pass (narrowest) |
| `granite-embedding-278m-multilingual` (q8_0) | 0.807 vs 0.659 | 0.658 vs 0.595 | pass, narrow |
| `jina-v5-small` (q4_k) | 0.715 vs 0.098 | 0.776 vs 0.095 | **strong** |
| `jina-v5-nano` (q4_k) | 0.707 vs 0.052 | 0.774 vs 0.070 | strong |
| `all-MiniLM-L6-v2` (q4_k) | 0.774 vs 0.728 | 0.027 vs 0.154 | **DO NOT USE for AR** |
| `all-mpnet-base-v2` (q8_0) | 0.850 vs 0.801 | -0.031 vs 0.044 | **DO NOT USE for AR** |

**Key finding — Arabic margins are narrower than Japanese across the board.**
The granite-107m AR margin is +0.12 (vs JA +0.53), and several models cluster
in the 0.09–0.15 range. This does not mean Arabic is broken — all multilingual
models pass all three checks — but the separation is thinner, so a retrieval
system on Arabic may need a stronger model (e5-large or paraphrase-multi) than
it would for Japanese.

**Negative controls work:** both EN-only models show near-chance cross-lingual
scores (negative margins) and AR paraphrase margins under 0.05.

### Korean

| Model (quant tested) | KO paraphrase | KO cross-lingual | Verdict |
|---|--:|--:|---|
| `granite-embedding-107m-multilingual` (q4_k) | 0.890 vs 0.526 | 0.801 vs 0.510 | **strong** |
| `paraphrase-multilingual-MiniLM-L12-v2` (q8_0) | 0.941 vs -0.050 | 0.748 vs -0.044 | **strongest separation** |
| `multilingual-e5-large` (q8_0, no prefix) | 0.988 vs 0.822 | 0.883 vs 0.740 | strong (narrow margin) |
| `multilingual-e5-base` (q8_0, no prefix) | 0.973 vs 0.791 | 0.882 vs 0.720 | strong (narrow margin) |
| `multilingual-e5-small` (q8_0, no prefix) | 0.967 vs 0.803 | 0.840 vs 0.726 | pass |
| `granite-embedding-278m-multilingual` (q8_0) | 0.890 vs 0.465 | 0.799 vs 0.458 | **strong** |
| `jina-v5-small` (q4_k) | 0.938 vs 0.042 | 0.830 vs 0.039 | **strong, sharp** |
| `jina-v5-nano` (q4_k) | 0.946 vs 0.037 | 0.853 vs 0.046 | **strong, sharp** |
| `all-MiniLM-L6-v2` (q4_k) | 1.000 vs 0.990 | 0.049 vs 0.167 | **DO NOT USE for KO** |
| `all-mpnet-base-v2` (q8_0) | 1.000 vs 0.992 | -0.015 vs 0.141 | **DO NOT USE for KO** |

**Key finding — Korean tracks close to Japanese** for most models, with strong
margins. The jina models show particularly sharp separation (0.04 unrelated
cosine). However, the EN-only controls are **even more degenerate on Korean
than Japanese**: unrelated-KO cosine is 0.99 (vs JA 0.33), meaning the
30k WordPiece tokenizer collapses ALL Korean input to near-identical vectors.
The `CRISPEMBED_WARN_UNK` warning fires on these models for Korean input.

**Cross-language comparison (margin on monolingual paraphrase check):**

| Model | JA margin | AR margin | KO margin |
|---|--:|--:|--:|
| granite-107m-multi | +0.53 | +0.12 | +0.36 |
| paraphrase-multi-MiniLM | +1.04 | +0.76 | +0.99 |
| e5-large (no prefix) | +0.18 | +0.12 | +0.17 |
| jina-v5-small | +0.88 | +0.62 | +0.90 |

Arabic is consistently the weakest language; Korean is close to Japanese.

## Reranker models — JA + AR + KO (verified 2026-08-08, extended 2026-08-17)

All three shipped rerankers verified on Japanese (E2, 2026-08-08) and extended
to Arabic + Korean (E3, 2026-08-17). Each language has 2 fixture queries,
scored against a relevant vs irrelevant document in that language.

| Model (quant) | JA cats | JA cooking | AR cats | AR cooking | KO cats | KO cooking | EN ctrl | Tokenizer |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `bge-reranker-v2-m3` (q4_k) | +17.13 | +10.07 | +10.90 | +9.64 | +16.17 | +9.04 | +10.62 | SentencePiece 250k |
| `jina-reranker-v2-base-multilingual` (q4_k) | +4.57 | +2.14 | +3.56 | +1.32 | +3.56 | +1.76 | +4.09 | SentencePiece 250k |
| `bge-reranker-base` (q4_k) | +14.75 | +14.84 | +11.84 | +7.93 | +11.52 | +7.92 | +13.65 | SentencePiece 250k |

Columns show the score gap (relevant minus irrelevant document score); positive
= correct ranking. All three rerankers pass all three languages.

**AR/KO findings:** Arabic and Korean gaps are slightly narrower than Japanese
on `bge-reranker-v2-m3` (AR cats +10.9 vs JA +17.1) but still large. The
jina reranker shows a narrower Arabic cooking gap (+1.32) — the smallest gap
in the table, but still clearly positive. Korean tracks close to Arabic.

**Evidence caveat (unchanged):** all three rerankers use 250k SentencePiece/
XLM-R vocabularies. There is no English-only 30k-WordPiece reranker in the
registry to serve as a negative control. `bge-reranker-base` was intended as
one but turned out to be multilingual at the tokenizer level. The embedder
table carries stronger evidence because it includes genuine EN-only controls.

The `CRISPEMBED_WARN_UNK` warning would not fire on any reranker, for any
language — SentencePiece has byte-fallback, so no `[UNK]` tokens are produced.
This is correct behaviour.

Fixture: `tests/reranker_language_eval.py`. Reproduce:
```bash
python tests/reranker_language_eval.py ./build/crispembed <models-dir> out.json
```

## Vocabulary scanner vs measured eval — cross-reference (E7, 2026-08-17)

`tools/scan_model_languages.py` counts script-range code points in a GGUF's
tokenizer vocabulary. For **OCR recognizers** this is reliable (the dictionary
gates emittable characters). For **embedders** it is not — the E3 results
prove it is unreliable in **both directions**:

| Model | Tokenizer | kana | hangul | arabic | JA eval | KO eval | AR eval |
|---|---|---:|---:|---:|---|---|---|
| granite-107m-multi | SP 250k | 9729 | 9917 | 61802 | PASS strong | PASS strong | PASS narrow |
| paraphrase-multi | SP 250k | 9729 | 9917 | 61802 | PASS best | PASS strong | PASS best XL |
| e5-small | SP 250k | 9729 | 9917 | 61802 | PASS | PASS | PASS |
| jina-v5-small | BPE 152k | **0** | **0** | **0** | PASS strong | PASS strong | PASS strong |
| jina-v5-nano | BPE 128k | **0** | **0** | **0** | PASS strong | PASS strong | PASS strong |
| all-MiniLM-L6-v2 | WP 30k | 188 | 70 | 91 | BROKEN | BROKEN | BROKEN |
| all-mpnet-base-v2 | WP 30k | 188 | 70 | 91 | BROKEN | BROKEN | BROKEN |

**False positive (non-zero but broken):** all-MiniLM-L6-v2 has kana=188,
hangul=70, arabic=91 in its vocabulary, yet produces degenerate embeddings
for all three scripts (bit-identical JA vectors, KO unrelated cos=0.99).

**False negative (zero but works):** jina-v5-small/nano scan as latin-only
(kana=0, hangul=0, arabic=0), yet pass all three language checks with strong
margins (JA paraphrase +0.88, KO +0.90, AR +0.62). Root cause: BPE
tokenizers encode non-Latin scripts as byte sequences — the script characters
don't appear as code points in the token strings, but the model handles them
via byte-level encoding.

**E7 decision: do NOT surface scanner output as a `--list-models` column for
embedders or rerankers.** The scanner remains valid and useful for OCR
recognizers (where the dictionary genuinely gates output characters). For
embedders, use `tests/embed_language_eval.py` — the measured eval tables
above are authoritative.

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
