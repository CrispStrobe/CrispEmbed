# CrispEmbed v0.17.8

A regression-fix and tokenizer-parity release — 15 commits since v0.17.7.
The headline: **v0.17.7 shipped the PP-OCRv6 pipeline ~10–18% slower on
Metal/CPU boxes** (issue #45, reported with an exemplary A/B corpus by
@niksedk). It is root-caused and fixed here, and the same pipeline now runs
**~25–38% faster than v0.17.6**, byte-identical output. The second theme:
the WordPiece and SentencePiece tokenizers now match HuggingFace exactly on
accented, CJK and typographic-punctuation text — which **changes embeddings
for such text** (toward the reference), so read *Changed defaults* if you
pin embedding vectors.

Every performance number below is a measurement recorded in
`PERFORMANCE.md` or the corresponding commit, with its fixture and box.

---

## Changed defaults

| Default | Was | Now | Evidence |
|---|---|---|---|
| CLI / server thread count when `-t` is absent | 1 | **min(4, cores)** | issue #45: the v0.17.7 n_threads audit pinned the det CPU graph from ggml's implicit 4 threads to the CLI's 1; `-t 4` on unmodified v0.17.7 fully restores v0.17.6 (M1 + reporter's M4) |
| C API `n_threads = 0` | clamped to 1 thread | **auto (min(4, cores))**, as the header always documented | applied at all 36 `extern "C"` init boundaries; un-cripples Flutter (defaults `nThreads = 0` across ~20 classes) and Rust callers |
| PP-OCRv6 medium/large recognizer scalar convs | reference loop | **R6 im2col + mk micro-kernel** (`CRISPEMBED_CONV2D_MK=0` restores) | the medium rec has no GGML graph, every crop runs these convs; decode byte-identical on EN/DE/FR/JA/ZH strips |
| WordPiece HF parity (`CRISPEMBED_WORDPIECE_HF_{NORM,PRETOK,UNK}`) | historical byte-level behavior | **on** | token ids 4/24 → 24/24 and 35/35 exact vs HF on 3 models; accented e2e cos vs ONNX 0.6466 → 1.000000, ASCII bit-identical |
| SentencePiece `nmt_nfkc` charsmap (`CRISPEMBED_SPM_HF_NORM`, embedding path) | not implemented | **on** | 4837-codepoint table generated from HF's own `Precompiled` component, byte-identical across all six shipped multilingual embedders; charsmap-material cos vs ONNX 0.9070 → 0.9876 |
| SigLIP text canonicalization (`clip_text`) | none | **lowercase + strip punctuation + collapse whitespace + charsmap**, per SigLIP | `"A photo of a CAT, running fast!"` cos vs HF 0.810971 → 0.999503; the all-lowercase stock fixture unchanged |
| FireRedPunc punctuation tokenizer | ASCII-whitespace-only split | **HF BertTokenizer-exact** | upstream uses plain `BertTokenizer`; token ids 2/9 → 9/9 exact, the duplicate label-alignment loop removed per the upstream blueprint |

Each keeps an env override with absolute precedence in both directions.

---

## Fixed: v0.17.7 ran the PP-OCRv6 Metal/CPU pipeline ~10–18% slower (issue #45)

v0.17.7's (correct) audit made `ppocrv6_det`/`ppocrv6_ocr`/`pplcnet` honor
the thread count they are handed — but the CLI's blanket default was
`-t 1`, so the detector's persistent CPU ggml graph (the production det
path on non-CUDA boxes) dropped from ggml's implicit 4 threads to 1. The
ggml pin bump and the sched replay change were exonerated: `-t 4` on the
unmodified v0.17.7 binary fully restores v0.17.6 timing on both an M1 and
the reporter's M4.

With the thread default fixed **and** the recognizer's mk adoption (the
reporter's `CRISPEMBED_CONV2D_MK=1` observation led straight to the
recognizer's un-adopted scalar convs):

- M1, subtitle-style strips, fresh process per run: 1920×200 two-liner
  **3.98 → 2.48 s (−38%)**, 64 px strip **1.97 → 1.28 s (−35%)** vs v0.17.6.
- Reporter's M4 corpus (30 invocations): **−24.8% vs v0.17.6** (v0.17.7 had
  been +9.9%).
- Decoded output byte-identical across every arm, fixture and language.

Pinned to v0.17.7? `-t 4` plus `CRISPEMBED_CONV2D_MK=1` gets you within a
few percent of this release.

## Fixed: embedding tokenizers diverged from HuggingFace on accented and CJK text

Three independent WordPiece defects, each able to change the token
sequence: a per-byte `tolower` instead of HF's `BertNormalizer`
(`café` → `caf`+`[UNK]`), a per-byte `isspace/ispunct` split instead of
`BertPreTokenizer` (CJK glued into one `[UNK]`, `“hello”` mis-split), and
kept-prefix output where HF emits one whole-word `[UNK]`. Ordinary English
with typographic punctuation was affected too (one sentence measured at
cos 0.430 vs the reference before the fix). The normalization tables are
generated from HF's own **Rust** normalizer — `unicodedata` disagrees with
it on 441 late-Unicode combining marks and would have shipped divergences.

SentencePiece embedders gained the `nmt_nfkc` precompiled charsmap
(fullwidth forms, U+3000, ligatures, `…` → `...`, ①→1, ㎏→kg) — routine in
JA/ZH text, i.e. exactly the retrieval case the multilingual embedders are
recommended for. New parity harnesses drive the real GGUFs through the
public C API (`tests/embed_tokenizer_parity.py`, `tests/dump_token_ids.cpp`,
`tests/clip_text_tokenizer_parity.py`).

## Added: server `--ocr-engine` / `--ocr-cls`

The server's flat pipeline det slot hard-codes the DBNet loader, so the
ppocrv6 pipeline (and every other CLI `--ocr-engine` lane) was unreachable
from the server. It now mirrors the CLI's single-stage builder — same
engine names, same per-engine registry defaults — so
`crispembed-server --ocr-pipeline --ocr-engine ppocrv6` alone is a valid
startup. A det+rec-only server invocation is no longer rejected at the
startup gate, and the server intentionally skips the CLI's one-shot
CPU-pick heuristic: a warm server amortises the recognizer's Metal init
across requests. Validated: `/ocr/pipeline` decodes the issue-#45 fixtures
byte-equal to the CLI.

---

Thanks to @niksedk (Subtitle Edit) for the interleaved A/B corpus that made
issue #45 a same-day root-cause, and for re-validating the fix on M4.
