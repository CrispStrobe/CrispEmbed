# Unlimited-OCR-2 published-model trial (Kaggle CPU)

Kernel `crispembed-unlimited-ocr2-trial` v1 at commit `267e47433da1e83a6cdab982e83b27df1d5bb4fd`; Hub revision `e3af0c0c4bfa241c161db64ca36f3c63d3ca8378`; Intel(R) Xeon(R) CPU @ 2.20GHz x4, 31.3 GB RAM; build 256 s; elapsed 4.81 h.

CER is recomputed offline from the recorded texts with the v2 normalizer (the in-kernel v1 normalizer left the native
CLI's inline `label[x0, y0, x1, y1]` prefixes in and is kept as `cer_harness_v1`). Layout labels, boxes and HTML tags are
stripped before scoring; table cells are scored as space-separated text.

## Alias resolution (empty cache, then cache hit)

| alias | fresh exit / s | SHA-256 verified | cached exit / s | re-downloaded | file sha256 == pin |
|---|---|---|---|---|---|
| `unlimited-ocr2-q8` | 0 / 291.77 | True | 0 / 240.19 | False | True |
| `unlimited-ocr2` | 0 / 384.22 | True | 0 / 280.47 | False | True |

Probe runs used the default (exact) decoder on 8 tokens, so their times are dominated by load + exact prefill.

## OCR runs

| page | model | tiled | decoder | wall s | peak RSS MB | IDs | CER | IDs == original |
|---|---|---|---|---|---|---|---|---|
| paragraphs | Q8_0 | False | fast | 46.18 | 6869 | 146 | 0.0 | False |
| paragraphs | Q8_0 | True | fast | 175.26 | 6869 | 146 | 0.0 | True |
| table | Q8_0 | False | fast | 46.27 | 6868 | 138 | 0.1257 | False |
| table | Q8_0 | True | fast | 170.7 | 6869 | 138 | 0.1257 | False |
| receipt | Q8_0 | False | fast | 57.31 | 6869 | 306 | 0.0 | False |
| receipt | Q8_0 | True | fast | 250.41 | 6869 | 306 | 0.0 | False |
| german | Q8_0 | False | fast | 48.52 | 6868 | 162 | 0.0 | False |
| german | Q8_0 | True | fast | 174.05 | 6869 | 164 | 0.0 | True |
| two-column | Q8_0 | False | fast | 42.15 | 6869 | 125 | 0.0 |  |
| two-column | Q8_0 | True | fast | 172.52 | 6869 | 131 | 0.0 |  |
| a4-dense | Q8_0 | False | fast | 66.26 | 6869 | 421 | 0.0 |  |
| a4-dense | Q8_0 | True | fast | 169.22 | 6869 | 421 | 0.0 |  |
| scan-sim | Q8_0 | False | fast | 57.85 | 6869 | 158 | 0.0249 |  |
| scan-sim | Q8_0 | True | fast | 117.31 | 6868 | 160 | 0.0277 |  |
| paragraphs | F16 | False | fast | 77.44 | 12821 | 146 | 0.0 | False |
| paragraphs | F16 | True | fast | 297.12 | 12821 | 146 | 0.0 | True |
| table | F16 | False | fast | 79.68 | 12821 | 138 | 0.1257 | False |
| table | F16 | True | fast | 297.99 | 12821 | 138 | 0.1257 | True |
| receipt | F16 | False | fast | 87.97 | 12821 | 306 | 0.0 | False |
| receipt | F16 | True | fast | 380.68 | 12821 | 306 | 0.0 | True |
| german | F16 | False | fast | 87.01 | 12821 | 162 | 0.0 | False |
| german | F16 | True | fast | 317.04 | 12821 | 164 | 0.0 | True |
| two-column | F16 | False | fast | 68.8 | 12821 | 125 | 0.0 |  |
| two-column | F16 | True | fast | 298.76 | 12821 | 131 | 0.0 |  |
| a4-dense | F16 | False | fast | 110.75 | 12821 | 421 | 0.0 |  |
| a4-dense | F16 | True | fast | 225.93 | 12821 | 421 | 0.0 |  |
| scan-sim | F16 | False | fast | 74.41 | 12821 | 158 | 0.0249 |  |
| scan-sim | F16 | True | fast | 189.55 | 12821 | 158 | 0.0249 |  |
| receipt | Q8_0 | True | exact | 2018.12 | 7359 | 306 | 0.0 | False |
| receipt | F16 | True | exact | 2453.34 | 12821 | 306 | 0.0 | True |
| paragraphs | Q8_0 | True | exact | 1456.87 | 7306 | 146 | 0.0 | True |
| paragraphs | F16 | True | exact | 1713.89 | 10495 | 146 | 0.0 | True |
| german | Q8_0 | True | exact | 1499.44 | 7306 | 164 | 0.0 | True |
| german | F16 | True | exact | 1757.05 | 10267 | 164 | 0.0 | True |

## Original PyTorch reference (F32, CPU)

| page | seconds | CER | first native-tiled-F16-fast ID difference |
|---|---|---|---|
| paragraphs | 111.9 | 0.0 | None |
| table | 117.7 | 0.0 | None |
| receipt | 161.1 | 0.0 | None |
| german | 113.6 | 0.0 | None |

## Findings

- Aliases: both download, verify SHA-256 against the pinned value, and reuse the cache without re-downloading. No alias defect.
- Tiled F16 (fast and exact) reproduces the original token IDs exactly on all four reference pages (paragraphs, table, receipt, german).
  Q8_0 tiled matches on paragraphs and german; on table it diverges at ID 7 and on receipt at ID 131 (box coordinates), with identical text.
- Untiled runs diverge from the original at ID 4-13 (box coordinates only: the original tiles 1024x768 pages); text and CER are unchanged.
- Fast decoder == exact decoder token-for-token on every paired run; 6-20x faster (e.g. receipt F16 tiled 381 s vs 2453 s).
- CER 0 on paragraphs, receipt, german, two-column (correct reading order) and A4-dense, both precisions, tiled and untiled.
- **Native CLI table output loses cell structure**: IDs equal the original, but stdout prints `<table>ProductRegionUnits...</table>` without
  `<tr>/<td>` (the original decodes them). The decoder drops these special tokens. Table CER 0.126 is entirely missing separators.
- **Native CLI prints raw layout prefixes** (`text[47, 92, 644, 128]`, `title<LEVEL1>[...]`) inline once <BLOCK>/<FORK> are stripped;
  a user gets neither clean text nor parseable markup.
- scan-sim CER 0.025-0.028 is a harness flaw: the rendered lines run past the right page edge, so the image itself is clipped
  ("Erdgesc", "Fre"); the model transcribes what is visible.
- Peak RSS: Q8_0 6.9-7.4 GB, F16 10.3-12.8 GB. Tiling costs 2-4x wall time (fast decoder) and gives no CER gain on these pages.
- No public-domain scans were used (no pinned, license-verified source); scan-sim stands in.
