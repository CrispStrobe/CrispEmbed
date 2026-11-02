# OCR on real scans (crispembed-ocr-real-scans)

Shards: native-uocr2 @ `2acee0b1c920` Intel(R) Xeon(R) CPU @ 2.20GHz x4 31.3 GB, 2.88 h; ref-uocr2 @ `4daea2037135` AMD EPYC 7B12 x4 31.3 GB, 1.42 h; native-minicpm @ `2acee0b1c920` AMD EPYC 7B12 x4 31.3 GB, 8.97 h; ref-minicpm @ `2acee0b1c920` Intel(R) Xeon(R) CPU @ 2.20GHz x4 31.3 GB, 1.43 h

Metrics after the normalisation in `score.py` (markup stripped). CER / CER without spaces / WER / word recall; wall s; peak RSS MB. `-` = not run (time budget) or failed (see JSON).

| page | uocr2-f16-fast-tiled | uocr2-original-f32 | minicpm-f16 | minicpm-original-f32 |
|---|---|---|---|---|
| de-antiqua-kafka-schloss-211 | 0.006 / 0.007 / 0.011 / 0.989; 227.4 s; 12820.9 MB | 0.006 / 0.007 / 0.011 / 0.989; 111.7 s; 18008.9 MB | 0.007 / 0.007 / 0.022 / 0.984; 1080.53 s; 5051.0 MB | 0.007 / 0.007 / 0.022 / 0.984; 205.9 s; 8264.6 MB |
| de-degraded-freiligrath-46 | 0.041 / 0.049 / 0.127 / 0.873; 200.32 s; 12821.1 MB | 0.041 / 0.049 / 0.127 / 0.873; 97.8 s; 18008.9 MB | 0.039 / 0.047 / 0.139 / 0.861; 623.08 s; 5049.5 MB | 0.039 / 0.047 / 0.139 / 0.861; 144.2 s; 8264.6 MB |
| de-fraktur-grimm-319 | 0.022 / 0.027 / 0.060 / 0.940; 324.0 s; 12821.6 MB | 0.022 / 0.027 / 0.060 / 0.940; 125.3 s; 17972.4 MB | 0.027 / 0.033 / 0.107 / 0.893; 926.6 s; 5092.6 MB | 0.027 / 0.033 / 0.107 / 0.893; 202.5 s; 8264.6 MB |
| en-form-economic-journal-67 | 0.035 / 0.021 / 0.003 / 1.000; 339.85 s; 12821.2 MB | 0.020 / 0.021 / 0.003 / 1.000; 252.7 s; 18008.9 MB | 0.017 / 0.021 / 0.000 / 1.000; 1744.63 s; 5058.4 MB | 0.017 / 0.021 / 0.000 / 1.000; 273.0 s; 8264.6 MB |
| en-prose-constitution-13 | 0.046 / 0.047 / 0.029 / 1.000; 386.05 s; 12821.0 MB | 0.046 / 0.047 / 0.029 / 1.000; 135.7 s; 17972.4 MB | 0.011 / 0.012 / 0.000 / 1.000; 788.04 s; 5051.2 MB | 0.011 / 0.012 / 0.000 / 1.000; 188.8 s; 8132.4 MB |
| en-table-climatic-26 | 0.029 / 0.026 / 0.052 / 0.971; 581.51 s; 12821.0 MB | 0.027 / 0.026 / 0.033 / 0.998; 255.3 s; 18008.9 MB | 0.005 / 0.006 / 0.004 / 0.998; 1871.72 s; 5051.2 MB | 0.005 / 0.006 / 0.004 / 0.998; 274.8 s; 8264.6 MB |
| receipt-cord-test-29 | 0.996 / 0.995 / 0.983 / 0.018; 390.71 s; 12821.0 MB CAP | 0.996 / 0.995 / 0.983 / 0.018; 251.8 s; 18008.9 MB CAP | 0.035 / 0.028 / 0.018 / 1.000; 637.52 s; 5049.5 MB | 0.035 / 0.028 / 0.018 / 1.000; 142.6 s; 8264.6 MB |
| de-2col-meyers-b13-103 | 0.037 / 0.040 / 0.170 / 0.841; 1491.7 s; 12821.0 MB | 0.037 / 0.040 / 0.171 / 0.840; 736.4 s; 18008.9 MB | 0.047 / 0.053 / 0.227 / 0.778; 5432.3 s; 5049.2 MB | 0.047 / 0.053 / 0.227 / 0.778; 556.6 s; 8264.6 MB |
| de-antiqua-kafka-betrachtung-94 | 0.008 / 0.010 / 0.013 / 0.987; 431.93 s; 12820.6 MB | 0.008 / 0.010 / 0.013 / 0.987; 213.3 s; 18008.9 MB | 0.011 / 0.012 / 0.032 / 0.968; 1338.09 s; 5049.1 MB | 0.011 / 0.012 / 0.032 / 0.968; 193.7 s; 8484.1 MB |
| de-table-rgbl-1867-403 | 0.375 / 0.429 / 0.107 / 0.893; 550.59 s; 12821.2 MB | 0.375 / 0.429 / 0.107 / 0.893; 306.7 s; 18008.9 MB | 0.389 / 0.242 / 0.101 / 0.899; 3090.49 s; 5145.4 MB | 0.389 / 0.242 / 0.101 / 0.899; 409.6 s; 8484.1 MB |
| en-2col-eb1911-v04-246 | 0.061 / 0.062 / 0.071 / 0.999; 592.35 s; 12820.8 MB | 0.061 / 0.062 / 0.071 / 0.999; 321.2 s; 18008.9 MB | 0.132 / 0.133 / 0.148 / 0.918; 2367.06 s; 5145.3 MB | 0.132 / 0.133 / 0.149 / 0.917; 340.2 s; 8484.1 MB |
| en-report-amtrak-23 | 0.006 / 0.000 / 0.053 / 0.947; 465.46 s; 12821.2 MB | 0.000 / 0.000 / 0.000 / 1.000; 245.8 s; 18008.9 MB | 0.001 / 0.001 / 0.002 / 0.998; 1792.88 s; 5099.1 MB | 0.001 / 0.001 / 0.002 / 0.998; 287.9 s; 8484.1 MB |
| receipt-cord-test-16 | 0.137 / 0.037 / 0.698 / 0.302; 221.47 s; 12820.8 MB | 0.041 / 0.037 / 0.046 / 0.954; 89.1 s; 18008.9 MB | 0.041 / 0.037 / 0.070 / 0.930; 513.3 s; 5049.6 MB | 0.041 / 0.037 / 0.070 / 0.930; 133.4 s; 8484.1 MB |
| receipt-cord-test-4 | 7.585 / 7.400 / 5.774 / 0.000; 423.79 s; 12821.3 MB CAP | 7.585 / 7.400 / 5.774 / 0.000; 263.6 s; 18008.9 MB CAP | 0.061 / 0.064 / 0.161 / 0.903; 484.41 s; 5049.5 MB | 0.061 / 0.064 / 0.161 / 0.903; 131.9 s; 8484.1 MB |
| de-mixed-welzheim-188 | 0.015 / 0.016 / 0.059 / 0.949; 604.28 s; 12820.9 MB | 0.015 / 0.016 / 0.059 / 0.949; 274.2 s; 18008.9 MB | 0.026 / 0.029 / 0.118 / 0.905; 2455.45 s; 5143.5 MB | 0.026 / 0.029 / 0.118 / 0.905; 353.0 s; 8484.1 MB |
| en-2col-eb1911-v25-5 | 0.112 / 0.123 / 0.013 / 0.993; 768.0 s; 12820.6 MB | 0.112 / 0.123 / 0.015 / 0.991; 398.4 s; 18180.5 MB | 0.106 / 0.122 / 0.007 / 0.994; 2721.21 s; 5123.8 MB | 0.106 / 0.122 / 0.007 / 0.994; 380.9 s; 8484.1 MB |
| en-old-statutes-v1-772 | 0.043 / 0.042 / 0.038 / 0.993; 601.54 s; 12820.7 MB | 0.034 / 0.033 / 0.033 / 0.995; 293.9 s; 18008.9 MB | 0.041 / 0.040 / 0.038 / 0.974; 1896.76 s; 5049.7 MB | 0.041 / 0.040 / 0.038 / 0.974; 294.4 s; 8484.1 MB |
| en-prose-constitution-81 | 0.039 / 0.039 / 0.031 / 1.000; 430.58 s; 12820.8 MB | 0.040 / 0.039 / 0.031 / 1.000; 155.8 s; 18008.9 MB | 0.002 / 0.001 / 0.006 / 1.000; 814.54 s; 5051.4 MB | 0.002 / 0.001 / 0.006 / 1.000; 186.3 s; 8484.1 MB |
| receipt-cord-test-14 | 0.122 / 0.005 / 0.868 / 0.132; 388.02 s; 12821.0 MB | 0.004 / 0.005 / 0.019 / 0.981; 133.8 s; 18180.5 MB | 0.004 / 0.005 / 0.019 / 0.981; 662.24 s; 5049.5 MB | 0.004 / 0.005 / 0.019 / 0.981; 172.4 s; 8484.1 MB |
| receipt-cord-test-20 | 3.098 / 3.211 / 2.381 / 0.714; 271.79 s; 12443.8 MB | 3.196 / 3.276 / 2.500 / 0.714; 141.7 s; 18180.5 MB | 0.027 / 0.033 / 0.095 / 0.905; 483.72 s; 5049.3 MB | 0.027 / 0.033 / 0.095 / 0.905; 131.3 s; 8484.1 MB |

## Native vs original token agreement

| page | engine | ids exact | first diff | prefix agr. | native n | ref n | CER native | CER original |
|---|---|---|---|---|---|---|---|---|
| de-2col-meyers-b13-103 | Unlimited-OCR-2 | False | 2419 | 0.9084 | 2663 | 2663 | 0.0369 | 0.0371 |
| de-2col-meyers-b13-103 | MiniCPM-OCR | True | None | 1.0 | 2270 | 2270 | 0.047 | 0.047 |
| de-antiqua-kafka-betrachtung-94 | Unlimited-OCR-2 | False | 349 | 0.5232 | 667 | 667 | 0.008 | 0.008 |
| de-antiqua-kafka-betrachtung-94 | MiniCPM-OCR | True | None | 1.0 | 539 | 539 | 0.0106 | 0.0106 |
| de-antiqua-kafka-schloss-211 | Unlimited-OCR-2 | True | None | 1.0 | 356 | 356 | 0.0061 | 0.0061 |
| de-antiqua-kafka-schloss-211 | MiniCPM-OCR | True | None | 1.0 | 328 | 328 | 0.007 | 0.007 |
| de-degraded-freiligrath-46 | Unlimited-OCR-2 | True | None | 1.0 | 289 | 289 | 0.0412 | 0.0412 |
| de-degraded-freiligrath-46 | MiniCPM-OCR | True | None | 1.0 | 170 | 170 | 0.0392 | 0.0392 |
| de-fraktur-grimm-319 | Unlimited-OCR-2 | True | None | 1.0 | 292 | 292 | 0.0221 | 0.0221 |
| de-fraktur-grimm-319 | MiniCPM-OCR | True | None | 1.0 | 245 | 245 | 0.0271 | 0.0271 |
| de-mixed-welzheim-188 | Unlimited-OCR-2 | True | None | 1.0 | 936 | 936 | 0.0151 | 0.0151 |
| de-mixed-welzheim-188 | MiniCPM-OCR | True | None | 1.0 | 897 | 897 | 0.0258 | 0.0258 |
| de-table-rgbl-1867-403 | Unlimited-OCR-2 | False | 273 | 0.28 | 975 | 975 | 0.3751 | 0.3751 |
| de-table-rgbl-1867-403 | MiniCPM-OCR | True | None | 1.0 | 1172 | 1172 | 0.3893 | 0.3893 |
| en-2col-eb1911-v04-246 | Unlimited-OCR-2 | True | None | 1.0 | 1124 | 1124 | 0.0614 | 0.0614 |
| en-2col-eb1911-v04-246 | MiniCPM-OCR | False | 124 | 0.1473 | 842 | 842 | 0.1318 | 0.1321 |
| en-2col-eb1911-v25-5 | Unlimited-OCR-2 | False | 175 | 0.1338 | 1308 | 1308 | 0.1117 | 0.1117 |
| en-2col-eb1911-v25-5 | MiniCPM-OCR | True | None | 1.0 | 1024 | 1024 | 0.1064 | 0.1064 |
| en-form-economic-journal-67 | Unlimited-OCR-2 | True | None | 1.0 | 934 | 934 | 0.0354 | 0.0202 |
| en-form-economic-journal-67 | MiniCPM-OCR | False | 103 | 0.1643 | 627 | 627 | 0.0174 | 0.0174 |
| en-old-statutes-v1-772 | Unlimited-OCR-2 | False | 358 | 0.3862 | 917 | 927 | 0.0433 | 0.0339 |
| en-old-statutes-v1-772 | MiniCPM-OCR | True | None | 1.0 | 689 | 689 | 0.041 | 0.041 |
| en-prose-constitution-13 | Unlimited-OCR-2 | True | None | 1.0 | 282 | 282 | 0.046 | 0.046 |
| en-prose-constitution-13 | MiniCPM-OCR | True | None | 1.0 | 198 | 198 | 0.0107 | 0.0107 |
| en-prose-constitution-81 | Unlimited-OCR-2 | False | 10 | 0.0309 | 324 | 324 | 0.0389 | 0.0399 |
| en-prose-constitution-81 | MiniCPM-OCR | True | None | 1.0 | 206 | 206 | 0.0021 | 0.0021 |
| en-report-amtrak-23 | Unlimited-OCR-2 | True | None | 1.0 | 763 | 763 | 0.0062 | 0.0 |
| en-report-amtrak-23 | MiniCPM-OCR | True | None | 1.0 | 620 | 620 | 0.001 | 0.001 |
| en-table-climatic-26 | Unlimited-OCR-2 | True | None | 1.0 | 806 | 806 | 0.0285 | 0.0268 |
| en-table-climatic-26 | MiniCPM-OCR | True | None | 1.0 | 678 | 678 | 0.0049 | 0.0049 |
| receipt-cord-test-14 | Unlimited-OCR-2 | True | None | 1.0 | 255 | 255 | 0.1218 | 0.0042 |
| receipt-cord-test-14 | MiniCPM-OCR | True | None | 1.0 | 145 | 145 | 0.0042 | 0.0042 |
| receipt-cord-test-16 | Unlimited-OCR-2 | True | None | 1.0 | 230 | 230 | 0.1371 | 0.0406 |
| receipt-cord-test-16 | MiniCPM-OCR | True | None | 1.0 | 120 | 120 | 0.0406 | 0.0406 |
| receipt-cord-test-20 | Unlimited-OCR-2 | False | 33 | 0.0685 | 484 | 482 | 3.0978 | 3.1957 |
| receipt-cord-test-20 | MiniCPM-OCR | True | None | 1.0 | 107 | 107 | 0.0272 | 0.0272 |
| receipt-cord-test-29 | Unlimited-OCR-2 | True | None | 1.0 | 1024 | 1024 | 0.9961 | 0.9961 |
| receipt-cord-test-29 | MiniCPM-OCR | True | None | 1.0 | 174 | 174 | 0.0349 | 0.0349 |
| receipt-cord-test-4 | Unlimited-OCR-2 | False | 690 | 0.6738 | 1024 | 1024 | 7.585 | 7.585 |
| receipt-cord-test-4 | MiniCPM-OCR | True | None | 1.0 | 106 | 106 | 0.0612 | 0.0612 |

| engine | paired pages | IDs exact | normalised text equal | mean CER native | mean CER original |
|---|---|---|---|---|---|
| Unlimited-OCR-2 | 20 | 12 | 10 | 0.6407 | 0.6333 |
| MiniCPM-OCR | 20 | 18 | 19 | 0.0515 | 0.0515 |

## Findings

Arms: native Unlimited-OCR-2 F16 published alias `unlimited-ocr2`, fast decoder, gundam tiles; native MiniCPM-OCR F16 alias
`minicpm-ocr`, prompt `Text Recognition:`; the original PyTorch / Transformers models at F32 on CPU. Each arm ran in its own
Kaggle CPU session (4 vCPU, 31 GB). All 20 pages ran in every arm, with no time-budget skips. **The sessions landed on different
CPUs** (see the Shards line above), so wall times compare across hosts and are only diagnostic. The JSON receipts in this
directory hold per-page text, IDs, timings and RSS.

| arm | host | 15 scans mean / median CER | mean WER | word recall | 5 receipts word recall | median wall s per scan | peak RSS MB |
|---|---|---|---|---|---|---|---|
| Unlimited-OCR-2 native F16 | Xeon 2.2 GHz | 0.0584 / 0.0369 | 0.0558 | 0.958 | 0.233 | 465 | 12822 |
| Unlimited-OCR-2 original F32 | EPYC 7B12 | 0.0563 / 0.0339 | 0.0508 | 0.964 | 0.533 | 253 | 18181 (process) |
| MiniCPM-OCR native F16 | EPYC 7B12 | 0.0574 / 0.0258 | 0.0635 | 0.945 | 0.944 | 1793 | 5145 |
| MiniCPM-OCR original F32 | Xeon 2.2 GHz | 0.0574 / 0.0258 | 0.0636 | 0.945 | 0.944 | 275 | 8484 (process) |

- **MiniCPM-OCR native matches the original**: 18/20 token-identical, 19/20 normalised text equal, identical mean CER.
  One of the two ID mismatches (economic-journal-67) comes from the input, not the port: Wikimedia served two different
  thumbnail byte streams for that page, and the sha256 mismatch is recorded in native-uocr2 and ref-minicpm. Pin this
  image better in v2. The other mismatch (eb1911-v04-246, first difference at ID 124) changes CER from 0.1321 to 0.1318.
- **Native MiniCPM looks about 6.5x slower than Transformers F32** (28,943 s vs 4,292 s summed over the scans), even
  allowing for the different hosts. Profile it on one host. Memory is lower (5.1 vs 8.5 GB).
- **Unlimited-OCR-2 native text quality is on par for prose and scans**: 12/20 token-identical. Where IDs diverge, the
  divergence is late or in box coordinates, and CER stays within about 0.01 of the original, apart from the table defect
  described next.
- **Native Unlimited-OCR-2 table defect, confirmed on real receipts**: the CLI prints `<table>` without `<tr>/<td>`, so
  the cells run together (receipt-14 word recall is 0.13 native vs 0.98 original with identical IDs; receipt-16 is 0.30 vs 0.95).
  TODO: emit the row and cell tokens.
- **Unlimited-OCR-2 fails on these receipts in both implementations**: receipt-29 loops on empty `<td>` up to the
  1024-token cap; receipt-4 hallucinates LaTeX; receipt-20 repeats. This is source-model behaviour that the port
  reproduces. MiniCPM reads all 5 receipts (recall 0.90-1.00).
- Both models score badly on the dot-leader page rgbl-1867-403 (CER 0.38-0.39, from leader dots and padding). Both drop the
  EB1911 superscript footnote markers, and both read Fraktur long s as f (`Brautfchau`).
- The CORD ground truth is partial (annotated fields only), so word recall is the meaningful metric for receipts, not CER.
