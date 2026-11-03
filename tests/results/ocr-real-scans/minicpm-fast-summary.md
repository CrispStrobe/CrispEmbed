# OCR on real scans (crispembed-ocr-real-scans)

Shards: native-minicpm-fast @ `dded13842101` Intel(R) Xeon(R) CPU @ 2.20GHz x4 31.3 GB, 1.58 h

Metrics after the normalisation in `score.py` (markup stripped). CER / CER without spaces / WER / word recall; wall s; peak RSS MB. `-` = not run (time budget) or failed (see JSON).

| page | uocr2-f16-fast-tiled | uocr2-original-f32 | minicpm-f16 | minicpm-f16-fast | minicpm-original-f32 |
|---|---|---|---|---|---|
| de-antiqua-kafka-schloss-211 | - | - | - | 0.007 / 0.007 / 0.022 / 0.984; 224.45 s; 8172.2 MB | - |
| de-degraded-freiligrath-46 | - | - | - | 0.039 / 0.047 / 0.139 / 0.861; 161.11 s; 7937.9 MB | - |
| de-fraktur-grimm-319 | - | - | - | 0.027 / 0.033 / 0.107 / 0.893; 221.44 s; 8268.3 MB | - |
| en-form-economic-journal-67 | - | - | - | 0.017 / 0.021 / 0.000 / 1.000; 274.19 s; 8180.2 MB | - |
| en-prose-constitution-13 | - | - | - | 0.011 / 0.012 / 0.000 / 1.000; 206.13 s; 8172.1 MB | - |
| en-table-climatic-26 | - | - | - | 0.005 / 0.006 / 0.004 / 0.998; 286.18 s; 8172.3 MB | - |
| receipt-cord-test-29 | - | - | - | 0.035 / 0.028 / 0.018 / 1.000; 158.86 s; 7944.5 MB | - |
| de-2col-meyers-b13-103 | - | - | - | 0.047 / 0.053 / 0.227 / 0.778; 517.41 s; 7893.4 MB | - |
| de-antiqua-kafka-betrachtung-94 | - | - | - | 0.011 / 0.012 / 0.032 / 0.968; 179.17 s; 7731.7 MB | - |
| de-table-rgbl-1867-403 | - | - | - | 0.389 / 0.242 / 0.101 / 0.899; 396.22 s; 8377.1 MB | - |
| en-2col-eb1911-v04-246 | - | - | - | 0.132 / 0.133 / 0.148 / 0.918; 345.28 s; 8376.8 MB | - |
| en-report-amtrak-23 | - | - | - | 0.001 / 0.001 / 0.002 / 0.998; 286.93 s; 8304.6 MB | - |
| receipt-cord-test-16 | - | - | - | 0.041 / 0.037 / 0.070 / 0.930; 147.16 s; 7944.5 MB | - |
| receipt-cord-test-4 | - | - | - | 0.061 / 0.064 / 0.161 / 0.903; 145.69 s; 7943.8 MB | - |
| de-mixed-welzheim-188 | - | - | - | 0.026 / 0.029 / 0.118 / 0.905; 363.43 s; 8374.7 MB | - |
| en-2col-eb1911-v25-5 | - | - | - | 0.106 / 0.122 / 0.007 / 0.994; 358.76 s; 8317.7 MB | - |
| en-old-statutes-v1-772 | - | - | - | 0.041 / 0.040 / 0.038 / 0.974; 277.24 s; 8168.6 MB | - |
| en-prose-constitution-81 | - | - | - | 0.002 / 0.001 / 0.006 / 1.000; 193.39 s; 8172.3 MB | - |
| receipt-cord-test-14 | - | - | - | 0.004 / 0.005 / 0.019 / 0.981; 189.24 s; 8189.7 MB | - |
| receipt-cord-test-20 | - | - | - | 0.027 / 0.033 / 0.095 / 0.905; 161.35 s; 7944.4 MB | - |

## Native vs original token agreement

| page | engine | ids exact | first diff | prefix agr. | native n | ref n | CER native | CER original |
|---|---|---|---|---|---|---|---|---|

| engine | paired pages | IDs exact | normalised text equal | mean CER native | mean CER original |
|---|---|---|---|---|---|
