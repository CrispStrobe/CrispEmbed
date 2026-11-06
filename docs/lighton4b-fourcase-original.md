# Frozen four-case BF16 original capture

Actual CPU-only source42e5ce6f9bc6bd01865ac4f21d2157c98d3475f1 completes the
predeclared four-case protocol3a1738c26cf5a34680fea1468bea07c06ee26659f7ddb7e249299677746ba36d
and input manifest a07af76028a0d53c8bfc0ccafedac348eaeb725c8c8c55e3136906f202b8f65d.
All four PNG/RGB/official processor grids/token counts and prepared input hashes
are checked before one BF16 model load. Preparation-only and actual-run input
metadata hashes match for every field and case. The previous177-token form is
excluded and was not rerun. Runtime/checkpoint/tokenizer pins remain unchanged.

| Case | Split | Generated IDs/full score rows | Original stop |
| --- | --- | ---: | --- |
| simple_table | calibration |222 |EOS |
| arabic_printed_line | calibration |6 |EOS |
| receipt_example | heldout |256 |Frozen cap, no EOS |
| handwritten_letter | heldout |65 |EOS |

The heldout receipt is **source-completion negative**; it ends mid-table at the
unchanged256-token cap. This blocks complete four-case qualification even if a
native engine faithfully reproduces it. Preserve the image, prompt, cap, scorer,
original bytes and negative receipt. Do not substitute another heldout case.

The frozen fixture named`handwritten_letter` visibly contains a red serif A
logo. Its original output describes that logo; this case does not demonstrate
handwriting transcription. Both calibration/heldout splits stay frozen. The
other source outputs also establish no representative human OCR benchmark.

The [original batch](../tests/results/qwen35-ocr/lightonocr3-4b-fourcase-original/batch.json)
has SHA`bd8b4c4c095f17c7873c510d97b07f02868b62592251581d07c0362f3d762502`.
[Independent small-report review](../tests/results/qwen35-ocr/lightonocr3-4b-fourcase-original/review.json)
checks all549 position/ID/shape/byte-count/hash records. Each row has248320
F32 scores/993280 bytes; raw score payloads total545310720 bytes and remain
remote. Only small reports were downloaded; future consumers must rehash the
actual prepared-input and score payloads before using them as ground truth.
Exact per-case capture hashes are retained in the review and unaltered case
receipts. Private provider identity stays in ignored environment notes.

There are549 original forwards,723 BF16 parameters and9078531072 parameter bytes.
Measured process HWM is9945726976 bytes; cgroup peak22290255872 bytes includes
file-cache/resource overhead. Batch time is2068.56 seconds on its hosted worker;
these are original end-to-end measurements, not native speed comparisons.
No native inference or source-quality/model publication qualification follows.
New CI dispatches remain paused for user-requested consolidation.
