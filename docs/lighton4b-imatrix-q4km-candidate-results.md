# Weighted Q4KM-style candidate: conversion and first scores

The original-F16 one-pass candidate completed on Kaggle CPU in1,796.45 seconds
from tested source [`a4f2c908f`](https://github.com/CrispStrobe/CrispEmbed/commit/a4f2c908ffaf3f17eb770973017e3de8a08bb121).
The [complete small receipts and raw logs](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/evidence.json)
preserve the exact controls,723 payload hashes and source identities. This is a
first-score measurement; complete decoded qualification is still pending.

The file is4,041,905,312 bytes, SHA256
`5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29`.
It contains216 Q4_K,33 Q6_K and474 F32 tensors. All248 body projections actually
used the adapted importance matrix; the tied embedding/head is deliberately
unweighted Q6_K because the published statistics contain no head entry. All474
protected F32 payloads equal the admitted null file. Local consistency checks
confirm the complete723-hash audit, all249 changed quantized payloads and all474
protected equalities. The original source and adapted-statistics full SHA gates
passed before conversion; neither Q4 nor Q8 weights were requantized.

The custom profile is `lighton4b-weighted-q4km-style-q6head-v1`. Its33 higher
precision tensor names match the pinned publisher's named type schedule. Native
HF grouping, quant-block boundaries, source carrier and arithmetic prevent a
claim of publisher-byte equivalence. The publisher's text-statistics corpus and
exact conversion tool provenance remain unverified; no image-imatrix claim is
made. The model and521,472,000-byte raw Q6 head fixture remain remote, together
with a hashed runtime bundle. Kaggle retention has no asserted90-day guarantee.

| Case / first-score arm | Selected ID | Probability of BF16's first ID | Raw logit RMS / max absolute error vs BF16 |
|---|---:|---:|---:|
| Table: unweighted plain Q4 |20353|0.405578|0.706260 /4.226563|
| Table: weighted plain Q4, prior experiment |27|0.590433|0.865078 /4.351563|
| Table: weighted Q4KM-style with Q6 head |27|0.503789|0.776426 /3.792969|
| Arabic: unweighted plain Q4 |67321|0.871767|0.364631 /2.953125|
| Arabic: weighted plain Q4, prior experiment |67321|0.892794|0.387335 /2.218750|
| Arabic: weighted Q4KM-style with Q6 head |67321|0.858263|0.336954 /2.093750|

The table first ID now agrees with BF16, but its probability is lower than the
prior weighted plain-Q4 result (`0.503789` versus `0.590433`). Arabic retains the
same first ID while that probability decreases. Raw RMS also moves in different
directions: a shared additive logit shift changes RMS while leaving softmax
probabilities unchanged. These metrics therefore cannot select a decoded-quality
winner. The earlier weighted plain-Q4 table run timed out after225 rows; it
provides no complete table text, CER/WER or EOS result.

Both frozen unweighted null first-score hashes reproduced exactly, and the
post-merge model-free controls passed on the actual producer source. The initial
unqualified graph-reuse fixture failure is retained in the
[earlier controls evidence](lighton4b-imatrix-q4km-controls.md); this producer
uses the already passing original arithmetic. No performance optimization,
heldout read, default, shipping approval or release follows from this result.

A separate continuation uses this exact model and preserved binaries, rehashes
the complete mounted model, requires exact producer prefill/stage hashes, then
runs table/Arabic to actual EOS or256 tokens with75/20-minute child budgets.
A separate form diagnostic follows against frozen BF16 and the pinned llama.cpp
Q4_K_M transcription, with original prepared-input hashes and a producer-score
platform control. All three cases require complete independent token/text/EOS
receipts; the previously observed form is post-selection, not heldout.

The initial complete-decode continuation stopped before model forward: this
Kaggle image has no `/usr/bin/time`. Its8 small raw files are
[preserved](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/decode-v1-negative/evidence.json).
Model/runtime authentication, Arabic preflight and all five original prepared
input hashes passed first. The corrected source `e822a6de8` uses Python child
resource accounting and checks both executable loaders/argument guards and all
calibration asset paths before copying the model. Three local trust-boundary
controls pass. A single corrected continuation is running on the same sealed
candidate and original producer binaries; no quantization rerun or quality result
is inferred from that setup correction.
