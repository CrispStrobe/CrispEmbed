# Weighted Q4KM-style full calibration result

The fixed custom candidate beats the observed llama.cpp Q4_K_M table output
against frozen BF16, and matches Arabic exactly. Both native outputs reach EOS;
form is still pending in a separate post-selection diagnostic. This is an
observed-case improvement, not BF16 parity or shipping approval.

| Frozen case | Candidate output | Candidate CER / WER vs BF16 | Pinned llama Q4_K_M CER / WER vs the same BF16 |
|---|---|---|---|
| Table |214 IDs, EOS, no cap|160/607 (0.263591) /47/106 (0.443396)|315/607 (0.518946) /88/106 (0.830189);256 IDs, cap/no EOS|
| Arabic |6 IDs, EOS, no cap|0/12 /0/2, exact IDs/text|0/12 /0/2, exact IDs/text/EOS|

The table still has 160 character edits and 47 word edits against BF16; its first
ID difference is at position 39. Native text contains 673 codepoints/120 words,
versus 607/106 in BF16 and 769/151 in the capped upstream output. These are raw
model-agreement CER and whitespace-word WER, not human OCR accuracy. No whitespace,
case, punctuation or Unicode normalization was applied.

The exact small transcriptions are retained:

- Table: [candidate](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/lighton-q4km-decode-results/simple_table/native/generated.txt), [BF16](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/references/simple_table/original-bf16.txt), [llama Q4_K_M](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/references/simple_table/pinned-llama-q4km.txt).
- Arabic: [candidate](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/lighton-q4km-decode-results/arabic_printed_line/native/generated.txt), [BF16](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/references/arabic_printed_line/original-bf16.txt), [llama Q4_K_M](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/references/arabic_printed_line/pinned-llama-q4km.txt), all exactly `أبجدية عربية`.

## Source and byte controls

The [raw receipts and independent review](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/full-calibration-v2/evidence.json)
preserve 16 API files, actual decoded/token bytes, derived frozen reference texts
and their hashes. Source is
[`e822a6de8`](https://github.com/CrispStrobe/CrispEmbed/commit/e822a6de84c8fb546623a464c7e9c79a62e19bbc),
bootstrap SHA `8bc640586a09baa5953ba6c1bcd7dac6f5431281e5c1f9460f7a84400f469f5d`,
result SHA `1f296b54b74e2660f81f7fa7d9aae9850bb3d22fb19c5024c7a28908248090e7`.
The mounted 4,041,905,312-byte candidate was independently rehashed on the worker:
`5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29`.
It reuses the producer's exact executable/library bundle and original arithmetic;
no graph-reuse experiment is present. The [producer evidence](lighton4b-imatrix-q4km-candidate-results.md)
retains the 248 actual weighted applications,216 Q4/33 Q6/474 protected F32 schedule,
all 723 payload hashes and both frozen plain-Q4 score controls.

All five original prepared inputs per case matched. The first vocabulary row and
70 common native-producer stages per case matched byte-for-byte. These 140 pairs
are repeatability against the same native producer, not 140 BF16-parity stages.
No new source activation capture was made at the later ID divergence.

The hosted controller independently reads every returned vocabulary row, checks
finite values, full 248320-column shape and the selected argmax, including actual
EOS. The fresh read-only review rechecked actual small token/text files against
the receipt, canonical bytes, EOS and all 220 row-manifest entries, then independently
recomputed CER/WER against both frozen transcriptions. It downloaded no model or
large score arrays and ran no new inference. All 220 raw rows remain remote.

## Timing and limits

The controller took 3,927.82 seconds. Arabic's native child took 210.24 seconds;
table took 3,676.83 seconds (61.28 minutes), within its 75-minute bound. The earlier
45-minute/225-row timeout remains a negative; this result has complete independent
IDs, text and EOS. The initial missing-timer setup failure remains preserved too.

For table, child user/system CPU totals were 3,744.58/2,954.60 seconds. The recorded
completed-child cumulative peak RSS is 4,487,450,624 bytes; it is not an isolated
per-child peak claim. Kaggle provides no asserted fixed 90-day retention guarantee.
The tested native path uses CPU, bounded dequantization and quantized storage;
no original F32 neural model or new BF16 reference forward ran.

The profile, external text statistics, fixed input/control hashes and precision
gates did not change after first-score screening. This is a calibration-only
comparison between independently quantized/runtime-layout variants; the separate
physical same-weight test addresses runtime attribution. Publisher checkpoint
identity is declared in its header but not proven from quantized tensors alone.
Form remains post-selection, other heldout cases are untouched, and neither the
public-model fence nor defaults/release artifacts are changed by these results.
