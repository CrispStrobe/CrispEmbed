# Weighted Q4KM-style form: valid result, criterion failed

The fixed candidate reaches EOS but fails the predeclared form criterion:
character agreement with BF16 is worse than pinned llama.cpp Q4_K_M, while word
agreement is better. The candidate therefore does **not** meet all three observed
case floors and remains unqualified for shipping. Passing table/Arabic or choosing
only WER cannot override this negative.

| Output on the same form | IDs / stop | CER versus BF16 | WER versus BF16 |
|---|---|---|---|
| BF16 original |177 / EOS|0/656|0/90|
| Pinned llama Q4_K_M |170 / EOS|189/656 (0.288110)|46/90 (0.511111)|
| Fixed weighted candidate |182 / EOS|217/656 (0.330793)|43/90 (0.477778)|

The first native ID difference from BF16 is at position 11. Candidate text has
673 codepoints and 94 whitespace words, versus 656/90 in BF16 and 647/86 in the
publisher output. These raw agreement metrics apply no case, punctuation,
whitespace or Unicode normalization and are not human OCR accuracy.

Read the exact [candidate text](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/form-postselection-v1/lighton-q4km-form-results/simple_form/native/generated.txt),
[BF16 text](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/form-postselection-v1/references/original-bf16.txt)
and [llama Q4_K_M text](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/form-postselection-v1/references/pinned-llama-q4km.txt).
The candidate begins `Figure – Interface de saisie` and includes altered wording
and labels; its complete independent IDs, actual EOS and bytes are preserved.

## Authentication and independent review

[Thirteen raw API files, reference copies and review](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/form-postselection-v1/evidence.json)
retain source
[`b978b1cef`](https://github.com/CrispStrobe/CrispEmbed/commit/b978b1cefb9cafee88a70e23f4f4faf1b6ab90a5),
bootstrap SHA `0faac598a3d5c87a0f441578fc55ce35bb065e79802ee11079a22a5ad629d0a3`
and result SHA `a483fe4d56cfdde2c3835d2cf7ca1f322d09c4594664405e0faa8da073230643`.
The model remains
`5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29`.
The worker independently rehashed the complete mounted file and reused the
producer's exact executable/library bundle and original arithmetic.

Before form inference, its full table vocabulary row and 70 common producer
stages matched exactly. All five form prepared-input arrays matched the original
BF16 capture's canonical NumPy hashes. The original capture is
`170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a`:
177 IDs/EOS, original checkpoint revision and all 723 parameters in BF16. Neither
source inputs nor logits are substituted into native inference.

The hosted controller checked every form vocabulary row's shape, finite values
and selected argmax. A fresh read-only review authenticated actual small IDs/text,
canonical byte hashes, EOS, all 182 row-manifest entries, platform stages and
independently recomputed distances against both references. Direct candidate-to-
llama differences are 220/647 characters and 45/86 words; the criterion uses the
shared BF16 denominator above. No new inference or model/large-array download ran
in this review. All raw score arrays remain remote without a fixed retention
expiry guarantee.

The captured worker identifies four logical/affinity CPUs, `GenuineIntel`,
`Intel(R) Xeon(R) CPU @ 2.20GHz`. This fixes future metadata collection; it does
not retroactively assign CPU identity to the earlier producer's blank fields.
The controller took 3,030.72 seconds, including 135.82 seconds for its platform
control and 2,857.43 seconds (47.62 minutes) for form generation. Form child
user/system CPU totals are 3,011.08/1,896.27 seconds. Recorded cumulative completed-
child peak RSS is 4,542,083,072 bytes, not an isolated per-child peak claim.

## Scope and next bounded test

[Table and Arabic](lighton4b-imatrix-q4km-full-calibration-results.md) retain their
positive observed results on the same unchanged candidate. This form was already
observed and is explicitly post-selection, not heldout. There is no aggregate,
threshold change, recipe revision, default flip or release promotion.

The [225/249 payload comparison](lighton4b-weighted-publisher-payload-comparison.md)
narrows the remaining coordinate question but does not explain this CER loss by
itself. Next is an original-F16 tensor reproducer for the 24 recurrent output
matrices in upstream tiled columns, using raw upstream importance coordinates.
Only after strict payload checks will a distinct HF-tiled-out tag/gate be tested:
699 other payloads and HF gamma/log-A semantics remain unchanged, with only the
matched activation permutation before output projection. It must not set the
physical-format6 scalar semantics on an HF container. The original candidate and
all negative evidence remain intact; no new conversion or inference follows from
this report alone.
