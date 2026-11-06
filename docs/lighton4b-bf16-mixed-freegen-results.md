# Genuine BF16 mixed full generation: native residual confirmed

The original BF16 model with the fixed mixed quantized weights generates 178
tokens and reaches EOS. Corrected native mixed generates 156 tokens and reaches
EOS; their first 33 IDs agree, then zero-based ID 33 differs. With genuine BF16
mixed as the reference, raw native agreement is CER `19.85185%` (134 edits / 675
reference code points), WER `31.57895%` (30 edits / 95 reference words). The
decoded residual persists with the same quantized tensor provenance; comparing
native only with the unquantized official BF16 model had confounded these effects.

| Independent model output | Tokens / stop | First ID difference versus official BF16 | Raw agreement against official BF16 |
| --- | --- | ---: | --- |
| Official original BF16 | 177 / EOS | none | exact reference |
| Genuine original BF16 mixed | 178 / EOS | 25 | CER 180/656 = 27.43902%; WER 40/90 = 44.44444% |
| Corrected native mixed | 156 / EOS | 25 | CER 197/656 = 30.03049%; WER 40/90 = 44.44444% |

These are model-output agreement measures, not human-ground-truth OCR accuracy
or architecture-quality selection. CER preserves Unicode case, punctuation,
accents and newlines; WER uses Unicode whitespace splitting with case and
punctuation retained. Direction matters: reversing the primary comparison and
using native text as denominator gives CER `22.40803%` (134/598), WER `37.03704%`
(30/81). Both directions are retained explicitly in the review.

The genuine BF16 mixed text includes `Rechercher...` and `Modifier l'utilisateur`
where native produced `Researcher` and `Visible button`. Both remain different
from the official reference. [Full genuine BF16 mixed text](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-freegen/decoded.txt),
[native and official texts](lighton4b-corrected-mixed-full-results.md) and all IDs
are preserved; no reference is regenerated or replaced.

One CPU-only worker, source `89eda20c6558b9439c2bc3bc78ea395a6c463a1c`, follows
the [predeclared full-generation protocol](https://github.com/CrispStrobe/CrispEmbed/blob/89eda20c6558b9439c2bc3bc78ea395a6c463a1c/docs/lighton4b-bf16-mixed-freegen.md).
Original source remains `8d3b2331d1b57e285f7ba348fc1eacd689b79aa8`. All 723
original BF16 parameter hashes and all 249 effective mixed BF16 parameter hashes
match the prior independent one-position oracle. The same fixed 153 Q8
embedding/attention + 96 Q4 FFN matrices are installed by authenticated bounded
streaming. All 474 protected original tensors, buffers, parameter identity,
storage, dtype and tied head are checked. Full original checkpoint SHA/size,
complete Q8/Q4 file hashes and five prepared input hashes pass.

The official one-token score SHA remains exactly
`a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd`;
mixed first score remains exactly
`bea1064566634615c955582808c75b0003dc8f24983cbd12cbb661a1b5c33ae9`.
Independent mixed greedy cap is 256, sampling disabled and cache enabled. Actual
counts are 178 mixed root/text forwards and one mixed vision forward; the
unchanged original driver independently records 179 root calls including the
one-token official anchor. Effective EOS is 248046, source text/pad 248044; no
tokens follow EOS and the cap is not hit. No teacher-forced replay runs.

All 178 full-vocabulary score rows are finite, 993,280 bytes each, totaling
176,803,840 bytes. Their immutable hashes are in the complete oracle report.
Generated ID file SHA is
`786f4f36146fa87e563af90ccaeed9ff1318e8dc7ab09da0d9969ad2a7febc14`;
decoded UTF-8 SHA is
`c9075ce2be0b510cb812d929a5fef7bdc8144c99d39addaf7c1677556bbd1292`.
No intermediate activations or caches were captured in this run. The worker
compares decoded output using public native report SHA
`2853afb5518c283f6eea356208c7f034a3c7b2e1ed56bfe61b62eab757303ee8`,
source `189fc91bb`; it receives no GitHub artifact credential. Numerical native
score comparison is deferred and must use identical causal prefixes only.

The result does not alone isolate an implementation defect. Native retains F32
dequantized weights; the genuine BF16 original rounds those parameters to BF16.
All original operators and boundaries also retain their genuine BF16 semantics.
The next bounded activation replay targets shared-prefix decision 33 to separate
earlier precision/implementation residuals from the final score boundary. The
runtime agent's score-margin evidence and gated head-rounding candidate remain
separate; full decoded agreement must be measured rather than inferred from a tie.

Full generation takes 596.86 seconds. Process HWM is 13,883,699,200 bytes; cgroup
peak is 27,614,916,608 of 32,212,254,720 bytes. Capacity/cache guards pass, no OOM or
retry. The terminal error is intentional rejection of the already known official
first-score gate (mixed max `1.578125`, RMS `0.2857999084`); original driver returns
zero, no report error exists, and full capture/audits complete. Seventeen hosted
controls actually pass 6 original + 6 mixed + 5 full. Their logs retain the
nonfatal startup `sitecustomize` / missing `wrapt` warning. A historical reporting
correction is also retained: the prior mixed bootstrap ran its mixed six tests
twice, while the original six passed locally; this worker executes the three
distinct paths explicitly. No numerical evidence is overwritten by that correction.

Complete small evidence:

- [Full oracle, 178 score hashes, source parameter hashes and controls](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-freegen/oracle.json)
- [Final original-driver/checkpoint audit](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-freegen/original-audit.json)
- [CPU worker/bootstrap resource receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-freegen/bootstrap.json)
- [Primary/reverse metrics, exact receipt/service identities and limits](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-mixed-freegen/review.json)

Large score payloads remain with CPU service version v1
`lighton4b-bf16-mixed-free-generation-v1`; only small reports/control logs are
downloaded. No 90-day retention extension is claimed for that service output.
The prior native artifacts retain their separately recorded expiries. This
observed form remains excluded from held-out calibration. No F32 model,
publication, default, release/app-pin change, new PR or extra model worker is
part of this completed capture.
