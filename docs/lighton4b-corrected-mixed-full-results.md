# Corrected mixed form: full score and decoded rejection

The corrected native mixed model reaches EOS after 156 tokens on the observed
form, versus the original BF16 model's 177. Its first 25 IDs agree; zero-based
ID 25 first differs. Complete decoded text and IDs reject exact agreement.
All 177 authenticated original-prefix score rows are finite and fail the
unchanged maximum-error `0.002` / cosine `0.99999` gate. Corrected vision does
not establish full decoder or recognition parity.

| Actual crispembed-diff row | Max absolute error | RMS error | Cosine | Original/native L2 magnitude |
| --- | ---: | ---: | ---: | --- |
| First, position 0 | 1.48261547 | 0.284899443 | 0.992587984 | 1103.18604 / 1141.71387 |
| Worst, position 5 | 10.1740723 | 1.51052773 | 0.763862431 | 1030.22205 / 1139.82166 |
| Last, position 176 | 3.07944798 | 0.55710572 | 0.969954610 | 1077.86047 / 1136.61646 |

Positions are zero-based. Earliest observed score difference and budget failure
are both 0. Per-row maximum error ranges from `0.962980747` to `10.1740723`, RMS
from `0.179244876` to `1.51052773`, cosine from `0.763862431` to `0.998720229`.
Every row fails both budgets. At the first position, maximum absolute original
score magnitude is `22.875`, native `22.5501289368`. Complete metrics for every
position are retained in the raw comparison and structured review below.

Raw decoded agreement against the frozen BF16 reference is CER `30.03049%`
(197 Unicode-code-point edits / 656 reference code points) and WER `44.44444%`
(40 word edits / 90 reference words). CER retains case, accents, punctuation
and newlines with no normalization. WER uses Unicode whitespace splitting and
preserves case and punctuation. The native text has 598 code points / 81 words.
These are agreement measures against model output, not human-ground-truth OCR
accuracy. Both outputs describe a French figure, but native labels include
`Researcher` and `Visible button` where the original includes `Rechercher...`
and `Modifier l'utilisateur`; full text is preserved for direct review:

- [Original BF16 decoded text](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/original-decoded.txt)
- [Corrected native mixed decoded text](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/native-decoded.txt)
- [Raw CER/WER definitions, counts and text hashes](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/decoded-agreement.json)

Native run [42109519607](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42109519607)
completed successfully, attempt 1, source
`189fc91bb1cc3a49ec918c804c30d072135546af`. Immutable mixed model SHA is
`d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474`;
its complete conversion/artifact provenance remains in the producer report.
Five prepared inputs and all 29 original BF16 vision stages are byte-exact on
this image. Corrected rotary table/profile identities and gates are verified;
decoder BF16 boundary, direct-quant/cache and unproven alternatives remain off.
The independent greedy run uses cap 256 and reaches effective EOS 248046, with
no tokens after EOS. Source text/pad ID remains 248044. EOS completion does not
override the decoded or numerical rejection.

The single read-only comparison
[42118693149](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42118693149),
source `957eb951387976fe8beef921eea83d812f89e607`, authenticates the complete
437,001,835-byte native archive, producer receipt, raw vision/score/generated
payloads and comparer-derived normalized receipt. The producer did not emit a
legacy `native.json`; normalization is labeled explicitly and independently
hash-admitted, preserving the mixed recipe identity. Original frozen protocol
SHA `4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc`,
receipt SHA `170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a`,
all 177 original row hashes, IDs, text and EOS are unchanged and authenticated
against the exact original BF16 provider version/bootstrap. No F32 prerequisite
or new original/native inference occurs in the comparer.

Admission controls and actual Ref equal/change/rejection/tampered-payload controls
all pass. Actual diff binary SHA is
`d6898491984110b0536dcd43ddb1091f6559495a79dfb23389baedb8dc4ae246`.
The red comparison is the expected numerical/decoded result: status `compared`,
return code 1, all 177 rows complete, rather than a blocked retrieval or harness
failure. The native run measures 2,896.15 seconds wall time, 1,697.34 seconds user
and 1,760.77 seconds system CPU, peak RSS 5,198,012,416 bytes. This single diagnostic
is not a throughput benchmark.

The separate genuine BF16 mixed-weight oracle covers only the first position.
Here native first-score RMS `0.2848994466` versus original is close to that
oracle's `0.2857999084`, while native versus that oracle still has maximum
`0.1633567810`, RMS `0.0274295614`, cosine `0.9999284248`, a strict rejection.
Native uses F32 dequantized weights; the genuine BF16 oracle rounds those
parameters to BF16. This residual mixes runtime and effective-precision effects;
it is not an isolated implementation error. First-position evidence cannot
partition all 177 score errors or the full decoded change into quantization
versus runtime. The comparer reports no new internal decoder-stage attribution.

This form was already observed and remains excluded from held-out calibration
or architecture-quality selection. Earlier Q8/Q4, mixed calibration and reference
control negatives remain preserved. No default, alias, release, app-pin, model
publication, new PR or further inference/comparison job follows this bounded run.

Complete small receipts:

- [Producer report: full decoded output, hashes, controls and resources](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/native-report.json)
- [Explicit comparer-derived receipt](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/comparer-derived-native.json)
- [Authentication/build/control audit](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/full-replay.json)
- [Actual full177-row comparison and independent decoding](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/comparison.json)
- [Structured177 rows, artifact/receipt identities and limits](../tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/review.json)

Raw native artifact `13457482504`, digest
`24aad9c72f67d645d705d2b45bf09b7260f144ff5d785a8856fb35046cb7b7aa`,
remains remote until `2027-02-03T21:06:10Z`. Comparison artifact `13457253394`,
374,854 bytes, digest
`682683af1e78d59b94402ae08a2b220b1dae55e9f535654f8928b30ad86b92ab`,
retains all row/control/build logs until `2027-02-03T22:07:06Z`. Its complete ZIP
hash is independently verified. Only needed small reports/text were downloaded;
the large native payload stays remote.

The earlier mixed model transport artifact `13405370923` is a separate
reproducibility dependency: 3,830,524,370 bytes, digest
`6d7e72a423ae0e0b1ab5fd331bad240174c2b29421fe7b585130868cb8a0ee10`,
expires `2026-11-19T09:57:52Z`. Its immutable model hash, packing source and existing
public Q8/Q4 source identities remain recorded. This run does not extend that
transport's retention or publish/restage a model payload.
