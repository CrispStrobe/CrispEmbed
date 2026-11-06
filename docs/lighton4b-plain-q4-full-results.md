# Unchanged Q4: genuine BF16 source and corrected native

The genuine original BF16 model with the unchanged Q4 checkpoint also generates
an HTML table and hits256 tokens without EOS. Corrected native shares171 token
IDs, then differs. This reproduces the major output change inside the original
framework with quantized weights, while retaining a real native numerical and
decoded residual. It does not establish absolute OCR quality or qualify Q4.

## Authenticated source and controls

The [source receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-full/oracle.json)
comes from immutable [bd8759304](https://github.com/CrispStrobe/CrispEmbed/tree/bd87593040c9a31cf5ca6ec2fd2369de6f34b995),
service version4, bootstrap SHA
`b94074a5eb158a4ea6a861b0571f7bc2948f9dab2c7a61edcc4cf16bbcf68602`.
Its complete report SHA is
`1096b5f0f6ce26f0fa26ff9cc86dd606fbb3da24a3feca06768f2f8cd66d6d1b`.
All723 original BF16 parameter hashes and249 effective quantized parameter hashes
match the retained one-position oracle;474 protected tensors and the tied head
remain exact. The file is unchanged`f873eb3d…`:248 Q4_K matrices and its original
Q8_0 embedding, with no precision overrides. F32 exists only in bounded
independent dequantization chunks before BF16 parameter copies; no F32 model runs.

The successful worker is the frozen Intel AVX2 profile: vendorGenuineIntel,
family6/model79, Torch git`cf30153…`, two threads and one interop thread. Both the
official original`a4bece23…` and genuine Q4`e59d6054…` first-score vectors are
byte-exact. The actual frozen driver audit counts257 model forwards: one original
anchor plus256 independent Q4 forwards, with one Q4 vision invocation. All29
model-free controls pass. Full generation takes758.8 seconds; cgroup peak is
29,806,710,784 bytes. The expected numerical rejection remains red.

Three earlier failures are retained separately. The
[first attempt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-source-initial/oracle.json)
and [AVX2 retry](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-source-avx2-retry/oracle.json)
produce`02ca259b…`, rejecting the frozen original anchor before Q4 streaming.
The retry records AMD family23/model49 with otherwise matching weights, inputs,
four framework wheel hashes and AVX2 settings. The
[AMD-only attempt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-amd-profile-placement/oracle.json)
lands on Intel6/79 and correctly rejects its AMD anchor after reproducing frozen
`a4`. The final protocol selects these two closed profiles solely by measured
vendor/family/model before forward; unknown hardware is rejected. This measures a
platform-dependent original output, without identifying the responsible operator
or replacing the frozen reference. The successful full run uses the frozen Intel
profile, so the AMD deviation is not a caveat on this particular full comparison.

## Independent decoded outputs

[Native42183620520](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42183620520)
at`4443861693fc7bc4afeaa85f97e4cb2820250610` enables the corrected vision gates
and BF16 head, with mixed precision disabled. Its authenticated
[report](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-full/native-report.json)
has SHA`c726b09da906562a7f8cdd20a696d7e7d634745f7b329400f056fc4275e45d4e`.
Native artifact13488399938 contains181,791,946 bytes, SHA
`e896e23af456ad3bc1a1a482aa2420f5dfbfc93fafbf7def27a76c80a0f7a61e`.
Both outputs are independent greedy generations; both hit256 tokens without EOS
and end in an incomplete table. Neither is a complete-document result.

With genuine BF16 Q4 as the reference, native matches171 IDs, then differs at
zero-based row171. Raw text agreement is CER110/583=18.8679% and
WER17/48=35.4167%, with no normalization. The source Q4 text itself differs from
the frozen unquantized original: CER550/656=83.8415%, WER89/90=98.8889%; the
original completes177 tokens with EOS. These are reference agreement metrics,
not human OCR accuracy, and the capped outputs limit the comparison.

The first score remains distinct from runtime residual: genuine BF16 Q4 versus
unquantized original has RMS0.59156984; native versus that same Q4 source has
RMS0.0267200. The Q4 source prefers token20353 at22.0 over token27 at21.0,
whereas the original prefers27 at22.875 over20353 at20.75. This is a changed
model preference under quantization; a single token alone is not decoded-quality
proof, and scalar RMS values cannot be subtracted as causal error components.

## Remaining actual row171 boundary

The [authenticated selected rows](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-full/same-prefix-selected-scores.json)
use the existing raw source/native vectors, with matching causal token contexts.
At row171 the source chooses27287 (`Radio`) at20.5 over1058 at20.375. Native has
both at20.5 and correctly chooses the lower ID1058. The head is already rounded
to BF16. Maximum score error is0.203125, RMS0.03414869, cosine0.99984631.
Row170 has the same winning ID29 and RMS0.03572288. Changing argmax tie handling
would conceal the numerical difference rather than match original semantics.

No hidden or operator arrays at171 were captured in either full run. Existing
scores therefore identify the decision boundary but cannot identify its earliest
operator. Native retains F32 dequantized parameters while the genuine BF16 oracle
rounds them to BF16; that representation difference remains a confound. A future
bounded passive source/native replay could stop at172 tokens, require all172 old
score hashes and IDs, and capture row171 input/positions, all decoder layers,
final norm, pre-round head and layer0 operators. Actual retained inputs and both
parameter representations must be distinguished before proposing a runtime fix.
No such new inference has been dispatched for this report.

## Complete read-only archival and causal numerical progression

[Consumer42192667516](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42192667516)
authenticates the version4 source bootstrap/report and complete native archive.
It preserves all256 raw source rows (254,279,680 bytes), IDs/text, source audit and
172 native rows whose previous token IDs match. Rows172–255 have different
contexts and are excluded from numerical comparison. The actual portable Ref
binary SHA is`d6898491984110b0536dcd43ddb1091f6559495a79dfb23389baedb8dc4ae246`,
identical to the earlier mixed consumer. Equal/changed controls pass; all172 raw
comparisons fail the retained max0.002/mincos0.99999 budgets. The workflow is red
for this expected numerical failure, with report status`compared` and no error.
The [complete report](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-plain-q4-full/plain-q4-comparison.json)
retains every magnitude, causal context check, score hash and Ref result.

The residual is not monotonically increasing or uniformly small:

| Row | Maximum absolute error | RMS | Cosine | Winning IDs agree |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.15625 | 0.02672002 | 0.99994326 | yes |
| 33 | 0.25 | 0.04264112 | 0.99992412 | yes |
| 100 | 0.5625 | 0.08100626 | 0.99976118 | yes |
| 125 | 3.03125 | 0.49875835 | 0.98704768 | yes |
| 126 | 0.46875 | 0.07585576 | 0.99979287 | yes |
| 170 | 0.34375 | 0.03572288 | 0.99996496 | yes |
| 171 | 0.203125 | 0.03414869 | 0.99984631 | no |

Row125 is the largest RMS/absolute error despite choosing the same ID371:
source38.0 versus native38.75. This spike remains a meaningful numerical negative;
it cannot be dismissed because the decoded prefix still matches. Future stage
attribution should include125 as well as the171 decision boundary. Neither has
retained hidden/operator arrays, so this consumer makes no earliest-operator claim.

Artifact13491795001 contains177,270,896 ZIP bytes, SHA
`f19ddbdcc806b322b1d705442e1a02b9b089b9cd94d407c1d537a055dda5ad9d`, retained
through2027-02-04T09:07:44Z. Large payloads remain remote; only small reports and two
source/native rows were downloaded for review. No inference, default, release,
model alias, model upload or precision recipe is promoted.
