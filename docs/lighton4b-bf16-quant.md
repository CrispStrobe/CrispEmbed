# BF16 original versus native Q8/Q4 diagnostic

This separate protocol uses the completed original BF16 free-generation capture;
there is no F32 prerequisite. It preserves document-v1/v2 and their admissions.
The exact original receipt SHA, all177 IDs, decoded text and177 score payload
hashes are frozen in [the protocol](lighton4b-bf16-quant-v1.json). The separate registry now admits the two authenticated complete native
executions for diagnostic comparison only. Both fail full numerical and decoded
parity; see the final results below. No model/precision quality is qualified.

Retrieve the completed original capture from its retained hosted artifact into
an empty directory, including `capture.json` and every `scores-NNN.f32` file.
The consumer authenticates retrieved bytes against the frozen receipt/payload
hashes; it never fetches a model or regenerates the reference. Remote artifacts
may be retrieved directly on the hosted worker; no shared-host model is needed.

A future actual producer must write `native.json` with schema
`lighton4b-bf16-quant-native/v1`, precision `q8_0` or `q4_k`, this protocol's
SHA256, `original_capture_sha256`, exact `teacher_forced_ids`, `scores`,
`free_generation_independent=true`, independent `generated_ids`, `decoded`,
`stop`, `cap_hit`, `native_forward_executed=true`, `producer_source_sha`,
`producer_run_url` and `weights_sha256`. Every score record has the original
position, file, shape `[1,248320]`, bytes993280 and SHA256. Teacher forcing must
consume the original prompt and original IDs before that position, never its
own free-generation prefix. Independent generation starts afresh with the
frozen image-only prompt, cache, greedy decoding, EOS248046 and cap256.
A separately reviewed native-receipt-SHA256/source/run/weight/precision tuple must first be added to
`native_producers` in [the separate registry](lighton4b-bf16-quant-admissions-v1.json); arbitrary producer claims and synthetic fixtures fail closed.

On hosted CI, build target `crispembed-diff-lighton4b-document`, then run:

```sh
python tools/lighton4b_bf16_quant.py \
  --original "$ORIGINAL" --native "$NATIVE" \
  --binary "$DIFF_BINARY" --output "$REPORT"
```

The consumer compares every original-prefix score through unchanged actual
`crispembed_diff::Ref` and bounded one-position GGUF pairs. It preserves max/mean
absolute error, RMS, minimum row cosine and both magnitudes, strict0.002/0.99999
flags, first observed failing score position, and complete original/native
IDs/text/EOS. Numerical and decoded parity are separate required results.
The original contains no internal-stage trace; output-position differences do
not identify the earliest internal operator. Observer-off original equivalence,
representative OCR quality, model/publication qualification and intermediate
attribution remain outside this diagnostic.

[Hosted controls](../.github/workflows/lighton4b-bf16-quant-controls.yml) use
synthetic full-vocabulary rows with the actual Ref binary: equal rows, an error
at position1 and tampered bytes; contract negatives reject missing prefixes,
missing scores, independent-generation omission, wrong precision and stops.
Synthetic controls retain `native_execution_admitted=false` and cannot admit
weights or native execution. Large artifacts stay remote.

Actual [hosted41997183136](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41997183136)
at `a5b005b368e6f47530eea01bdc81d07df4aadf44` passes both control methods,
including actualRef equal/error/tamper and contract-negative subcases. The
[receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-controls.json)
retains exact source and executable hashes. Initial41997070623 was cancelled
while queued after the public-path/forged-producer control amendment; no model
ran in either workflow. The initial transient Git push rejection and absent
new-workflow default-branch dispatch404 preceded the successful exact-source
existing `heavy-cpu.yml` dispatch. Native parity remains pending.

The Q4 candidate is the actual plain `q4_k` recipe, not a Q4_K_M mixture:
`lightonocr3-4b-q4_k-vision-f32.gguf`, SHA256
`f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91`.
The prior preparatory `q4_k_m` label remains in Git history; it was never an
admitted or executed native producer. Receipt-digest binding was added before
any native admission. Current protocol digest must be used by future captures.

Final actual-Q4_K/receipt-bound
[controls41998539516](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41998539516)
at `acc4e5dece62bd49c576528d63fd30e33dd4739b` pass both methods, including
below-budget nonzero drift, first threshold failure, corrupted bytes, forged
producer and changed receipt-digest negatives. The
[final receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-controls-final.json)
retains exact hashes. Earlier41998345877 independently passes receipt binding
before the Q4 recipe-label amendment. Superseded41998180454 and41998220523
were cancelled before execution. No actual native producer was admitted.

A separate bounded first-score diagnostic is prepared in
`tools/ci-heavy/lighton4b_smoke_compare.py`. Dispatch existing `heavy-cpu.yml`
with explicit `--recipe q8_0` or `--recipe q4_k`, `--native-run`,
`--native-source`, `--artifact-id`, `--artifact-sha256` and
`--receipt-sha256` arguments after the successful native smoke receipt is
reviewed. It authenticates exact run/attempt/source, artifact digest, native
receipt and corresponding immutable candidate weight hash before reading one
native row. Read-only provider credentials retrieve only original version1's
993,280-byte first BF16 score row, whose frozen SHA is independently verified.
No original forward is repeated and no model is acquired by this comparer.
The unchanged actualRef report preserves raw errors/magnitudes and native
partial decoded text. One-position evidence never admits a177-position producer
or qualifies independent full decoded parity. Actions access is read-only and
its token is supplied only for this exact script selector; no provider write
operation is implemented.

Admission tuples live in the separate protocol-bound registry. The frozen
protocol file remains byte-identical at SHA256
`4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc`.
This prevents a circular dependency: admitting a receipt digest must not change
the protocol digest already embedded in that receipt. Empty registry remains
fail-closed until actual complete native outputs are independently reviewed.

[Registry controls42001553561](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001553561)
at `5eb0735c0` pass all three methods, including actualRef equal/error/tamper,
below-budget drift, receipt-digest admission and wrong-registry-protocol
rejection. The [receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-registry-controls.json)
retains exact hashes. Initial first-score jobs42000741513/42000808436 stopped
before comparison because Kaggle imports undeclared `six`; it is now explicitly
pinned. Corrected42001333434/42001338099 then stopped at provider HTTP before
comparison: the official SDK documents that historical version labels require
`v1`, not bare `1`. Exact `v1` requests replace that spelling; no current-version
fallback or source/hash substitution is permitted. All four initial blocked
receipts remain preserved and are not numerical failures.

## Actual first-position evidence

Exact-version corrected [Q842001979476](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001979476)
and [plain Q4_K42001983230](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001983230)
compare all248,320 actual first-position scores against the authenticated
original BF16 row. Both are finite and **fail** unchanged numerical gates:

| Candidate | Maximum absolute error | Mean absolute error | RMS | Row cosine | Reference/native norm | Two-token smoke text |
|---|---:|---:|---:|---:|---:|---|
| Q8_0 |0.568259001|0.108710214|0.130495042|0.998469710|1103.18604/1078.98657|`<figure`|
| Q4_K |2.49422884|0.466762543|0.571912766|0.976987541|1103.18604/1238.58728|`<table>`|

Original first two IDs are `[27,17251]`; Q8 matches that prefix, whereas plain
Q4_K emits `[20353,29]`. Both smokes intentionally cap at two tokens without
EOS. Earliest observed score difference and numerical failure are position0;
no original internal trace exists to attribute an earlier operator. Full177
teacher-forced scores and complete independent generation remain separate.
Successful workflow completion here means the diagnostic ran, not that parity
passed. Exact receipts retain [Q8](../tests/results/qwen35-ocr/lightonocr3-4b-q8-firstscore.json)
and [Q4_K](../tests/results/qwen35-ocr/lightonocr3-4b-q4-firstscore.json) source,
artifact, row and executable hashes. No reference or model forward was repeated
by either comparer. The native smoke producer source is
`fd150b98f488a6884a6e8d8127e7dfb263cab237`.

`tools/ci-heavy/lighton4b_full_compare.py` is prepared for the complete panels.
It requires both reviewed report and native-receipt digests, an exact admitted
native receipt tuple, immutable artifact/source/weight checks, original `v1`
capture and all177 authenticated rows, then actualRef plus complete decoded/EOS
comparison. It performs no inference, retains large inputs in hosted scratch,
and rejects empty/unreviewed admission before transfer. The protocol remains
byte-identical; source/run/weight/capture tuples belong in the separate registry.

## Retained-score rank and gated-path diagnostics

No inference was repeated by these comparers. Authenticated retained-vector
[Q8 patterns42003109155](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42003109155)
preserve all top8 ranks: original ID27 probability0.821857 becomes0.826126;
ID20353 remains second. [Q4 patterns42003112983](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42003112983)
reverse the two winners: ID20353 probability0.525659 versus ID27 probability
0.245700; table-minus-figure gap changes from-2.125 to+0.760540. A positive
affine fit (slope0.979528/intercept0.406107) leaves RMS0.428556/max2.468551
residual, including winner residuals-1.635999/+1.206038. This rank reversal is
not explained by a global positive scale or offset. Fitting is diagnostic;
no logits, arrays, thresholds or inference were changed. Exact source/artifact/
row proofs and top8 probabilities remain in the [Q8](../tests/results/qwen35-ocr/lightonocr3-4b-q8-score-patterns.json)
and [Q4](../tests/results/qwen35-ocr/lightonocr3-4b-q4-score-patterns.json) receipts.

| Retained native path | Q8 max/RMS/cosine | Q4 max/RMS/cosine |
|---|---|---|
| Legacy (`fd150b98`) |0.568259/0.130495/0.998469710|2.494229/0.571913/0.976987541|
| Direct quant (`6d94fd9b`) |0.647667/0.170557/0.997532368|2.658515/0.606225/0.974644244|
| BF16 vision (`891d07da`) |1.092199/0.200537/0.996400595|2.610071/0.545670/0.978054762|

All six raw numerical results fail; each retains finite248,320-element scores
and unchanged Q8 `<figure`/Q4 `<table>` two-token prefixes. Direct-quant
[Q842004610956](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42004610956)
and [Q442004615158](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42004615158)
are numerically worse than legacy on this position. BF16-vision
[Q842006393935](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42006393935)
and [Q442006480602](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42006480602)
do not establish strict parity or repair the Q4 prefix; Q4 RMS improves while
its maximum error worsens. Local operator/component improvements are not a
composed decoded-quality claim. Both new paths stay default-off; old full
captures and every rejected result remain unchanged. These observed-form
findings may not select tensors or tune a new heldout panel.
## Final complete plain-Q8/Q4 comparison

Actual [Q842012564139](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42012564139)
and [Q442012559942](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42012559942)
authenticate and compare every177 original-prefix row, all248,320 vocabulary
scores per position. All rows are finite; both have0/177 strict passes and
first observed/budget failure0. Q8 maximum absolute error is2.93674088 and
minimum row cosine0.958800077; Q4 maximum11.067977 and minimum0.666163504.
Q8 independently ends at160 IDs/EOS with changed French figure text; Q4 reaches
cap256/noEOS with an incomplete table, versus original177/EOS. Exact raw
scores/IDs/text proof, magnitudes, source pins and initial failed harness
attempts are linked in [the final umbrella](lighton4b-qualification-consolidation.md).
The retry workflows are red because numerical/decoded rejection is expected;
their drivers explicitly report `status=compared`, not a harness failure.
No original model forward was repeated, no F32 reference was required, and no
threshold, source output, prompt, cap or default was changed to pass a gate.
