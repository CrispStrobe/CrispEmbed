# LightOn4B qualification evidence umbrella

This child of [runtimePR155](https://github.com/CrispStrobe/CrispEmbed/pull/155)
consolidates retained dense arithmeticPR152, original BF16 prefillPR157 and
independent four-case original capturePR161. The
[source/hash manifest](lighton4b-qualification-consolidation-v1.json) preserves
exact source heads and copied probe/receipt bytes. Historical runtime code was
not imported: every `src/` byte, the ggml gitlink and VERSION remain identical
to runtime spine `8eec65b865003c643a848af8aa8dd2fe3143be1f`. Producer branches
and already running jobs remain unchanged; retain their source refs when
closing superseded PRs.

Included evidence preserves source BF16 free-generation and observer-neutral
first-prefill scores; all71 original stage records; shared BF16/F32 vision
errors; Q8/Q4 recipe divergence from decoder layer0; exact raw first-score,
rank/margin and gated-path negatives; separate receipt-bound full admissions;
predeclared independent image inputs; mixed byte-packing audit; and the PR161
original capture script. Local-only negative BF16-vision evidence from
`2a62e29e771eb747bee7e867789c147f8a36ec28` is retained byte-for-byte.

## Complete native captures

| Native run | Teacher positions | Independent result | Parity against original177-ID EOS figure |
|---|---:|---|---|
|[Q842001328483](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001328483)|177|160 IDs, EOS|Different French figure text; first ID difference9|
|[Q442001333207](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001333207)|177|256 IDs, cap, no EOS|Incomplete HTML table, truncated at `<td`|

Exact native receipts and fidelity metrics are retained in
[the full review](../tests/results/qwen35-ocr/lightonocr3-4b-full-native-review.json).
Both authenticated captures are admitted only to diagnostic comparison. They
are not quality qualified. Fixed0.002/0.99999 numerical gates fail at every one of177 positions for both
recipes; the complete authenticated Ref comparisons are recorded below. The protocol SHA
`4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc`
is unchanged. No F32 original is required and no working alternative/default,
model alias, runtime release or app pin changes.

## Validation and remaining work

Existing exact-source hosted proof includes dense trace41984176263 (native
route observed; Torch internal order unproven), tokenizer41999152372 (154
prompt IDs,177-ID decode, seven Unicode/added-token cases, six negatives),
registry42001553561 (three methods with actualRef/digest/registry negatives),
input freeze42004846562 (all four PNG/RGB hashes), and packing42006450732
(two controls plus all723 persisted payload hashes). First-score and BF16
candidate-stage failures remain explicit. The independent four-case original batch is complete, with the capped receipt
and logo coverage limits retained. One separately authorized lean calibration
job42013855293 now covers only table/Arabic; no heldout qualification is claimed.

This combined tree received static Python AST/YAML, exact copied-file hashes,
`src/`/ggml/VERSION identity and whitespace review. Combined-tree CI was not
run during the user-requested pause. Historical secret-free extraction and
attribution jobs remain isolated; a separately selected read-only job supplies
Actions/Kaggle read credentials only to the three fixed reference comparers.
Do not broaden credential forwarding or treat old producer results as new
combined-source qualification. After the single `[skip ci]` consolidation push,
only the specifically authorized Q4 and Q8 full comparers may be dispatched;
all unrelated CI and new PRs remain paused.

The initial targeted full-comparison jobs42011206392/42011236969 stopped before
`main()` with an omitted legacy I/O import, comparing zero positions and
emitting no artifact. A reviewed failure record preserves that distinction
from numerical rejection. The scoped repair extracts only bounded standard-
library score I/O, preserving the exact diagnostic GGUF encoding, and restores
the exact standalone Ref C++ file/target. Both comparer CLI import/help checks
then succeed without a model, suite or CI matrix; all runtime bytes stay intact.
Only targeted retries are authorized.

The completed independent BF16 batch is retained from local evidence
`b05ff20325e113eefb27c2a59def235d00df22f0`: table222/EOS, Arabic6/EOS,
receipt256/cap/noEOS and logo65/EOS (549 score rows,545,310,720 bytes remotely).
The frozen case named `handwritten_letter` is actually a red serif A logo;
it does not cover handwriting. Both this input limitation and heldout receipt
completion failure remain explicit. Frozen cases, prompts and caps are not
changed after observing outputs; no new native panel jobs were dispatched.

## Complete authenticated177-position Ref results

[Q8 replay42012564139](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42012564139)
and [Q4 replay42012559942](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42012559942)
complete successfully as comparers and intentionally return failure for the
rejected numerical/decoded gates. Each authenticates the full immutable native
archive, exact admitted native receipt, originalKaggle `v1` capture and every
one of177 original/native score payloads. All43,952,640 value pairs per recipe
are finite; neither comparer repeats model/reference inference.

| Precision | Strict positions passed | Maximum absolute error | Minimum row cosine | First score failure | Final decoding |
|---|---:|---:|---:|---:|---|
|Q8_0|0/177|2.93674088 (position119)|0.958800077 (position123)|0|160 IDs/EOS; different text|
|Q4_K|0/177|11.067977 (position5)|0.666163504 (position5)|0|256 IDs/cap/noEOS; incomplete table|

The exact unchanged Ref executable SHA is
`d6898491984110b0536dcd43ddb1091f6559495a79dfb23389baedb8dc4ae246`.
Raw per-position errors, RMS, both magnitudes, complete ID arrays and full
original/native decoded texts remain verbatim in the
[Q8 comparison](../tests/results/qwen35-ocr/lightonocr3-4b-q8-full-comparison.json)
and [Q4 comparison](../tests/results/qwen35-ocr/lightonocr3-4b-q4-full-comparison.json).
[The independent review](../tests/results/qwen35-ocr/lightonocr3-4b-full-score-comparison-review.json)
recomputes maxima/minima/counts and records source/report hashes. At the Q8
maximum-error row reference/native norms are1213.35876/1152.56775; at the Q4
maximum they are1030.22205/1185.22864. The separate initial capture review's
reference-relative CER/WER is0.2134146/0.3888889 forQ8 and0.8277439/0.9888889
forQ4; those are fidelity to the original output, not human OCR accuracy.
Both candidates remain rejected for parity; no precision/model/default is promoted.

## Completed sequential mixed calibration

[42013855293](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42013855293)
at`a3671602edfadb44a86bc8785fff3d34f94acc02` completed both predeclared calibration
cases sequentially with the exact attention-Q8/FFN-Q4 candidate. The red workflow
is an **actual numerical/decoded rejection**, not a harness error or timeout.
Both native input preflights matched the authenticated official pixels, tokens,
modalities and grids before model loading; both independent generations reach EOS.
All228 original-prefix score rows (56,616,960 value pairs) are finite, but none
passes unchanged0.002/0.99999 numerical gates. The unchanged actualRef executable
SHA remains`d6898491984110b0536dcd43ddb1091f6559495a79dfb23389baedb8dc4ae246`.

| Case | Original/native IDs | Decoded result | Maximum absolute error | Minimum row cosine |
|---|---:|---|---:|---:|
|simple_table|222/218, both EOS|Different English/French figure descriptions; first ID difference7|5.72646046 at5|0.886965096 at14|
|arabic_printed_line|6/6, both EOS|Exact IDs/text: `أبجدية عربية`|2.24062681 at0|0.978771031 at1|

The table's raw reference-output CER/WER is0.6738056013/0.9622641509; Arabic is0/0.
These compare frozen generated outputs, not human OCR transcriptions. The table
changes language and numerical content; complete original/native strings, IDs,
per-position metrics, hashes and limits are preserved in the
[review](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-calibration/review.json),
[actual comparison](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-calibration/comparison.json)
and [decoded differences](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-calibration/decoded-differences.json).
At maximum table error, reference/native norms are1178.21594/1344.91064;
Arabic's maximum-error norms are1052.64636/1064.06604.

The exact4,683,041,248-byte mixed model SHA is
`d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474`.
Run artifact13413777229 retains599,197,464 bytes with GitHub-declared archive
SHA`989e25be9af1d2ad280be8c76efc24f437258d4bd2d9767a548b1fb622d91d37`.
Only792,422 ZIP-range bytes were fetched for small reports/text; arrays remain
remote. There was no native heldout run. The heldout original receipt still caps
without EOS, and the frozen logo case does not establish handwriting coverage.
Arabic's small decoded success does not qualify this recipe or prove an improvement
against plain Q4 on the same input. Original internal stages were not captured
for this panel, so a score failure is not an earliest-operator diagnosis. No
threshold, cap, prompt, default, alias or runtime release changes follow.
