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
are not quality qualified. Fixed0.002/0.99999 first-score numerical gates already
fail; complete177-position Ref comparisons remain separate. The protocol SHA
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
candidate-stage failures remain explicit. The independent four-case original
worker is still executing; no new native case jobs were dispatched after the
user requested queue consolidation.

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
