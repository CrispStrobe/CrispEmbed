# OCR qualification continuation — 4 November 2026

This is the current executable entry for [PR141](https://github.com/CrispStrobe/CrispEmbed/pull/141).
Read it before the historical dispatch tables in [the expansion record](ocr-benchmark-expansion.md).
Re-query exact run IDs before dispatching; do not duplicate a running case. Large
archives, conversions, builds and inference stay hosted. Keep all negative results.

## Regression review

Main was fast-forwarded to `2f748825ff3b56b8f2b89d156986b94218ae28c3`.
Published v0.17.14 native/WASM assets, scheduled OCR run41823109685 and main-health
41848426718 pass their bounded checks. On PR141 source `f27223fc2`, hosted native,
mobile, WASM, Rust, inpainting, formatting, environment and model-free controls
pass. Hosted [Build41899765931](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41899765931)
at sourceea2edb5bf now passes every selected job, including macOS/iOS. The earlier
Apple capacity failures remain historical evidence; the M1 preflight correctly
stopped under pressure without builds/inference. No additional M1 work is needed
for this build milestone. These are actual
build/control results, not an exhaustive all-model regression guarantee.

The historical MoEdern Q8 source-closure test rejects the changed family engine
(7/8 controls pass). PLAN records both exact hashes and its current-source
qualification task. Preserve the old closure; do not silently repin it. No new
numerical regression is inferred from that guard failure. The 51 UltraBERT,
13 base MoEdern, 13 Josev (one additional skip) and 18 main-health controls passed
the bounded review. Existing strict numerical debts remain open.

## Completed bounded evidence

- **Xiaomi0:** metadata/frontend41870593556 is pixel/patch exact. Neural run
  41871383235 completes two originals and four F32/F16 arms; all four final-logit
  bars and decoded IDs/text/stops pass. Minimum cosine is0.9999999999994335;
  six of160 strict intermediate comparisons fail. Every output reaches the
  declared16-token cap. [Full report](../tests/results/ocr-benchmark-expansion/xiaomi0-inference-41871383235.json)
  and [review](../tests/results/ocr-benchmark-expansion/xiaomi0-inference-review-41871383235.json).
- **MinerU2605 receipt:** aligned recipe run41871894063 at frozen `1bb0ddb1d`
  completes all208 positions and exact EOS-ended decoded requests at every
  precision; same-backend helper JSON/Markdown agree on8 blocks/9 requests.
  Minimum cosines F32/F16/Q8/Q4-mix are0.9999999999966631,
  0.9999999999964981,0.9993957644539981,0.9916272908500863. Q4-mix is
  1,278,646,496 bytes:384 F32,273 Q8_0,24 genuine Q4_K tensors. Unaligned896-wide
  matrices explicitly stay Q8; the parser still rejects undeclared Q4_0 fallback.
  Strict intermediate failures remain, worst row cosine0.399606049.
  [Review](../tests/results/ocr-benchmark-expansion/mineru2605-aligned-review-41871894063.json)
  preserves hashes, artifact bindings and full precision reports. Do not promote
  a model from this single case.
- **LightOn3-1B initial full panel:** exact namespace repair `4476be6eb` loads all
  532 checkpoint tensors into533 model keys, with only the declared tied-head
  omission. Run41873656972 passes all12 F32/F16 final-logit bars, but only10
  decodes match; form grounding differs at both precisions, and8 arms hit128
  tokens. [All twelve comparisons and reviewed original summary](../tests/results/lightonocr3-1b/native-initial-41873656972/review.json)
  retain the negative decoded/strict fields. The genuine vision RMSNorm epsilon
  mismatch and its gated repair are described below.
- **LightOn3-0.8B:** authenticated extraction41869721715 identifies1759 changed
  CPU/probability fields, maximum selected-probability delta1.72895655e-6;
  IDs/text/all other original fields match. The historical exact guard remains
  failed and native never ran. [Complete differences](../tests/results/ocr-benchmark-expansion/lighton08-original-diff-41810175624.json).
- **Nemotron frontend:**41810180035 passes11 geometry/packing cases,
  16,665,600 values with no changed values, and9 negative controls. It covers
  original-processor resized pixels, not native resizing or neural inference.

## Current LightOn result and cache experiment

The [complete epsilon review](../tests/results/lightonocr3-1b/epsilon-ab-v1/full-review-41889499814.json)
is a negative acceptance result, not an executing job. Run41889499814 at0473dd5e8
completes24 rows/72 native processes across both phases, with zero new original
forwards. Every cache-old gate0 tensor/decoded trace reproduces exactly. Corrected
epsilon improves first norm and passes all12 final numerical bars; F32 matches
all6 original free outputs, F16 matches5. Form-transcription F16 changes at
position6, ID2918→3377, and the generated description switches English→French.
Strict failures and8 capped outputs remain. The epsilon correction stays default-off.

Native always stores persistent K/V in F16, including when the model weights are
F32; the pinned original uses float32 with caching. A separate
[CPU cache-precision first-case protocol](lighton3-kv-f32-ab-v1.md) tests that
precision difference before any tensor recipe. Source3e82a1205/run**41897136488**
retains10 calls before its prefill check incorrectly includes postdecode IDs.
The [two-call completion41901879295](lighton3-kv-complete-first-v1.md) passes,
with an exactly reproduced executable and unchanged old receipts. Corrected
F32/F16 free IDs/text now match; minimum cosines0.99999999994/0.99999999992,
step6 max errors0.000143/0.000101. The separate5-case remainder **41905258340**
at558825720 is running, imports4 first rows and schedules60 new native calls.
No first-case repetition or new originals. The fixed experiment uses: fixed corrected epsilon1, unchanged F32/F16 GGUFs, form-transcription
only, cache0/1 × forced/observed-free/plain-free,12 native requests. Cache0 must
reproduce the retained negative, cache1 must keep prefill bytes exact, improve
step6 errors and match original free IDs/text. It records full-vector hashes and
original/native top2 margins; component probes are not quality qualification.
No new original forwards, default or publication follows. The positive first result opens only the actual remaining continuation above.
Review its24 combined rows/72 native calls before any default decision.

## Completed MinerU eight-case panel

[Complete independent review](../tests/results/mineru2605/aligned-panel-20261104/eight-case-review.json)
aggregates all8 frozen cases at1bb0ddb1d:32 numerical arms pass,6192 full151936-wide
native logit positions cover1548 original positions. Every request reaches EOS
without its2048-token cap; all8 same-native-backend helper JSON/Markdown pairs
match. This helper proof is separate from original per-request neural parity.

| Format | Minimum final-logit cosine | Exact decoded cases | Actual GGUF bytes |
|---|---:|---:|---:|
| F32 |0.9999999999678552|8/8|4630070400|
| F16 |0.9999999999673693|8/8|2591511712|
| Q8_0 |0.9985343305283868|6/8|1330944224|
| aligned Q4_K/Q8 |0.9902117584412018|6/8|1278646496|

All candidate hashes match across cases, and actual Q4 census remains
384F32/273Q8/24Q4_K with no Q4_0 fallback. All strict intermediate failures remain.
Layout-page Q8/Q4 each change a coordinate digit. Rotated-layout Q8 changes two
coordinates (maximum ordinal delta0.015); Q4 changes14/25 ordinal regions,
including5 labels (max delta0.172), with551 versus550 tokens and65 character edits.
These are retained-original comparisons, not annotated layout scores. Numerical
acceptance does not hide those structured-output negatives.

| Case | Producer run | Review/extraction |
|---|---|---|
| receipt |41871894063|41876269745, all4 numerical/decoded pass |
| formula |41877117676|41889185669, all4 numerical/decoded pass |
| japanese |41877121416|41889185669, all4 numerical/decoded pass |
| arabic |41877125216|41889185669, all4 numerical/decoded pass |
| table |41877129385|41889185669, all4 numerical/decoded pass |
| german |41877132652|41895937896, all4 numerical/decoded pass |
| layout-page |41877137082|41889185669, all4 numerical bars; Q8/Q4 decoded differences |
| layout-rot90 |41877141270|41891577778, all4 numerical bars; Q8/Q4 decoded differences |

The [execution ledger](../tests/results/ocr-benchmark-expansion/mineru2605-aligned-panel-20261104.json)
and per-case reports retain full rows, artifact identities, strict errors,
complete token/text differences and tensor census. All selected-member ZIP hashes
and400+80+80 text members were verified in addition to the first receipt's80.
Do not repeat the original panel or fit a new recipe on these frozen negatives.

## Independent Apple and downstream evidence

Earlier GitHub Apple jobs never acquired runners; Build41889197876 correctly
failed its aggregate. M1 preflight then reported8180 new swapout pages and about
16GB swap in use, stopping before any contract/library/style/inference call.
Only4 owned/11 protected source checks at0473dd5e8 were reported exact. Those
facts remain preserved; no work was forced onto the pressured machine.

Hosted capacity subsequently returned. Build41899765931 at**ea2edb5bf** and all
ordinary checks are green, including actual macOS/iOS libraries and279 model-free
controls. The current runtime source files are unchanged from that tested head.
This closes the native compilation milestone, not Apple model-inference parity.
The completion's8 additional focused controls also passed on its hosted runner.

[CrispSorter PR20](https://github.com/CrispStrobe/CrispSorter/pull/20) is open at
0af4c342f0cef8e4fda1b629d5b9fdfc82ed15b4. Its
[reported feature check](https://github.com/CrispStrobe/CrispSorter/pull/20#issuecomment-6639976078)
completes `cargo check -p crispsorter --lib --features crispembed-metal` against
the actual immutable0.17.12 pin, exit0 in27m53s. The prior wrong-sibling API errors
are retracted; image-indexing backfill is on that branch. This is useful stable
consumer/type-check evidence. Current0.17.15 experimental backend compilation,
production linking/loading and live OCR are separate. PR20 still needs integration;
its Windows helpers/live-model checks remain unexecuted, and related consent PR18
is draft. No downstream source or pin was changed by this OCR session.

## Remaining tasks and completion gates

1. **LightOn3-1B Q8/Q4:** after cache attribution and its complete panel, freeze the chosen math/source
   and add a real `crispembed-quantize` branch. The current converter `--dtype`
   accepts F32/F16 only; do not pass quantized names to it. Preserve critical
   tensor bytes at declared precision, record actual role census and compare
   original full-position logits, earliest-stage errors and free outputs.
2. **MinerU quantized layout:** preserve the frozen panel, including rotated
   Q4's5 changed labels/14 changed ordinal regions and max0.172 coordinate delta.
   Declare separate licensed calibration and held-out data before a new mixed
   recipe. Compare same-input Python/native stages and Q8-versus-Q4 added error
   with the dumper/diff harness; retain precisely identified sensitive down
   projections/vision roles at higher precision with payload hashes. Freeze
   actual C-quantizer output/census and qualify full logits plus free structured
   output on held-out data. Do not tune on or overwrite these old negative cases.
3. **Xiaomi EOS/document + Q8/Q4:** declare a new cap/EOS/corpus protocol; retain
   the successful16-token panel separately. Add forced-prefix all-vocabulary
   dumps when quantized free IDs diverge. External PPDocLayoutV3 detector
   composition needs its own original/native document experiment and annotations.
4. **LightOn0.8 retained continuation:** follow [the exact retained-native implementation scope](lighton08-retained-native-continuation.md),
   authenticate all905 old arrays, source,
   image, prompt, cap, token IDs and score coverage; run native only against
   retained references with the historical guard failure explicit. Keep all
   three original cases distinct. Original CPU reproduction is a separate
   attribution experiment, never a blind original rerun or ignored-field waiver.
5. **Model publication:** only after required task/public-interface evidence,
   export standard GGUF metadata (`--standard-file-type` where applicable), prove
   per-tensor payload preservation, run metadata/parser/readback checks, upload
   immutable Hub assets, empty-cache readback, SHA-pinned registry aliases and
   Python/Rust/Dart/CLI consumers. Current scratch GGUF hashes are not public aliases.
6. **Other bounded work:** native Nemotron resize/learned positions/neural backend,
   legacy LightOn epsilon attribution against its own pinned original, and the
   MoEdern current-source Q8 closure remain separately scoped in PLAN. Existing
   broader model, release and downstream work remains intact.

## Resources and preservation

The measured Kaggle CPU-only capacity probe verifies network access and separate
scratch/output filesystems for that exact worker; no model inference was run.
[Public receipt](../tests/results/ocr-benchmark-expansion/kaggle-cpu-capacity-20261104.json)
and [probe source](../tools/kaggle/ocr_cpu_capacity_probe.py) preserve evidence.
The optional MinerU CPU adapter reserves all four retained GGUFs; no duplicate
Kaggle model batch was launched while GitHub's seven cases were active.
[The M1 fallback](handover-m1-mineru2605.md) separates build-only checks and the
not-yet-implemented Darwin provider adapter; no M1 run is claimed.

The original18 November expiry no longer threatens these ten ZIPs: both
[raw preservation41894992782](../tests/results/ocr-benchmark-expansion/raw-preservation-41894992782.json)
and [German supplement41897911047](../tests/results/ocr-benchmark-expansion/raw-preservation-41897911047.json)
complete authenticated byte-copy and full remote readback, zero model/build calls.
All8 MinerU captures and both initial MinerU/LightOn0.8 failure archives retain
unchanged ZIP bytes through**2 February2027**. The two small receipts were
independently checksummed and reviewed; large archives never reached the shared
host. Raw artifacts are13355747765 (9919709356 bytes, SHA256
`43f955f22569f20557075612943ba6f9f6fa1c1427b3ee7f860b7b762cb77ab0`)
and13357368062 (2543809832 bytes, SHA256
`36f78190bb14412005f47b5dffefc2ff712d6b1e596c540003542a2c95bba77a`).
They contain original ZIPs as members; do not confuse them with text-only reports.
A later native continuation must explicitly authenticate this preserved transport
and each inner original hash, without weakening historical model/source guards.

The new LightOn epsilon/cache raw artifacts preserve reference arrays for90d;
the current remaining panel must retain its copied references and results too.
Keep weights, credentials, private machine locations and operator identities
outside public Git. No new release/model publication follows from preservation.
