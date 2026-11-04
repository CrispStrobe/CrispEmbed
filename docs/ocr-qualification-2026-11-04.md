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
pass; check the remaining macOS Build leg before integration. These are actual
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

## Execute next: source-correct LightOn3 vision norm

Pinned Pixtral uses1e-5 in its three vision RMSNorm call sites; native used1e-6.
Exact pixels precede patch-conv max9.06e-6, first norm max7.33e-5 and vision
layer23 max4898.416 at reference RMS2834.331. Form-grounding diverges at generated
position9, ID24 versus23: box `999` becomes `899`. This is why a high final cosine
is insufficient. Fix math before searching quantization recipes.

[The frozen A/B protocol](lighton3-pixtral-epsilon-ab-v1.md) and
[JSON source ownership](lighton3-pixtral-epsilon-ab-v1.json) define the exact
retained original archive, model hashes, first-stage barrier and full continuation.
Only the new LightOn3-1B profile can opt into
`CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS=1`; default/0 and legacy LightOn stay unchanged.
All version surfaces now identify0.17.15 development, unreleased. No tag,
application upgrade, publication or default flip follows from this experiment.

First run41885090925 stopped before build/acquisition/inference because shallow
checkout lacked the baseline object. The fixed-fetch retry **41885702781** at
`f27223fc276674c4621f32b709aa03ef0d035f09` is executing. Check it first.
It uses zero original forwards and twelve independent native request processes:
form-grounding × F32/F16 × gates0/1 × forced/observed-free/plain-free.
Gate0 must replay every retained tensor byte and non-timing decoded trace field;
gate1 must improve first-norm error, pass unchanged final bars and match original
free IDs/text. No budget, gold, prompt, cap or original equation may be changed.

After independent positive first-artifact review, launch the full command in
the protocol document with the exact first run, small-artifact ID and ZIP SHA.
The remaining five cases require60 new native requests,24 combined gate/precision
rows and72 requests across both stages. Keep all strict failures and cap statuses.
If either stage is negative, preserve it and attribute the earliest differing
operator; do not weaken replay or decoded checks. Full default/public-consumer
qualification is a separate decision after this bounded experiment.

## Execute next: MinerU seven remaining cases

All use the unchanged producer `1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6`, branch
`audit/mineru2605-aligned-q4-20261104`, script
`tools/ci-heavy/mineru2605_qualify.sh`, arguments
`CASE f32,f16,q8_0,q4_k 2048 3 1`, portable, ubuntu-latest, distinct isolated lanes.

| Case | Run | Last reviewed execution state |
|---|---|---|
| formula |41877117676| success; small-report extraction/review pending |
| japanese |41877121416| success; small-report extraction/review pending |
| arabic |41877125216| success; small-report extraction/review pending |
| table |41877129385| success; small-report extraction/review pending |
| german |41877132652| in progress |
| layout-page |41877137082| success; small-report extraction/review pending |
| layout-rot90 |41877141270| in progress |

The [machine-readable panel ledger](../tests/results/ocr-benchmark-expansion/mineru2605-aligned-panel-20261104.json)
records all eight run IDs. Add exact finished artifact IDs/sizes/digests and each
producer's exact branch to the closed follow-up extraction list, retain the
existing archive/security helper unchanged, run hosted extraction once, then
verify every small-file hash. Do not download large original archives locally.
Review all32 arms: all original positions/full vocabulary, precision bars,
free IDs/text/EOS/caps, repeats, helper outputs, actual tensor census and matching
candidate hashes. Green job status alone is not qualification. If a case fails,
retain it without fitting the frozen recipe on held-out failures.

## Remaining tasks and completion gates

1. **LightOn3-1B Q8/Q4:** after the epsilon experiment, freeze the chosen math/source
   and add a real `crispembed-quantize` branch. The current converter `--dtype`
   accepts F32/F16 only; do not pass quantized names to it. Preserve critical
   tensor bytes at declared precision, record actual role census and compare
   original full-position logits, earliest-stage errors and free outputs.
2. **Xiaomi EOS/document + Q8/Q4:** declare a new cap/EOS/corpus protocol; retain
   the successful16-token panel separately. Add forced-prefix all-vocabulary
   dumps when quantized free IDs diverge. External PPDocLayoutV3 detector
   composition needs its own original/native document experiment and annotations.
3. **LightOn0.8 retained continuation:** authenticate all905 old arrays, source,
   image, prompt, cap, token IDs and score coverage; run native only against
   retained references with the historical guard failure explicit. Keep all
   three original cases distinct. Original CPU reproduction is a separate
   attribution experiment, never a blind original rerun or ignored-field waiver.
4. **Model publication:** only after required task/public-interface evidence,
   export standard GGUF metadata (`--standard-file-type` where applicable), prove
   per-tensor payload preservation, run metadata/parser/readback checks, upload
   immutable Hub assets, empty-cache readback, SHA-pinned registry aliases and
   Python/Rust/Dart/CLI consumers. Current scratch GGUF hashes are not public aliases.
5. **Other bounded work:** native Nemotron resize/learned positions/neural backend,
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

Original failure archives expire18 November2026. Small extraction receipts have
90-day retention, but text does not preserve original arrays. The epsilon raw
artifacts copy the reference arrays selected by each phase; preserve all still
needed uncopied references remotely before expiry. Never include weights,
credentials, private machine locations or operator identities in public Git.
