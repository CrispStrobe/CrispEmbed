# CrispEmbed plan — open work (updated 5 November 2026)

The user requires LightOn4B Q4 quality at least comparable to the measured
llama.cpp Q4_K_M baseline before shipping a Q4 model. The existing plain Q4,
custom mixed recipes and weighted experiments are unqualified; keep their gates
default-off and do not publish, register, or use them as an app release. The
[pinned independent baseline](docs/lighton4b-independent-llama-baseline.md)
uses the same frozen form, table and Arabic inputs, with Q4_K_M form CER/WER
189/656 and 46/90, table 315/607 and 88/106 (capped), and Arabic 0/12 and 0/2.
On those diagnostics compare complete decoded text, EOS/cap, CER and WER against
the identical BF16 reference and the upstream outcome; record both wins and
losses. These are not human transcription accuracy. Require additional fresh,
representative OCR cases before any broader quality or release decision. Preserve
the distinction between plain Q4_K, llama.cpp Q4_K_M and each explicit custom
recipe; Q6 or F16/BF16 tensors may be used where measured and declared. The user
selects the BF16 original reference: an F32 model/reference is not a prerequisite.
Use Kaggle rather than GitHub-hosted runners for this work.

The bounded default-off `qwen35_lighton4b_text.h` component maps only the exact
426 text tensors from immutable LightOn4B source
`b71010f095dc3735de7ee6b9969bff8f8f50b39a`: 32 layers, full-attention and
16-key/32-value-head recurrent geometry, tied head and strict shape/type/identity
ownership. It leaves the old mapper, source closures and format-5 admission
unchanged. Hosted `tools/ci-heavy/lighton4b_text_owner.py` must compare all owner
identities against the independently captured original checkpoint header and
reject per-tensor shape/type/contiguity/missing/alias drift, with no payload
allocation. Actual [hosted41929343665](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41929343665)
at `291b0a171` passes all3,498 controls; independent review verifies426 unique
original rows,249 matrices/177 protected tensors and the exact control census.
The [repaired hosted41930138239](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41930138239)
also passes3498 owner and36 Python protocol/closure controls after clean-exit,
formatting and synthetic historical-fixture corrections; original source guards
remain unchanged and every initial PR failure is retained in HISTORY. Final-source
CI at `2d36eeb94` now passes all13 selected workflows, including actual Build
aggregate41930954667; the [receipt](tests/results/qwen35-ocr/lightonocr3-4b-text-owner-ci.json)
closes the initial source-CI follow-up without changing historical guards.
This is metadata ownership only. Vision mapping, actual composed
decoder dispatch/cache behavior, bounded full-vocabulary projection, original
intermediates/decoded outputs, tokenizer and public inference remain separate
gates; no native model support or quality claim follows.

Only open or active work lives here; completed runs move to [HISTORY](HISTORY.md) with their
receipts, measured numbers go to [PERFORMANCE](PERFORMANCE.md). Start with
[the 3 November handoff](docs/handoff-2026-11-03.md) and
[current state](docs/current-state-and-next-steps.md). Branches named below are merged
into `main` unless the row says otherwise; start new work from `origin/main`.

| Lane | State | Next action | Done when |
|---|---|---|---|
| OCR model gaps / LightOnOCR-3-4B (`feat/ocr-next-models-20261104`, [PR143](https://github.com/CrispStrobe/CrispEmbed/pull/143)) | Main `17c96639`/PR141 evidence is integrated; all seven workflows pass at `91b24764`. Explicit converter has 16 passing export controls at `aa661568` (333 total); its inline workflow-contract regression is repaired. [Default-off grouped-head APIs and pinned original/meta protocols](docs/lightonocr3-4b.md#default-off-grouped-head-component-and-hosted-reference) are implemented | Review hosted meta construction, 208 original/native component stages and cache controls, plus combined regression CI. Then add bounded vocabulary projection and explicit larger-model ownership | Native support remains false until complete original/native F32/F16 and document evidence, followed by separate precision/publication gates |
| Q4_K_MIX publications (MiniCPM-OCR, Unlimited-OCR-2, MoEdern 27M/115M, Josev, UltraBERT) | Recipes and receipts on main. TeleOCR and MinerU Q4_K_MIX are already published and registered (`teleocr-q4_k-mix`, `mineru25-pro-q4_k-mix`, PR #116); the rest are not | Coordinator dispatches the cosine-gated publication per model (sections below); Josev needs a quantized branch in `josev_publish.py`, MoEdern needs SciFact/client panels first | Hub upload, empty-cache readback, registry alias with SHA-256 |
| `env_gate.h` false/off/no lockstep | Claimed by Codex; CrispASR #572 and companion update in progress | Handoff 3 Nov lane B | Identical header in both repos, tests in CI, sibling sync green |
| OXR native-doc parity | Rerun 41664201815 failed again (`native=fail`, not the trace-key bug) | Handoff 3 Nov lane C | Five-case parity receipt or scoped runtime bug |
| Unlimited-OCR-2 Q8_0 on CUDA | 0.9555 < 0.97 (CPU 0.9739) | Search a CUDA-qualifying Q8_0 mix (F16 shared experts/attention) | ≥ 0.97 on CPU and CUDA |
| GOT-OCR2 Q4_K drift | F16-r2 exact vs original; Q4_K diverges at 83/20; no final-logit cosine measured, so the 1 November precision policy cannot yet be applied | Teacher-forced final-logit cosine of the published Q4_K vs the F16-r2 pipeline (as for Unlimited/MiniCPM), then a mixed search only if < 0.94. Owner: the model is research-licensed (Tongyi), so whether a Q4_K is worth more compute is an owner call (handoff "Owner decisions") | Cosine > 0.94 recorded, or 3/3 token-exact, or owner accepts/declines |
| Server load control (PR124) | Merged, gates default-off | Live A/B with a real GGUF; concurrent latency; chat-route parity | Receipts per item in the section below |
| Reranker residual (~0.2 logits) | Pair layout/truncation fixed (PR122) | Fix the stage-diff harness, Spearman/top-k instead of exact argsort | Attributed or accepted |
| CUDA / Metal audit follow-ups | Matrices in `docs/gpu-support-matrix.md` | One gated GPU path or A/B at a time | Measured A/B before any default flip |
| Ovis, Josev, MoEdern, UltraBERT, LFM2.5, EmbeddingGemma precision | See the 29 October table below | One bounded protocol at a time | Per-row gates in that table |
| Issue 71 inpainting | Engines published (default-off); stream and grain correction [PR162](https://github.com/CrispStrobe/CrispEmbed/pull/162) merged | Representative subtitle/video quality, MI-GAN512 Q4_K caveat (0.9019) | Owner-reviewed quality panel |
| Releases / pub.dev / app pin | v0.17.15 published, pub.dev unqualified, app on 0.17.12 | Qualified assets and fresh consumers first | Release procedure in the pub.dev section |
| OXR-1.0 quantized + GPU | Q8_0-large / Q4_K-head pass the owner bar (cos ≥ 0.998 vs F16, CPU/Metal/CUDA); German diverges 1 token from decode-step activation quantization | Opt-in F32-activation decode A/B (the recipes are already published and registered as `oxr-1.0-q8_0` / `oxr-1.0-q4_k-head`, PR #97) | A/B receipt; German page token-exact or the divergence attributed |
| OXR native document route / D-FINE | Hosted helpers/replay/native pass (41249898971), `dfine` stage failed there; D-FINE regions match original 5/5 but 2.6x slower | Re-run `oxr_native_doc.sh` `dfine native` stages on main; encoder per-op profile | Hosted parity on all five cases; profile receipt |
| MinerU2.5-Pro full native doc route | 8/8 model-backed cases identical (41251475795); default-off | Owner: default-on or not (handoff "Owner decisions"); image-bearing-table fixture | Decision recorded; fixture receipt |
| Marker-style pipeline / TeleOCR | Implemented and bounded-validated, default-off; TeleOCR strict numerics false at every precision | Representative licensed multilingual/layout gold corpus (needs owner-provided licensed annotations) | Corpus frozen; CER/WER/order receipts |
| Model-weight licences in apps | Registry/cards/gates corrected; CrispSorter PR20 at0af4c34 adds fail-closed OCR consent, typed Gemma terms, real0.17.12 pin and image backfill. Its reported Metal-feature cargo check passes; PR18 remains draft | Review/integrate [CrispSorter20](https://github.com/CrispStrobe/CrispSorter/pull/20) alongside18; retain Windows-helper/live-OCR gaps. Check CrispCalc consent separately. No vendor/app pin upgrade | Downstream acknowledgement PRs merged and actual consumer checks recorded |
| **Release v0.17.15** | **Published 5 November 2026** from `d28ca2ea4`; full candidate gates and all 16 native/WASM assets verified | Fresh downstream consumer checks and owner-controlled pin updates; package publication separate | Consumer upgrades reviewed against immutable 0.17.15 artifacts |
| CI selection after PR #134 | Merged; docs/evidence-only pushes run nothing, so the main head may have no Build run | Before tagging: `gh workflow run build.yml --ref main` (and build-wasm, build-wasm-embed, rust-crates, inpaint-cli); confirm the dynamic desktop matrix in build.yml evaluates on hosted runners | One full green Build on the main head |
| Failing main workflows owned by the integration session | Recorded as non-release-blocking on 4 November 2026 (HISTORY): `MoEdern real padded Q4_K diagnostic` stops at authenticated input preparation (run 41587641989), and `Layer0 attention op-by-op same-input replay v1` is a manual experiment with no main run | Re-run only if the authenticated inputs are authorized; otherwise no release action | Recorded; re-run or leave closed |
| Embedder/tokenizer residuals (CUDA audit, PR #115/#122) | Fixed: SPM space collapse, XLM-R pair layout, 512 truncation | `xlmr-ner-hrl` GGUF has no Unigram scores (reconvert with scores); `paraphrase-multilingual` native truncates at 512, original at 128 (honour `max_seq_length`); Ettin reranker unchecked (tokenizer needs transformers 5) | Each matches the original on the 77-text panel |
| EasyOCR residuals (PR #100/#110) | x86 line drops fixed; Python-parity work merged | `scan_page_pd` page output differs between Apple arm64 and x86/CUDA (first diverging region/stage); remaining punctuation/spacing differences vs Python EasyOCR on low-confidence lines | Attributed and fixed, or documented as a float-order tie |
| MI-GAN256 Metal/Q4_K | Metal F32/F16/Q8_0 pass; Q4_K fails (masked 0.81); GPU default off | Q4_K probes with sensitive layers at Q8_0 (`CRISPEMBED_QUANTIZE_MIGAN_Q8`); flip to the padded-K Metal path only after measuring delivery-5 CPU kernels on Apple CPU | Q4_K ≥ 0.94 masked or declined; A/B before any default flip |
| Big-LaMa (research only) | Weights' rights unclear; kept on `feat/issue71-lama-research`, private Hub copy only | Nothing until the owner clarifies the weight licence | Licence clarified (then publish) or lane closed |
| Ovis 2048 / final-integrated source | 1024 default; 2048 has six composed final-norm numeric failures with exact text | Same-input attribution (section "Ovis 2048-patch") | Attributed; or 2048 stays unqualified |

## Post-release v0.17.15 — 5 November 2026

[Native/WASM v0.17.15](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.15)
is published from `d28ca2ea43a95934d673d1a54177fdf721614ade` after all full
candidate gates. All 16 assets and seven full hosted native readbacks are verified
in [the receipt](tests/results/releases/v0.17.15.json); completed sources/runs and
limits are in HISTORY. Draft PRs #155/#157 were explicitly excluded from this release
and remain unmerged with their owners. Preserve the tag and source receipts.

Remaining work is downstream consumer qualification and owner-controlled app pin
updates; the application still pins 0.17.12. PyPI/pub.dev publication stays separate
and disabled. Do not infer new model or precision qualification from a release
build. Keep LightOn3 correction gates off unless explicitly enabled, and retain
all prior capped/negative OCR evidence. The current release notes remain in
[RELEASE_NOTES_v0.17.15.md](RELEASE_NOTES_v0.17.15.md); eight older tracked snapshots
were removed while GitHub release bodies and Git history preserve their content.

## Regression review and OCR continuation — 4 November 2026

The [current executable qualification continuation](docs/ocr-qualification-2026-11-04.md)
records reviewed evidence, exact live run IDs, preservation and remaining acceptance
gates. It supersedes historical queued/dispatch language. Heavy execution stays
hosted. Merged PR141 includes source-correct, default-off LightOn3 vision
epsilon/CPU-cache corrections; development0.17.15 is unreleased, published assets remain0.17.14.

A bounded review against main `9c790c81a` found that
`tests/test_moedern_q8_profile.py` passes7/8 controls and rejects its old source
closure: `src/moedern_family_engine.cpp` expected
`e3c95c082a3e8ca4325829dcb2d153664ea017c6eb583a486392df013ca4fc9e`,
current SHA256 `410c27ea7179b346debbabf5257d5b0fc159f026d0f8c7b4274520e3e3641cf3`.
Later mixed-Q4 source `ff1b8bbaf` changed that file. Preserve the frozen protocol
`docs/moedern-q8-f32-critical-v1.json` and its historical evidence. This is a
current-source qualification gap, not evidence of an F32/Q8 inference regression.
Before reusing this harness on current main, review the complete source delta,
declare a separate versioned closure/driver and run hosted authored profile,
operator, storage and gate controls, then original/native vectors, retrieval and
public consumers as required by the model lane. Do not simply replace its hash
or weaken the frozen guard. The base13 MoEdern controls,51 UltraBERT controls,
13 Josev controls (one additional test skipped) and18 main-health controls pass.
These checks and green ordinary CI do not establish all-model numerical parity.

PR141's compilation/regression milestone is complete: all10 ordinary workflows
at**ddb666997** pass, including actual macOS/iOS/native/mobile/WASM/Rust builds,
35 selected checks and294 model-free controls in26 suites. Three declared skips
remain explicit in [the receipt](tests/results/ocr-benchmark-expansion/source-ci-ddb666997.json).
Earlier Apple capacity cancellations and the M1 pressure stop remain in HISTORY;
no further M1 work is needed for compilation. This is bounded regression evidence,
not every-model/Apple-inference quality or a runtime release.

### LightOn4B retained synthetic recurrence replay — 4 November 2026

The [retained-only replay receipt](docs/lighton4b-retained-gdn-replay.md) closes a
test-harness state-reset defect from negative41923599391, without a new original
forward. Corrected41924863763 and independent small-report review41925188488
pass all208 original stage comparisons (52 per case), including full-prefix
output/state and an explicit cleared-cache-null control. Maximum absolute error
is3.397465e-6; minimum row cosine0.999999945275. All checkpoint/export/quality/
alias flags remain false. Preserve the original negative and its frozen arrays;
full-width native ownership, vocabulary logits and decoded checkpoint/EOS parity
remain separate acceptance gates. Temporary review-only workflow wiring is removed.

### Current bounded OCR execution and completion gates

See the [reviewed reports and repairs](docs/ocr-benchmark-expansion.md#reviewed-retained-reports-and-further-repairs)
for exact source commits, artifacts, initial failures and error magnitudes.

| Lane | Current evidence / job | Exact next action and completion gate |
|---|---|---|
| MinerU2605 artifact/public-consumer qualification | Complete8-case panel:32 numerical arms pass,28 decodes exact; F32/F16 all8 exact. Q8/Q4 differ on the2 layout cases. [Complete review](tests/results/mineru2605/aligned-panel-20261104/eight-case-review.json) | All10 raw ZIPs now have verified90d preservation through2 February2027; retain quantized layout negatives. Before any publication finish required task quality/public surfaces, export standard GGUF metadata with per-tensor payload proof, empty-cache Hub readback and SHA-pinned consumers. No repeat panel or promotion from cosine alone |
| LightOn3-4B (`feat/ocr-next-models-20261104`, [PR143](https://github.com/CrispStrobe/CrispEmbed/pull/143)) | Source evidence and 16 export controls pass at `aa661568` (333 total). Five ordinary workflows stopped on one inline-command assertion; original step restored without weakening it. Separate grouped-head APIs, exact original-module/meta probes, 51 protocol/selector controls and hosted job are implemented; old equal-head bodies and source receipts remain unchanged | Run/review the original meta schema and four synthetic GDN cases: 208 stages, 106 named controls, independent native caches. Then bounded full-vocabulary projection; actual conversion/readback, full model ownership, original/native F32/F16 document evidence and publication remain separate |
| LightOn3-1B corrected-default decision | Full cache+epsilon41905258340 passes: all12 corrected outputs exact, worst cosine0.999999999771; strict432/1842 and8 caps remain. Epsilon-only negative preserved; both gates default-off; actual Apple compilation passes | After positive complete cache/decoded panel and actual Apple evidence, adopt/replay the paired CPU changes per [the separate default scope](docs/lighton3-pixtral-epsilon-ab-v1.md#subsequent-default-decision-is-separate): preserve old closures, literal0 old paths/legacy and default GPU isolation, new source controls and actual unset-vs-EPS1/KV1 output replay for all6 modes×2 formats. No new originals or capped-document/publication claims |
| LightOn3-1B EOS/document scope | Current six request modes use cap128;8/12 original outputs are capped, even if corrected native IDs match | Freeze a separate licensed document/EOS protocol. The current `tests/lightonocr3_trace.cpp` and generation validation hard-limit256 tokens: stay within256 or version the harness/comparer with explicit memory/time bounds before a larger cap. Preserve128-token receipts and strict failures; require complete EOS-ended original/native outputs, full-position logits and independent plain generation before document claims |
| MinerU2605 quantized layout quality | Existing aligned recipe passes numerical bars but Q8 changes rotated box coordinates and Q4 changes5 region labels/14 ordinal records (max coordinate delta0.172) | Preserve original eight-case evaluation. Freeze a separate licensed calibration/held-out corpus; use Python dumps and native same-input intermediates/crispembed-diff to locate additional Q4-vs-Q8 error before choosing higher-precision down projections or vision roles. Record every retained tensor's type/payload hash, C-quantizer recipe, full-position logits and free structured outputs. New recipe requires independent held-out layout/EOS/quality/public-consumer evidence; do not fit on the frozen negative cases or replace their verdicts |
| Xiaomi0 | Metadata/frontend41870593556 passes exact pixels/patches; inference41871383235 at3a5a9a1dc passes all4 arms and exact IDs/text at the16-token cap | Bounded two-case integration complete. Preserve6 strict stage failures and capped outputs; freeze separate EOS/document and Q8/Q4 protocols before expanded precision/publication. External detector composition remains a separate experiment |
| LightOn3-0.8B | Extraction diagnoses1759 float-metric/probability differences; IDs/text/all other original fields exact; no native calls | Preserve the old guard failure. Implement/review the [exact retained-capture continuation](docs/lighton08-retained-native-continuation.md), authenticating all905 arrays and old source/input/score identities; no blind original rerun or ignored-field broadening. Keep the first source result and all three-case requirements distinct |
| OvisOCR3 document task | Separate pinned Markdown task/source and cleanup audit exists; the old plain-OCR capped-page failure is preserved | Independent licensed input/gold, actual original task captures and distinct native task admission; upstream two-call Markdown/layout scores are not plain-OCR evidence |
| Nemotron frontend | Actual11/11 geometry/packing and9/9 negatives pass | Bounded milestone complete. Native resize/positions and neural backend remain separate work in the model document |

The original OCR source integration and first-batch dispatch record are archived
in [HISTORY](HISTORY.md#4-november-2026--original-ocr-source-integration-and-first-batch-snapshot).
PR138 is merged. Its old initial failures/queued statuses do not describe current
qualification; use the bounded execution table above and exact retained receipts.

LightOn4B work preserves the existing equal-head runtime and historical source guards.
The new primitive path is default-off through `core_env`; its hosted reference
executes pinned original functions and reports stage magnitudes and absolute
errors. Synthetic component agreement is not model or document qualification.
The exact larger-model converter and 24-layer precision recipe remain distinct
from the historical 12-layer recipe and from native/public admission.

## Release v0.17.14 — 4 November 2026

Owner decision: cut the next release once every PR is resolved and main is green. Nothing is
tagged until then.

Gate status (4 November 2026): **met**. The open-PR queue is empty (#135, #138 and #139 merged),
`main` at `2f1bbdb1` has all 16 workflows green including all 11 native build jobs, and the two
manual diagnostics are recorded non-release-blocking below and in HISTORY. Versions agree on
0.17.14; `tests/test_pubdev_release_gate.py` passes; the changelog entry and
[published release notes](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.14)
record that release. Its old Markdown snapshot was later removed during cleanup.

1. **Gate.** `gh pr list --state open` is empty (or the owner excludes the rest). On the current
   main head, dispatch the full set because docs-only pushes no longer start Build (PR #134):
   `gh workflow run build.yml --ref main`, and likewise `build-wasm.yml`, `build-wasm-embed.yml`,
   `rust-crates.yml`, `inpaint-cli.yml`. All must be green. The two integration-session
   diagnostics in the index above must be green or recorded as non-blocking.
2. **Preflight.** VERSION, `flutter/crispembed/pubspec.yaml`, crate/package versions and
   `flutter/crispembed/CHANGELOG.md` agree on 0.17.14; run `python tests/test_pubdev_release_gate.py`
   (PR #91). Write `RELEASE_NOTES_vX.Y.Z.md` for the candidate; it and the
   changelog entry must list the default changes since 0.17.12 (EasyOCR LSTM state reset,
   GOT-OCR2 pipeline parity, SPM space collapse, XLM-R pair layout, position-table truncation,
   CLIP EOS keep, CUDA defaults for ESRGAN/InstructIR/PAN/Restormer/LightOnOCR, Metal defaults
   for TeleOCR/MinerU/OXR/D-FINE/heron, MiniCPM/Unlimited-OCR-2 speed defaults), each with its
   `=0` opt-out from `docs/environment-variables.md`.
3. **Tag** `v0.17.14` on the verified main head and push the tag; `release.yml` builds every
   platform. Check the "Verify bundled dylib set (macOS)" step (PR #128) and that the release
   carries `crispembed-macos-arm64.tar.gz`, `crispembed-ios-arm64.tar.gz` and the Android/Linux/
   Windows assets. The pub.dev publisher job stays disabled (PR #91 follow-ups).
4. **After.** Tell downstream apps to drop `CRISPEMBED_PREBUILT_VERSION`; Android builds from
   main fetch matching libraries again. App pin updates are separate owner decisions.

Published 4 November 2026 from `9c790c81` (tag `v0.17.14`): 16 assets, including every required
platform archive, both WASM bundles, the JS/WASM files and the CUDA/Vulkan extras. The macOS
archive was downloaded and inspected: `crispembed` and `crispembed-quantize` plus the dylibs are
present, install names are unversioned `@rpath/lib*.dylib`, and `libcrispembed.dylib` reports
current version 0.17.14. Only step 4 (downstream switch) is still open; the pub.dev publish job
stayed skipped because that package is not qualified (PR #91 follow-ups).

Operational note for the next cut: v0.17.15's tag push automatically started
`release.yml`, `release-wasm.yml` and preview Python wheels. Inspect exact tag/SHA
runs before dispatching anything; do not duplicate a healthy automatic build.
If a workflow genuinely did not start, dispatch it at the immutable tag ref
(`gh workflow run release.yml --ref vX.Y.Z`, or `gh workflow run release-wasm.yml
--ref vX.Y.Z -f tag=vX.Y.Z`). Attach the reviewed body with
`gh release edit vX.Y.Z --notes-file RELEASE_NOTES_vX.Y.Z.md` and verify it survives
uploads. Retry only failed transient upload jobs; never move/re-cut a public tag.

## Quantizer changes and the OCR quality contract

Since PR #108, any change to `tools/quantize.cpp` or the pinned Qwen3.5 sources fails the OCR
quality contract until it is recorded: run
`python tools/qwen35_source_pins.py --amend <file> --reason "<why the qwen35_ocr default output is unchanged>"`,
commit `tests/results/qwen35-ocr/source-pin-amendments.json`, and confirm with
`python tools/qwen35_source_pins.py --check`. Amend only with a true reason; if a default path
changed, report it instead. The repo pre-commit formatter rewrites files on commit, so run
`--check` again after committing.

## MiniCPM-OCR Q8_0 / Q4_K — 3 November 2026 (`perf/q4k-minicpm-ocr-20261103`)

Source: `docs/minicpm-ocr-q4k.md`. Final-logit cosine against the original model
(teacher-forced on the frozen IDs, every position, five cases, hosted CPU portable):
Q8_0 0.9998; `minicpm_q4k_v1` 0.9671 (971 MiB); `minicpm_q4k_min_v1` 0.9489 (902 MiB).

- Coordinator: the policy receipt `tests/results/minicpm-ocr/q4k-precision.json` is on main
  (PR131); dispatch the `Q4_K_MIX` publication (cosine gate). Done when the file is on the Hub,
  read back from an empty cache and registered with its SHA-256. (Q8_0 is already published as
  `minicpm-ocr-q8`.)
- F32 arm: closed by decision (3 November). Its stop was fixed in `8c8c8be6` and hosted
  parity is 5/5 on the F32 GGUF; an F32 cosine row adds nothing to the Q4_K publication gate,
  so the arm is not re-run (HISTORY 3 November).
- `para-en` diverges at token 54 in every 4-bit arm (box glyphs read as `×`
  instead of the original `x`/dropped). Representative real-page quality is still open.

## Server usage, load control and OpenAI-shaped OCR — 2 November 2026

Merged as PR124 (fold of 3 November). These are general server improvements. Every new
surface is default-off, except the accounting fix, which has an explicit
opt-out:

- `crispembed_last_prompt_tokens()` C API, plus observed `usage.prompt_tokens`
  on `/v1/embeddings` and `/api/embed` `prompt_eval_count`. These used to be
  hard-coded `0` and text count. `CRISPEMBED_SERVER_USAGE_TOKENS=0` restores
  the old values. LFM2/decision/UltraBERT report "not observed" (-1 → 0).
- `CRISPEMBED_SERVER_METRICS=1` → `GET /metrics` (Prometheus, crispembed_*
  series). "Running" means holding a model lock, counted by `counted_mutex`.
- `CRISPEMBED_SERVER_MAX_INFLIGHT=N` → 429 admission on POST instead of
  invisible queueing behind the model lock. `CRISPEMBED_SERVER_RETRY_AFTER`
  is optional.
- `CRISPEMBED_SERVER_OPENAI_CHAT=1` → `POST /v1/chat/completions` and
  `GET /v1/models` for the `--ocr` model. Each request takes one base64
  data-URL image, and anything the OCR model can't honour is rejected.
- Host test `test-server-load-control`, added to the Unix model-free CI step.

Local evidence. This used a synthetic random-weight 2-layer BERT GGUF, which
tests plumbing only and says nothing about quality:

- `prompt_tokens` 9 equals the `/tokens` id counts 4 + 5.
- `CRISPEMBED_SERVER_USAGE_TOKENS=0` gives 0, and the vectors are identical
  with the gate on and off.
- `MAX_INFLIGHT=2` with 24 concurrent clients: 2× 200, 22× 429, and the
  in-flight count scraped from `/metrics` peaks at 2.
- `RETRY_AFTER=2` sends the header.
- With the gates off: `/metrics` and `/v1/chat/completions` return 404, and
  12/12 requests return 200.
- The chat route was checked only for its 503 without `--ocr` and by host
  parser tests. No OCR model was available here.
- httplib caps `application/x-www-form-urlencoded` bodies at 8 KiB, and that is
  curl `-d`'s default content type. This predates the branch and is documented
  in the README.

Open, with no quality or throughput claim made yet:

- [ ] Live A/B with a real embedding GGUF: identical vectors with the gates on
      and off, and `prompt_tokens` equal to the `/tokens` id count.
- [ ] Measured latency/throughput under concurrent load, with and without
      `MAX_INFLIGHT` (p50/p95, 429 rate). Use hosted CI, not a shared host.
- [ ] `/v1/chat/completions` decoded-text parity with `/ocr/model` on the same
      image, prompt and cap, for one prompt-following VLM. OvisOCR2 is served
      through the orchestrator selector and is not reachable from this route.
- [ ] Prompt-token count for OCR. This needs an ABI accessor, and until it
      exists the route reports 0 instead of an estimate.
- [ ] Rust/Dart wrappers for `crispembed_last_prompt_tokens()`.

## Wiring and environment-variable audit — 2 November 2026

Reference: [docs/wiring-audit-20261102.md](docs/wiring-audit-20261102.md) and
[docs/environment-variables.md](docs/environment-variables.md). Mechanical fixes
(named stage ids 13/23 in every binding, `unlimited-ocr[2]` CLI/server aliases)
are merged. Open, scoped:

- [ ] CrispCalc: no engine-id selector; add catalog entries only after its app pin moves past
      0.17.12 (exact TODO in the wiring audit). CrispSorter PR #19 is merged (HISTORY).
- [ ] Optional, cross-repo: make `core_env::on`/`explicitly_off` treat `false`/`off`/`no`
      as off. This needs a matching edit to CrispASR's `src/core/env_gate.h` in the same
      change, because the Build sibling check requires the two copies to be code-identical.

## GOT-OCR2 Q4_K drift — 3 November 2026

F16-r2 (`got-ocr2-f16`, Hub `c0f1aa7b`) is token-exact against the original stepfun-ai pipeline
on fox/receipt_example/simple_form (run 41588457181; HISTORY 2–3 November). Open:
- The published Q4_K (`got-ocr2`) is not exact: receipt_example first divergence 83 (ratio
  0.8041), simple_form 20 (0.9679). Next: a mixed Q4_K recipe search against the exact F16
  pipeline (`tools/ci-heavy/got_ocr2_f16_republish.sh` compares IDs against the original).
  Done when a recipe is token-exact on all three cases or the drift is accepted by the owner.

## CUDA audit follow-ups — 2 November 2026

Source: `docs/gpu-support-matrix.md` §CUDA audit (branch `feat/cuda-audit`, Kaggle T4).

- EasyOCR x86 CPU dropped lines: **fixed in PR #100** (second BiLSTM layer's initial
  state never reset; see HISTORY 2 November). Still open: native page crops vs Python
  EasyOCR's own recognizer on the same crops agree on 1/2 (fox), 6/11 (scan_strip),
  21/36 (scan_page_pd) lines; every difference is punctuation/spacing on a low-confidence
  line, and the same on CUDA/Metal (pre-existing). Confidences differ too (fox line 2
  0.60 native vs 0.95 Python), which points at crop/resize preprocessing in
  `Reader.recognize` (e.g. `int()` vs `ceil()` resized width) rather than the recognizer,
  which matches Python per stage. Next: dump Python's per-crop recognizer input and diff
  `input_image` with test-easyocr-diff. Also: scan_page_pd's page output differs between
  the M1 (b3318ffa) and x86/CUDA (ad8d340b) on both the fixed and legacy paths, so it is
  not caused by this fix; attribute it (DBNet boxes vs recognizer).
- Below-bar artifacts: f2llm/CLIP/BGE-M3 heads repaired and republished (fix/below-bar-artifacts,
  [summary](tests/results/below-bar/SUMMARY.md)). GOT-OCR2 F16: root cause found (see below). Still open:
  - A CUDA re-run of the new files.
  - A ColBERT-qualifying Q4_K recipe for BGE-M3 (all tried mixes 0.37-0.60).
  - Metal A/B of `bge-m3-heads` (enc0-15 F16 mix, published `283645d8`; CPU 0.9911, CUDA 0.9922).
    0-13 and 0-17 each flip one ColBERT token ([summary](tests/results/below-bar/SUMMARY.md)).
- Pre-existing defects that are independent of the tokenizer switch
  ([A/B notes](tests/results/spm-collapse-ab/SUMMARY.md)):
  - Reranker parity after the PR122 fixes (truncation and XLM-R pair layout are done; HISTORY
    3 November). Open:
    - Attribute the remaining ~0.2 (Q8_0 files vs fp32 HF; expected quantization, unproven). The
      harness's per-layer stage diff came back empty for every model (`stage_diff_worst` None), so it needs a
      fix and a rerun, ideally also on the F16/F32 GGUFs.
    - The harness's ranking-order check (`order_equal`, exact argsort over 77 scores) is False for every
      model in both arms, including ms-marco. It is too strict to read; replace it with Spearman/top-k.
    - ettin-reranker-17m was not checked: its tokenizer needs transformers 5 (the harness error made run
      41588538331 red).
  - The xlmr-ner-hrl GGUF has no Unigram scores (12/76 texts tokenize like HF).
  - paraphrase-multilingual: native truncates at 512, sentence-transformers at 128.
- Stage-diff references that fail on both arms: PP-OCRv6 small rec fox ref, PAN, HAT.
- Not reached on CUDA: PP-LCNet, CRAFT, TexTeller/Pix2TeX, BERT-NER, LFM2.5-VL, LFM2
  encoder, Unlimited/DeepSeek-OCR2, Text-SR, SAFMN, face recognition, BiDirLM audio.
- GPU ports still open (CPU-only graphs): UltraBERT, MoEdern, Josev2B, PP-FormulaNet,
  MixTeX, HMER/BTTR/PosFormer, FLOVA; `wload` SR family (SwinIR/HAT/DAT/SCUNet/TBSRN/
  Text-SR/AdaIR) — the PAN/Restormer pattern (weights + sched on the GPU) gave 8-18x.
- Metal A/B of the new `CRISPEMBED_PAN_GPU` / `CRISPEMBED_RESTORMER_GPU` (CUDA-only
  defaults until measured there).

## Unlimited-OCR-2 Q4_K recipes — 3 November 2026 (perf/q4k-unlimited-ocr2-20261103)

Source: `docs/unlimited-ocr2-q4k-recipes.md`. Both named policies clear the Q4_K bar on
hosted CPU (portable) and Kaggle T4; `Q4_K_MIX` (`unlimited_ocr2_q4k_shared8_v1`) is
prepared in the publication protocol, not published.

- Coordinator: dispatch `Q4_K_MIX` publication (cosine gate); the fresh F16 conversion must
  equal the measured published F16 SHA `487b887e...` or the receipt is re-scored.
- TODO: Q8_0 is 0.9555 on CUDA (< 0.97; CPU 0.9739). Find a CUDA-qualifying Q8_0 recipe
  (F16 shared experts / attention) and re-score the published F32-sourced Q8_0.
- Metal arm not measured.

## Josev Q4_K sweep — 3 November 2026 (perf/q4k-josev-20261103)

Source: `docs/josev2b-q4k-sweep.md`. Portable hosted CPU, frozen original reference.
`josev_q4k_down8_v1` (2,952 MiB vs Q8_0 3,462) clears the Q4_K bar on head logits
(0.99739) and final norm (0.94911); Q8_0 clears 0.97 (0.99995/0.99944).

- Coordinator: Josev Q8_0/Q4_K_MIX publication needs a quantized branch in
  `tools/ci-heavy/josev_publish.py` first (`docs/josev2b-q4k-publication-v1.json`
  is prepared, `dispatch_ready` false). Not published.
- No Q4_K recipe is typed-JSON exact (0/4; argmax 4/4). Strict 0.002/0.99999 stage
  budgets stay failed diagnostics.
- CUDA: Josev's packed/composed runtime is CPU-only; a CUDA arm needs a backend path.

## TeleOCR / MinerU Q4_K-class recipes — 2 November 2026 (feat/qwen2vl-q4k-recipes)

Source: `docs/qwen2vl-q4k-recipes.md`. Winners clear the Q4_K bar on CPU, Metal and
CUDA; `Q4_K_MIX` publication recipes are prepared, not published.

- Coordinator: dispatch `Q4_K_MIX` publication for TeleOCR and MinerU (cosine gate),
  then add registry aliases with the read-back SHA-256.
- Neither winner is token-exact free-running (TeleOCR table@35, three CJK characters;
  MinerU German masthead/"Infrafttreten" and a layout box coordinate). A token-exact
  Q4-class recipe would need more Q8_0 (MinerU `r4` keeps the same two divergences).
- TeleOCR decoder Q4_K remains out of reach (best whole-decoder 0.870 with imatrix);
  a larger, in-distribution calibration set for the opt-in imatrix path is untested.
- [x] Hosted CPU builds used `-march=native`: MinerU's CPU cosine moved by up to 0.015 with
  the runner ISA on identical bytes. Fixed on `fix/portable-qual-build`: heavy-cpu.yml
  `build_mode` (default `portable`) exports `CRISPEMBED_BUILD_MODE`, CMakeLists.txt maps
  `portable` to `GGML_NATIVE=OFF`/`CRISPEMBED_NATIVE=OFF`, and every run writes
  `build-mode.txt`. Receipts state the mode (docs/precision-acceptance.md). MinerU r2/r4 re-measured portable
  three times: 0.952109 / 0.970185, byte-identical cosine reports
  ([receipt](tests/results/portable-build/SUMMARY.md)). Open: TeleOCR winner in portable mode
  before its `Q4_K_MIX` publication; an Intel-runner portable run (all three landed on EPYC 9V45).

## Metal GPU audit follow-ups — 1 November 2026

Source: `docs/gpu-support-matrix.md` (branch `feat/metal-audit`).

- CUDA A/B (Kaggle) of `CRISPEMBED_ESRGAN_GPU=1` and `CRISPEMBED_DBNET_DECONV_MATMUL=1`
  (DBNet det is already GPU-default on CUDA; the deconv rewrite may speed it further).
- Not reached on Metal (no A/B yet): VLM lanes beyond the queued GOT/SmolDocling/H2OVL/
  LightOnOCR/Granite runs, GLM vision drift root cause, Surya, SAM2, GLiNER, LiLT,
  FireRedPunc/PCS, CLIP/ViT/CNN, BiDirLM, NAFNet, SMT/TrOMR/Transcoda, PP-FormulaNet-L.
- CPU-forced engines with ggml graphs that still need a gated GPU path: UltraBERT,
  EmbeddingGemma2 decision (needs a GPU form of its CPU custom GELU), MoEdern, Josev2B,
  Qwen3.5-OCR/Ovis, MiniCPM, PP-FormulaNet, MixTeX, HMER/BTTR/PosFormer, FLOVA,
  Restormer, the `wload` SR family (SwinIR/HAT/DAT/PAN/SCUNet/TBSRN/Text-SR/AdaIR).

# Older lanes (29 October wrap-up and earlier)

## TeleOCR / MinerU on Metal — follow-ups (1 November 2026)

- A/B `CRISPEMBED_QWEN2VL_METAL_SPLIT_K` (TeleOCR vision FFN-down 3420) and
  `CRISPEMBED_TELEOCR_REUSE_DECODE_GRAPH` on Metal before any default change.
- No Q4_K-decoder recipe meets the 0.94 final-output bar for either model on
  any device; search recipes (per-layer sensitivity, Q6_K/Q8_0 for the
  weakest matrices) rather than runtime changes. MinerU's passing
  "q4_k-dq8"/"q4_k-df16" were renamed `q8_0-all` / `q8_0-vision-f16-decoder` (no Q4_K tensor).
- CUDA F16/Q8_0 show prompt-row cosine 0.86-0.90 (CUDA half-precision
  activations; Metal fixed the analogous trap): check cuBLAS/MMQ precision.
- Merge the ggml branch `crispembed/metal-mul-mat-prec-hp` into the fork's
  main line before this branch merges.

## Issue 71 latent / MI-GAN512 qualification — 1 November 2026

Branch `feat/issue71-latent-qual` (five-engines c4bb1f93 + graphs delivery 4
06d75fb8). Not merged into feat/teleocr or main; nothing published or
registered. Status and numbers: [precision guide](docs/inpainting-precision.md),
HISTORY 1 November.

- MI-GAN512: F32 eager and graph pass the frozen 8-fixture panel; F16/Q8_0/Q4_K
  pass the owner cosine gate on all four execution arms. Live real-photo
  evaluation recorded. Open: representative subtitle/video quality, a reviewed
  author digest (only first-observed), publication decision.
- Moebius / PixelHacker: original CUDA captures exist (Kaggle T4; PixelHacker
  with the recorded torch 2.6.0 deviation). Eager F32 Moebius reproduces the
  original image (RGB max diff 1) but exceeds the strict 0.002 component budget
  on VAE moments and late UNet steps; keep that failure recorded and unchanged.
  Pending in hosted lanes: PixelHacker native, all graph arms after the pin
  fix, precision arms and live photos.
- Graph core (delivery 4): initializers and parameters are now pinned against
  gallocr reuse; graph controls pass on CPU (41302790327). CUDA/Metal graph
  qualification needs provisioned runners.
- Speed: the eager latent path takes 6–9 min per 512 image on 4 vCPU; graph is
  about 25% faster. Profile before any default change.
- HF private storage for the account is full; further private preservation
  needs the owner to free space or upgrade.

## Native Marker-style document pipeline — 31 October 2026

The requested implementation and bounded validation are complete on
`feat/marker-native`, preserving the completed TeleOCR
branch and merges `origin/main@b8cd249` before code changes. Marker source is
pinned to `71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4` (2.1.0, Apache-2.0).
The requested GGUF/GGML route uses the existing native TeleOCR backend and a
C++ document pipeline. Current Surya checkpoint inference/parity is outside
this alternative backend: its separately restricted models are not downloaded.

- Native C ABI/orchestration, optional PDFium, layout/crop OCR, document
  processors/renderers, CLI, Python/Rust/Dart and HTTP are implemented. See
  [usage](docs/marker.md), [coverage](docs/marker-feature-coverage.md),
  [licenses](docs/marker-license.md) and [validation](docs/marker-validation.md).
- Final document/backend/C API/save panels pass 86/39/38/38 checks; public
  integration 112, compiled Rust/Dart 13/14, final ASAN/UBSAN 86 and actual
  Python packaging 26. The independent native/reference panel passes 86
  checks across eight PDFs/ten pages. Initial failures and corrections remain.
- Real model runs use a declared, tiny owned panel. Final-source F16/Q8_0/Q4_K
  recover the three-line image exactly; the initial F16 full panel also covers
  neural PDF layout with text reuse and a 4×3 table. Keep raw label errors and
  table-header semantics limitations visible. The separately declared final
  force-OCR PDF also matches authored text with embedded extraction bypassed.
  Seven mode/build/precision trials reuse three distinct inputs; complete
  per-build receipts remain in [the aggregate](tests/results/marker/summary.json).
- Keep `CRISPEMBED_MARKER` and `CRISPEMBED_TELEOCR` default-off. No model
  publication, registry alias, runtime release or numerical qualification change.
- Remaining qualification is representative multilingual/layout/reading-order
  quality, more than two columns, complex tables, large-document resource
  boundaries and non-Linux backends/consumers. This implementation does not
  claim the original Surya model behavior, Office/EPUB providers or hosted LLM
  processors. Upstream models retain their separate restrictions.
- Resource check before bounded work: idle host, CPU-only inference sequentially,
  no simultaneous model forwards. Reuse the frozen TeleOCR numerical evidence;
  this pipeline does not waive its failed strict intermediate gates.

## TeleOCR GGUF/GGML integration — 31 October 2026

The requested native implementation and bounded CPU validation are complete
on `feat/teleocr-ggml`, based on `15adb672f71f8470a23db1284dafa102515adae9`;
ggml remains pinned to `a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. The complete
source delivery includes ggml, an applicable patch and retained evidence.
See [usage](docs/teleocr.md), [license audit](docs/teleocr-license.md),
[validation](docs/teleocr-validation.md) and the [matrix](tests/results/teleocr/summary.json).

- Sources: `XingChen-AGI/TeleOCR@b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d`
  and Apache-2.0 `rzafiamy/teleocr-rs@957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe`.
  The pinned HF model module has an explicit Apache source header. Official
  unlicensed document-client/postprocessor files were not directly translated.
- Converter checks 701 source tensors and writes 700; F32/F16 and protected
  Q8_0/Q4_K artifacts were actually converted and tested. C/Python/CLI/HTTP,
  stage 24, compiled Rust/Dart and the native-backed document parser are wired.
- All 24 CPU cases reach EOS with complete 75-stage coverage and valid
  confidences. F32/F16/Q8_0 each match six complete original sequences; Q4_K
  matches five and adds a trailing newline on code. Raw and normalized scores
  remain separate. Final loader guards pass 43 metadata/header cases and the
  rebuilt production CLI repeats the full 66-token text result exactly.
- **Strict numerical qualification remains false for every precision** under
  the unchanged 0.002 absolute / 0.99999 row-cosine gates. Preserve the initial
  failures, original references and current decoded results. The limited
  F32 ULP/propagation diagnosis does not waive the gates or explain all errors.
- Keep `CRISPEMBED_TELEOCR=1` default-off and CPU placement default. Next work is
  numerical qualification against the same frozen source, representative
  independent OCR/layout gold, then actual CUDA/Metal/Vulkan/mobile/WASM and
  resource-boundary qualification. Quantized raw differences must remain visible.
- Model publication, immutable registry aliases, release tags and downstream
  app changes remain separate. No runtime version or default is promoted here.

Unrelated portfolio work and downstream application pins remain with their
existing owners. Completed details are recorded in HISTORY and PERFORMANCE.

The 29 October [handoff](docs/handoff-2026-10-29.md) is superseded by
[the 3 November handoff](docs/handoff-2026-11-03.md); the older one keeps its dated
sources, job states, inputs, commands, limits and acceptance gates. Completed
results and superseded task claims are in [HISTORY](HISTORY.md); measurements
are in [PERFORMANCE](PERFORMANCE.md). The full previous plan is preserved in
[the dated archive](HISTORY.md#archived-roadmap-snapshot-29-october2026).

This is an open roadmap. No new portfolio release, default flip or quantized
qualification is claimed. Development VERSION is0.17.14; the application stays
on immutable0.17.12. Do not bump versions for this documentation handoff.

## pub.dev publication handoff (31 October 2026)

The published Flutter package is 0.16.1; the development package is 0.17.14.
The existing native release workflow uses `v<version>` tags, while the current
pub.dev trusted publisher uses `crispembed-v{{version}}` (administrator
confirmed). A normal native release tag therefore does not run that publisher.
Do not change the pub.dev setting until a replacement is qualified and merged.

[Draft PR #91](https://github.com/CrispStrobe/CrispEmbed/pull/91) proposes a
release preflight, but leaves the new publisher disabled and retains the old
publisher. It is not a completed publishing solution: the preflight checks
asset identity, not source qualification or consumer functionality, and GitHub
release assets may become public before it runs. The package changelog also lacks
the current development version. The draft Android version change needs an
actual Gradle consumer test, and existing cached libraries are not verified
against the requested version. Preserve the current publisher until replacement.

The draft's model-free preflight has seven hosted contract tests passing. A
[hosted Apple baseline](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41213168996)
using the already-published 0.16.1 package built a clean Flutter app for macOS
and iOS; the macOS build warned about arm64 libraries in its x86_64 link, and
the app did not call native symbols. This does not qualify the development
package, Apple FFI runtime, Android, Linux, Windows, OIDC, or publication.
Before a release, verify source-matched assets and fresh consumers, coordinate
the trusted tag pattern, update the changelog, and confirm the actual pub.dev
publication by registry readback. No version bump or publication is made by
this handoff.

## Immediate executable order

### Model weight licenses (31 October 2026)

[The registry audit](tests/results/licenses/registry-qwen-audit.json) found
five families labelled apache-2.0 whose weights inherit stricter Qwen terms:
`qwen2vl-3b`, `nanonets-ocr-s`, `uni-mumer-qwen2.5-vl-3b` and `teleocr-*`
(Qwen2.5-VL-3B: Qwen RESEARCH LICENSE, non-commercial) and `got-ocr2`
(Qwen1.5-0.5B decoder: Tongyi Qianwen RESEARCH LICENSE, non-commercial and no
use to improve other LLMs). Registry licenses, download acceptance, engine gates
(`CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE`, `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE`)
and the published cstr/* cards are corrected. Open: CrispSorter/CrispCalc must
not offer these models without the acknowledgement; Apache-2.0 replacements for
GOT's tasks are MinerU2.5-Pro, FireRed-OCR, Uni-MuMER-Qwen3-VL-2B and PaddleOCR-VL.


### OXR / D-FINE on Metal and precision recipes (feat/metal-oxr, 1 November 2026)

Done (receipts `tests/results/oxr/metal-m1/`, `tests/results/oxr-dfine/metal-m1/`, hosted
`tests/results/oxr*/hosted-413*`; details in docs/oxr.md "GPU: Metal" and docs/oxr-dfine.md):
OXR F16 on Metal token-exact 5/5 and now the Apple default for that recipe; D-FINE F32 on
Metal region-exact 5/5 (opt-in); ggml patch `GGML_METAL_PREC_F32_MM`; D-FINE F16 re-scored
(passes the owner bar); OXR Q8_0 with quantized large tables (1.93 GB, F16 vision) passes;
Q4_K-head mixed recipes pass; plain Q4_K LLM layers fail (table 0.896).

Open, in order:

1. Heron: done; GPU default now PREC_F32 + `_hp` (regions equal CPU 5/5). D-FINE is Metal by
   default on Apple (region-exact 5/5, 2.1-2.2x CPU).
2. OXR quantized: CPU `CRISPEMBED_OXR_DEQUANT_VISION=1` makes Q8_0-everything 5/5 token-exact
   (41340880860); Metal five-case A/B done (Q8_0-large and Q4_K-head: 4/5, logits cos >= 0.9995 vs
   Metal F16). Remaining German divergence comes from decode-step activation quantization
   (integer mat-vec on every backend); an opt-in F32-activation decode path is the next A/B.
   Publication decision for the qualifying recipes is the owner's (none made here).
3. OXR F32 on Metal (needs a local F32 conversion; volume was below its free-space floor).
4. CUDA: done on T4 (kernel v2): F16 5/5 token-exact with the new dense attention; Q8_0-large
   and Q4_K-head 4/5 (German 55/209), logits cos >= 0.998 vs CUDA F16. CUDA quantized products
   still quantize activations like the CPU; a CUDA analogue of the vision dequant gate is open.
5. Dense GPU decode copies the whole V cache per layer per token (permute+cont); a
   transposed-V cache would cut it. Flash is 1.85x faster on German but less close to CPU.

### OXR-1.0 native integration (31 October 2026)

The isolated `feat/oxr-1.0` worktree retains the earlier MinerU integration and
merges upstream `b8cd2497765f2fd8db0b9c713d952898bdd9b996`. The OXR model pin is
`c708407f9d2b4df658f2543d9132b62bb7221df4`; the document helper pin is
`c1a1dcb312ee9f691404f127785510453c0d9350`. A dedicated native graph, strict
converter, document adapter, public bindings, and original-reference qualification
tools are implemented behind `CRISPEMBED_OXR=1`. This is a separate source lane
under OXR's additional license terms, not an addition to the permissive-model
publication queue. [The guide](docs/oxr.md) records exact scope and commands.

Completed component evidence is in
[the implementation receipt](tests/results/oxr/implementation-validation.json).
Full library linking, the linked GGML contracts, checkpoint conversion/inference,
all 66 original/native stages, representative document metrics, same-host F32
speed, reusable-graph A/B, GPU/backends and Rust/Dart runtime checks remain open.
The supplied [hosted workflow](.github/workflows/oxr.yml) has not been dispatched.
Do not publish aliases, promote defaults or infer vLLM throughput from component
checks. The original D-FINE layout service remains separate; supplied regions
allow the adapter to avoid that service. The native D-FINE layout detector now
lives on `feat/oxr-dfine` (see below).

### OXR-1.0 D-FINE layout detector (feat/oxr-dfine, 1 November 2026)

Native `dfine` architecture of `layout_detect` ([docs/oxr-dfine.md](docs/oxr-dfine.md)),
gated by `CRISPEMBED_OXR=1` + `CRISPEMBED_ACCEPT_OXR_LICENSE=1`. Hosted run
41250159815: F32 OXR-selected regions match the original service on all five
pages (boxes <= 4.4e-4 px, scores <= 1.8e-4), preprocessing bit-exact, heron
byte-identical; stage budget 5e-3 fails in decoder layers 2-3 on receipt/german
(native closer to float64 than PyTorch FP32 on 55-56/56 stages); F16 rejected.
Open, in order:

1. Speed: native 1.44 s vs PyTorch 0.52 s per page (run 41251744740, 4 threads;
   2.6-2.7x). Opt-in dw-direct is 18% faster, same regions, one extra stage-budget
   failure; conv-direct is slower. Next: per-op profile of the encoder graph
   (im2col 3x3 convs, concat chains) before any default change; keep paths gated.
2. End-to-end `CrispOXR(layout_model=)` document output vs the service-backed
   pipeline on the same pages (JSON/Markdown), once the document-assembly lane lands.
3. Optional: `class_confidences`, GPU backend qualification, Rust/Dart runtime tests.
No publication, alias, merge into feat/teleocr, tag or default flip follows.

**Native OXR document route (`feat/oxr-native-doc`, `CRISPEMBED_OXR_NATIVE_DOC=1`, default off).**
The pinned helper's `process_pages`, minus the layout detector, is translated to C++
([guide](docs/oxr.md#native-c-document-route-experimental-crispembed_oxr_native_doc1)).
Local model-free parity on arm64 matches 2,801 helper checks and 60/60 scripted documents.
The only differences are arm64-wheel `INTER_AREA` pixels
([receipt](tests/results/oxr/native-doc-local-macos-arm64.json)). Open items:
hosted x86-64 helpers/replay/native parity on the five original-D-FINE cases
(`tools/ci-heavy/oxr_native_doc.sh`); Markdown-table output, table-image placeholders
and `asset_dir` (rejected, not ported); and plugging the native D-FINE provider
into `from_layout_detect` after `layout_detect::region` carries raw labels. Keep the route
default-off until hosted parity holds on every case.

The shared-operator audit additionally replaces MinerU-only QuickGELU's F16 LUT
with explicit F32 math. Its linked operator check and full-model replay remain
pending; the earlier MinerU component receipt retains its original source hashes.

### MinerU2.5-Pro-2604 integration (31 October 2026)

The isolated `feat/mineru25-pro-2604` source delivery adds the checkpoint
`opendatalab/MinerU2.5-Pro-2604-1.2B` at revision
`d3f5e08d073c21466bbabe21c71bb1e9c2e595da`. The implementation reuses the
Qwen2-VL GGML graph behind `CRISPEMBED_MINERU25_PRO=1`. Its converter,
preprocessing/token contracts, document adapter and qualification tools are
tracked in [the integration guide](docs/mineru25-pro.md). Model-backed
intermediate, decoded-quality and same-host speed gates must pass before this
path is qualified or receives a published model alias. No existing model default,
release version, application pin or remote model artifact is changed here.

Full native document route (`feat/mineru-native-doc`): C++ port of every
mineru-vl-utils branch behind `CRISPEMBED_MINERU25_PRO_FULL_DOC` (default off),
composed from the layout/router/Marker-document components with an exact
upstream output profile. Weight-free parity is identical (see the guide).
Model-backed parity on hosted run 41251475795: 8/8 protocol cases identical.
Next: review and decide default-on (keep basic/Python routes); add an
image-bearing-table fixture to exercise masking with the real model; token-glyph pixels in image-bearing tables and
cross-page table merge remain documented exceptions.


Choose one bounded lane. The source table and current remote states in the
handoff take precedence over historical commands.

1. (Closed 3 November: Unlimited 40868634374 superseded, Ultra 40871150002 reviewed; receipts
   and HISTORY 3 November.)
3. Independently review MoE real-router source81fa4e26, then execute its
   component-only replay using exact-v4 attached captures. No whole-model or
   original forward. Require own-byte controls before cross-input conclusions.
4. Implement Gemma's frozen live-snapshot/768-wide-adapter design0ee3fb674.
   It is **not executable**. Review the implementation, then activate only one
   18-ID probe/three native calls. Review actual live/late/pooled evidence before
   separately declaring any full145-stage/typed/Q4 retry.
5. Independently review Josev imatrix source208fbefe, register its workflow, then
   execute its fixed40-prefill development panel on a host satisfying14GiB
   RAM/64GiB disk. Do not silently lower its resource/source guards for Kaggle.
6. Complete the remaining precision/task/client gates below, then integrate
   reviewed opt-ins and qualify final runtime/assets/consumers. Each new model
   or arithmetic source needs its own reference/quality/consumer evidence.

## Unlimited-OCR-2 vision speed — 3 November 2026

The F32-vision default and its post-flip real-scan confirmation are done (HISTORY 3 November). Open:
- Speed: remaining time is prefill (38 s) and decode (~66 s) on the table page; the global
  1024 SAM view is the next vision target. `TILE_BATCH` stays off (~2%).

## Required Q8_0 and Q4_K coverage for new permissive models (29 October)

Use actual `crispembed-quantize`, original Python/ONNX dumps and actual
`crispembed-diff`. Keep critical tensor types/payload hashes, axes, compiler/
backend provenance, absolute/RMS errors, magnitudes, cosine and final task
outputs. Numerical and task verdicts are distinct; preserve negative flags.
[Measured critical-tensor evidence](docs/critical-tensor-precision-20261029.md)
explains why a generic exemption list is insufficient.

| Front | Current boundary | Remaining work and completion gate |
|---|---|---|
| Josev-2B |F32 composition227 stages/four typed requests is established; protected/single-projection quantization negatives are retained. Final-pair Q8 passes a limited decision screen but75 strict stages/full serialization remain negative. |Execute reviewed calibrated/uncalibrated final-pair Q4 experiment; keep292 critical F32/148 F16/2 Q4 and exact uncalibrated model continuity. Do not select/tune on NEW20. A separately declared independent task panel, actual six-client/public/error/recovery checks and immutable HF readback remain before publication. Q8 and Q4 qualification are both open. |
| MoEdern1.5M |Actual F32 capture localizes one near-boundary expert change; router weights already F32. Component-only replay source is ready, execution unreviewed. |Separate same-input operator rounding from upstream input effects; preserve exact-tie guards and all original matrices. Only then declare a gated correction, original-bound full5483-vector/300-query replay and Q8/Q4 retrieval/client panels. Keep explicit query/document roles and late-interaction semantics. |
| MoEdern27M/115M |Original token checks complete but exact-router-tie failures prevent full reference completion. Authored Q8/top-k tests are component evidence only. |Use source-backed routing semantics and bounded failed-row attribution before separately frozen original/full native controls. Preserve115M dilated-conv behavior. Validate actual20/20/52 padded Q4 and protected Q8 tensors, full retrieval and all public consumers; no component-to-model promotion. |
| UltraBERT |Frozen5483-row F32 control/300 rankings pass. Full 8-shard Q4 aggregate 40871150002 (`q4k-last-four-ffn`, portable) is complete: min embedding cosine 0.9779 clears the owner Q4_K bar (> 0.94); nDCG@10 -0.0018; strict v1 vector/top-1 (0.973) gates fail and stay recorded ([receipt](tests/results/ultrabert/remaining-q4-shards-v1-review.json)). |Q4_K: coordinator publication dispatch (with the Q4_K_MIX queue). Keep .001 task-degradation/.99 top-one gates and strict vector flags separately. Q8 needs explicit metadata/type admission; standard-header normalization, public clients, independent activation and publication/readback remain open. Public pairs/F16/long-context/GPU/performance are distinct follow-ups. |
| Unlimited-OCR-2 |The last1 per-stage panel 40868634374 stopped after 1/4 Q4 cases on its disk-reserve preflight ([receipt](tests/results/unlimited-ocr2/last1-replay-v1-initial-failure.json)); superseded by the 3 November final-logit Q4_K recipe lane. |See "Unlimited-OCR-2 Q4_K recipes" and "Unlimited-OCR-2 Q8_0 on CUDA" above; no further per-stage panel. |
| MiniCPM-OCR |All30 decoded cases exact; vision130/180 pass,50 fail;96 crop parts unmapped. Q4 protects vision while generic Q8 quantizes it. |Declare decoder-only Q8 on the SAME186 decoder matrices; keep593 source-model exclusions byte-exact. Independently attribute earliest F32 vision differences with same inputs/weights and document unmapped crop limits. Preserve original23 plus added7 outputs; qualify all five cases/intermediates/clients before publication. No original corpus recapture or generic quantizer default change. |
| LFM2.5 masked230M |Published Q8/Q4/calibrated/mixed profiles exist, but mixed330 still changes one masked prediction. |Reuse immutable original/same-weight references; isolate remaining prediction-sensitive groups. Preserve attention/ShortConv/FFN-down high precision supported by actual ablations. Freeze independent task inputs before selection; extend quantized/GPU/client evidence separately. Do not confuse this with the350M retrieval model. |
| Typed EmbeddingGemma |Q8 approximation published; first new Q4 worker stops at F32 saved-stage/adapter failures. All five pooled F32 embeddings pass. |Implement/review design-only repair, three-call first probe, then separately activate full145/eight-typed/30-bank F32 barrier before unchanged broad/protected-input Q4 recipes. Q8 current-source replay and Rust/Dart model-backed paths remain open; old wrapper helpers hardcode F32. Independent task/GPU/performance/publication gates remain separate. |
| OvisOCR2/OvisOCR3 |Ovis2 F16 is published/consumer-verified. All existing Q8/plain/calibrated/mixed Q4 candidates are rejected. Ovis3 faithfully matches an original capped/poor-quality page. |Keep Ovis2 F16/alias; new precision selection needs fresh development protocol after completed same-weight attribution. Ovis3 needs a separate original-first task/evaluation protocol before quantization, alias or publication. Never raise caps/change prompt/gold to pass old quality gates. |

For each candidate: authenticate all declared inputs before model use, preserve
ordinary/observer/repeat and rejection/recovery controls, compare earliest actual
intermediates, and evaluate complete decoded/typed outputs or retrieval rankings.
Only a separately qualified candidate proceeds to all public surfaces, immutable
HF upload/empty-cache readback, registry pins and consumers. Stored files alone
are not qualification. Calibration is development-only; never invent reference
activations or waive historical budgets.

## Active Ovis completion sequence

Draft [PR89](https://github.com/CrispStrobe/CrispEmbed/pull/89) holds integration
`8da6636b35d33623e143ae34120c09fb9878a460`. Its frozen test-source formatting
failure must be repaired **only in a new integration closure**, preserving every
producer's source/receipts. Merge carefully: old branches can delete unrelated
registered workflows/evidence; review all deletions against current main.

After final code decisions, declare a new complete source closure. Require
original-bound Ovis fourteen-arm plus first-F16 eleven-case replay on BOTH gates,
all frozen IDs/text/EOS/caps/confidences, appropriate public interfaces and the
full required CI. Then create fresh native/WASM assets and verify actual Linux,
Apple, Windows, Android/browser consumers, hashes, baseline CPU portability,
installed loads and aggregation. Only then use the version/release procedure,
immutable model aliases and a separate downstream app-pin upgrade.

The old frozen full-source guards intentionally reject a changed runtime. Do
not weaken them, relabel another source or infer asset qualification from CI.

## Ovis 2048-patch numerical attribution — before broader input claims

Default1024 stays. Frozen/1024/1200 pass14 arms;2048 has six composed final-norm
failures despite exact text/EOS. Reuse original embeddings and retained inputs;
isolate the first divergent position/interpolation/norm boundary with actual
source-fed operators before any broader-page claim. Preserve the four opt-in
large-image gates and existing strict/calibrated budgets.

## Other open numerical, performance and consumer lanes

Each row requires a new bounded protocol, frozen inputs, original/reference
comparison, final-output checks and actual hosted evidence before adoption.
Detailed prior acceptance clauses remain in the archived roadmap/model docs.

| Lane | First bounded action | Gate/dependency |
|---|---|---|
| WASM layer3 |Complete corrected one-sentence live operand capture; use actual13 norms/23 layer3 boundaries and same-input replays. |Preserve full norm/stored-F32/rounding negatives and default-off gates.208 local/196 natural comparisons and native/old-WASM byte anchors stay distinct. No full ten-case retry or arithmetic change before attribution. |
| LightOnOCR-3-0.8B |Use retained book isolation to investigate upstream vision, with one fixed input/source-fed replay. |Exact independent-document text retains numerical debt; all quantized receipt/book recipes remain rejected. No publication/default from literal EOS or table-cell equality. |
| OCR real pages/benchmarks |Freeze licensed annotations/reading order and consistent timing spans for common pages across supported engines. |Report CER/WER, boxes, order, EOS/caps, confidence and paired latency/RSS; upstream OCRBench/OmniDocBench/olmOCRBench scores are not reproduced merely by integration. Tau-squared is an agent benchmark. Preserve Mini's one and Unlimited's19 real-page token mismatches. |
| Tesseract/Fraktur |Freeze exact line/crop geometry and compare official recoder/decoder/dictionary/spacing outputs. |Keep multi-code recoding/default DBNet and beam alternatives. Certainty, line counts, CER/WER and crop batching must be measured independently. |
| PP-OCRv6/orientation/EasyOCR |Reproduce one detector geometry/readback or SE/depthwise outlier; freeze word boxes/order for structured LayoutLM inputs. |Artifact-backed sizes, original/native crops/recognition and actual device evidence; retain CPU fallback and provenance. |
| PreprocessingT2/O10, batchingT3–T6 |Choose one cleanup/dewarp/binarization/resize/depthwise/crop-batch change against annotated pages. |Recognition/geometry quality first; identical input/output timing spans. Do not select preprocessing from synthetic-reference similarity alone. |
| Layout/TableFormerT16 |Freeze document structure/reading-order gold, then profile one GPU value projection/hot spot. |Actual decoded structure/boxes and device execution, not allocation backend alone. |
| Qwen/formula/backendR3/R6/O3/O7 |Profile one measured decoder/encoder/GEMM/thread-propagation or residency hot spot. |Preserve CPU/serial/parallel alternatives; actual F32/cache/decoded and paired memory/timing controls. Graph setup was negligible in old profiling; do not optimize it by analogy. |
| Ovis decoder cache/F2 |Freeze representative same-host cold/warm initialization, repeated recognition and one/two-context latency/RSS. |256MiB cache remains off after only0.797% warm median gain/increased RSS. Full970MiB head remains fallback. Metal cap adoption needs actual device evidence. |
| Pipeline cache identity |Start with legacy engine20 and two distinct model paths. |Force second-model/rejection/recovery proof before an engine-specific key fix; preserve completed Ovis stage-owned path identity. |
| F3–F6/VLM frontends |Separately scope CLI thread default, SmolDocling vision scheduling, DeepSeek-OCR2 crop/F16-KV or Qwen3-VL/jina resize/tiling contracts. |Current source/licensing, original input geometry and final outputs/performance; do not port caches or defaults family-wide. |
| Backend bring-up |Use actual CUDA/Metal/Vulkan hardware for a single proven compute/scheduling gap. |Verify numerical/decoded quality, actual compute placement, residency and memory; host allocation is insufficient. |
| E4/encoder ecosystem |Freeze German/Japanese representative quality inputs; reconcile feature matrix/Nomic MoE opt-in independently. |MTEB/JMTEB task quality, actual representative latency and consumer compatibility; byte equality alone is not a speed claim. |
| Restoration/OMR |Validate Q8 ESRGAN/SCUNet or one eligible notation/vocabulary candidate under its own original blueprint. |Do not publish known bad ESRGAN Q4. Transcoda remains outside permissive queue; oemer remains reference-only. Preserve completed TrOMR/Flova evidence. |
| Coverage/T9/F10/SAM |Audit one native output/control gap; coordinate sibling changes explicitly. |No silent CrispASR changes. SAM stays with its existing owner; leave unrelated work intact. |

## Permissive OCR queue

Audit actual code, checkpoint, base and dependencies first: MIT/Apache/BSD or
similarly permissive terms only. For each model use the contribution checklist:
converter/full tensor census; actual Python/ONNX blueprint and intermediate
reference; gated native math with earliest-stage/final-output comparison; C ABI,
orchestrator, CLI/server, Python/Rust/Dart; actual Q8/Q4_K task qualification;
immutable upload/readback, registry hashes and real consumers.

The [4 November model-gap review and expansion guide](docs/ocr-benchmark-expansion.md)
supersedes the initial candidate order. Current main already registers
PaddleOCR-VL-1.6, GLM-OCR, olmOCR2, FireRed, Nanonets-OCR2-1.5B, DeepSeek-OCR2,
MiniCPM-OCR, Unlimited-OCR2, SmolDocling and earlier LightOn/MinerU/Ovis versions.
Keep exact checkpoint versions separate from generic family coverage.

1. Finish existing MinerU2605, LightOn3-1B/0.8B, Xiaomi and Ovis3 qualification
   under their already declared protocols. PR141 owns the active norm/cache work.
2. **LightOnOCR-3-4B (active source/bootstrap lane):** Apache-2.0 difficult-text,
   handwriting/math/degraded-page candidate. Pin its source and complete tensor
   inventory first. Current native 1024-wide/24-layer admission, equal linear
   key/value heads, vocabulary projection allocation and 12-layer vision precision
   recipe need separate changes; generic Qwen3.5 support is insufficient.
3. **Qianfan-OCR:** Apache-2.0 document-understanding/KIE candidate. Audit exact
   vision/projector mappings, Qwen3 decoder and actual processor defaults before
   native work; configuration/template names alone do not prove tiling behavior.
4. **Logics-Parsing-V3 (conditional):** prioritize compact multi-page hierarchy
   and continuation support once the publisher's weight grant is explicit. Its
   Apache-2.0 code can support a separate model-independent document-state/DSL
   contract. Do not transfer Logics v2's score or third-party GGUF license labels.
5. **Nemotron 3.5 Parse / OpenDoc-0.1B:** footprint-driven feasibility work after
   existing quality closure. Nemotron needs its full C-RADIO/causal-mBART graph;
   OpenDoc includes a layout model plus enhanced UniRec, not a standalone 0.1B
   complete page model. Measure total memory/latency and full pipeline quality.
6. **Infinity-Parser2-Flash:** later dense-family extension for a demonstrated
   specialty. **Infinity-Parser2-Pro / Qwen3.6-27B:** separate server-size ports;
   Pro's 35.1B total MoE weights, 40 layers and 256 experts require storage and
   backend planning. General Qwen3.5-9B OCR is lower priority than the named tasks.

TeleOCR remains in its existing `qwen-research` acknowledgement/qualification
lane, not this permissive queue. Indra remains public-code-analysis-only.
**dots.mocr** is the leading conditional new quality/multilingual candidate;
its Qwen2 decoder has a custom Dots vision tower. Its MIT label includes
supplemental terms requiring a policy decision before adoption. OCRVerse and
ABot-OCR remain conditional on explicit weight rights; OCRVerse is a specialist
visual/code reconstruction option. Dolphin1.5 (MIT) is a possible compact fallback,
whereas Dolphin-v2 has Qwen Research weight terms. Chandra2/Surya2.5 require
separate commercial/competition eligibility decisions; Youtu and Hunyuan have
territorial limitations. Existing `surya-det` is not Surya2.5 OCR. Read exact
checkpoint/code/base terms independently; benchmark rank is not permission.

## Issue71, packaging and downstream delivery

- Bounded source delivery, 1 November 2026: issue71 CLI implementation on
  `feat/issue71-inpainting`, based on `b8cd2497765f2fd8db0b9c713d952898bdd9b996`.
  Adds a default-off, exact-model-pinned MI-GAN CLI, native256 tiling, explicit or
  DBNet-derived masks, PNG provenance and persistent directory/TSV batches.
  The existing MI-GAN graph/reference receipts remain unchanged. The
  [implementation and validation contract](docs/inpainting-cli.md) distinguishes
  model-free wrapper checks from pending native hosted execution and real-scene
  quality. Thirty-six dependency-free Python checks, six C++ syntax-only checks,
  Python/YAML syntax, formatting and the clean-exit guard pass. Compiled native
  tests, actual CLI execution and checkpoint replication remain unexecuted;
  the new three-platform workflow and opt-in model smoke are prepared, not
  observed hosted results. This source delivery creates no model alias, runtime release, shared
  C ABI or downstream pin change; temporal/video and LaMa remain separate.
- `feat/metal-migan` (1 November 2026): gated MI-GAN GPU/Metal path, `migan-pointwise-v1`
  F16/Q8_0/Q4_K containers ([doc](docs/migan-metal.md)). Metal F32 exact on all eight
  fixtures (0/1896 strict failures); `PAD_K` 284 ms vs CPU ~650 ms per tile on the M1.
  Owner bars: F32/F16/Q8_0 pass on CPU and Metal; Q4_K fails (generated 0.9395,
  masked 0.805). TODO: (1) finish the queued Q4_K sensitive-layer (`CRISPEMBED_QUANTIZE_MIGAN_Q8`)
  and `DEQUANT_WEIGHTS` probes and rerun the eight-fixture panel for any candidate;
  (2) run delivery-5's CPU-kernel graph on the M1 CPU; (3) reconcile with
  `feat/issue71-five-engines-graphs` and only then flip GPU via `explicitly_off`
  with `PAD_K`; (4) attribute the ~2x per-tile gap of 1080p frames vs isolated
  calls; (5) CUDA/Vulkan untested.
- Issue71 MI-GAN: fixed256 CPU/F32 prototype and494/eight-fixture audit are
  complete. Qualify the new CLI separately, then freeze licensed real subtitle/
  scene images and masks with perceptual/seam criteria. C ABI/orchestrator/server/
  bindings, registry and downstream consumers remain separate. Video/motion/
  flicker and text-derived masks need their own gold. Other model resolutions,
  F16/quantization and GPU backends remain separate. LaMa is held on checkpoint
  terms/FFT semantics. Keep issue71 open.
- CPU inpainting execution, 1 November 2026: native builds, unit checks and
  live model tests were executed for this authorized run. Persistent-input
  allocation and long GGUF tensor-name loading defects are corrected; all
  2,080 latent tensor payloads remain unchanged. The source package records
  executed checks separately from the remaining qualification gaps. See the
  [completed-results report](docs/inpainting-optimization-results.md).
- MI-GAN256/512 baseline and optimized paths each pass 20 original-model cases
  and 1,070 stage comparisons; corresponding native stage hashes are identical.
  Ordinary image API medians improve 2.61–3.27 times over eager on the admitted
  256/512, four/eight-thread arms. The 512/eight-thread quiet confirmation
  improves 3.24 times. Initial contended samples remain excluded and retained.
- Moebius completes the four-case original replay with local/relative lambda
  reassociation. Its integrated optional CPU depthwise path preserves all eight
  prior generated/composited RGB outputs and all 608 reference metric records;
  the same 35 strict absolute-stage failures remain. A completed ordinary API
  pair also has exact RGB and known pixels, but the warm median changes from
  142.796 to 181.397 seconds under contention. No complete-image speedup is
  qualified; keep depthwise outside the recommended throughput preset.
- The resumed PixelHacker eager/candidate pair completes one cold and three
  warm calls per arm on the same executable. Native RGB agreement passes:
  filled-channel maximum difference 1, mean 0.0000949436 and known errors 0.
  Warm medians are 310.958/144.243 seconds; the observed 2.156 ratio remains a
  diagnostic result from one fixture and fixed run order. Preserve the earlier
  cancelled eager arm and timing-only revalidation unchanged. Original CUDA/FLA
  parity, portable-only graph timing and representative quality remain open.
- Optional SHA-NI model verification hashes every byte against the same digest,
  with observed hardware/block counts and scalar fallback. Its bounded memory
  hashing benchmark measures 4.6835 times the scalar throughput, separate from
  image inference. Production CPU convolution stays at 32 MiB; the private
  128 MiB workspace screen was mixed and is not integrated. The resumed control
  suites pass 244 Python/CLI tests, 20 native configurations and 21 additional
  private API-controller controls; exact reports are linked from the results.
- Actual production CLI checks cover both MI-GAN sizes,
  with empty masks, exact API outputs and batch-context reuse. GPU execution,
  original PixelHacker parity, representative perceptual/video quality,
  quantization, public bindings and downstream release qualification remain
  separate. The current SHA-enabled CLI panel passes for every profile of that
  research build:
  12 invocations/16 PNGs on one binary, with exact ordinary-API outputs, empty
  masks and both latent filled/empty/filled batches. The earlier CLI panels
  remain separate. Keep issue71 open, all candidates default off and the
  model/source pins intact. Follow-up work must preserve prior receipts and
  fixed numerical budgets, including every original intermediate failure.
- Source/wheel success is not release qualification. Require fresh portable
  native/WASM producers, installed bundle/ABI/platform consumers and full
  aggregation for the final source, then update release metadata coherently.
- PosFormer/handwriting: retain7/50 original versus0/50 candidate, old legacy=0
  and full-checkpoint/sequence parity debt. No new training or quality claim
  without licensed inputs and an original blueprint.
- CrispMath stays immutable0.17.12. After compatible qualified assets exist,
  separately update its pin/actual consumers. Physical-device/accessibility/
  localization/narrow-layout and App Review/store work remain consumer tasks;
  cloud/collaboration are optional separate decisions.
- CrispSorter: use the actual Rust/Tauri/Svelte extractor/config/selector paths,
  orchestrator stage21 for qualified Ovis2; stage22/Ovis3 remains unqualified.
  Require selected-library identity, fresh model hashes, outputs/confidence,
  prompt/cap rejection and fallback behavior. Do not reuse legacy VLM number space.

## Executable audit continuation — 27 October 2026

This anchor is retained for older links. Its dated commands/status snapshots
are archived in HISTORY. Use [the 3 November handoff](docs/handoff-2026-11-03.md)
for current sources, jobs and exact next actions; do not repeat completed runs.

### Portfolio Q4_K/Q8 branch (PR89) open items

MoEdern Q4_K lane (3 November, [doc](docs/moedern-q4k-mixed.md)): 115M `moedern115m_q4k_v1` and 27M
`moedern27m_q4k_v1` clear the Q4_K token-vector bar on x86/ARM; 1.5M has no Q4_K recipe. Open:
(1) SciFact retrieval/ranking and public-client panels for both packages before any publication
dispatch (`docs/moedern-q4k-publication-v1.json`, cosine gate prepared); (2) Q8_0 misses 0.97 on
27M/115M through single-token routing flips; a flip-robust Q8 recipe or a ranking-level judgement
is undecided.

Q8 admission source is being prepared on `feat/portfolio-q8-profiles-20261029`:
Josev gets an explicitly named `crispembed-josev-packed-mixed-v2` profile;
header7 requires actual Q8_0 and no Q4_K, while legacy Q4 requires real Q4_K.
UltraBERT uses the same storage/header consistency. The existing quantization
gates remain mandatory. A bounded `--metadata-string KEY=VALUE` C-quantizer
option replaces existing STRING keys before header layout, enabling genuine
profile stamps without ad hoc byte-length patches. Numeric or unknown metadata
keys reject before output creation. Actual compiler/metadata fixtures and the
original-bound Q8 tensor/intermediate/decoded panels still gate qualification.

The default-off MoEdern Q8 runtime and source-bound component caller are
integrated in the portfolio feature branch. All eight MoEdern runtime/profile
stdlib controls pass there. The caller's separate frozen whole-SRC assertion
correctly rejects this combined tree: its immutable1f source differs only in
UltraBERT's later UINT32 file-type admission check. The reviewed actual fixture
runs on exactdd592, not the integrated tree; no assertion or source gate was
weakened to claim combined-source qualification. Final source declarations and
actual composite/runtime/platform gates remain required.


## Atomic LightOn4B model ownership — 5 November 2026

Active bounded lane `feat/lighton4b-model-owner-20261105` stacks exact PR145 `882b8dded` text/grouped/export dependencies with unchanged PR144 `84e19504e` vision source/header. Adds atomic723-tensor borrowed ownership behind a separate default-off gate, globally rejects pointer alias/strides/unknown prefixes and preserves protected F32 norms/biases. Hosted header-derived controls only; no checkpoint payload, format5 admission, frontend, decoder execution, release or decoded parity claim. Grouped execution and streamed projection remain separate dependencies for future composed inference.

Initial hosted owner run [41973459077](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41973459077) builds successfully and runs5281 controls but fails: the new test incorrectly treated literal `false` as off, contrary to the unchanged shared helper contract (only unset, empty and literal0 are off). Correct the test to exercise unset/empty/0; preserve the runtime gate semantics and the negative run. No model payload or inference occurred.

Corrected atomic owner [41973787198](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41973787198) at `8ea8f4d03` passes5281 header-derived controls and36 protocol/closure controls. Independent small-artifact review verifies12 recorded source digests, exact723-row schema hash,426/297 partition,197 protected vision vectors, inherited PR144/145 component identity and the control census. [Scope/evidence](docs/lighton4b-model-owner.md) and [receipt](tests/results/qwen35-ocr/lightonocr3-4b-model-owner.json) retain initial41973459077. Borrowed metadata only; no payload-backed native model, format5 admission or decoded parity. Draft stacked PR146 stays unmerged pending checks.

## Full-width LightOn4B linear decoder block — 5 November 2026

Active isolated lane `feat/lighton4b-linear-block-20261105`, stacked on PR146 `8157a3137`. Adds a separate default-off one-block CPU component combining zero-centered RMS, grouped16:32 GDN, residuals and9216-wide SwiGLU FFN at actual hidden2560. Compact convolution/recurrent caches are immutable explicit inputs, with exact32768/524288 counts. Existing decoder/cache formulas remain untouched; full32-layer execution, full-attention composition and streamed vocabulary projection remain separate. Hosted original Transformers5.5.4 decoder-layer reference uses synthetic full-width weights and original-first prefill/cached/full-prefix arrays; no checkpoint acquisition, logits, format5 admission or decoded-quality claim. Preserve original source pins and report earliest stage errors/magnitudes.

Full-width synthetic linear-block [41976928016](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41976928016) at `5b1c813dc` passes33 original/native stages,52 controls and9 byte-exact original observer-neutrality checks. Independent source/raw-array review recomputes first input-norm max3.5762786865e-7 (magnitude1.429962039), worst FFN activation2.8610229492e-6, final hidden prefill/fullprefix1.9073486328e-6 and cached1.3113021851e-6. [Exact scope](docs/lighton4b-linear-block.md), [review receipt](tests/results/qwen35-ocr/lightonocr3-4b-linear-block.json). CPU/F32 one synthetic linear block only; no checkpoint/logits/decoded output. Existing decoder sources unchanged. Public report sanitization is reporting-only.

## Full-width LightOn4B full-attention decoder block — 5 November 2026

Active isolated lane `feat/lighton4b-attention-block-20261105`, stacked on PR148 `fd970f348`. Separate default-off full-attention block with exact16Q/4KV,head256,partial rotary64 and hidden2560/FFN9216; old linear/attention/decoder sources remain untouched. Hosted original DecoderLayer(index3), eager causal masks, distinct nonzero multimodal axes and native-owned K/V cache exercise prefill2+cached1/fullprefix3, with51 original/native stage comparisons and observer-neutrality/negative controls. Synthetic CPU/F32 only; no checkpoint, vocabulary projection,32-layer decoder,format5 admission or decoded-quality claim.

Full-attention block [41977890812](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41977890812) at `9f64a417a` passes51 original/native stages,72 controls and9 bit-exact observer-neutrality checks. Independent source/raw review recomputes first norm3.5762786865e-7 (magnitude1.429962039), worst cached FFN activation5.7220458984e-6 (magnitude4.790956020), final hidden prefill/fullprefix3.7550926208e-6 and cached3.0994415283e-6. [Exact scope](docs/lighton4b-attention-block.md), [review receipt](tests/results/qwen35-ocr/lightonocr3-4b-attention-block.json). Reporting-only correction labels inherited GDN function checks as auxiliary source verification, not executed attention functions; original whole-module hash pins actual execution. No source closure/budget changes, checkpoint/logits/decoded model proof or default promotion.

## LightOn4B32-layer hidden-state composition — 5 November 2026

Active isolated `feat/lighton4b-decoder-compose-20261105`, stacked on PR149 `a657860f5`. Adds a separate default-off exact32-layer LLLF orchestrator using tested block entries unchanged, tagged independently owned native caches and final zero-centered norm. Hosted genuine original TextModel forward uses two explicitly shared synthetic full-width weight sets across32 distinct original modules/caches and a meta/unallocated unused embedding, avoiding17GB checkpoint-like allocation. Original-first prefill2/cached1/fullprefix3 will compare all32 hidden stages, final norm and every cache. Vocabulary projection remains separate: exact e42de7a3c implementation cannot be integrated here without changing frozen source closures; no alternate projection or fake logits. No model/checkpoint/format5/decoded-quality claim; hosted only.

Initial composition [41979168300](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41979168300) at `bf6a7f930` is retained negative: all347 controls and290/291 comparisons pass, but prefill layer21 recurrent minimum row cosine0.999980919 fails unchanged0.99999. State max error1.0477379e-7 versus magnitude0.01271605; worst row1326 has magnitude9.5542184e-8, L2=2.2861583e-7 and error5.6026082e-9. Upstream hidden20 error9.6082687e-5 versus24.0699482; final norm errors remain1.88e-5–2.21e-5. Exact retained-input block21 attribution uses a dedicated read-only Actions artifact job, regenerates and checks all26 synthetic weight hashes, replays only original/native block21 on source/native hidden20 plus a test-only source-normalized GDN boundary. No32-layer forward repeat, runtime/budget change or qualification promotion.

Retained attribution [41980636649](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41980636649) at `fbab5cf9f` verifies26 regenerated weight hashes and all6 original/native frozen anchors byte-exact.22 same-input block21 comparisons+4 source-normalized GDN comparisons pass unchanged budgets; recurrent same-input cosines0.9999998702/0.9999998445. Original block21 supplied only the retained native hidden20 input reproduces strict cache rejection (max1.0617077e-7,cos0.9999773187), establishing accumulated upstream input drift as sufficient cause rather than a localized GDN mismatch. The original32-layer panel remains negative290/291; no full decoder/original rerun, runtime/budget change or model qualification. [Scope and exact limits](docs/lighton4b-decoder-compose.md), [source-verified combined receipt](tests/results/qwen35-ocr/lightonocr3-4b-decoder-compose-attribution.json). Scoped PR workflow-prefix guard failure41980642106 is repaired by appending the same selector exclusion after existing conditions; old security tests unchanged.

### LightOn4B quantized payload-owner continuation — 5 November 2026

The bounded `feat/lighton4b-gguf-decoder-20261105` lane connects the immutable
format-5 inspector to a CPU payload owner, admits only the frozen Q8_0 or
Q4_K-with-Q8-embedding text recipes, and retains protected vision/norm/recurrent
storage. `CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT` and
`CRISPEMBED_LIGHTON4B_QUANT` are default-off in addition to the existing component
and quant gates. Grouped projections and both decoder block validators now admit
those gated quantized matrices; existing F32/F16 paths remain available.
Chunked prefill preserves the old sixteen-token component boundary and uses
independent layer caches up to the existing4096-token context ceiling.

Hosted portable [controls41997276699](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41997276699)
pass native compilation and nine gate/type/negative controls at`fe27027bb`.
The [receipt](tests/results/qwen35-ocr/lightonocr3-4b-native-context-controls.json)
retains the initial missing-declaration compiler failure and corrected positive run.
Next gate: actual
Q8/Q4 payload admission and native document inference against the pinned original
BF16 execution. Full-position logits, earliest differing stage with magnitudes,
free IDs/text/EOS and held-out documents remain required. The historical negative
290/291 synthetic decoder receipt and its exact thresholds remain unchanged;
it is evidence to preserve, not a prerequisite requiring an F32 model run.
No native quality or release qualification follows from structural admission.
### LightOn4B image composition — 5 November

`feat/lighton4b-image-compose-20261105` composes the PR146 mapped vision geometry,
shared still-image processor and2560-wide image-prefill/positions behind a new
default-off gate. Existing shared engine sources and historical closures remain
unchanged. [Component scope](docs/lighton4b-image-compose.md) records the exact
source and1024-patch limit. Model-free hosted controls are prepared; actual native
processor/vision arrays, quantized full inference/logits and EOS parity remain
open. No Q8/Q4 qualification or public model admission follows from composition.

### Actual LightOn4B native document assembly — 5 November 2026

`feat/lighton4b-native-document-20261105` assembles the tested quantized owner,
image composition, separate format-5 tokenizer and exact quantized streamed-head
function from`eb6b9cb40`. The test-only binary takes source RGB pixels, dimensions
and teacher token IDs; it computes processor patches, native vision/tokenizer,
embedding splice, all32 decoder layers and vocabulary logits. It retains all177
teacher-forced positions separately from independent greedy generation. No
original hidden state or logit is accepted as native input. Initial hosted build
controls precede actual Q8/Q4 smoke and full original BF16 comparison. No synthetic
F32 correction or F32 checkpoint is required to execute that separate experiment.

Integrated hosted [build41998205271](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41998205271)
passes actual native-runner compilation, nine payload-owner controls and all
streamed projection controls (including quantized dequant/dot/seam and gate
checks). [Receipt](tests/results/qwen35-ocr/lightonocr3-4b-native-document-build.json).
Actual separately bounded Q8 [smoke41998868478](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41998868478)
and Q4 [smoke41998910591](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41998910591)
are pending at`b34a07e82`. They use one teacher score and two free tokens only;
full177-position/complete greedy execution and paired BF16 score review remain
separate. Sources/images/candidates are pinned; preflight uses effective
cgroup/host available RAM and free disk, with peakRSS and timing retained.
No model quality follows from building or launching these smoke runs.

Initial Q8/Q4 smoke runs41998868478/41998910591 authenticated both full
candidate hashes and completed resource preflight, then exited4 at tokenizer
metadata admission before native vision/decoder execution: the adapter expected
an obsolete source-only profile. [Q8](tests/results/qwen35-ocr/lightonocr3-4b-q8-native-smoke-initial.json)
and [Q4](tests/results/qwen35-ocr/lightonocr3-4b-q4-native-smoke-initial.json)
retain the exact initial source/hash/gates/preflight. The corrected adapter
matches the exported`lightonocr3-4b` profile. Corrected smoke waits for its hosted
metadata/tokenizer controls. Initial subsecond sampledRSS is not a true process
peak; corrected workers use GNU time child rusage for peakRSS/CPU/wall timing.

Corrected actual smoke at`fd150b98f` is complete: Q8
[41999580781](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41999580781)
and Q4[41999584163](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41999584163)
each execute native154-token image prefill, one full-vocabulary score and two
independent greedy tokens. Q8 yields`[27,17251]`/`<figure`, matching the original
prefix; Q4 yields`[20353,29]`/`<table>` and already fails at the first token.
[Reviewed receipts/resources](tests/results/qwen35-ocr/lightonocr3-4b-native-smoke-review.json)
preserve all exact sources and initial profile failures. No EOS/full-document
quality follows. Guarded original first-score comparisons are pending as
[Q8 42000741513](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42000741513)
and[Q4 42000808436](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42000808436);
then execute full177 teacher positions and independent cap256 generation for
both candidates, preserving the Q4 regression. Original BF16 internal tracing
is a separate continuation; current original capture contains scores only.

The user-requested full runs now execute on the exact reviewed smoke source
`fd150b98f488a6884a6e8d8127e7dfb263cab237`, preserved as
`audit/lighton4b-native-full-fd150-20261105`: [Q8 42001328483](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001328483)
and [Q4 42001333207](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001333207).
Each retains all177 original-prefix score rows and independent greedy generation
at unchanged cap256 with a180-minute hosted bound. Scalar teacher/cache stepping
is unchanged from smoke; no new batching, reference precision or threshold
waiver. Smoke's127/133-second prefill/two-token measurements suggest a potentially
long full autoregressive panel; no full timing or completion is inferred yet.
Retain partial/failure artifacts and the Q4 first-token rejection even if later
positions improve. Final paired full-vocabulary/decoded review remains pending.

### LightOn4B qualification consolidation and completed captures — 5 November 2026

[The qualification umbrella](docs/lighton4b-qualification-consolidation.md)
folds source/evidencePR152/157/161 into a child of runtimePR155 without changing
any runtime `src/` byte, ggml gitlink or VERSION. Exact source heads, raw
receipts, initial failures and local2a62 candidate-negative evidence remain
preserved. No old runtime branch was merged over current source. Existing
producer branches/jobs remain immutable; close supersededPR152/161 only without
deleting their source refs. No new PR or unrelated CI is authorized.

Both actual plain captures completed at exactfd150b98 source: Q842001328483
retains177 teacher rows and160 independently decoded IDs/EOS, differing from
original177-ID French figure at ID9; Q442001333207 retains177 teacher rows and
256 independently decoded IDs, cap/noEOS, incomplete HTML table ending `<td`.
Their exact native receipt hashes are separately admitted for comparison only;
quality remains negative/unqualified. Targeted177-position Ref comparisons are
the next specifically authorized jobs after one consolidation push. Original
reference, prompt/cap, thresholds, defaults, release and app pin stay unchanged.
The independent four-case BF16 capture is complete, including the retained
receipt cap/noEOS source failure. Mixed candidate bytes are packed/readback-
verified; its separately authorized two-case calibration closeout is below.

Scoped full-comparer retries are required after42011206392/42011236969 stopped
at a missing diagnostic I/O import before any comparison. The repair imports
no old protocol/runtime tree: it restores bounded score I/O plus the exact
standalone Ref source/target, with successful import/help checks. Preserve the
reviewed zero-position failure record; do not call it numerical rejection.
Completed independent BF16 reference cases retain549 scores: table222EOS,
Arabic6EOS, receipt256cap/noEOS and65EOS for the frozen `handwritten_letter`
case, which is actually a red serif A logo. Receipt completion and handwriting-
coverage limitations block broad qualification; never replace/tune frozen cases.

### Completed actual BF16/Q8/Q4 parity comparison — 5 November 2026

Corrected targeted [Q842012564139](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42012564139)
and [Q442012559942](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42012559942)
complete every177 original-prefix248,320-score comparison. All rows finite,
0/177 strict passes each, first observed/budget failure0. Q8max2.93674088/
mincos0.958800077; Q4max11.067977/mincos0.666163504. Their red job conclusions
are expected numerical/decoded rejection, with `status=compared` and no harness
error. Q8 independently160IDs/EOS differs from177-ID original; Q4 caps256/noEOS
and leaves incomplete HTML. [Exact reports/recomputed review](docs/lighton4b-qualification-consolidation.md)
retain all sources, payload/report hashes, per-position errors/magnitudes and
complete decoded outputs. Original BF16 `v1` remains frozen; no model/reference
forward, F32 prerequisite or budget/default change was introduced. Both plain
candidates remain unqualified. Only PR155/runtime and PR157/evidence remain open.
The separately authorized lean mixed calibration42013855293 is complete for
table/Arabic only: both native generations reachEOS, table218 differs from222
source IDs and Arabic6 IDs/text match. All228 score rows are finite,0 strict
passes; red is actual rejection, not timeout/harness failure. See
[completed calibration evidence](docs/lighton4b-qualification-consolidation.md#completed-sequential-mixed-calibration).
Receipt source-completion and logo/handwriting-coverage limitations remain; no
native heldout or model qualification follows. Next runtime claim requires
same-input earliest-operator attribution, not another blind model batch.

### Reviewed checkpoint layer0 same-input attribution — 5 November 2026

[The bounded layer0 attribution](docs/lighton4b-layer0-attribution-results.md)
retains exact source-environment BF16 hidden reproduction (394,240 values),
initial hosted one-ULP failure42022722133, and explicit unanchored operator
run42023499031. All80 same-input/same-weight native stage checks pass unchanged
budgets; no local layer0 implementation defect is identified. Original Python
with dequantized Q8/Q4 weights itself reaches hidden max0.03973293/0.18586349
against frozen BF16, versus official same-weight F32 max0.03580666. These are
single-layer F32 operator diagnostics, not new model references or pure quant-only
errors. BF16 original source rounds activations at boundaries that native keeps
F32, including Q/K normalization before F32 recurrence. Keep full decoded Q8/Q4
rejections and hosted BF16 replay failures unchanged. A separately gated BF16
boundary candidate requires an original same-input oracle, local operator
measurements and actual decoded A/B before any runtime/default/quality decision.
No production source, VERSION, ggml, reference, budget or application pin changed.


### Reviewed LightOn4B same-input vision attribution — 5 November 2026

[Operator42023264258](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42023264258)
atf9a14c1be reconstructs all573,440 original layer0 values bit-exactly with
original torch2.13.0+cpu/numpy2.3.5/two-thread settings. Source-fed norm1 first
differs at87 values/max0.001953125/RMS4.82146e-6; independent QKV/attention/MLP
differences remain. See [complete operator metrics](tests/results/qwen35-ocr/lightonocr3-4b-vision-layer0-source-fed.json).

[Mixed-type norm A/B42024288946](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42024288946)
at3b194f40b preserves exact original reconstruction and all frozen baseline
composed hashes, but is negative for promotion: norm1 differences87→52, norm2
73→76; merged RMS0.0223861→0.0199821 while first-logit RMS worsens
0.545670→0.591688 (max2.61007→2.67622). Both independent ordinary Q4 arms
decode`<table>` (IDs20353,29), versus source`<figure` prefix. The
[full negative A/B receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-norm-ab.json)
retains magnitudes/errors and source/artifact pins. Experimental norm code remains
on the existing vision feature branch; this qualification update imports evidence
only. Both experimental gates remain default-off and no quality/release/default
promotion follows. Next: same-input normalization moments/affine contraction
isolation using retained13414500337, before more ordinary full-model smokes.
Initial42022655060 stopped before build/model access on obsolete Kaggle1.8.3
authentication; known-working2.2.2 corrects only the producer dependency.

### Rejected linear-block BF16 activation candidate — 5 November 2026

The first separately gated linear BF16-boundary candidate42026091939/f201bc8c9
is independently rejected: earliest input norm max0.015625, then QKV0.0625 and
hidden official/Q80.03125/Q40.015625 against the original activation oracle.
All legacy recovery hashes remain exact. Its wrapper's green conclusion is a
reporting defect (retained stage flags fail); successor aggregation returns
nonzero for local failures. Pure same-input original F32 quantization separately
starts at QKV: Q80.08341026/Q40.90670824 against magnitude24.30494, reaching
hidden0.01743889/0.15934277. These are actual original-code candidate losses,
not native implementation proof. Bounded norm-only42027758298 at28e4180b8
compares existing sequential/precise RMS and BF16 rounding before any new full
candidate or decoded claim; source/cache-mode controls remain isolated.

### LightOn4B decoder norm operator continuation — 5 November 2026

Norm-only42027758298/28e4180b8 completes: precise RMS reduces F32 maximum
0.0001449585→0.00000762939 and BF16 maximum0.015625→0.00390625, still failing
0.002. At flat330868, native0.587890625 is a BF16 midpoint rounding down while
original0.5878906846 is one F32 ULP above and rounds up. Preserve both red
comparisons and wrapper aggregation limitation; native legacy/cache-mode controls
pass. [The report and limits](docs/lighton4b-layer0-attribution-results.md#earliest-input-normalization-reduction-precise-alone-remains-negative)
remain distinct from runtime qualification. Source mean/scale isolation42028839817
atbdefd15c4 is executing with exact original norm reconstruction required before
source-fed rsqrt/affine alternatives. No expanded decoder or new quality/default
claim follows before these actual first-operator results.

Source mean/scale isolation42028839817/bdefd15c4 completes exact original norm
reconstruction: original means fed to native F32 sqrt/reciprocal yield all154
scales and394,240 F32/BF16 outputs bit-exact; source-scale affine also exact.
F64-rsqrt is negative (3 BF16 differences/max0.00390625). This narrows the local
cause to mean reduction before rsqrt. Pinned Torchcf30153c4c13 sum/mean/AVX2 source
motivates test-only grouped-order alternatives42029945232/215f463bc, with no
production RMS change until actual mean/output comparison. All prior strict and
decoded rejections remain. See [the completed operator split](docs/lighton4b-layer0-attribution-results.md#source-meanscale-isolation-closes-rsqrt-and-affine).


### LightOn4B verified vision math, bounded decoded negative — 5 November 2026

The vision feature branch retains all default-off alternatives.
[Source-header42026490991](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42026490991)
atd7415c4db isolates compiler contraction: original PyTorch2.13 BF16 moments
and affine code compiled with `-ffp-contract=fast` matches both frozen norms,
while `off` reproduces the separate-affine negative.
[Actual production-helper42027253266](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42027253266)
at9c11a924c then matches both573,440-value norms and all1,120 mean/rstd entries
per norm exactly. Unset/empty/0 retain prior bytes. Independent small-manifest
range review confirms identical SHA values, including signed-zero bits; no large
raw arrays were downloaded for review. See the
[operator attribution](tests/results/qwen35-ocr/lightonocr3-4b-vision-norm-operator-attribution.json)
and [production controls](tests/results/qwen35-ocr/lightonocr3-4b-vision-fma-norm-operators.json).

Initial[42028040544](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42028040544)
atfd102fa25 fails safely before native candidate/ordinary inference: an AVX512
runner changes reconstructed original layer0 at3,965 values/max0.015625. Its
[negative receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-fma-norm-avx512-initial.json)
remains. Corrected[42028668863](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42028668863)
at6c3cb6ed9 explicitly pins original AVX2 ATen/oneDNN/MKL dispatch, preserves
exact original layer0 and both source-fed norms, and passes all frozen portable
native baseline hashes. Contracted norm reduces layer0 RMS0.000781620→0.000691054
and merged RMS0.0223861→0.00973525 (max2.46484→1.21094), below legacy shared
F32 merged RMS0.0112389. Q4 first-logit RMS still worsens0.545670→0.560135
despite max2.61007→2.38528. Both independent ordinary arms decode`<table>`
(IDs20353,29), versus original`<figure` prefix.
[Complete A/B](tests/results/qwen35-ocr/lightonocr3-4b-vision-fma-norm-ab.json)
retains magnitudes, controls and source/artifact pins. This is a verified operator
fix/component improvement with negative decoded selection, not model qualification
or a cross-platform claim. Runtime defaults, release and app pin stay unchanged.

The next independent boundary is source-fed QKV.
[Retained-only42030085653](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42030085653)
at403d94e21 reconstructs all1,720,320 original QKV values byte-exactly. Original
eight-register/eight-lane reduction matches every171 native difference plus2,207
controls (2,378 total) for both FMA and separate mul/add. Double variants still
differ68/71 and serial FMA78; generic extra precision is not source math.
[Selected receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-selected-dot.json)
is selected-only evidence, not full-operator or decoded proof. Next: preserve
the gated verified norm improvement in existing PR155, then gated source-order
linear implementation and full source-fed operator validation before more ordinary
smokes. Large vectors remain in pinned remote artifacts. This commit imports
evidence only, with no runtime source changes.

The pinned mean-order comparison42029945232/215f463bc is complete: source-derived
eight-lane grouping matches all154 means/scales and394,240 F32/BF16 norm values
bit-exactly. Sixteen lanes and precision-only sum→mean rounding retain3/2 BF16
differences at0.00390625. [Complete result](docs/lighton4b-layer0-attribution-results.md#exact-source-derived-mean-order-at-the-retained-boundary)
retains source hashes, unchanged budgets and all old negatives. Measured order is
now scoped only to default-off BF16 linear RMS in5ba80426d; actual-helper
42031297677 covers widths2560/128 before composed layer0 or expanded decoder.
No qualified runtime/source/default/alias/release or app-pin change is implied.

Actual gated-helper42031297677/5ba80426d is complete: all154 input scales,
394,240 F32/BF16 input-norm outputs and630,784 real original GDN128 pre-weight
normalized BF16 values are bit-exact. Legacy/cache/direct-entry controls pass;
old alternatives remain red. [Complete helper evidence](docs/lighton4b-layer0-attribution-results.md#actual-gated-rms-helper-both-widths-exact)
is bounded normalization proof, not complete GDN/decoder quality. Composed
same-input layer0 replay42032626361/5b47ed44e now measures the next projection,
with actual same-process mode interleaving and immutable/mixed-mode cache checks.
No decoder expansion/default or decoded-quality promotion precedes that result.

Composed exact-RMS42032626361/5b47ed44e completes all arms and is correctly red:
input norm exact, first remaining QKV max official0.015625/Q8-Q40.03125. Hidden
improves to official/Q40.00390625 and Q80.0078125, still over0.002. All112 legacy
stages plus genuine same-process mode interleave/immutable/mixed-cache controls
pass. [Complete composed limits](docs/lighton4b-layer0-attribution-results.md#composed-layer0-after-exact-rms-next-qkv-failure-retained)
preserve every old rejection. Focused Q8 dot42036721968/f62b28879 now anchors
source normalized/projection bytes and observes actual F32 operands/route before
pre/post-BF16 accumulation hypotheses; no decoder/GDN/model forward is repeated.
Only verified default-off RMS/BF16-block code is a candidate for scoped PR155
integration; QKV/full-block/decoded/platform/release gates remain open.

### Completed real Q8 QKV accumulation attribution — 5 November2026

Focused42036721968/f62b28879 observes20 accepted actual native F32 llamafile
calls with full independent weight/input byte matches and exact observer/repeat
neutrality. All1,261,568 native outputs match eight-lane FMA. Pre-round error
max2.67029e-5/RMS7.38513e-7 passes; BF16 rounding creates841 differences/max0.03125.
At flat18513, source7.890624523 and native7.890625477 cross midpoint7.890625.
Sixty-four-lane FMA/unfused/F64 alternatives retain564/568/502 BF16 differences,
all max0.03125; none is a proven dot fix. Original GEMM partial sums remain
unobserved. [Complete route/operand/hypothesis evidence](docs/lighton4b-layer0-attribution-results.md#real-q8-qkv-dot-tiny-pre-round-error-crosses-bf16-boundaries)
retains all strict and decoded failures. Only already-verified default-off RMS/
linear-boundary code is integrated into existing PR155 at7d1e73464; old path,
quantized storage and known QKV failure remain. Assembled model-free/vision and
full decoder/decoded/platform/release checks remain separate; no default change.


### LightOn4B actual vision operators integrated, Q4 decoded negative — 5 November2026

PR155 preserves all earlier alternatives/defaults and decoder source. Full
original/actual source-fed nn.Linear QKV/out/up/down bytes match at1/4 threads
in42032345547. Original Conv3D42036525469 identifies four partial sums reduced
left-associatively; actual separate gated Conv42037359930 matches all573,440
patch bits at1/4 threads and independently of NN/PATCH alternatives. The earlier
wrong Conv-as-linear path remains separately gated and its negative42033651799
receipt is preserved. No matrix algebra/precision/cosine shortcut is promoted.

[Assembled42038802495](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42038802495)
atedbd2b782 passes actual norm/full NN/full Conv/gate/native build controls and
decoder20 controls. Exact source ZIP13414500337 is preserved in artifact13420770687
member`linear/retained-source-13414500337.zip`, SHA77682c5a,238,129,493 bytes,
expiring3 February2027. Only task-owned script retention changes; no bulk local
arrays, original regeneration or runtime release. All receipts are under
`tests/results/qwen35-ocr/`.

Initial final-smoke42040210848 stops before ordinary execution on an unnecessary
missing historical norm-off receipt, despite complete norm-only overrides. Its
negative is retained; the scoped fallback repair changes no source vector/budget.
[Actual42041219151](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42041219151)
at3a0c7ffc6 then preserves all30 fixed old baseline hashes with decoder gate0.
Patch/position/embedded match original raw SHA values. Layer0 RMS improves
0.000691054→0.000141771 (35,896→1,466 differences); merged RMS improves
0.00973525→0.00866825 but max worsens1.21094→1.67969. First-logit RMS worsens
0.560135→0.568348/max2.38528→2.45592. Both independent two-token decodes remain
wrong`<table>` (20353,29), versus source`<figure`. The complete30-stage smoke
receipt preserves this real quality negative despite intermediate improvements.
All gates stay off; no model/EOS/document/quant/publication/release/app promotion.

Next is one retained-only source-fed attention pooling probe (65 values/max0.00390625),
with exact original and actual-native byte controls before rotation/QK/softmax/PV
attribution. Tiny GELU tails remain separate. This commit imports evidence only;
actual runtime stays in existing PR155.


### 2026-11-05: source-exact vision attention, combined Q4 quality negative

PR155 runtime head `26bce947d` preserves separate default-off gates. Retained
source-fed QK/PV attention reduced 65 to 20 pooled differences (42047075420),
then original AVX2 vector-header attribution 42052003211 and self-contained
SLEEF-compatible production 42053439752 close all 5,017,600 F32 probabilities,
8,960 reciprocals and 573,440 actual pooled values to bit-exact. Initial setup
failure 42050508555 remains retained. Final integration 42055105090 passes
old-gate bytes, exact pool, no invalid/divzero, native/context builds, context 9
and decoder 20 model-free controls. Gate requires x86_64 AVX2/FMA and nearest
rounding; no Torch runtime dependency. License and runtime stay in PR155 only.

Ordinary Q4 A/B 42056882067 at `88aa296dc653c87c493f1357605196df1799f0e9`
preserves all 30 fixed baseline hashes and decoder candidate is explicitly 0.
Candidate patch/position/embedded are exact; layer 0 differences 35,896 to 40,
RMS 0.000691054 to 0.0000239534. But merged RMS worsens 0.00973525 to
0.01839050, first-logit RMS 0.56013479 to 0.59042975, and both decodes still
wrong `<table>` against original `<figure`. Quality remains negative; no default,
release, app-pin or decoded promotion. Runtime-generated rotary and source-fed
activation (14,029 F32 differences/max 1.5367e-7) are next attribution boundaries.
Full source/operator/negative receipts and 90-day artifact identities are under
`tests/results/qwen35-ocr/lightonocr3-4b-{vision-attention,softmax}-*.json`.
This qualification commit changes evidence only, not runtime or workflow.


### 2026-11-05: remaining first-block differences fully attributed

Retained-only 42060487656 at `915c1cd325a0bbe9e77885c17354d0c706e6a3bd`
passes original byte controls and exactly reproduces actual candidate layer0
from verbatim production rotary and GELU expressions. Rotary alone causes 32
final differences; GELU alone 8; both 40. No unexplained first-block remainder.
Native cosine differs at 2,488 and sine at 696 values (max 5.96e-8), leading to
2 rotated-Q / 1 rotated-K differences and 5 pooled differences. Native GELU
differs at 14,029 values (max 1.5367e-7), while installed AVX2 Gelu.h strict/
contracted helpers match all original activation bytes. Rotary is earliest.
Complete receipt `tests/results/qwen35-ocr/lightonocr3-4b-first-block-boundaries.json`
contains exact source/artifact identity (13432171897, SHA
`77fac62ebb8f4e3bb073984927cc0158ec8977e38788b34fe24f808399ca32ac`, 90-day retention).
Prior combined Q4 decoded-quality negative remains; no runtime/default promotion
in this evidence-only qualification commit.


### 2026-11-05: verified GELU and actual MKL rotary backend, Q4 still negative

PR155 head `fb94a5715` preserves all gates off. Initial source-math 42061899237
at f0ea7ff1b is a retained numerical failure: inverse frequencies all exact,
but fused SLEEF trig grid differs at 633 cosine and 237 sine values. Direct installed
SLEEF headers in 42062871623 give those same differences and nonfused is also
negative, so ROTARY_AVX2 is rejected and stays0 in quality candidates.

Self-contained source-derived GELU matches all 2,293,760 activation bytes. Final
integration 42064278334 at `9da1757cd3103180cfa809406452719e2368a87e` passes
old/unset/empty/0, original/combined first-block controls, native/context builds,
context 9 and decoder 20 plus env docs. Original build is oneMKL 2024.2 (20240605),
AVX2; direct reference-only vmsCos/vmsSin HA calls match all 2,000 original grid
angles bit-exact. No MKL/Torch product dependency or reference table is added.

Ordinary Q4 GELU 42065352208 at `40c68f35f81c8403eb22b27b642037df3fd34c8d`
preserves all 30 baseline hashes, decoder 0 and rejected rotary 0. Actual layer 0
has 32 differences (RMS 2.28155e-5 / max 0.0078125), matching known rotary attribution.
Merged RMS improves baseline 0.00973525 to 0.00775425 (max 1.21094 to 0.734375), much
better than previous candidate 0.01839050. First-logit RMS 0.56432682 remains worse
than baseline 0.56013479 (max 2.48687 versus 2.38528), though better than prior
candidate 0.59042975. Both decodes remain wrong `[20353,29]`, `<table>`, against
original `<figure>`. Quality stays negative/unqualified; no default/release/app
promotion. Native elapsed 177.66 s, peak 4,510,707,712 bytes; not a speed claim.

That run's test-only activation replay still hardcodes legacy GELU, so its 14,029
activation differences do not observe the verified production helper. Actual
native composed score/decoded path is unaffected. Observer is fixed in PR155
for future runs, without runtime change or a duplicate model job. Complete
source/failure/MKL/negative receipts and 90-day artifact identities are under
`tests/results/qwen35-ocr/lightonocr3-4b-{source-trig-initial-negative,trig-dispatch-gelu-proof,gelu-integration-mkl-control,gelu-composed-negative}.json`.
No further model jobs in this lane. A possible next bounded operator-only action
is retained same-input merger GELU/erf attribution, keeping earliest MKL rotary
dependency explicit. This qualification commit changes evidence only.


### 2026-11-05: final measured merger correction, Q4 quality still negative

PR155 final head `579aa53c1` preserves default-off corrections and alternatives.
Merger attribution 42069010152 at `c398b710b34987b960ce54dddfb8feeed35a3b5d`
passes complete original and actual merged byte controls using retained final
vision-layer inputs and a bounded 54.5 MB original parameter range. Norm/up and
installed original Gelu.h are exact. Native std::erf differs at 33 source-input
and 26 actual-input activations (max 1.52588e-5), propagating to 84/58 merged
values (max 0.00048828125). Source-derived FMA erf plus verified exp closes all
573,440 activation and 358,400 merged values on each input to exact bytes.

Focused production 42069996673 at `ca8e6dcade7fc85ec7adecb8dc904ee64c0db009`
passes those complete original/actual controls and actual gated production
outputs, unset/empty/0 preservation, nonfinite/unsupported-rounding guards,
native/context builds, context 9, decoder 20 and environment docs. Separate
MERGER_ERF_AVX2 gate remains off by default and uses the supported x86_64
AVX2/FMA nearest-rounding profile; existing std::erf and fallback are preserved.
Runtime and workflow changes remain in PR155 only, not this evidence commit.

Final ordinary Q4 42070806605 at `e7730d91e7b7b6f9bec5c59b33c612ddcc97254b`
uses all verified vision corrections, with rejected rotary and decoder gates
explicitly 0. All 30 fixed baseline hashes reproduce; all eight source-fed
first-block operators have zero differences, including the corrected activation
observer. Ordinary native inference is independent of the source-fed arrays.
Actual layer 0 retains exactly the 32 known MKL-rotary differences, RMS 2.28155e-5
and max 0.0078125. Merged RMS improves baseline 0.0097352499 to 0.0077542505
(max 1.21094 to 0.734375). First-logit RMS 0.5643236502 remains worse than
baseline 0.5601347910 (max 2.48687124 versus 2.38527584), though slightly better
than prior GELU candidate 0.5643268159. Both decodes remain wrong `[20353,29]`,
`<table>`, unlike the original `<figure` prefix. Quality stays negative and
unqualified; no default, release or app-pin promotion.

Same-run native elapsed is 105.17 s baseline / 115.84 s candidate, user/system
CPU 75.42/56.22 s versus 100.42/55.81 s, peak RSS 4,524,982,272 versus
4,510,666,752 bytes. This is one Q4 image and two-token smoke, not a throughput
benchmark, EOS or full-document qualification. Actual GGUF merger weights were
not separately downloaded; complete actual merged byte control establishes
compatibility of the original parameter range. The oneMKL rotary dependency
remains unresolved in the product; no reference table or Torch/MKL product
dependency is added. Exact sources, raw-byte controls, observer scope, timings,
90-day artifact identities and limitations are in the three merger-erf receipts
under `tests/results/qwen35-ocr/`. No further work or runs in this lane.


### Genuine BF16 quantized-weight first-position oracle — complete, rejected

Source `048fd683f` completed the single bounded CPU worker against immutable
original `8d3b2331`. The original full score SHA `a4bece2310…5ffd` is exact.
All three arms use genuine BF16 parameters, one fresh image/text prefill each;
no F32 model or native forward is run. Complete mapping/readbacks verify all
723 tensors, 249 changed quantized text matrices and 474 protected originals,
unchanged buffers and tied embedding/head. Q8 first ID27 stays `<` but score
max/RMS error is0.47265625/0.07158037, cosine0.99953436. Q4 changes toID20353
`<table` with max/RMS2.546875/0.59156984, cosine0.97588336. Both strict gates
reject independently of native implementation. Dequantized weights are explicitly
rounded to BF16 (maximum rounding Q8.0009689331/Q4.0018157959), distinct from the
prior F32-dequant activation diagnostic. Original/native full decoded negatives,
EOS policy and frozen177-position reference remain unchanged. Worker terminal
error is intentional numerical rejection after complete capture, not an OOM;
peak cgroup31,310,606,336 of32,212,254,720 bytes.
[Full exact receipts and limits](docs/lighton4b-bf16-quant-first-score-results.md).
No runtime/default/release/app-pin change follows.

A possible subsequent source-BF16 mixed diagnostic is plan-only: Q8 embedding1,
linear-attention120 and dense-attention32 matrices; Q4 FFN96; preserve474 originals.
It would authenticate both full existing public files sequentially and stream
selected BF16 subsets into one resident original model, without publishing a
conversion. Freeze exact name sets, source/candidate hashes, one-position cap,
prepared inputs and comparison gates before dispatch. This existing image is
exploratory mechanism evidence after the Q4 observation, never new held-out
calibration or architecture-quality selection. No mixed worker is dispatched.


### Fixed mixed original BF16 exploratory mechanism check — complete, rejected

Source `50307be7f` completed exactly two one-token original BF16 prefills under
the predeclared153Q8 embedding/attention +96Q4 FFN policy. Original first-score
SHA and all723 original parameter hashes agree with the prior oracle; all249
final mixed hashes exactly match their corresponding earlier Q8/Q4 arm hashes.
All474 protected originals and tied output head remain unchanged. Mixed selects
ID27 (`<`) with probability0.76989954, but complete-score max1.578125/RMS0.28579991/
cosine0.99251625 rejects unchanged gates. This existing-image experiment was
declared after the Q4 first-token observation: it is exploratory mechanism
evidence, not held-out calibration or architecture-quality selection. No native
inference ran, and one token cannot qualify native/full-original decoding orEOS.
Smaller chunks/cache advice pass capacity guards; cgroup peak29,645,406,208 bytes
of32,212,254,720, no OOM/retry. [Full receipts and exact scope](docs/lighton4b-bf16-mixed-first-score-results.md).
Two small authenticated legacy-native first rows also give joint native-versus-
BF16-quant residual Q8 max0.46874571/RMS0.09304085, Q4 max0.76940751/RMS0.08628066.
This mixes runtime and effective-weight precision, not isolated implementation
error or subtractable quantization loss. No additional worker/runtime fix, model
upload, default, alias, release or app-pin promotion follows.


### 2026-11-05: complete configured rotary arithmetic, vision raw parity and Q8 score improvement

PR155 evidence/runtime handoff is `269ed8dbf993ed5e60dfef2bddc79e6066abcbed`.
The complete rotary asset is configuration-defined arithmetic, not captured
image/activation outputs: source width 1024 / heads 16 gives 16 frequencies, original
default theta 10000, native axes bounded 1024 gives coordinates 0..1023. All
coordinates/frequencies/functions occupy 128 KiB. Public reproducible generator,
source config/default-function pins, exact raw/text hashes and MIT/generated
arithmetic plus Apache2 config provenance are retained in PR155. No MKL code,
binary or product dependency is introduced. Separate ROTARY_TABLE gate is off
by default and preserves old/rejected alternatives and unsupported fallbacks.

Domain proof 42096520795 at `79fbff4e74ba084c35804ba96178a15d2e3e2da0` checks
independent regeneration, all 32768 direct original MKL values and compiled
bytes. Production integration 42098420346 at
`7648267c79453b277b6ae3f8c43aa54769903264` repeats 681,459,712 F32 values over
3,873 layouts, all 3,280 admitted single-frame h/w pair and 1,024 total-token
counts plus temporal/multigrid edges. Mathematical coordinate-domain coverage
is complete; finite packing cases are not a claim of enumerating all combined
grids. All 22 reader/production fallback/precedence controls, source tables,
attention and full first-block bytes, native/context builds, context 9, decoder 20
and environment docs pass. Raw bank SHA is
`30ea508c92d92a8400604e276fac161a1a85f7e89cd79877f227684a6d330e0c`.
Reference generator/code remains outside product artifacts; only the configured
constant bank and reader enter the gated runtime.

Q4 ordinary 42099925931 at `a1bb8c924607e2631435a973512aeee94ea5f0be` and
independent retained-only hash audit 42102908743 establish all 29 emitted
processor/vision stages SHA-identical to original for one image. All 66 shared
float files and 30 fixed baseline hashes were independently checked. Five
original-only intermediates (merger norm/up/activation and rotary cos/sin) are
not emitted by ordinary runner; they have separate operator controls. Position
raw layouts differ (original chronology+THW, native THW), so raw inequality there
is not alone a position-math failure. First downstream shared mismatch is text
embedding: 391,330/394,240 values, max 0.000380992889, RMS 6.30394e-5; later error
is not attributed solely to embedding. Q4 first-logit RMS worsens 0.5601347910 to
0.5879641585 (max 2.38528 to 2.44462), and both decodes remain wrong `<table>` /
[20353,29] versus source [27,17251] `<figure` prefix. Native elapsed baseline/
candidate 130.86/153.89 s, peaks 4,524,945,408/4,510,699,520 bytes: diagnostic
one-case measurements, not throughput claims.

One corrected Q8 smoke 42104609427 at
`f120d61a97a82acc7d3e4a2f2e02a5a1354e5ee4` passes immutable 5,815,503,232-byte
model SHA, 10 GiB available-RAM / 12 GiB scratch guards, all 5 original canonical
prepared-input hashes and 29 raw vision hashes. Decoder candidate and rejected
SLEEF rotary remain 0, caches off. Original first-row RMS/max is now
0.0742289135 / 0.460906029. Compare both same-model/same-image historical sources:
fd150 original F32-vision smoke approximately 0.130495042 / 0.568259001 (independent
float64 review RMS 0.1304950347489), and 891d early BF16 candidate
0.2005369525 / 1.092198968. Source commits and hosted workers differ; no same-source
A/B, table-only causal or speed claim. All three Q8 prefixes were already correct
[27,17251], `<figure`, so no new-prefix win is claimed. Current Q8 native elapsed
139.13 s, user/system 134.01/57.80 s, peak 6,317,207,552 bytes; fd150 127.69 s /
6,316,945,408 bytes, 891d 105.43 s / 6,317,084,672 bytes, diagnostic across hosts.

Full 177-token/EOS and whole-document quality remain untested in this lane. All
248,320 Q8 logits still differ from BF16; Q4 prefix remains incorrect. Future
observer now separates exact_original_prefix_ids from full exact_original_ids;
the latter cannot test a two-token prefix against 177-token source completion.
Independent prefix audit protects current evidence. Raw/profile/integration/Q4/
Q8 receipts contain exact sources, both historical baselines, complete artifact
hashes / 90-day retention, performance and limitations. All defaults, release and
app pin stay unchanged. This PR157 follow-up changes only receipts and plan/
history; runtime and generated asset remain in PR155.

### Corrected native mixed form diagnostic


### Frozen-case runner and mixed storage admission — 5 November

`feat/lighton4b-frozen-cases-20261105` extends the exact`fd150b98f` prototype in a
separate isolated lane. The immutable active full panels remain unchanged. An
explicit default-off`CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4` admits only the
fixed policy: all96 FFN matrices Q4_K, all152 linear/full-attention matrices Q8_0,
the same Q8_0 tied embedding/head, and the existing protected F32 vision/norm/
recurrent tensors. Uniform old Q8/Q4 policies remain distinct and arbitrary
mixed storage is rejected. Metadata-policy controls precede actual file use.
The generalized worker will authenticate frozen image/cap/original-reference
case manifests and weights; no source image, prompt, gold or budget tuning.

Hosted[42005163384](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42005163384)
at`7f49713bc` passes native preflight compilation,74 metadata-policy controls,
old owner and streamed projection controls without model payload access.
Subsequent source binds the mixed metadata key exactly and adds the generalized
worker`tools/ci-heavy/lighton4b_frozen_case.py`. It authenticates the reviewed
four-case protocol/PNG/RGB freeze and explicit original-capture/weight SHA256s,
then compares native pixel/ID/modality/grid arrays exactly to authenticated
original prepared inputs before any payload-backed inference. Original arrays
are comparison-only, never inputs to the native decoder. Source-capped cases
remain negative diagnostics. The frozen cap256 and calibration/heldout split
are unchanged. Actual mixed weights, original four-case admissions and decoded
panels remain pending; no admission or quality is inferred from preparation.

Final-source frozen worker[42006252927](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42006252927)
at`84691b26494e61d82efcc436b083b48a1a639236` passes all three fixed-case
contract tests,74 exact mixed/uniform storage controls, nine owner controls and
old streamed projection controls; both actual native/preflight binaries compile.
[Receipt](tests/results/qwen35-ocr/lightonocr3-4b-frozen-case-build.json) retains
exact executable hashes. No model payload or source capture was consumed.
The worker remains blocked on independently reviewed original per-case capture
SHA256s and actual mixed pack SHA256; these are explicit input admissions rather
than inferred quality. Preserve active full panels and all initial negatives.

### Consolidated LightOn4B native spine — 5 November

PR155 now folds PR159's exact mixed policy/frozen-case worker and preserves
PR154's already-contained payload-owner history. Unique PR144 vision-owner
controls/platform wiring, PR147 structural inspection controls/header receipts,
and PR153 actual preprocessing replay/initial dependency negatives are retained
with their original source/run identities. Existing implementation blobs were
kept; the newer mixed recipe accessor is not overwritten by the older inspector.
Archived completed component records live in HISTORY. Historical original guards,
arrays, budgets, negative outcomes and source branches remain unchanged.

The failed direct-quant precision alternative stays only on its retained PR156
branch; it is not added to this spine. Original full runs42001328483/42001333207
continue on exact`fd150b98f`; no run is cancelled, repinned or interpreted as
complete. New model dispatches and new PRs are paused while the user-requested
PR/CI consolidation proceeds. No new runtime release or quality promotion.

Consolidated`cf1e7248c` automatic checks exposed four missing test clean-exit
routes and owned-file clang18 style violations; both are repaired locally.
Aggregate red/skipped build jobs followed failed static prerequisites and are
not execution evidence of a platform/numerical regression. The source-pin failure
for`qwen35_linear.cpp`/`qwen35_math.h` predates consolidation: their pre-merge
bytes were unchanged from PR155. Reviewed exact-byte amendments use the existing
append-only journal and leave every frozen receipt/hash/budget intact. Four old
mathematical function bodies remain byte-identical to main.
[Static repair record](tests/results/qwen35-ocr/lightonocr3-4b-consolidation-static-repair.json)
records initial runs and limits. All713 formatting checks,264 clean-exit mains,
three case contracts,13 quality-audit controls and15 git-verified source pins
pass locally; no model inference/build or new CI dispatch is implied. Repairs
are held locally for the coordinated queue-preserving push.

Both exact`fd150b98f` full native captures completed: Q8
[42001328483](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001328483)
and Q4[42001333207](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42001333207)
retain all177 original-prefix score rows. Independent Q8 generates160 IDs and
EOS but differs from the original177 IDs at position9; raw original-output
CER0.2134146/WER0.3888889. Q4 generates256 IDs, reaches the unchanged cap without
EOS and truncates an HTML table; CER0.8277439/WER0.9888889. These are explicit
decoded-parity negatives, not quality qualification.
[Exact reviewed receipts/resources](tests/results/qwen35-ocr/lightonocr3-4b-full-native-review.json)
retain remote archive/source/weight/protocol identities and limits. Full177-row
numerical comparison is a separately controlled continuation; no reference, cap,
prompt or strict budget changes. Large arrays stay remote.

### Single sequential mixed calibration job

The reviewed mixed candidate is4,683,041,248 bytes, SHA256
`d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474`,
retained by pack42006450732/artifact13405370923. A single scoped hosted job
stages this candidate once, builds once and runs only the two predeclared
calibration cases sequentially, with original222/6 EOS-ended positions.
[Scope and bounds](docs/lighton4b-native-calibration.md) preserve the heldout
receipt's capped-without-EOS source failure and prohibit qualification.
The actual four-case capture schema is bound by source/protocol/case hashes;
it has no invented `requested_dtype` field. Reviewed source42e5ce6f9 and batch
metadata fix BF16. Reused binaries require exact source/build-receipt/executable
hashes. Local transfer-overrun/hash/source/binary-tamper and selector controls
pass; no broad CI or new PR is required.


### PR155 bounded gated BF16 vision norm integration — 5 November 2026

Preserve all existing mixed-policy/preflight/inspection targets and legacy shared
vision. The gated BF16 vision owner now retains both previous norm alternatives
and `CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM`, whose actual source-fed operators
match original AVX2 bytes in42027253266. All gates remain default-off. The
[candidate guide](docs/lighton4b-vision-norm-candidate.md) records exact sources
and **negative decoded qualification**: composed42028668863 improves merged
RMS0.0223861→0.00973525 but worsens Q4 first-logit RMS0.545670→0.560135 and
keeps incorrect`<table>` output. No release, default or app pin changes.

Required assembled-source gate is the single targeted hosted
`tools/ci-heavy/lighton4b_vision_norm_integration.py`: actual production norm/raw
SHA/mean-rstd and unset/empty/0 controls, finite/tie BF16 rounding, native-context
and image-compose controls, compiled ordinary native runner argument rejection
and environment-doc freshness. No model inference or broad CI is requested.
Source-fed QKV selected-dot42030085653 is positive on2,378 positions, including
all171 historical differences, but a linear runtime change needs full operator
validation before another composed smoke. It is not part of this norm change.


Initial assembled[42031440977](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42031440977)
atb75e722a7 stops before artifact transfer/build/model execution because this
branch's generic heavy job does not expose its read-only actions token. The
[initial failure](tests/results/qwen35-ocr/lightonocr3-4b-vision-norm-integration-initial.json)
remains. The scoped workflow repair adds only read-only actions permission and
`GH_TOKEN` to that existing job, preserving calibration-native exclusions/jobs.
Runtime source and budgets stay unchanged; only the targeted assembled job retries.


[Assembled retry42032227007](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42032227007)
at16b849573 passes exact norm/source/raw-SHA/moments/gate controls and builds
all four targets. Native-context9 controls pass. It exposes an existing renamed
image-compose test helper falling through without `return0`:45 passing assertions
print success, followed by undefined execution/spurious line98 failure. The
[second initial receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-norm-integration-test-initial.json)
is retained. Add only explicit helper `return0` and clearer failed-control
metadata; preserve every assertion, runtime byte and numerical gate. Retry only
the same targeted assembled-source job.


[Assembled42033147414](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42033147414)
at43b2d2c35 passes actual production norm/source-SHA/moments/gate controls,
BF16 finite/tie rounding, native-context9 and image-compose45 controls, ordinary
native-runner compilation/bad-argument rejection and environment-doc freshness.
The [complete integration receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-norm-integration.json)
pins all seven relevant runtime hashes; BF16 source09394e54 and shared768 norm
remain unchanged by this evidence-only update. Both initial workflow/test-helper
failures remain preserved. Gated norm integration build/control gate is closed;
model/decoded quality remains negative and every new default remains off.

### Scoped decoder BF16 linear-boundary integration — 5 November2026

Existing PR155 receives only the verified default-off linear-block boundary path,
source-order RMS and precision-tagged caches, preserving old F32 execution and
both vision gates. [Scope and measured limits](docs/lighton4b-decoder-bf16-boundaries.md)
retain exact input/GDN RMS controls and composed42032626361's first-QKV failure;
no dot alternative/default/quality promotion is included. New model-free gate/
rounding/delta-isolation controls require actual assembled hosted execution.
Full attention/final norm, full decoded output and final platform/release checks
remain separate; source-fed arrays/probes stay outside production artifacts.


### PR155 scoped vision nn.Linear and Conv3D assembly — 5 November 2026

Preserve decoder7d1e73464 and all earlier mixed-policy/preflight/inspection
sources/targets. `CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR` now applies only to
true nn.Linear. Four complete actual operators match original bytes at1/4
threads in42032345547. Conv3D is a separate default-off
`CRISPEMBED_LIGHTON4B_VISION_BF16_CONV` path with source four-partial/left-fold
GEMM order:42037359930 matches all573,440 original patch values at1/4 threads
and independently of NN/PATCH gates. Old F32 patch and rejected patch-as-linear
remain behind existing/default-off `CRISPEMBED_LIGHTON4B_VISION_BF16_PATCH_LINEAR`.
Unset/empty/0 controls preserve all historical bytes. Shared768 vision stays.

Combined42033651799's negative remains in the full receipt: applying nn.Linear
order to Conv3D changed20→34 patch values, while merged RMS improved
0.00973525→0.00772041 but first-score RMS worsened0.560135→0.568035 and both
ordinary Q4 arms still output wrong`<table>`. Those results do not qualify current
NN-only/Conv-corrected composition. All runtime defaults remain off; no release,
model/decoded/EOS/document/publication or app-pin promotion.

Single targeted `tools/ci-heavy/lighton4b_vision_linear_integration.py` must replay
complete actual norm/NN/Conv original-byte/gate controls, native builds/controls,
and decoder20-control census on assembled source. It preserves exact original
ZIP13414500337 (SHA77682c5a,238,129,493 bytes) remotely without regeneration.
Only this script's artifact retention is90 days; all other workflow retention
stays unchanged. Reference/source probes remain test-only, with no bulk local
artifact download or model forward in this assembled gate.


[Assembled42038802495](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42038802495)
atedbd2b782 passes exact norms, all four complete nn.Linear outputs at1/4
threads, original Conv3D573,440 outputs at1/4/independent gates, old-path byte
controls, native builds/controls and decoder20 controls. The
[complete integration receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-linear-integration.json)
pins nine actual runtime files. Exact source ZIP13414500337 is preserved as
`linear/retained-source-13414500337.zip` in artifact13420770687, SHA77682c5a,
238,129,493 bytes. Artifact API confirms expiry3 February2027; no original arrays
were regenerated or bulk downloaded locally. This closes assembled operator/build
controls, not full-model quality. Runtime bytes are unchanged by this receipt.

Next single targeted Q4 `tools/ci-heavy/lighton4b_vision_linear_conv_ab.py` holds
decoder candidate explicitly0 to isolate verified vision norm+NN+Conv changes.
The paired ordinary baseline must match all30 fixed norm-only source42028668863
stage hashes before judging candidate first logits/independent two-token text.
Source/operator rechecks remain required; no default/quality/release promotion.
Only the two task-owned integration/final-smoke scripts retain artifacts90 days;
all unrelated workflow retention stays unchanged.


Initial final-smoke[42040210848](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42040210848)
at80a553c7b passes original layer0 and source-fed corrected operators, then
stops before ordinary inference on an unnecessary absent historical norm-off
receipt. Complete fixed norm-only30-stage overrides were already supplied. The
[initial failure](tests/results/qwen35-ocr/lightonocr3-4b-vision-linear-conv-smoke-initial.json)
remains. The scoped driver repair uses complete explicit overrides directly and
retains the old fallback protocol otherwise; no runtime, hashes, gold or budgets
change. Retry only the same targeted Q4/legacy-decoder smoke. Previous actual
source-order native candidate42033651799 completed in180.3s/4.51GB peak RSS;
this diagnostic is not a speed claim or current whole-source qualification.


[Final Q4 vision smoke42041219151](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42041219151)
at3a0c7ffc6 completes with decoder candidate explicitly0 and all30 fixed
baseline hashes preserved. Actual patch/position/embedded now equal original
SHA bytes exactly. Layer0 RMS improves0.000691054→0.000141771 and differing
values35,896→1,466 (max0.03125→0.015625). Merged RMS improves
0.00973525→0.00866825 but max worsens1.21094→1.67969; first-logit RMS worsens
0.560135→0.568348/max2.38528→2.45592. Both independent outputs remain
incorrect`<table>` (20353,29) versus original`<figure` prefix. The
[complete30-stage receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-linear-conv-smoke.json)
retains absolute magnitudes/errors, sources and actual90-day raw artifact expiry.
Quality remains negative/unqualified; no defaults, budget, release or app change.

Next bounded operator lane uses retained QKV/cos/sin/pool arrays to isolate first
remaining source-fed attention-pooling mismatch:65 values/max0.00390625. Recreate
original pooled bytes, require actual candidate output to match retained old bytes,
then compare Q/K rotation, BF16 QK matmul, F32 softmax and BF16 PV accumulation
on identical original operands. Tiny GELU tail differences remain separate.
No new full model batch or runtime attention fix precedes that attribution.


### PR155 partial attention dot-order correction — softmax still incomplete

Source-fed42044794137 establishes exact Q/K rotations, first QK mismatch33 values
(max0.125), exact source eight-register QK, and exact four-partial PV. Gated
production42047075420 reduces pooled65→20 differences, max0.00390625→0.001953125
and RMS6.969e-6→3.994e-6, with actual/observer/disabled-gate byte controls exact.
The default-off `CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION` applies these QK/PV
orders only; previous attention and all earlier paths remain preserved.

Whole attention is **not exact**. Even original softmax reciprocal plus source/std
exp leaves12 probability differences; best vector normalization leaves15. No
softmax guess, full-model quality or default promotion follows. Actual Q4 decoded
quality remains negative from42041219151. Focused assembled
`tools/ci-heavy/lighton4b_vision_attention_integration.py` must reproduce exact
primitives, verified partial candidate bytes, old gates, native build/context and
decoder controls. Reference probes stay test-only; no model forward at this gate.


[Focused assembled42048810358](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42048810358)
atc3abc6057 passes exact QK/PV source primitives, original/actual/observer and
disabled-gate controls, and the verified partial-candidate SHA4de43a53. Native
runner/context build and decoder20 controls pass; environment docs are fresh.
The [integration receipt](tests/results/qwen35-ocr/lightonocr3-4b-vision-attention-integration.json)
preserves65→20 pooled differences (max0.00390625→0.001953125) explicitly as
**incomplete whole attention**, with no new model run or decoded promotion.
Softmax still has12 differences even under forced original reciprocal, and best
reduction15; all new gates remain off. Runtime bytes do not change in this receipt.

Prepared next attribution: original installed PyTorch vector-header softmax
controls, compiled at original AVX2 and default dispatch with strict/contracted/
fast-math flags, versus the retained full F32 softmax/probability/reciprocal bytes.
Check actual header/git/library identities and exp0 before source inference; stop
on linkage/original controls. No guessed exp approximation is imported into runtime.

### 2026-11-05: exact source-fed vision softmax integration

Reference retry [42052003211](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42052003211)
at `13909256012d8e85483a121e1c9c26db884dc3aa` reproduces all 5,017,600
F32/BF16 probabilities and 8,960 reciprocals with the original AVX2 vector headers.
The initial namespace-only compile failure 42050508555 is retained; no arrays or
model were transferred before it failed. Standalone Torch.exp differs at 493,502
values, so it cannot replace the original softmax vector exp.

Production [42053439752](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42053439752)
at `7284cec3e988bd302f1b8529bb08201ddea00abc` validates a self-contained adaptation
of pinned SLEEF xexpf (Boost 1.0 notice retained), explicit FMAs and original
8-lane reduction. All F32 probabilities, reciprocals and 573,440 actual pooled
values match original bytes: old 65 differences, partial QK/PV 20, exact softmax 0.
Empty/0 and ATTENTION-off preserve previous bytes. No Torch dependency is added.

The separate default-off `CRISPEMBED_LIGHTON4B_VISION_BF16_SOFTMAX_AVX2` requires
ATTENTION and x86_64 AVX2/FMA with nearest rounding. Unsupported platforms,
nonfinite input and unsupported shapes preserve the previous softmax. Integration
also skips an unused legacy division on the new path; focused final verification
is pending. Source-fed cosine/sine remain distinct from runtime-generated rotary
tables. Prior negative decoded evidence remains; no quality, default, release or
app-pin promotion. Exact source, artifact identities and 90-day expiry are in
`tests/results/qwen35-ocr/lightonocr3-4b-softmax-{vector-initial,vector-source,production}.json`.

Final focused integration [42055105090](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42055105090) at `102994438f5b25db1d40dfc6b452da9c4b3102ca` passed all exact source F32 probability/reciprocal/pooling checks, preserved old gates, checked no invalid/divide-by-zero exception on the exact path, built native/context and passed context 9 + decoder 20 model-free controls and env docs. Artifact `13429762220`, SHA `d74896b641020d77bd326179763d4cd9ca4412b81ec989d3a0f161b2f45c078b`, retained 90 days. Next: one ordinary Q4 composed first-score/two-token A/B with decoder candidate explicitly 0, all vision corrections enabled only in candidate, and all 30 fixed-baseline hashes required. Runtime-generated rotary and decoded quality remain unqualified until measured.

### 2026-11-05: exact-attention candidate Q4 quality remains negative

Ordinary [42056882067](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42056882067)
at `88aa296dc653c87c493f1357605196df1799f0e9` passes all 30 fixed baseline hashes
and source-fed norms/QKV/pooling/out/up/down byte controls. With decoder candidate
explicitly 0 and all verified vision gates enabled, patch/position/embedded match
original bytes. Layer 0 differences fall 35,896 to 40, RMS 0.000691054 to
0.0000239534 (max 0.03125 to 0.0078125). This is still not composed parity.

Merged RMS worsens 0.00973525 to 0.01839050 (max 1.21094 to 2.57422), and first-logit
RMS worsens 0.56013479 to 0.59042975 (max 2.38528 to 2.55264). Both independent
smokes still decode `[20353,29]`, `<table>`, against original `<figure`. **Quality
remains negative and unqualified; all correction gates stay off.** Source-fed
activation still has 14,029 small F32 differences (max 1.5367e-7), and runtime
rotary generation remains a separate boundary. Next bounded job must isolate
those on retained identical inputs before any further correction.

Full numerical/decoded/resource receipt: `tests/results/qwen35-ocr/lightonocr3-4b-softmax-composed-negative.json`.
Artifact `13430388075`, SHA `1e45293758c7c914da4d40d8c96bb4a67a3b3af4a33cd4eaab50a4db201f264b`,
retained 90 days. Two-token smoke is not EOS or full-document qualification.

Bounded retained-only next attribution prepares verbatim extracted production rotary/GELU expression helpers, original installed AVX2 Gelu.h controls (strict/contracted), and four first-block compositions (original, native rotary, native activation, both). Require original bytes and combined reproduction of retained actual 40-difference candidate before attribution. No model download/full-model smoke, production correction or gate change. Reference target is excluded unless its generated-source directory is explicitly supplied.

### 2026-11-05: all 40 remaining first-block differences attributed

Retained-only [42060487656](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42060487656)
at `915c1cd325a0bbe9e77885c17354d0c706e6a3bd` passes original controls and
reproduces the actual combined candidate first-block bytes exactly from verbatim
extracted production rotary + GELU expressions. Native rotary alone causes 32
final differences; native GELU alone 8; combined 40. No first-block remainder.

Native cosine differs at 2,488 and sine at 696 values (max 5.96e-8), producing 2
rotated-Q and 1 rotated-K differences, then 5 pooled differences (max 0.00012207).
This is the earliest remaining first-block mismatch. Native GELU differs at
14,029 values (max 1.5367e-7), while installed original AVX2 Gelu.h strict and
contracted helpers reproduce all source activation bytes. No production change
is inferred without equivalent self-contained math and old-gate verification.
Prior combined Q4 quality remains negative. Receipt and artifact identity are in
`tests/results/qwen35-ocr/lightonocr3-4b-first-block-boundaries.json` (artifact
`13432171897`, SHA `77fac62ebb8f4e3bb073984927cc0158ec8977e38788b34fe24f808399ca32ac`, 90 days).

Prepared source-derived default-off GELU_AVX2 and ROTARY_AVX2 helpers: pinned
SLEEF double-float tanh plus original GELU cube grouping, and small-range u35
sin/cos. Nonfused operations retain explicit F32 rounding; Boost 1.0 retained.
Eligibility remains x86_64 AVX2/FMA and nearest rounding; rotary supports fixed
64-dim/theta10000 with coordinates below125, otherwise old path. Pending bounded
operator checks include full supported trig grid/inv-frequency, all retained
activation bytes, original/old-gate controls and corrected composed first block.
No production math is promoted before those checks or ordinary decoded A/B.

Production-math operator 42061899237 at f0ea7ff1b is **rejected**, before any quality
smoke: old/unset/empty/0 and prior actual 40-difference controls pass; all 16
inverse frequencies are exact. But fused u35 trig grid differs at 633 cosine and
237 sine entries (max 1.19e-7 / 5.96e-8), so the new rotary gate is not qualified
and stays off. GELU had not yet been evaluated when the strict trig guard stopped.
Initial numerical failure receipt is retained. Next one bounded job compares
installed vector-header trig and nonfused source alternative, and evaluates GELU
independently even if trig remains negative; no smoke until exact correction.

Operator 42062871623 at d6686c456 validates self-contained production GELU:
all 2,293,760 retained activation values are bit-exact; original AVX2 Gelu.h and
old/unset/empty/0 controls pass. Rotary remains rejected: direct installed AVX2
sin/cos headers give the same 633/237 trig-grid differences as fused SLEEF,
and nonfused SLEEF is also negative. Combined rejected rotary changes 67 first-
block values, so it must not enter a quality candidate.

Source inspection identifies a separate potential backend: UnaryOpsKernel.cpp
uses at::vml::vcos/vsin, and vml.h specializes F32 to MKL vmsCos/vmsSin with HA,
FTZDAZ_OFF and ERRMODE_IGNORE when MKL is enabled. Next focused integration
confirms MKL build/symbol availability and a direct HA reference control if
available, without adding any MKL dependency to product code. GELU-only candidate
keeps original rotary path, verifies first-block parity against original GELU
with native rotary, builds native/context and runs model-free gate controls.
Only after this passes may one ordinary Q4 A/B enable GELU. Prior quality remains
negative; rejected ROTARY_AVX2 gate stays off.

GELU-only integration [42064278334](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42064278334)
at `9da1757cd3103180cfa809406452719e2368a87e` passes all 2,293,760 activation
bytes, old gate controls, and exact first-block reproduction of original GELU
with native rotary (32 differences remain, attributed to rotary). Native/context
builds pass, context 9 + decoder 20 controls pass, env docs current. Rejected
ROTARY_AVX2 stays explicitly 0 in the next ordinary Q4 A/B; decoder candidate 0.

Actual backend control is now confirmed: original CPU build has oneMKL 2024.2
(build 20240605), AVX2. Direct reference-only vmsCos/vmsSin HA mode calls on all
2,000 finite grid angles match Torch exactly, while installed SLEEF vector
headers do not. No MKL/Torch dependency is added to product. Retain this as a
rotary backend dependency, not permission to import closed-source math or a
reference lookup table. All exact source/config/control and artifact identities
are in `tests/results/qwen35-ocr/lightonocr3-4b-gelu-integration-mkl-control.json`.
Quality remains negative pending the authorized GELU-only composed Q4 A/B.

### 2026-11-05: verified GELU improves deep vision, decoded prefix still negative

Ordinary [42065352208](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42065352208)
at `40c68f35f81c8403eb22b27b642037df3fd34c8d` preserves all 30 fixed baseline
hashes with decoder and rejected rotary candidates explicitly 0. Candidate
first block has exactly the attributed 32 rotary differences, RMS 2.28155e-5,
max 0.0078125. Merged RMS improves 0.00973525 to 0.00775425 (max 1.21094 to
0.734375), also improving the prior exact-attention candidate RMS 0.01839050.
First-logit RMS is 0.56432682 against baseline 0.56013479 (max 2.48687 against
2.38528): better than prior candidate 0.59042975, still worse than baseline.
Both independent smokes remain wrong `<table>` / `[20353,29]` against original
`<figure>`. **Decoded quality remains negative and unqualified; defaults stay off.**

The test-only replay in that run still hardcodes legacy GELU, so its 14,029
activation differences are not a measurement of the gated production helper.
Production GELU is independently byte-exact in 42064278334; actual native
composed score/decoded paths use production and are unaffected. Fix the replay
observer to use the actual gated math for future runs, with no runtime change
and no duplicate model job. Full numerical/decoded/resource receipt and 90-day
artifact identity: `tests/results/qwen35-ocr/lightonocr3-4b-gelu-composed-negative.json`.
No further model jobs authorized in this lane. The earliest remaining first-
block mismatch is confirmed MKL rotary; another bounded operator-only option is
retained same-input merger GELU/erf attribution, which must not obscure that
backend dependency or claim decoded improvement.

### 2026-11-05: bounded retained same-input merger/erf attribution

Next one operator-only job uses retained original and actual layer 23/merged
arrays, plus only the 54.5 MB pinned original merger-parameter range. Require
complete original BF16 merger bytes and complete actual native merged bytes
before attribution; actual GGUF weights are not separately downloaded, so record
that parameter-compatibility limitation explicitly. Reconstruct only norm/up/
activation/down for the merger, not the vision model or decoder. Compare native
verbatim std::erf expression and installed original Gelu.h on identical up inputs.
A test-only source erf polynomial/exp alternative may be measured, but no
production change or quality claim before actual/source controls pass. Preserve
all reports/arrays remotely for 90 days; unrelated workflow retention unchanged.

Merger attribution [42069010152](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42069010152)
at `c398b710b34987b960ce54dddfb8feeed35a3b5d` passes complete original and
actual merged byte controls, norms/up and installed original Gelu.h. Native
std::erf differs at 33 original-input and 26 actual-input activation values
(max 1.52588e-5), propagating to 84/58 merged values (max 0.00048828125).
Test-only source FMA erf + verified exp matches all 573,440 activation and
358,400 merged values on both inputs exactly. A separate default-off
`CRISPEMBED_LIGHTON4B_VISION_BF16_MERGER_ERF_AVX2` now adapts only this measured
operator; unsupported platforms/rounding/nonfinite inputs retain the preceding
path, and existing std::erf path is preserved. Source BSD/SLEEF notices remain.
Focused source/actual/gate/guard/build verification pending. No full-model smoke
or decoded-quality claim; latest ordinary Q4 result remains negative.

Final focused merger integration [42069996673](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42069996673)
at `ca8e6dcade7fc85ec7adecb8dc904ee64c0db009` passes original and actual
complete merger controls. Gated production matches all 573,440 activations and
358,400 merged outputs on each input; unset/empty/0 preserve preceding bytes,
nonfinite/unsupported-rounding guards reject unchanged. Native/context builds,
context 9, decoder 20 and env docs pass. Final receipt pins runtime/artifact
identities, retained 90 days. Operator success still does not qualify quality.
One final authorized ordinary Q4 A/B now includes all verified vision corrections,
with rejected rotary explicitly 0 and decoder candidate 0, all 30 fixed baseline
hashes required, first scores/independent two-token decodes and native timings.

### 2026-11-05: final verified-vision Q4 result remains decoded-negative

Final ordinary [42070806605](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42070806605)
at `e7730d91e7b7b6f9bec5c59b33c612ddcc97254b` includes norm, true Linear,
Conv3D, attention, tanh GELU and measured merger-erf corrections. Rejected rotary
and decoder candidates are explicitly 0 in the wrapper. All 30 fixed baseline
hashes reproduce; corrected source-fed layer 0 has zero differences for all
eight observed operators, with the replay activation observer now corrected.
Ordinary inference is independent of those source-fed arrays.

Actual layer 0 retains the 32 attributed MKL-rotary differences (RMS 2.28155e-5,
max 0.0078125). Merged RMS improves baseline 0.0097352499 to 0.0077542505
(max 1.21094 to 0.734375). First-logit RMS is 0.5643236502 against baseline
0.5601347910 (max 2.48687124 against 2.38527584); it is only slightly better
than prior GELU candidate 0.5643268159. Both native outputs remain `[20353,29]`,
`<table>`, unlike the original `<figure` prefix. **Decoded quality remains
negative and unqualified. All correction gates stay off by default.**

Same-run native elapsed: baseline 105.17 s, candidate 115.84 s; user/system CPU
75.42/56.22 s versus 100.42/55.81 s; peak RSS 4,524,982,272 versus 4,510,666,752
bytes. These are one-case measurements, not a throughput benchmark. Two-token
smoke does not qualify EOS/full-document behavior. Actual merger parameters were
not independently fetched from GGUF, but complete original and actual merged
byte controls establish compatibility of the bounded original parameter range.

Full source/operator/stage/decoded/timing/limit receipt:
`tests/results/qwen35-ocr/lightonocr3-4b-merger-erf-composed-negative.json`.
Artifact `13436852207`, SHA `1165cfd00895e99840e939f79fc6aa00c730f3aa57316f132c70dec386f1e4e8`,
retained through 2027-02-03. No further work or runs in this lane; remaining
MKL rotary dependency is explicit and no source lookup table or new product
Torch/MKL dependency has been introduced. No release or app-pin change.

### 2026-11-05: complete model-arithmetic rotary profile proof, before integration

The next bounded lane generates configuration-defined rotary constants, not
captured outputs: complete coordinates 0..1023, all 16 frequencies, sine/cosine,
128 KiB raw. Native geometry caps h,w at 1024, so frames/multiple grids cannot
escape that domain. Generator enforces source config SHA, Transformer default
theta/head dimension, frozen Torch git/NumPy/oneMKL2024.2 AVX2 HA. No image,
activation or checkpoint is read. MIT generator/output and existing Apache2
config provenance, no vendor implementation or binary shipped.

Production is unchanged pending proof. Targeted job regenerates twice, checks
all constants against direct original MKL, compares compiled reader against all
3280 admitted single-frame h/w pairs and every total token count4..4096 plus
frame/multigrid boundaries, tests disabled/unsupported/fallback output retention,
and checks no native MKL/Torch dependency. Unsupported profile/CPU/rounding or
range must fall back. Scope and reproduction are in
`docs/lighton4b-rotary-arithmetic-profile.md`. No fixture fitting or quality claim.

Complete arithmetic profile [42096520795](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42096520795)
at `79fbff4e74ba084c35804ba96178a15d2e3e2da0` is proved: independent duplicate
generation, all 32,768 original MKL constants, compiled 128 KiB raw bytes, and
681,459,712 source-vs-reader F32 values across 3,873 layouts are bit-exact.
This includes all 3,280 admitted h/w pairs and all 1,024 total token counts plus
frame/multigrid boundaries. All 17 reader fallback controls pass; native binary
has no Torch/MKL dependency. No image, checkpoint or captured activation is read.
Raw SHA `30ea508c92d92a8400604e276fac161a1a85f7e89cd79877f227684a6d330e0c`.

The generated model-arithmetic bank and reader now enter production behind a
separate default-off ROTARY_TABLE gate. Exact profile/CPU/rounding/length/range
checks precede lookup; on rejection the original rotary generator remains.
This is a complete configuration-derived arithmetic asset with explicit MIT /
Apache2 configuration provenance, not a reference fixture or fitted image table.
Focused integration will regenerate committed data, repeat the whole domain
through the actual private production helper, verify new fallback/precedence
controls, source tables and first-block bytes, native/context builds and decoder
controls. No quality claim before actual composed inference is measured.

Production table integration [42098420346](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42098420346)
at `7648267c79453b277b6ae3f8c43aa54769903264` passes full committed-asset
regeneration and the actual private runtime helper: 681,459,712 F32 values across
3,873 layouts, all 3,280 h/w pairs and 1,024 token counts, remain bit-exact.
All 22 reader/production fallback and precedence controls pass; native reader
links no Torch/MKL. Retained source controls now match all original cosine/sine,
pooled attention and complete first-block bytes, closing the 32 rotary
mismatches. Preceding first-block bytes are preserved when table disabled.
Native/context builds, context 9, decoder 20 and env docs pass. Exact generator
manifest is updated from the actual qualified run, not manually recomputed data.

One source-pinned ordinary Q4 A/B now enables every verified vision gate plus
the complete rotary profile, rejected SLEEF rotary0 and decoder candidate0.
Require all 30 frozen baseline stage hashes, independent decoded outputs and
first-score magnitudes, plus both native timings. No quality/default/release
promotion from source/component parity alone; latest decoded result is negative.

Ordinary Q4 table run 42099925931 at a1bb8c924 passes all 30 fixed baseline
hashes; all 29 processor/vision stages have zero numerical differences through
layer 23 and merged output. Raw-byte equality is not inferred from numerical
zero: an independent report-only audit now verifies every archived shared float
stage against the pinned original manifest, baseline against frozen hashes, and
original two-token prefix separately from the full 177-token completion. It may
fetch only the earliest downstream source float array for absolute/RMS error.
No checkpoint load or inference is performed by that audit.

Q4 decoded prefix remains wrong `<table>`/[20353,29] versus source [27,17251],
and logit RMS 0.5879641585 remains worse than baseline 0.5601347910. Measured
native elapsed is 130.86 s baseline / 153.89 s candidate; detailed final receipt
follows raw audit. Correct future native observer metadata: `exact_original_ids`
means full-completion equality, whereas smoke needs `exact_original_prefix_ids`
and the compared source prefix. Earlier incorrect `<table>` conclusions remain
valid, but a two-token smoke cannot equal the full 177-token ID sequence.

### 2026-11-05: single-image ordinary vision raw parity, Q4 still decoded-negative

Q4 [42099925931](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42099925931)
at `a1bb8c924607e2631435a973512aeee94ea5f0be` closes every recorded ordinary
processor/vision stage through all 24 layers and merged output. Independent
report-only [42102908743](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42102908743)
verifies the complete retained ZIP, every one of 66 shared F32 files, all 30
frozen baseline hashes, and **29 processor/vision SHA identities** against the
pinned original manifest. This is one image, not whole-model/corpus qualification.
Five original-only vision intermediates (merger norm/up/activation and rotary
cos/sin) are not emitted by ordinary runner; separate operator controls cover
them. Position raw layouts differ (original chronology+THW, native THW), so that
raw inequality is not alone a position-math failure.

First downstream shared difference is text.embedded: 391,330/394,240 values,
max absolute 0.000380992889, RMS 6.30394e-5. Later decoder/quantization errors
are not attributed solely to embedding. First-logit RMS worsens baseline
0.5601347910 to 0.5879641585 (max 2.38528 to 2.44462), and both Q4 prefixes
remain wrong `<table>`/[20353,29] versus original [27,17251] (`<figure`). Full
completion has 177 tokens; the old `exact_original_ids` field compares that full
sequence and is not a smoke-prefix test. Independent prefix audit confirms the
Q4 mismatch and future metadata now distinguishes prefix from completion.

Same-run native elapsed baseline/candidate 130.86/153.89 s, user CPU
104.36/143.37 s, system CPU 64.92/68.73 s, peak RSS 4,524,945,408/4,510,699,520
bytes. This is not a throughput benchmark. Exact source, raw stages, first-score,
decode, timing and limitations are in the rotary-table Q4 and raw-audit receipts.
Defaults and quality remain unqualified, with no release/app-pin promotion.

### 2026-11-05: one authorized corrected Q8 smoke

Q8 immutable file is 5,815,503,232 bytes; prior observed native peak is
6,317,084,672 bytes. Dedicated new smoke requires at least 10 GiB actual available
RAM and 12 GiB free scratch, four threads and caches off. It loads only Q8 once,
checks full model SHA, canonical original prepared-input hashes, all 29 raw
vision stages, original first-row scores, two-token prefix and timings. Historical
Q8 source `891d07da39e9c361a206f2ec4b40fe8339712e0f` already matched `<figure`
prefix; its scores are compared with immutable report/ZIP hashes, explicitly
not a same-source A/B or table-only effect. No 177-token run or broad CI.

### 2026-11-05: corrected Q8 first-score improvement, two-token limit explicit

Single corrected [Q8 42104609427](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42104609427)
at `f120d61a97a82acc7d3e4a2f2e02a5a1354e5ee4` passes the complete immutable
5,815,503,232-byte model SHA, 10 GiB available-RAM / 12 GiB disk guards, all five
canonical original prepared-input hashes, and all 29 recorded processor/vision
raw SHA identities. All verified vision gates and complete table are enabled;
rejected SLEEF rotary and decoder candidate remain 0, caches off, four threads.

Against original BF16 first-row logits, corrected Q8 RMS is 0.0742289135 and
max absolute error 0.460906029. Compare both historical same-model/same-image
receipts: original F32-vision `fd150b98f` smoke RMS approximately 0.13049504
(published 0.130495042; pinned float64 stage review 0.1304950347489), max
0.568259001; earlier BF16-vision `891d07da` RMS 0.2005369525, max 1.092198968.
These are different runtime sources and hosted workers, not a same-source A/B,
a table-only causal claim or a speed benchmark. All three Q8 runs already match
the original two-token prefix `[27,17251]`, `<figure`; there is no new-prefix
win. No full 177-token completion, EOS or full-document qualification is claimed.
All 248,320 current logits still differ from BF16 original; downstream residual
remains, and Q4 still generates the incorrect `<table>` prefix.

Current Q8 native elapsed 139.13 s, user/system CPU 134.01/57.80 s, peak RSS
6,317,207,552 bytes. Historical fd150 elapsed 127.69 s, peak 6,316,945,408;
891d elapsed 105.43 s, peak 6,317,084,672. Timings are diagnostic across sources/
hosts. Full source, original and both historical score comparisons, prompt/vision
controls, prefix/completion distinction, capacity and timing are retained in
`tests/results/qwen35-ocr/lightonocr3-4b-rotary-table-q8-smoke.json`. Artifact
`13450873762`, SHA `9a16e85ba59ca254a2d2d509cf21f0864fe7da4911b49c8312d6afdd2e461c09`,
retained through 2027-02-03. No default, release or app-pin promotion follows.

### Corrected native mixed form diagnostic

One hosted first-score/two-token smoke uses the immutable mixed candidate from
[42006450732](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42006450732),
the complete rotary table and all verified vision gates. It authenticates the
original BF16 prefill and independently captured BF16 mixed-recipe first score,
and requires all29 vision stages and five prepared inputs to match the original
bytes. This observed form is excluded from calibration/heldout qualification;
two prefix IDs cannot establish document completion or EOS parity. Decoder BF16
rounding, direct quantization and rejected rotary alternatives remain off.

The scoped [corrected mixed smoke42107457845](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42107457845)
completed at9e24c7c16. All29 vision stages and five prepared inputs are original-exact.
Independent IDs27,17251 decode `<figure`, matching the first2 original IDs.
First-score versus original BF16: max1.48261547, RMS0.284899447, cosine0.992588001;
versus independently captured BF16 mixed recipe: max0.163356781, RMS0.0274295614,
cosine0.999928425. Both strict numerical comparisons fail. Native time141.19s,
peak RSS5,179,695,104B. This separates the remaining runtime residual from recipe
loss without establishing a full decoded result; no EOS/full/heldout claim.
[Complete report and receipt](tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-form/native-report.json)
preserve exact model/table/profile/reference hashes and active/disabled gates.


### Corrected mixed form full177 comparison — prepared, waiting for native output

Native job42109519607/source189fc91bb is still in progress. One bounded
read-only comparator is prepared against the unchanged frozen177-row original
BF16 protocol and actual crispembed-diff target. It requires independently
reviewed complete artifact/producer/normalized-receipt hashes, fixed mixed model
provenance, all5 input and29 raw vision controls, all177 native row hashes and
independent full greedy IDs/text/EOS. The normalized receipt is explicitly
comparer-derived; no mixed-to-Q4 relabeling or frozen protocol/budget change.
[Exact admission, resource bounds and dispatch prerequisites](docs/lighton4b-corrected-mixed-full-comparison.md).
Four new admission controls and existing model-free controls pass; actual Ref
controls run inside the single hosted comparison. No dispatch before native
completion; observed form remains exploratory, not held-out quality selection.
No native/original inference, model upload, runtime/default/release change or newPR.


### Corrected mixed full form — complete numerical and decoded rejection

Native42109519607/source189fc91bb completes177 teacher rows and independent
greedy156 tokens/EOS248046, versus frozen BF16177 tokens. First differing ID is
zero-based25. Single read-only comparison42118693149/source957eb9513 authenticates
complete native/archive/producer/normalized receipt and original BF16 source
identities; all5 inputs and29 vision raw-byte controls pass. All177 rows are finite
and all177 fail unchanged0.002/0.99999 gates. First row max1.48261547/RMS0.284899443/
cos0.992587984; worst row5 max10.1740723/RMS1.51052773/cos0.763862431. Earliest
observed score difference and budget failure remain0. Exact independent IDs/text
reject despiteEOS; raw BF16-agreement CER197/656=30.03049%, WER40/90=44.44444%
(case/punctuation retained, Unicode whitespace words), not human-ground-truth OCR
accuracy. [Complete receipts,177 metrics, texts, artifact identities and limits](docs/lighton4b-corrected-mixed-full-results.md).
The comparer performs no inference and passes actual Ref equal/change/tamper
controls; its red status is complete numerical/decoded rejection, not harness
failure. Both artifacts remain remote90days. Existing observed form is not heldout
or architecture-quality selection. First-position genuine-BF16 mixed oracle
cannot partition full177-row/decoded errors into quantization versus runtime.
No additional jobs, runtime/default/release/app-pin change, newPR or model upload.

## 6 November: four-case vision generalization closed within measured scope

[Actual source/native vision42166765060](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42166765060)
at`6838c8106` passes all34 retained form source-control hashes and all136 native
processor/vision stage hash pairs across frozen table, Arabic, receipt and
letter. All maximum/RMS errors0; actual rotary arrays exact for all four grids.
No vision divergence or runtime correction was found; no decoder/generation ran.
[Evidence and limitations](docs/lighton4b-vision-generalization.md) preserve the
initial dependency-only failure, immutable source pins, original receipt cap,
historical CPU-dispatch uncertainty and90-day remote arrays. Next correctness
attribution belongs at the downstream native text/decoder boundary; full decoded
quality remains open and all new gates remain default-off. No new PR or release.


### Genuine BF16 mixed full generation — complete, native residual confirmed

Source89eda20c6 completed one CPU worker:178-token genuine BF16 mixed EOS versus
native156/EOS and official177/EOS. Source/native share33IDs, diverge at33. With
genuine BF16 mixed as reference, nativeCER134/675=19.85185%,WER30/95=31.57895%;
reverse-denominator metrics labeled separately. Source mixed still differs from
official BF16 at25 (CER27.43902%,WER44.44444%). All178 finite score hashes retained
remotely; all249 effective mixed/723 original parameter hashes,474 protected/tied
head, both score anchors and actual179 totalroot calls pass.17 hosted controls
pass distinct6+6+5 paths; benignwrapt startup warning and historical duplicate
mixed-test coverage are preserved. [Complete evidence and precision limits](docs/lighton4b-bf16-mixed-freegen-results.md).
Genuine BF16 rounding differs from native F32-dequant storage, so residual alone
is not an isolated logic defect. Next boundedsource replay captures actualshared-
prefix step33 operators/all32hidden/headinput, checking all34scores against this
retained capture; runtimeagent independently probes finalhead BF16 rounding.
No F32 model, gold rewrite, heldout qualification, default/release/app-pin or upload.

The next single scoped corrected mixed form capture preserves177 teacher-score
rows plus independent greedy output up to256 tokens, using the same immutable
model, original BF16 source and independent mixed BF16 first row. All prepared
input/29 vision-stage controls remain mandatory. Expected CPU duration60–100min;
180min cap, retained failure diagnostics, no new PR or broad CI. The first-row
comparison does not substitute for the subsequent complete177-row replay.

### Bounded parallel quantized conversion candidate

A separate default-off candidate schedules existing Q8_0/Q4_K dequantization
on complete block boundaries, retaining the original converters, F32 matmul,
quant admission gates and bounded transient allocations. First acceptance is
48 model-free serial/parallel byte-equality cases, uneven partitions, boundary
sentinels and explicit-off/type/size/quant-gate controls on hosted CPU. No full
model F32 cache or decoded-quality claim. Only after these pass may one bounded
actual same-input A/B measure timing and score/decoded equality. Historical
run sources, receipts and the rejected direct-quant branch remain unchanged.

Focused [42164147108](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42164147108)
passes all48 conversion byte-equality cases at35cf71ec1. Next single bounded
native A/B uses one build/model/prepared input, serial first then parallel,
1 teacher row and2 independent greedy tokens per arm; every emitted artifact
must be byte-identical. Serial-first cache warmth is explicitly uncontrolled,
so wall ratios are diagnostic, not a clean isolated speed claim. No full decoded
quality inference follows from a prefix or conversion equality.

The bounded [actual A/B42164504290](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42164504290)
passes at37beb2e4a: all76 emitted files are byte-identical between conversion
gate0/1, including decoder stages, scores and independent `<figure` prefix.
Serial177.97s/5,179,629,568B RSS versus parallel130.42s/5,176,004,608B RSS;
wall ratio1.3646 is diagnostic because serial-first cache warmth is uncontrolled.
User CPU165.08→169.41s and system75.8→95.0s increase. Original BF16 score remains
strictly failing (RMS0.284899447); no full decoded/quality qualification. Candidate
remains separate/default-off; no additional job. Full receipts preserve exact
model/source/table/reference pins in
[the A/B report](tests/results/qwen35-ocr/lightonocr3-4b-parallel-dequant/native-ab.json).

The next single controlled worker runs conversion gate order0,1,1,0 (two
opposite-order pairs), sharing one immutable model/build/prepared input and
requiring every emitted artifact to remain byte-identical. Per-arm GNU time,
RSS and repeated-arm load/cgroup memory are retained. Warm reverse-pair timing
is reported separately from the coldest first arm. Default-off integration
waits for these checks; no additional matrix or full-document rerun.

Balanced [42165624595](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42165624595)
passes atbf10b06d2: serial/parallel/parallel/serial wall111.21/82.38/84.06/116.36s,
with all76 emitted stage/score/ID/text files byte-identical across all4 arms.
Warm reverse pair116.36/84.06 yields1.38425 ratio; mean ratio1.36728. RSS remains
about5.18GB, while parallel user/systemCPU increase. This supports the opt-in
conversion scheduling on this worker; no full-document/general speed or quality
qualification. Only1 teacher row and2 IDs per arm; original numerical failure
is unchanged. The source-pin amendment preserves all historical receipt bytes,
and the nonpthreadWASM path explicitly excludes thread launch. Default stays off.
[Full balanced receipt](tests/results/qwen35-ocr/lightonocr3-4b-parallel-dequant/balanced-ab.json).


### Mixed step33 precision attribution — complete, decoded correction verified

Original source8eeb captures50actual stages; all34 IDs/score hashes remain exact
against178-row mixedBF16 source. Native headRNE AB42168142185/b988 preserves229
OFF artifacts/156IDs, whileON reproduces all178 source IDs/text/EOS. All47
nonhead step33 arrays are byte-identical. Read-only Ref42174566100/5222 completes
96raw stage comparisons plus26 same-causal-context official/mixed/native triples.
Embedding max0.000105858 becomes source-bit-exact withBF16RNE; firstbudgetfailure
is input_norm max0.0165167/RMS0.00256818. Later raw numerical debt remains: headON
max0.125/RMS0.0302963 and all178 scoreSHA differ despite exactdecodedoutput.
Actualhead401/1778 tie is18.5/18.5; independentRNE of all248,320OFF values exactly
equalsON vector. Row5 official→mixedBF16 max10.578125/RMS1.54315937 confirms
large quant-model loss without native implementation; mixed→native residual
max0.656324/RMS0.0785952. [All stages, triples, controls and source versions](docs/lighton4b-mixed-step33-results.md).
Artifact13483410105 preserves178source rows/all rawpaired stages/26official and
native rows90days until2027-02-04. Complete numerical failure is expected, no
setupfailure, inference or F32model in consumer. Runtimeagent owns gated correction;
no defaults/quality qualification. Official177 output remains different, observed
form remains excluded fromheldout/architecture selection. No additional jobs.

### Mixed full generation first differing token33

Authenticated original-BF16 mixed and native mixed have identical33-token
independent prefix. Source row33 has an exact top tie: IDs401/1778 both18.5;
Torch argmax selects lower401. Native F32 logits are18.551120758/18.558383942,
selecting1778 by0.007263184. Offline BF16 RNE maps both to18.5 and restores401.
Row32 still picks68100; row33 RMS0.0299801/max0.136086/cosine0.9999511. This
identifies an output dtype boundary sufficient for the first token difference,
without attributing all residuals to it or claiming later/full parity.

Next single hosted same-worker experiment tests HEAD_BF16 default-off against
on with full independent greedy cap256 (one teacher row each), preserving exact
original178-row mixed receipt and prior native156-ID EOS baseline. Model-free
RNE/special-value/lowest-ID/source-row tests must pass before model acquisition.
Selected step33 cached hidden/operator probes are test-only and their recomputed
layer0 hidden must match actual forward bytes. No source weights, prompt, EOS,
protocol budgets or existing gates change. Expected45–70min total,180min cap.

### Full mixed runtime head-output dtype A/B

[42168142185](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42168142185)
completes atb98883abe. Optional HEAD_BF16 output rounding reproduces all178
original-BF16 **mixed-recipe** IDs, EOS and686-byte text exactly (SHA c9075ce2…).
Gate-off reproduces prior156-ID EOS output, divergence33, and229 shared
stage/score hashes. All5 prepared inputs/29 vision stages remain exact;
66 prefill artifacts and47 selectedstep33 hidden/operator arrays are unchanged
across arms. RNE/NaN/Inf/lowest-ID/source-tie and explicit-off controls pass
before model acquisition. Full raw report and hashes are retained in
[the receipt](tests/results/qwen35-ocr/lightonocr3-4b-head-bf16/native-ab.json).

This establishes decoded runtime fidelity for the observed mixed form, not
unquantized-original or heldout quality. All178 score hashes still differ
from the BF16 mixed source; actual row33 max0.125/RMS0.0302963/cosine0.999949698
fails the unchanged strict budget. Against the unquantized BF16 original,
firstID difference25 persists, raw CER improves197/656→180/656 (30.03→27.44%),
and WER remains40/90 (44.44%). No full178 numerical acceptance claim.

OFF156tokens takes1388.25s/RSS5,185,196,032B; ON178tokens1561.55s/5,185,175,552B.
Different completion lengths prevent a speed interpretation. Gate stays
default-off, no release/default/model alias change. Existing historical
source-pin amendments and15git-verified frozen receipt pins remain intact.

The [pinned upstream blueprint audit](docs/lighton4b-upstream-blueprint-audit.md)
compares llama.cpp`bbc2e4b65705132843b9d142f4471dfb8085b233` and official
Qwen3.5-4B geometry without inference. It finds no demonstrated missing native
GDN/attention operation. Upstream's tiled V-head storage permutation and
pretransformed A/RMS parameters differ from our HF-order runtime contract; they
must not be copied selectively. All4096 lane indices pass the integer layout
control. UpstreamQ4_K_M protects selected FFN-down/V tensors; our frozen
mixersQ8/FFNQ4 is a different policy. Any next recipe needs separate calibration,
actual importance-vector coverage and heldout decoded/EOS evidence after
same-weight arithmetic controls. No runtime or recipe change, CI or release.

## 6 November: bounded Q8 restoration sensitivity measured

[Five-arm calibration42185018279](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42185018279)
at`67f47d435` completes table/Arabic only; heldout receipt/letter untouched.
Exact donor groups0/152/32/96/248 and payload hashes were frozen before forward;
all248 loaded base/donor pairs and475 protected tensors pass. Table first-score
RMS Q4/mixers/down/FFN/fullQ8 is0.706260/0.654575/0.650900/0.632844/0.0712989;
Arabic0.364631/0.293539/0.325577/0.291611/0.0394947. Mixer/fullQ8 fix table's
observed two-token`<figure` prefix; down/FFN do not. All strict score gates still
fail and no full decoded/EOS qualification follows. First recorded decoder
response is block0; mixer block0 max/RMS0.155319/0.00558722(table),
0.110185/0.00433838(Arabic), growing through32 blocks. These are changes versus
nativeQ4, not original internal stages or operator attribution.
[Complete evidence](docs/lighton4b-restoration-sensitivity.md) preserves timing,
firstscore/block hashes,predeclared protocol and90-day remote arrays. FullQ8
reduces RMS about90%, while selectiveFFN restoration alone is insufficient on
table; further policies require separate calibration/source controls. Current
owner has noQ6 support. No production admission/runtime/default/release change.

### Corrected plain Q4 remains capped and unqualified

[Native42183620520](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42183620520)
completes execution at444386169 using the immutable plainQ4f873eb3d… payload,
all verified vision gates and BF16 head output (mixed gate0). Five prepared
inputs and29 vision arrays match the frozen original exactly; head RNE/gate
controls pass. The independent generation still hits256 tokens without EOS,
ending in incomplete HTML `<td`. All256 IDs and581-byte text (SHA9b2b5352…)
are identical to historical native42001333207/fd150b98, before vision/head fixes.
This is a preserved quality failure, not a fixed plainQ4 result.

The authenticated **same-weight BF16 first row** (048fd683f/03119ea4…/e59d6054…)
already selects20353 `<table`, matching native and differing from unquantized
original27 `<figure`. Original figure/table margin+2.125 becomes−1.0 under Q4
in the genuine original code: the first layout regression is weight-induced.
Corrected native versus same-Q4 BF16 firstrow max0.15625/RMS0.0267200/
cosine0.999943264 still fails strict tolerance. Versus unquantized original,
max2.453125/RMS0.587978416/cosine0.976136772 fail; decoded rawCER543/656
(82.774%) andWER89/90(98.889%) remain unchanged.

Native time2618.62s, peakRSS4,526,112,768B. One teacher row and all256 independent
greedy rows are retained remotely, with complete hashes in
[the native receipt](tests/results/qwen35-ocr/lightonocr3-4b-corrected-plain-q4/native-report.json).
New full same-weight BF16 source attempts rejected an originala4→02ca baseline
drift before quantization despite exact723 original tensors/five inputs;
dual-vendor CPU-profile diagnostics are separate and never replace the frozen
a4/e59 anchors. No full same-weight reference admission, EOS/document, heldout,
strict score, default or release qualification follows.

## 6 November: mixer architecture partition measured

[Six-arm42187953019](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42187953019)
finds table prefix fixed by Q8GDN120,not full-attention32 alone. Table first-score
RMS baseline/GDN/attention/mixers/mixers+down/fullQ8 is
0.706260/0.642260/0.664900/0.654575/0.558216/0.0712989;
Arabic0.364631/0.336366/0.327429/0.293539/0.256652/0.0394947. All12strict gates
still fail; two-token prefixes only,heldout untouched. Attention-only preserves
blocks0–2 bytes and first responds at3; GDN responds at0. Repeated controls versus
first panel pass198/198 fullscore/layer hashes. All248 donor pairs/475protected
hashes and5prepared-input hashes/case pass. [Evidence](docs/lighton4b-mixer-partition-sensitivity.md)
retains raw hashes,timing,stage magnitudes and the disjoint projection classes of
GDN120: QKV24,Z24,a+b48,out24. Next calibration question can split these classes
with high-precision controls; no production/model/default/release promotion.


### Unchanged Q4 full original/native mechanism comparison

The [authenticated full Q4 comparison](docs/lighton4b-plain-q4-full-results.md)
uses sourcebd8759304 with exact frozen Intel originala4/Q4e59 anchors and all
723/249/474 hashes. Genuine BF16 Q4 independently generates256 tokens/noEOS and
an incomplete HTML table, reproducing the main format/cap change inside the
original framework. Corrected native42183620520 matches171 IDs, then differs;
raw native agreement against genuine Q4 is CER110/583 and WER17/48. This is not
absolute OCR accuracy, EOS completion or a quality promotion. Existing row171
vectors show source27287=20.5 versus1058=20.375, while native ties both20.5 and
correctly selects1058. RMS0.03414869 remains; no hidden/operator arrays at171
exist, and F32-dequant versus BF16-rounded parameter representations remain
separate. All three source-profile failures and the measured AMD/Intel split are
retained. Next is complete read-only archival and, only if authorized, one
observer-neutral same-prefix172-token stage replay; no new inference is dispatched.


The read-only [Q4 archival consumer42192667516](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42192667516)
finishes with expected numerical failure and reportstatus`compared`: actualRef
controls pass, all172 shared-causal rows fail retained budgets, and all256 source
rows plus172 native rows are remotely preserved90days in artifact13491795001.
No inference runs. RMS is nonmonotonic: row0=.02672002, row125=.49875835
(max3.03125, cosine.98704768, samewinningID), row171=.03414869
(max.203125, firstIDdifference). These are retained numerical negatives, not
uniformly small residuals. Future observer-neutral attribution should capture125
and171 with actual inputs/weights; no hidden/operator arrays exist at either row.

## 6 November: GDN projection-class screen measured

[Eight-arm42191354208](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42191354208)
completes table/Arabic calibration only. Table first-score RMS Q4/QKV/Z/a+b/out/
GDN/mixers+down/fullQ8 is0.706260/0.668973/0.698002/0.707848/0.616934/
0.642260/0.558216/0.0712989; Arabic0.364631/0.348032/0.357053/0.364840/
0.352079/0.336366/0.256652/0.0394947. QKV/Z/out restore table's two-tokenprefix;
a+b does not and slightly worsens both RMS values. Output-only24 adds125829120
bytes and is best table singleclass,while QKV is best Arabic singleclass.
All16strict original score gates fail; heldout/fullEOS untested. Controls248donor/
475protected plus5inputs/case pass; repeated264/264 score/block hashes match
prior panel. [Evidence](docs/lighton4b-gdn-projection-sensitivity.md) retains
all32 stages,magnitudes,hashes,timing and90-day remote arrays. A separately named,
fixed Q4_K+24GDNoutputQ8 exploratory recipe can next receive complete calibration
output comparison; it is notQ4_K_M and does not qualify a model/default/release.
### Optional dequantized text weight dtype isolation

A separate default-off BF16_QUANT_WEIGHTS experiment casts only admitted named
Q8_0/Q4_K text weights to BF16 then back to F32 in existing transient buffers.
It covers ordinary matrices, gathered embedding rows and tied head tiles;
protected F32 vision/norm/conv and unrelated types/names remain untouched.
No quantized storage change or full-model F32 cache. Full4B decoder caches
record the mode and reject toggles between cached steps. First hosted gate
is checkpoint-free projection/embedding/streamed-head byte equivalence to
independently rounded F32 matrices, explicit-off/storage controls and genuine
positive cache-toggle controls with the existing shared synthetic decoder.
Historical synthetic numerical rejection remains separate, unchanged and
insufficient for checkpoint qualification. Source-fed actual weight hashes
and operator inputs are next; no full generation or speed claim yet.

Focused [42196101066](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42196101066)
passes16 dequantized-weight/operator controls and353 decoder controls including
valid cache-mode toggle rejection at7ff9d2ded. The separate synthetic F32
numerical comparison still rejects prefill.cache.21.recurrent, with its full
original receipt unchanged/preserved; this is not waived or required to qualify
the optional storage-dtype mechanism. No checkpoint or full-model storage.

Next one hosted source-fed test streams all249 actual plainQ4 tensors in
≤65536-element chunks, hashes raw and BF16 values against the authenticated
03119ea4 oracle, and feeds the admitted5aa5a802 step33 embedding/head/FFN inputs
through OFF/ON/OFF-repeat operators. Selected sharedQ8 embedding/head and Q4FFN
source weight hashes must agree across oracle/trace. Exact source embedding and
gate-off repeat are required; all projection numerical residuals are reported
without threshold waivers. No autoregressive generation or speed/quality claim.

The separate plainQ4 full producer444386169/42183620520 is now retained with
this controller: corrected vision+head still yields the identical256-ID capped
table as historical42001333207, noEOS. Its unchanged reportc726b09d… and genuine
BF16 plainQ4 full reference1096b5f0… at4939c9a6e70a6651fa6072aab71665a30db0c71e
remain separately frozen. First171 native/source IDs agree before a BF16 tie
residual; no unquantized-original quality or document/EOS qualification.

### BF16 dequantized text-weight isolation

Checkpoint-free [42196101066](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42196101066)
passes16 projection/embedding/head/gate/storage checks and353 decoder controls,
including valid cached-mode toggle rejection. The separate synthetic original
numerical rejection remains preserved atprefill.cache.21.recurrent; no waiver.
Actual source-fed [42198497821](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42198497821)
atbe9eb625b verifies all249 raw quantized and BF16 weight hashes against the
authenticated original oracle, in≤65536-element chunks, including tied embedding/
head. Source-fed ON embedding/up/down arrays are byte-exact; gate differs1/9216
(max0.0004883), head24/248320 (max0.03125/RMS0.00007093, still strict failure).
OFF-repeat raw bytes are identical. Time38.49s/peakRSS2,928,726,016B. These fixed
source inputs isolate weight dtype; no generation, speed or quality claim.
[Full receipts and hash proof](tests/results/qwen35-ocr/lightonocr3-4b-bf16-weight-precision/source-operators.json).

Next one real same-Q4 image OFF/ON smoke uses BF16 head and verified vision,
1 teacher score plus independent16-token cap per arm. OFF must reproduce all
shared frozen correctedQ4 stage/score bytes and16IDs; prepared/vision/positions
stay exact across arms, ON embedding must equal independent BF16 RNE of OFF.
The new native-only weight option stays off on WASM; a standalone simulated
preprocessor guard runs before model staging, explicitly not a WASM runtime test.
Existing source pin chain and cache controls remain intact. No full Q4 quality
claim follows from this bounded smoke; integration waits for its checks.

### Real plainQ4 weight-dtype smoke: partial negative

[42200784414](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42200784414)
completes at4bd327394 with BF16_QUANT_WEIGHTS OFF/ON. OFF reproduces87 frozen
correctedQ4 shared stage/score hashes;34 prepared/vision/position artifacts
remain identical across arms, and ON embedding equals independent BF16 RNE.
Both produce the same16 IDs, matching the genuine same-Q4 source prefix,
ending in partial `<table>…<th` without EOS. No full-document test. The
standalone simulatedWASM preprocessor guard passes and disables this native
oracle option there; no actualWASM runtime qualification.

Same-weight BF16 firstscore RMS changes0.0267200→0.0257732, but max error
worsens0.15625→0.1796875; both fail unchanged strict tolerance. Against the
unquantized original, ON RMS0.5882684/max2.46875 also fail. This option is not
a plainQ4 quality fix. OFF takes195.46s, ON417.00s (observed cost, not a speed
benchmark; serial-first/cachewarmth uncontrolled), peakRSS4.525/4.522GB.
The gate stays default-off, with immutable quant storage, bounded transient
weight/row/tile conversion and cached-mode toggle rejection. All historical
source pins/negative evidence remain unchanged.
[Full measured receipt](tests/results/qwen35-ocr/lightonocr3-4b-bf16-weight-precision/native-smoke.json).

## 6 November: named24GDNoutputQ8 full calibration improves but remains negative

[Actual four-run42199032449](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42199032449)
at26e6ae663 finishes EOS for both Q4/candidate on table/Arabic. Candidate table
CER441→224edits/607(72.65%→36.90%),WER99→61/106(93.40%→57.55%),matchingprefix0→36;
212→228tokens versus222original. Both Arabic outputs exactly match6original
IDs/text/EOS. Strict firstscore budgets still fail; table remains wrong. Verified
file4156787616bytes SHA40e54560ed349c077100c4e39276d29c02b98d6bec3611ac7b3a88d5a771c710
keeps699payloads and replaces24wholeout matrices. All723 producer hashes,112recipe
controls,48parallel cases and132serializedsmoke SHA pairs pass. Independent raw
ID/text and character/word edit audits reproduce metrics.
[Evidence](docs/lighton4b-gdn-out-full-calibration.md) records complete decoded
outputs,source/protocol pins,RSS/time and remote90-day candidate/arrays. No broader
qualification or default/release follows calibration. Next one bounded job must
first independently rehash whole prior archive/candidate/all723tensor payloads,
then unchanged candidate versus plain baseline on heldout legacyhandwritten_letter,
which is actually an A/logo65-tokenEOS source,not representative paragraph OCR.
Receipt stays deferred/capped. Coordinate latest PR1554850481ba before integration;
current model measurement remains frozen26e6ae663.


### 6 November2026 — unchanged GDN-out recipe heldout diagnostic completed

[42209944112](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42209944112)
at0effa01fd5 freezes calibration engine26e6 and verifies the complete prior
archive/model,723 payloads,1004 F32 arrays and four fresh calibration closures
before the predeclared A/logo case. Both Q4 and unchanged24-out-Q8 candidate
produce `A`/EOS in2 IDs; original BF16 has65 IDs and a figure description.
RMS improves0.643224→0.597031 but decoded agreement does not improve; both
CER215/216 and WER37/38. Native105.98/105.96 seconds, RSS4,523,798,528/4,647,583,744.
[Full evidence](docs/lighton4b-gdn-out-logo-heldout.md) preserves independent
source/raw-text metrics and closes prior whole-archive rehash TODO. Legacy
`handwritten_letter` is a red A/logo, not paragraph OCR. Receipt remains untouched
and originally capped; no promotion. Any next candidate/heldout or integration
requires separate coordination and preservation of the frozen source/gates.


### Actual original-BF16 selective Q6 candidate: measured calibration only

[Q6 conversion/screen42212846887](docs/lighton4b-q6-candidate-screen.md) atc3d6b5f88
builds4,195,863,232-byte candidate76911f05… by direct original-BF16 quantization of
32 matrices and691 byte-exact copies, with Q8 tied head and full723/native/Python
hash audits. Old plain first-score controls match exactly. Table RMS.70626033→
.70573124 barely changes while maxerror4.22656→4.35938 and original-token
probability.40558→.27142 worsen; Arabic RMS.36463124→.31669830 and probability
.87177→.90418 improve. First tokens are unchanged; all4 cap1 withoutEOS and strict
Ref comparisons fail. No complete decoded/heldout quality or default promotion.
Initial Q6 loader failure and713-pass/converter proofs are retained. Candidate and
raw native first-score/prefill data are90-day remote artifact13500128433; no hub
upload or further model job. Current decision is whether this mixed calibration
screen justifies any later evaluation, separately from the held-out lane.


### Fixed24 GDN-output Q6 screen: first table token restored, full decode pending

[Actual42219710049](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42219710049)
at10d3fad88 converts24 matrices directly from original BF16, copies699 payloads
exactly including the Q8 tied head, and passes both immutable Q4/Q8-output score
controls on table/Arabic. [Complete evidence](docs/lighton4b-gdn-out-q6-first-score.md)
preserves all723 audits,700 new/713 old Q6/112 Q8-output model-free controls and
six actual strict-score failures. Table first token changes20353→27, matching
original; RMS .706260→.604549 (Q8-output .616934), original-ID probability
.405578→.544192. Arabic keeps67321 but original-ID probability worsens
.871767→.840039; RMS .364631→.359155. Every arm is cap1/noEOS, so the complete
table is not yet proven fixed. All protected vision/input hashes match; first
retained difference is decoder layer0, with actual magnitudes/errors preserved.
Candidate83e065c4…/4,095,839,136 bytes and raw arrays remain remote90days in
artifact13503268909. No heldout, F32 model, new reference forward, model-hub upload,
default or release change. Next: review one bounded full-generation calibration
worker for this exact candidate against frozen original/Q4/Q8-output decodes.


### Exact24-Q6 full calibration: equal Q8 decodes, residual original disagreement

[Full42223984643](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42223984643)
at30c2332da passes the700 controls and both smoke/full70-stage/prefill/argmax/EOS
barriers. [Complete evidence](docs/lighton4b-gdn-out-q6-full-calibration.md) records
Q6-output24 exactly matching retained Q8-output24 IDs/text/EOS on table228 IDs and
Arabic6 IDs, both under cap256. Arabic is exact genuine BF16. Table remains
non-exact at ID36: CER224/607=.369028, WER61/106=.575472 versus genuine BF16,
compared with plainQ4 .726524/.933962. All234 Q6/Q8 raw score hashes differ;
six strict first-score negatives remain. This is decoded agreement, not numerical
parity or human OCR accuracy. Candidate83e065c4… saves60,948,480 bytes versus Q8;
full raw scores remain90days in artifact13506428243. Q6 observed table/Arabic
seconds1085.18/76.67 and peak RSS4.526/4.535GB are retained; older Q8 workers are
not a controlled speed comparison. PR155 head90d2 was reverified; policies stay
default-off. No heldout, release/model alias, new reference forward, full F32
model, cap waiver or further model job. Remaining original-table disagreement
and Arabic probability regression remain explicit limits.


### Fixed-Q6 observed-logo diagnostic: decoded disagreement unchanged

[Post-selection42230729575](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42230729575)
atc7e6720ab preserves the frozen24-output83e065 candidate. [Full negative evidence](docs/lighton4b-q6-logo-post-selection.md)
records Q6 returning exactly prior Q4/Q8 `A`, IDs32/248046, primary EOS/no cap,
versus the65-ID BF16 caption: CER215/216=.995370, WER37/38=.973684.700 controls,
70-hash table anchor, actual154-token/560-patch input admission and36 protected
logo stage hashes pass. Q6 first-score RMS.593816 (Q8.597031/Q4.643224) and
original-ID probability.091132 do not repair output. Layer0 Q6/Q8 RMS.001250457,
max.01825738 and magnitudes are retained, with163.11s/4.585GB peak RSS. Raw arrays
remain90days in artifact13510375701. This case was already observed: explicitly
post-selection, not clean heldout or representative handwritten-letter OCR.
No recipe change, quality/default/release promotion, F32 model or new reference
forward. Next bounded work is source/tool integration for reproducible BF16→Q6
conversion and the tested closed harnesses in existing PR155; evidence stays PR157.

### Default-off exact selective Q6 LightOn4B candidate

User-requested Q6 support is prepared separately from the running calibration
source. [The candidate](docs/lighton4b-selective-q6.md) gates Q6 ownership and
native admission behind`CRISPEMBED_LIGHTON4B_Q6_K`, an exact custom marker,
UINT32 M-header15 and full32Q6/216Q4/1Q8 layout. The tied Q8 output and474F32
protected tensors remain explicit; old recipes stay unchanged. Selection follows
pinned llama.cpp value-family/FFN-down higher-bit layers, with the Q8 tied-head
exception clearly labeled. Source must original BF16 or unquantized F16, never
Q8→Q6 requantization. Model-free controls cover independent Q6 scalar decoding,
actual projection/BF16 conversion and header/type negatives; one bounded hosted
model-free build is next. No model job or quality claim before calibration review.


Initial Q6 model-free42206305308 compiles all targets but fails control703:
actual project_rows_cpu still rejects Q6 while full projection/dequant controls
pass. The negative is retained. Both streamed projection and grouped-GDN type
validators now gain the same default-off native Q6 admission, with explicit
stream and synthetic grouped prefill/cached-decode equivalence tests. One bounded
model-free retry follows; no checkpoint or calibration job is started.


Q6 structural/operator controls now pass713 in42207472045, with old native/mixed/
BF16 controls and native/preflight builds retained. Initial negative42206305308
remains preserved. Guarded Q6 support is ready for scoped PR155 integration;
actual candidate conversion/calibration remains pending the current recipe review.
No Q6 model job has run.


### Integrated default-off24-output Q6 and verified Q8 native admission

PR155 selectively incorporates tested source10d3fad88 owner/typed metadata/profile
selection/preflight and the700/112-check C++ fixtures, preserving old713-check32-Q6
and all earlier gates. Exact635be5d9 ancestry and currentmain were verified first;
no conversion worker or reference probe is added to runtime artifacts. See
[native policy details](docs/lighton4b-gdn-output-native-policies.md). Hosted
42218056668 and actual42219710049 supply the unchanged-source controls. No default,
registry, release, EOS or full-decoded-quality promotion: separate calibration
full generation remains pending; PR157 retains all score negatives and provenance.


### Reproducible BF16→Q6 tooling integrated with the guarded native owner

PR155 now carries the exact tested streaming row helper/converter and closed
32-Q6/24-output first-score, full-calibration and observed-logo harnesses, plus
source protocols and scoped controls. See the
[end-to-end reproduction guide](docs/lighton4b-q6-conversion-and-calibration.md).
Fresh Q6 weights come directly from original BF16; protected quantized payloads
and the Q8 tied head are copied exactly. CMake's helper staysEXCLUDE_FROM_ALL;
production artifacts gain no reference probe or model alias. Workflow changes
are four exact read-only/manual selectors plus Q6-only retention routing; all
unrelated jobs, steps and gates remain intact. Source files match the tested
c7e6720ab tree, with compiled runtime hashes pinned to10d3/90d2. Calibration and
logo evidence stay in PR157, including the remaining table/known-logo negatives.
No additional model job, default/quality/release promotion or model-hub upload.


### External-text imatrix format/coordinates verified on Kaggle CPU

[Complete verification](docs/lighton4b-imatrix-verification.md) preserves the
cancelled queued GH job and Kaggle v1/v3 fixture negatives alongside v2/v4 passes.
Actual C++ means match all248/897,024 normalized values, including24 inverse V-head
input permutations. Eight Python controls, strict production target rejection,
legacy valid loading and no-silent-fallback checks pass; no weight quantization
or model forward ran. Public header census distinguishes plainQ4 from publisher
Q4_K_M (216Q4+33Q6 including head), separate F16 projector and external-text
calibration. Publisher English KLD is not OCR evidence. Source corpus contents,
license and calibration source weights remain unauthenticated; no pretrained image
imatrix is assumed. Next is reviewed Kaggle-only fresh-source weighted plainQ4,
with exact unweightedf873 control and248 actual application checks, preserving
protected tensors/gates and all reference/model-quality limits.


### Independent upstream llama.cpp CPU baseline completed on observed cases

[Full source/result evidence](docs/lighton4b-independent-llama-baseline.md) pins
upstream6c01da627, publisher8406c78c and testedworker4d2184ede. Authorized Kaggle
CPUv3 finishes all9 Q4_K_M/Q6_K/Q8×form/table/Arabic arms with exact original
prompt-ID hashes and geometry, no original rerun. FullQ6 table222IDs/text/EOS
and Arabic6 exactly match BF16; form still154/656character edits. Q4M form170EOS
189edits improves over historical capped plainQ4 but uses different weights/vision;
Q4M table256cap/noEOS315edits remains negative. Q8 table223EOS10edits and
form160EOS140edits remain nonexact. All Arabic exact. Independent raw hashes,
EOS/cap,27output hashes and full-matrix edit audits pass. No same-weight/operator,
human accuracy or broad qualification claim. GH cancellation and Kagglev1/v2
setup failures preserved; no new GH dispatch or PR/default/release/model upload.
Next bounded work requires a separately frozen conversion/arithmetic attribution,
not promotion based solely on these observed three-case outputs.

### External-text weighted plain Q4: verified application, partial decode negative

[Complete calibration evidence](docs/lighton4b-imatrix-q4-calibration-results.md)
retains the initial Kaggle ensurepip failure and the authorized source 3ee2b69dd
retry. Regenerated f873 and both frozen first-score hashes are exact. All 248
actual applications/normalized mean hashes and 475 protected payloads pass;
all 248 Q4 projections change, tied Q8 head/vision remain exact. Candidate
65fcdbd4… (4,030,958,240 bytes) is retained privately in Kaggle output with full
producer SHA, Range-verified bytes and all 734 output paths. No GH dispatch,
F32 model, model-hub upload, recipe search or heldout claim.

Table first ID 20353→27 matches BF16 and one-position KLD improves .181511→.125953;
raw RMS worsens .706260→.865078 due in part to a logit shift (centered RMS improves).
Arabic KLD improves .030166→.015504; all 6 full IDs/EOS/text remain exactly BF16,
CER 0/12 and WER 0/2, also matching retained Q4/Q8. Table full inference hits the
strict 45-minute child limit after 225 score rows 0–224, with no generated IDs/text
or completed-EOS summary; no table CER/WER or decoded fix is claimed. All raw
negative/status flags are retained and their incomplete-run meanings explained.
Independent read-only arithmetic and source/hash controls pass. No retry or
extra inference; further table completion/quality qualification remains pending.

### Publisher source-revision declaration correction

The bounded publisher Q4_K_M header declares official b710 in `general.name`;
[the correction receipt](tests/results/qwen35-ocr/lightonocr3-4b-publisher-declared-source-revision.json)
pins the exact original spelling, header/range hashes and full LFS identity.
Earlier wording that the revision was undisclosed was too strong. A declared
revision is not independent source tensor-content verification, and does not pin
the producing converter/quantizer commit or calibration corpus. Old raw evidence
and its hashes remain unchanged; no quality, same-weight or release promotion.


### Physical upstream import audit completed; runtime bridge next

[Authorized CPU audit5199609f4](docs/lighton4b-upstream-physical-import.md)
verifies all249 actual Q4/Q6 matrices native/independent F32 hashes bit-exact,
426 frozen name/shape/type mappings,96 lossless row maps and168 out-dot bridges.
Out-column128/256 block incompatibility is confirmed; negative unpermuted dots
err0.00356691–2.58265. Inverse RMS subtraction loses4 values and inverse A log
loses106, so physical source gamma/A must be consumed directly; conv/dt/linear
norm inverse maps are exact. No inference/F32 model/requantization or quality
claim. Root authorized narrow default-off runtime/import implementation plus
focused Kaggle controls; coordinate owner/header integration with the separate
weighted-HF recipe and avoid duplicate expensive workers. Same-source decoder
comparison must freeze F32 head output (HEAD_BF16=0), not silently reuse the
genuine-BF16-oracle head cast. All old alternatives/defaults remain unchanged.


### Format6 physical import controls pass; same-feature decode next

[Focused Kaggle controls](docs/lighton4b-upstream-physical-import.md#format6-runtime-and-no-checkpoint-controls)
at0610af26d pass557857 physical checks,1147 HF-weighted,713 oldQ6,84 tied-head,
39 publicfence with four actual header-only public rejections, all native/mixed/
BF16 legacy controls and actual native-document/preflight builds. Six Python
transport/header/map cases pass. Initialv1 missing-protocol setup failure remains
retained;v2 compiles/tests without checkpoints. All19 actual source blobs and30
small outputs independently rehashed. No full imported candidate or decoded
quality proof yet; current goal remains matching actual upstreamQ4M outputs.
Next source-fed upstream MTMD feature+native decoder and composed three-case
comparison are separately pinned and queued after the weighted candidate job.
No default/alias/release/public admission or full F32 model.

### Guarded 33-Q6-head capability and public no-ship boundary verified

[Complete model-free evidence](docs/lighton4b-imatrix-q4km-controls.md) preserves
Kaggle namespace/link fixture negatives and source 6d751828e PASS: new 1,147,
old 713/700/112, independent head 84/65,792 values, public boundary 39, all 248
importance hashes and 15 historical source pins. PR155 integration 69abebb7e
merges main additively and preserves 21 tested source blobs; raw upstream qwen35
and experimental formats 5/6 reject before public backend/payload loading, with
no registry/release alias or shipping approval. A later allocation-reuse
preflight 11131b7fc stops before any checkpoint; its 31 raw logs remain, and the
quality producer a4f2c908f restores original streamed arithmetic. Original-F16
one-pass weighted 216-Q4/33-Q6 and complete table/Arabic decoded evaluation remain
pending; the previous 225-row timeout is not decoded-quality proof. No GH job,
F32 reference model, heldout read, new PR or quality promotion.

### Weighted Q4KM-style candidate first-score evidence — 7 November

[Producer receipts](docs/lighton4b-imatrix-q4km-candidate-results.md) preserve
actual sourcea4f2c908f, bootstrap0f53bde5, controller1,796.45s and candidate
SHA5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29
(4,041,905,312bytes). All248 weighted body applications,216Q4/33Q6/474F32 types,
723 payload hashes and474 protected equalities passed; both frozen null score
hashes reproduced. Table firstID20353→27, BF16-first-ID probability.405578→.503789,
but the prior weighted plain-Q4 achieved.590433. Arabic firstID67321 unchanged,
probability.871767→.858263; rawRMS.364631→.336954. Table rawRMS worsens
.706260→.776426; RMS and probability are not decoded-quality selection evidence.
No originalF32 model, requantization, heldout read, shipping/default or release.
All63 small files are retained with exact hashes; full candidate and rawhead stay
remote without an asserted fixed Kaggle retention expiry. Complete table/Arabic
continuation is running from sealed source191406b82 with exact preserved producer
binaries; separate post-selection form follows, then the physical same-weight lane.

Complete-decodev1 setup negative is retained with8 raw files: missing
`/usr/bin/time` stopped native process creation after authenticated runtime/model
and exact Arabic prepared inputs. Corrected sourcee822a6de8 replaces the external
timer with Python child resource accounting and checks native executable loaders,
argument guards and calibration asset paths before model copy. Three local
trust-boundary controls pass; one corrected Kaggle CPUv2 is submitted with the
same immutable candidate/binaries/precision and original budgets. No model rerun
for conversion, recipe change or decoded-quality claim.

Producer CPU metadata omission is explicit:4 logical/affinity CPUs and cgroup
quota400000/100000 are recorded, but vendor/model/flags are blank due to unstripped
`/proc/cpuinfo` keys. Raw receipts are unchanged and exact null firstscore gates
still pass. Future sourceb978b1cef strips keys and covers tabs with a fixture test;
no conversion rerun or inferred vendor follows from this metadata correction.

Read-only independent score consumer authenticates six selected BF16-expanded
rows and reproduces firstscore controls within1e-12, with no model/original forward.
Table centeredRMS.685013→.569416 and source||candidateKLD.181511→.137862
(prior weighted plainQ4.623312/.125953); Arabic centeredRMS.364349→.330996 while
KLD.030166→.032228 (prior weighted plainQ4.360166/.015504). These mixed one-position
posthoc metrics do not select a quality winner or compare with publisher corpus
KLD. Reproducible source8bae598d3 has3 focused mathematical controls; full decoded
calibration remains running on its immutable source, and the recipe is unchanged.

The [physical import setup review](docs/lighton4b-upstream-physical-import.md#actual-persisted-physical-import-and-initial-decoder-setup-negative)
now preserves the actual 723-tensor readback and full candidate hash, both built
executables and initial missing-PyYAML failure before either engine forward.
The corrected source-feature-only Kaggle retry is pending its measured result;
no decoded quality, performance or release qualification follows from this proof.

### Weighted Q4KM-style complete calibration — 7 November

[Full receipts/transcriptions](docs/lighton4b-imatrix-q4km-full-calibration-results.md)
from immutable sourcee822a6de8 finish table214IDs/EOS (no cap), CER160/607=.263591
and WER47/106=.443396 against frozenBF16, beating the pinned llamaQ4_K_M table
315/607=.518946 and88/106=.830189 with256cap/noEOS. Arabic6IDs/EOS/text match both
exactly (CER/WER0). CandidateSHA5d133ae0… and original producer binaries are fixed;
wholemodelrehash, five inputs/case,140 same-native-producer stage pairs and220 raw
vocabulary rows pass controls. Actual small token/text bytes and independently
recomputed distances are retained, with16 raw API files and exact hashes. Table
stilldiffers fromBF16 atID39 and has160char/47word edits; these are not BF16stage
parity or humanOCR accuracy claims. Controller3927.82s/table3676.83s/Arabic210.24s;
CPU/system/explicitlycumulativeRSS scope and earlier negatives remain preserved.
No new original/F32model, heldout read, recipe/default/shipping/release change.
Separate fixed-model post-selection form is running sourceb978b1cef/bootstrap0faac598…;
physical same-weight diagnostics proceed in parallel and remain separate.

### Named quantized-payload attribution — 7 November

[Read-only named comparison](docs/lighton4b-weighted-publisher-payload-comparison.md)
joins the original-F16 weighted723-payload audit with the completed physical
import:225/249 quantized tensors byte-identical after96 verified whole-row maps,
including all33Q6/tiedhead and192Q4 (2,552,432,640 bytes). Only24GDN out_proj raw
hashes differ (141,557,760 bytes), with HF versus retained physical column order.
Every native256-element block maps across2physical256blocks under the exact128
head-tile permutation; source/maps/all249namedSHA/type/shape/offset records are
retained. Raw out-hash differences are NOT a canonical dequantized-weight error
measurement or decoded-quality cause. Floating norm/A, vision inputs and actual
quantized-activation versus dequant-F32 arithmetic remain separate. No model,
quantization, reference forward, default or shipping change; fullform is pending.

### Weighted Q4KM-style form negative — 7 November

[Complete form bytes/review](docs/lighton4b-imatrix-q4km-form-result.md), source
b978b1cef/bootstrap0faac598, retain182 independentIDs/EOS (no cap), valid full
model/runtime/platform/input guards, but CER217/656=.330793 is worse than pinned
llama189/656=.288110; WER43/90=.477778 improves46/90=.511111. Both are mandatory,
so no aggregate/one-metric promotion: this fixedcandidateFAILS the all3 observed
floor, despite table/Arabic wins.13rawAPI files, actualtext/IDs, frozenreference
texts and independenthash/EOS/182-row-manifest/edit-distance review are saved.
Controller3030.72s/form2857.43s; CPUmetadata nowcapturedIntel4cores, cumulativeRSS
scope and previousmetadataomission remainexplicit. No neworiginal/F32model,
heldoutclaim, recipe/default/release/shippingchange. Root-approvednext bounded
lane is original-F16→24tiledQ4 tensorreproduction, then distinctHF-tiled-out tag
with699otherpayloads/HFgamma/logA unchanged andONLY outputactivationP; do not apply
physical-format6 scalar mode to HF tensors. No newmodeljob from this evidence.

### Physical same-weight baseline complete: no promotion

The [actual three-case source-feature decode](docs/lighton4b-upstream-physical-decoded-results.md)
is complete at frozen `ac34f72ca`: exact physical text/captured-feature/prompt
controls pass, but form174/EOS has202 character edits versus upstream189;
table256/cap has no EOS; Arabic6/EOS matches. Word edits improve on form, both
edit distances improve on table, but the required floor/completeness is not met.
All81 small raw outputs, full ID/text evidence, source/max-error magnitudes and
independent metrics are preserved; arrays stay remote. The separate arithmetic
smoke and actual-main primitive closure are independent receipts. No alias,
release, default, heldout or full-runtime-parity qualification.

Physical native-ops actual negative and actual-main trace closure are preserved in
[the bounded result](docs/lighton4b-physical-native-ops-negative.md):38 OFF stage
hashes exactly close to fresh ac34; ON firstscore worsens RMS0.1023485→0.1296631
and max0.576516→0.752944 despite faster ordered prefix timing. Cap2/noEOS is not
quality proof. Independent read-only closure verifies664 raw per-case hashes and
actual first4 input/RMS/QKV/Z/a/b plus first16 layer0 main output. Keep master off;
next is source-fed GDN/attention attribution and focused closed ablation controls,
not promotion or another unbounded model panel.


### LightOn4B tiled-output helper controls — 7 November 2026

Kaggle CPU 143168048/v1 at617534eff passed19 synthetic checks plus19 retained-bundle repeats and5 Python protocol tests in76.18s, with zero checkpoint/inference. [Receipts and limitations](docs/lighton4b-hf-tiled-out-controls-results.md) retain exact source/helper/runtime hashes. Original-F1624tensor reproduction is a separate pending gate; no new runtime recipe or quality promotion.


### LightOn4B actual original-F16 output-coordinate proof — 7 November2026

Kaggle CPU143168450/v1 at24208e2ac reproduced all24 old-HF payloads and all24 publisher-tiled payloads directly from immutable original F16, with48 statistics hashes and72 independent dequant hashes exact,294.67s/no forward. [Full evidence and magnitudes](docs/lighton4b-hf-tiled-out-controls-results.md) preserve canonical weight RMS0.000596409/max0.0468936 and slightly worse tiled unweighted source RMSE. The remaining24 packed differences are coordinate-dependent quantization; no decoded promotion follows. A distinctly gated HF-only output-layout candidate/699 protected-payload audit and decoded tests remain next.


### Same-input GDN localization completed; no production correction

Corrected operator-only Kaggle `143170539/v2`, source `0bbe9d876`, passes
actual native helper/state, first-sixteen actual attention and all six observer
neutrality byte controls. Raw convolution is exact; earliest SiLU RMS is
`1.04e-8` (max `9.54e-7`), and final gated post-norm RMS `8.23e-9`
(max `4.77e-7`). Backend controls have tiny nonzero differences. Initial v1
raw RMS comparison is invalid because fusion skipped its buffer; output flags
correct materialization and change only those source RMS hashes, preserving all
other stage hashes. No model/decode/full-F32-model job or production GDN change
was made. Source-fed native inputs, zero-state and four-vs-sixteen batch limits
remain explicit; see [raw evidence and source blueprint](docs/lighton4b-gdn-source-primitives-results.md).

[Six same-feature cap2 arithmetic ablations](docs/lighton4b-physical-prefix-ablations.md)
complete at7ccb0c468: all38 OFF/all golden hashes close; none improves firstscore
RMS (OFF0.1023485/all0.1296631/quant0.1287035/RMS0.1023507/layers0.1269996/
head0.1039377). Every arm is only `<figure`, cap2/noEOS. No promotion or recipe
selection; ordered timings cannot waive quality. Next bounded lane is actual-input
source attention/F16-cache/softmax primitive attribution, with no speculative
production change or full-model qualification claim.


### LightOn4B HF-only tiled-output native controls — 7 November2026

Kaggle CPU143171211/v3 atc89aaa4cb passed66,710 profile/type/metadata/permutation +39 full-width output/cache controls and all selected old recipe/physical/Q6/stream/public-fence controls,134.98s/no checkpoint. [Raw receipts and both preserved fixture negatives](docs/lighton4b-hf-tiled-out-controls-results.md). Exact runner/library bundle retained remotely. Next is independently audited699-protected/24-replaced candidate packing and full observed form/table/Arabic decode with adequate90min case caps; default/release/quality fences remain unchanged.


### HF tiled-output complete decoded negative (2026-11-07)

Kaggle CPU143174221/v1 completed atB2/C89 with candidateSHA`cc6ce27eded435b8ac278229a8c6ff553f1654d865ff14a1b67c597fa0ca5254`:699other payloads protected,24original-F16-proven tiled Q4 outputs exact. Form182EOS remains exactoldHF candidate andCER217/656 (llama189/656),WER43/90; table256cap/noEOS worsens fromold160/607,47/106 to237/607,70/106; Arabic6EOSexact. Independent rawIDs/text/EOS/Unicode-metric audit passes. No quality/shipping/heldout claim. Full report63a586c0 and allsmall artifacts/actualtranscriptions are preserved in [decoded results](docs/lighton4b-hf-tiled-out-decoded-results.md). Separate artifact-only common-prefix/stage/body audit pending, no new inference.


### HF tiled-output artifact-only causal audit (2026-11-07)

Kaggle143195841/v1 source0d90fd0b2 passes actual4GB model/all723payload hashes and independently authenticated699old/24original-F16 publisher payloads, allscore argmax/hash/finite controls. All3prefill inputs+positions exactold; earliest observed difference layer0 (tableerrorRMS0.00273095/max0.0423252, magnitudes0.202175→0.202302), layer31RMS0.0532514/max1.26601, firstlogitsRMS0.162988/max1.0. Formall182/Arabic6 causalrows compared; tableonly0..113 (firstselectiondiff303→430),142laterrows excluded. Full32stage/rawmagnitudes/counts and actualtranscriptions retained in [decoded negative](docs/lighton4b-hf-tiled-out-decoded-results.md). No new inference or same-weight originalBF16 parity claim; quality remains negative.

### Physical F16 full-form initial setup negative

The first full-form worker at `fb8a5ecf1` stopped before entering the model worker: a hosted `sitecustomize` warning polluted the stdlib guard test stdout. [Raw setup evidence](tests/results/qwen35-ocr/lightonocr3-4b-physical-f16-attention/full-form-v1-setup-negative/evidence.json) is retained. The one-line test isolation at `4a63d73be` passes all eight focused checks; numerical code, immutable weights/features and release fences are unchanged. The bounded retry is submitted; no decoded or speed result is claimed yet.

### Main SDK linkage and first real-WASM control

The actual native main archive now passes all five closed-helper link checks. The first Emscripten 6.0.2 control compiled the actual shared attention/helper objects, then failed final linking because its standalone harness omitted production `EMSCRIPTEN_SYSTEM_PROCESSOR=wasm` and excluded the vendor WASM quantizer source. [Raw native pass and WASM setup negative](tests/results/qwen35-ocr/lightonocr3-4b-physical-f16-attention/sdk-link-v1-negative/evidence.json) are preserved; `d7a35a13d` corrects only the harness profile and adds source-membership guards. No full WASM or decoded qualification is claimed.

### Physical F16 attention full-form measured result

[Authenticated full-form evidence](tests/results/qwen35-ocr/lightonocr3-4b-physical-f16-attention/full-form-v2/evidence.json): source `4a63d73be`, unchanged physical weights/features, frozen old-ON baseline 38 stages and both score rows exact. The new gate completes all 177 teacher positions and independent 167-token EOS generation. Against original BF16, native CER 175/656 and WER 45/90 improve on retained upstream 189/656 and 46/90 for this form. It remains numerically different (upstream first-score RMS 0.105889/max 0.558670), and is not release qualification. Full arm including teacher scoring took 241.94 s; sampled RSS 3,023,495,168 B. Independent teacher-row numerical review and same-configuration table/Arabic plus representative held-out/composed-vision gates remain pending.

### Actual native SDK and real WASM linkage verified

[Corrected Emscripten control](tests/results/qwen35-ocr/lightonocr3-4b-physical-f16-attention/sdk-link-v2/evidence.json) at `d7a35a13d` passes the actual main SDK archive link and real Emscripten 6.0.2 shared-decoder/helper closure plus Node execution (five checks each, 456.24 s). The public helper remains the unsupported closed stub; only the experimental decoder target receives the private ABI definition. This verifies linkage, not a complete OCR WASM artifact or model qualification. V1 negative remains preserved.


### Actual177 original-prefix teacher scores after physical F16 attention (2026-11-07)

Artifact-only Kaggle143200239/v3 sourcebcbd50b18 passes177whole-vocabulary originalBF16/nativehash+causalcontextguards, no inference. All43,952,640logits: RMS0.386425/max8.82935, original/nativeRMSmagnitudes2.28140/2.26667; meanKLD(BF16||native)0.0123239,169/177top1, nativeoriginaltokenprobabilitymean0.861789. Row0RMS0.335973/max1.60669;row1710.257192/1.35267;EOS1760.655251/3.33821. Pinnedupstreamonlyrow0 admitted(RMS0.105889/max0.558670,exactproducerclosure); no misalignedgreedyrowssubstituted. BothsetupnegativesV1/V2 retained, V3fourcontrols+independentaggregateauditpass. [Fullrawreceipts,magnitudes/probabilities/limits](docs/lighton4b-f16-teacher-results.md). Differentweight/vision/arithmeticdomainsremainexplicit; no sameweightparity/generalshippingclaim.

### Same-configuration table/Arabic calibration negative

[Raw calibration evidence](tests/results/qwen35-ocr/lightonocr3-4b-physical-f16-attention/calibration-v1/evidence.json) at `fa3aab156` reuses the exact completed-form binary, ten library/binary hashes, gate tuple and physical weight bytes. All 222 table and six Arabic teacher positions complete. Table independent generation hits cap256 without EOS: BF16 CER316/607 and WER88/106 versus retained upstream315/607 and88/106; this is not a qualification pass. Arabic completes six tokens/EOS with exact IDs and text against both references. Worker413.19s; table219.14s, Arabic13.90s. The original cap256 negative stays fixed; a separately frozen extended-token upstream/native table comparison is authorized to determine completion, without promoting the gate.


### Frozen held-out selection (2026-11-07; no inference)

Prepared [LightOn4B held-out v1](docs/lighton4b-heldout-v1.md) and its hashed-input [protocol](docs/lighton4b-heldout-v1.json): historical receipt, two-column page, owned inventory grid; two real documents plus one controlled table. Six source/gold hashes checked. No candidate outputs inspected or model jobs launched. Current LightOn4B cases, identical receipt aliases and derivatives excluded; historically used other OCR lanes, not pretraining-clean. German official-page reserve is explicitly outside primary quota. Prior visual annotations require expert-human adjudication, and embedded literary-text rights remain a distribution limitation. Execution blocked on generic cap1536/context2048 test driver with prompt<=512, exact source/binary/gate/prompt and geometry admission, licensing/reference review and separate hosted capacity authorization. Native composed vision is primary; source-feature decoder isolation cannot rescue a failed case. All three cases must independently meet actual upstream Q4_K_M CER/WER and completion, with positional inventory cell controls; no aggregate hides failures. Resource quotas and partial/raw retention are frozen.


### Composed projector executor controls (2026-11-07)

[Model-free hosted controls](docs/lighton4b-composed-vision-controls.md) source91025 passed actual supportedCPU/privateABI255297 and nine actual CLI guards, unchanged production/CMake bytes,116.32s. No model acquired/forward executed. Raw receipt52fd2c and file inventory preserved; no qualification claim. Next authorized observed-form paired composed/source-feature diagnostic uses same decoder and complete greedy or explicit cap; canonical projector weight/dtype identity remains unproven, actual upstream visual intermediates unsupported.

### Extended table completes but does not remove the regression

[Authenticated cap512 evidence](tests/results/qwen35-ocr/lightonocr3-4b-physical-f16-attention/table512-v1/evidence.json) at `f789609c2` completes upstream and native generation at 282 tokens/EOS. Native BF16 CER349/607 remains one character worse than upstream348/607; both WER97/106. The original cap256 result is not replaced. All old256 native raw-score hashes, all38 prefill stages and IDs reproduce exactly after the explicit test-driver-only cap opt-in; full upstream observer OFF/ON IDs, probabilities, decoded text and EOS are exact. Source69.47s/native182.16s on this worker are diagnostic timings, not a promotion. Actual loaded metadata confirms selected Q4 matrices use CPU_REPACK/extra while QKV Q6 and tied head use mapped buffers. A same-input operator probe must establish numerical consequences before any physical-only runtime fix; no global vendor change or release is authorized by this result.


### Observed composed-vision negative (2026-11-07)

[Paired form result](docs/lighton4b-composed-vision-negative.md) source8be/receipt1458: source-fed167EOS175/656char45/90word vsoriginalreference, composed194EOS259/65647/90, actualupstream189/46. Both floors fail for composed; no ship. Oldsourcefed allrow+38stageanchors exact, pairedpatches/textembeddings/positions exact, earliestcomparable mergedvision RMS.011237/max1.68147; firstscoreRMS.192548 vsfed.105889. Independentrawtext/IDs/edits and12 vector hashes checked; firstscoreRMS FP64 reduction differences2/1ULP explicitly retained, maxexact. Fullvectors remainremote. No human-GT claim or canonicalprojectorweight-equivalence claim. Hold additional table/Arabic model runs while tracing actual F16projector canonicalweights/precision and earliestavailable sourcevisionstages; implement only measured gated corrections.


### Projector-only attribution initial setup negative (2026-11-07)

[Preserved first attempt](docs/lighton4b-projector-attribution.md) source0f92/receipt5f684: standaloneCPP andmtmdcompiled, finalreference linkfailedobsolete-lcommon at250.86s. Build-before-transferguard stoppedall2GBparametertransfer/canonical/visionforward; no numericalstageclaim. Narrowreference-onlyc010 removesunneededcommondependency, no math/profile change. Preservefailurebeforeboundedretry.

### Physical Q4 repack route measured, integration remains experimental

[Operator evidence](docs/lighton4b-q4-repack-operator-results.md) preserves exact
source `3b41475c8`, raw receipt `4dc28c17`, two build profiles and bootstrap. All
five actual N16 raw projections close; preferred Q4 routes match across both
pinned backends while changing raw arithmetic slightly. Q6 N16 retains a
separate backend residual. No decoded attribution follows. Next is bounded
default-off physical buffer-cache lifecycle/source-fed verification, then
same-feature decoded comparison; current Q4 is not qualified.


### ProjectorV2 qualified evidence and observer invalidation (2026-11-07)

[PreservedV2](docs/lighton4b-projector-attribution.md): raw64eb/sourcec010 completes318.87s, canonical297/298 allauthenticated333.5Mvalues (198F32exact/99F16 max2^-25;287nativeF16roundexact/10edgecasesunresolved), actualsourceOFF/ONmergedbyteclosure and validpreprocessingRMS.000221817/max.0156863. Postcomputeintermediatebuffersalias, so rejectALLintermediatemagnitudes despite rawmeasuredstatus/nonzero/finalneutrality. CorrectedreferenceLIVEcallback+arithmeticclosures+RGB/resize/normalizationdecompositionpreparedc1ed, no productfix orqualityclaim; awaitpeerreview/boundedretry.

Physical Q4 repack initial hosted runtime controls at `26763167f` pass57
supported lifecycle checks and2 main closed-stub checks, then stop before
source-fed work on a reference-adapter missing-declaration compile error.
[Preserved partial receipt](docs/lighton4b-q4-repack-operator-results.md) keeps
that distinction explicit. Reviewed include/build-order repair `5f6dfb9f0`
leaves runtime bytes unchanged; one bounded retry is pending, with no model
inference or promotion.

Q4 repack cache retry source `5f6dfb9f0` passes the complete bounded control
worker (`c7130c6d`,491.644s): all30 actual-input OFF/ON projection bytes exact,
lifecycle57 plus255297/142/39/84/5/2 compatibility controls.
[Raw and independently qualified evidence](docs/lighton4b-q4-repack-operator-results.md)
preserves the initial compile negative and stale outer-bootstrap inventory
limitation. No model forward/decoded or new real-WASM qualification. Next is
the reviewed table512 OFF282row/38stage anchor then ON comparison; no promotion
until all observed/composed/held-out gates pass.

### Physical Q4 repack model negative: keep gate off

[Completed same-feature table OFF/ON](docs/lighton4b-q4-repack-operator-results.md)
source `f079dcec6`, receipt `5acbdab9`,684.813s: OFF closesall282raw/38stage
anchors and282EOS; ON295EOS regresses CER349→365/607 andWER97→104/106.
First-scoreRMS0.188629→0.202156; actual216-entry1.663GBquantcache/zeroF32
weightcache costs sampledRSS2.959→4.632GB. Alternative staysdefault-off and
unqualified, with no speed or quality promotion. NativepairedstageL0RMS
3.266e-5 growsL31RMS0.07623; actualupstreamtextinternalstageswerenotcaptured.
Prioritize composed-vision attribution/fix; do not infer a new text-kernel fix
from this negative. Preserve all raw text,IDs,receipts and immutable sources.

### Live projector V3 and precision-policy closure (2026-11-07)

Preserved valid live-callback projectorV3 receipt427b/sourcec1ed, all32 callbacks once, patch/position and FFN residual identities, actual upstream OFF/ON merged-byte closure. Independent17 small-array hashes verify source normalization/HWC→CHW and341 differing resize channels/max2. Dtype-aware audit closes198 F32 parameters exactly and99 native matrices/100 F16 source tensors after required rounding; V2's10 F16 mismatches were sourceF32 parameters incorrectly judged after an unnecessary cast. All297/298 canonical transforms close; original checkpoint identity remains unproven. Valid actual patch RMS.0008215/layer0 .003005/layer23 7.98955/merged .011237 locate growing arithmetic divergence; report-derived metrics and independent raw subset are distinguished. V2 aliased stage claims remain rejected. Separate default-off Pillow22/F16-matrix/F32-state production draft awaits hosted component controls and complete decode; no quality/default/release promotion.

### Gated vision model-free controls preserved (2026-11-07)

Source0eca's separate Pillow22/F16-vision draft passes690.72s hosted controls, raw31e6d6e5. Independent58hashes,8actual-upstream resize pairs and actualformRGB/packedpatch bytes all exactV3, closing earliest pixel/normalization divergence. Synthetic24block2×2 graph37/mainstub7/Pillow17+main17 pass; no checkpoint acquired, no actual-weight/decoded proof. Initial title/slug400 rejected submission retained. Published raw controls/cache/logs/bootstrap and limits; next pinned vision-only297tensor actual-stage worker must reproduce oldlegacy stages before comparing newsource-compatible output. Default gates/public release remain unchanged.

### Actual vision-component map-gate setup negative (2026-11-07)

Source d338 worker stopped131.69s at reference map admission: OCR flag was enabled after mapper call while controller correctly cleared inherited flags. All297 tensor SHAs/source precision policy and synthetic graph passed;1.334GBvision-only parameters staged, no actual-weight vision/text forward or stage metrics. Original raw760db939 retained privately/remotely; public redacted receipt/logs bind original+public hashes without access locators. Scoped b59e reference adapter gate-order fix and actual metadata-only sparse-map control run before transfer; same297 schema1de40, seven local tests and peer review pass. Production/GGML/CMake/numerical helper unchanged; retry remains new evidence, no promotion.

### Actual source-compatible vision byte proof (2026-11-07)

Corrected b59 component completed280.59s, raw039b024f: actual sparse metadata admission before1.334GBtransfer, all297tensorSHA/source-declared precision exact. All29legacy anchors remain exact; newPillow/F16matrix/F32state pipeline matchesALL28actual live upstream arrays (patch/position/embedded/24blocks/merged), RMS/max0. Independent9rawfiles18.2MB +full cross-manifest hash census pass; full73owned/28source remote rehash required before composed-form model staging. Ordered vision-only138.00s→15.47s/~1.50GBRSS is component evidence, not balanced/fullspeed proof. No text checkpoint/originalBF16/upstream rerun; no decoded/heldout/general claim or default promotion. Full composed form follow-up preservesold194EOS/allraw+text/vision anchors before only2gatechange; fixed167source-feature parity is measured independently.
