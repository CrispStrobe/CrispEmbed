# CrispEmbed plan — open work (updated 4 November 2026)

Only open or active work lives here; completed runs move to [HISTORY](HISTORY.md) with their
receipts, measured numbers go to [PERFORMANCE](PERFORMANCE.md). Start with
[the 3 November handoff](docs/handoff-2026-11-03.md) and
[current state](docs/current-state-and-next-steps.md). Branches named below are merged
into `main` unless the row says otherwise; start new work from `origin/main`.

| Lane | State | Next action | Done when |
|---|---|---|---|
| Q4_K_MIX publications (MiniCPM-OCR, Unlimited-OCR-2, MoEdern 27M/115M, Josev, UltraBERT) | Recipes and receipts on main. TeleOCR and MinerU Q4_K_MIX are already published and registered (`teleocr-q4_k-mix`, `mineru25-pro-q4_k-mix`, PR #116); the rest are not | Coordinator dispatches the cosine-gated publication per model (sections below); Josev needs a quantized branch in `josev_publish.py`, MoEdern needs SciFact/client panels first | Hub upload, empty-cache readback, registry alias with SHA-256 |
| `env_gate.h` false/off/no lockstep | CrispASR #572 open (46 checks pass); CrispEmbed side not written | Handoff 3 Nov lane B | Identical header in both repos, tests in CI, sibling sync green |
| OXR native-doc parity | Rerun 41664201815 failed again (`native=fail`, not the trace-key bug) | Handoff 3 Nov lane C | Five-case parity receipt or scoped runtime bug |
| Unlimited-OCR-2 Q8_0 on CUDA | 0.9555 < 0.97 (CPU 0.9739) | Search a CUDA-qualifying Q8_0 mix (F16 shared experts/attention) | ≥ 0.97 on CPU and CUDA |
| GOT-OCR2 Q4_K drift | F16-r2 exact vs original; Q4_K diverges at 83/20; no final-logit cosine measured, so the 1 November precision policy cannot yet be applied | Teacher-forced final-logit cosine of the published Q4_K vs the F16-r2 pipeline (as for Unlimited/MiniCPM), then a mixed search only if < 0.94. Owner: the model is research-licensed (Tongyi), so whether a Q4_K is worth more compute is an owner call (handoff "Owner decisions") | Cosine > 0.94 recorded, or 3/3 token-exact, or owner accepts/declines |
| Server load control (PR124) | Merged, gates default-off | Live A/B with a real GGUF; concurrent latency; chat-route parity | Receipts per item in the section below |
| Reranker residual (~0.2 logits) | Pair layout/truncation fixed (PR122) | Fix the stage-diff harness, Spearman/top-k instead of exact argsort | Attributed or accepted |
| CUDA / Metal audit follow-ups | Matrices in `docs/gpu-support-matrix.md` | One gated GPU path or A/B at a time | Measured A/B before any default flip |
| Ovis, Josev, MoEdern, UltraBERT, LFM2.5, EmbeddingGemma precision | See the 29 October table below | One bounded protocol at a time | Per-row gates in that table |
| Issue 71 inpainting | Engines published (default-off); stream merged | Representative subtitle/video quality, MI-GAN512 Q4_K caveat (0.9019) | Owner-reviewed quality panel |
| Releases / pub.dev / app pin | v0.17.14 published, pub.dev unqualified, app on 0.17.12 | Qualified assets and fresh consumers first | Release procedure in the pub.dev section |
| OXR-1.0 quantized + GPU | Q8_0-large / Q4_K-head pass the owner bar (cos ≥ 0.998 vs F16, CPU/Metal/CUDA); German diverges 1 token from decode-step activation quantization | Opt-in F32-activation decode A/B (the recipes are already published and registered as `oxr-1.0-q8_0` / `oxr-1.0-q4_k-head`, PR #97) | A/B receipt; German page token-exact or the divergence attributed |
| OXR native document route / D-FINE | Hosted helpers/replay/native pass (41249898971), `dfine` stage failed there; D-FINE regions match original 5/5 but 2.6x slower | Re-run `oxr_native_doc.sh` `dfine native` stages on main; encoder per-op profile | Hosted parity on all five cases; profile receipt |
| MinerU2.5-Pro full native doc route | 8/8 model-backed cases identical (41251475795); default-off | Owner: default-on or not (handoff "Owner decisions"); image-bearing-table fixture | Decision recorded; fixture receipt |
| Marker-style pipeline / TeleOCR | Implemented and bounded-validated, default-off; TeleOCR strict numerics false at every precision | Representative licensed multilingual/layout gold corpus (needs owner-provided licensed annotations) | Corpus frozen; CER/WER/order receipts |
| Model-weight licences in apps | Registry/cards/gates corrected; CrispSorter PR20 at0af4c34 adds fail-closed OCR consent, typed Gemma terms, real0.17.12 pin and image backfill. Its reported Metal-feature cargo check passes; PR18 remains draft | Review/integrate [CrispSorter20](https://github.com/CrispStrobe/CrispSorter/pull/20) alongside18; retain Windows-helper/live-OCR gaps. Check CrispCalc consent separately. No vendor/app pin upgrade | Downstream acknowledgement PRs merged and actual consumer checks recorded |
| **Release v0.17.14** (owner: release once every PR is resolved and main is green) | **Published 4 November 2026** from `9c790c81`; 16 assets (all platform archives, both WASM bundles, CUDA/Vulkan extras); macOS archive checks `0.17.14` and unversioned `@rpath` install names. Downstream apps still pin prebuilt libs to 0.17.12 via `CRISPEMBED_PREBUILT_VERSION` (PR #132) | Section "Release v0.17.14" below: only step 4 remains — tell downstream to drop the override | Downstream told to drop the override; app pins updated by their owners |
| CI selection after PR #134 | Merged; docs/evidence-only pushes run nothing, so the main head may have no Build run | Before tagging: `gh workflow run build.yml --ref main` (and build-wasm, build-wasm-embed, rust-crates, inpaint-cli); confirm the dynamic desktop matrix in build.yml evaluates on hosted runners | One full green Build on the main head |
| Failing main workflows owned by the integration session | Recorded as non-release-blocking on 4 November 2026 (HISTORY): `MoEdern real padded Q4_K diagnostic` stops at authenticated input preparation (run 41587641989), and `Layer0 attention op-by-op same-input replay v1` is a manual experiment with no main run | Re-run only if the authenticated inputs are authorized; otherwise no release action | Recorded; re-run or leave closed |
| Embedder/tokenizer residuals (CUDA audit, PR #115/#122) | Fixed: SPM space collapse, XLM-R pair layout, 512 truncation | `xlmr-ner-hrl` GGUF has no Unigram scores (reconvert with scores); `paraphrase-multilingual` native truncates at 512, original at 128 (honour `max_seq_length`); Ettin reranker unchecked (tokenizer needs transformers 5) | Each matches the original on the 77-text panel |
| EasyOCR residuals (PR #100/#110) | x86 line drops fixed; Python-parity work merged | `scan_page_pd` page output differs between Apple arm64 and x86/CUDA (first diverging region/stage); remaining punctuation/spacing differences vs Python EasyOCR on low-confidence lines | Attributed and fixed, or documented as a float-order tie |
| MI-GAN256 Metal/Q4_K | Metal F32/F16/Q8_0 pass; Q4_K fails (masked 0.81); GPU default off | Q4_K probes with sensitive layers at Q8_0 (`CRISPEMBED_QUANTIZE_MIGAN_Q8`); flip to the padded-K Metal path only after measuring delivery-5 CPU kernels on Apple CPU | Q4_K ≥ 0.94 masked or declined; A/B before any default flip |
| Big-LaMa (research only) | Weights' rights unclear; kept on `feat/issue71-lama-research`, private Hub copy only | Nothing until the owner clarifies the weight licence | Licence clarified (then publish) or lane closed |
| Ovis 2048 / final-integrated source | 1024 default; 2048 has six composed final-norm numeric failures with exact text | Same-input attribution (section "Ovis 2048-patch") | Attributed; or 2048 stays unqualified |

## Regression review and OCR continuation — 4 November 2026

The [current executable qualification continuation](docs/ocr-qualification-2026-11-04.md)
records reviewed evidence, exact live run IDs, preservation and remaining acceptance
gates. It supersedes historical queued/dispatch language. Heavy execution stays
hosted. PR141 includes a source-correct, default-off LightOn3 vision epsilon
experiment; development0.17.15 is unreleased, published assets remain0.17.14.

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

PR141's actual native compilation milestone is now complete: all ordinary
checks at**ea2edb5bf** are green, including macOS/iOS Build41899765931 and279
model-free controls. Earlier Apple capacity cancellations and the M1 pressure
stop remain in HISTORY; no further work is needed on that M1 for compilation.
This does not establish Apple model-inference quality or release new assets.
The separately staged completion/remaining controllers add focused controls;
observe their actual final-head CI before integrating them.

### Current bounded OCR execution and completion gates

See the [reviewed reports and repairs](docs/ocr-benchmark-expansion.md#reviewed-retained-reports-and-further-repairs)
for exact source commits, artifacts, initial failures and error magnitudes.

| Lane | Current evidence / job | Exact next action and completion gate |
|---|---|---|
| MinerU2605 artifact/public-consumer qualification | Complete8-case panel:32 numerical arms pass,28 decodes exact; F32/F16 all8 exact. Q8/Q4 differ on the2 layout cases. [Complete review](tests/results/mineru2605/aligned-panel-20261104/eight-case-review.json) | All10 raw ZIPs now have verified90d preservation through2 February2027; retain quantized layout negatives. Before any publication finish required task quality/public surfaces, export standard GGUF metadata with per-tensor payload proof, empty-cache Hub readback and SHA-pinned consumers. No repeat panel or promotion from cosine alone |
| LightOn3-1B CPU KV precision A/B | Full epsilon A/B41889499814 is negative: all12 bars pass, F32 all6 exact, F16 form-transcription changes language at ID6. Persistent native K/V is F16 versus float32 original | [Two-call completion41901879295](docs/lighton3-kv-complete-first-v1.md) passes using10 retained calls and the byte-identical binary. Corrected F32/F16 minimum cosine exceeds0.99999999992 and free outputs match. Remaining5-case/60-call41905258340 at558825720 is running; authenticate24 final rows/72 total calls, all old-path bytes, original full logits/free outputs and strict/cap flags. No duplicate first case, default or weight fitting |
| LightOn3-1B corrected-default decision | Full epsilon verdict remains negative; first cache proof is positive and remaining panel is running, both gates default-off; actual Apple compilation now passes | After positive complete cache/decoded panel and actual Apple evidence, follow [the separate default scope](docs/lighton3-pixtral-epsilon-ab-v1.md#subsequent-default-decision-is-separate): preserve old closures, literal0 old paths/legacy isolation, new source controls and actual unset-vs-corrected1 output replay for all6 modes×2 formats. No new originals or capped-document/publication claims |
| LightOn3-1B EOS/document scope | Current six request modes use cap128;8/12 original outputs are capped, even if corrected native IDs match | Freeze a separate licensed document/EOS protocol. The current `tests/lightonocr3_trace.cpp` and generation validation hard-limit256 tokens: stay within256 or version the harness/comparer with explicit memory/time bounds before a larger cap. Preserve128-token receipts and strict failures; require complete EOS-ended original/native outputs, full-position logits and independent plain generation before document claims |
| MinerU2605 quantized layout quality | Existing aligned recipe passes numerical bars but Q8 changes rotated box coordinates and Q4 changes5 region labels/14 ordinal records (max coordinate delta0.172) | Preserve original eight-case evaluation. Freeze a separate licensed calibration/held-out corpus; use Python dumps and native same-input intermediates/crispembed-diff to locate additional Q4-vs-Q8 error before choosing higher-precision down projections or vision roles. Record every retained tensor's type/payload hash, C-quantizer recipe, full-position logits and free structured outputs. New recipe requires independent held-out layout/EOS/quality/public-consumer evidence; do not fit on the frozen negative cases or replace their verdicts |
| Xiaomi0 | Metadata/frontend41870593556 passes exact pixels/patches; inference41871383235 at3a5a9a1dc passes all4 arms and exact IDs/text at the16-token cap | Bounded two-case integration complete. Preserve6 strict stage failures and capped outputs; freeze separate EOS/document and Q8/Q4 protocols before expanded precision/publication. External detector composition remains a separate experiment |
| LightOn3-0.8B | Extraction diagnoses1759 float-metric/probability differences; IDs/text/all other original fields exact; no native calls | Preserve the old guard failure. Implement/review the [exact retained-capture continuation](docs/lighton08-retained-native-continuation.md), authenticating all905 arrays and old source/input/score identities; no blind original rerun or ignored-field broadening. Keep the first source result and all three-case requirements distinct |
| Nemotron frontend | Actual11/11 geometry/packing and9/9 negatives pass | Bounded milestone complete. Native resize/positions and neural backend remain separate work in the model document |

## Active work in flight — benchmark OCR expansion, 4 November 2026

Owner-authorized follow-up to the OmniParseBench / OmniDocBench review. Isolated
implementation lanes start from current main; each preserves existing negative evidence
and model defaults. New registry aliases require qualified immutable artifacts.

| Lane / branch | Scope | Completion gate |
|---|---|---|
| `fix/ocr-qualification-nov4` | Owns the first hosted-run review, LightOn1B dependency repair, Xiaomi processor representation repair and bounded extraction of retained failure reports | Reviewed exact failures; corrected bounded jobs at one recorded source, with original guards and negative evidence retained |
| `feat/ocr-mineru2605` | Exact 2605 profile and original-first portable job integrated into PR138; 21 new/17 preserved conversion controls pass | Full-position F32/F16/Q8/Q4 comparisons and symmetric complete 2.0.7/native helper traces across the frozen eight cases |
| `feat/ocr-lighton3-1b` | Exact transcription/grounding profile, actual GGUF/source census and portable original-first trace job; 19 controls pass | Hosted original/native pixels, stages, full-vocabulary logits, free generation and structured output; old profile behavior preserved |
| `feat/ocr-xiaomi0` | Format-4 profile, public task/cap admission, original external PP-DocLayoutV3 producer, strict OTSL and bounded document assembly; 26 controls pass | Hosted metadata/frontend and two-case F32/F16 execution, then a separately frozen document experiment; detector remains external |
| `feat/ocr-benchmark-integration` | Shared document evaluation and candidate/license matrix; 32 document controls pass | Hosted source checks, original-first qualification and representative licensed document quality before alias/publication work |
| LightOn3-0.8B portable replay | Separate current-source F16/Q8 job preserves the old three cases, prompt, 1024 cap, recipes and receipts; 8 portable/6 retained-review controls pass | All three one-case jobs at one reviewed commit; complete final-logit coverage, completion, text differences and explicit precision decision |
| `feat/ocr-ovis3-task-audit` | Separate pinned Markdown task/source and cleanup audit; 14 controls pass and the old plain-OCR failure is preserved | Independent licensed input/gold, actual original Markdown captures, then distinct native task admission; no old gate is changed |
| `feat/ocr-nemotron-geometry` | Hosted processor comparison41810180035 passes all11 cases exactly and all9 negative controls | Native resize, learned position interpolation and the complete neural backend remain separate |

Checkpoint: [integration PR #138](https://github.com/CrispStrobe/CrispEmbed/pull/138)
contains the reviewed source at `7ac5b594e63a34b53a753d9520fccff4ceb888d0`.
All 162 focused OCR controls pass locally and the
[shared hosted controls](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41735520715)
pass. All15 workflows are green, including all11 native-build jobs and both WASM
workflows. [The CI record](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/ocr-benchmark-expansion/source-ci-7ac5b594.json)
preserves the source identities and earlier invocation/setup/WASM-link failures
with their corrections. The source PR merged on 4 November 2026. The
[first five hosted jobs](tests/results/ocr-benchmark-expansion/initial-hosted-20261104.json)
ran at `2f1bbdb1`: Nemotron's bounded frontend passes; LightOn1B dependency
resolution and Xiaomi metadata validation fail before inference. LightOn0.8
executes the original capture but fails its frozen-original guard before native
execution. MinerU's exact failed stage still needs its retained reports extracted.
No new model is qualified. The next jobs and evidence gates are in
[the expansion guide](docs/ocr-benchmark-expansion.md); do not repeat the original
batch or infer precision failure from a setup failure.

This work does not adopt restricted terms or change the immutable app runtime.
OvisOCR3's frozen capped-page failure and all old strict LightOnOCR-3-0.8B verdicts
remain recorded. A separate review of complete retained logits finds F16/Q8 clear
the current owner numerical bars; Q4 does not. Fresh current-source portable
qualification is still required. See PR138's `docs/lighton08-precision-next.md`.

## Release v0.17.14 — 4 November 2026

Owner decision: cut the next release once every PR is resolved and main is green. Nothing is
tagged until then.

Gate status (4 November 2026): **met**. The open-PR queue is empty (#135, #138 and #139 merged),
`main` at `2f1bbdb1` has all 16 workflows green including all 11 native build jobs, and the two
manual diagnostics are recorded non-release-blocking below and in HISTORY. Versions agree on
0.17.14; `tests/test_pubdev_release_gate.py` passes; the changelog entry and
`RELEASE_NOTES_v0.17.14.md` are in this change.

1. **Gate.** `gh pr list --state open` is empty (or the owner excludes the rest). On the current
   main head, dispatch the full set because docs-only pushes no longer start Build (PR #134):
   `gh workflow run build.yml --ref main`, and likewise `build-wasm.yml`, `build-wasm-embed.yml`,
   `rust-crates.yml`, `inpaint-cli.yml`. All must be green. The two integration-session
   diagnostics in the index above must be green or recorded as non-blocking.
2. **Preflight.** VERSION, `flutter/crispembed/pubspec.yaml`, crate/package versions and
   `flutter/crispembed/CHANGELOG.md` agree on 0.17.14; run `python tests/test_pubdev_release_gate.py`
   (PR #91). Write `RELEASE_NOTES_v0.17.14.md` like the earlier release-notes files; it and the
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

Operational note for the next cut: pushes of `v*` tags start no workflow in this repository, so
both `release.yml` and `release-wasm.yml` must be dispatched at the tag ref
(`gh workflow run release.yml --ref vX.Y.Z` and the same for `release-wasm.yml` with
`-f tag=vX.Y.Z`). `release.yml` creates the release with an empty body, so attach the notes with
`gh release edit vX.Y.Z --notes-file RELEASE_NOTES_vX.Y.Z.md`. Its `Create GitHub Release` step
can fail transiently while uploading the multi-hundred-MB CUDA bundles; rerun the failed jobs
once the release exists instead of re-cutting the tag.

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

The owner-authorized benchmark expansion is tracked in PR138 and its
`docs/ocr-benchmark-expansion.md`. Prioritize exact supported-family upgrades
before the larger new backends:

1. MinerU2.5-Pro-2605-1.2B: Apache-2.0 checkpoint and separately pinned helper;
   exact 2605 profile and original-first qualification, preserving 2604.
2. LightOnOCR-3-1B and Xiaomi-OCR-0: exact profile/frontend/native reuse,
   grounding and explicit document-stage dependencies. Community GGUFs are not
   native qualification.
3. LightOnOCR-3-0.8B F16/Q8 portable replay and OvisOCR3's separately declared
   document task protocol; preserve existing strict and capped-page evidence.
4. Nemotron 3.5 Parse: pinned OpenMDW-1.1 source, bounded frontend, then its
   new C-RADIO/causal-mBART composition.
5. LightOnOCR-3-4B, Infinity-Parser2-Flash and Qwen3.5-9B: separate memory,
   long-page and backend protocols after small dense-model parity.
6. Qwen3.6-27B and Infinity-Parser2-Pro: separate larger dense/MoE ports.
   Pro has 40 layers and 256 experts; active parameters do not determine GGUF size.

TeleOCR remains in its existing `qwen-research` acknowledgement/qualification
lane, not this permissive queue. Indra remains public-code-analysis-only.
dots.mocr is a distinct candidate from the removed dots.ocr entry: its MIT tag
has supplemental terms and needs a policy decision before adoption.
Chandra-OCR-2, Surya2.5, Hunyuan-OCR and other restricted candidates retain
separate explicit eligibility decisions. Do not infer weight permission from
a public repository or apply a full application's license to a separate model
without reading the exact materials.

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
