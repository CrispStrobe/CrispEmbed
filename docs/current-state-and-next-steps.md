# CrispMath and CrispEmbed: current state and executable lanes

Consumer snapshot verified5 October2026; vendor roadmap updated 16 October 2026.
This handoff scopes follow-up work; it does not say the
lanes below have been implemented. Read the status, choose one lane, and record
its exact source, evidence and remaining dependencies before handing it on.
All references are public repository paths, workflow runs or published services.

## What is done

| Surface | Current verified state | Public evidence |
| --- | --- | --- |
| CrispMath main | PRs 1–3 merged; application revision 4f45017534720c96d46b5c0874a3100e9cf28e7e has successful post-merge platform, CI/CD, Feature and Pages runs | [PR 3](https://github.com/CrispStrobe/CrispMath/pull/3), [Feature validation](https://github.com/CrispStrobe/CrispMath/actions/runs/37240927201), [CI/CD](https://github.com/CrispStrobe/CrispMath/actions/runs/37240927178) |
| Apple binary | 1.2.0 (23), source 637b07ec0e13c428385dc670cdc91f64919777b6, VALID, beta APPROVED, Public Beta assigned, internal/external IN_BETA_TESTING | [Signed upload](https://github.com/CrispStrobe/CrispMath/actions/runs/37230996262), [Apple read-back](https://github.com/CrispStrobe/CrispMath/actions/runs/37231506974), [TestFlight](https://testflight.apple.com/join/E6HdVhTx) |
| Public web | Pages reports 4f45017; Vercel reports 637b07e. The subsequent application change is a DEBUG-only Mac capture-window size, with release behavior unchanged | [Pages](https://crispstrobe.github.io/CrispMath/), [Pages verification](https://github.com/CrispStrobe/CrispMath/actions/runs/37240927177), [Vercel](https://crisp-math.vercel.app/), [Vercel verification](https://github.com/CrispStrobe/CrispMath/actions/runs/37230684265) |
| Mathematical audit | All 789 accumulated cases pass independently on Linux/native, WASM and packaged macOS. All 52 round-eleven references also pass actual desktop/phone UI checks | [Native/WASM audit](https://github.com/CrispStrobe/CrispMath/actions/runs/37228244814), [Mac audit](https://github.com/CrispStrobe/CrispMath/actions/runs/37228248552), [frozen references and initial failures](https://github.com/CrispStrobe/CrispMath/blob/main/docs/round11-math-audit-52.md) |
| Guided workflow and UX | Real menu creates a fresh a=3, f(x)=x²+a, f(4)=19 worksheet; 19→21→19, linked graph, checkpoints, downloads, reload and existing-document preservation pass release/debug on desktop, phone and tablet. Long math and enlarged-text result actions scroll | [Guided checks](https://github.com/CrispStrobe/CrispMath/blob/main/tool/check_guided_worksheet_browser.py), [main browser evidence](https://github.com/CrispStrobe/CrispMath/actions/runs/37240927201) |
| Cloud | Off without configuration; local work needs no account. Real browser profiles emit zero backend requests. Configured-cloud tests use a disposable backend, not a production project | [Cloud behavior](https://github.com/CrispStrobe/CrispMath/blob/main/docs/workspace-backups.md), [contract CI](https://github.com/CrispStrobe/CrispMath/actions/runs/37230602216) |
| Native screenshots | 33 populated captures pass: eleven each for iPhone, iPad and Mac. New Mac renders are 2560×1600; four selected Mac scenes are visually reviewed. Both iOS simulators pass first attempt, no recovery; sixteen tool controls pass | [Native run](https://github.com/CrispStrobe/CrispMath/actions/runs/37239843806), [selection/provenance](https://github.com/CrispStrobe/CrispMath/blob/main/docs/native-apple-gallery.md) |
| CrispEmbed app runtime | Published 0.17.12 at 78493a31fc3043f3af4ab6fd2f197a32da6f07f0 contains decoder input normalization and ceil-pooling fixes. Fourteen CPU/Apple/Android/WASM assets have verified release hashes; the app pins the release | [Release](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.12), [native assets](https://github.com/CrispStrobe/CrispEmbed/actions/runs/37222310628), [WASM assets](https://github.com/CrispStrobe/CrispEmbed/actions/runs/37222310650) |
| Browser OCR packaging | Hosted builds stage the checksummed published JS/WASM. A bounded upstream UTF-8 compatibility patch handles resizable buffers; downloaded/staged JS hashes are separate and WASM unchanged. Actual module/Unicode/heap checks pass | [Staging helper](https://github.com/CrispStrobe/CrispMath/blob/main/tool/stage_ocr_wasm_runtime.py), [model-free browser gate](https://github.com/CrispStrobe/CrispMath/blob/main/tool/check_ocr_wasm_runtime_browser.py) |

Earlier full app validation recorded 5,774 Dart passes/eight skips, 511 focused
feature passes, 19 guided/math passes, eleven cloud-widget passes and clean
analysis. These invocations overlap: do not sum them as unique tests. Prefer
the exact latest workflow reports when recording new totals.

## What remains open

- Physical-device testing and Apple's September 4.3(a) App Review rejection.
  External TestFlight approval does not settle that rejection. No new App
  Review submission or store-image upload has occurred.
- Handwriting quality: unchanged original benchmark 7/50; expanded-vocabulary
  candidate 0/50, despite 49/50 references being lexically representable.
  Corrections establish bounded forward correctness, not improved recognition.
  Original training-checkpoint/full-sequence parity remains unproved.
- CrispEmbed Python-wheel run 37222310629 failed on macOS arm64 during wheel
  repair. Linux x86_64/aarch64 and Windows builds passed; aggregation skipped.
  This is separate from successful app native/WASM release assets. The
  [bounded repairPR70](https://github.com/CrispStrobe/CrispEmbed/pull/70) aligns
  cibuildwheel with the existing12.0 native target and actually loads the
  installed bundled library. [Full wheel matrix39001056271](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39001056271)
  is queued at56b5af0c; repaired artifacts/installed checks remain unverified.
- The minimal runtime correction is published but its draft PR 60 remains open.
  Before this Markdown update, vendor code baseline 28d5a61 and release 78493a3
  diverge: main has 64
  distinct commits, release has seven. Do not merge unrelated work into the app
  dependency or claim vendor main already contains the release correction.
- No production cloud backend is selected. Provisioning one is optional future
  work, not a blocker for offline use or App Review demonstration.

## Execution and evidence rules for every lane

Keep each lane in an isolated change with one owner; coordinate edits to shared
fixtures, workflows, package pins and release metadata. Consult each repository's
PLAN and contributing instructions before changing code. Run builds, full suites,
corpus batches, browser batches, native captures and model benchmarks on hosted
CI. Use authorized GPU infrastructure for training. Keep large models/bundles
remote; fetch only the reports/images needed for review.

Freeze independent references before app outputs. Preserve initial failures and
unchanged historical fixtures. Exercise actual CLI/native/browser paths; do not
inject correct results or weaken comparisons to make a gate pass. Check full
workflow conclusions and explicit assertion reports, not merely step progression.
Distinguish numerical approximations from exact answers, component parity from
decoded quality, simulator checks from device checks, and beta from App Review.

Runtime work must follow the real inference blueprint and compare intermediate
values, magnitudes and final decoded outputs. Keep experimental/performance
paths gated; a speed improvement without equal output is not promotion evidence.
Read [CrispEmbed contribution requirements](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/contributing.md)
and [model policy](https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md).

Every finished lane delivers a bounded patch, meaningful positive/negative
regressions, exact source/workflow links, retained initial/final evidence and
updated status here and in both affected PLAN files. Never publish host paths,
credentials, private service references or operator contact details.

## Vendor status — 16 October 2026

Use [PLAN.md](../PLAN.md) for active tasks and ownership, [HISTORY.md](../HISTORY.md)
for completed work and frozen handoffs, and [PERFORMANCE.md](../PERFORMANCE.md)
for measured quality/timing. The earlier vendor milestones are now archived;
this status does not promote experimental vendor main into the app's pinned release.

LightOnOCR-3-0.8B has a native Qwen3.5 engine and public C/orchestrator/CLI/server/
Python/Rust/Dart wiring. The [external-document panel](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38674281196)
finds exact F16 outputs/stops on4 cases, but three exceed the prior0.002 cache-key
budget (worst0.00622). Q8 matches2/3 EOS-completed cases; Q8/Q6 matches1/3.
All receipt arms truncate at384 tokens. The completed
[larger-input controls](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38698415749)
show that all-F32 text and precise RMS do not remove strict failures: worst key
errors remain 0.00620–0.00625. Original CPU-backend variation also exceeds0.002;
no tolerance is relaxed. The completed
[operator audit](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38719205543)
reproduces the baseline exactly; both existing vision precision gates reduce20
prefill failures to11, while attention alone regresses to23. Same-input operators
mostly satisfy raw0.002. The completed
[native block audit](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38726666455)
passes678 exact controls: original block inputs reduce final-block error sharply,
but ordinary prefill failures remain20/11 and two earlier suffix resets regress.
The [completed native norm audit](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38764189561)
passes 225 exact controls and reproduces693 prior prefill records. Actual computed
cascade outputs match75 original-input norms exactly; split variance rounding
resolves the earlier validation-code mismatch and compiler output confirms it.
Prefill failures remain20/11/14 for baseline/cascade/cascade-attention; the older
combined F64 arm also had 11, with different trade-offs. Three small-document
ID/text/EOS arms pass, but no large-document quality or default promotion follows.
The [four-arm block replay](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38775800804)
passes 1,356 exact controls and preserves 924 prior prefill records. Both cascade
arms make all 36 original-input norm1 outputs exact, but local block11 errors
remain and cascade's early suffix resets still fail twice. All four small-fixture
ID/text/EOS arms pass; the sweep retains 56 prefill and four suffix failures.
The [completed attention trace](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38800963470)
passes all 72 native equality controls and preserves924 prefill, 108 shared vision
and 1,302 original CPU-control records. Two compiler-rounding differences in the
validation copy are corrected; production kernels remain unchanged. All 840
conditional vectors pass raw 0.002. Original-input F32 sampled QK scores and
original-probability pooling are exact; softmax probability errors reach 1.88e-6.
The sweep retains 56 prefill failures and passes four small-fixture ID/text/EOS
arms. The [completed decoded panel](../tests/results/qwen35-ocr/decoded-panel.json)
now matches all 16 fixed-cap ID/text/stop comparisons and 2,216 teacher argmaxes
across all four arithmetic settings. Each arm reaches EOS on three inputs;
receipt outputs remain 384-token prefixes. Confidence error is at most 3.28822e-6,
all 924 prefill and 1,302 original CPU-control records remain exact, and all 56
prefill failures remain. The first run times out after 14 cases; a bounded
[completion run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38830295916)
finishes only the missing two, reproducing original references/confidence/CPU
controls exactly. Reference parity does not establish transcription accuracy;
existing form/receipt gold errors remain. Native baseline receipt recognition
takes 585.82 s in that single observation. The completed
[profiling audit](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38850631263)
at `6895c69a` reproduces all prior controls and 56 failures, with zero output or
reproduction failures. All 18 paired runs preserve IDs/text/stops and native
probabilities. F32 projection materialization consumes 62–88% of profiled wall time;
head materialization alone 28–40%. Unprofiled medians are 225.81s form, 38.65s German,
67.33s for a 32-token receipt prefix. Counter overhead is 4.46–9.79%; peak child RSS
is 3723.38–3725.35 MiB including load; those nested timings measure attribution.
The completed [parallel-F16 trial](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38860469683)
at `1e75f698` verifies all 473 converted tensors/852,985,920 F32 values bitwise
against NumPy and native serial values, with zero new output/reproduction failures.
All 924 prefill and 1,302 CPU-control records and four baseline full frozen-cap
outputs reproduce. Unprofiled paired speedups are 1.67× form, 1.37× German and 1.41×
for a 32-token receipt prefix; medians fall 225.89→135.21s, 39.16→28.50s and 68.01→48.09s.
Peak child RSS stays around 3725.36 MiB including load; incremental inference memory
is not isolated. All 37 Python and Linux/macOS/Windows native controls pass;
all 56 numerical failures and 384-token receipt truncation remain. The conversion
gate stays default-off. The completed
[receipt/prompt audit](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38870347237)
at `a07e83fb` reaches EOS at455/434 tokens (image-only/literal), with exact original
Python IDs/text/stops and all889 teacher-score vectors passing unchanged gates.
Both outputs still contain HTML tables. Lower raw CER is a serialization change,
not proof of improved recognition. All308 earlier receipt prefill records reproduce
with27 failures; the literal prompt adds77 records/seven failures. This workflow
correctly fails for34 prefill failures, with zero incomplete/output/teacher-score
failures. Other616 historical prefill records were not rerun; full-panel56 failures
remain retained. All46 Python and Linux/macOS/Windows native controls pass
[38870034506](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38870034506).
The [v1 independent quality contract](ocr-quality.md) now freezes two real-document
transcripts and an owned table before model output, with raw CER/WER, ordered
content and position-indexed cell comparisons. [Completed x86 preparation](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38879605915)
and [independent ARM preparation](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38879334940)
produce identical PNG/RGB/font/source/gold/split hashes. All three images are
visually reviewed; [the retained report](../tests/results/ocr-quality-prepared.json)
keeps the receipt damage/clipped footer and small book text unchanged. Unavailable
table annotations stay unmeasured. The guarded original/F16 driver requires those
reviewed hashes before model imports/downloads. All22 hosted controls and frozen
image/scorer/hash replay pass [38882346793](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38882346793); the completed
[F16 audit](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38882488689) at
`4c4b3ee4` reaches all3 EOS outputs with exact1,404 IDs/text/stops/teacher argmaxes.
Inventory ordered content and15/15 cells are exact; receipt/book retain recognition
errors shared with Python. The workflow correctly fails for6 new prefill and13
book-logit failures, worst0.0060377121 against unchanged0.002. Zero incomplete,
output or quality-parser failures. [Full report](../tests/results/qwen35-ocr/quality.json),
[raw/secondary scores and timing limits](../PERFORMANCE.md). Archived924/56
prefill debt was not rerun or waived. The completed
[book continuation isolation](../tests/results/qwen35-ocr/book-isolation.json) at
`4323dc67` reproduces the composed baseline and removes all book prefill/score/
selected-probe failures with original prefill embeddings. Worst isolated score
error0.00012421608; all683 argmaxes match. This identifies an upstream boundary,
not a production correction. All30 hosted controls, frozen replay and three-OS
native controls pass. Bounded CI drift is now repaired in
[PR61](https://github.com/CrispStrobe/CrispEmbed/pull/61): all six affected artifacts
independently byte-verified, previous digests retained, strict304/304 freshness
replay and all PR build/regression/WASM/packaging checks pass.
[Verification](evidence/registry-pin-verified.json),
[complete freshness](evidence/registry-pin-current.json). Quantized quality runs,
larger pages, further materialization/default adoption and HF publication remain
open;1024 patches is the native bound.
[Detailed evidence](qwen35-ocr.md). The completed
[book vision trace](../tests/results/qwen35-ocr/book-vision.json) ataccd903c/
[run38917305082](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38917305082)
passes24 exact native controls, reproduces434 original CPU/77 baseline prefill
records and independently replays all12 logs. Existing settings retain5/6/5/8
prefill failures despite improved merger errors; original-input block11 down
error0.00268554688 remains and no single causal defect is proved. All280 conditional
attention vectors pass raw0.002. All34 source controls and frozen preparation
replay pass; all ten PR62 Build jobs (including Flutter), WASM, contract and
secret checks now pass at17b4b7dc. [Platform evidence38922363407](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38922363407).
PR63's complete ten-job Build (including Flutter), WASM, contract and secret checks
now pass; [platform38933245371](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38933245371).
Latest PR67/merged-main broader checks remain pending. Continue independent-document
quantized evaluation while retaining projection/reduction attribution as open
numerical debt. No runtime/default/weight/release/app-pin promotion.
The first [independent Q8 inventory audit](../tests/results/qwen35-ocr/quality-quant-q8-inventory-table.json)
atcbd3852f/[run38924337127](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38924337127)
reproduces original/F16 controls and all quantized file/protected tensor identities.
Q8 preserves248 IDs/text/EOS,15/15 cells and zero ordered CER/WER, but retains75
prefill/248 score failures (min cosine0.999256611, worst abs0.440546274). All41
source controls, frozen preparation and source secret scan pass; all five logs
independently reparse. PR63 merges asd34a64d2. The completed
[Q8 receipt](../tests/results/qwen35-ocr/quality-quant-q8-historical-receipt.json)
atfdf83339/[run38928823914](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38928823914)
reproduces original/F16/weight controls and all raw logs, but changes8PK→SPK
(first token divergence303); content CER/WER worsen3.39%/13.19%→3.52%/13.89%.
Q8 retains75/473 prefill/score failures versus1/0 for F16. Both reach EOS.
The completed [Q8 book](../tests/results/qwen35-ocr/quality-quant-q8-two-column-page.json)
ata42a28b7/[run38931861995](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38931861995)
reproduces original/F16/weight controls and all raw logs. Q8 reaches EOS with684
versus683 tokens, first divergence130; content CER/WER decrease slightly
11.47%/17.63%→11.20%/17.45%, while text changes and recognition errors remain.
Q8 retains75/683 prefill/score failures versus F16's5/13. Q8 panel:3/3 EOS,1/3
original ID/text matches; retain225 prefill/1,404 score failures. Three preselected
mixed runs usef43c4b77. The completed
[mixed inventory](../tests/results/qwen35-ocr/quality-quant-q8-q6-inventory-table.json)
[run38942897105](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38942897105)
reproduces all original/F16/weight controls and raw logs, preserving248 IDs/text/EOS
and15/15 cells. It retains75/248 prefill/score failures, with worse score fidelity
than Q8 (min cosine0.998120904, worst abs1.08967495). The completed
[mixed receipt](../tests/results/qwen35-ocr/quality-quant-q8-q6-historical-receipt.json)
[run38942903455](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38942903455)
reproduces all controls/logs, but changes8PK→SPK and drops a bullet, reaching EOS
with471 versus473 tokens. Content CER decreases3.39%→3.26%; WER stays13.19%.
Mixed retains75/473 prefill/score failures (min cosine0.997738481, worst abs1.08097315).
The completed [mixed book](../tests/results/qwen35-ocr/quality-quant-q8-q6-two-column-page.json)
[run38942909059](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38942909059)
reproduces all original/F16/weight controls and raw logs. Mixed reaches685 EOS
tokens versus683, first divergence130; content CER rises11.47%→11.51% while WER
falls17.63%→17.45%. It retains75/683 prefill/score failures (min cosine0.998528719,
worst abs1.22282577). Mixed panel:3/3 EOS,1/3 exact original IDs/text,225/1,404
numerical failures; score fidelity is worse than Q8 on every input. Q4 [inventory](../tests/results/qwen35-ocr/quality-quant-q4-inventory-table.json),
[receipt](../tests/results/qwen35-ocr/quality-quant-q4-historical-receipt.json) and
[book](../tests/results/qwen35-ocr/quality-quant-q4-two-column-page.json) at8bf55424
are complete measurements. All original/F16/weight records and five logs each verify.
Inventory preserves248 IDs/text/EOS and15/15 cells. Receipt diverges at25, changes
to HTML and truncates at1024 without EOS; unclosed HTML leaves content/cells unscored.
Book reaches693 EOS tokens versus683, diverges at26, has equal content CER11.47%
and worse WER17.63%→17.81%. Q4 retains225 prefill/1,404 score failures, minimum
score cosine0.928607225 and worst abs7.40867901; gates/caps remain unchanged.

The full nine-run panel uses three unique inputs. Q8/mixed reach3/3 EOS, Q4 reaches
2/3; all three quantized recipes match original IDs/text only on inventory. F16
matches all originals but retains6/13 numerical failures. [Complete comparison](../PERFORMANCE.md)
and [source/report matrix](../HISTORY.md). The public model-free auditor preserves
all failures when records verify; all50 hosted controls pass at5dbabaa1/
[38957526079](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38957526079).
No candidate is promoted or published, and the app stays on immutable0.17.12.
UltraBERT is the next independent implementation lane; OCR numerical attribution,
broader quality and validated publication remain pending. Newer platform CI
conclusions remain separate from the exact completed source controls above.

The iOS standalone-target repair now passes the complete
[ten-job main platform workflow](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38697864752),
including Flutter bundling. All 38 bundled native library entries match the seven
platform artifacts by size/ZIP CRC. The scoped 18-case position probe and initial
failures remain in HISTORY. This does not change the app's runtime pin or establish
installed-plugin/device inference validation.

The [same-input decoder experiment](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38702930399)
removes all 20 strict stage failures when original embeddings are supplied only
to diagnostic decoder arms; worst F16 key error becomes 0.0000406504. The composed
baseline exactly reproduces its prior failures. The subsequent fixed-cap decoded
A/B now passes on the four-document panel while retaining numerical debt. Use the
faithful attention trace for any new divergence; the receipt now completes, while
broader accuracy and performance remain open.

LFM2.5 Encoder and EmbeddingGemma typed-decision integration/publication are
archived as complete, with GPU, independent task-quality and long-context/perf
follow-ups in PLAN. [UltraBERT](ultrabert.md) has a pinned source/metadata feasibility
audit and a proposed port order. [PR69](https://github.com/CrispStrobe/CrispEmbed/pull/69)
adds strict conversion and all24-layer original-Python reference captures;
[hosted build](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38972505497)
and24 source/log/operator-parser controls pass at a8b129f4. The
[complete original panel38997562520](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38997562520)
freezes all seven cases/2,730 stages and both218-tensor conversions after repairing
a negative-control inference-tensor mutation. The
[first native panel38997900303](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38997900303)
completes all14 arms and700 exact controls. F32 final CLS cosines round1.0, worst
raw error0.000067234;183 intermediate failures first exceed budget at layer14
FFN output. F16 retains1,294 failures and worse raw CLS errors. All measurements
and unchanged0.99999/0.002 failures remain in PR69; no candidate is promoted.
The [original-input operator audit38998432137](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38998432137)
completes at a8b129f4 with all ordinary metrics/exits reproduced exactly. All28
logs/84 local records verify: F32 passes41/42 probes and F16 retains24 failures.
Unicode whole-MLP error0.0048828125 falls to0.000671386719 with original gate/up;
local norm probes pass. Next capture original SiLU/product/down inputs and inspect
projection rounding before selecting a correction; preserve all initial debt.
The [isolated linear audit39000683936](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39000683936)
is queued at c7c5ef30. It adds separate original-module SiLU/product observations
with bit-exact old gate/up/output anchors, then isolates native down projection;
new controls, compilation and measurements are pending. The requested
[three-PR review](pr-review-2026-10-17.md) is complete; PR68 fork CI is now approved
and queued, PR60 wheel11/12 target inconsistency remains, and PR69 stays draft. Native tokenization, production backend,
public interfaces, task retrieval, quantized quality and publication remain open.
The independently scoped [wheel repairPR70](https://github.com/CrispStrobe/CrispEmbed/pull/70)
sets cibuildwheel to the existing macOS12.0 native floor and verifies actual
installed native loading/C-ABI use. [Four-platform matrix39001056271](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39001056271)
is queued at56b5af0c; wheel repair/aggregation is not yet verified. No release or
app pin change accompanies this patch.
Indra remains public-code analysis only under the selected permissive-model scope.

The independent-document quantization panel is complete. No tested quantized
recipe qualifies for promotion from this panel; publication remains conditional
on a validated candidate. UltraBERT is the next independent implementation lane,
then the permissive OCR queue. Projection/reduction attribution remains open;
preserve all numerical failures and frozen quality inputs. The consumer lanes
below remain separate, with their original verification dates.

## Recommended order and dependencies

Start W1 (wheel repair) and V1 (vendor reconciliation) as separate changes.
M1 (fresh mathematics) and U1 (UX coverage) can proceed independently on hosted
runners. O1 can extend diagnostics now; O2 waits for checkpoint/license/access
requirements. P1 waits for a device session; A1 follows P1 and final metadata
review. R1 is used only when a production app change is ready to ship. C1 and
F1 require an explicit backend/product decision. B1 is an optional bounded
math-engine investigation, not a reason to rebuild everything immediately.

### W1 — Repair macOS arm64 Python-wheel packaging

**Repository:** CrispEmbed. **Ready now.**

First inspect the exact [failed wheel run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/37222310629).
The failure is in delocate repair and reports staged dylibs with minimum target
12.0; the log asks to align MACOSX_DEPLOYMENT_TARGET. Treat this as a packaging
lead, not a proven complete fix.

Work in the [wheel workflow](https://github.com/CrispStrobe/CrispEmbed/blob/main/.github/workflows/python-wheels.yml)
and Python build/package configuration. Compare CMake/ggml deployment settings,
wheel tags, cibuildwheel environment and every bundled dylib's minimum OS target.
Reproduce one failing Python version first, align the supported minimum honestly,
then run the complete wheel matrix. Do not simply change a filename or suppress
repair errors. Test an installed wheel's actual library load and an available
small public API operation on the declared platform.

**Done when:** Mac wheels repair/load successfully, other three platform jobs
remain green and aggregation succeeds, with source/version/artifact provenance.
The workflow produces preview artifacts; it does not currently publish to PyPI.
Do not overwrite published app archives or claim PyPI publication.

### V1 — Reconcile the released hotfix with vendor main

**Repository:** CrispEmbed. **Ready now; keep separate from W1.**

Start with [draft PR 60](https://github.com/CrispStrobe/CrispEmbed/pull/60)
and the [release-to-main comparison](https://github.com/CrispStrobe/CrispEmbed/compare/78493a31fc3043f3af4ab6fd2f197a32da6f07f0...28d5a61ce45af6a3a547802e82b8e553c6fe3d5d).
Inventory the seven release-side commits and identify the two runtime repairs,
their tests, release/provenance work and any already-equivalent main changes.
Port only necessary fixes to current main and resolve conflicts against the
actual [PosFormer implementation](https://github.com/CrispStrobe/CrispEmbed/blob/main/src/posformer_ocr.cpp).
Preserve independent concurrent vendor features. Follow the vendor version-bump
procedure for an actual new release; do not move the existing release tag.

Re-run synthetic positive/negative normalization/pooling controls, independent
reference comparisons and actual normal-API Linux/Mac transcripts on the frozen
fifty images. Inspect packaged libraries for unintended diagnostic payloads.
Reconcile the draft only after equivalent behavior is evidenced on current main.

**Done when:** main contains the measured correction with reproducible evidence
and the draft status explains its disposition. Keep CrispMath on immutable
0.17.12 until a separately validated successor release exists.

### M1 — Fifty new independent mathematical problems and systemic repairs

**Repository:** CrispMath. **Ready now.**

First read the [round-eleven reference/failure record](https://github.com/CrispStrobe/CrispMath/blob/main/docs/round11-math-audit-52.md)
and prior corpus only to avoid semantic duplicates. Draft fifty fresh problems
with independently derived full answers, domains, multiplicities, dimensions
and rounding tolerances before running the app. Cover symbolic branches/domain
holes, calculus endpoints, matrices/systems, statistics tails, composite units
and complete constraint solutions. Include failures and unsupported boundaries.

Freeze a round-twelve reference document and fixtures in Git. Extend the actual
[CLI audit](https://github.com/CrispStrobe/CrispMath/blob/main/tool/crispmath_cli.dart),
[runtime workflow](https://github.com/CrispStrobe/CrispMath/blob/main/.github/workflows/workflow-task-audit.yml),
[packaged Mac audit](https://github.com/CrispStrobe/CrispMath/blob/main/.github/workflows/build-macos.yml)
and [strict UI checker](https://github.com/CrispStrobe/CrispMath/blob/main/tool/check_round11_math_browser.py)
for the new fixtures. Register them in deployment checks as well; a fixture
unreferenced by a workflow is not coverage. Keep all 789 old cases unchanged.

Preserve the first outputs; trace each discrepancy through parser, dispatcher,
engine, numerical method and presentation before patching the responsible layer.
Add a negative control for every checker change and a real UI regression for
each entry-point/presentation gap. Require complete sets/matrix cells, genuine
factor structure and visible precision rather than text fragments.

**Done when:** all 839 accumulated cases have explicit native/WASM/Mac outcomes,
every new question has an actual UI check or a documented limitation, required
units/analysis pass, and Pages/Vercel Playwright checks prove deployed behavior.
Report genuine unsupported tasks as open; do not silently drop or redefine them.

### U1 — Accessibility, localization and narrow-layout coverage

**Repository:** CrispMath. **Ready now; coordinate shared UI edits with M1.**

Begin from the [guided checks](https://github.com/CrispStrobe/CrispMath/blob/main/tool/check_guided_worksheet_browser.py),
[native gallery test](https://github.com/CrispStrobe/CrispMath/blob/main/integration_test/product_gallery_test.dart)
and result presentation regressions. Audit real graph controls, photo/ink import
dialogs, variable/function editing and export dialogs on phone/tablet/desktop
at normal and enlarged text, and in English/German/French/Spanish. An off-screen
item in an intentional scroll region is not by itself an overflow defect.

Record failures before editing. Fix shared layout/action helpers where several
flows share a cause. Preserve complete accessible values and actual clipboard
payloads. If adding a graph scroll affordance, prove it exposes real trace/table
commands without covering the plot. Use stable real clicks and rendered geometry,
not forced clicks, retries masking errors or injected state.

**Done when:** observed defects have targeted widget controls and real release/
debug browser checks, no framework/page errors, readable populated screenshots
and reachable final actions. Physical VoiceOver verification belongs to P1.

### O1 — Complete the handwriting diagnosis before choosing new weights

**Repositories:** CrispEmbed runtime; CrispMath benchmark/consumer. **Partly ready now.**

Read the [measured findings](https://github.com/CrispStrobe/CrispMath/blob/main/docs/handwriting-quality-findings.md)
and [independent reference run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/37221078035).
Preserve the frozen fifty references, rasterization, image/model hashes and
token scoring. Bounded exported-FP32 comparisons account for 270 tensors,
45 encoder and 275 decoder stage comparisons per arm. They do not prove the
original training checkpoint or unrestricted autoregressive decode.

Inspect [conversion](https://github.com/CrispStrobe/CrispEmbed/blob/main/models/convert-posformer-to-gguf.py),
[runtime](https://github.com/CrispStrobe/CrispEmbed/blob/main/src/posformer_ocr.cpp)
and [app reference harness](https://github.com/CrispStrobe/CrispMath/blob/main/tool/handwriting_reference_parity.py).
Extend bounded coverage through preprocessing, positional conventions, masks,
token IDs, EOS/length handling and complete decoded outputs. Compare magnitudes
as well as cosine; localize the first actual divergence. Keep probe code out
of production artifacts. Separate FP32 structural parity from Q8 drift.

**Done when:** a pinned full-sequence reference and runtime agree on the same
inputs or the first remaining divergence is reproduced with a regression.
Original-checkpoint checks require legitimately accessible weights. If unavailable,
finish available exported-weight diagnostics and record that dependency instead
of repeating blind encoder changes or claiming recognition is fixed.

### O2 — Obtain and evaluate suitable licensed recognition weights

**Repositories:** CrispEmbed conversion/training; CrispMath consumer. **Conditional.**

Depends on O1's inference contract and legitimate checkpoint/dataset access.
Start by verifying the real checkpoint architecture, vocabulary and license
chain; a public GGUF or wider token dictionary alone is not proof of provenance
or quality. Use the existing [training tools](https://github.com/CrispStrobe/CrispEmbed/tree/main/tools/kaggle/posformer-train)
only after confirming they match the verified inference path and authorized
remote resources. Split training/evaluation data without reusing the frozen
fifty examples for training or parameter selection.

Produce pinned conversion/reference artifacts and compare decoded results with
the unchanged baseline plus a separate held-out corpus. Record exact matches,
error categories, invalid/empty output, latency/memory and numeric handoff.
Require a predeclared quality threshold; never substitute benchmark success
for user review. Do not add the current 0/50 candidate to the default catalog.

**Done when:** provenance/licensing, reference parity and held-out quality justify
a separately reviewed model promotion, with opt-in/rollback controls and no
unverified reliable-recognition claim. No training has been authorized by this
handoff document itself.

### P1 — Physical Apple device session

**Repository:** CrispMath. **Waiting for the deferred device session.**

Use approved build 23 from [TestFlight](https://testflight.apple.com/join/E6HdVhTx)
and follow the [device matrix](https://github.com/CrispStrobe/CrispMath/blob/main/docs/apple-review-evidence.md#later-physical-device-session).
Record device model, OS, install/update route and actual build; exercise offline
guided 19→21→19, graph trace/table, background/relaunch/rotation, Files round trips,
photo/camera cancellation and permission behavior, enlarged text/VoiceOver,
native URL/shortcut behavior, and supported Pencil/multitasking.

**Done when:** both device categories have recorded results and first-failure
reproductions, with fixes re-tested on the affected device. Simulator/Python CI
must not be entered as a physical pass. Ink/import tests do not prove recognition.

### A1 — Finish store presentation and resolve App Review 4.3(a)

**Repository:** CrispMath. **Drafting is ready; submission is deferred.**

Start with the [review packet](https://github.com/CrispStrobe/CrispMath/blob/main/docs/apple-review-evidence.md)
and [native selection](https://github.com/CrispStrobe/CrispMath/blob/main/docs/native-apple-gallery.md).
Keep the connected parameter→function→graph→checkpoint→export narrative. Select
actual populated images with original source manifests and supported slots;
Mac files must use listed 16:10 dimensions. Check every description against the
current binary, default-off cloud behavior and experimental handwriting limits.
Remove unsupported feature/reliability claims from any proposed store text.

After P1, assemble the exact screenshots, description and review response for
review. Explain concrete distinct workflow/content; do not allege an automated
similarity check or use source-line/test counts as proof of distinctiveness.
The next agent may prepare drafts; sending correspondence, uploading store
metadata/images and a new App Review submission need the user's explicit scope.

**Done when:** the packet matches a tested binary and verified device/screenshots;
after an authorized submission, record Apple's actual response. Do not mark the
4.3(a) rejection resolved merely because the beta is approved.

### R1 — Promote the next production app change through all release gates

**Repository:** CrispMath; coordinate a dependency change with CrispEmbed. **Conditional.**

Use only after a production change from another lane is ready. Inspect the
[release workflow](https://github.com/CrispStrobe/CrispMath/blob/main/.github/workflows/ios-release.yml)
and [validation/source guard](https://github.com/CrispStrobe/CrispMath/blob/main/tool/wait_release_validation.py).
Bump the build number for a new Apple binary; never reuse23 after its upload.
Pin a vendor successor's source/version and published native/WASM hashes
atomically, preserving the UTF-8 staging provenance and model-free module gate.

Require appropriate units/analysis, accumulated CLI and actual UI audits,
release/debug browser checks, supported native screenshots, packaged metadata
and deployed Pages/Vercel evidence on the real candidate. Validate privacy
purpose strings inside the signed app. Upload once, wait for actual Apple VALID
processing, then perform metadata-only internal/external assignment checks.

**Done when:** exact-source gates pass, deployed fingerprints are recorded and
the authorized new beta has read-back evidence. Do not rebuild/reupload for a
retry of metadata assignment, for documentation, or the DEBUG-only capture fix.
App Review remains a separate action in A1.

### C1 — Optional configured-cloud deployment

**Repository:** CrispMath. **Waiting for an explicit backend choice.**

Keep cloud off by default. Do not create a project just to make an optional
checklist green. If cloud is selected, use the [workspace contract](https://github.com/CrispStrobe/CrispMath/blob/main/docs/workspace-backups.md)
and [disposable CI contract](https://github.com/CrispStrobe/CrispMath/blob/main/.github/workflows/sync-backend-contract.yml)
as the starting point. Receive configuration through a secure mechanism;
publish no service-role keys, credentials, sessions or operator identities.

Verify migration/RLS with two independent users, then real authenticated
devices/browsers with a local worksheet evaluating to 21 and a separately
imported/cloud worksheet evaluating to 23, preserving both conflicting versions,
reload/offline recovery, safe sign-out, cancellation, cleanup and backup round
trips. Re-run fresh unconfigured profiles to ensure zero backend requests.

**Done when:** real chosen-backend/two-device evidence exists without losing
local work; otherwise record this lane as intentionally unconfigured, not broken.

### B1 — Optional math-stack capabilities and plotting performance

**Repositories:** CrispMath and its publicly declared math-stack dependency.
**Investigation ready; implementation needs a bounded demonstrated use case.**

Start with [CrispMath PLAN](https://github.com/CrispStrobe/CrispMath/blob/main/PLAN.md)
and [dependency declarations](https://github.com/CrispStrobe/CrispMath/blob/main/pubspec.yaml).
Measure a real repeated graph-evaluation or rigorous-numerics workload before
proposing LLVM/ARB changes. ECM/PRIMESIEVE are conditional on an observed large
factorization limitation. Follow the dependency's own public build instructions;
do not guess its repository or enable all optional libraries together.

Change one capability at a time, preserve fallback behavior and compare the
same results/domains, latency, memory, startup and binary size across native
and WASM. Run hosted build/package/CLI/UI controls and document unsupported
platforms. A faster benchmark with changed output is not a successful default.

**Done when:** a bounded use case has a measured improvement and passing parity,
or a reproducible finding explains why the capability remains disabled.

### F1 — Future collaboration/product expansion

**Repository:** CrispMath. **Needs a separate product/backend decision.**

The older plan's multi-user editing proposal is not part of the shipped app.
First define the collaboration unit, offline ownership, conflict policy and
account/privacy model. Preserve portable local worksheets and source-only
imports; do not turn the default-off backup service into mandatory login.
Build a small disposable-backend two-editor prototype before proposing a
production service, with deterministic concurrent-edit and recovery tests.

**Done when:** the prototype has reviewed semantics and real two-client tests;
a production rollout has its own authorized scope, migration and device gates.
Do not treat this longer-term idea as an unfinished build 23 feature.
