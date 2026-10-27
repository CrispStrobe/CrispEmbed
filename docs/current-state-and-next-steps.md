# CrispMath and CrispEmbed: current state and executable lanes

Consumer snapshot verified5 October2026; vendor roadmap updated26 October2026.
For current vendor work, start with the [executable handoff](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/next-agent-handoff.md).
The application snapshot below keeps its original verification dates. Read the status, choose one lane, and record
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
| Python wheel preview | Four platforms/23 repaired-wheel installs/native loads/C-ABI checks and aggregation pass; no PyPI/runtime release | [39003273845](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39003273845), [full matrix evidence](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/wheel-matrix.json) |
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
- The selective PosFormer source/default restoration is merged via PR72/PR73;
  historical PR60 is closed. Linux published0.17.12 and the current default match
  all50 frozen IDs/text/EOS. A successor release still needs actual portable
  artifacts/hashes and consumer qualification; the app remains pinned0.17.12.
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

## Vendor status — updated26 October2026

[Executable next-agent tasks](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/next-agent-handoff.md) give exact sources, tool paths,
corrected audit, artifact expiry and acceptance checks. [PLAN](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md) contains
pending work, [HISTORY](https://github.com/CrispStrobe/CrispEmbed/blob/main/HISTORY.md) completed milestones and every initial
failure, [PERFORMANCE](https://github.com/CrispStrobe/CrispEmbed/blob/main/PERFORMANCE.md) numerical/quality/storage limits.
Completed wheel/source changes and test-only UltraBERT tools are integrated;
merged-source platform CI and Pages/live metadata verification pass (HISTORY23 October).
The app remains pinned to immutable0.17.12, not vendor main.

| Front | Verified result | Remaining work |
| --- | --- | --- |
| Wheels | All4 platform legs/23 installed native-load/C-ABI checks/aggregation pass at e75a32f1 | Version/release reconciliation and any future PyPI publication are separate |
| Social preview | PR68 merged after native/Flutter/WASM/secret success | Completed deployment39009694992 at f6f84aba; live HTML/PNG verified23 October |
| UltraBERT source/conversion | All218 tensors in both F32/F16 exports and original7-case/2,730-stage capture pass | Gated CPU/F32 text dispatch mergedPR74; PR75 Rust/Dart wrapper integration merged2f078406 after actual39979622930; pairs, numerical attribution and release gates remain |
| UltraBERT tokenizer | All8 native BPE/test-template rows match original IDs/masks; both wrong alternatives fail8/8 | Scoped public F32 IDs/masks pass; pair/reserved-token profiles remain separate |
| UltraBERT numerical | Initial14-arm panel/700 exact controls complete,183 F32/1,294 F16 numerical failures retained | [Corrected local audit39009703362](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39009703362) complete: all21 F32 local probes pass; six F16 down failures; reference/projection attribution and both strict replay failures remain |
| OCR quantization | Original/F16 plus all9 independent Q8/mixed/Q4 audits and auditor controls complete | No tested recipe qualifies; retained numerical/recognition/format/cap debt blocks promotion |
| LFM2.5 Encoder | Official F16 CPU/CUDA T4 audit passes594 direct checks and both15-case ordinary API panels; [full report](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/lfm2-encoder/cuda-f16.json). Paired v2 performance passes independent recipe/placement/hash/420-sample review; all15 cases faster on CUDA in one T4 session | Pooled9/20/649-token medians27.93x/27.64x/120.11x are session-scoped; other GPU/precision/MMQ, throughput, long-context and task quality remain open; GPU default unchanged |
| Typed Gemma | Integration/publication completed in its model document | Independent task quality, GPU and performance remain open |
| Permissive OCR queue | OvisOCR2 export/frontend/original GPU pass; reviewed [native40049244654](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-model-audit.json) passes14/14 F32/F16 arms on943 frozen fields, [public40053283383](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-public-surfaces.json) passes27 C/Python/CLI/HTTP/Rust/Dart arms, [original HF eleven-case quality40054043629](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-original-quality.json) is frozen, and [first native F16 quality40056641800](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-f16-quality.json) has a positive independent eleven-case review | [Protected Q8 40061433357](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-q8-quality.json) is complete and rejected for numerical and two-column text regression. All four quantization attempts are rejected; qualified F16 is selected. F16 publication40074370335 and immutable readback pass; real precision alias is wired. Published CLI/HTTP/offline consumer40080458492 passes; PR79 and final platform/Flutter/WASM/style/security checks are integrated. The bounded cache experiment is complete and remains default-off: only0.797% median warm gain with increased RSS. First-FFN attribution reproduces the rejected Q4 error in original Python; no kernel mismatch or new precision is selected. Initial12/14 vision failures and four strict0.002 exceedances remain recorded; F16 publication and actual source alias consumer are verified. Indra public-code-only |
| PosFormer/app | PR73 merged75e87ea6 after all source790f9a6b checks; unset/1 trained,0 legacy. Actual Linux released/default50-case outputs and bounded references pass | Successor native/WASM artifacts/hashes and consumer qualification remain separate; app stays0.17.12 |
| Development desktop bundles | [PR76](https://github.com/CrispStrobe/CrispEmbed/pull/76) merged1f8ff4d5; [39979940916](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39979940916) passes Linux/macOS/Windows CPU-baseline configure/build/smoke/install | [Flag/log review](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/development-cpu-baseline.json) is scoped to development bundles; successor release assets and cross-host consumer checks remain |

UltraBERT original arrays were verified and copied by preservation39913800444;
the remote copy expires21 January2027. Original producer/manifest/payload
identities remain unchanged. [Preservation provenance](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/ultrabert/reference-preservation.json).
Numerical replay drift and cause remain unresolved. Component cosine, exact
tokenizer IDs or local operator agreement do not establish full model/task quality.
Default-off CPU/F32 public text support merged viaPR74 at3f3d023a after complete
source4be9f864 platform/Flutter/WASM/style/security checks. Hosted39943940607
passes actual C API/Python/CLI/HTTP and all twelve pooled/seven text-row checks.
[API evidence](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/ultrabert/public-f32-api.json). Original-first
retrieval39950831201 at812e56e5/native4be9f864 passes all32 IDs/64 vectors and
8/8 exact top1; original/native Recall@1/MRR@10/NDCG@10 all1.0. Raw max7.34e-5,
normalized5.51e-7, minimum cosine0.999999999999019. [Full smoke evidence](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/ultrabert/retrieval-smoke.json).
This authored panel does not establish representative MTEB/BEIR quality.
Actual Rust/Dart39979622930 passes all seven rows/42 pooled checks; PR75 merged
at2f078406. [Reviewed binding evidence](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/ultrabert/public-bindings.json)
retains raw maxima6.723403930664062e-5/5.941838026046753e-7/6.07222318649292e-7
for raw/dense/normalized-token CLS, both languages, and all14 binding ID rows.
Public pairs, long-context/GPU/performance and validated
quantization/publication remain U3/U4. Initial harness failures and183/1294
intermediate failures are retained. No new runtime tag/model alias/app pin.
The [M1 native handover](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/handover-m1-ultrabert-f32-api.md) supplies
complementary Apple coverage, with no repeated Python audit. Original-backend reproducibility39915714794 completes all four controls without
recovering frozen anchors. The [M1 follow-up](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/handover-m1-validation.md) is complete:
raw/normalized CLS max6.52e-5/6.41e-7, long hidden0.00579. Independent review
checks all returned stage identities and same-input hashes. Preserve the full audit
and scope F32 pooled-output/API/retrieval qualification separately; no repeated arms.
Selective PosFormer [PR72](https://github.com/CrispStrobe/CrispEmbed/pull/72) is
merged at7f1ecd33 after full sourcef7c53803 platform/Flutter/style/security and
bounded original/exported-weight reference success. PR60 is closed. Development
manifests0.17.13 are not a release or app upgrade. The new main integration runs
are separate checks. Released-binary compatibility is measured; PR73 default restoration merged75e87ea6 after full exact-source CI. Atomic
release/consumer qualification remains open.
Other portfolio/app owners remain separate.

## 26 October executing closure

The following reviewed receipts and running checks extend the dated snapshots
above. They create no runtime release, model promotion or application-pin change.

| Lane | Actual evidence | Remaining gate |
| --- | --- | --- |
| Ovis2048 | Norm40286291355 excludes a local RMS defect; source-order position40328446801 matches all five operator grids. Both quality40334662677 arms retain1841 IDs/text/EOS; off confidences exact, on maximum drift0.000024796. | Corrected four-grid composed40376882009 at8bb9f074 has both frozen and1024-square controls14/14 per gate;1200-landscape and2048-boundary pairs remain pending after four retained pre-inference path-redaction failures and two unexecuted cancellations. Keep the alternative default-off and2048 unqualified. |
| MI-GAN256 | [PR85](https://github.com/CrispStrobe/CrispEmbed/pull/85) merged at1b74b12a after29 successful checks. Original40330674649 and corrected40344396533 pass494 comparisons/eight authored images; ordinary final SHAs and known pixels remain exact. Initial256 stale observer captures are retained. | Production APIs/registry/publication, real subtitle/perceptual quality and video remain separate. Issue71 stays open. |
| Remote preservation | [Ovis3 copy40287253703](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr3-vector-preservation.json) verifies all1,662 payloads and unchanged original inner ZIP before/after actual remote readback. | Renew the preserved copy before24 January2027 and other original/Ovis/Ultra archives before their individual January expiries. |
| UltraBERT | [PR84](https://github.com/CrispStrobe/CrispEmbed/pull/84) merged; RoPE passes1344 comparisons/336 controls, empty rotation exact. Canonical SciFact preparation40372423343 passes276 source hashes,5483 actual wrapper token/mask rows and13 neutral timing probes;460 documents retain fixed512-token truncation. | [All eight originals and freeze40375626336](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/ultrabert/scifact-original-freeze.json) at immutablee35cd1a0 are independently reviewed with exact5,483-row union before native. Finish eight native shards and aggregation. Strict-v1 data rejections and separately declared canonical-v2 inputs stay retained, as do historical183/1294 and strict replay failures. |
| Immutable delivery | Seven native/mobile archives40339796226 and two WASM bundles40339800364 complete at40a82158. Initial consumers40354251257 pass Linux/macOS desktop, mobile cross-links and Rust/Dart, but overall fail. | Windows UTF-8 auditor repair awaits hosted retry. Initial Linux corpus40376735641 rejects omitted notice before PNG comparison; [corrected exact-byte40386968045](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/posformer-corpus-proof.json) atf284b1f6 passes all50 frozen PNG hashes/manifest/dimensions/notice. [Actual Mac-only40399358753](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/successor-mac-only.json) at7d561149 verifies artifacts/RGB and all50 released/default/1 IDs, text, EOS and confidences exactly; legacy0 remains. Initial40 compatibility is complete; historical7/50 quality does not improve. Fresh final-source producers/all consumers remain required; no tag or app upgrade. |
| WASM attribution | [Actual trace40379523009](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/successor-wasm-source-trace.json) reproduces eight output anchors; embedding norm exact, first captured divergence attention0, final max0.00837269/cos0.99864984 remains negative. Same-input Q8 activation rounding differs despite exact scales. | Actual40391544739 at580e3ae3 passes independently reviewed [weight-free operator controls](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/wasm-rounding-operators.json), including158/189 matmul outputs and backend/plan isolation. The [completed model A/B](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/wasm-rounding-ab-v1.json) preserves legacy/observer/context controls but rejects8/10 final cases. Attention0 improves to1.19e-7 and layers0–2 stay below9.54e-7; layer3 max0.00291893 is the first remaining failure, encoder_out0.02850568. Keep default off. Test-only [layer3 attribution40415950220](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40415950220) atbbc80ae2 must preserve old stage/final anchors before same-input Q8_0/Q8_K operator replay; actual output is pending. Budgets remain0.002/0.99999. |
| Requested permissive models | MoEdern1.5M native vectors and exact34-case Unigram roles pass; Josev raw-request40449265616 passes227 stages/four complete typed responses; sources are permissively licensed and pinned | MoEdern public query/document roles and repack; Josev packed/public interfaces, independent quality, wider MoEdern variants and qualified publication |
| Cross-platform aggregation | [Auditor repair](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/packaging/successor-desktop-vector-review-repair.json) rejects synthetic orthogonal vectors hidden behind positive local self-comparisons;14 controls pass. Retained Linux ARM/x86 max0.000119589/cos0.999999698 passes without inference. | Apply independent model/text/vector and unchanged0.002/0.99999 comparisons to all four actual final-source desktops. Two-platform reanalysis does not close the original negative batch. |

Use the [active vendor plan](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md)
for exact current commits, acceptance gates and initial failures. Queued/prepared
work is not qualification. Independent model/reference arrays stay remote;
small reports, source identities, magnitudes and actual decoded output support
review. The app consumes immutable0.17.12 rather than arbitrary vendor main.

## Recommended order and dependencies

W1 packaging is complete and archived in [HISTORY](https://github.com/CrispStrobe/CrispEmbed/blob/main/HISTORY.md).
Continue vendor UltraBERT tasks from [the executable handoff](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/next-agent-handoff.md);
V1 (selective PosFormer reconciliation) remains a separate bounded change.
M1 (fresh mathematics) and U1 (UX coverage) can proceed independently on hosted
runners. O1 can extend diagnostics now; O2 waits for checkpoint/license/access
requirements. P1 waits for a device session; A1 follows P1 and final metadata
review. R1 is used only when a production app change is ready to ship. C1 and
F1 require an explicit backend/product decision. B1 is an optional bounded
math-engine investigation, not a reason to rebuild everything immediately.

### V1 — Qualify actual successor assets and consumer behavior

**Repository:** CrispEmbed. **Source/default restoration complete; release gate open.**

Selective source PR72 merged7f1ecd33; historical PR60 closed without merging its
old release branch or moving the0.17.12 tag. [PR73](https://github.com/CrispStrobe/CrispEmbed/pull/73)
merged at `75e87ea6d0a5bbb1b3fb06362e05135b0a77b47d` after all exact-source790f9a6b
platform/style/security/WASM and ten [Build/Flutter jobs39933251542](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39933251542)
pass. `CRISPEMBED_POSFORMER_TRAINED_FORWARD` is captured at model init: unset/1
select the trained correction, matching published0.17.12; literal0 retains legacy.

[Hosted39933245147](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39933245147)
and independent raw-report review pass actual unset/released Linux0.17.12
IDs/text/EOS/confidence for all50 cases, unset/1 and zero/legacy agreement,
public API/printed regressions, primitive controls and both five-case/five-step
exported-F32/original-code reference paths. Recognition remains7/50; original
training-checkpoint/BN and unrestricted full-sequence parity are unproved.
[All raw reports/limits](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/posformer/released-default-validation.json).
Earlier actual-binary comparison retains up to2.0265579223632812e-6 confidence
drift, without fitting a tolerance or claiming general bitwise parity.

Before a successor tag, build portable native/WASM release assets using the
release workflow's CPU-baseline gate; verify archive/library hashes and normal
consumer behavior on each affected platform. Include an actual Apple released/
new-default binary comparison; Linux evidence does not certify an Apple asset.
Preserve native per-case logs in future producer artifacts; this run retains
hashes but did not upload those logs. Exact source checks do not replace actual
artifact qualification. Development0.17.13 creates no tag, publication or app pin
change; the app remains on immutable0.17.12.

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

## Published OvisOCR2 baseline — 24 October 2026

Frontend PR78 and the controlled original T4/FP16/TRITON_ATTN reference pass.
The initial [native40041958521](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/initial-ovisocr2-model-vision-failure.json)
retains its12/14 result and vision failures. Existing precision A/B40047163838
selects the explicit `both` vision preset. The reviewed frozen
[replay40049244654](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-model-audit.json) passes
all14 F32/F16 arms without recapturing the943 original fields; nine generated
IDs reach EOS248046 and decode to `Hello OCR\n\n12345`. Four vision stages still
exceed raw0.002 absolute error and pass the unchanged original backend-calibrated
budgets. [Public audit40053283383](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-public-surfaces.json)
passes27 C/Python/CLI/HTTP/Rust/Dart arms with exact text and bounded confidence.
The frozen [original HF eleven-case quality reference40054043629](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-original-quality.json)
has eight synthetic WER0 cases, all EOS/no cap, and15/15 inventory table cells;
newline CER and strict-v1 border parse failures remain visible.

The [first native F16 quality run40056641800](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-f16-quality.json)
at638380f6 has a positive independent review. Its eleven native cases retain
all1,841 generated IDs and match the frozen original HF IDs, raw text and
processed text exactly; all stop at EOS248046 without a cap hit. Eight authored
cases have WER0 and the inventory table has15/15 cells. The paired synthetic
and three-document quality screens pass within their bounded panel. This is
the immutable first F16 baseline for subsequent quantized comparisons.
Its native CPU was AMD EPYC7763 with two threads; the original HF CPU was a
different model. The recorded per-case times and cumulative child RSS are
diagnostics, not a causal paired speed result.

The [quality workflow](https://github.com/CrispStrobe/CrispEmbed/blob/main/.github/workflows/ovisocr2-quality.yml) now selects
one same-run `dev-hello` F16 host control for quant phases after exact first
F16 receipt, source, build and runtime checks. Direct driver calls still default
to the full eleven-case control, and the workflow accepts `full` explicitly.
The short control does not establish eleven-case cross-host F16 identity; its
paired scores come from the reviewed first F16 run. Every quantized candidate
still runs all eleven cases and the complete numerical and 473-tensor checks;
calibrated Q4 also retains the development-only importance proof. Each new Q4
run captures a same-run F16 first-linear control, candidate first-linear probe
and full decoder probe with unchanged thresholds. Raw vectors stay in a separate
remote artifact; small manifests and all failures stay in the report. Protected
[Q8 40061433357](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-q8-quality.json) is
complete and rejected: its first decoder-layer error is0.00376427 with minimum
cosine0.999920726, and the two-column page changes `Lorem Ipsum` to `Lorem Ispum`
with697→700 IDs and increased raw/content CER/WER. All eleven cases reach EOS;
the frozen first F16 baseline remains qualified. Reviewed Q8 diagnostic
completion permits separate Q4 experiments, without qualifying Q8 publication.
Actual [plain Q4](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-q4-quality.json) and
[calibrated Q4](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-q4-calibrated-quality.json)
are complete and rejected for numerical and document nonregression failures.
All11 cases in each reach EOS without caps; F16 local controls pass18/18.
The72-vector [development-only calibration proof](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-q4-calibrated-calibration.json)
is positive separately from candidate qualification. All473 tensor records and
261 raw manifest entries per Q4 arm are retained; large payloads stay remote.
[Q8 candidate-only replay](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-q8-raw-diff.json)
retains243 vectors and first QKV max0.0564889908, without replacing the rejected
Q8 quality wrapper. Earlier pre-model failures and raw strict flags stay in
HISTORY and their immutable receipts.

[Completed selection](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-quant-attempts.json)
chooses qualified first F16. [Actual publication40074370335](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-publication.json)
at121dc943 passes upload and immutable empty-cache full readback at
[Hub b7229e46](https://huggingface.co/cstr/OvisOCR2-GGUF/tree/b7229e46f31bcf342d0999966bae8a03bf02bb07).
Anonymous tree/small-file verification passes; no probes/test weights are
included. The [initial403](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/initial-ovisocr2-publication-authorization-failure.json)
stays separate; the temporary publisher binding is removed after success.
The real `ovisocr2-f16` URL/SHA alias and both loader fixes are wired. All25
recorded native source files and qualified `src`/ggml/CMake objects remain
unchanged; ten current consumer controls pass, including native-source drift
rejection and Darwin RSS units. Actual
[consumer40080458492](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-published-consumer.json)
at5296e955 passes two fresh-cache full model hash checks, offline CLI cache
reuse and actual CLI/HTTP text, region and existing confidence budgets. Both
independent reviews verify all seven report members/six logs, seven helper and
25 native source hashes. Public APIs expose no IDs/stops; the separately frozen
native nine-ID/EOS proof remains unchanged. PR79 is merged after final-source
and merged-source checks. The qualification loop is closed; next select the
bounded latency/RSS profiling lane in PLAN.
Model publication does not upgrade the app or create a runtime release.
Follow the [executable handoff](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/next-agent-handoff.md) and [PLAN](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md).
The app remains on immutable0.17.12; published F16 and wired source alias do not upgrade it.


## Ovis follow-up outcomes — 25 October 2026

Profiling, source feasibility, selective quantization and larger-input audits
have reviewed outcomes, including the OvisOCR3 paired/interface diagnostics.
The remaining work is source-quality/numerical attribution and the separately
scoped implementation, benchmark, preservation and immutable-release lanes. Use the [executable vendor handoff](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/next-agent-handoff.md) and
[PLAN](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md) for current dependencies. The application remains pinned to
published0.17.12; development0.17.13 and model experiments do not upgrade it.

- [Profile40171223309](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40171223309)
  is independently reviewed: three alternating off/on pairs preserve all nine
  IDs, raw text, EOS and probabilities. Median wall time is38.108/38.058 seconds,
  with about3.32 GiB process RSS including load. Eight decode steps materialize
  24.06 GB of F32 data, spending11.16 seconds materializing versus0.57 computing
  projections. Nested counters are not additive and this is no broad speed claim.
  [Full evidence](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-profiling.json).
- [Issue71 feasibility](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/inpainting-feasibility.md) is complete at source level.
  MI-GAN has explicit MIT code/weight terms; one independently implemented fixed-
  resolution graph with original RGB/mask/block/output references is the first
  bounded prototype. Payload identity and any copied helper notices remain gates.
  LaMa checkpoint terms and FFT/complex implementation remain unresolved.
  No inpainting execution, video-quality claim or issue closure follows.
- [Attention-F16/FFN-Q4 attempt40172527229](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40172527229)
  is complete and rejected. Its72 FFN matrices use Q4_K;401 tensors, including78
  attention matrices, remain byte exact. The first full layer has max error
  0.0328895 despite global cosine0.999563; receipt and two-column CER/WER regress.
  [Four-attempt selection](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-quant-attempts-v2.json)
  keeps the published F16 candidate. Independent same-quantized-weight operator
  attribution precedes any newly declared precision-selection protocol. The
  [bounded first-FFN run40231922668](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-first-ffn-attribution.json)
  now passes: independent dequantization and original/native anchors are bit
  exact, same-weight operator errors stay below1.669e-6, and original Python
  with the same Q4 weights reproduces the0.032889 layer error. No local operator
  mismatch is found at that boundary. Document-quality rejection remains;
  further precision selection requires a fresh development/evaluation protocol.
  PR82 is merged at`160520e8`; all22 final PR checks and11 fresh-main
  workflows pass. The next numerical lane is2048 attribution.
- [Larger-input audit40177774574](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-larger-inputs.json)
  is complete: frozen/1024-square/1200-landscape pass14/14;2048 boundary passes8/14.
  Six composed decoder arms fail prefill/full-prefix norm, F32/F16 max
  0.003945589/0.004386902, despite exact nine IDs/text/EOS. Original-embedding
  replay passes; root cause remains open. The2048 experiment stays unqualified,
  default1024 unchanged. All four full producer reports are retained in Git.
  [Final-source frozen14](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-final-runtime-frozen-replay.json)
  and [eleven-case F16 replay](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr2-final-runtime-f16-quality.json)
  pass with exact1,841 first-F16 outputs, stops and confidences.
- OvisOCR3 actual controlled T4 execution, processor anchors and original943-stage
  CPU/F32 capture are reviewed. [Final-source native14](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr3-native-model.json)
  passes14/14 with exact nine-ID smoke output and profile-bound isolation.
  Its original plain OCR11 capture is complete, but two-column output caps1024
  without EOS (CER0.52365/WER0.83993). [Paired F16 run40187405735](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr3-f16-quality.json)
  faithfully matches all11 cases/1,679 IDs/raw text/stops/caps. [Public-interface
  diagnostic40196755934](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr3-public-diagnostic.json)
  passes all29 recorded checks across six clients, ABI/cache and negative guards.
  It retains model-qualified=false/publication-eligible=false. The initial
  missing-artifact-name dispatch failed safe before dependencies/model transfer/build.
  A faithful diagnostic cannot override source-quality failure: no qualification,
  model alias or publication. Preserve unchanged prompts/gold/caps/scorers.
  [Original quality evidence](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/qwen35-ocr/ovisocr3-original-quality.json).

## Current audit continuation — 27 October 2026

The application remains pinned to immutable0.17.12. Vendor work is staged in
[draft PR86](https://github.com/CrispStrobe/CrispEmbed/pull/86), with separate
immutable experiment branches. No requested model is published or qualified by
source preparation, component cosine or a queued job. Use
[PLAN's executable continuation](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md#executable-audit-continuation--27-october-2026)
and [the next-agent handoff](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/next-agent-handoff.md)
for current sources and acceptance gates.

| Front | Completed evidence | Remaining gate |
| --- | --- | --- |
| Ovis interpolation | All8 grid/gate replays complete; gate0 reproduces old vectors/six boundary failing commands, gate1 passes all14 calibrated arms; all16 owning outputs exact. Boundary F32/F16 max0.00363088/0.00338554 remains above strict0.002 | Global default stays off and2048 unqualified; exact final integration frozen14/eleven-case replay, actual successor producers/consumers and main closure |
| WASM attribution | Corrected layer3 capture preserves old ordinary/observer/final anchors. All12 same-input projections have exact quantized bytes/scales and max≤1.431e-6 | Composed FFN-down error0.00331104 and prior full-panel rejection remain; no same-input kernel defect proved, default stays off |
| UltraBERT SciFact | Unicode A/B40434717493 passes all5,483 original IDs/masks/full hashes and14 authored cases; unset/empty/0 preserve old24/25 mismatches | Corrected fresh-process loader/native-v3 d993038e in40452077037 against frozen originals; preserve initial loader failure, truncation, numerical debt and ranking budgets |
| MoEdernColBERT | Native-v2 passes10 cases/177 checks/three MaxSim pairs; Unigram40451408614 passes34 cases/1040 normalization checks; old byte rejection preserved | Raw-text40455093591 passes ten cases/twenty vectors/three scores; next public roles/repack,27M/115M and representative retrieval |
| Josev-2B | Raw-request40449265616 passes227 stages/four complete typed responses; worst logit error0.0000168085. Component CPU/CUDA head and backbone proofs retained | Packed loader build passes; execute byte-preserving packing and actual packed/public parity. New SystemOne root build40455424346 is queued; independent decision quality remains open |

Public detailed evidence is in [HISTORY](https://github.com/CrispStrobe/CrispEmbed/blob/main/HISTORY.md)
and [PERFORMANCE](https://github.com/CrispStrobe/CrispEmbed/blob/main/PERFORMANCE.md).
Josev package metadata and original preflight failures are retained; neither is
an unresolved license blocker or a native inference result. The MoEdern strict
original continuity rejection remains active; do not regenerate gold or waive it.

Large artifacts stay hosted. Any new source/default choice requires an explicit
closed source contract; the historical final-v1 guard must keep rejecting the
added regression tree. Runtime assets, real C/Python/CLI/Rust/Dart/browser/Apple
consumers and the application pin remain separate release gates.
