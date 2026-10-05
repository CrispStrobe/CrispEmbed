# CrispMath and CrispEmbed: current state and executable lanes

Verified 5 October 2026. This handoff scopes follow-up work; it does not say the
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
  This is separate from successful app native/WASM release assets.
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
