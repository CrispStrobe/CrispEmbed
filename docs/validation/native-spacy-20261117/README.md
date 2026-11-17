# Native spaCy implementation — 17 November 2026

## Current checkpoint: frozen native acceptance and bounded W6 grammar complete

The complete first held-out native result is saved at
[`2176c896dc6bb8c832a9d03b04fd07c7c991fdc5`](https://github.com/CrispStrobe/CrispEmbed/commit/2176c896dc6bb8c832a9d03b04fd07c7c991fdc5).
All eight cases pass: 98 tokens, 13 sentences, 201 native parser transitions,
294 classifier choices and 851 numerical stages / 1,068,891 F32 elements.
Annotations, original-source spans, parser features, legal masks and actions
match exactly. Maximum absolute numerical error is `5.340576171875e-5` under
the unchanged `2e-4 + 2e-5*abs(reference)` elementwise budget.

The [complete first result](acceptance-first.json.gz),
[summary](acceptance-summary.json), [protocol](acceptance-protocol.json) and
[pre-execution remote verification](acceptance-publication-order.json) preserve
the gate and publication sequence. The eight-case split is consumed. No
candidate, reference, input or budget was tuned after that first execution.
The [W5 recovery audit and source inventory](resume-w5-source-snapshot.json)
rehashes the retained artifacts, checks all 15 archived installed files and 22
candidate source files, and independently compares saved annotations/parser
traces with the retained reference. It recomputes counts and aggregate stage
metrics from the saved report without rerunning native acceptance.

W5 now qualifies the optional Dart worker and manual Studio analysis on Linux
x86-64/AVX2; [the cross-repository evidence](https://github.com/CrispStrobe/quds_office_suite/tree/e7c67f29b34f1c1909e51aa25997fc774bb8ac80/docs/validation/native-spacy-w5-20261117)
and the final section below record its separate adapter and editor controls.

The [W6 handoff](../native-spacy-w6-20261117/README.md) records the separate
first guarded-agreement qualification against Quds main `6245a695`, including
its German phrase-completion and editor-formatting updates. The protocol was
amended before qualification with the original panel and thresholds unchanged.
The [immutable first W6 result](https://github.com/CrispStrobe/quds_office_suite/blob/3a8bca4f8f8744bbb0fbd6809a700b5768d5ceba/docs/validation/native-spacy-w6-20261117/qualification-first.json)
passes: baseline 5/30, native and combined 22/30 exact errors (+17), with zero
findings on 42 valid inputs and ten guards. Four unsupported locales are
explicit. All 22 corrected rechecks have no further finding; 18 preserve a
compatible lexical reading and four abstain. Eight missed errors remain in
the report. This authored finite panel supports bounded incremental benefit.

Native/model bytes and the consumed native acceptance remain unchanged. The
optional manual grammar provider API defaults off, and Studio does not
register it automatically. W7 target packaging, device qualification,
model/native distribution and production activation remain open.
`CRISPEMBED_SPACY` stays default off; no GitHub Actions query, wait or rerun
was used for W6.

The sections below preserve the chronological checkpoints. Statements that
acceptance was unused or decoding unfinished describe their original cutoffs;
the current status is the completed native acceptance, Linux W5 analysis and
bounded W6 grammar work above.

## Source reconciliation (W0)

This implementation starts from current CrispEmbed `b34a132c6c204b71f5c58ba8bc1705b8769edb9b`
and Quds `081af8a5` (the full commit/tree/ref inventory is in
[source-snapshot.json](source-snapshot.json)). Both repositories use isolated
worktrees. Origin branch and PR-head refs were refreshed; Quds upstream branch
and PR-head refs were refreshed too. Upstream main has no commits missing from
Quds main at this cutoff. No Actions, checks, or status endpoints were queried.

Quds includes the handoff PR #131 and subsequent ODF story/header/bookmark fixes
in PR #132. The current deterministic German context rules stay the grammar
baseline; historical AOT remains excluded. CrispEmbed main has no changes since
the handoff's inspected base. Its newer unmerged GECToR branch at `957ffa3b` is
a standalone-build/reference precedent, not a dependency to merge; the unrelated
worksheet perspective work at `0eca8e35` remains separately owned. Other fetched
feature/PR heads are inventoried, not represented as integrated.

## Owned lane and constraints

- Native source: [CrispEmbed feat/native-spacy-20261117](https://github.com/CrispStrobe/CrispEmbed/tree/feat/native-spacy-20261117).
- Developer oracle: [Quds feat/native-spacy-reference-20261117](https://github.com/CrispStrobe/quds_office_suite/tree/feat/native-spacy-reference-20261117).
- Decision: German `de_core_news_sm` 3.8.0, CPU/F32 first, no app Python, daemon,
  automatic model download, or implicit grammar activation.
- The user explicitly requires local validation and no CI waiting. This
  supersedes the generic hosted-CI guidance for this bounded implementation.
- Save meaningful commits remotely with `[skip ci]`, verifying each remote head.

## W1 inputs

The official wheel was acquired and matched **14,639,490 bytes**, SHA-256
`fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7`.
It remains a developer input outside ordinary Git. Converter outputs must be
reproducible from that authenticated input; model notices must remain intact.

The handoff's spaCy source links used a nonexistent `v3.8.18` tag. The correct
tag is `release-v3.8.18`, commit
`43311999028032fd81bef77a75fdb4fa96803c98`; this corrects source identity without
changing the reference version. Thinc `v8.3.14` is
`e617de4ea2530c7091ea2110e011807c33c366f0`.

Next checkpoint: deterministic model conversion, explicit tensor/lexical
contract, hash-bound reference environment, and nonempty full-analysis fixtures.
Numerical porting follows that remote checkpoint. Missing fixtures must fail
qualification. No native analysis, grammar, mobile, or distribution readiness
is claimed by this source checkpoint.

## W1 converted bundle and reference checkpoint

[converter-validation.json](converter-validation.json) records 33 passing
model-free controls and two independent actual conversions: all five output
files match byte-for-byte, all 39 trained F32 tensors pass independent GGUF
readback, and both model notices match the wheel verbatim. The separately
retained final export matches the same manifest SHA-256:
`d599768375de3dc56abcca5cf1d49da0015857f9d3a10640aebb5317ef5d427e`.

The 7,383,104-byte model GGUF is
`fae2e6ab8d6c611d7a9dddd4b3259a871a1d2f95cacc8259f49c1e4ff31ff6cd`.
The complete [bundle manifest](bundle-manifest.json) binds tensor layouts,
resource hashes, source files, capabilities, versions, and limits.
[environment.json](environment.json) records two matching hash-locked Linux
reference installs, including 4,181 verified installed RECORD entries each.
The full contract and reproduction commands are in
[tools/spacy](../../../tools/spacy/README.md).

The paired Quds reference checkpoint is
[`dcf2ec9b4d6fba05fa5eec2634c6654266332cc9`](https://github.com/CrispStrobe/quds_office_suite/commit/dcf2ec9b4d6fba05fa5eec2634c6654266332cc9).
Its complete nonempty development captures are stored remotely before numerical
porting: 16 cases, 113 tokens, 1,201 numerical stages, and 244 parser transitions.
Acceptance inputs remain untouched. No native inference, full analysis, grammar,
mobile, or shipping claim follows from this checkpoint. Original third-party
notices retain their exact bytes, including upstream trailing whitespace.

## Native numerical checkpoint (W3 development stage)

The standalone C++ library now performs exact-feature hash embedding, the
shared CNN, tagger/morphologizer/lemma-class scores, and parser scoring on
supplied states. It uses scalar F32 arithmetic with unfused operations; ggml
provides GGUF loading and CPU weight ownership. It does **not** implement
raw-text tokenization, parser transitions, edit-tree application, complete
analysis, a Dart worker, or grammar checking. Capabilities advertise only the
implemented numerical services.

| Evidence | Result |
| --- | --- |
| [Final installed-library parity](native-parity-final.json) | 16 development cases / 113 tokens / 1,201 stages / 1,406,553 F32 elements; all budgets pass |
| Classifier decisions | 339 exact argmax choices |
| Parser scorer replay | 244 exact legal action choices on reference-supplied states and masks; not native parsing |
| Maximum absolute numerical error | 3.814697265625e-5; predeclared tolerance is 2e-4 + 2e-5 × absolute reference value |
| [Native build/control receipt](native-validation.json) | Two native test executables pass; numeric ASan/UBSan control passes; missing reference fails explicitly |
| Installed target | Linux x86-64 / repository AVX2 baseline; ten installed library/header/notice files verified |
| Runtime dependencies | C/C++/math/system loader only in the inspected ELF; no Python library dependency |

The [compact summary](native-summary.json) independently recomputes the final
report denominators and maximum errors. The [first run](native-parity-first.json)
is retained with its original source hashes. The final run binds the updated
CMake test/install recipe and executes the installed library; both runs use the
same cases and must not be counted as extra unique evidence. No tolerance or
golden output changed after native inference.

The initial C ABI test stopped before its synthetic fixture could be created
in the unwritable default temporary directory. CTest now provides a build-owned
private temporary directory, and both controls pass. The initial stale install
recipe omitted the added notice rules; regenerating CMake and verifying the
actual install closed that setup issue. Neither was a model inference failure;
both observations are retained in the build receipt.

[Build/API instructions](../../../tools/spacy/README.md) reproduce the exact
boundary. `CRISPEMBED_SPACY` stays default off. Model weights remain outside
ordinary Git; the authenticated immutable source wheel and deterministic
converter reproduce them. Native/model release artifacts are not published yet.

## W2 lexical core continuation checkpoint

The native tokenizer and lexical core now implement the original German model
resources, spaCy reserved string symbols and MurmurHash64A IDs, source-compatible
normalization and shape, and original UTF-8/scalar/UTF-16 spans. The source text
is never normalized before alignment. Python remains developer-only. The PCRE2
regex engine is a new native dependency; its shorthand character classes are
translated using the pinned CPython 3.12.14 / Unicode 15 data to avoid host-table
drift. The generated supplement includes contextual-final-sigma classes and
reserved symbols absent from the original bundle; the bundle bytes are unchanged.

[lexical-parity-w2.json](lexical-parity-w2.json) records 16 cases, 113 tokens and
2,260 exact field comparisons through the separately compiled developer probe.
The [controls](lexical-controls-w2.json) record 28 model-free checks, 12
resource-backed checks, generated-data reproduction and explicit missing-probe
failure. [The first receipt](lexical-parity-first.json) retains its earlier
formatting/test-census source hashes; it is the same corpus, not extra cases.
Reproduction commands and exact probe/resource/source hashes are in the receipt.
The developer probe is not installed into the runtime.

This W2 checkpoint precedes composed raw-text analysis. The public ABI at this
checkpoint still exposes prepared numerical inputs; the separately owned text
API and W4 decoders are the next integration. Input bounds are 1 MiB of UTF-8
and 2,048 tokens, with explicit malformed/over-limit rejection, no truncation
and cancellation. Other platforms and broader tokenizer behavior remain to
qualify. The eight acceptance inputs remain unused.

The paired Quds handoff is now saved at
[`e87494ada0729878981dc0cc60fdb07bcd5116d4`](https://github.com/CrispStrobe/quds_office_suite/commit/e87494ada0729878981dc0cc60fdb07bcd5116d4),
[draft PR #139](https://github.com/CrispStrobe/quds_office_suite/pull/139). It
incorporates current Quds main `d6010059644b34accd6f188e0e2a4863ab63e318`,
including incoming Office/proofing changes, and restores the exact earlier W3
handoff. CrispEmbed main remains `b34a132c6c204b71f5c58ba8bc1705b8769edb9b` and
is already an ancestor. Remote source refs were refreshed without CI endpoints.

## Remaining work after W2

At the W2 cutoff, owned text integration and W4 linguistic decoding were the
next dependencies. The following section records their later completion for
the development panel; the earlier lexical checkpoint remains source-bound.

## W2–W4 composed raw-text development candidate

The installed native library now accepts only the original counted UTF-8 text
and generates its own tokens, feature IDs, parser states, legal masks and greedy
actions. It returns original byte/scalar/UTF-16 spans, fine and coarse POS,
morphology, lemmas, dependency heads/labels and sentence boundaries. No reference
tokens, states, masks or scores are supplied to the analyzer. The separate
prepared-feature API remains available unchanged, and the gate stays default off.

| Evidence | Result |
| --- | --- |
| [Composed summary](text-summary.json) and [complete installed comparison](text-parity-final.json.gz) | 16 cases / 113 tokens / 22 sentences, exact annotations and source spans |
| Native decisions | All 244 transitions, 33,428 legal-mask fields and 339 classifier choices match |
| Numerical stages | All 1,201 stages / 1,406,553 F32 elements pass; maximum absolute error `3.814697265625e-5` under the unchanged budget |
| [Expanded lexical comparison](lexical-extended-final.json) | 4,658 inputs / 13,689 token instances / 273,780 exact field comparisons |
| [Decoder isolation and sanitizers](decoder-validation-final.json) | 61 model-free assertions plus full frozen-score replay; ASan/UBSan/leak checks pass |
| [Text API controls](text-api-final.json) | 21 real-model control groups, zero failures/errors/skips |
| [Build/dependency/install receipt](text-validation.json) | Four CTest executables, CPU baseline, gate documentation, and 15 installed files verified |

The expanded lexical cases cover all 417 tokenizer exceptions in six contexts,
all 457 reserved symbols, all 1,433 scalar lowercase entries with sigma context,
16 explicit Unicode controls and 250 deterministic authored mixtures. These
test lexical implementation parity; they are not 4,658 full linguistic analyses
or a grammar-quality corpus. The original 16 development documents remain the
same documents across lexical, isolated decoder and composed repeats.

The new additive C ABI authenticates both GGUF and resources before parsing,
uses counted strings (including embedded NUL), and owns immutable documents
independently of contexts. Legacy contexts report numeric capabilities only;
resource-aware contexts report complete analysis. Controls cover wrong assets,
malformed UTF-8, byte/token limits, disabled gates, cancellation, trace abortion,
concurrency, failed-output reset and document lifetime after context destruction.

Independent review found cancellation gaps in infix iteration and dependency
reconstruction. Both now poll within those loops; authored regression controls
prove cancellation and recovery. [The lexical controls](lexical-controls-final.json)
retain the old implementation's negative. [The first decoder receipt](decoder-validation.json)
and its [exact original source archive](decoder-first-source.tar.gz) remain intact.
The [first composed run](text-parity-first.json.gz) passed before this hardening;
the installed final run confirms unchanged annotations and numerical output.
The [first API receipt](text-api-first.json) remains separately source-bound.

The expanded lexical harness initially split raw U+2028/U+2029 inside JSON with
Python `splitlines()`; literal LF framing corrected that setup failure without
changing native output. An initial attempt to invoke the environment-documentation
script through unittest discovered zero tests and was not counted as success;
its direct executable invocation passes all 499 documented variable entries.

PCRE2 10.46 is the added native runtime dependency. The inspected ELF links
PCRE2, C/C++/math/system libraries and no Python library. All five original model
bundle files remain byte-identical. Installed notices now include PCRE2,
CPython and Unicode provenance in addition to the existing model/source notices.
The build is Linux x86-64 with the repository AVX2 baseline; other platforms and
mobile distribution are not qualified by these results.

## Acceptance sequence and remaining application work

The eight frozen acceptance inputs remain unused at this candidate checkpoint.
`tests/spacy_acceptance.py` freezes the exact remote source commit, installed
binary, model, inputs, original exporter, numerical budgets and gate sources.
Six model-free admission controls verify that mutable or missing source/assets
cannot reach native execution. Publish the candidate and protocol, export and
save the original acceptance reference, then run the frozen candidate exactly
once for qualification. Preserve failures without tuning on this split.

W5 may then add the optional Dart worker/analysis UI. W6 requires incremental
grammar benefit over current deterministic rules; W7/W8 need actual packaging,
platforms, device evidence and final application qualification. No existing
Quds provider, app pin, native model alias or release default was changed. No
CI endpoint was queried, polled, awaited or rerun.

## Recovered candidate and acceptance freeze

The integrated code and development evidence were recovered from remote commit
`51c2ac328ef40b00f18eb9b03130fb67c3f11313` after the previous session ended.
The source is unchanged. [The rebuild receipt](resume-build.json) and
[environment verification](resume-environment.json) reproduce the pinned
CPython, 46 package payloads / 4,181 RECORD entries, 24 blueprint files, native
toolchain, five bundle files and 15 installed files. All four CTests, six
acceptance admission controls, CPU baseline and environment documentation checks
pass. The receipt retains the initial zero-test invocation and its correction;
only the actual six-control execution counts as success. Sanitized full build
logs are retained in [the log archive](resume-build-logs.tar.gz).

The rebuilt binary has SHA-256
`ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977`.
It differs from the historical binary because source paths are embedded, so
it was independently revalidated before freezing. [The complete rerun](resume-development-text.json.gz)
passes the same 16 cases / 113 tokens / 22 sentences / 244 transitions /
339 classifier choices / 1,201 numerical stages, with unchanged maximum
absolute error `3.814697265625e-5`. [All 21 API groups](resume-development-api.json)
also pass. These repeat the existing development panel, not additional cases.

[The frozen acceptance protocol](acceptance-protocol.json) binds this installed
binary, exact remote candidate sources, original oracle, bundle, input bytes
and unchanged policy. [The freeze receipt](resume-freeze.json) binds the
protocol and development evidence; the build receipt additionally records the
six resolved runtime dependency binary hashes, including PCRE2. At this
checkpoint no acceptance reference has been exported and no native acceptance
case has run. Save and verify this checkpoint remotely before the original
export, then save and verify the complete original reference before native
execution. The previous candidate/receipts remain intact.

The exact 15-file installed candidate is also retained in
[a deterministic validation archive](acceptance-candidate-linux-x86_64.tar.gz),
with [its provenance and extraction verification](acceptance-candidate-linux-x86_64.json).
This preserves the protocol's exact binary across workspace loss. It contains
the library, header and notices; models and system dependencies are not bundled.
The archive is Linux x86-64/AVX2 validation evidence, not a runtime release or a
consumer package. Its library hash is the frozen hash above.

## First held-out acceptance and durable handoff

The frozen source is `51c2ac328ef40b00f18eb9b03130fb67c3f11313` and the installed
binary remains
`ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977`.
The protocol was saved at `8fac62023e257ccf431e3daee25e6b2f15f182c8`, the exact
installed archive at `84327ac9af9f8925ff1f0723e027d6b76b2a2dc2`, and the complete
original reference in Quds at
[`4ae8e9e0e9b83b1a01176485bae55d9595e99b1c`](https://github.com/CrispStrobe/quds_office_suite/commit/4ae8e9e0e9b83b1a01176485bae55d9595e99b1c).
Remote source/tree/file verification preceded the single native execution;
the original UTF-8 was the only native inference input. The first full result
was then saved at `2176c89`, with [unchanged-asset evidence](acceptance-assets-after.json).

The compressed first-result SHA-256 is
`967593d587acf51557a4c27fb1cc813d3c348b91af2413a3a6453e83b5819caa`;
its uncompressed JSON SHA-256 is
`5b9c42ffc6d25b1c747ed005361b8834084f6e42e65aff50adfeb584282a77d7`.
The reference manifest is
`1b60f2e99ebacfe0a5795548482d6210f6a1e2c9b0543cf7d7192eebee2c8b25`.
The later recovery audit verifies those identities, all 851 reference trace
segments and the retained exact annotations/parser records. It aggregates the
recorded numerical comparisons; it does not claim a new model execution or
recompute native numerical arrays. The original first run remains authoritative.

This is parity to the pinned German model on eight authored held-out cases.
It does not establish grammar correction benefit, broader linguistic quality,
device performance, Dart/editor behavior or a release. Continue W5 with the
current source reconciliation and keep those later gates explicit. No Actions,
checks or status endpoints were queried or awaited during recovery.

## W5: optional Dart worker and manual Studio analysis

[The Quds W5 evidence](https://github.com/CrispStrobe/quds_office_suite/tree/e7c67f29b34f1c1909e51aa25997fc774bb8ac80/docs/validation/native-spacy-w5-20261117) is saved on
`feat/native-spacy-reference-20261117`, with the UI checkpoint at `b5b1fa99`
and the complete indexed evidence/handoff at `e7c67f29`.
The same frozen binary `ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977`
and model run through a counted-string binding and a persistent isolate worker.
No native source or held-out reference changed, and acceptance was not rerun.

Direct FFI and actual-worker replays each pass the same 16 development cases,
113 tokens, 22 sentences and 2,571 exposed-field comparisons with zero
differences, plus ten authored controls each. These executions do not increase
the unique development-document count. Twenty-nine package tests cover source
mapping, bundle admission and worker lifecycle. A compiled malformed-native
fixture passes 20 real-FFI controls with 30 document frees and zero live
documents/contexts; that fixture does not perform trained inference.

Twenty-two Studio snapshot/widget controls and one actual-native widget gate
pass. The actual gate analyzes a selection with nonzero UTF-16 origin and
navigates its token back to the exact original paragraph without editing it.
Seventy-five existing proofing regressions pass. Package and Studio browser
imports compile and return explicit unavailability without loading FFI.

The view requires a manual action and validates original text, document/story,
locale and configuration before displaying or navigating results. Cancellation
and timeout settle the caller while the physical worker slot remains owned
until native completion; disposal joins safe cleanup. No grammar provider is
registered and defaults remain off. These are bounded Linux adapter/editor
results; final checks on merge `8d89e4f7` also pass with incoming main
`4b2b4e78`. All 29 package tests, 22 UI controls, 75 existing proofing
regressions, the real-native widget gate and static analysis pass. The rendered
panel is byte-identical to the earlier reviewed UI image.
W6 must establish incremental grammar benefit against the latest deterministic
baseline. Other platforms, distribution and release remain separate gates.

## PR236 owner review — 17 November 2026

[The owner merge review](owner-merge-review.json) independently rehashes all 22
frozen sources and 15 installed archive files, verifies eight original-reference
case files at `4ae8e9e0`, and compares every retained annotation/source span and
all parser actions/features/masks. The original eight-case/851-stage result and
unchanged budgets pass retained-record review; six model-free admission controls
also pass. No native acceptance, new model matrix or W6 work was executed.
Main has no protection or applicable branch rules, hence zero required checks
or approvals. This local review uses no Actions/status query, wait or rerun.
The later W6 documentation checkpoint is preserved as separate pending work.
