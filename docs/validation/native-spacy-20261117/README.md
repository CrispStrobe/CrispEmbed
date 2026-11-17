# Native spaCy implementation — 17 November 2026

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
