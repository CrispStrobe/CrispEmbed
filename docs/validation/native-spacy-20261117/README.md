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
