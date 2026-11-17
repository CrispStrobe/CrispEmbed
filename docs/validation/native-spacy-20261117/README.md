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
